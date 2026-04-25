"""
ml/nn_trainer.py
================
Alternative tabular neural baseline (MLP) for AMT.

This DOES NOT replace XGBoost production baseline.
It provides a comparable model family for fair benchmarking.

Features:
- Small MLP (tabular)
- Dropout
- Weight decay
- Early stopping
- Class weighting (pos_weight in BCEWithLogitsLoss)
- Walk-forward fold metrics

Usage:
    python -m ml.nn_trainer
    python -m ml.nn_trainer path/to/amt_ml_dataset.db --epochs 80 --patience 10
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from dataclasses import dataclass

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from ml.dataset_builder import FEATURES, get_xy


WF_TRAIN_FRACTION = 0.70
WF_VAL_SIZE = 50000
WF_STEP = 25000


def walk_forward_splits(n_samples: int, train_fraction: float, val_size: int, step: int):
    train_start = 0
    train_end = int(n_samples * train_fraction)

    while True:
        val_start = train_end
        val_end = min(val_start + val_size, n_samples)
        if val_end - val_start < 1000:
            break

        yield np.arange(train_start, train_end), np.arange(val_start, val_end)

        train_start += step
        train_end = min(train_start + (val_end - val_start), n_samples)


class TabularMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, int] = (64, 32), dropout: float = 0.20):
        super().__init__()
        h1, h2 = hidden_dims
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class NNTrainConfig:
    hidden_dims: tuple[int, int] = (64, 32)
    dropout: float = 0.20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 100
    patience: int = 10
    batch_size: int = 256
    seed: int = 42
    max_folds: int = 20


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_tensor(x: np.ndarray, y: np.ndarray | None = None, device: torch.device = torch.device("cpu")):
    x_t = torch.tensor(x, dtype=torch.float32, device=device)
    if y is None:
        return x_t, None
    y_t = torch.tensor(y, dtype=torch.float32, device=device)
    return x_t, y_t


def _fit_one_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    cfg: NNTrainConfig,
    device: torch.device,
):
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_val_s = scaler.transform(x_val)

    model = TabularMLP(input_dim=x_train_s.shape[1], hidden_dims=cfg.hidden_dims, dropout=cfg.dropout).to(device)

    pos = float((y_train == 1).sum())
    neg = float((y_train == 0).sum())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    x_train_t, y_train_t = _to_tensor(x_train_s, y_train, device=device)
    x_val_t, y_val_t = _to_tensor(x_val_s, y_val, device=device)

    best_state = None
    best_val_loss = float("inf")
    best_epoch = 0

    n = len(x_train_s)
    rng = np.random.default_rng(cfg.seed)
    for epoch in range(1, cfg.epochs + 1):
        model.train()

        idx = rng.permutation(n)
        for start in range(0, n, cfg.batch_size):
            batch_idx = idx[start:start + cfg.batch_size]
            xb = x_train_t[batch_idx]
            yb = y_train_t[batch_idx]

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(x_val_t)
            val_loss = float(criterion(val_logits, y_val_t).item())

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        elif (epoch - best_epoch) >= cfg.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        val_logits = model(x_val_t)
        val_proba = torch.sigmoid(val_logits).detach().cpu().numpy()

    return model, scaler, val_proba, best_epoch, best_val_loss


def train(db_path: str = config.DB_PATH, cfg: NNTrainConfig | None = None):
    cfg = cfg or NNTrainConfig()
    _set_seed(cfg.seed)

    print("=" * 60)
    print("🧠 AMT NN Trainer (Tabular MLP)")
    print("=" * 60)

    X, y, _, encoders = get_xy(db_path)
    x_np = X.astype(np.float32).values
    y_np = y.astype(np.float32).values

    if len(x_np) < 500:
        print("⚠️  Warning: fewer than 500 samples — model may be unreliable.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Walk-forward folds for stability metrics
    fold_aucs = []
    print(f"\n📊 Walk-forward validation (train={WF_TRAIN_FRACTION}, val_size={WF_VAL_SIZE}, step={WF_STEP})...")

    for fold, (tr_idx, val_idx) in enumerate(walk_forward_splits(len(x_np), WF_TRAIN_FRACTION, WF_VAL_SIZE, WF_STEP), 1):
        if fold > cfg.max_folds:
            break
        # last 10% of train window used for early stopping validation; fold val kept untouched for metric
        tr_cut = max(100, int(len(tr_idx) * 0.90))
        tr_main = tr_idx[:tr_cut]
        tr_es = tr_idx[tr_cut:]
        if len(tr_es) < 50:
            tr_es = tr_idx[-50:]
            tr_main = tr_idx[:-50]

        model, scaler, _, best_epoch, best_val_loss = _fit_one_model(
            x_train=x_np[tr_main],
            y_train=y_np[tr_main],
            x_val=x_np[tr_es],
            y_val=y_np[tr_es],
            cfg=cfg,
            device=device,
        )

        x_val_fold = scaler.transform(x_np[val_idx])
        x_val_t, _ = _to_tensor(x_val_fold, None, device=device)
        with torch.no_grad():
            proba = torch.sigmoid(model(x_val_t)).detach().cpu().numpy()

        auc = roc_auc_score(y_np[val_idx], proba)
        fold_aucs.append(float(auc))
        print(
            f"   Fold {fold}: AUC={auc:.4f} | train={len(tr_idx):,} val={len(val_idx):,} "
            f"| best_epoch={best_epoch} val_loss={best_val_loss:.5f}"
        )

    print(f"\n   Mean AUC = {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
    print(f"   Total folds: {len(fold_aucs)}")

    # Final model training using last 10% as early-stop slice
    print("\n🏋️  Training final MLP model...")
    split = max(100, int(len(x_np) * 0.90))
    x_train_full = x_np[:split]
    y_train_full = y_np[:split]
    x_val_full = x_np[split:]
    y_val_full = y_np[split:]

    # If dataset is tiny and no val slice, fallback to duplicate a tiny chunk
    if len(x_val_full) < 50:
        cut = max(1, int(len(x_np) * 0.85))
        x_train_full = x_np[:cut]
        y_train_full = y_np[:cut]
        x_val_full = x_np[cut:]
        y_val_full = y_np[cut:]

    final_model, final_scaler, val_proba_full, best_epoch, best_val_loss = _fit_one_model(
        x_train=x_train_full,
        y_train=y_train_full,
        x_val=x_val_full,
        y_val=y_val_full,
        cfg=cfg,
        device=device,
    )

    y_pred_full = (val_proba_full >= 0.5).astype(int)
    print("\n📋 Classification report (final early-stop slice):")
    print(classification_report(y_val_full.astype(int), y_pred_full, target_names=['SKIP', 'TRADE']))

    os.makedirs('ml', exist_ok=True)
    torch.save(
        {
            'state_dict': final_model.state_dict(),
            'input_dim': int(x_np.shape[1]),
            'hidden_dims': list(cfg.hidden_dims),
            'dropout': float(cfg.dropout),
        },
        config.ML_MLP_MODEL_PATH,
    )
    joblib.dump(final_scaler, config.ML_MLP_SCALER_PATH)
    joblib.dump(encoders, config.ML_MLP_ENCODERS_PATH)

    meta = {
        'backend': 'mlp',
        'features': FEATURES,
        'hidden_dims': list(cfg.hidden_dims),
        'dropout': cfg.dropout,
        'lr': cfg.lr,
        'weight_decay': cfg.weight_decay,
        'epochs': cfg.epochs,
        'patience': cfg.patience,
        'batch_size': cfg.batch_size,
        'seed': cfg.seed,
        'cv_mean_auc': float(np.mean(fold_aucs)) if fold_aucs else None,
        'cv_std_auc': float(np.std(fold_aucs)) if fold_aucs else None,
        'wf_train_fraction': WF_TRAIN_FRACTION,
        'wf_val_size': WF_VAL_SIZE,
        'wf_step': WF_STEP,
        'wf_n_folds': len(fold_aucs),
        'n_train_samples': int(len(x_np)),
        'n_positive': int((y_np == 1).sum()),
        'n_negative': int((y_np == 0).sum()),
        'best_epoch_final': int(best_epoch),
        'best_val_loss_final': float(best_val_loss),
    }
    with open(config.ML_MLP_META_PATH, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ MLP model   → {config.ML_MLP_MODEL_PATH}")
    print(f"✅ MLP scaler  → {config.ML_MLP_SCALER_PATH}")
    print(f"✅ MLP encoders→ {config.ML_MLP_ENCODERS_PATH}")
    print(f"✅ MLP meta    → {config.ML_MLP_META_PATH}")
    print("=" * 60)

    return final_model


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train AMT tabular MLP model")
    p.add_argument('db_path', nargs='?', default=config.DB_PATH)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--dropout', type=float, default=0.20)
    p.add_argument('--h1', type=int, default=64)
    p.add_argument('--h2', type=int, default=32)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-folds', type=int, default=20)
    return p


if __name__ == '__main__':
    args = _build_parser().parse_args()
    cfg = NNTrainConfig(
        hidden_dims=(args.h1, args.h2),
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        seed=args.seed,
        max_folds=args.max_folds,
    )
    train(db_path=args.db_path, cfg=cfg)
