"""
ml/evaluator.py
===============
Comparative evaluator for AMT model backends (XGB vs MLP).

Metrics covered:
- AUC
- accuracy
- precision / recall
- calibration (Brier + ECE)
- performance by regime
- stability by fold

Usage:
    python -m ml.evaluator
    python -m ml.evaluator --db-path amt_ml_dataset.db --threshold 0.6
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, brier_score_loss

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from ml.dataset_builder import FEATURES, load_dataset, engineer_features
from ml.nn_trainer import TabularMLP, walk_forward_splits, NNTrainConfig, _fit_one_model, _set_seed


WF_TRAIN_FRACTION = 0.70
WF_VAL_SIZE = 50000
WF_STEP = 25000


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    n = len(y_true)
    if n == 0:
        return 0.0

    for i in range(bins):
        left, right = edges[i], edges[i + 1]
        mask = (y_prob >= left) & (y_prob < right if i < bins - 1 else y_prob <= right)
        if not np.any(mask):
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def _metrics_frame(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        'auc': float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else np.nan,
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'brier': float(brier_score_loss(y_true, y_prob)),
        'ece': float(expected_calibration_error(y_true, y_prob, bins=10)),
    }


def _predict_xgb(xgb_model, X: np.ndarray) -> np.ndarray:
    return xgb_model.predict_proba(X)[:, 1]


def _predict_mlp(mlp_payload: dict, scaler, X: np.ndarray, device: torch.device) -> np.ndarray:
    model = TabularMLP(
        input_dim=int(mlp_payload['input_dim']),
        hidden_dims=tuple(mlp_payload.get('hidden_dims', [64, 32])),
        dropout=float(mlp_payload.get('dropout', 0.2)),
    ).to(device)
    model.load_state_dict(mlp_payload['state_dict'])
    model.eval()

    x_scaled = scaler.transform(X)
    xt = torch.tensor(x_scaled, dtype=torch.float32, device=device)
    with torch.no_grad():
        p = torch.sigmoid(model(xt)).detach().cpu().numpy()
    return p


def _select_regime_col(df: pd.DataFrame) -> str | None:
    if 'market_state' in df.columns:
        return 'market_state'
    if 'session_state' in df.columns:
        return 'session_state'
    return None


def _load_evaluation_dataset(db_path: str):
    df_raw = load_dataset(db_path)
    if len(df_raw) < 1000:
        print("⚠️  Small dataset; fold metrics may be noisy.")

    xgb_enc = joblib.load(config.ML_ENCODERS_PATH) if os.path.exists(config.ML_ENCODERS_PATH) else None
    fit_new = xgb_enc is None
    df, _ = engineer_features(df_raw, encoders=xgb_enc, fit=fit_new)
    x = df[FEATURES].astype(float).values
    y = df['target'].astype(int).values
    return df, x, y


def _compute_holdout(df: pd.DataFrame, x: np.ndarray, y: np.ndarray, threshold: float, xgb_model, mlp_payload, mlp_scaler, device):
    split_idx = int(len(x) * 0.80)
    x_hold, y_hold = x[split_idx:], y[split_idx:]
    df_hold = df.iloc[split_idx:].copy()

    xgb_prob = _predict_xgb(xgb_model, x_hold)
    mlp_prob = _predict_mlp(mlp_payload, mlp_scaler, x_hold, device)

    holdout = {
        'xgb': _metrics_frame(y_hold, xgb_prob, threshold),
        'mlp': _metrics_frame(y_hold, mlp_prob, threshold),
    }
    return split_idx, df_hold, y_hold, xgb_prob, mlp_prob, holdout


def _compute_regime_performance(df_hold: pd.DataFrame, split_idx: int, y_hold: np.ndarray, xgb_prob: np.ndarray, mlp_prob: np.ndarray, regime_col: str | None):
    regime_perf = {}
    if regime_col is None:
        return regime_perf

    for regime, grp in df_hold.groupby(regime_col):
        idx = grp.index - split_idx
        idx = idx.values
        if len(idx) < 20:
            continue
        y_r = y_hold[idx]
        regime_perf[str(regime)] = {
            'xgb_auc': float(roc_auc_score(y_r, xgb_prob[idx])) if len(np.unique(y_r)) > 1 else np.nan,
            'mlp_auc': float(roc_auc_score(y_r, mlp_prob[idx])) if len(np.unique(y_r)) > 1 else np.nan,
            'n': int(len(idx)),
        }
    return regime_perf


def _compute_fold_stability(
    x: np.ndarray,
    y: np.ndarray,
    threshold: float,
    xgb_model,
    device: torch.device,
    max_folds: int,
):
    fold_rows = []
    _set_seed(42)
    mlp_cfg = NNTrainConfig()

    for fold, (tr_idx, val_idx) in enumerate(walk_forward_splits(len(x), WF_TRAIN_FRACTION, WF_VAL_SIZE, WF_STEP), 1):
        if fold > max_folds:
            break
        y_val = y[val_idx]

        xgb_p = _predict_xgb(xgb_model, x[val_idx])
        xgb_m = _metrics_frame(y_val, xgb_p, threshold)

        tr_cut = max(100, int(len(tr_idx) * 0.90))
        tr_main = tr_idx[:tr_cut]
        tr_es = tr_idx[tr_cut:]
        if len(tr_es) < 50:
            tr_es = tr_idx[-50:]
            tr_main = tr_idx[:-50]

        model_fold, scaler_fold, _, _, _ = _fit_one_model(
            x_train=x[tr_main],
            y_train=y[tr_main],
            x_val=x[tr_es],
            y_val=y[tr_es],
            cfg=mlp_cfg,
            device=device,
        )
        x_val_scaled = scaler_fold.transform(x[val_idx])
        xv_t = torch.tensor(x_val_scaled, dtype=torch.float32, device=device)
        model_fold.eval()
        with torch.no_grad():
            mlp_p = torch.sigmoid(model_fold(xv_t)).detach().cpu().numpy()
        mlp_m = _metrics_frame(y_val, mlp_p, threshold)

        fold_rows.append(
            {
                'fold': fold,
                'xgb_auc': xgb_m['auc'],
                'mlp_auc': mlp_m['auc'],
                'xgb_acc': xgb_m['accuracy'],
                'mlp_acc': mlp_m['accuracy'],
                'xgb_ece': xgb_m['ece'],
                'mlp_ece': mlp_m['ece'],
                'n_val': int(len(val_idx)),
            }
        )

    return pd.DataFrame(fold_rows)


def compare_models(
    db_path: str = config.DB_PATH,
    threshold: float = config.ML_CONFIDENCE_THRESHOLD,
    max_folds: int = 20,
) -> dict:
    print("=" * 70)
    print("📊 AMT Comparative Evaluator (XGB vs MLP)")
    print("=" * 70)

    df, x, y = _load_evaluation_dataset(db_path)
    regime_col = _select_regime_col(df)

    # Load trained artifacts
    xgb_model = joblib.load(config.ML_MODEL_PATH)
    mlp_payload = torch.load(config.ML_MLP_MODEL_PATH, map_location='cpu')
    mlp_scaler = joblib.load(config.ML_MLP_SCALER_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Holdout comparison (last 20%)
    split_idx, df_hold, y_hold, xgb_prob, mlp_prob, holdout = _compute_holdout(
        df=df,
        x=x,
        y=y,
        threshold=threshold,
        xgb_model=xgb_model,
        mlp_payload=mlp_payload,
        mlp_scaler=mlp_scaler,
        device=device,
    )

    # Regime performance on holdout
    regime_perf = _compute_regime_performance(
        df_hold=df_hold,
        split_idx=split_idx,
        y_hold=y_hold,
        xgb_prob=xgb_prob,
        mlp_prob=mlp_prob,
        regime_col=regime_col,
    )

    # Stability by fold (retraining MLP each fold, fixed XGB model inference)
    fold_df = _compute_fold_stability(
        x=x,
        y=y,
        threshold=threshold,
        xgb_model=xgb_model,
        device=device,
        max_folds=max_folds,
    )

    summary = {
        'holdout': holdout,
        'regime_performance': regime_perf,
        'fold_stability': {
            'xgb_auc_mean': float(fold_df['xgb_auc'].mean()) if not fold_df.empty else np.nan,
            'xgb_auc_std': float(fold_df['xgb_auc'].std()) if not fold_df.empty else np.nan,
            'mlp_auc_mean': float(fold_df['mlp_auc'].mean()) if not fold_df.empty else np.nan,
            'mlp_auc_std': float(fold_df['mlp_auc'].std()) if not fold_df.empty else np.nan,
            'xgb_ece_mean': float(fold_df['xgb_ece'].mean()) if not fold_df.empty else np.nan,
            'mlp_ece_mean': float(fold_df['mlp_ece'].mean()) if not fold_df.empty else np.nan,
            'n_folds': int(len(fold_df)),
        },
    }

    os.makedirs('ml', exist_ok=True)
    comp_table = pd.DataFrame(
        {
            'metric': ['auc', 'accuracy', 'precision', 'recall', 'brier', 'ece'],
            'xgb_holdout': [holdout['xgb'][m] for m in ['auc', 'accuracy', 'precision', 'recall', 'brier', 'ece']],
            'mlp_holdout': [holdout['mlp'][m] for m in ['auc', 'accuracy', 'precision', 'recall', 'brier', 'ece']],
        }
    )
    comp_table['winner'] = np.where(
        comp_table['metric'].isin(['brier', 'ece']),
        np.where(comp_table['xgb_holdout'] <= comp_table['mlp_holdout'], 'xgb', 'mlp'),
        np.where(comp_table['xgb_holdout'] >= comp_table['mlp_holdout'], 'xgb', 'mlp'),
    )

    comp_csv = os.path.join('ml', 'model_comparison.csv')
    fold_csv = os.path.join('ml', 'model_fold_stability.csv')
    json_path = os.path.join('ml', 'model_comparison_report.json')
    comp_table.to_csv(comp_csv, index=False)
    fold_df.to_csv(fold_csv, index=False)
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print("\nHoldout comparison:")
    print(comp_table.to_string(index=False))

    print("\nFold stability:")
    if fold_df.empty:
        print("  No walk-forward folds available")
    else:
        print(fold_df[['fold', 'xgb_auc', 'mlp_auc', 'xgb_ece', 'mlp_ece', 'n_val']].to_string(index=False))

    print("\nArtifacts:")
    print(f"  - {comp_csv}")
    print(f"  - {fold_csv}")
    print(f"  - {json_path}")
    print("=" * 70)

    return summary


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare AMT models: XGB vs MLP")
    p.add_argument('--db-path', default=config.DB_PATH)
    p.add_argument('--threshold', type=float, default=config.ML_CONFIDENCE_THRESHOLD)
    p.add_argument('--max-folds', type=int, default=20)
    return p


if __name__ == '__main__':
    args = _build_parser().parse_args()
    compare_models(db_path=args.db_path, threshold=args.threshold, max_folds=args.max_folds)
