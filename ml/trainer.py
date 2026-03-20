"""
ml/trainer.py
=============
Trains an XGBoost classifier on AMT signals.
Uses TimeSeriesSplit — NEVER shuffle financial time series.

Usage:
    python ml/trainer.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    classification_report, roc_auc_score,
    precision_score, recall_score, f1_score
)
from dataset_builder import get_X_y, FEATURES

MODEL_PATH = "ml/amt_model.pkl"
META_PATH  = "ml/amt_model_meta.json"
N_SPLITS   = 5


def train(db_path: str = "amt_ml_dataset.db"):
    print("=" * 60)
    print("🤖 AMT ML Trainer")
    print("=" * 60)

    X, y, df = get_X_y(db_path)

    if len(X) < 500:
        print("⚠️  Warning: fewer than 500 samples — model may be unreliable.")

    # ── TimeSeriesSplit cross-validation ───────────────────────────────────────
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    cv_aucs = []

    print(f"\n📊 Cross-validation ({N_SPLITS}-fold TimeSeriesSplit)...")
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        m = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=10,   # avoid overfitting on rare signals
            scale_pos_weight=(y_tr == 0).sum() / max((y_tr == 1).sum(), 1),
            eval_metric='logloss',
            random_state=42,
            verbosity=0,
        )
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        auc = roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])
        cv_aucs.append(auc)
        print(f"   Fold {fold}: AUC = {auc:.4f} | val_size = {len(y_val):,}")

    print(f"\n   Mean AUC = {np.mean(cv_aucs):.4f} ± {np.std(cv_aucs):.4f}")

    # ── Final model — train on ALL data ───────────────────────────────────────
    print("\n🏋️  Training final model on full dataset...")
    final_model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1),
        eval_metric='logloss',
        random_state=42,
        verbosity=0,
    )
    final_model.fit(X, y)

    # ── Last fold evaluation report ────────────────────────────────────────────
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    y_pred = final_model.predict(X_val)

    print("\n📋 Classification report (last fold):")
    print(classification_report(y_val, y_pred, target_names=["SKIP", "TRADE"]))

    # ── Feature importance ────────────────────────────────────────────────────
    print("🔍 Feature importances:")
    importances = dict(zip(FEATURES, final_model.feature_importances_))
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
        bar = "█" * int(imp * 40)
        print(f"   {feat:<25} {bar} {imp:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs("ml", exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)

    import json
    meta = {
        "features": FEATURES,
        "cv_mean_auc": float(np.mean(cv_aucs)),
        "cv_std_auc": float(np.std(cv_aucs)),
        "n_train_samples": int(len(X)),
        "n_positive": int(y.sum()),
        "n_negative": int((y == 0).sum()),
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ Model saved  → {MODEL_PATH}")
    print(f"✅ Metadata     → {META_PATH}")
    print("=" * 60)
    return final_model


if __name__ == "__main__":
    import sys
    db = sys.argv[1] if len(sys.argv) > 1 else "amt_ml_dataset.db"
    train(db)
