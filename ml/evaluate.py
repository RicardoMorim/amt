"""
ml/evaluate.py  (D1)
====================
Holdout evaluation of the trained AMT model.

Outputs:
  - Per-signal-type win rate
  - Simulated PnL vs random baseline (fixed 0.5% TP / 0.25% SL)
  - ROC-AUC and classification report on the holdout set

Usage:
    python ml/evaluate.py
    python ml/evaluate.py path/to/amt_ml_dataset.db 0.20
"""

import os, sys, json
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.dataset_builder import load_dataset, engineer_features, FEATURES
import config


def evaluate(
    db_path:       str   = config.DB_PATH,
    holdout_frac:  float = 0.20,
    model_path:    str   = config.ML_MODEL_PATH,
    encoders_path: str   = config.ML_ENCODERS_PATH,
    tp_pct:        float = 0.005,   # 0.5% take-profit
    sl_pct:        float = 0.0025,  # 0.25% stop-loss
    threshold:     float = config.ML_CONFIDENCE_THRESHOLD,
):
    print("=" * 60)
    print("📊 AMT Model Evaluation")
    print("=" * 60)

    # Load and split — NEVER shuffle financial time-series
    df_raw = load_dataset(db_path)
    if len(df_raw) < 100:
        print("⚠️  Not enough data for evaluation.")
        return

    encoders = joblib.load(encoders_path) if os.path.exists(encoders_path) else None
    df, _    = engineer_features(df_raw, encoders=encoders, fit=(encoders is None))

    split_idx = int(len(df) * (1 - holdout_frac))
    df_hold   = df.iloc[split_idx:].copy()

    if df_hold.empty:
        print("⚠️  Holdout set is empty.")
        return

    print(f"   Holdout size: {len(df_hold):,} signals ({holdout_frac:.0%} of dataset)")

    X_hold = df_hold[FEATURES].astype(float)
    y_hold = df_hold['target']

    model  = joblib.load(model_path)
    probas = model.predict_proba(X_hold)[:, 1]
    preds  = (probas >= threshold).astype(int)

    # ── ROC-AUC ───────────────────────────────────────────────────────────────
    from sklearn.metrics import roc_auc_score, classification_report
    try:
        auc = roc_auc_score(y_hold, probas)
        print(f"\n   ROC-AUC  : {auc:.4f}")
    except Exception:
        print("   ROC-AUC  : N/A (single class in holdout)")

    print("\n📋 Classification report:")
    print(classification_report(y_hold, preds, target_names=['SKIP', 'TRADE']))

    # ── Per signal-type win rate ───────────────────────────────────────────────
    print("\n📈 Win rate by signal type (holdout):")
    df_hold['pred']  = preds
    df_hold['proba'] = probas

    for sig_type, grp in df_hold.groupby('signal_type'):
        pos_rate = grp['target'].mean()
        model_wr = grp.loc[grp['pred'] == 1, 'target'].mean() if (grp['pred'] == 1).any() else float('nan')
        print(
            f"   {sig_type:<35} base={pos_rate:.1%}  "
            f"model_wr={model_wr:.1%}  n={len(grp):,}"
        )

    # ── Simulated PnL vs random baseline ──────────────────────────────────────
    print(f"\n💰 Simulated PnL (TP={tp_pct:.1%} / SL={sl_pct:.1%}):")

    def simulate_pnl(mask, label=''):
        trades = df_hold.loc[mask].copy()
        if trades.empty:
            print(f"   {label}: no trades")
            return
        wins  = trades['label_win_pct']  >= tp_pct
        total = len(trades)
        win_n = wins.sum()
        pnl   = win_n * tp_pct - (total - win_n) * sl_pct
        print(
            f"   {label}: {total:,} trades | "
            f"WR={win_n/total:.1%} | "
            f"PnL={pnl*100:+.2f}% total | "
            f"avg={pnl/total*100:+.3f}% per trade"
        )

    simulate_pnl(df_hold['pred'] == 1,               label='Model (threshold)')
    simulate_pnl(pd.Series(True, index=df_hold.index), label='Baseline (all signals)')
    rng_mask = pd.Series(np.random.rand(len(df_hold)) >= 0.5, index=df_hold.index)
    simulate_pnl(rng_mask,                            label='Random (50%)')

    print("\n" + "=" * 60)


if __name__ == '__main__':
    db   = sys.argv[1] if len(sys.argv) > 1 else config.DB_PATH
    frac = float(sys.argv[2]) if len(sys.argv) > 2 else 0.20
    evaluate(db_path=db, holdout_frac=frac)
