"""
ml/dataset_builder.py
=====================
Loads and prepares the AMT signals dataset for ML training.
Target: binary classification — is this signal worth trading?
"""

import sqlite3
import numpy as np
import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────────
DB_PATH       = "amt_ml_dataset.db"
MIN_RR_RATIO  = 1.5   # Minimum reward/risk ratio to label as positive
MIN_WIN_PCT   = 0.003  # Minimum 0.3% gain to count as a real win

FEATURES = [
    'distance_to_poc_pct',
    'volume_zscore',
    'delta_zscore',
    'cvd_slope_short',
    'cvd_slope_long',
    'signal_type_enc',
    'direction_enc',
    'session_state_enc',
    'is_composite',
    'timeframe_secs',
]

# ── Loader ─────────────────────────────────────────────────────────────────────
def load_dataset(db_path: str = DB_PATH) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("""
        SELECT *
        FROM signals
        WHERE is_labeled = 1
          AND label_win_pct  IS NOT NULL
          AND label_loss_pct IS NOT NULL
        ORDER BY timestamp_event ASC
    """, conn)
    conn.close()
    print(f"✅ Loaded {len(df):,} labeled signals from {db_path}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # ── Categorical encoding ───────────────────────────────────────────────────
    df['signal_type_enc']   = pd.Categorical(df['signal_type']).codes
    df['direction_enc']     = pd.Categorical(df['direction']).codes        # buy=0 sell=1
    df['session_state_enc'] = pd.Categorical(df['session_state']).codes

    # ── Target label ───────────────────────────────────────────────────────────
    # Positive = win gain is meaningful AND reward/risk ratio is good
    loss_abs = df['label_loss_pct'].abs().replace(0, 1e-6)
    rr       = df['label_win_pct'] / loss_abs
    df['target'] = ((df['label_win_pct'] >= MIN_WIN_PCT) & (rr >= MIN_RR_RATIO)).astype(int)

    # ── Sanity stats ───────────────────────────────────────────────────────────
    pos = df['target'].sum()
    neg = len(df) - pos
    print(f"   Class balance → Positive (trade): {pos:,} ({pos/len(df)*100:.1f}%) | Negative (skip): {neg:,}")

    # ── Drop rows with NaN in any feature ─────────────────────────────────────
    df = df.dropna(subset=FEATURES + ['target'])
    print(f"   After dropna → {len(df):,} clean rows")

    return df


def get_X_y(db_path: str = DB_PATH):
    df = load_dataset(db_path)
    df = engineer_features(df)
    X  = df[FEATURES].astype(float)
    y  = df['target']
    return X, y, df
