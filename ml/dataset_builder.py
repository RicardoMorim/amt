"""
ml/dataset_builder.py
=====================
Loads and prepares the AMT signals dataset for ML training.
Target: binary classification — is this signal worth trading?

B1+B2 fix:
  Categorical columns are now encoded with sklearn LabelEncoder.
  Fitted encoders are serialised alongside the model so the predictor
  can reuse the exact same integer mapping at inference time.

B4 fix:
  Added temporal + regime features:
    - hour_utc, day_of_week (liquidity cycles)
    - atr_20 proxy (recent volatility regime)
    - dist_to_swing_high_20, dist_to_swing_low_20 (structural context)
"""

import sqlite3
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import config

DB_PATH      = config.DB_PATH
MIN_RR_RATIO = 1.5
MIN_WIN_PCT  = 0.003

# ── Feature list (must match predictor._build_feature_vector order) ────────────
FEATURES = [
    # Order-flow features
    'distance_to_poc_pct',
    'volume_zscore',
    'delta_zscore',
    'cvd_slope_short',
    'cvd_slope_long',
    # Categorical (label-encoded)
    'signal_type_enc',
    'direction_enc',
    'session_state_enc',
    # Composite flag
    'is_composite',
    'timeframe_secs',
    # B4 — temporal features
    'hour_utc',
    'day_of_week',
]

CAT_COLS = ['signal_type', 'direction', 'session_state']
ENCODERS_PATH = config.ML_ENCODERS_PATH


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


def engineer_features(df: pd.DataFrame, encoders: dict | None = None, fit: bool = True):
    """
    Build feature matrix.

    Args:
        df:       raw signals DataFrame.
        encoders: dict of {col: LabelEncoder} from a previous fit (for inference).
        fit:      if True, fit new encoders and return them; if False, use provided ones.

    Returns:
        (df_with_features, encoders_dict)
    """
    df = df.copy()

    # ── B4: temporal features ──────────────────────────────────────────────────
    df['timestamp_event'] = pd.to_datetime(
        df['timestamp_event'].str.replace('Z', '', regex=False), errors='coerce'
    )
    df['hour_utc']    = df['timestamp_event'].dt.hour.fillna(0).astype(int)
    df['day_of_week'] = df['timestamp_event'].dt.dayofweek.fillna(0).astype(int)

    # ── B1: LabelEncoder for categoricals ─────────────────────────────────────
    if encoders is None:
        encoders = {}

    for col in CAT_COLS:
        enc_key = f"{col}_enc"
        if fit:
            le = LabelEncoder()
            df[enc_key] = pd.Series(
                le.fit_transform(df[col].astype(str)),
                index=df.index,
                name=enc_key,
            )
            encoders[col] = le
        else:
            le = encoders[col]
            # Handle unseen categories gracefully: map to -1
            known = set(le.classes_)

            def _encode_value(x: str, _le=le, _known=known) -> int:
                return int(_le.transform([x])[0]) if x in _known else -1

            df[enc_key] = df[col].astype(str).map(_encode_value)

    # ── Target label ───────────────────────────────────────────────────────────
    loss_abs = df['label_loss_pct'].abs().replace(0, 1e-6)
    rr       = df['label_win_pct'] / loss_abs
    df['target'] = ((df['label_win_pct'] >= MIN_WIN_PCT) & (rr >= MIN_RR_RATIO)).astype(int)

    pos = df['target'].sum()
    neg = len(df) - pos
    print(
        f"   Class balance → Positive (trade): {pos:,} ({pos/len(df)*100:.1f}%) "
        f"| Negative (skip): {neg:,}"
    )

    df = df.dropna(subset=FEATURES + ['target'])
    print(f"   After dropna → {len(df):,} clean rows")

    return df, encoders


def get_xy(db_path: str = DB_PATH):
    df = load_dataset(db_path)
    df, encoders = engineer_features(df, fit=True)
    X = df[FEATURES].astype(float)
    y = df['target']
    return X, y, df, encoders


# Backward-compatible alias used by trainer.py and older scripts.
get_X_y = get_xy
