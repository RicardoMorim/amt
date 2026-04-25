"""
ml/dataset_builder.py
=====================
Loads and prepares the AMT signals dataset for ML training.
Target: binary classification — is this signal worth trading?

FASE 1 updates:
  - Explicit temporal ordering
  - Optional structural features joined from local SQLite candles table
  - Strictly past-only rolling features (shift(1) before merge)
  - No global normalization using future information
"""

import sqlite3
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import config

DB_PATH = config.DB_PATH
MIN_RR_RATIO = 1.5
MIN_WIN_PCT = 0.003

# ── Feature list (must match predictor order for live inference) ───────────────
FEATURES = [
    # Core order-flow features
    'distance_to_poc_pct',
    'volume_zscore',
    'delta_zscore',
    'cvd_slope_short',
    'cvd_slope_long',
    # Categorical (label-encoded)
    'signal_type_enc',
    'direction_enc',
    'session_state_enc',
    'market_state_enc',
    # Core context
    'is_composite',
    'timeframe_secs',
    'hour_utc',
    'day_of_week',
    # FASE 1 structural additions
    'false_breakout_flag',
    'price_vs_poc_norm',
    'price_vs_value_area',
    'atr_20_norm',
    'dist_to_swing_high_20',
    'dist_to_swing_low_20',
]

CAT_COLS = ['signal_type', 'direction', 'session_state', 'market_state']
ENCODERS_PATH = config.ML_ENCODERS_PATH


def _load_candles_for_join(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load candles table if available. Returns empty DataFrame otherwise."""
    try:
        candles = pd.read_sql_query(
            """
            SELECT symbol, timeframe_secs, timestamp, open, high, low, close
            FROM candles
            ORDER BY timestamp ASC
            """,
            conn,
        )
    except Exception:
        return pd.DataFrame()

    if candles.empty:
        return candles

    candles['timestamp'] = pd.to_datetime(candles['timestamp'], utc=True, errors='coerce')
    candles = candles.dropna(subset=['timestamp', 'open', 'high', 'low', 'close'])
    candles['symbol'] = candles['symbol'].astype(str)
    candles['timeframe_secs'] = candles['timeframe_secs'].astype(int)

    candles = candles.sort_values(['symbol', 'timeframe_secs', 'timestamp']).copy()

    # True Range + ATR (past-only via shift)
    candles['prev_close'] = candles.groupby(['symbol', 'timeframe_secs'])['close'].shift(1)
    tr_1 = candles['high'] - candles['low']
    tr_2 = (candles['high'] - candles['prev_close']).abs()
    tr_3 = (candles['low'] - candles['prev_close']).abs()
    candles['tr'] = np.nanmax(np.vstack([tr_1.values, tr_2.values, tr_3.values]), axis=0)

    candles['atr_20'] = (
        candles.groupby(['symbol', 'timeframe_secs'])['tr']
        .transform(lambda s: s.rolling(window=20, min_periods=20).mean())
        .shift(1)
    )
    candles['close_lag1'] = candles.groupby(['symbol', 'timeframe_secs'])['close'].shift(1)
    candles['atr_20_norm'] = candles['atr_20'] / candles['close_lag1'].replace(0, np.nan)

    # Swing distances (past-only)
    candles['swing_high_20'] = (
        candles.groupby(['symbol', 'timeframe_secs'])['high']
        .transform(lambda s: s.shift(1).rolling(window=20, min_periods=20).max())
    )
    candles['swing_low_20'] = (
        candles.groupby(['symbol', 'timeframe_secs'])['low']
        .transform(lambda s: s.shift(1).rolling(window=20, min_periods=20).min())
    )
    candles['dist_to_swing_high_20'] = (
        (candles['close_lag1'] - candles['swing_high_20']) / candles['close_lag1'].replace(0, np.nan)
    )
    candles['dist_to_swing_low_20'] = (
        (candles['close_lag1'] - candles['swing_low_20']) / candles['close_lag1'].replace(0, np.nan)
    )

    candles = candles.rename(columns={'symbol': 'asset'})
    return candles[[
        'asset', 'timeframe_secs', 'timestamp',
        'atr_20_norm', 'dist_to_swing_high_20', 'dist_to_swing_low_20'
    ]]


def load_dataset(db_path: str = DB_PATH) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            """
            SELECT *
            FROM signals
            WHERE is_labeled = 1
              AND label_win_pct  IS NOT NULL
              AND label_loss_pct IS NOT NULL
            ORDER BY timestamp_event ASC
            """,
            conn,
        )

        # Optional leakage-safe structural join from candles table
        candles = _load_candles_for_join(conn)
        if not candles.empty and not df.empty:
            df = df.copy()
            df['timestamp_event'] = pd.to_datetime(df['timestamp_event'], utc=True, errors='coerce')
            df = df.sort_values(['asset', 'timeframe_secs', 'timestamp_event'])
            candles = candles.sort_values(['asset', 'timeframe_secs', 'timestamp'])

            df = pd.merge_asof(
                left=df,
                right=candles,
                left_on='timestamp_event',
                right_on='timestamp',
                by=['asset', 'timeframe_secs'],
                direction='backward',
                allow_exact_matches=False,  # conservative: exclude same timestamp
            )
            if 'timestamp' in df.columns:
                df = df.drop(columns=['timestamp'])

        print(f"✅ Loaded {len(df):,} labeled signals from {db_path}")
        return df
    finally:
        conn.close()


def engineer_features(df: pd.DataFrame, encoders: dict | None = None, fit: bool = True):
    """
    Build feature matrix with strict time-safe transformations.

    Args:
        df:       raw signals DataFrame.
        encoders: dict of {col: LabelEncoder} from a previous fit (for inference).
        fit:      if True, fit new encoders and return them; if False, use provided ones.

    Returns:
        (df_with_features, encoders_dict)
    """
    df = df.copy()

    # ── Explicit temporal ordering ─────────────────────────────────────────────
    df['timestamp_event'] = pd.to_datetime(df['timestamp_event'], utc=True, errors='coerce')
    df = df.sort_values('timestamp_event').reset_index(drop=True)

    # ── Temporal features ──────────────────────────────────────────────────────
    df['hour_utc'] = df['timestamp_event'].dt.hour.fillna(0).astype(int)
    df['day_of_week'] = df['timestamp_event'].dt.dayofweek.fillna(0).astype(int)

    # ── Structural/contextual features (no future dependency) ──────────────────
    df['market_state'] = df.get('session_state', 'UNKNOWN').astype(str)
    df['false_breakout_flag'] = (df.get('signal_type', '').astype(str).str.upper() == 'FALSE_BREAKOUT').astype(int)

    df['distance_to_poc_pct'] = pd.to_numeric(df.get('distance_to_poc_pct', 0.0), errors='coerce').fillna(0.0)
    df['price_vs_poc_norm'] = df['distance_to_poc_pct']

    # price_vs_value_area: >0 above VAH, <0 below VAL, 0 inside VA
    if {'trigger_price', 'vah', 'val'}.issubset(df.columns):
        tp = pd.to_numeric(df['trigger_price'], errors='coerce').replace(0, np.nan)
        vah = pd.to_numeric(df['vah'], errors='coerce')
        val = pd.to_numeric(df['val'], errors='coerce')
        above = (tp - vah) / tp
        below = (tp - val) / tp
        df['price_vs_value_area'] = np.where(tp > vah, above, np.where(tp < val, below, 0.0))
        df['price_vs_value_area'] = pd.to_numeric(df['price_vs_value_area'], errors='coerce').fillna(0.0)
    else:
        df['price_vs_value_area'] = 0.0

    # Optional candle-derived features from load_dataset merge
    for col in ['atr_20_norm', 'dist_to_swing_high_20', 'dist_to_swing_low_20']:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    # Ensure expected base columns exist
    defaults = {
        'volume_zscore': 0.0,
        'delta_zscore': 0.0,
        'cvd_slope_short': 0.0,
        'cvd_slope_long': 0.0,
        'is_composite': 0,
        'timeframe_secs': 0,
        'signal_type': 'UNKNOWN',
        'direction': 'UNKNOWN',
        'session_state': 'UNKNOWN',
        'market_state': 'UNKNOWN',
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    # ── LabelEncoder for categoricals ─────────────────────────────────────────
    if encoders is None:
        encoders = {}

    for col in CAT_COLS:
        enc_key = f"{col}_enc"
        if fit:
            le = LabelEncoder()
            df[enc_key] = pd.Series(le.fit_transform(df[col].astype(str)), index=df.index, name=enc_key)
            encoders[col] = le
        else:
            le = encoders[col]
            known = set(le.classes_)

            def _encode_value(x: str, _le=le, _known=known) -> int:
                return int(_le.transform([x])[0]) if x in _known else -1

            df[enc_key] = df[col].astype(str).map(_encode_value)

    # ── Target label ───────────────────────────────────────────────────────────
    loss_abs = pd.to_numeric(df['label_loss_pct'], errors='coerce').abs().replace(0, 1e-6)
    win_pct = pd.to_numeric(df['label_win_pct'], errors='coerce')
    rr = win_pct / loss_abs
    df['target'] = ((win_pct >= MIN_WIN_PCT) & (rr >= MIN_RR_RATIO)).astype(int)

    pos = int(df['target'].sum())
    neg = int(len(df) - pos)
    if len(df) > 0:
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
