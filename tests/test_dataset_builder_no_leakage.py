import sqlite3
import pandas as pd

from ml.dataset_builder import load_dataset, engineer_features


def _build_base_db(db_path: str, future_spike: bool = False):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE signals (
            id TEXT PRIMARY KEY,
            timestamp_event TEXT,
            asset TEXT,
            timeframe_secs INTEGER,
            signal_type TEXT,
            direction TEXT,
            session_state TEXT,
            is_composite INTEGER,
            trigger_price REAL,
            distance_to_poc_pct REAL,
            volume_zscore REAL,
            delta_zscore REAL,
            cvd_slope_short REAL,
            cvd_slope_long REAL,
            label_win_pct REAL,
            label_loss_pct REAL,
            is_labeled INTEGER
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE candles (
            symbol TEXT,
            timeframe_secs INTEGER,
            timestamp TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL
        )
        """
    )

    # 35 candles so ATR/swing windows have enough history
    start = pd.Timestamp('2026-01-01T00:00:00Z')
    for i in range(35):
        ts = start + pd.Timedelta(minutes=15 * i)
        base = 100.0 + i * 0.1
        high = base + 0.2
        low = base - 0.2
        close = base + 0.05

        # optional future spike (after signal time) to test leakage resistance
        if future_spike and i == 30:
            high = base + 20.0
            close = base + 10.0

        cur.execute(
            "INSERT INTO candles VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ('btcusdt', 900, ts.isoformat(), base, high, low, close, 1000.0),
        )

    # signal at candle index 25 timestamp
    sig_ts = (start + pd.Timedelta(minutes=15 * 25)).isoformat()
    cur.execute(
        """
        INSERT INTO signals VALUES
        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            'sig_1', sig_ts, 'btcusdt', 900, 'FALSE_BREAKOUT', 'LONG', 'BALANCE', 0,
            102.5, 0.01, 0.5, -0.2, 1.0, 0.8,
            0.006, -0.003, 1,
        ),
    )

    conn.commit()
    conn.close()


def test_dataset_builder_adds_structural_features_and_keeps_order(tmp_path):
    db = tmp_path / 'dataset_base.db'
    _build_base_db(str(db), future_spike=False)

    df = load_dataset(str(db))
    df_feat, _ = engineer_features(df, fit=True)

    assert len(df_feat) == 1
    for col in [
        'market_state_enc',
        'false_breakout_flag',
        'price_vs_poc_norm',
        'price_vs_value_area',
        'atr_20_norm',
        'dist_to_swing_high_20',
        'dist_to_swing_low_20',
        'target',
    ]:
        assert col in df_feat.columns


def test_no_lookahead_from_future_candle_spike(tmp_path):
    db_a = tmp_path / 'dataset_a.db'
    db_b = tmp_path / 'dataset_b.db'
    _build_base_db(str(db_a), future_spike=False)
    _build_base_db(str(db_b), future_spike=True)

    df_a = load_dataset(str(db_a))
    feat_a, _ = engineer_features(df_a, fit=True)

    df_b = load_dataset(str(db_b))
    feat_b, _ = engineer_features(df_b, fit=True)

    # If builder is leakage-safe, future spike must not change features for earlier signal.
    cols = ['atr_20_norm', 'dist_to_swing_high_20', 'dist_to_swing_low_20']
    for col in cols:
        a = float(feat_a.iloc[0][col])
        b = float(feat_b.iloc[0][col])
        assert abs(a - b) < 1e-12, f"Leakage detected in {col}: {a} != {b}"
