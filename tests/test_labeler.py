import sqlite3
from datetime import datetime, timezone

import pandas as pd

from ml.labeler import (
    LabelStatus,
    LabelerConfig,
    SameCandlePolicy,
    SQLiteSignalLabeler,
)


def _mk_candles(ts_start: str, rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    idx = pd.date_range(pd.Timestamp(ts_start, tz='UTC'), periods=len(rows), freq='15min')
    data = []
    for i, (o, h, l, c) in enumerate(rows):
        data.append({
            'timestamp': idx[i],
            'open': o,
            'high': h,
            'low': l,
            'close': c,
            'volume': 1.0,
        })
    return pd.DataFrame(data)


def test_label_one_long_win():
    cfg = LabelerConfig(horizon_candles=3, tp_pct=0.005, sl_pct=0.003, fee_pct=0.0, slippage_pct=0.0)
    conn = sqlite3.connect(':memory:')
    labeler = SQLiteSignalLabeler(conn=conn, config=cfg)

    signal = pd.Series({
        'id': 's1',
        'timestamp_event': '2026-01-01T00:00:00Z',
        'direction': 'LONG',
        'trigger_price': 100.0,
    })
    candles = _mk_candles('2026-01-01T00:15:00Z', [
        (100.0, 100.6, 99.9, 100.4),
        (100.4, 100.5, 100.1, 100.2),
        (100.2, 100.3, 100.0, 100.1),
    ])

    out = labeler.label_one(signal, candles)
    assert out.status == LabelStatus.WIN
    assert out.reason == 'tp_hit'
    assert out.is_labeled == 1
    assert out.label_win_pct > 0


def test_label_one_short_loss():
    cfg = LabelerConfig(horizon_candles=3, tp_pct=0.005, sl_pct=0.003, fee_pct=0.0, slippage_pct=0.0)
    conn = sqlite3.connect(':memory:')
    labeler = SQLiteSignalLabeler(conn=conn, config=cfg)

    signal = pd.Series({
        'id': 's2',
        'timestamp_event': '2026-01-01T00:00:00Z',
        'direction': 'SHORT',
        'trigger_price': 100.0,
    })
    candles = _mk_candles('2026-01-01T00:15:00Z', [
        (100.0, 100.4, 99.9, 100.2),
        (100.2, 100.3, 100.0, 100.1),
        (100.1, 100.2, 99.9, 100.0),
    ])

    out = labeler.label_one(signal, candles)
    assert out.status == LabelStatus.LOSS
    assert out.reason == 'sl_hit'
    assert out.is_labeled == 1
    assert out.label_loss_pct <= 0


def test_label_one_timeout():
    cfg = LabelerConfig(horizon_candles=2, tp_pct=0.01, sl_pct=0.01, fee_pct=0.0, slippage_pct=0.0)
    conn = sqlite3.connect(':memory:')
    labeler = SQLiteSignalLabeler(conn=conn, config=cfg)

    signal = pd.Series({
        'id': 's3',
        'timestamp_event': '2026-01-01T00:00:00Z',
        'direction': 'LONG',
        'trigger_price': 100.0,
    })
    candles = _mk_candles('2026-01-01T00:15:00Z', [
        (100.0, 100.2, 99.9, 100.1),
        (100.1, 100.3, 99.95, 100.0),
    ])

    out = labeler.label_one(signal, candles)
    assert out.status == LabelStatus.TIMEOUT
    assert out.reason == 'timeout'
    assert out.is_labeled == 1


def test_same_candle_tp_sl_sl_first():
    cfg = LabelerConfig(
        horizon_candles=2,
        tp_pct=0.005,
        sl_pct=0.003,
        fee_pct=0.0,
        slippage_pct=0.0,
        same_candle_policy=SameCandlePolicy.SL_FIRST,
    )
    conn = sqlite3.connect(':memory:')
    labeler = SQLiteSignalLabeler(conn=conn, config=cfg)

    signal = pd.Series({
        'id': 's4',
        'timestamp_event': '2026-01-01T00:00:00Z',
        'direction': 'LONG',
        'trigger_price': 100.0,
    })
    candles = _mk_candles('2026-01-01T00:15:00Z', [
        (100.0, 100.6, 99.6, 100.0),
        (100.0, 100.1, 99.9, 100.0),
    ])

    out = labeler.label_one(signal, candles)
    assert out.status == LabelStatus.LOSS
    assert out.reason == 'both_hit_sl_first'
    assert out.ambiguity_flag == 1


def test_same_candle_tp_sl_skip_policy():
    cfg = LabelerConfig(
        horizon_candles=2,
        tp_pct=0.005,
        sl_pct=0.003,
        fee_pct=0.0,
        slippage_pct=0.0,
        same_candle_policy=SameCandlePolicy.SKIP,
    )
    conn = sqlite3.connect(':memory:')
    labeler = SQLiteSignalLabeler(conn=conn, config=cfg)

    signal = pd.Series({
        'id': 's5',
        'timestamp_event': '2026-01-01T00:00:00Z',
        'direction': 'LONG',
        'trigger_price': 100.0,
    })
    candles = _mk_candles('2026-01-01T00:15:00Z', [
        (100.0, 100.6, 99.6, 100.0),
        (100.0, 100.1, 99.9, 100.0),
    ])

    out = labeler.label_one(signal, candles)
    assert out.status == LabelStatus.SKIP
    assert out.is_labeled == 0
    assert out.reason == 'ambiguous_both_hit'


def test_strict_future_only_no_entry_candle_lookahead():
    cfg = LabelerConfig(horizon_candles=1, tp_pct=0.005, sl_pct=0.003, fee_pct=0.0, slippage_pct=0.0)
    conn = sqlite3.connect(':memory:')
    labeler = SQLiteSignalLabeler(conn=conn, config=cfg)

    signal = pd.Series({
        'id': 's6',
        'timestamp_event': '2026-01-01T00:15:00Z',
        'direction': 'LONG',
        'trigger_price': 100.0,
    })

    # Candle exactly at signal timestamp has TP hit (must be ignored), next candle does not hit.
    candles = pd.DataFrame([
        {'timestamp': pd.Timestamp('2026-01-01T00:15:00Z'), 'open': 100, 'high': 101, 'low': 99.5, 'close': 100, 'volume': 1},
        {'timestamp': pd.Timestamp('2026-01-01T00:30:00Z'), 'open': 100, 'high': 100.1, 'low': 99.9, 'close': 100.0, 'volume': 1},
    ])

    out = labeler.label_one(signal, candles)
    assert out.status == LabelStatus.TIMEOUT


def test_sqlite_label_signals_updates_metadata(tmp_path):
    db_path = tmp_path / 'label_test.db'
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE signals (
            id TEXT PRIMARY KEY,
            timestamp_event TEXT,
            asset TEXT,
            timeframe_secs INTEGER,
            direction TEXT,
            trigger_price REAL,
            label_win_pct REAL,
            label_loss_pct REAL,
            is_labeled INTEGER DEFAULT 0
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

    cur.execute(
        "INSERT INTO signals (id, timestamp_event, asset, timeframe_secs, direction, trigger_price, is_labeled) VALUES (?, ?, ?, ?, ?, ?, 0)",
        ('sig1', '2026-01-01T00:00:00Z', 'btcusdt', 900, 'LONG', 100.0),
    )
    cur.executemany(
        "INSERT INTO candles VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ('btcusdt', 900, '2026-01-01T00:15:00Z', 100.0, 100.6, 99.9, 100.4, 10),
            ('btcusdt', 900, '2026-01-01T00:30:00Z', 100.4, 100.5, 100.2, 100.3, 10),
        ],
    )
    conn.commit()

    labeler = SQLiteSignalLabeler(
        conn=conn,
        config=LabelerConfig(horizon_candles=2, tp_pct=0.005, sl_pct=0.003, fee_pct=0.0, slippage_pct=0.0, relabel_all=False),
    )
    stats = labeler.label_signals(symbol='btcusdt', timeframe_secs=900)
    assert stats['processed'] == 1
    assert stats['WIN'] == 1

    row = conn.execute(
        "SELECT is_labeled, label_status, label_reason, label_exit_price, label_horizon_candles FROM signals WHERE id='sig1'"
    ).fetchone()
    assert row[0] == 1
    assert row[1] == 'WIN'
    assert row[2] in ('tp_hit', 'both_hit_tp_first', 'both_hit_sl_first')
    assert row[4] == 2

    conn.close()
