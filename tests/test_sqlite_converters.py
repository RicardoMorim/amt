import sqlite3
import pytest
from datetime import datetime, timezone
from core.converters import (
    sqlite_row_to_candle,
    sqlite_row_to_signal,
    sqlite_row_to_prediction,
    _sqlite_ts_to_iso
)
from core.contracts import Candle, AMTSignal, KronosPrediction, Direction, SignalType

@pytest.fixture
def db_conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()

def test_sqlite_row_to_candle_dict():
    row = {
        "symbol": "BTC/USDT",
        "timestamp": "2023-01-01T00:00:00Z",
        "open": 16000.0,
        "high": 16100.0,
        "low": 15900.0,
        "close": 16050.0,
        "volume": 10.5,
        "amount": 168000.0
    }
    candle = sqlite_row_to_candle(row)
    assert isinstance(candle, Candle)
    assert candle.symbol == "BTC/USDT"
    assert candle.open == pytest.approx(16000.0)
    assert candle.amount == pytest.approx(168000.0)

def test_sqlite_row_to_candle_row_factory(db_conn):
    db_conn.execute("CREATE TABLE candles (symbol TEXT, timestamp TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL, amount REAL)")
    db_conn.execute("INSERT INTO candles VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                   ("ETH/USDT", "2023-01-01T00:00:00Z", 1200.0, 1210.0, 1190.0, 1205.0, 100.0, 120500.0))
    row = db_conn.execute("SELECT * FROM candles").fetchone()

    candle = sqlite_row_to_candle(row)
    assert isinstance(candle, Candle)
    assert candle.symbol == "ETH/USDT"
    assert candle.close == pytest.approx(1205.0)

def test_sqlite_row_to_candle_field_mapping():
    # Test 'instrument' instead of 'symbol' and 'value' instead of 'amount'
    row = {
        "instrument": "SOL/USDT",
        "timestamp": "2023-01-01T00:00:00Z",
        "open": 20.0,
        "high": 22.0,
        "low": 19.0,
        "close": 21.0,
        "value": 1000.0
    }
    candle = sqlite_row_to_candle(row)
    assert candle.symbol == "SOL/USDT"
    assert candle.amount == pytest.approx(1000.0)

def test_sqlite_row_to_candle_tuple():
    # keys = ["symbol", "timestamp", "open", "high", "low", "close", "volume", "amount"]
    row = ("BNB/USDT", "2023-01-01T00:00:00Z", 300.0, 310.0, 290.0, 305.0, 50.0, 15250.0)
    candle = sqlite_row_to_candle(row)
    assert candle.symbol == "BNB/USDT"
    assert candle.close == pytest.approx(305.0)

def test_sqlite_row_to_candle_malformed():
    row = {"symbol": "BTC/USDT"} # Missing mandatory fields
    assert sqlite_row_to_candle(row) is None

    assert sqlite_row_to_candle(None) is None
    assert sqlite_row_to_candle("not a row") is None

def test_sqlite_row_to_signal():
    row = {
        "symbol": "BTC/USDT",
        "direction": "LONG",
        "signal_type": "FALSE_BREAKOUT",
        "timestamp_event": "2023-01-01T00:00:00Z"
    }
    sig = sqlite_row_to_signal(row)
    assert isinstance(sig, AMTSignal)
    assert sig.direction == Direction.LONG
    assert sig.signal_type == SignalType.FALSE_BREAKOUT

def test_sqlite_row_to_prediction():
    row = {
        "symbol": "BTC/USDT",
        "timestamp": "2023-01-01T00:00:00Z",
        "direction": "SHORT",
        "confidence": 0.8
    }
    pred = sqlite_row_to_prediction(row)
    assert isinstance(pred, KronosPrediction)
    assert pred.direction == Direction.SHORT
    assert pred.confidence == pytest.approx(0.8)

def test_sqlite_ts_to_iso():
    # None
    assert _sqlite_ts_to_iso({}) is not None # Returns current time

    # Unix seconds
    iso = _sqlite_ts_to_iso({"timestamp": 1672531200})
    assert "2023-01-01T00:00:00" in iso

    # Unix milliseconds
    iso_ms = _sqlite_ts_to_iso({"timestamp": 1672531200000})
    assert "2023-01-01T00:00:00" in iso_ms

    # ISO string with Z
    iso_z = _sqlite_ts_to_iso({"timestamp": "2023-01-01T00:00:00Z"})
    assert "+00:00" in iso_z

    # Invalid timestamp (should treat as string)
    iso_inv = _sqlite_ts_to_iso({"timestamp": "invalid"})
    assert iso_inv == "invalid"

    # Extreme value (should be handled by try-except in _sqlite_ts_to_iso)
    iso_extreme = _sqlite_ts_to_iso({"timestamp": 1e20})
    assert iso_extreme == str(1e20)
