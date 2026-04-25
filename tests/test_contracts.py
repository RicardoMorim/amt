"""Unit tests for shared data contracts (FASE 0)."""

import json
import sqlite3
from datetime import datetime, timezone
from io import StringIO

import pandas as pd
import pytest

from core.contracts import (
    AMTContextSnapshot,
    AMTSignal,
    Candle,
    CandleWindow,
    FusionDecision,
    KronosPrediction,
    MarketState,
    SignalType,
    SessionState,
    TradeAction,
    TradeLabel,
    Direction,
    contracts_from_json,
    contracts_to_json,
)
from core.converters import (
    adapt_prediction_dict,
    adapt_signal_dict,
    candles_to_df,
    df_to_candles,
    df_to_signals,
    signals_to_df,
    sqlite_row_to_candle,
    sqlite_row_to_signal,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_ts():
    return datetime(2025, 6, 15, 14, 30, 0, tzinfo=timezone.utc)


@pytest.fixture
def candle(sample_ts):
    return Candle(
        symbol="NQ1",
        timestamp=sample_ts,
        open=18000.0,
        high=18050.0,
        low=17980.0,
        close=18025.0,
        volume=1500.0,
        amount=27037500.0,
        timeframe_secs=60,
    )


@pytest.fixture
def signal(sample_ts):
    return AMTSignal(
        symbol="NQ1",
        timestamp_event=sample_ts,
        direction=Direction.LONG,
        signal_type=SignalType.FALSE_BREAKOUT,
        session_state=SessionState.NY,
        is_composite=True,
        confidence=0.82,
        probability=0.75,
        distance_to_poc_pct=1.23,
        volume_zscore=2.45,
        delta_zscore=-1.67,
        cvd_slope_short=0.034,
        cvd_slope_long=-0.012,
    )


@pytest.fixture
def prediction(sample_ts):
    return KronosPrediction(
        symbol="NQ1",
        timestamp=sample_ts,
        direction=Direction.SHORT,
        confidence=0.71,
        probability_long=0.29,
        probability_short=0.71,
        context_length=512,
        timeframe_secs=300,
    )


@pytest.fixture
def fusion(sample_ts):
    return FusionDecision(
        symbol="NQ1",
        timestamp=sample_ts,
        action=TradeAction.BUY,
        direction=Direction.LONG,
        amt_score=0.65,
        kronos_score=0.48,
        fusion_score=0.57,
        confidence=0.68,
        probability=0.62,
        contributing_signals=["FALSE_BREAKOUT", "KRONOS_PREDICTION"],
    )


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------

class TestEnums:
    def test_direction_from_raw(self):
        assert Direction.from_raw("LONG") == Direction.LONG
        assert Direction.from_raw("buy") == Direction.LONG
        assert Direction.from_raw(1) == Direction.LONG
        assert Direction.from_raw("SHORT") == Direction.SHORT
        assert Direction.from_raw(-1) == Direction.SHORT
        assert Direction.from_raw("FLAT") == Direction.FLAT
        assert Direction.from_raw("HOLD") == Direction.FLAT

    def test_market_state_from_raw(self):
        assert MarketState.from_raw("BALANCE") == MarketState.BALANCE
        assert MarketState.from_raw("IMBALANCE_UP") == MarketState.IMBALANCE_UP
        assert MarketState.from_raw("invalid") == MarketState.UNKNOWN

    def test_signal_type_from_raw(self):
        assert SignalType.from_raw("FALSE_BREAKOUT") == SignalType.FALSE_BREAKOUT
        assert SignalType.from_raw("volume_imbalance") == SignalType.VOLUME_IMBALANCE
        # Unknown falls back to FUSION_DECISION
        assert SignalType.from_raw("unknown_kind") == SignalType.FUSION_DECISION

    def test_trade_action_from_raw(self):
        assert TradeAction.from_raw("BUY") == TradeAction.BUY
        assert TradeAction.from_raw("SELL") == TradeAction.SELL
        assert TradeAction.from_raw("LONG") == TradeAction.BUY  # alias
        assert TradeAction.from_raw("FLAT") == TradeAction.HOLD

    def test_session_state_from_raw(self):
        assert SessionState.from_raw("LONDON") == SessionState.LONDON
        assert SessionState.from_raw("unknown") == SessionState.UNKNOWN


# ---------------------------------------------------------------------------
# Candle tests
# ---------------------------------------------------------------------------

class TestCandle:
    def test_to_dict_roundtrip(self, candle):
        d = candle.to_dict()
        restored = Candle.from_dict(d)
        assert restored.symbol == candle.symbol
        assert abs(restored.open - candle.open) < 1e-6
        assert abs(restored.close - candle.close) < 1e-6
        assert restored.volume == candle.volume
        assert restored.timeframe_secs == candle.timeframe_secs

    def test_from_pandas_row(self):
        row = pd.Series({
            "timestamp": "2025-06-15T14:30:00+00:00",
            "open": 180.0,
            "high": 182.0,
            "low": 179.5,
            "close": 181.0,
            "volume": 500,
        })
        c = Candle.from_pandas_row(row, symbol="ES1", timeframe_secs=60)
        assert c.symbol == "ES1"
        assert abs(c.open - 180.0) < 1e-6
        assert c.timestamp.tzinfo is not None

    def test_from_unix_timestamp(self):
        row = pd.Series({
            "timestamp": 1749923400,  # unix seconds
            "open": 100.0,
            "high": 105.0,
            "low": 99.0,
            "close": 103.0,
            "volume": 200,
        })
        c = Candle.from_pandas_row(row)
        assert c.timestamp.tzinfo is not None

    def test_timestamp_utc(self, candle):
        assert candle.timestamp.tzinfo is not None


# ---------------------------------------------------------------------------
# CandleWindow tests
# ---------------------------------------------------------------------------

class TestCandleWindow:
    def test_from_candles(self, sample_ts):
        candles = [
            Candle(symbol="NQ1", timestamp=sample_ts.replace(minute=m), open=100+m, high=105+m, low=98+m, close=102+m, timeframe_secs=60)
            for m in range(5)
        ]
        window = CandleWindow.from_candles(candles)
        assert window.symbol == "NQ1"
        assert window.length == 5
        d = window.to_dict()
        restored = CandleWindow.from_dict(d)
        assert restored.length == 5

    def test_empty_window(self):
        w = CandleWindow(symbol="TEST")
        assert w.length == 0


# ---------------------------------------------------------------------------
# AMTSignal tests
# ---------------------------------------------------------------------------

class TestAMTSignal:
    def test_to_dict_roundtrip(self, signal):
        d = signal.to_dict()
        restored = AMTSignal.from_dict(d)
        assert restored.symbol == signal.symbol
        assert restored.direction == Direction.LONG
        assert restored.signal_type == SignalType.FALSE_BREAKOUT
        assert abs(restored.confidence - 0.82) < 1e-3

    def test_from_legacy_dict(self):
        legacy = {
            "symbol": "NQ1",
            "direction": "LONG",
            "signal_type": "FALSE_BREAKOUT",
            "timestamp_event": "2025-06-15T14:30:00+00:00",
            "session_state": "NY",
            "is_composite": True,
            "distance_to_poc_pct": 1.5,
            "volume_zscore": 2.0,
        }
        sig = AMTSignal.from_legacy_dict(legacy)
        assert sig.direction == Direction.LONG
        assert sig.signal_type == SignalType.FALSE_BREAKOUT

    def test_serialization(self, signal):
        d = signal.to_dict()
        assert "direction" in d
        assert d["direction"] == "LONG"
        assert "signal_type" in d
        assert d["signal_type"] == "FALSE_BREAKOUT"


# ---------------------------------------------------------------------------
# AMTContextSnapshot tests
# ---------------------------------------------------------------------------

class TestAMTContextSnapshot:
    def test_from_volume_profile(self):
        profile = {"poc": 18000, "vah": 18100, "val": 17900}
        snap = AMTContextSnapshot.from_volume_profile("NQ1", profile, current_price=18050)
        assert snap.poc == 18000.0
        assert snap.vah == 18100.0
        assert snap.market_state in (MarketState.BALANCE, MarketState.IMBALANCE_UP)

    def test_from_volume_profile_no_data(self):
        snap = AMTContextSnapshot.from_volume_profile("NQ1", None, current_price=100)
        assert snap.poc == 0.0
        assert snap.market_state == MarketState.UNKNOWN


# ---------------------------------------------------------------------------
# KronosPrediction tests
# ---------------------------------------------------------------------------

class TestKronosPrediction:
    def test_roundtrip(self, prediction):
        d = prediction.to_dict()
        restored = KronosPrediction.from_dict(d)
        assert restored.direction == Direction.SHORT
        assert abs(restored.confidence - 0.71) < 1e-3

    def test_from_legacy_fields(self):
        legacy = {
            "symbol": "NQ1",
            "prediction": "SHORT",
            "timestamp": "2025-06-15T14:30:00+00:00",
            "conf": 0.8,
        }
        pred = KronosPrediction.from_dict(legacy)
        assert pred.direction == Direction.SHORT


# ---------------------------------------------------------------------------
# FusionDecision tests
# ---------------------------------------------------------------------------

class TestFusionDecision:
    def test_roundtrip(self, fusion):
        d = fusion.to_dict()
        restored = FusionDecision.from_dict(d)
        assert restored.action == TradeAction.BUY
        assert "FALSE_BREAKOUT" in restored.contributing_signals


# ---------------------------------------------------------------------------
# TradeLabel tests
# ---------------------------------------------------------------------------

class TestTradeLabel:
    def test_roundtrip(self, sample_ts):
        label = TradeLabel(
            symbol="NQ1",
            timestamp=sample_ts,
            action=TradeAction.BUY,
            entry_price=18000.0,
            exit_price=18150.0,
            pnl_pct=0.83,
            holding_bars=5,
        )
        d = label.to_dict()
        restored = TradeLabel.from_dict(d)
        assert abs(restored.pnl_pct - 0.83) < 1e-3


# ---------------------------------------------------------------------------
# Converter tests
# ---------------------------------------------------------------------------

class TestConverters:
    def test_df_to_candles(self):
        df = pd.DataFrame({
            "timestamp": ["2025-06-15T14:00:00+00:00", "2025-06-15T14:01:00+00:00"],
            "open": [180.0, 181.0],
            "high": [182.0, 183.0],
            "low": [179.0, 180.0],
            "close": [181.5, 182.0],
            "volume": [100, 200],
        })
        candles = df_to_candles(df, symbol="ES1", timeframe_secs=60)
        assert len(candles) == 2
        assert candles[0].symbol == "ES1"

    def test_candles_to_df(self):
        candles = [
            Candle(symbol="NQ1", timestamp=datetime(2025, 6, 15, 14, i, tzinfo=timezone.utc),
                   open=100+i, high=105+i, low=98+i, close=102+i, volume=50)
            for i in range(3)
        ]
        df = candles_to_df(candles)
        assert len(df) == 3
        assert "timestamp" in df.columns

    def test_df_to_signals(self):
        df = pd.DataFrame([{
            "symbol": "NQ1",
            "direction": "LONG",
            "signal_type": "FALSE_BREAKOUT",
            "timestamp_event": "2025-06-15T14:30:00+00:00",
        }])
        signals = df_to_signals(df)
        assert len(signals) == 1
        assert signals[0].direction == Direction.LONG

    def test_signals_to_df(self, signal):
        df = signals_to_df([signal])
        assert len(df) == 1
        assert "direction" in df.columns

    def test_adapt_signal_dict(self):
        legacy = {
            "instrument": "ES1",
            "side": "SHORT",
            "type": "VOLUME_IMBALANCE",
            "timestamp": "2025-06-15T14:30:00+00:00",
        }
        sig = adapt_signal_dict(legacy)
        assert sig.symbol == "ES1"
        assert sig.direction == Direction.SHORT

    def test_adapt_prediction_dict(self):
        legacy = {
            "ticker": "NQ1",
            "pred": "LONG",
            "timestamp": "2025-06-15T14:30:00+00:00",
            "score": 0.7,
        }
        pred = adapt_prediction_dict(legacy)
        assert pred.symbol == "NQ1"
        assert pred.direction == Direction.LONG


# ---------------------------------------------------------------------------
# SQLite tests
# ---------------------------------------------------------------------------

class TestSQLite:
    def test_sqlite_row_to_candle(self):
        # Use a dict (sqlite3.Row is not constructible in all Python versions)
        row = {
            "symbol": "NQ1",
            "timestamp": "2025-06-15T14:30:00+00:00",
            "open": 180.0,
            "high": 182.0,
            "low": 179.0,
            "close": 181.0,
            "volume": 500,
        }
        candle = sqlite_row_to_candle(row)
        assert candle is not None
        assert candle.symbol == "NQ1"

    def test_sqlite_unix_timestamp(self):
        row = {
            "symbol": "ES1",
            "timestamp": 1749923400,
            "open": 5200.0,
            "high": 5210.0,
            "low": 5195.0,
            "close": 5205.0,
            "volume": 300,
        }
        candle = sqlite_row_to_candle(row)
        assert candle is not None

    def test_sqlite_row_to_signal(self):
        row = {
            "symbol": "NQ1",
            "direction": "LONG",
            "signal_type": "FALSE_BREAKOUT",
            "timestamp_event": "2025-06-15T14:30:00+00:00",
        }
        sig = sqlite_row_to_signal(row)
        assert sig is not None
        assert sig.direction == Direction.LONG


# ---------------------------------------------------------------------------
# JSON serialization tests
# ---------------------------------------------------------------------------

class TestJSONSerialization:
    def test_contracts_to_json(self, candle, signal, prediction):
        contracts = [candle, signal, prediction]
        json_str = contracts_to_json(contracts)
        data = json.loads(json_str)
        assert len(data) == 3
        assert data[0]["symbol"] == "NQ1"

    def test_contracts_from_json(self):
        data = [
            {"symbol": "NQ1", "timestamp_event": "2025-06-15T14:30:00+00:00",
             "direction": "LONG", "signal_type": "FALSE_BREAKOUT"},
            {"symbol": "NQ1", "timestamp": "2025-06-15T14:30:00+00:00",
             "prediction": "SHORT", "conf": 0.7},
        ]
        json_str = json.dumps(data)
        contracts = contracts_from_json(json_str)
        assert len(contracts) == 2
        assert isinstance(contracts[0], AMTSignal)
