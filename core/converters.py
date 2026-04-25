"""
Conversion utilities for AMT / Kronos contracts.

FASE 0 — Bridges between pandas/SQLite/dict and the typed contracts.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, TypeVar, Union

import pandas as pd

from .contracts import (
    AMTSignal,
    Candle,
    Contract,
    FusionDecision,
    KronosPrediction,
    TradeLabel,
)


# ---------------------------------------------------------------------------
# Pandas helpers
# ---------------------------------------------------------------------------

def df_to_candles(df: pd.DataFrame, symbol: str = "", timeframe_secs: int = 60) -> List[Candle]:
    """Convert an OHLCV DataFrame to a list of Candle objects."""
    candles = []
    for _, row in df.iterrows():
        try:
            candle = Candle.from_pandas_row(row, symbol=symbol, timeframe_secs=timeframe_secs)
            candles.append(candle)
        except (KeyError, ValueError, TypeError):
            # Skip malformed rows silently — caller can inspect later
            continue
    return candles


def df_to_signals(df: pd.DataFrame) -> List[AMTSignal]:
    """Convert a DataFrame of signal records to AMTSignal objects."""
    signals = []
    for _, row in df.iterrows():
        try:
            # Try legacy format first (dict-like), then typed dict
            if isinstance(row, dict):
                sig = AMTSignal.from_legacy_dict(row)
            else:
                sig = AMTSignal.from_legacy_dict(row.to_dict())
            signals.append(sig)
        except Exception:
            continue
    return signals


def df_to_predictions(df: pd.DataFrame) -> List[KronosPrediction]:
    """Convert a DataFrame of prediction records to KronosPrediction objects."""
    preds = []
    for _, row in df.iterrows():
        try:
            d = row.to_dict() if not isinstance(row, dict) else row
            preds.append(KronosPrediction.from_dict(d))
        except Exception:
            continue
    return preds


def candles_to_df(candles: List[Candle]) -> pd.DataFrame:
    """Convert a list of Candle objects back to an OHLCV DataFrame."""
    if not candles:
        return pd.DataFrame(columns=["symbol", "timestamp", "open", "high", "low", "close", "volume", "amount"])

    records = [c.to_dict() for c in candles]
    df = pd.DataFrame(records)
    # Parse ISO timestamps back to datetime
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def signals_to_df(signals: List[AMTSignal]) -> pd.DataFrame:
    """Convert a list of AMTSignal objects to a DataFrame."""
    if not signals:
        return pd.DataFrame(columns=["symbol", "timestamp_event", "direction", "signal_type"])

    records = [s.to_dict() for s in signals]
    df = pd.DataFrame(records)
    if "timestamp_event" in df.columns:
        df["timestamp_event"] = pd.to_datetime(df["timestamp_event"], utc=True)
    return df


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------

def sqlite_row_to_candle(row: Dict[str, Any], schema: str = "ohlcv") -> Optional[Candle]:
    """Convert a SQLite row (dict-like) to a Candle.

    Args:
        row: A dict or sqlite3.Row object from cursor.fetchone().
             Note: For sqlite3.Row support, set conn.row_factory = sqlite3.Row
             before executing queries.
        schema: Expected column naming convention ("ohlcv" or "trades").
    """
    # Normalize to a plain dict — handles both real dicts and sqlite3.Row objects
    if hasattr(row, 'keys'):
        # Dict-like (sqlite3.Row with row_factory set, or plain dict)
        d = {k: row[k] for k in row.keys()}
    elif hasattr(row, '__iter__') and not isinstance(row, str):
        # Tuple-like (cursor.fetchone() without row_factory)
        keys = ["symbol", "timestamp", "open", "high", "low", "close", "volume", "amount"]
        d = {k: v for k, v in zip(keys, row) if v is not None}
    else:
        return None

    try:
        return Candle.from_dict({
            "symbol": str(d.get("symbol", d.get("instrument", ""))),
            "timestamp": _sqlite_ts_to_iso(d),
            "open": float(d["open"]),
            "high": float(d["high"]),
            "low": float(d["low"]),
            "close": float(d["close"]),
            "volume": float(d.get("volume", 0)),
            "amount": float(d.get("amount", d.get("value", 0))),
        })
    except (KeyError, ValueError, TypeError):
        return None


def sqlite_row_to_signal(row: Dict[str, Any]) -> Optional[AMTSignal]:
    """Convert a SQLite row (dict-like) to an AMTSignal."""
    try:
        d = {k: row[k] for k in row.keys()}
        return AMTSignal.from_legacy_dict(d)
    except Exception:
        return None


def sqlite_row_to_prediction(row: Dict[str, Any]) -> Optional[KronosPrediction]:
    """Convert a SQLite row (dict-like) to a KronosPrediction."""
    try:
        d = {k: row[k] for k in row.keys()}
        return KronosPrediction.from_dict(d)
    except Exception:
        return None


def _sqlite_ts_to_iso(row: sqlite3.Row, ts_col: str = "timestamp") -> str:
    """Convert various SQLite timestamp formats to ISO-8601 string."""
    raw = row.get(ts_col)
    if raw is None:
        return datetime.now(tz=timezone.utc).isoformat()

    # If it's already a number (unix ts), convert
    try:
        ts_num = float(raw)
        # Heuristic: if > 1e12, it's milliseconds
        if ts_num > 1e12:
            ts_num /= 1000
        return datetime.fromtimestamp(ts_num, tz=timezone.utc).isoformat()
    except (ValueError, TypeError, OSError):
        pass

    # Otherwise treat as string
    s = str(raw)
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return s


# ---------------------------------------------------------------------------
# Legacy dict adapters — for gradual migration
# ---------------------------------------------------------------------------

def adapt_signal_dict(d: Dict[str, Any]) -> AMTSignal:
    """Adapt any legacy signal dict to the new AMTSignal contract."""
    # Normalize common field name variations
    normalized = {
        "symbol": d.get("symbol", d.get("instrument", d.get("ticker", "UNKNOWN"))),
        "direction": d.get("direction", d.get("side", d.get("action", "FLAT"))),
        "signal_type": d.get("signal_type", d.get("type", d.get("kind", "FUSION_DECISION"))),
        "timestamp_event": d.get("timestamp_event", d.get("timestamp", d.get("ts", None))),
        "confidence": d.get("confidence", d.get("conf", d.get("score", 0.5))),
    }
    # Copy through any extra fields the caller might need
    normalized.update(d)
    return AMTSignal.from_legacy_dict(normalized)


def adapt_prediction_dict(d: Dict[str, Any]) -> KronosPrediction:
    """Adapt any legacy prediction dict to the new KronosPrediction contract."""
    normalized = {
        "symbol": d.get("symbol", d.get("instrument", d.get("ticker", "UNKNOWN"))),
        "direction": d.get("direction", d.get("prediction", d.get("pred", "FLAT"))),
        "timestamp": d.get("timestamp", d.get("target_time", d.get("ts", None))),
        "confidence": d.get("confidence", d.get("conf", d.get("score", 0.5))),
    }
    normalized.update(d)
    return KronosPrediction.from_dict(normalized)


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Row = Union[Dict[str, Any], pd.Series, sqlite3.Row]
