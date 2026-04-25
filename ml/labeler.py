"""
ml/labeler.py
==============
Robust, audit-friendly historical labeling for AMT signals.

Design goals (FASE 1):
- Deterministic and easy to audit
- SQLite-friendly
- Walk-forward friendly (labels use only candles strictly AFTER signal time)
- Conservative handling of ambiguity (TP and SL in same candle)

Outputs written to `signals` table:
- label_win_pct
- label_loss_pct
- is_labeled
- label_status          (WIN/LOSS/TIMEOUT/SKIP)
- label_reason          (tp_hit/sl_hit/timeout/ambiguous_both_hit/...)
- label_exit_price
- label_exit_time
- label_horizon_candles
- label_tp_pct
- label_sl_pct
- label_fee_pct
- label_slippage_pct
- label_timeout_return_pct
- label_ambiguity_flag

Candles source:
- SQLite `candles` table with schema (minimum):
  symbol TEXT, timeframe_secs INTEGER, timestamp TEXT, open REAL, high REAL, low REAL, close REAL
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timezone
from typing import Any, Iterable, Optional
import logging
import sqlite3

import pandas as pd

logger = logging.getLogger(__name__)


class LabelStatus(str, Enum):
    WIN = "WIN"
    LOSS = "LOSS"
    TIMEOUT = "TIMEOUT"
    SKIP = "SKIP"


class SameCandlePolicy(str, Enum):
    """How to resolve TP+SL touched in the same candle."""

    SL_FIRST = "SL_FIRST"   # conservative default
    TP_FIRST = "TP_FIRST"
    SKIP = "SKIP"


@dataclass(frozen=True)
class LabelerConfig:
    horizon_candles: int = 8
    tp_pct: float = 0.005            # 0.50%
    sl_pct: float = 0.003            # 0.30%
    fee_pct: float = 0.0004          # 0.04% each side
    slippage_pct: float = 0.0002     # 0.02% each side
    same_candle_policy: SameCandlePolicy = SameCandlePolicy.SL_FIRST
    min_forward_candles_required: int = 1
    relabel_all: bool = True

    @property
    def roundtrip_cost_pct(self) -> float:
        return 2.0 * (self.fee_pct + self.slippage_pct)


@dataclass
class LabelOutcome:
    signal_id: str
    status: LabelStatus
    reason: str
    is_labeled: int

    label_win_pct: float
    label_loss_pct: float

    exit_price: float
    exit_time: Optional[str]

    timeout_return_pct: float
    ambiguity_flag: int


def _to_utc_ts(value: Any) -> pd.Timestamp:
    """Parse input to timezone-aware UTC timestamp."""
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid timestamp: {value}")
    return ts


def _normalize_direction(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip().upper()
    if s in {"LONG", "BUY", "1"}:
        return "LONG"
    if s in {"SHORT", "SELL", "-1"}:
        return "SHORT"
    return None


def _directional_return_pct(direction: str, entry: float, price: float) -> float:
    if entry <= 0:
        return 0.0
    if direction == "LONG":
        return (price - entry) / entry
    return (entry - price) / entry


def _effective_label_pcts(max_fav_pct: float, min_adv_pct: float, roundtrip_cost_pct: float) -> tuple[float, float]:
    """Apply costs conservatively:
    - win reduced by roundtrip costs
    - loss made worse by roundtrip costs
    """
    win_net = max(0.0, max_fav_pct - roundtrip_cost_pct)
    loss_net = min(0.0, min_adv_pct - roundtrip_cost_pct)
    return float(win_net), float(loss_net)


class SQLiteSignalLabeler:
    def __init__(self, conn: sqlite3.Connection, config: LabelerConfig):
        self.conn = conn
        self.config = config

    # -------------------------------
    # Schema / IO
    # -------------------------------

    def ensure_schema(self) -> None:
        """Add labeling metadata columns if they don't exist."""
        cur = self.conn.cursor()
        cur.execute("PRAGMA table_info(signals)")
        existing_cols = {row[1] for row in cur.fetchall()}

        required: dict[str, str] = {
            "label_status": "TEXT",
            "label_reason": "TEXT",
            "label_exit_price": "REAL",
            "label_exit_time": "TEXT",
            "label_horizon_candles": "INTEGER",
            "label_tp_pct": "REAL",
            "label_sl_pct": "REAL",
            "label_fee_pct": "REAL",
            "label_slippage_pct": "REAL",
            "label_timeout_return_pct": "REAL",
            "label_ambiguity_flag": "INTEGER DEFAULT 0",
        }

        for col, col_type in required.items():
            if col not in existing_cols:
                cur.execute(f"ALTER TABLE signals ADD COLUMN {col} {col_type}")
        self.conn.commit()

    def load_signals(self, symbol: Optional[str] = None) -> pd.DataFrame:
        where = []
        params: list[Any] = []

        if not self.config.relabel_all:
            where.append("COALESCE(is_labeled, 0) = 0")
        if symbol:
            where.append("LOWER(asset) = LOWER(?)")
            params.append(symbol)

        where_sql = f"WHERE {' AND '.join(where)}" if where else ""

        query = f"""
            SELECT
                id,
                timestamp_event,
                asset,
                timeframe_secs,
                direction,
                trigger_price
            FROM signals
            {where_sql}
            ORDER BY timestamp_event ASC
        """
        return pd.read_sql_query(query, self.conn, params=params)

    def load_candles(self, symbol: Optional[str] = None, timeframe_secs: Optional[int] = None) -> pd.DataFrame:
        where = []
        params: list[Any] = []
        if symbol:
            where.append("LOWER(symbol) = LOWER(?)")
            params.append(symbol)
        if timeframe_secs is not None:
            where.append("timeframe_secs = ?")
            params.append(int(timeframe_secs))

        where_sql = f"WHERE {' AND '.join(where)}" if where else ""

        query = f"""
            SELECT symbol, timeframe_secs, timestamp, open, high, low, close, volume
            FROM candles
            {where_sql}
            ORDER BY timestamp ASC
        """
        try:
            df = pd.read_sql_query(query, self.conn, params=params)
        except Exception:
            return pd.DataFrame()

        if df.empty:
            return df

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp", "open", "high", "low", "close"])
        return df

    # -------------------------------
    # Label logic
    # -------------------------------

    def label_one(self, signal_row: pd.Series, candles_df: pd.DataFrame) -> LabelOutcome:
        sig_id = str(signal_row["id"])

        # Validate signal
        direction = _normalize_direction(signal_row.get("direction"))
        entry = float(signal_row.get("trigger_price", 0.0) or 0.0)
        if direction is None or entry <= 0:
            return LabelOutcome(
                signal_id=sig_id,
                status=LabelStatus.SKIP,
                reason="invalid_direction_or_entry",
                is_labeled=0,
                label_win_pct=0.0,
                label_loss_pct=0.0,
                exit_price=entry,
                exit_time=None,
                timeout_return_pct=0.0,
                ambiguity_flag=0,
            )

        try:
            sig_ts = _to_utc_ts(signal_row["timestamp_event"])
        except Exception:
            return LabelOutcome(
                signal_id=sig_id,
                status=LabelStatus.SKIP,
                reason="invalid_timestamp",
                is_labeled=0,
                label_win_pct=0.0,
                label_loss_pct=0.0,
                exit_price=entry,
                exit_time=None,
                timeout_return_pct=0.0,
                ambiguity_flag=0,
            )

        # Strictly future candles only (no lookahead leakage from entry candle)
        future = candles_df.loc[candles_df["timestamp"] > sig_ts].head(self.config.horizon_candles)

        if len(future) < self.config.min_forward_candles_required:
            return LabelOutcome(
                signal_id=sig_id,
                status=LabelStatus.SKIP,
                reason="insufficient_forward_candles",
                is_labeled=0,
                label_win_pct=0.0,
                label_loss_pct=0.0,
                exit_price=entry,
                exit_time=None,
                timeout_return_pct=0.0,
                ambiguity_flag=0,
            )

        tp_level = entry * (1 + self.config.tp_pct) if direction == "LONG" else entry * (1 - self.config.tp_pct)
        sl_level = entry * (1 - self.config.sl_pct) if direction == "LONG" else entry * (1 + self.config.sl_pct)

        max_fav_pct = 0.0
        min_adv_pct = 0.0

        status = LabelStatus.TIMEOUT
        reason = "timeout"
        exit_price = float(future.iloc[-1]["close"])
        exit_time = pd.Timestamp(future.iloc[-1]["timestamp"]).isoformat()
        ambiguity_flag = 0

        for _, row in future.iterrows():
            high = float(row["high"])
            low = float(row["low"])
            close = float(row["close"])
            c_ts = pd.Timestamp(row["timestamp"]).isoformat()

            if direction == "LONG":
                fav = (high - entry) / entry
                adv = (low - entry) / entry
                tp_hit = high >= tp_level
                sl_hit = low <= sl_level
            else:
                fav = (entry - low) / entry
                adv = (entry - high) / entry
                tp_hit = low <= tp_level
                sl_hit = high >= sl_level

            max_fav_pct = max(max_fav_pct, float(fav))
            min_adv_pct = min(min_adv_pct, float(adv))

            if tp_hit and sl_hit:
                ambiguity_flag = 1
                if self.config.same_candle_policy == SameCandlePolicy.SKIP:
                    status = LabelStatus.SKIP
                    reason = "ambiguous_both_hit"
                    exit_price = close
                    exit_time = c_ts
                    break
                if self.config.same_candle_policy == SameCandlePolicy.SL_FIRST:
                    status = LabelStatus.LOSS
                    reason = "both_hit_sl_first"
                    exit_price = sl_level
                    exit_time = c_ts
                    break
                status = LabelStatus.WIN
                reason = "both_hit_tp_first"
                exit_price = tp_level
                exit_time = c_ts
                break

            if tp_hit:
                status = LabelStatus.WIN
                reason = "tp_hit"
                exit_price = tp_level
                exit_time = c_ts
                break

            if sl_hit:
                status = LabelStatus.LOSS
                reason = "sl_hit"
                exit_price = sl_level
                exit_time = c_ts
                break

        roundtrip_cost = self.config.roundtrip_cost_pct
        label_win_pct, label_loss_pct = _effective_label_pcts(max_fav_pct, min_adv_pct, roundtrip_cost)

        timeout_return_pct = 0.0
        if status == LabelStatus.TIMEOUT:
            timeout_return_pct = _directional_return_pct(direction, entry, float(exit_price)) - roundtrip_cost

        # Conservative: SKIP rows are NOT considered labeled.
        is_labeled = 0 if status == LabelStatus.SKIP else 1

        return LabelOutcome(
            signal_id=sig_id,
            status=status,
            reason=reason,
            is_labeled=is_labeled,
            label_win_pct=label_win_pct,
            label_loss_pct=label_loss_pct,
            exit_price=float(exit_price),
            exit_time=exit_time,
            timeout_return_pct=float(timeout_return_pct),
            ambiguity_flag=int(ambiguity_flag),
        )

    def label_signals(self, symbol: Optional[str] = None, timeframe_secs: Optional[int] = None) -> dict[str, int]:
        self.ensure_schema()

        signals_df = self.load_signals(symbol=symbol)
        if signals_df.empty:
            return {"processed": 0, "WIN": 0, "LOSS": 0, "TIMEOUT": 0, "SKIP": 0}

        candles_df = self.load_candles(symbol=symbol, timeframe_secs=timeframe_secs)
        if candles_df.empty:
            raise RuntimeError(
                "No candles available in SQLite table `candles`. "
                "Run historical backfill with candle persistence first."
            )

        # Pre-group candles by (symbol, timeframe) for efficient lookup
        candles_groups: dict[tuple[str, int], pd.DataFrame] = {}
        for (sym, tf), grp in candles_df.groupby(["symbol", "timeframe_secs"], sort=False):
            candles_groups[(str(sym).lower(), int(tf))] = grp.sort_values("timestamp")

        stats = {"processed": 0, "WIN": 0, "LOSS": 0, "TIMEOUT": 0, "SKIP": 0}
        updates = []

        for _, sig in signals_df.iterrows():
            sig_symbol = str(sig.get("asset", "")).lower()
            sig_tf = int(sig.get("timeframe_secs", 0) or 0)

            # Prefer exact (symbol,timeframe), then fallback by symbol only
            cdf = candles_groups.get((sig_symbol, sig_tf))
            if cdf is None:
                fallback = [g for (s, _tf), g in candles_groups.items() if s == sig_symbol]
                cdf = fallback[0] if fallback else pd.DataFrame()

            outcome = self.label_one(sig, cdf)
            stats["processed"] += 1
            stats[outcome.status.value] += 1

            updates.append(
                (
                    outcome.label_win_pct,
                    outcome.label_loss_pct,
                    outcome.is_labeled,
                    outcome.status.value,
                    outcome.reason,
                    outcome.exit_price,
                    outcome.exit_time,
                    int(self.config.horizon_candles),
                    float(self.config.tp_pct),
                    float(self.config.sl_pct),
                    float(self.config.fee_pct),
                    float(self.config.slippage_pct),
                    outcome.timeout_return_pct,
                    outcome.ambiguity_flag,
                    outcome.signal_id,
                )
            )

        self.conn.executemany(
            """
            UPDATE signals
            SET
                label_win_pct = ?,
                label_loss_pct = ?,
                is_labeled = ?,
                label_status = ?,
                label_reason = ?,
                label_exit_price = ?,
                label_exit_time = ?,
                label_horizon_candles = ?,
                label_tp_pct = ?,
                label_sl_pct = ?,
                label_fee_pct = ?,
                label_slippage_pct = ?,
                label_timeout_return_pct = ?,
                label_ambiguity_flag = ?
            WHERE id = ?
            """,
            updates,
        )
        self.conn.commit()
        return stats


def relabel_sqlite(
    db_path: str,
    config: LabelerConfig,
    symbol: Optional[str] = None,
    timeframe_secs: Optional[int] = None,
) -> dict[str, int]:
    """Convenience entry point for scripts/CLI."""
    conn = sqlite3.connect(db_path)
    try:
        labeler = SQLiteSignalLabeler(conn=conn, config=config)
        stats = labeler.label_signals(symbol=symbol, timeframe_secs=timeframe_secs)
        logger.info("Labeling complete: %s", stats)
        return stats
    finally:
        conn.close()
