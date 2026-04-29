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
import multiprocessing
import sqlite3

import numpy as np
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
    horizon_candles: int = 25
    tp_pct: float = 0.007
    sl_pct: float = 0.0025
    fee_pct: float = 0.0004
    slippage_pct: float = 0.0002
    same_candle_policy: SameCandlePolicy = SameCandlePolicy.TP_FIRST
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


def _effective_label_pcts(
    max_fav_pct: float,
    min_adv_pct: float,
    roundtrip_cost_pct: float,
) -> tuple[float, float]:
    win_net = max(0.0, max_fav_pct - roundtrip_cost_pct)
    loss_net = min(0.0, min_adv_pct - roundtrip_cost_pct)
    return float(win_net), float(loss_net)


# ---------------------------------------------------------------------------
# Core label logic (pure function — safe to call from worker processes)
# ---------------------------------------------------------------------------

def _label_one_raw(
    sig_id: str,
    direction: Optional[str],
    entry: float,
    sig_ts_ns: int,                 # UTC nanoseconds (int64)
    ts_ns_arr: np.ndarray,          # int64 array of candle timestamps
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray,
    config_horizon: int,
    config_tp_pct: float,
    config_sl_pct: float,
    config_roundtrip_cost: float,
    config_same_candle_policy: str,
    config_min_forward: int,
) -> tuple:
    """
    Pure-function label logic operating on raw numpy arrays.
    Uses searchsorted for O(log n) candle lookup instead of O(n) .loc filter.
    Returns a flat tuple ready for executemany.
    """
    # O(log n) binary search for first candle strictly after signal timestamp
    start_idx = int(np.searchsorted(ts_ns_arr, sig_ts_ns, side="right"))
    end_idx = min(start_idx + config_horizon, len(ts_ns_arr))
    n_forward = end_idx - start_idx

    _SKIP = "SKIP"

    if direction is None or entry <= 0:
        return (0.0, 0.0, 0, "SKIP", "invalid_direction_or_entry",
                entry, None, config_horizon, config_tp_pct, config_sl_pct,
                0.0, 0, sig_id)

    if n_forward < config_min_forward:
        return (0.0, 0.0, 0, "SKIP", "insufficient_forward_candles",
                entry, None, config_horizon, config_tp_pct, config_sl_pct,
                0.0, 0, sig_id)

    if direction == "LONG":
        tp_level = entry * (1.0 + config_tp_pct)
        sl_level = entry * (1.0 - config_sl_pct)
    else:
        tp_level = entry * (1.0 - config_tp_pct)
        sl_level = entry * (1.0 + config_sl_pct)

    max_fav_pct = 0.0
    min_adv_pct = 0.0

    status = "TIMEOUT"
    reason = "timeout"
    ambiguity_flag = 0
    exit_price = float(close_arr[end_idx - 1])
    exit_ts_ns = int(ts_ns_arr[end_idx - 1])

    for i in range(start_idx, end_idx):
        high = float(high_arr[i])
        low = float(low_arr[i])
        close = float(close_arr[i])
        c_ts_ns = int(ts_ns_arr[i])

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

        if fav > max_fav_pct:
            max_fav_pct = fav
        if adv < min_adv_pct:
            min_adv_pct = adv

        if tp_hit and sl_hit:
            ambiguity_flag = 1
            if config_same_candle_policy == "SKIP":
                status, reason, exit_price, exit_ts_ns = "SKIP", "ambiguous_both_hit", close, c_ts_ns
                break
            if config_same_candle_policy == "SL_FIRST":
                status, reason, exit_price, exit_ts_ns = "LOSS", "both_hit_sl_first", sl_level, c_ts_ns
                break
            status, reason, exit_price, exit_ts_ns = "WIN", "both_hit_tp_first", tp_level, c_ts_ns
            break

        if tp_hit:
            status, reason, exit_price, exit_ts_ns = "WIN", "tp_hit", tp_level, c_ts_ns
            break

        if sl_hit:
            status, reason, exit_price, exit_ts_ns = "LOSS", "sl_hit", sl_level, c_ts_ns
            break

    label_win_pct, label_loss_pct = _effective_label_pcts(
        max_fav_pct, min_adv_pct, config_roundtrip_cost
    )

    timeout_return_pct = 0.0
    if status == "TIMEOUT":
        timeout_return_pct = _directional_return_pct(
            direction, entry, float(exit_price)) - config_roundtrip_cost

    # Convert exit_ts_ns back to ISO string
    exit_time = pd.Timestamp(exit_ts_ns, unit="ns",
                             tz="UTC").isoformat() if exit_ts_ns else None

    is_labeled = 0 if status == "SKIP" else 1

    return (
        label_win_pct, label_loss_pct, is_labeled, status, reason,
        float(exit_price), exit_time,
        config_horizon, config_tp_pct, config_sl_pct,
        float(timeout_return_pct), int(ambiguity_flag),
        sig_id,
    )


# ---------------------------------------------------------------------------
# Worker function (runs in a separate process)
# ---------------------------------------------------------------------------

def _label_group_worker(args: tuple) -> tuple[list[tuple], dict[str, int]]:
    """
    Labels all signals for one (symbol, timeframe) group.
    Called by ProcessPoolExecutor — must be picklable (top-level function).

    Returns (updates_list, stats_dict).
    """
    (
        signals_records,   # list of dicts
        ts_ns_arr,         # np.ndarray int64
        high_arr,
        low_arr,
        close_arr,
        config_horizon,
        config_tp_pct,
        config_sl_pct,
        config_roundtrip_cost,
        config_same_candle_policy,
        config_min_forward,
    ) = args

    updates = []
    stats = {"WIN": 0, "LOSS": 0, "TIMEOUT": 0, "SKIP": 0}

    for sig in signals_records:
        direction = _normalize_direction(sig.get("direction"))
        entry = float(sig.get("trigger_price", 0.0) or 0.0)
        sig_id = str(sig["id"])

        try:
            sig_ts_ns = int(pd.to_datetime(
                sig["timestamp_event"], utc=True).value)
        except Exception:
            updates.append((
                0.0, 0.0, 0, "SKIP", "invalid_timestamp",
                entry, None, config_horizon, config_tp_pct, config_sl_pct,
                0.0, 0, sig_id,
            ))
            stats["SKIP"] += 1
            continue

        row = _label_one_raw(
            sig_id=sig_id,
            direction=direction,
            entry=entry,
            sig_ts_ns=sig_ts_ns,
            ts_ns_arr=ts_ns_arr,
            high_arr=high_arr,
            low_arr=low_arr,
            close_arr=close_arr,
            config_horizon=config_horizon,
            config_tp_pct=config_tp_pct,
            config_sl_pct=config_sl_pct,
            config_roundtrip_cost=config_roundtrip_cost,
            config_same_candle_policy=config_same_candle_policy,
            config_min_forward=config_min_forward,
        )
        updates.append(row)
        stats[row[3]] = stats.get(row[3], 0) + 1  # row[3] is status string

    return updates, stats


# ---------------------------------------------------------------------------
# Main labeler class
# ---------------------------------------------------------------------------

class SQLiteSignalLabeler:
    def __init__(
        self,
        conn: sqlite3.Connection,
        config: LabelerConfig,
        n_workers: Optional[int] = None,
    ):
        self.conn = conn
        self.config = config
        # Default: leave one core free for the OS / SQLite writes
        self.n_workers = n_workers or max(
            1, (multiprocessing.cpu_count() or 2) - 1)

    # ------------------------------------------------------------------
    # Schema / IO
    # ------------------------------------------------------------------

    def ensure_schema(self) -> None:
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

    def load_candles(
        self,
        symbol: Optional[str] = None,
        timeframe_secs: Optional[int] = None,
    ) -> pd.DataFrame:
        where, params = [], []
        if symbol:
            where.append("LOWER(symbol) = LOWER(?)")
            params.append(symbol)
        if timeframe_secs is not None:
            where.append("timeframe_secs = ?")
            params.append(int(timeframe_secs))

        where_sql = f"WHERE {' AND '.join(where)}" if where else ""
        query = f"""
            SELECT symbol, timeframe_secs, timestamp, open, high, low, close, volume
            FROM candles {where_sql}
            ORDER BY timestamp ASC
        """
        try:
            df = pd.read_sql_query(query, self.conn, params=params)
        except Exception:
            return pd.DataFrame()

        if df.empty:
            return df

        df["timestamp"] = pd.to_datetime(
            df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp", "open", "high", "low", "close"])
        return df

    # ------------------------------------------------------------------
    # label_one kept for backwards compat / unit tests
    # ------------------------------------------------------------------

    def label_one(self, signal_row: pd.Series, candles_df: pd.DataFrame) -> LabelOutcome:
        """Label a single signal. Uses searchsorted internally."""
        sig_id = str(signal_row["id"])
        direction = _normalize_direction(signal_row.get("direction"))
        entry = float(signal_row.get("trigger_price", 0.0) or 0.0)

        try:
            sig_ts_ns = int(_to_utc_ts(signal_row["timestamp_event"]).value)
        except Exception:
            return LabelOutcome(
                signal_id=sig_id, status=LabelStatus.SKIP,
                reason="invalid_timestamp", is_labeled=0,
                label_win_pct=0.0, label_loss_pct=0.0,
                exit_price=entry, exit_time=None,
                timeout_return_pct=0.0, ambiguity_flag=0,
            )

        ts_ns_arr = candles_df["timestamp"].values.astype("int64")
        high_arr = candles_df["high"].values.astype("float64")
        low_arr = candles_df["low"].values.astype("float64")
        close_arr = candles_df["close"].values.astype("float64")

        row = _label_one_raw(
            sig_id=sig_id, direction=direction, entry=entry,
            sig_ts_ns=sig_ts_ns,
            ts_ns_arr=ts_ns_arr, high_arr=high_arr,
            low_arr=low_arr, close_arr=close_arr,
            config_horizon=self.config.horizon_candles,
            config_tp_pct=self.config.tp_pct,
            config_sl_pct=self.config.sl_pct,
            config_roundtrip_cost=self.config.roundtrip_cost_pct,
            config_same_candle_policy=self.config.same_candle_policy.value,
            config_min_forward=self.config.min_forward_candles_required,
        )

        status_map = {
            "WIN": LabelStatus.WIN, "LOSS": LabelStatus.LOSS,
            "TIMEOUT": LabelStatus.TIMEOUT, "SKIP": LabelStatus.SKIP,
        }
        return LabelOutcome(
            signal_id=sig_id,
            status=status_map[row[3]],
            reason=row[4],
            is_labeled=row[2],
            label_win_pct=row[0],
            label_loss_pct=row[1],
            exit_price=row[5],
            exit_time=row[6],
            timeout_return_pct=row[10],
            ambiguity_flag=row[11],
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def label_signals(
        self,
        symbol: Optional[str] = None,
        timeframe_secs: Optional[int] = None,
    ) -> dict[str, int]:
        self.ensure_schema()

        candles_df = self.load_candles(
            symbol=symbol, timeframe_secs=timeframe_secs)
        if candles_df.empty:
            raise RuntimeError(
                "No candles available in SQLite table `candles`. "
                "Run historical backfill with candle persistence first."
            )

        # Build per-group numpy arrays once (avoids repeated DataFrame slicing in workers)
        candles_groups: dict[tuple[str, int], dict] = {}
        for (sym, tf), grp in candles_df.groupby(["symbol", "timeframe_secs"], sort=False):
            grp_sorted = grp.sort_values("timestamp")
            candles_groups[(str(sym).lower(), int(tf))] = {
                "ts_ns":  grp_sorted["timestamp"].values.astype("int64"),
                "high":   grp_sorted["high"].values.astype("float64"),
                "low":    grp_sorted["low"].values.astype("float64"),
                "close":  grp_sorted["close"].values.astype("float64"),
            }

        # Load all signals into memory (grouped by asset+timeframe)
        where, params = [], []
        if not self.config.relabel_all:
            where.append("COALESCE(is_labeled, 0) = 0")
        if symbol:
            where.append("LOWER(asset) = LOWER(?)")
            params.append(symbol)

        where_sql = f"WHERE {' AND '.join(where)}" if where else ""
        query = f"""
            SELECT id, timestamp_event, asset, timeframe_secs, direction, trigger_price
            FROM signals {where_sql}
        """
        signals_df = pd.read_sql_query(query, self.conn, params=params)

        total = len(signals_df)
        print(
            f"Total signals to label: {total:,}  |  workers: {self.n_workers}")

        # Group signals by (asset, timeframe) and build worker argument tuples
        worker_args = []
        for (sym, tf), sig_grp in signals_df.groupby(
            [signals_df["asset"].str.lower(), signals_df["timeframe_secs"]], sort=False
        ):
            key = (str(sym).lower(), int(tf))
            arrays = candles_groups.get(key)
            if arrays is None:
                # Try symbol-only fallback
                fallback = next(
                    (v for (s, _), v in candles_groups.items()
                     if s == str(sym).lower()),
                    None,
                )
                if fallback is None:
                    logger.warning(
                        "No candles for (%s, %s) — all SKIP", sym, tf)
                    continue
                arrays = fallback

            worker_args.append((
                sig_grp.to_dict(orient="records"),
                arrays["ts_ns"],
                arrays["high"],
                arrays["low"],
                arrays["close"],
                self.config.horizon_candles,
                self.config.tp_pct,
                self.config.sl_pct,
                self.config.roundtrip_cost_pct,
                self.config.same_candle_policy.value,
                self.config.min_forward_candles_required,
            ))

        stats = {"processed": 0, "WIN": 0, "LOSS": 0, "TIMEOUT": 0, "SKIP": 0}

        try:
            from tqdm import tqdm
        except ImportError:
            def tqdm(iterable, **kwargs):
                return iterable

        UPDATE_SQL = """
            UPDATE signals
            SET
                label_win_pct = ?, label_loss_pct = ?, is_labeled = ?, label_status = ?,
                label_reason = ?, label_exit_price = ?, label_exit_time = ?,
                label_horizon_candles = ?, label_tp_pct = ?, label_sl_pct = ?,
                label_timeout_return_pct = ?, label_ambiguity_flag = ?
            WHERE id = ?
        """

        use_mp = self.n_workers > 1 and len(worker_args) > 1

        if use_mp:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            print(f"Launching {self.n_workers} worker processes...")
            pbar = tqdm(total=total, desc="Labeling", unit="sig")
            with ProcessPoolExecutor(max_workers=self.n_workers) as pool:
                futures = {pool.submit(
                    _label_group_worker, arg): arg for arg in worker_args}
                for fut in as_completed(futures):
                    try:
                        updates, grp_stats = fut.result()
                    except Exception as exc:
                        logger.error(
                            "Worker failed: %s — falling back to SKIP for group", exc)
                        continue

                    self.conn.executemany(UPDATE_SQL, updates)
                    self.conn.commit()

                    n = len(updates)
                    stats["processed"] += n
                    for k in ("WIN", "LOSS", "TIMEOUT", "SKIP"):
                        stats[k] += grp_stats.get(k, 0)
                    pbar.update(n)
            pbar.close()
        else:
            # Single-threaded fallback (e.g. only 1 group or 1 CPU)
            print("Running single-threaded (1 group or 1 CPU)...")
            pbar = tqdm(total=total, desc="Labeling", unit="sig")
            for arg in worker_args:
                updates, grp_stats = _label_group_worker(arg)
                self.conn.executemany(UPDATE_SQL, updates)
                self.conn.commit()
                n = len(updates)
                stats["processed"] += n
                for k in ("WIN", "LOSS", "TIMEOUT", "SKIP"):
                    stats[k] += grp_stats.get(k, 0)
                pbar.update(n)
            pbar.close()

        return stats


def relabel_sqlite(
    db_path: str,
    config: LabelerConfig,
    symbol: Optional[str] = None,
    timeframe_secs: Optional[int] = None,
    n_workers: Optional[int] = None,
) -> dict[str, int]:
    """Convenience entry point for scripts/CLI."""
    conn = sqlite3.connect(db_path)
    try:
        labeler = SQLiteSignalLabeler(
            conn=conn, config=config, n_workers=n_workers)
        stats = labeler.label_signals(
            symbol=symbol, timeframe_secs=timeframe_secs)
        logger.info("Labeling complete: %s", stats)
        return stats
    finally:
        conn.close()
