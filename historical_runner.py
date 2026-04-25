"""
historical_runner.py
====================
Downloads years of tick-by-tick BTCUSDT Futures data from Binance Vision,
streams it through the exact same AMTSession engine as the live bot,
and populates amt_ml_dataset.db with labeled training data.

Features:
  - Parallel downloads: fetches N days concurrently (configurable)
  - Resume support: skips dates already processed (tracked in SQLite)
  - Progress bar via tqdm (install with: pip install tqdm)
  - Configurable symbol, date range, and timeframe

Performance fixes (2026-03-30):
  - _aggregate_by_seconds: replaced Python loop with vectorized pandas resample
    (10-50x speedup on large tick datasets)
  - SQLite WAL mode + NORMAL synchronous pragma for faster concurrent writes
  - SQLite cache_size and mmap_size tuned for large datasets
  - Default parallel_downloads increased to 16
  - Daily profile reset: SessionProfileManager.reset() called between days
"""

import asyncio
import io
import logging
import sqlite3
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from threading import Lock

import pandas as pd
import numpy as np
import requests

import sys
import config


class UnbufferedHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.stream.flush()


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[UnbufferedHandler(sys.stdout)]
)

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

    class tqdm:  # type: ignore
        def __init__(self, iterable, **kwargs):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable)

        def update(self, n=1): pass
        def set_postfix(self, d): pass
        def close(self): pass
    logging.warning("tqdm not installed. Run 'pip install tqdm' for a progress bar.")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ───────────────────────────────────────────────────────────────────────────────
# THREAD-SAFE DATABASE MANAGER
# ───────────────────────────────────────────────────────────────────────────────
class ThreadSafeDBManager:
    """Thread-safe SQLite with WAL mode and performance pragmas per connection."""

    def __init__(self, db_path: str, pool_size: int = 5):
        self.db_path = db_path
        self.pool_size = pool_size
        self.lock = Lock()
        self._local = threading.local()

    def get_connection(self):
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
            conn.isolation_level = None
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=-131072")   # 128 MB
            conn.execute("PRAGMA mmap_size=268435456")  # 256 MB
            conn.execute("PRAGMA temp_store=MEMORY")
            self._local.conn = conn
        return self._local.conn

    def execute(self, query: str, params: tuple = (), fetch: bool = False):
        with self.lock:
            conn = self.get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(query, params)
                if fetch:
                    return cursor.fetchall()
                conn.commit()
                return None
            except sqlite3.OperationalError as e:
                logger.warning(f"DB operational error (retry): {e}")
                conn.commit()
                raise

    def close_all(self):
        if hasattr(self._local, 'conn') and self._local.conn:
            self._local.conn.close()
            self._local.conn = None


class EmptyAlertSender:
    def send(self, signal):
        pass


def _download_day_sync(
    symbol: str,
    date_str: str,
    max_retries: int = 3,
    aggregate_secs: int = 1,
    timeout_secs: int = 180,
) -> pd.DataFrame:
    """
    Download and parse a single day's zip from Binance Vision.
    Includes retry logic with exponential backoff.
    """
    import threading
    url = (
        f"https://data.binance.vision/data/futures/um/daily/trades/"
        f"{symbol.upper()}/{symbol.upper()}-trades-{date_str}.zip"
    )

    thread_id = threading.current_thread().name
    logger.info(f"[{thread_id}] [{date_str}] ⬇️ Starting download...")

    for attempt in range(max_retries):
        try:
            logger.info(
                f"[{thread_id}] [{date_str}] 🌐 Attempt {attempt + 1}/{max_retries}: requesting from Binance Vision...")
            import time as time_module
            start_req = time_module.time()
            response = requests.get(url, timeout=timeout_secs)
            elapsed = time_module.time() - start_req
            logger.info(f"[{thread_id}] [{date_str}] ✓ HTTP {response.status_code} (took {elapsed:.1f}s)")

            if response.status_code == 404:
                logger.info(f"[{thread_id}] [{date_str}] ℹ️ No data (404)")
                return pd.DataFrame()
            elif response.status_code != 200:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"[{thread_id}] [{date_str}] ⚠️ HTTP {response.status_code} — retry in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"[{thread_id}] [{date_str}] ❌ HTTP {response.status_code} — giving up")
                    return pd.DataFrame()

            logger.info(f"[{thread_id}] [{date_str}] 📦 Parsing ZIP file...")
            t_parse_start = time.time()

            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                csv_filename = z.namelist()[0]
                with z.open(csv_filename) as f:
                    df = pd.read_csv(
                        f,
                        header=0,
                        names=['id', 'price', 'qty', 'quote_qty', 'time', 'is_buyer_maker'],
                    )

            df = df[pd.to_numeric(df['time'], errors='coerce').notnull()].copy()
            df['timestamp'] = pd.to_datetime(df['time'].astype(float).astype('int64'), unit='ms')
            df['side'] = df['is_buyer_maker'].map(
                {True: 'sell', False: 'buy', 'True': 'sell', 'False': 'buy'}
            )
            df['price'] = pd.to_numeric(df['price'])
            df['volume'] = pd.to_numeric(df['qty'])
            df = df[['timestamp', 'price', 'volume', 'side']].dropna()

            t_agg_start = time.time()
            if aggregate_secs > 0:
                df = _aggregate_by_seconds(df, aggregate_secs)
            t_agg = time.time() - t_agg_start

            t_parse_total = time.time() - t_parse_start
            logger.info(
                f"[{thread_id}] [{date_str}] ✅ SUCCESS: {len(df):,} aggregated units "
                f"(parsed+agg: {t_parse_total:.2f}s, agg: {t_agg:.2f}s)"
            )
            return df

        except requests.Timeout:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.warning(f"[{thread_id}] [{date_str}] ⏱️ Timeout — waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                logger.error(f"[{thread_id}] [{date_str}] ❌ Timeout after {max_retries} attempts")
                return pd.DataFrame()
        except (requests.RequestException, zipfile.BadZipFile) as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.warning(f"[{thread_id}] [{date_str}] ⚠️ {type(e).__name__} — waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                logger.error(f"[{thread_id}] [{date_str}] ❌ Failed: {type(e).__name__}: {str(e)[:150]}")
                return pd.DataFrame()

    return pd.DataFrame()


def _download_with_date(
    symbol: str,
    date_str: str,
    max_retries: int,
    aggregate_secs: int,
    timeout_secs: int,
) -> tuple[str, pd.DataFrame]:
    df = _download_day_sync(symbol, date_str, max_retries, aggregate_secs, timeout_secs)
    return date_str, df


def _aggregate_by_seconds(df: pd.DataFrame, secs: int) -> pd.DataFrame:
    """
    Aggregate tick-by-tick trades into N-second buckets using VWAP for price.
    Fully vectorized with pandas resample — 10-50x faster than a Python loop.
    """
    if df.empty:
        return df

    df = df.set_index('timestamp')
    rule = f'{secs}s'

    weighted_price = (df['price'] * df['volume']).resample(rule).sum()
    total_volume   = df['volume'].resample(rule).sum()
    vwap           = weighted_price / total_volume

    # Fix: compute buy/sell volume on the full resampled index to avoid
    # misaligned-Series errors when one side has zero trades in a bucket.
    buy_vol  = df.loc[df['side'] == 'buy', 'volume'].resample(rule).sum()
    sell_vol = df.loc[df['side'] == 'sell', 'volume'].resample(rule).sum()

    # Reindex both to the full resampled index so they share identical labels
    idx = weighted_price.index
    buy_vol  = buy_vol.reindex(idx, fill_value=0.0)
    sell_vol = sell_vol.reindex(idx, fill_value=0.0)

    dominant_side = pd.Series(
        np.where(buy_vol >= sell_vol, 'buy', 'sell'),
        index=idx, name='side'
    )

    result = pd.DataFrame({
        'price':  vwap,
        'volume': total_volume,
        'side':   dominant_side,
    }).dropna(subset=['price', 'volume']).reset_index()

    result['side'] = result['side'].fillna('buy')
    return result


def _mark_date_processed(db_manager: ThreadSafeDBManager, date_str: str):
    try:
        db_manager.execute(
            "INSERT OR IGNORE INTO processed_dates (date_str) VALUES (?)",
            (date_str,)
        )
    except Exception as e:
        logger.error(f"Failed to mark {date_str} as processed: {e}")


def _get_processed_dates(db_manager: ThreadSafeDBManager) -> set:
    try:
        rows = db_manager.execute("SELECT date_str FROM processed_dates", fetch=True)
        return {r[0] for r in rows} if rows else set()
    except Exception as e:
        logger.warning(f"Could not retrieve processed dates: {e}")
        return set()


def _ensure_resume_table(db_manager: ThreadSafeDBManager):
    try:
        db_manager.execute(
            "CREATE TABLE IF NOT EXISTS processed_dates (date_str TEXT PRIMARY KEY)"
        )
    except Exception as e:
        logger.error(f"Failed to create resume table: {e}")


def _ensure_candles_table(db_manager: ThreadSafeDBManager):
    """Create local candles table used by robust historical labeler."""
    try:
        db_manager.execute(
            """
            CREATE TABLE IF NOT EXISTS candles (
                symbol TEXT NOT NULL,
                timeframe_secs INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                PRIMARY KEY (symbol, timeframe_secs, timestamp)
            )
            """
        )
    except Exception as e:
        logger.error(f"Failed to create candles table: {e}")


def _upsert_candles(
    db_manager: ThreadSafeDBManager,
    symbol: str,
    timeframe_secs: int,
    ohlcv_df: pd.DataFrame,
):
    """Insert/ignore candles into SQLite for offline deterministic relabeling."""
    if ohlcv_df.empty:
        return

    rows = []
    for ts, row in ohlcv_df.iterrows():
        if pd.isna(row.get('open')) or pd.isna(row.get('high')) or pd.isna(row.get('low')) or pd.isna(row.get('close')):
            continue
        ts_dt = pd.Timestamp(ts)
        if ts_dt.tzinfo is None:
            ts_iso = ts_dt.tz_localize('UTC').isoformat()
        else:
            ts_iso = ts_dt.tz_convert('UTC').isoformat()
        rows.append((
            str(symbol).lower(),
            int(timeframe_secs),
            ts_iso,
            float(row['open']),
            float(row['high']),
            float(row['low']),
            float(row['close']),
            float(row.get('volume', 0.0) or 0.0),
        ))

    if not rows:
        return

    with db_manager.lock:
        conn = db_manager.get_connection()
        conn.executemany(
            """
            INSERT OR IGNORE INTO candles
            (symbol, timeframe_secs, timestamp, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()


async def run_historical_backfill(
    symbol: str,
    start_date_str: str,
    end_date_str: str,
    candle_timeframe_seconds: int = 900,
    tick_size: float = 0.1,
    parallel_downloads: int = 16,
    db_path: str = 'amt_ml_dataset.db',
    aggregate_secs: int = 1,
):
    """
    Streams historical tick data through the AMT engine to populate the ML dataset.
    The session's volume profile is reset between days so each day's POC/VAH/VAL
    reflects only that day's traded volume — matching live-bot behaviour.
    """
    from main import AMTSession

    logger.info(f"🚀 Starting historical backfill for {symbol.upper()}")
    logger.info(f"   Threads: {parallel_downloads} parallel downloads")
    logger.info(
        f"   Aggregation: {aggregate_secs}s bucketing"
        if aggregate_secs > 0
        else "   Aggregation: tick-by-tick (no bucketing)"
    )
    logger.info(f"   Python Version: {__import__('sys').version.split()[0]}")

    db_manager = ThreadSafeDBManager(db_path, pool_size=parallel_downloads + 2)
    _ensure_resume_table(db_manager)
    _ensure_candles_table(db_manager)
    already_done = _get_processed_dates(db_manager)

    session = AMTSession(
        symbol=symbol.lower(),
        source='binance',
        candle_timeframe_seconds=candle_timeframe_seconds,
        alert_dispatcher=EmptyAlertSender(),
        tick_size=tick_size,
        preload_history=False,
    )

    start = datetime.strptime(start_date_str, "%Y-%m-%d")
    end   = datetime.strptime(end_date_str,   "%Y-%m-%d")
    all_dates = []
    current = start
    while current <= end:
        ds = current.strftime("%Y-%m-%d")
        if ds not in already_done:
            all_dates.append(ds)
        current += timedelta(days=1)

    total_days = len(all_dates)
    skipped = (end - start).days + 1 - total_days
    logger.info(
        f"📅 Date range: {start_date_str} → {end_date_str} "
        f"({(end-start).days+1} days total, {skipped} already done, {total_days} to fetch)"
    )

    if not all_dates:
        logger.info("✅ All dates already processed. Nothing to do!")
        return

    total_trades = 0
    total_trades_lock = Lock()

    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(
        max_workers=parallel_downloads,
        thread_name_prefix="HistoricalThread"
    )

    elapsed_total = 0

    try:
        logger.info(f"📤 Submitting {total_days} download tasks to {parallel_downloads} threads...")
        logger.info("=" * 70)

        all_futures = []
        for date_idx, date_str in enumerate(all_dates, 1):
            future = loop.run_in_executor(
                executor,
                _download_with_date,
                symbol,
                date_str,
                3,
                aggregate_secs,
                180,
            )
            all_futures.append(future)
            if date_idx % 100 == 0 or date_idx == 1:
                logger.info(f"   ... submitted {date_idx}/{total_days} tasks")

        logger.info("=" * 70)
        logger.info(f"✅ All {len(all_futures)} tasks submitted. Processing results as they complete...")
        logger.info("=" * 70)

        completed_count = 0
        failed_count = 0
        start_time = time.time()

        for coro in asyncio.as_completed(all_futures):
            try:
                date_str, df = await coro
                completed_count += 1
                elapsed = time.time() - start_time
                rate = completed_count / elapsed if elapsed > 0 else 0
                eta_secs = (len(all_futures) - completed_count) / rate if rate > 0 else 0

                if df.empty:
                    logger.info(
                        f"  [{completed_count:4d}/{len(all_futures)}] [{date_str}] "
                        f"ℹ️ NO DATA (404) | Rate: {rate:.2f} d/s | ETA: {int(eta_secs // 60)}m"
                    )
                    _mark_date_processed(db_manager, date_str)
                    continue

                logger.info(
                    f"  [{completed_count:4d}/{len(all_futures)}] [{date_str}] "
                    f"✅ {len(df):>6,} units | Rate: {rate:.2f} d/s | ETA: {int(eta_secs // 60)}m"
                )

                # Reset profile at start of each day so POC/VAH/VAL are day-scoped
                session.profile_mgr.reset()

                t_stream_start = time.time()
                trades_list = df.to_dict('records')
                for trade in trades_list:
                    session.on_trade(trade)
                t_stream = time.time() - t_stream_start

                with total_trades_lock:
                    total_trades += len(trades_list)

                logger.debug(f"    [{date_str}] ⏳ Streamed {len(trades_list):,} trades in {t_stream:.2f}s")

                t_label_start = time.time()
                df_indexed = df.set_index('timestamp')
                # ohlc() already creates high/low/open/close columns — no need to recalculate
                ohlcv = df_indexed['price'].resample('1min').ohlc()
                # persist session-timeframe candles for deterministic offline relabeling
                tf_rule = f"{int(candle_timeframe_seconds)}s"
                ohlcv_tf = df_indexed['price'].resample(tf_rule).ohlc()
                ohlcv_tf['volume'] = df_indexed['volume'].resample(tf_rule).sum()
                _upsert_candles(db_manager, symbol=symbol, timeframe_secs=candle_timeframe_seconds, ohlcv_df=ohlcv_tf)
                # Ensure index is timezone-naive for consistent comparison in label_all_pending
                if hasattr(ohlcv.index, 'tz') and ohlcv.index.tz is not None:
                    ohlcv.index = ohlcv.index.tz_localize(None)
                session.ml_collector.label_all_pending(ohlcv)
                t_label = time.time() - t_label_start

                logger.debug(f"    [{date_str}] 🏷️ Labeled pending signals in {t_label:.2f}s")

                _mark_date_processed(db_manager, date_str)

            except Exception as e:
                failed_count += 1
                elapsed = time.time() - start_time
                rate = completed_count / elapsed if elapsed > 0 else 0
                eta_secs = (len(all_futures) - completed_count) / rate if rate > 0 else 0
                _date = date_str if 'date_str' in dir() else "unknown"
                logger.error(
                    f"  [{completed_count + 1:4d}/{len(all_futures)}] [{_date}] "
                    f"❌ {type(e).__name__}: {str(e)[:80]} | Rate: {rate:.2f} d/s | ETA: {int(eta_secs // 60)}m"
                )
                # NOTE: Do NOT mark failed dates as processed — they need to be retried

        elapsed_total = time.time() - start_time
        logger.info("=" * 70)
        logger.info("📊 BATCH COMPLETE")
        logger.info(f"   ✅ Succeeded   : {completed_count}")
        logger.info(f"   ❌ Failed      : {failed_count}")
        logger.info(f"   ⏱️  Total time  : {elapsed_total:.0f}s ({elapsed_total // 60:.0f}m {int(elapsed_total % 60)}s)")
        if completed_count > 0:
            logger.info(f"   📈 Avg speed   : {completed_count / elapsed_total:.2f} days/sec")
        logger.info("=" * 70)

    finally:
        elapsed_total = time.time() - start_time if 'start_time' in dir() else 0
        executor.shutdown(wait=True)
        db_manager.close_all()
        session.ml_collector.close()


if __name__ == "__main__":
    asyncio.run(
        run_historical_backfill(
            symbol=config.SYMBOL,
            start_date_str=config.BACKFILL_START,
            end_date_str=config.BACKFILL_END,
            candle_timeframe_seconds=config.TIMEFRAME_SECS,
            tick_size=config.TICK_SIZE,
            aggregate_secs=config.AGGREGATE_SECS,
            parallel_downloads=config.PARALLEL_DOWNLOADS,
            db_path=config.DB_PATH,
        )
    )
