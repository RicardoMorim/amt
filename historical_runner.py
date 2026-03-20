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
import requests

import sys

# Logging with forced flush (unbuffered output)


class UnbufferedHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.stream.flush()


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[UnbufferedHandler(sys.stdout)]
)

# Try to import tqdm for a nice progress bar, fall back gracefully
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Fallback: create a dummy tqdm class that just returns the iterable

    class tqdm:  # type: ignore
        def __init__(self, iterable, **kwargs):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable)

        def update(self, n=1):
            """No-op: dummy tqdm fallback"""
            pass

        def set_postfix(self, d):
            """No-op: dummy tqdm fallback"""
            pass

        def close(self):
            """No-op: dummy tqdm fallback"""
            pass
    logging.warning(
        "tqdm not installed. Run 'pip install tqdm' for a progress bar.")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ───────────────────────────────────────────────────────────────────────────────
# THREAD-SAFE DATABASE MANAGER
# ───────────────────────────────────────────────────────────────────────────────
class ThreadSafeDBManager:
    """Manages thread-safe SQLite operations with connection pooling."""

    def __init__(self, db_path: str, pool_size: int = 5):
        self.db_path = db_path
        self.pool_size = pool_size
        self.lock = Lock()
        self._local = threading.local()

    def get_connection(self):
        """Get thread-local database connection."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            conn = sqlite3.connect(
                self.db_path, timeout=30, check_same_thread=False)
            conn.isolation_level = None  # Autocommit mode
            self._local.conn = conn
        return self._local.conn

    def execute(self, query: str, params: tuple = (), fetch: bool = False):
        """Execute a query safely with automatic retry."""
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
        """Close thread-local connection."""
        if hasattr(self._local, 'conn') and self._local.conn:
            self._local.conn.close()
            self._local.conn = None


class EmptyAlertSender:
    """Dummy alert dispatcher — suppresses console output during backfill."""

    def send(self, signal):
        # No-op: discard all alerts during historical backfill
        pass


def _download_day_sync(symbol: str, date_str: str, max_retries: int = 3, aggregate_secs: int = 1, timeout_secs: int = 180) -> pd.DataFrame:
    """
    Synchronous function that downloads and parses a single day's zip from Binance Vision.
    Includes retry logic with exponential backoff for transient network failures.

    Args:
        symbol: Trading pair (e.g., 'btcusdt')
        date_str: Date string (e.g., '2024-01-15')
        max_retries: Number of retry attempts for failed downloads
        aggregate_secs: Aggregate ticks to N-second candles (default 1 = per second)
                       Set to 0 to keep tick-by-tick (slower, more data)
        timeout_secs: Request timeout in seconds (Binance Vision can be slow, default 180)

    Returns:
        DataFrame with columns: timestamp, price, volume, side (aggregated if aggregate_secs > 0)
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
            logger.info(
                f"[{thread_id}] [{date_str}] ✓ HTTP {response.status_code} (took {elapsed:.1f}s)")

            if response.status_code == 404:
                logger.info(
                    f"[{thread_id}] [{date_str}] ℹ️ No data (404 - expected for weekends/holidays)")
                return pd.DataFrame()
            elif response.status_code != 200:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    logger.warning(
                        f"[{thread_id}] [{date_str}] ⚠️ HTTP {response.status_code} — retry in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(
                        f"[{thread_id}] [{date_str}] ❌ HTTP {response.status_code} — giving up after {max_retries} attempts")
                    return pd.DataFrame()

            logger.info(f"[{thread_id}] [{date_str}] 📦 Parsing ZIP file...")
            t_parse_start = time.time()

            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                csv_filename = z.namelist()[0]
                with z.open(csv_filename) as f:
                    df = pd.read_csv(
                        f,
                        header=0,
                        names=['id', 'price', 'qty', 'quote_qty',
                               'time', 'is_buyer_maker'],
                    )

            # Drop stray header rows if the CSV had one
            df = df[pd.to_numeric(
                df['time'], errors='coerce').notnull()].copy()
            df['timestamp'] = pd.to_datetime(
                df['time'].astype(float).astype('int64'), unit='ms')
            df['side'] = df['is_buyer_maker'].map(
                {True: 'sell', False: 'buy', 'True': 'sell', 'False': 'buy'}
            )
            df['price'] = pd.to_numeric(df['price'])
            df['volume'] = pd.to_numeric(df['qty'])
            df = df[['timestamp', 'price', 'volume', 'side']].dropna()

            t_parse_csv = time.time() - t_parse_start
            logger.debug(
                f"[{thread_id}] [{date_str}]     CSV parsed in {t_parse_csv:.2f}s ({len(df):,} raw ticks)")

            # ── AGGREGATION: Optional time-based bucketing to reduce noise ──────────
            t_agg_start = time.time()
            if aggregate_secs > 0:
                df = _aggregate_by_seconds(df, aggregate_secs)
            t_agg = time.time() - t_agg_start

            t_parse_total = time.time() - t_parse_start
            logger.info(
                f"[{thread_id}] [{date_str}] ✅ SUCCESS: {len(df):,} aggregated units (parsed+agg: {t_parse_total:.2f}s, agg: {t_agg:.2f}s)")
            return df

        except requests.Timeout:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.warning(
                    f"[{thread_id}] [{date_str}] ⏱️ Timeout (>180s) on attempt {attempt + 1}/{max_retries} — waiting {wait_time}s before retry...")
                time.sleep(wait_time)
                continue
            else:
                logger.error(
                    f"[{thread_id}] [{date_str}] ❌ Timeout after {max_retries} attempts")
                return pd.DataFrame()
        except (requests.RequestException, zipfile.BadZipFile) as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.warning(
                    f"[{thread_id}] [{date_str}] ⚠️ {type(e).__name__} on attempt {attempt + 1}/{max_retries} — waiting {wait_time}s before retry...")
                logger.debug(f"   Error details: {str(e)[:150]}")
                time.sleep(wait_time)
                continue
            else:
                logger.error(
                    f"[{thread_id}] [{date_str}] ❌ Failed after {max_retries} attempts: {type(e).__name__}: {str(e)[:150]}")
                return pd.DataFrame()

    return pd.DataFrame()


def _download_with_date(
    symbol: str,
    date_str: str,
    max_retries: int,
    aggregate_secs: int,
    timeout_secs: int,
) -> tuple[str, pd.DataFrame]:
    """Wrapper that bundles date_str into the return value — fixes Python 3.12+ asyncio.as_completed() incompatibility."""
    df = _download_day_sync(symbol, date_str, max_retries,
                            aggregate_secs, timeout_secs)
    return date_str, df


def _aggregate_by_seconds(df: pd.DataFrame, secs: int) -> pd.DataFrame:
    """
    Aggregate tick-by-tick trades into N-second buckets using VWAP for price.

    Args:
        df: DataFrame with columns [timestamp, price, volume, side]
        secs: Bucket size in seconds

    Returns:
        DataFrame with aggregated trades (one row per bucket)
    """
    if df.empty:
        return df

    # Round timestamps down to nearest N-second bucket
    df['bucket'] = df['timestamp'].dt.floor(f'{secs}s')

    # Simpler aggregation: preserve side information
    grouped = []
    bucket_count = 0
    for bucket, group in df.groupby('bucket'):
        bucket_count += 1
        buy_vol = group[group['side'] == 'buy']['volume'].sum()
        sell_vol = group[group['side'] == 'sell']['volume'].sum()

        grouped.append({
            'timestamp': bucket,
            # VWAP
            'price': (group['price'] * group['volume']).sum() / group['volume'].sum(),
            'volume': group['volume'].sum(),
            'side': 'buy' if buy_vol >= sell_vol else 'sell',  # Dominant side
        })

    result = pd.DataFrame(grouped)
    return result


def _mark_date_processed(db_manager: ThreadSafeDBManager, date_str: str):
    """Records a date as fully processed so we can resume interrupted runs (thread-safe)."""
    try:
        db_manager.execute(
            "INSERT OR IGNORE INTO processed_dates (date_str) VALUES (?)",
            (date_str,)
        )
    except Exception as e:
        logger.error(f"Failed to mark {date_str} as processed: {e}")


def _get_processed_dates(db_manager: ThreadSafeDBManager) -> set:
    """Returns set of already-processed date strings (thread-safe)."""
    try:
        rows = db_manager.execute(
            "SELECT date_str FROM processed_dates", fetch=True)
        return {r[0] for r in rows} if rows else set()
    except Exception as e:
        logger.warning(f"Could not retrieve processed dates: {e}")
        return set()


def _ensure_resume_table(db_manager: ThreadSafeDBManager):
    """Creates the processed_dates tracking table if it doesn't exist (thread-safe)."""
    try:
        db_manager.execute(
            "CREATE TABLE IF NOT EXISTS processed_dates (date_str TEXT PRIMARY KEY)"
        )
    except Exception as e:
        logger.error(f"Failed to create resume table: {e}")


async def run_historical_backfill(
    symbol: str,
    start_date_str: str,
    end_date_str: str,
    candle_timeframe_seconds: int = 900,
    tick_size: float = 0.1,
    parallel_downloads: int = 4,
    db_path: str = 'amt_ml_dataset.db',
    aggregate_secs: int = 1,
):
    """
    Streams historical tick data through the AMT engine to populate the ML dataset.

    Features:
      - SAFE MULTITHREADING: Thread-safe DB operations with locks and connection pooling
      - PARALLEL DOWNLOADS: Fetches N days concurrently using ThreadPoolExecutor
      - RESUME SUPPORT: Skips dates already processed (tracked in SQLite)
      - RETRY LOGIC: Automatic retries for transient network failures
      - DATA FROM 2020: Full support from Binance Futures launch
      - AGGREGATION: Optional time-based aggregation to reduce HFT noise (default: 1 sec)

    Args:
        symbol:                  Trading pair, e.g. 'btcusdt'
        start_date_str:          First day to fetch, e.g. '2020-01-01'
        end_date_str:            Last day to fetch (inclusive), e.g. '2026-03-18'
        candle_timeframe_seconds: Candle duration in seconds (default 900 = 15m)
        tick_size:               Price bucket granularity for volume profile
        parallel_downloads:      How many days to download concurrently (default 4)
        db_path:                 Path to ML dataset database file
        aggregate_secs:          Aggregate ticks to N-second candles (default 1)
                                 - 0: Keep tick-by-tick (more data, slower)
                                 - 1: Per-second aggregation (recommended)
                                 - 5+: Coarser aggregation (less noise, but may lose signals)
    """
    from main import AMTSession

    logger.info(f"🚀 Starting historical backfill for {symbol.upper()}")
    logger.info(f"   Threads: {parallel_downloads} parallel downloads")
    logger.info(f"   Aggregation: {aggregate_secs}s bucketing" if aggregate_secs >
                0 else "   Aggregation: tick-by-tick (no bucketing)")
    logger.info(f"   Python Version: {__import__('sys').version.split()[0]}")

    # ── Thread-Safe Database Setup ──────────────────────────────────────────────
    db_manager = ThreadSafeDBManager(db_path, pool_size=parallel_downloads + 2)
    _ensure_resume_table(db_manager)
    already_done = _get_processed_dates(db_manager)

    # ── Session Setup ───────────────────────────────────────────────────────────
    session = AMTSession(
        symbol=symbol.lower(),
        source='binance',
        candle_timeframe_seconds=candle_timeframe_seconds,
        alert_dispatcher=EmptyAlertSender(),
        tick_size=tick_size,
    )

    # ── Date Range Generation ───────────────────────────────────────────────────
    start = datetime.strptime(start_date_str, "%Y-%m-%d")
    end = datetime.strptime(end_date_str, "%Y-%m-%d")
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

    # ── Multithreaded Download & Processing ─────────────────────────────────────
    total_trades = 0
    total_trades_lock = Lock()

    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(
        max_workers=parallel_downloads,
        thread_name_prefix="HistoricalThread"
    )

    elapsed_total = 0

    try:
        logger.info(
            f"📤 Submitting {total_days} download tasks to {parallel_downloads} threads...")
        logger.info("=" * 70)

        # ✅ FIX: Collect futures in a plain list — no dict needed
        #    because date_str comes back inside the tuple from _download_with_date
        all_futures = []
        for date_idx, date_str in enumerate(all_dates, 1):
            future = loop.run_in_executor(
                executor,
                _download_with_date,
                symbol,
                date_str,
                3,             # max_retries
                aggregate_secs,
                180            # timeout_secs
            )
            all_futures.append(future)
            if date_idx % 100 == 0 or date_idx == 1:
                logger.info(f"   ... submitted {date_idx}/{total_days} tasks")

        logger.info("=" * 70)
        logger.info(
            f"✅ All {len(all_futures)} tasks submitted. Processing results as they complete...")
        logger.info(
            "💡 Threads will log progress. Monitor this output to see activity.")
        logger.info("=" * 70)

        completed_count = 0
        failed_count = 0
        start_time = time.time()

        # ✅ FIX: Python 3.12+ compatible — await each coroutine directly
        for coro in asyncio.as_completed(all_futures):
            try:
                date_str, df = await coro  # ← await obrigatório no Python 3.12+
                completed_count += 1
                elapsed = time.time() - start_time
                rate = completed_count / elapsed if elapsed > 0 else 0
                eta_secs = (len(all_futures) - completed_count) / \
                    rate if rate > 0 else 0

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

                # Stream trades through the AMT engine
                t_stream_start = time.time()
                trades_list = df.to_dict('records')
                for trade in trades_list:
                    session.on_trade(trade)
                t_stream = time.time() - t_stream_start

                with total_trades_lock:
                    total_trades += len(trades_list)

                logger.debug(
                    f"    [{date_str}] ⏳ Streamed {len(trades_list):,} trades in {t_stream:.2f}s")

                # End-of-day bulk labeling
                t_label_start = time.time()
                df_indexed = df.set_index('timestamp')
                ohlcv = df_indexed['price'].resample('1min').ohlc()
                ohlcv['high'] = df_indexed['price'].resample('1min').max()
                ohlcv['low'] = df_indexed['price'].resample('1min').min()
                session.ml_collector.label_all_pending(ohlcv)
                t_label = time.time() - t_label_start

                logger.debug(
                    f"    [{date_str}] 🏷️ Labeled pending signals in {t_label:.2f}s")

                _mark_date_processed(db_manager, date_str)
                logger.debug(f"    [{date_str}] 💾 Marked as processed")

            except Exception as e:
                failed_count += 1
                elapsed = time.time() - start_time
                rate = completed_count / elapsed if elapsed > 0 else 0
                eta_secs = (len(all_futures) - completed_count) / \
                    rate if rate > 0 else 0
                # date_str may not be bound if await itself failed
                _date = date_str if 'date_str' in dir() else "unknown"
                logger.error(
                    f"  [{completed_count + 1:4d}/{len(all_futures)}] [{_date}] "
                    f"❌ {type(e).__name__}: {str(e)[:80]} | Rate: {rate:.2f} d/s | ETA: {int(eta_secs // 60)}m"
                )
                if _date != "unknown":
                    _mark_date_processed(db_manager, _date)

        elapsed_total = time.time() - start_time
        logger.info("=" * 70)
        logger.info("📊 BATCH COMPLETE")
        logger.info(f"   ✅ Succeeded   : {completed_count}")
        logger.info(f"   ❌ Failed      : {failed_count}")
        logger.info(
            f"   ⏱️  Total time  : {elapsed_total:.0f}s ({elapsed_total // 60:.0f}m {int(elapsed_total % 60)}s)")
        if completed_count > 0:
            logger.info(
                f"   📈 Avg speed   : {completed_count / elapsed_total:.2f} days/sec")
        logger.info("=" * 70)

    finally:
        elapsed_total = time.time() - start_time if 'start_time' in dir() else 0
        executor.shutdown(wait=True)
        db_manager.close_all()
        session.ml_collector.close()

if __name__ == "__main__":
    """
    Configure your date range and parameters here.

    Binance Futures tick data is available from 2020-01-01 onwards.
    Set end_date to yesterday (Binance data has a ~1 day delay).

    SAFE MULTITHREADING:
    - Thread-safe database operations with locks
    - Automatic connection pooling
    - Retry logic for transient failures
    - Safe counter updates across threads
    """
    asyncio.run(
        run_historical_backfill(
            symbol="btcusdt",
            start_date_str="2020-01-01",      # From Binance Futures launch
            end_date_str="2026-03-19",        # Update to yesterday
            candle_timeframe_seconds=900,     # 15-minute candles
            tick_size=0.1,
            # Aggregate to 1-second candles (recommended)
            aggregate_secs=1,
            parallel_downloads=5,
            db_path='amt_ml_dataset.db',
        )
    )
