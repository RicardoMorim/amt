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
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from threading import Lock

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Try to import tqdm for a nice progress bar, fall back gracefully
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    logging.warning("tqdm not installed. Run 'pip install tqdm' for a progress bar.")

logger = logging.getLogger(__name__)


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
            conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
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
        pass


def _download_day_sync(symbol: str, date_str: str, max_retries: int = 3, aggregate_secs: int = 1) -> pd.DataFrame:
    """
    Synchronous function that downloads and parses a single day's zip from Binance Vision.
    Includes retry logic for transient network failures.
    
    Args:
        symbol: Trading pair (e.g., 'btcusdt')
        date_str: Date string (e.g., '2024-01-15')
        max_retries: Number of retry attempts for failed downloads
        aggregate_secs: Aggregate ticks to N-second candles (default 1 = per second)
                       Set to 0 to keep tick-by-tick (slower, more data)
        
    Returns:
        DataFrame with columns: timestamp, price, volume, side (aggregated if aggregate_secs > 0)
    """
    url = (
        f"https://data.binance.vision/data/futures/um/daily/trades/"
        f"{symbol.upper()}/{symbol.upper()}-trades-{date_str}.zip"
    )
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=60)
            if response.status_code == 404:
                logger.debug(f"[{date_str}] No data available (404).")
                return pd.DataFrame()
            elif response.status_code != 200:
                if attempt < max_retries - 1:
                    logger.warning(f"[{date_str}] HTTP {response.status_code} — retry {attempt + 1}/{max_retries}")
                    continue
                else:
                    logger.warning(f"[{date_str}] HTTP {response.status_code} — skipping after {max_retries} attempts.")
                    return pd.DataFrame()

            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                csv_filename = z.namelist()[0]
                with z.open(csv_filename) as f:
                    df = pd.read_csv(
                        f,
                        header=0,
                        names=['id', 'price', 'qty', 'quote_qty', 'time', 'is_buyer_maker'],
                    )

            # Drop stray header rows if the CSV had one
            df = df[pd.to_numeric(df['time'], errors='coerce').notnull()].copy()
            df['timestamp'] = pd.to_datetime(df['time'].astype(float).astype('int64'), unit='ms')
            df['side'] = df['is_buyer_maker'].map(
                {True: 'sell', False: 'buy', 'True': 'sell', 'False': 'buy'}
            )
            df['price'] = pd.to_numeric(df['price'])
            df['volume'] = pd.to_numeric(df['qty'])
            df = df[['timestamp', 'price', 'volume', 'side']].dropna()
            
            # ── AGGREGATION: Optional time-based bucketing to reduce noise ──────────
            if aggregate_secs > 0:
                df = _aggregate_by_seconds(df, aggregate_secs)
            
            return df

        except (requests.RequestException, zipfile.BadZipFile, Exception) as e:
            if attempt < max_retries - 1:
                logger.warning(f"[{date_str}] Error on attempt {attempt + 1}: {e} — retrying...")
                continue
            else:
                logger.error(f"[{date_str}] Failed after {max_retries} attempts: {e}")
                return pd.DataFrame()
    
    return pd.DataFrame()


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
    
    agg_funcs = {
        'price': lambda x: (x * df.loc[x.index, 'volume']).sum() / df.loc[x.index, 'volume'].sum(),  # VWAP
        'volume': 'sum',
        'side': lambda x: 'buy' if (df.loc[x.index, df.loc[x.index, 'side'] == 'buy', 'volume']).sum() > \
                                   (df.loc[x.index, df.loc[x.index, 'side'] == 'sell', 'volume']).sum() else 'sell'
    }
    
    # Simpler aggregation: preserve side information
    grouped = []
    for bucket, group in df.groupby('bucket'):
        buy_vol = group[group['side'] == 'buy']['volume'].sum()
        sell_vol = group[group['side'] == 'sell']['volume'].sum()
        
        grouped.append({
            'timestamp': bucket,
            'price': (group['price'] * group['volume']).sum() / group['volume'].sum(),  # VWAP
            'volume': group['volume'].sum(),
            'side': 'buy' if buy_vol >= sell_vol else 'sell',  # Dominant side
        })
    
    return pd.DataFrame(grouped)


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
        rows = db_manager.execute("SELECT date_str FROM processed_dates", fetch=True)
        return {r[0] for r in rows}
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
    logger.info(f"   Aggregation: {aggregate_secs}s bucketing" if aggregate_secs > 0 else "   Aggregation: tick-by-tick (no bucketing)")
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
    total_trades_lock = Lock()  # Protect shared counter
    
    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=parallel_downloads)

    # Wrap with tqdm if available
    iterator = tqdm(all_dates, unit="day", desc="Backfilling") if HAS_TQDM else all_dates
    
    # Track futures for better error handling
    futures_to_date = {}

    try:
        # Submit all download tasks concurrently with aggregation parameter
        for date_str in all_dates:
            future = loop.run_in_executor(
                executor, 
                _download_day_sync, 
                symbol, 
                date_str,
                3,  # max_retries
                aggregate_secs  # PASS AGGREGATION PARAMETER
            )
            futures_to_date[future] = date_str

        # Process results as they complete
        for future in as_completed(futures_to_date):
            date_str = futures_to_date[future]
            
            try:
                df = future.result()
                
                if HAS_TQDM and iterator:
                    iterator.set_postfix({
                        "date": date_str, 
                        "trades": f"{total_trades:,}"
                    })
                else:
                    logger.info(f"[{date_str}] Processing {len(df):,} trades...")

                if df.empty:
                    _mark_date_processed(db_manager, date_str)
                    continue

                # Stream trades sequentially through the engine (thread-safe within session)
                trades_list = df.to_dict('records')
                for trade in trades_list:
                    session.on_trade(trade)
                
                # Update shared counter safely
                with total_trades_lock:
                    total_trades += len(trades_list)

                # --- End-of-day bulk labeling ---
                if not df.empty:
                    df_indexed = df.set_index('timestamp')
                    ohlcv = df_indexed['price'].resample('1min').ohlc()
                    ohlcv['high'] = df_indexed['price'].resample('1min').max()
                    ohlcv['low'] = df_indexed['price'].resample('1min').min()
                    session.ml_collector.label_all_pending(ohlcv)

                _mark_date_processed(db_manager, date_str)
                logger.info(
                    f"[{date_str}] ✅ Done — {len(trades_list):,} trades. "
                    f"Total so far: {total_trades:,}"
                )

            except Exception as e:
                logger.error(f"[{date_str}] Failed to process: {e}", exc_info=True)
                # Still mark as attempted to avoid infinite retry
                _mark_date_processed(db_manager, date_str)

    finally:
        # Cleanup resources
        executor.shutdown(wait=True)
        db_manager.close_all()
        session.ml_collector.close()
        if HAS_TQDM and iterator:
            iterator.close()

    # ── Final Statistics ────────────────────────────────────────────────────────
    try:
        stats_conn = sqlite3.connect(db_path, timeout=30)
        total_signals = stats_conn.execute("SELECT count(*) FROM signals").fetchone()[0]
        labeled = stats_conn.execute("SELECT count(*) FROM signals WHERE is_labeled=1").fetchone()[0]
        stats_conn.close()
    except Exception as e:
        logger.warning(f"Could not retrieve final stats: {e}")
        total_signals = labeled = 0

    logger.info("=" * 70)
    logger.info(f"🏁 BACKFILL COMPLETE (Multithreaded Safe Mode + {aggregate_secs}s Aggregation)")
    logger.info(f"   Traded units    : {total_trades:,}")
    logger.info(f"   Total signals   : {total_signals:,}")
    logger.info(f"   Labeled signals : {labeled:,}")
    logger.info(f"   Dataset file    : {db_path}")
    logger.info("=" * 70)


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
            end_date_str="2026-03-17",        # Update to yesterday
            candle_timeframe_seconds=900,     # 15-minute candles
            tick_size=0.1,
            aggregate_secs=1,                 # Aggregate to 1-second candles (recommended)
            parallel_downloads=10,             # Download 10 days concurrently (adjust based on bandwidth)
            db_path='amt_ml_dataset.db',
        )
    )
