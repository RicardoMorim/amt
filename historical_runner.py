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
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Try to import tqdm for a nice progress bar, fall back gracefully
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    logging.warning("tqdm not installed. Run 'pip install tqdm' for a progress bar.")


class EmptyAlertSender:
    """Dummy alert dispatcher — suppresses console output during backfill."""
    def send(self, signal):
        pass


def _download_day_sync(symbol: str, date_str: str) -> pd.DataFrame:
    """
    Synchronous function that downloads and parses a single day's zip from Binance Vision.
    This is run inside a ThreadPoolExecutor to allow parallel downloads.
    """
    url = (
        f"https://data.binance.vision/data/futures/um/daily/trades/"
        f"{symbol.upper()}/{symbol.upper()}-trades-{date_str}.zip"
    )
    try:
        response = requests.get(url, timeout=60)
        if response.status_code != 200:
            logging.warning(f"[{date_str}] HTTP {response.status_code} — skipping.")
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
        return df[['timestamp', 'price', 'volume', 'side']].dropna()

    except Exception as e:
        logging.error(f"[{date_str}] Download/parse failed: {e}")
        return pd.DataFrame()


def _mark_date_processed(conn, date_str: str):
    """Records a date as fully processed so we can resume interrupted runs."""
    conn.execute(
        "INSERT OR IGNORE INTO processed_dates (date_str) VALUES (?)", (date_str,)
    )
    conn.commit()


def _get_processed_dates(conn) -> set:
    """Returns set of already-processed date strings."""
    rows = conn.execute("SELECT date_str FROM processed_dates").fetchall()
    return {r[0] for r in rows}


def _ensure_resume_table(conn):
    """Creates the processed_dates tracking table if it doesn't exist."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS processed_dates (date_str TEXT PRIMARY KEY)"
    )
    conn.commit()


async def run_historical_backfill(
    symbol: str,
    start_date_str: str,
    end_date_str: str,
    candle_timeframe_seconds: int = 900,
    tick_size: float = 0.1,
    parallel_downloads: int = 4,
):
    """
    Streams historical tick data through the AMT engine to populate the ML dataset.

    Args:
        symbol:                  Trading pair, e.g. 'btcusdt'
        start_date_str:          First day to fetch, e.g. '2024-01-01'
        end_date_str:            Last day to fetch (inclusive), e.g. '2026-03-16'
        candle_timeframe_seconds: Candle duration in seconds (default 900 = 15m)
        tick_size:               Price bucket granularity for volume profile
        parallel_downloads:      How many days to download concurrently
    """
    # Import here to avoid circular import when running as a module
    from main import AMTSession
    import sqlite3

    # ── Session Setup ───────────────────────────────────────────────────────────
    session = AMTSession(
        symbol=symbol.lower(),
        source='binance',
        candle_timeframe_seconds=candle_timeframe_seconds,
        alert_dispatcher=EmptyAlertSender(),
        tick_size=tick_size,
    )

    # ── Resume Support ──────────────────────────────────────────────────────────
    conn = session.ml_collector.conn
    _ensure_resume_table(conn)
    already_done = _get_processed_dates(conn)

    # ── Date Range ──────────────────────────────────────────────────────────────
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
    logging.info(
        f"📅 Date range: {start_date_str} → {end_date_str} "
        f"({(end-start).days+1} days total, {skipped} already done, {total_days} to fetch)"
    )

    if not all_dates:
        logging.info("✅ All dates already processed. Nothing to do!")
        return

    total_trades = 0
    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=parallel_downloads)

    # Wrap with tqdm if available
    iterator = tqdm(all_dates, unit="day", desc="Backfilling") if HAS_TQDM else all_dates

    for date_str in iterator:
        if HAS_TQDM:
            iterator.set_postfix({"date": date_str, "trades": f"{total_trades:,}"})
        else:
            logging.info(f"[{date_str}] Downloading...")

        # Download in a thread so async event loop isn't blocked
        df = await loop.run_in_executor(
            executor, _download_day_sync, symbol, date_str
        )

        if df.empty:
            _mark_date_processed(conn, date_str)  # Mark skipped days too
            continue

        # Stream trades sequentially through the engine
        trades_list = df.to_dict('records')
        for trade in trades_list:
            session.on_trade(trade)
            total_trades += 1

        _mark_date_processed(conn, date_str)
        logging.info(
            f"[{date_str}] ✅ Done — {len(trades_list):,} trades. "
            f"Total so far: {total_trades:,}"
        )

    executor.shutdown(wait=False)
    session.ml_collector.close()

    # Final stats
    import sqlite3
    conn2 = sqlite3.connect('amt_ml_dataset.db')
    total_signals = conn2.execute("SELECT count(*) FROM signals").fetchone()[0]
    labeled = conn2.execute("SELECT count(*) FROM signals WHERE is_labeled=1").fetchone()[0]
    conn2.close()

    logging.info("=" * 60)
    logging.info(f"🏁 BACKFILL COMPLETE")
    logging.info(f"   Trades processed : {total_trades:,}")
    logging.info(f"   Total signals    : {total_signals:,}")
    logging.info(f"   Labeled signals  : {labeled:,}")
    logging.info(f"   Dataset file     : amt_ml_dataset.db")
    logging.info("=" * 60)


if __name__ == "__main__":
    # ── Configure your date range here ──────────────────────────────────────────
    # Binance Futures tick data is available from ~2020 onwards.
    # Set end_date to yesterday (Binance data has a ~1 day delay).
    asyncio.run(
        run_historical_backfill(
            symbol="btcusdt",
            start_date_str="2024-01-01",   # ~2.25 years of data
            end_date_str="2026-03-16",
            candle_timeframe_seconds=900,  # 15-minute candles
            tick_size=0.1,
            parallel_downloads=4,          # Download 4 days at once
        )
    )
