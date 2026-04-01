"""
relabel.py
==========
Post-hoc re-labeling of signals in amt_ml_dataset.db.

Downloads lightweight 15m OHLCV candles from Binance Klines API
(NOT tick data — fast!) and labels each signal based on:

  - label_win_pct  : max favorable move in next N candles
  - label_loss_pct : max adverse move in next N candles (as negative pct)

Fixes vs previous version:
  - Timestamp parser now handles both ISO-8601 ("2024-01-01T12:00:00Z") and
    space-separated ("2024-01-01 12:00:00") formats via fromisoformat().
  - fetch_klines wrapped with retry + exponential backoff (3 attempts).
  - Rate limit raised to 0.12 s/req (~500 req/min, safely under Binance 1200/min).
  - relabel_all() now filters WHERE is_labeled = 0 by default; pass
    force=True to re-label everything.
  - Dead _klines_cache dict removed.

Usage:
    python relabel.py              # label only unlabeled rows
    python relabel.py --force      # re-label all rows
"""

import argparse
import sqlite3
import time
import requests
import pandas as pd
from datetime import datetime, timezone

# ── Config ─────────────────────────────────────────────────────────────────────
DB_PATH         = "amt_ml_dataset.db"
SYMBOL          = "BTCUSDT"
INTERVAL        = "15m"          # candle size (match candle_timeframe_seconds=900)
FORWARD_CANDLES = 8              # look 8×15m = 2 hours forward
KLINES_URL      = "https://fapi.binance.com/fapi/v1/klines"
RATE_LIMIT_SECS = 0.12           # ≤ 500 req/min (Binance limit: 1200/min)
MAX_RETRIES     = 3


def _parse_timestamp(ts_str: str) -> datetime | None:
    """
    Parse a timestamp string to a UTC-aware datetime.
    Accepts:
      - ISO-8601 with Z:          "2024-01-15T08:30:00Z"
      - ISO-8601 with offset:     "2024-01-15T08:30:00+00:00"
      - Space-separated (legacy): "2024-01-15 08:30:00"
      - Space-separated with µs:  "2024-01-15 08:30:00.123456"
    Returns None if parsing fails.
    """
    if not ts_str:
        return None
    try:
        # Normalise: replace Z, strip microseconds, replace space with T
        clean = ts_str.strip().replace("Z", "+00:00")
        if " " in clean and "T" not in clean:
            clean = clean.replace(" ", "T", 1)
        if "." in clean:
            clean = clean.split(".")[0] + "+00:00"
        dt = datetime.fromisoformat(clean)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def fetch_klines(symbol: str, start_ms: int, limit: int = 20) -> pd.DataFrame:
    """
    Fetch OHLCV candles from Binance Futures starting at start_ms.
    Retries up to MAX_RETRIES times with exponential backoff on failure.
    Raises requests.HTTPError on non-retryable server errors.
    """
    params = {
        "symbol":    symbol,
        "interval":  INTERVAL,
        "startTime": start_ms,
        "limit":     limit,
    }

    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(KLINES_URL, params=params, timeout=10)

            if r.status_code == 429:
                # Rate-limited: Binance sends Retry-After header; honour it
                retry_after = int(r.headers.get("Retry-After", 60))
                print(f"  ⏳ Rate-limited (429). Waiting {retry_after}s...")
                time.sleep(retry_after)
                continue

            r.raise_for_status()
            data = r.json()
            if not data:
                return pd.DataFrame()

            df = pd.DataFrame(data, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_vol", "trades", "taker_base", "taker_quote", "ignore"
            ])
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            df["high"]  = df["high"].astype(float)
            df["low"]   = df["low"].astype(float)
            df["close"] = df["close"].astype(float)
            return df.set_index("open_time")

        except (requests.Timeout, requests.ConnectionError) as e:
            last_exc = e
            wait = 2 ** attempt
            print(f"  ⚠️  Attempt {attempt + 1}/{MAX_RETRIES} failed ({type(e).__name__}). Retrying in {wait}s...")
            time.sleep(wait)

        except requests.HTTPError as e:
            # Non-retryable (4xx except 429, 5xx after retries)
            raise e

    raise last_exc or RuntimeError("fetch_klines exhausted retries with no exception")


def label_signal(row: dict) -> tuple[float, float]:
    """
    Compute forward win/loss pct for a single signal.
    Returns (win_pct, loss_pct) where loss_pct is negative.
    """
    ts_str    = row["timestamp_event"]
    direction = str(row.get("direction", "buy")).lower()
    entry     = float(row["trigger_price"])

    if not entry or entry == 0:
        return 0.0, 0.0

    dt = _parse_timestamp(ts_str)
    if dt is None:
        print(f"  ⚠️  Could not parse timestamp: {ts_str!r}")
        return 0.0, 0.0

    start_ms = int(dt.timestamp() * 1000)

    try:
        df = fetch_klines(SYMBOL, start_ms, limit=FORWARD_CANDLES + 2)
        time.sleep(RATE_LIMIT_SECS)
    except Exception as e:
        print(f"  ⚠️  Klines fetch error: {e}")
        return 0.0, 0.0

    if df.empty or len(df) < 2:
        return 0.0, 0.0

    # Skip first candle (entry candle), use next FORWARD_CANDLES
    forward = df.iloc[1:FORWARD_CANDLES + 1]
    highs   = forward["high"].values
    lows    = forward["low"].values

    if direction in ("buy", "long"):
        win_pct  = (highs.max() - entry) / entry   # max upside
        loss_pct = (lows.min()  - entry) / entry   # max downside (negative)
    else:  # sell / short
        win_pct  = (entry - lows.min())  / entry   # max downside profit
        loss_pct = (entry - highs.max()) / entry   # max adverse (negative)

    return float(win_pct), float(loss_pct)


def relabel_all(db_path: str = DB_PATH, batch_size: int = 50, force: bool = False):
    """
    Label signals in the database.

    Args:
        db_path:    Path to the SQLite database.
        batch_size: Number of rows per commit.
        force:      If True, re-label ALL signals (even already labeled).
                    If False (default), only label rows where is_labeled = 0.
    """
    conn   = sqlite3.connect(db_path)
    cursor = conn.cursor()

    where = "" if force else "WHERE is_labeled = 0"
    rows = cursor.execute(f"""
        SELECT id, timestamp_event, direction, trigger_price
        FROM signals
        {where}
        ORDER BY timestamp_event ASC
    """).fetchall()

    total   = len(rows)
    updated = 0
    errors  = 0

    mode = "ALL" if force else "UNLABELED ONLY"
    print(f"🏷️  Re-labeling {total:,} signals [{mode}] with {FORWARD_CANDLES}×{INTERVAL} forward window...")
    print(f"   Symbol: {SYMBOL} | Interval: {INTERVAL} | Rate limit: {RATE_LIMIT_SECS}s/req")
    print("=" * 60)

    for i, (sig_id, ts_str, direction, trigger_price) in enumerate(rows, 1):
        row = {
            "timestamp_event": ts_str,
            "direction":       direction,
            "trigger_price":   trigger_price,
        }

        win_pct, loss_pct = label_signal(row)

        if win_pct == 0.0 and loss_pct == 0.0:
            errors += 1

        cursor.execute("""
            UPDATE signals
            SET label_win_pct  = ?,
                label_loss_pct = ?,
                is_labeled     = 1
            WHERE id = ?
        """, (win_pct, loss_pct, sig_id))
        updated += 1

        if i % batch_size == 0:
            conn.commit()
            print(f"  [{i:>5}/{total}] committed batch | last win={win_pct:.4f} loss={loss_pct:.4f} | errors so far: {errors}")

    conn.commit()
    conn.close()

    print("=" * 60)
    print(f"✅ Done. {updated:,} signals labeled ({errors} returned 0.0/0.0).")
    print(f"   Run `python ml/trainer.py` next.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-label AMT signals")
    parser.add_argument("--force", action="store_true", help="Re-label all rows, including already labeled")
    parser.add_argument("--db", default=DB_PATH, help=f"DB path (default: {DB_PATH})")
    args = parser.parse_args()
    relabel_all(db_path=args.db, force=args.force)
