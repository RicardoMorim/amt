"""
ml/relabel.py
=============
Post-hoc re-labeling of all signals in amt_ml_dataset.db.

Downloads lightweight 15m OHLCV candles from Binance Klines API
(NOT tick data — fast!) and labels each signal based on:

  - label_win_pct  : max favorable move in next N candles
  - label_loss_pct : max adverse move in next N candles (as negative pct)

Usage:
    python ml/relabel.py
"""

import sqlite3
import time
import requests
import pandas as pd
from datetime import datetime, timezone

# ── Config ─────────────────────────────────────────────────────────────────────
DB_PATH        = "amt_ml_dataset.db"
SYMBOL         = "BTCUSDT"
INTERVAL       = "15m"          # candle size (match candle_timeframe_seconds=900)
FORWARD_CANDLES = 8             # look 8×15m = 2 hours forward
KLINES_URL     = "https://fapi.binance.com/fapi/v1/klines"

# ── Klines cache (avoid re-downloading same day repeatedly) ────────────────────
_klines_cache: dict[str, pd.DataFrame] = {}


def fetch_klines(symbol: str, start_ms: int, limit: int = 20) -> pd.DataFrame:
    """Fetch OHLCV candles from Binance Futures starting at start_ms."""
    params = {
        "symbol":    symbol,
        "interval":  INTERVAL,
        "startTime": start_ms,
        "limit":     limit,
    }
    r = requests.get(KLINES_URL, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","quote_vol","trades","taker_base","taker_quote","ignore"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["high"]  = df["high"].astype(float)
    df["low"]   = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    return df.set_index("open_time")


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

    # Parse timestamp → ms
    try:
        if "." in ts_str:
            ts_str = ts_str.split(".")[0]
        dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except Exception:
        return 0.0, 0.0

    start_ms = int(dt.timestamp() * 1000)

    try:
        df = fetch_klines(SYMBOL, start_ms, limit=FORWARD_CANDLES + 2)
        time.sleep(0.05)  # gentle rate limiting
    except Exception as e:
        print(f"  ⚠️  Klines fetch error: {e}")
        return 0.0, 0.0

    if df.empty or len(df) < 2:
        return 0.0, 0.0

    # Skip first candle (entry candle), use next FORWARD_CANDLES
    forward = df.iloc[1:FORWARD_CANDLES + 1]
    highs   = forward["high"].values
    lows    = forward["low"].values

    if direction == "buy":
        win_pct  = (highs.max() - entry) / entry   # max upside
        loss_pct = (lows.min()  - entry) / entry   # max downside (negative)
    else:  # sell / short
        win_pct  = (entry - lows.min())  / entry   # max downside profit
        loss_pct = (entry - highs.max()) / entry   # max adverse (negative)

    return float(win_pct), float(loss_pct)


def relabel_all(db_path: str = DB_PATH, batch_size: int = 50):
    conn   = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Fetch all signals (re-label everything for clean results)
    rows = cursor.execute("""
        SELECT id, timestamp_event, direction, trigger_price
        FROM signals
        ORDER BY timestamp_event ASC
    """).fetchall()

    total   = len(rows)
    updated = 0
    errors  = 0

    print(f"🏷️  Re-labeling {total:,} signals with {FORWARD_CANDLES}×{INTERVAL} forward window...")
    print(f"   Symbol: {SYMBOL} | Interval: {INTERVAL}")
    print("=" * 60)

    for i, (sig_id, ts_str, direction, trigger_price) in enumerate(rows, 1):
        row = {
            "timestamp_event": ts_str,
            "direction":       direction,
            "trigger_price":   trigger_price,
        }

        win_pct, loss_pct = label_signal(row)

        cursor.execute("""
            UPDATE signals
            SET label_win_pct  = ?,
                label_loss_pct = ?,
                is_labeled     = 1
            WHERE id = ?
        """, (win_pct, loss_pct, sig_id))
        updated += 1

        # Progress + commit every batch
        if i % batch_size == 0:
            conn.commit()
            non_zero = sum(1 for _, _, _, _ in rows[:i] if True)
            print(f"  [{i:>5}/{total}] committed batch | last win={win_pct:.4f} loss={loss_pct:.4f}")

    conn.commit()
    conn.close()

    print("=" * 60)
    print(f"✅ Done. {updated:,} signals re-labeled.")
    print(f"   Run `python ml/trainer.py` next.")


if __name__ == "__main__":
    relabel_all()
