"""
config.py
=========
Central configuration for the AMT engine.
Edit this file to change symbols, timeframes, tick sizes and DB paths.
All modules import from here — no more hardcoded values scattered around.
"""

# ── Asset Configuration ────────────────────────────────────────────────────────
SYMBOL          = "btcusdt"
TICK_SIZE       = 0.1          # minimum price bucket granularity
TIMEFRAME_SECS  = 900          # candle duration in seconds (900 = 15 min)

# ── Volume Profile ─────────────────────────────────────────────────────────────
VALUE_AREA_PCT  = 0.68         # 68% of volume = value area

# ── ML / Backfill ─────────────────────────────────────────────────────────────
DB_PATH                 = "amt_ml_dataset.db"
LOOK_FORWARD_MINUTES    = 15
FLUSH_EVERY             = 200

# ── Historical Runner ─────────────────────────────────────────────────────────
BACKFILL_START      = "2020-01-01"
BACKFILL_END        = "2026-03-29"   # update to yesterday before each run
PARALLEL_DOWNLOADS  = 16
AGGREGATE_SECS      = 1
