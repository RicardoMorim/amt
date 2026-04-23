"""
config.py — Central configuration for the AMT engine.
Edit here; all modules import from this file.
"""

# ── Asset ──────────────────────────────────────────────────────────────────────
SYMBOL          = "btcusdt"
TICK_SIZE       = 0.1
TIMEFRAME_SECS  = 900          # 15-minute candles

# ── Volume Profile ────────────────────────────────────────────────────────────
VALUE_AREA_PCT  = 0.68

# ── Signal thresholds (used in signals/balance_breakout.py) ───────────────────
BREAKOUT_VOL_MULTIPLIER  = 1.5    # candle volume must be >= avg * this
BREAKOUT_MIN_BODY_RATIO  = 0.6    # body/range must be >= this

# ── ML / Backfill ─────────────────────────────────────────────────────────────
DB_PATH                 = "amt_ml_dataset.db"
LOOK_FORWARD_MINUTES    = 15
FLUSH_EVERY             = 200

# ── Historical Runner ─────────────────────────────────────────────────────────
BACKFILL_START      = "2020-01-01"
BACKFILL_END        = "2026-03-29"
PARALLEL_DOWNLOADS  = 10
AGGREGATE_SECS      = 1

# ── ML predictor ──────────────────────────────────────────────────────────────
ML_CONFIDENCE_THRESHOLD = 0.60     # minimum model confidence to fire an alert
ML_MODEL_PATH           = "ml/amt_model.pkl"
ML_ENCODERS_PATH        = "ml/amt_encoders.pkl"
ML_META_PATH            = "ml/amt_model_meta.json"
ML_OPTUNA_TRIALS        = 50
ML_N_CV_SPLITS          = 5

# ── Retrain scheduler ─────────────────────────────────────────────────────────
RETRAIN_GROWTH_THRESHOLD = 0.20    # retrain when dataset grows by 20%
RETRAIN_MIN_ROWS         = 500
