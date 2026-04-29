"""
ml/retrain_scheduler.py  (D2)
==============================
Auto-retrain the model when the dataset has grown by RETRAIN_GROWTH_THRESHOLD
(default 20%) since the last training run.

Designed to be called:
  - As a daily cron job:   0 4 * * *  python ml/retrain_scheduler.py
  - After each backfill run completes
  - Manually:              python ml/retrain_scheduler.py

Writes a retrain_log.json tracking each retrain event.
"""

import json
import logging
import os
import sqlite3
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

LOG_PATH = "ml/retrain_log.json"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _get_labeled_count(db_path: str) -> int:
    try:
        conn = sqlite3.connect(db_path)
        cur  = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM signals WHERE is_labeled = 1")
        n = cur.fetchone()[0]
        conn.close()
        return n
    except Exception as e:
        logger.error(f"Cannot read DB: {e}")
        return 0


def _load_log() -> dict:
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH) as f:
            return json.load(f)
    return {'last_n_samples': 0, 'runs': []}


def _save_log(log: dict):
    os.makedirs('ml', exist_ok=True)
    with open(LOG_PATH, 'w') as f:
        json.dump(log, f, indent=2)


def maybe_retrain(
    db_path:           str   = config.DB_PATH,
    growth_threshold:  float = config.RETRAIN_GROWTH_THRESHOLD,
    min_rows:          int   = config.RETRAIN_MIN_ROWS,
    force:             bool  = False,
) -> bool:
    """
    Retrain if dataset has grown by growth_threshold since last run.
    Returns True if retrain was triggered, False otherwise.
    """
    current_n = _get_labeled_count(db_path)
    log       = _load_log()
    last_n    = log.get('last_n_samples', 0)

    logger.info(f"📊 Dataset size: {current_n:,} labeled signals (last run: {last_n:,})")

    if current_n < min_rows:
        logger.info(f"⏳ Not enough data yet ({current_n} < {min_rows} min). Skipping.")
        return False

    if not force and last_n > 0:
        growth = (current_n - last_n) / last_n
        if growth < growth_threshold:
            logger.info(
                f"📉 Growth {growth:.1%} < threshold {growth_threshold:.0%}. No retrain needed."
            )
            return False
        logger.info(f"📈 Growth {growth:.1%} ≥ threshold — triggering retrain.")
    elif force:
        logger.info("🔧 Force retrain requested.")
    else:
        logger.info("🆕 First training run.")

    # ── Retrain ────────────────────────────────────────────────────────────────
    try:
        from ml.trainer import train
        train(db_path)

        try:
            logger.info('Running post-training heuristic optimization...')
            from ml.hyper_optimizer import optimize_heuristics
            optimize_heuristics(db_path)
        except Exception as e:
            logger.error(f'Heuristic optimization failed: {e}')

        log['last_n_samples'] = current_n
        log['runs'].append({
            'timestamp': datetime.utcnow().isoformat(),
            'n_samples': current_n,
            'reason':    'force' if force else f'growth>={growth_threshold:.0%}',
        })
        _save_log(log)
        logger.info("✅ Retrain complete and log updated.")
        return True

    except Exception as e:
        logger.error(f"❌ Retrain failed: {e}")
        return False


if __name__ == '__main__':
    force = '--force' in sys.argv
    db    = next((a for a in sys.argv[1:] if not a.startswith('--')), config.DB_PATH)
    maybe_retrain(db_path=db, force=force)
