import sqlite3
import pandas as pd
import optuna
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def optimize_heuristics(db_path=config.DB_PATH, trials=30):
    try:
        conn = sqlite3.connect(db_path)
        query = "SELECT json_extract(signal_data, '$.volume_zscore') as vol_z, json_extract(signal_data, '$.body_ratio') as body_ratio, label_status FROM signals WHERE signal_type = 'INITIATIVE_BREAKOUT' AND is_labeled = 1 AND label_status IN ('WIN', 'LOSS')"
        df = pd.read_sql(query, conn)
        conn.close()
        
        if df.empty or len(df) < 100:
            logger.warning("Not enough labeled breakout signals to run optimization.")
            return None

        df["vol_z"] = pd.to_numeric(df["vol_z"], errors="coerce")
        df["body_ratio"] = pd.to_numeric(df["body_ratio"], errors="coerce")
        df = df.dropna()

        def objective(trial):
            vol_threshold = trial.suggest_float("vol_multiplier", 1.0, 3.0, step=0.1)
            body_threshold = trial.suggest_float("min_body_ratio", 0.4, 0.9, step=0.05)
            filtered = df[(df["vol_z"] >= (vol_threshold - 1.0)) & (df["body_ratio"] >= body_threshold)]
            if len(filtered) < 50:
                return -1.0
            win_rate = len(filtered[filtered["label_status"] == "WIN"]) / len(filtered)
            return win_rate * (len(filtered) / len(df))

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=trials)
        best_params = study.best_params
        logger.info(f"Optimization complete. Best params: {best_params}")
        
        with open("ml/optimized_heuristics.json", "w") as f:
            import json
            json.dump(best_params, f, indent=2)
        return best_params
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        return None

if __name__ == "__main__":
    optimize_heuristics()

