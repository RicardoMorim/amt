import sqlite3
import pandas as pd
import optuna
import logging
import sys
import os
import argparse
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def optimize_heuristics(db_path=config.DB_PATH, trials=30):
    """
    Optimizes volume_multiplier and min_body_ratio for INITIATIVE_BREAKOUT signals
    by joining signals with the candles table to compute missing metrics.
    """
    try:
        conn = sqlite3.connect(db_path)
        
        # Load filtered signals
        sig_query = """
        SELECT 
            asset, timeframe_secs, timestamp_event,
            volume_zscore as vol_z,
            label_status 
        FROM signals 
        WHERE signal_type = 'INITIATIVE_BREAKOUT' 
            AND is_labeled = 1 
            AND label_status IN ('WIN', 'LOSS')
        """
        signals = pd.read_sql(sig_query, conn)
        
        # Load all candles (relatively small: ~220k rows)
        candle_query = "SELECT symbol as asset, timeframe_secs, timestamp, open, high, low, close FROM candles"
        candles = pd.read_sql(candle_query, conn)
        conn.close()
        
        if signals.empty:
            logger.warning("No labeled breakout signals found.")
            return None
            
        # Normalize timestamps for merging (removing 'Z' and '+00:00' suffix issues)
        signals['ts_norm'] = pd.to_datetime(signals['timestamp_event'], utc=True)
        candles['ts_norm'] = pd.to_datetime(candles['timestamp'], utc=True)
        
        # Merge signals with candles
        df = pd.merge(
            signals, 
            candles[['asset', 'timeframe_secs', 'ts_norm', 'open', 'high', 'low', 'close']],
            on=['asset', 'timeframe_secs', 'ts_norm'],
            how='inner'
        )
        
        if df.empty or len(df) < 50:
            logger.warning(f"Not enough breakout signals with matching candle data ({len(df)}) to run optimization.")
            return None

        # Compute body_ratio: abs(close - open) / (high - low)
        candle_range = (df["high"] - df["low"]).replace(0, np.nan)
        df["body_ratio"] = (df["close"] - df["open"]).abs() / candle_range
        
        df["vol_z"] = pd.to_numeric(df["vol_z"], errors="coerce")
        df = df.dropna(subset=["vol_z", "body_ratio", "label_status"])

        def objective(trial):
            # vol_multiplier in heuristics is compared to avg, but here we use vol_z for filtering
            # We'll optimize a Z-score threshold instead, mapped roughly from the multiplier logic
            vol_z_threshold = trial.suggest_float("vol_z_threshold", 0.5, 2.5, step=0.1)
            body_threshold = trial.suggest_float("min_body_ratio", 0.4, 0.9, step=0.05)
            
            filtered = df[(df["vol_z"] >= vol_z_threshold) & (df["body_ratio"] >= body_threshold)]
            if len(filtered) < 20:
                return -1.0
            
            win_rate = len(filtered[filtered["label_status"] == "WIN"]) / len(filtered)
            # Objective: balance win rate with sample size
            return win_rate * (len(filtered) / len(df))

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=trials)
        best_params = study.best_params
        
        # Map vol_z_threshold back to a "vol_multiplier" heuristic if needed, 
        # or just save the best params found.
        logger.info(f"Optimization complete. Best params: {best_params}")
        
        output_path = os.path.join("ml", "optimized_heuristics.json")
        with open(output_path, "w") as f:
            json.dump(best_params, f, indent=2)
            
        return best_params
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optimize AMT breakout heuristics')
    parser.add_argument('--db', default=config.DB_PATH, help='Path to SQLite database')
    parser.add_argument('--trials', type=int, default=50, help='Number of Optuna trials')
    args = parser.parse_args()
    
    optimize_heuristics(db_path=args.db, trials=args.trials)

