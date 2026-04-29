import pandas as pd
import numpy as np
import config
import os
import json

# Self-Learning Logic: Load optimized heuristcs if they exist
vol_multiplier = config.BREAKOUT_VOL_MULTIPLIER
min_body_ratio = config.BREAKOUT_MIN_BODY_RATIO

optimized_path = os.path.join(config.ML_DIR, 'optimized_heuristics.json')
if os.path.exists(optimized_path):
    try:
        with open(optimized_path, 'r') as f:
            opts = json.load(f)
            if 'vol_multiplier' in opts:
                vol_multiplier = opts['vol_multiplier']
            if 'min_body_ratio' in opts:
                min_body_ratio = opts['min_body_ratio']
    except Exception as e:
        pass


def detect_balance_breakout(current_candle, cvd_data, profile_data, lookback_df):
    if not profile_data or lookback_df.empty or cvd_data.empty:
        return None

    vah = profile_data['vah']
    val = profile_data['val']

    close  = current_candle['close']
    open_p = current_candle['open']
    high   = current_candle['high']
    low    = current_candle['low']
    vol    = current_candle['volume']

    is_breaking_up   = close > vah and open_p <= vah
    is_breaking_down = close < val and open_p >= val

    if not (is_breaking_up or is_breaking_down):
        return None

    avg_vol = lookback_df['volume'].mean()
    if vol < (avg_vol * vol_multiplier):
        return None

    current_delta = cvd_data['delta'].iloc[-1]
    if is_breaking_up   and current_delta <= 0:
        return None
    if is_breaking_down and current_delta >= 0:
        return None

    candle_range = high - low
    if candle_range == 0:
        return None
    body_size = abs(close - open_p)
    if (body_size / candle_range) < min_body_ratio:
        return None

    direction = 'LONG' if is_breaking_up else 'SHORT'
    return {
        'signal_type':   'INITIATIVE_BREAKOUT',
        'direction':     direction,
        'trigger_price': close,
        'stop_loss':     vah if direction == 'LONG' else val,
        'confidence':    'HIGH',
    }
