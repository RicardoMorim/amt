import pandas as pd
import numpy as np
import config


def detect_balance_breakout(current_candle, cvd_data, profile_data, lookback_df):
    """
    Detects a valid Initiative Breakout from the Value Area.
    Thresholds are read from config so they can be tuned without touching this file.

    Args:
        current_candle: dict with 'close', 'high', 'low', 'open', 'volume'.
        cvd_data: recent dataframe with 'cvd' and 'delta' columns.
        profile_data: dict with 'vah', 'val', 'poc'.
        lookback_df: dataframe with recent history to calculate averages.

    Returns:
        dict describing the signal if valid, else None.
    """
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

    # Volume expansion check (threshold from config)
    avg_vol = lookback_df['volume'].mean()
    if vol < (avg_vol * config.BREAKOUT_VOL_MULTIPLIER):
        return None

    # CVD confirmation
    current_delta = cvd_data['delta'].iloc[-1]
    if is_breaking_up   and current_delta <= 0:
        return None
    if is_breaking_down and current_delta >= 0:
        return None

    # Body/range check (threshold from config)
    candle_range = high - low
    if candle_range == 0:
        return None
    body_size = abs(close - open_p)
    if (body_size / candle_range) < config.BREAKOUT_MIN_BODY_RATIO:
        return None

    direction = 'LONG' if is_breaking_up else 'SHORT'
    return {
        'signal_type':   'INITIATIVE_BREAKOUT',
        'direction':     direction,
        'trigger_price': close,
        'stop_loss':     vah if direction == 'LONG' else val,
        'confidence':    'HIGH',
    }
