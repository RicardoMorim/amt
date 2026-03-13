import pandas as pd
import numpy as np

def detect_balance_breakout(current_candle, cvd_data, profile_data, lookback_df):
    """
    Detects if a valid Initiative Breakout from the Value Area is occurring.
    Validates the breakout with Volume and Cumulative Volume Delta (CVD) aggression.
    
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
    
    close = current_candle['close']
    open_p = current_candle['open']
    high = current_candle['high']
    low = current_candle['low']
    vol = current_candle['volume']
    
    # 1. State Verification (Are we outside VA?)
    is_breaking_up = close > vah and open_p <= vah  # Just crossed above VAH
    is_breaking_down = close < val and open_p >= val # Just crossed below VAL
    
    if not (is_breaking_up or is_breaking_down):
        return None
        
    # 2. Volume Expansion Verification
    avg_vol = lookback_df['volume'].mean()
    # A true breakout should have significant volume participation (e.g. 150% of recent average)
    if vol < (avg_vol * 1.5):
        return None
        
    # 3. CVD Confirmation (Order Flow Aggression)
    # The delta of the current breakout candle must support the direction
    current_delta = cvd_data['delta'].iloc[-1]
    
    if is_breaking_up and current_delta <= 0:
        return None # Warning: Price broke up, but delta was negative (sell limit absorption)
        
    if is_breaking_down and current_delta >= 0:
        return None # Warning: Price broke down, but delta was positive (buy limit absorption)
        
    # 4. Range / Wick Rejection Check
    # Ensure the breakout candle isn't just a massive wick (rejection). 
    # Body should be at least 60% of the total candle range.
    candle_range = high - low
    if candle_range == 0:
        return None
        
    body_size = abs(close - open_p)
    if (body_size / candle_range) < 0.6:
        return None
        
    # Valid Signal Detected
    direction = 'LONG' if is_breaking_up else 'SHORT'
    return {
        'signal_type': 'INITIATIVE_BREAKOUT',
        'direction': direction,
        'trigger_price': close,
        'stop_loss': vah if direction == 'LONG' else val, # Aggressive stop just inside the VA
        'confidence': 'HIGH'
    }
