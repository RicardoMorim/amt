import pandas as pd
import numpy as np

def detect_cvd_divergence(price_series, cvd_series, window=5):
    """
    Detects divergence between Price and CVD, typical of exhaustion (Trap zones).
    e.g. Price makes a higher high, but CVD makes a lower high.
    
    Args:
        price_series: Pandas series of recent close prices.
        cvd_series: Pandas series of recent CVD values.
        window: Number of periods to look back for the swing pivots.
        
    Returns:
        dict describing the signal if valid, else None.
    """
    if len(price_series) < window + 2:
        return None
        
    current_price = price_series.iloc[-1]
    current_cvd = cvd_series.iloc[-1]
    
    # Simple pivot logic via rolling max/min
    recent_high_price = price_series.iloc[-(window+1):-1].max()
    recent_low_price = price_series.iloc[-(window+1):-1].min()
    
    recent_high_cvd = cvd_series.iloc[-(window+1):-1].max()
    recent_low_cvd = cvd_series.iloc[-(window+1):-1].min()
    
    # BEARISH DIVERGENCE (Exhaustion of buyers)
    # Price makes higher high, but CVD is failing to make higher highs (Limit Selling Absorption)
    if current_price > recent_high_price and current_cvd < recent_high_cvd:
        return {
            'signal_type': 'CVD_DIVERGENCE_EXHAUSTION',
            'direction': 'SHORT',
            'trigger_price': current_price,
            'description': 'Price made a New High, but CVD failed (Buyer Exhaustion).'
        }
        
    # BULLISH DIVERGENCE (Exhaustion of sellers)
    # Price makes lower low, but CVD is failing to make lower lows (Limit Buying Absorption)
    if current_price < recent_low_price and current_cvd > recent_low_cvd:
        return {
            'signal_type': 'CVD_DIVERGENCE_EXHAUSTION',
            'direction': 'LONG',
            'trigger_price': current_price,
            'description': 'Price made a New Low, but CVD failed (Seller Exhaustion).'
        }
        
    return None

def detect_aggression_spike(delta_series, lookback=20, std_dev_multiplier=2.5):
    """
    Detects sudden massive Delta spikes (Aggressive Buying/Selling) 
    that deviate greatly from standard volume distribution.
    """
    if len(delta_series) < lookback:
        return None
        
    recent_deltas = delta_series.iloc[-lookback:-1] # excluding current
    current_delta = delta_series.iloc[-1]
    
    mean_delta_abs = recent_deltas.abs().mean()
    std_delta_abs = recent_deltas.abs().std()
    
    # If the current candle has a delta explosion
    if abs(current_delta) > (mean_delta_abs + (std_dev_multiplier * std_delta_abs)):
        direction = 'LONG' if current_delta > 0 else 'SHORT'
        return {
            'signal_type': 'DELTA_SPIKE',
            'direction': direction,
            'magnitude': abs(current_delta) / mean_delta_abs,
            'description': f'Massive {direction} aggression detected (> {std_dev_multiplier} sigma).'
        }
        
    return None
