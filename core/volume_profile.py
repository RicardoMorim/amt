import numpy as np
import pandas as pd

def calculate_volume_profile(df, price_col='close', vol_col='volume', value_area_pct=0.68, tick_size=0.25):
    """
    Calculates Volume Profile, POC, VAH, and VAL from an OHLCV or tick dataframe.
    
    Args:
        df: DataFrame containing price and volume data.
        price_col: Name of the column representing price. Use 'close' for OHLCV, or 'price' for tick.
        vol_col: Name of the column representing volume.
        value_area_pct: Percentage of volume to include in the Value Area (typically 68% or 70%).
        tick_size: The minimum tick size of the instrument to bucket prices.
        
    Returns:
        dict containing 'poc', 'vah', 'val', and the full 'profile' Series.
    """
    if df.empty:
        return None
        
    # Round prices to nearest tick_size for bucket clustering
    df = df.copy()
    df['price_bucket'] = (df[price_col] / tick_size).round() * tick_size
    
    # Aggregate volume by price bucket
    profile = df.groupby('price_bucket')[vol_col].sum().sort_index()
    
    if profile.empty:
        return None
        
    # Total volume
    total_volume = profile.sum()
    
    # Point of Control (POC)
    poc = profile.idxmax()
    poc_vol = profile.max()
    
    # Value Area calculation
    target_vol = total_volume * value_area_pct
    current_vol = poc_vol
    
    # Start expanding from POC
    poc_idx = profile.index.get_loc(poc)
    up_idx = poc_idx + 1
    down_idx = poc_idx - 1
    
    vah = poc
    val = poc
    
    prices = profile.index.values
    volumes = profile.values
    
    while current_vol < target_vol and (up_idx < len(prices) or down_idx >= 0):
        # Peak at the immediate upper and lower price levels
        vol_up = volumes[up_idx] if up_idx < len(prices) else -1
        vol_down = volumes[down_idx] if down_idx >= 0 else -1
        
        if vol_up == -1 and vol_down == -1:
            break
            
        # Add the level with the highest volume (or standard double-tick expansion)
        # Using a simple greedy algorithm: take the side with more volume.
        if vol_up >= vol_down:
            current_vol += vol_up
            vah = prices[up_idx]
            up_idx += 1
        else:
            current_vol += vol_down
            val = prices[down_idx]
            down_idx -= 1
            
    # Ensure VAH > VAL (in case of weird single-tick profiles)
    if vah < val:
        vah, val = val, vah
        
    return {
        'poc': poc,
        'vah': vah,
        'val': val,
        'profile': profile
    }

class SessionProfileManager:
    """
    Manages building a live profile and updating VAH/VAL dynamically tick-by-tick or footprint-by-footprint.
    """
    def __init__(self, tick_size=0.25, value_area_pct=0.68):
        self.tick_size = tick_size
        self.value_area_pct = value_area_pct
        self.price_buckets = {}  # price -> volume
        self.total_volume = 0.0
        
    def update(self, price, volume):
        bucket = round(price / self.tick_size) * self.tick_size
        self.price_buckets[bucket] = self.price_buckets.get(bucket, 0) + volume
        self.total_volume += volume
        
    def get_levels(self):
        if not self.price_buckets:
            return None
            
        # Convert to Series for easier math
        profile = pd.Series(self.price_buckets).sort_index()
        
        poc = profile.idxmax()
        poc_vol = profile.max()
        
        target_vol = self.total_volume * self.value_area_pct
        current_vol = poc_vol
        
        prices = profile.index.values
        volumes = profile.values
        
        poc_idx = profile.index.get_loc(poc)
        up_idx = poc_idx + 1
        down_idx = poc_idx - 1
        
        vah = poc
        val = poc
        
        while current_vol < target_vol and (up_idx < len(prices) or down_idx >= 0):
            vol_up = volumes[up_idx] if up_idx < len(prices) else -1
            vol_down = volumes[down_idx] if down_idx >= 0 else -1
            
            if vol_up >= vol_down:
                current_vol += vol_up
                vah = prices[up_idx]
                up_idx += 1
            else:
                current_vol += vol_down
                val = prices[down_idx]
                down_idx -= 1
                
        return {
            'poc': poc,
            'vah': vah,
            'val': val
        }
