import pandas as pd

def calculate_cvd(df, price_col='close', vol_col='volume', side_col='side'):
    """
    Calculates Cumulative Volume Delta (CVD) from a dataframe of trades or footprint candles.
    
    Args:
        df: DataFrame with trades or candle data.
        price_col: Column for price (used as fallback logic if side isn't available).
        vol_col: Column for volume.
        side_col: Column specifying if the trade was 'buy' (aggressor buy / lifting the ask) 
                  or 'sell' (aggressor sell / hitting the bid). 
                  If not available, falls back to calculating via the tick-test approximation.
                  
    Returns:
        DataFrame with 'delta' and 'cvd' columns added.
    """
    df = df.copy()
    
    if side_col in df.columns:
        # Determine direction based on string 'buy'/'sell' or integer 1/-1
        if df[side_col].dtype == object:
            direction = df[side_col].apply(lambda x: 1 if str(x).lower() == 'buy' else -1)
        else:
            direction = np.where(df[side_col] > 0, 1, -1)
    else:
        # Tick test approximation: 
        # Valid when exact order book side is unknown.
        # If price > prev_price -> +1 (buy)
        # If price < prev_price -> -1 (sell)
        # If price == prev_price -> previous tick direction
        price_diff = df[price_col].diff()
        
        # 1 for positive diff, -1 for negative, 0 for no change
        # Pandas 2.x efficient assignment:
        direction = pd.Series(index=df.index, dtype='float64')
        direction[price_diff > 0] = 1
        direction[price_diff < 0] = -1
        direction[price_diff == 0] = 0
        
        # Forward fill the zeros to carry over the previous direction
        direction = direction.replace(0, pd.NA).ffill().fillna(1)
        
    df['delta'] = df[vol_col] * direction
    df['cvd'] = df['delta'].cumsum()
    
    return df
