import pandas as pd
import numpy as np

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
        # Determine direction based on string 'buy'/'sell' or integer 1/-1.
        if df[side_col].dtype == object:
            direction = df[side_col].astype(str).str.lower().map(lambda x: 1.0 if x == 'buy' else -1.0)
        else:
            direction = pd.Series(np.where(df[side_col] > 0, 1.0, -1.0), index=df.index, dtype='float64')
    else:
        # Tick test approximation:
        # If price > prev_price -> +1 (buy)
        # If price < prev_price -> -1 (sell)
        # If price == prev_price -> previous tick direction
        price_diff = df[price_col].diff()
        direction = pd.Series(
            np.select(
                [price_diff > 0, price_diff < 0, price_diff == 0],
                [1.0, -1.0, 0.0],
                default=np.nan,
            ),
            index=df.index,
            dtype='float64',
        ).replace(0.0, np.nan).ffill().fillna(1.0)

    df.loc[:, 'delta'] = df[vol_col].astype(float) * direction.astype(float)
    df.loc[:, 'cvd'] = df['delta'].cumsum()
    
    return df
