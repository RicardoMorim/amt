import pandas as pd
import numpy as np

_VALID_SIDE_STRINGS = frozenset({'buy', 'sell'})
_VALID_SIDE_INTS = frozenset({1, -1})


def calculate_cvd(df, price_col='close', vol_col='volume', side_col='side'):
    """
    Calculates Cumulative Volume Delta (CVD) from a dataframe of trades or footprint candles.

    Args:
        df: DataFrame with trades or candle data.
        price_col: Column for price (used for tick-test when side is not available).
        vol_col: Column for volume.
        side_col: Column specifying if the trade was 'buy' or 'sell' (or 1 / -1).
                  If the column is absent, falls back to the tick-test approximation.

    Returns:
        DataFrame with 'delta' and 'cvd' columns added.

    Raises:
        ValueError: if required columns are missing, if side_col contains unexpected values,
                    or if there is insufficient data for the tick-test.
        TypeError: if df is not a DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas DataFrame, got {type(df).__name__}")
    if df.empty:
        raise ValueError("df is empty — cannot calculate CVD")
    if vol_col not in df.columns:
        raise ValueError(f"volume column '{vol_col}' not found in df. Available columns: {list(df.columns)}")

    df = df.copy()

    if side_col in df.columns:
        if df[side_col].dtype == object:
            # Validate that only 'buy' / 'sell' values exist (case-insensitive)
            unique_vals = set(df[side_col].dropna().astype(str).str.lower().unique())
            invalid = unique_vals - _VALID_SIDE_STRINGS
            if invalid:
                raise ValueError(
                    f"side column '{side_col}' contains unexpected values: {sorted(invalid)}. "
                    f"Expected only: {sorted(_VALID_SIDE_STRINGS)}"
                )
            direction = df[side_col].astype(str).str.lower().map({'buy': 1.0, 'sell': -1.0})
        else:
            unique_vals = set(df[side_col].dropna().unique())
            invalid = {v for v in unique_vals if v not in _VALID_SIDE_INTS}
            if invalid:
                raise ValueError(
                    f"side column '{side_col}' contains unexpected integer values: {sorted(invalid)}. "
                    f"Expected only: {sorted(_VALID_SIDE_INTS)}"
                )
            direction = pd.Series(
                np.where(df[side_col] > 0, 1.0, -1.0),
                index=df.index,
                dtype='float64'
            )

        if direction.isna().any():
            n_nulls = direction.isna().sum()
            raise ValueError(
                f"side column '{side_col}' has {n_nulls} null values — clean the data before calling calculate_cvd()"
            )
    else:
        # Tick-test approximation
        if price_col not in df.columns:
            raise ValueError(
                f"price column '{price_col}' not found in df and '{side_col}' is absent. "
                f"Available columns: {list(df.columns)}"
            )
        if len(df) < 2:
            raise ValueError(
                "Tick-test requires at least 2 rows to compute price direction, got 1"
            )

        price_diff = df[price_col].diff()

        # Assign direction: +1 up-tick, -1 down-tick, carry forward on equal-tick
        raw_direction = pd.Series(
            np.select(
                [price_diff > 0, price_diff < 0],
                [1.0, -1.0],
                default=np.nan,
            ),
            index=df.index,
            dtype='float64',
        )

        # Forward-fill equal-ticks (standard tick-test rule)
        direction = raw_direction.ffill()

        # First row has NaN price_diff — if still NaN after ffill, there is no prior tick
        if direction.isna().any():
            raise ValueError(
                "Cannot determine tick direction for the first row because there is no prior tick. "
                "Provide data with a known starting side, or pass 'side_col' explicitly."
            )

    df.loc[:, 'delta'] = df[vol_col].astype(float) * direction.astype(float)
    df.loc[:, 'cvd'] = df['delta'].cumsum()

    return df
