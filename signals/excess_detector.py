"""
signals/excess_detector.py — AMT Excess / Single Print detector.

In Auction Market Theory, 'excess' is a long wick (thin tail) at the extreme
of the Volume Profile, signalling aggressive price rejection. It marks the
furthest point the market was willing to explore before responsive participants
pushed it back — a high-conviction reversal zone.

This module identifies excess at session highs and lows by comparing the
tail length of the current candle to the candle range and to the volume
distribution at the extreme price levels.
"""

import pandas as pd
from typing import Optional


def detect_excess(
    current_candle: dict,
    profile_data: dict,
    min_tail_ratio: float = 0.35,
    min_price_move_pct: float = 0.002,
) -> Optional[dict]:
    """
    Detects excess (price rejection / long wick) at VAH or VAL extremes.

    A bearish excess forms when the high probes above VAH but the close is
    well below the high — the market rejected upside. A bullish excess forms
    when the low probes below VAL but the close is well above the low.

    Args:
        current_candle: dict with keys 'open', 'high', 'low', 'close', 'volume'.
        profile_data: dict with 'vah', 'val', 'poc' from calculate_volume_profile().
        min_tail_ratio: Minimum ratio of (tail / candle_range) to qualify as excess.
                        Default 0.35 means the tail must be at least 35% of the range.
        min_price_move_pct: Minimum % the wick must extend beyond VAH/VAL to count.

    Returns:
        dict describing the signal, or None if no excess detected.

    Raises:
        ValueError: if required keys are missing or parameters are invalid.
        TypeError: if inputs are of the wrong type.
    """
    if not isinstance(current_candle, dict):
        raise TypeError(f"current_candle must be a dict, got {type(current_candle).__name__}")
    if not isinstance(profile_data, dict):
        raise TypeError(f"profile_data must be a dict, got {type(profile_data).__name__}")

    required_candle = {'open', 'high', 'low', 'close', 'volume'}
    missing = required_candle - set(current_candle.keys())
    if missing:
        raise ValueError(f"current_candle is missing keys: {sorted(missing)}")

    required_profile = {'vah', 'val', 'poc'}
    missing = required_profile - set(profile_data.keys())
    if missing:
        raise ValueError(f"profile_data is missing keys: {sorted(missing)}")

    if not (0 < min_tail_ratio < 1):
        raise ValueError(f"min_tail_ratio must be between 0 and 1, got {min_tail_ratio}")
    if min_price_move_pct <= 0:
        raise ValueError(f"min_price_move_pct must be positive, got {min_price_move_pct}")

    high  = float(current_candle['high'])
    low   = float(current_candle['low'])
    close = float(current_candle['close'])
    open_ = float(current_candle['open'])
    vah   = float(profile_data['vah'])
    val   = float(profile_data['val'])
    poc   = float(profile_data['poc'])

    candle_range = high - low
    if candle_range <= 0:
        return None  # Doji or zero-range — cannot determine excess

    # Upper tail (bearish excess): wick above VAH
    upper_tail = high - max(open_, close)
    upper_tail_ratio = upper_tail / candle_range
    probe_above_vah = (high - vah) / vah if vah > 0 else 0.0

    if (
        high > vah
        and probe_above_vah >= min_price_move_pct
        and upper_tail_ratio >= min_tail_ratio
        and close < vah  # closed back below VAH — true rejection
    ):
        return {
            'signal_type': 'EXCESS',
            'direction': 'SHORT',
            'trigger_price': high,
            'close_price': close,
            'rejection_level': vah,
            'tail_ratio': round(upper_tail_ratio, 4),
            'probe_pct': round(probe_above_vah * 100, 4),
            'description': f'Bearish excess above VAH ({vah:.2f}): market rejected upside aggressively.',
        }

    # Lower tail (bullish excess): wick below VAL
    lower_tail = min(open_, close) - low
    lower_tail_ratio = lower_tail / candle_range
    probe_below_val = (val - low) / val if val > 0 else 0.0

    if (
        low < val
        and probe_below_val >= min_price_move_pct
        and lower_tail_ratio >= min_tail_ratio
        and close > val  # closed back above VAL — true rejection
    ):
        return {
            'signal_type': 'EXCESS',
            'direction': 'LONG',
            'trigger_price': low,
            'close_price': close,
            'rejection_level': val,
            'tail_ratio': round(lower_tail_ratio, 4),
            'probe_pct': round(probe_below_val * 100, 4),
            'description': f'Bullish excess below VAL ({val:.2f}): market rejected downside aggressively.',
        }

    return None


def detect_excess_from_df(
    df: pd.DataFrame,
    profile_data: dict,
    min_tail_ratio: float = 0.35,
    min_price_move_pct: float = 0.002,
) -> list:
    """
    Runs detect_excess on every row of a DataFrame.

    Args:
        df: DataFrame with OHLCV columns ('open', 'high', 'low', 'close', 'volume').
        profile_data: Volume profile dict (vah, val, poc).
        min_tail_ratio: See detect_excess().
        min_price_move_pct: See detect_excess().

    Returns:
        List of (index, signal_dict) tuples for rows where excess was detected.

    Raises:
        TypeError: if df is not a DataFrame.
        ValueError: if required columns are missing.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas DataFrame, got {type(df).__name__}")

    required_cols = {'open', 'high', 'low', 'close', 'volume'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing columns: {sorted(missing)}")

    results = []
    for idx, row in df.iterrows():
        candle = {
            'open':   row['open'],
            'high':   row['high'],
            'low':    row['low'],
            'close':  row['close'],
            'volume': row['volume'],
        }
        signal = detect_excess(candle, profile_data, min_tail_ratio, min_price_move_pct)
        if signal is not None:
            results.append((idx, signal))
    return results
