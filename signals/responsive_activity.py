"""
signals/responsive_activity.py — AMT Responsive Activity detector.

In Auction Market Theory, markets alternate between two modes:
  - Initiative Activity: price moves away from value (breakout, trend).
  - Responsive Activity: participants respond to price moving too far from
    value, fading the move back toward POC/Value Area.

This module detects Responsive Activity: when price touches or exceeds
VAH/VAL and the CVD immediately reverses, indicating that responsive
participants are absorbing the initiative move.
"""

import pandas as pd
from typing import Optional


def detect_responsive_activity(
    current_candle: dict,
    cvd_data: pd.DataFrame,
    profile_data: dict,
    cvd_reversal_window: int = 3,
    min_cvd_reversal_pct: float = 0.05,
) -> Optional[dict]:
    """
    Detects Responsive Activity at VAH or VAL.

    Conditions for Responsive Sell (fade at VAH):
      1. Price touched or exceeded VAH this candle.
      2. The last N CVD readings show a declining slope (sellers absorbing buyers).
      3. Current candle closes inside or below VAH (responsive rejection).

    Conditions for Responsive Buy (fade at VAL):
      1. Price touched or went below VAL this candle.
      2. The last N CVD readings show a rising slope (buyers absorbing sellers).
      3. Current candle closes inside or above VAL (responsive rejection).

    Args:
        current_candle: dict with keys 'open', 'high', 'low', 'close', 'volume'.
        cvd_data: DataFrame with 'cvd' column (output of calculate_cvd()).
        profile_data: dict with 'vah', 'val', 'poc'.
        cvd_reversal_window: Number of recent CVD bars to check for reversal slope.
        min_cvd_reversal_pct: Minimum CVD change (as fraction of its absolute value)
                              to qualify as a meaningful reversal.

    Returns:
        dict describing the signal, or None if no responsive activity detected.

    Raises:
        ValueError: if required keys are missing or insufficient data.
        TypeError: if inputs are of the wrong type.
    """
    if not isinstance(current_candle, dict):
        raise TypeError(f"current_candle must be a dict, got {type(current_candle).__name__}")
    if not isinstance(cvd_data, pd.DataFrame):
        raise TypeError(f"cvd_data must be a DataFrame, got {type(cvd_data).__name__}")
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

    if 'cvd' not in cvd_data.columns:
        raise ValueError("cvd_data must have a 'cvd' column — run calculate_cvd() first")

    if cvd_reversal_window < 2:
        raise ValueError(f"cvd_reversal_window must be >= 2, got {cvd_reversal_window}")

    if len(cvd_data) < cvd_reversal_window:
        raise ValueError(
            f"cvd_data has {len(cvd_data)} rows but cvd_reversal_window requires {cvd_reversal_window}. "
            "Provide more CVD history."
        )

    if not (0 < min_cvd_reversal_pct < 1):
        raise ValueError(f"min_cvd_reversal_pct must be between 0 and 1, got {min_cvd_reversal_pct}")

    high  = float(current_candle['high'])
    low   = float(current_candle['low'])
    close = float(current_candle['close'])
    vah   = float(profile_data['vah'])
    val   = float(profile_data['val'])
    poc   = float(profile_data['poc'])

    recent_cvd = cvd_data['cvd'].iloc[-cvd_reversal_window:]
    cvd_start  = float(recent_cvd.iloc[0])
    cvd_end    = float(recent_cvd.iloc[-1])
    cvd_change = cvd_end - cvd_start

    # Avoid division-by-zero: if CVD is near zero, any reversal is meaningful
    cvd_magnitude = max(abs(cvd_start), 1.0)
    cvd_reversal_strength = abs(cvd_change) / cvd_magnitude

    # RESPONSIVE SELL: touched VAH but CVD is falling
    if (
        high >= vah
        and close <= vah  # closed back inside or below VAH
        and cvd_change < 0
        and cvd_reversal_strength >= min_cvd_reversal_pct
    ):
        return {
            'signal_type': 'RESPONSIVE_ACTIVITY',
            'direction': 'SHORT',
            'trigger_price': close,
            'target': poc,
            'rejection_level': vah,
            'cvd_change': round(cvd_change, 4),
            'cvd_reversal_strength': round(cvd_reversal_strength, 4),
            'description': (
                f'Responsive Sell at VAH ({vah:.2f}): CVD reversed {cvd_change:.0f} '
                f'over last {cvd_reversal_window} bars. Target POC: {poc:.2f}.'
            ),
        }

    # RESPONSIVE BUY: touched VAL but CVD is rising
    if (
        low <= val
        and close >= val  # closed back inside or above VAL
        and cvd_change > 0
        and cvd_reversal_strength >= min_cvd_reversal_pct
    ):
        return {
            'signal_type': 'RESPONSIVE_ACTIVITY',
            'direction': 'LONG',
            'trigger_price': close,
            'target': poc,
            'rejection_level': val,
            'cvd_change': round(cvd_change, 4),
            'cvd_reversal_strength': round(cvd_reversal_strength, 4),
            'description': (
                f'Responsive Buy at VAL ({val:.2f}): CVD reversed +{cvd_change:.0f} '
                f'over last {cvd_reversal_window} bars. Target POC: {poc:.2f}.'
            ),
        }

    return None
