import pandas as pd
import json
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class BreakoutConfig:
    vol_multiplier: float
    min_body_ratio: float

    def __post_init__(self):
        if self.vol_multiplier <= 0:
            raise ValueError(f"vol_multiplier must be positive, got {self.vol_multiplier}")
        if not (0 < self.min_body_ratio <= 1):
            raise ValueError(f"min_body_ratio must be between 0 and 1, got {self.min_body_ratio}")

    @classmethod
    def from_json(cls, path: str) -> 'BreakoutConfig':
        """
        Load config from a JSON file.
        Raises:
            FileNotFoundError: if the file does not exist.
            json.JSONDecodeError: if the file is malformed.
            KeyError: if required keys are missing.
            ValueError: if values are invalid.
        """
        with open(path, 'r') as f:
            opts = json.load(f)  # raises json.JSONDecodeError if malformed
        return cls(
            vol_multiplier=float(opts['vol_multiplier']),
            min_body_ratio=float(opts['min_body_ratio']),
        )

    @classmethod
    def from_defaults(cls) -> 'BreakoutConfig':
        """Load from config.py defaults."""
        import config
        return cls(
            vol_multiplier=config.BREAKOUT_VOL_MULTIPLIER,
            min_body_ratio=config.BREAKOUT_MIN_BODY_RATIO,
        )

    @classmethod
    def load(cls, optimized_path: Optional[str] = None) -> 'BreakoutConfig':
        """
        Load optimized config if it exists, otherwise fall back to defaults.
        If the optimized file exists but is invalid, raises immediately.
        """
        if optimized_path and os.path.exists(optimized_path):
            return cls.from_json(optimized_path)
        return cls.from_defaults()


class BreakoutDetector:
    """
    Detects initiative balance breakouts based on AMT principles:
    - Price closes outside the Value Area
    - Volume is above average (initiative activity)
    - Delta confirms the direction
    - Candle body is dominant (no indecision wick)
    """

    def __init__(self, config: BreakoutConfig):
        self.config = config

    def detect(self, current_candle: dict, cvd_data: pd.DataFrame,
               profile_data: dict, lookback_df: pd.DataFrame) -> Optional[dict]:
        """
        Raises:
            ValueError: if required keys are missing from inputs.
            TypeError: if inputs are of the wrong type.
        """
        self._validate_inputs(current_candle, cvd_data, profile_data, lookback_df)

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
        if vol < (avg_vol * self.config.vol_multiplier):
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
        if (body_size / candle_range) < self.config.min_body_ratio:
            return None

        direction = 'LONG' if is_breaking_up else 'SHORT'
        return {
            'signal_type':   'INITIATIVE_BREAKOUT',
            'direction':     direction,
            'trigger_price': close,
            'stop_loss':     vah if direction == 'LONG' else val,
            'confidence':    'HIGH',
        }

    @staticmethod
    def _validate_inputs(current_candle, cvd_data, profile_data, lookback_df):
        required_candle_keys = {'open', 'high', 'low', 'close', 'volume'}
        missing = required_candle_keys - set(current_candle.keys())
        if missing:
            raise ValueError(f"current_candle is missing keys: {sorted(missing)}")

        if not isinstance(cvd_data, pd.DataFrame) or cvd_data.empty:
            raise ValueError("cvd_data must be a non-empty DataFrame")
        if 'delta' not in cvd_data.columns:
            raise ValueError("cvd_data must have a 'delta' column — run calculate_cvd() first")

        required_profile_keys = {'vah', 'val', 'poc'}
        missing = required_profile_keys - set(profile_data.keys())
        if missing:
            raise ValueError(f"profile_data is missing keys: {sorted(missing)}")

        if not isinstance(lookback_df, pd.DataFrame) or lookback_df.empty:
            raise ValueError("lookback_df must be a non-empty DataFrame")
        if 'volume' not in lookback_df.columns:
            raise ValueError("lookback_df must have a 'volume' column")
