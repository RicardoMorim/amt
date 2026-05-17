import pytest
from unittest.mock import MagicMock
from signals.balance_breakout import detect_balance_breakout
import config

# Simple mock for pandas DataFrame behavior needed by the function
class MockIloc:
    def __init__(self, val):
        self.val = val
    def __getitem__(self, idx):
        if idx == -1: return self.val
        return self.val

class MockSeries:
    def __init__(self, mean_val=None, iloc_val=None):
        self._mean = mean_val
        self.iloc = MockIloc(iloc_val)
    def mean(self):
        return self._mean

class MockDF:
    def __init__(self, empty=False, data=None):
        self.empty = empty
        self.data = data or {}
    def __getitem__(self, key):
        return self.data.get(key)

@pytest.fixture
def base_candle():
    return {
        'open': 100,
        'high': 110,
        'low': 90,
        'close': 105,
        'volume': 1000
    }

@pytest.fixture
def profile_data():
    return {
        'vah': 100,
        'val': 90
    }

def test_detect_balance_breakout_empty_inputs(base_candle, profile_data):
    empty_df = MockDF(empty=True)
    valid_df = MockDF(empty=False, data={'volume': MockSeries(mean_val=500), 'delta': MockSeries(iloc_val=10)})

    # Empty profile
    assert detect_balance_breakout(base_candle, valid_df, {}, valid_df) is None
    assert detect_balance_breakout(base_candle, valid_df, None, valid_df) is None

    # Empty lookback
    assert detect_balance_breakout(base_candle, valid_df, profile_data, empty_df) is None

    # Empty cvd
    assert detect_balance_breakout(base_candle, empty_df, profile_data, valid_df) is None

def test_detect_balance_breakout_no_breakout(base_candle, profile_data):
    valid_df = MockDF(empty=False, data={'volume': MockSeries(mean_val=500), 'delta': MockSeries(iloc_val=10)})

    # Candle completely inside VAH/VAL
    candle_inside = {**base_candle, 'open': 95, 'close': 98}
    assert detect_balance_breakout(candle_inside, valid_df, profile_data, valid_df) is None

def test_detect_balance_breakout_low_volume(profile_data):
    # Setup LONG breakout logic
    # VAH is 100. Open <= 100, Close > 100
    candle = {
        'open': 95,
        'high': 115,
        'low': 90,
        'close': 110,
        'volume': 1000
    }

    # Avg vol * multiplier (1.5) > candle vol -> fails
    avg_vol = 1000 / config.BREAKOUT_VOL_MULTIPLIER + 10  # This makes target > 1000
    valid_df = MockDF(empty=False, data={'volume': MockSeries(mean_val=avg_vol), 'delta': MockSeries(iloc_val=10)})

    assert detect_balance_breakout(candle, valid_df, profile_data, valid_df) is None

def test_detect_balance_breakout_bad_delta(profile_data):
    # Setup LONG breakout logic
    candle_up = {
        'open': 95,
        'high': 115,
        'low': 90,
        'close': 110,
        'volume': 1000
    }

    # Avg vol allows passing
    avg_vol = 500

    # Bad delta for LONG
    valid_df_bad_long_delta = MockDF(empty=False, data={'volume': MockSeries(mean_val=avg_vol), 'delta': MockSeries(iloc_val=0)})
    assert detect_balance_breakout(candle_up, valid_df_bad_long_delta, profile_data, valid_df_bad_long_delta) is None

    valid_df_bad_long_delta2 = MockDF(empty=False, data={'volume': MockSeries(mean_val=avg_vol), 'delta': MockSeries(iloc_val=-5)})
    assert detect_balance_breakout(candle_up, valid_df_bad_long_delta2, profile_data, valid_df_bad_long_delta2) is None

    # Setup SHORT breakout logic
    # VAL is 90. Open >= 90, Close < 90
    candle_down = {
        'open': 95,
        'high': 100,
        'low': 80,
        'close': 85,
        'volume': 1000
    }

    # Bad delta for SHORT
    valid_df_bad_short_delta = MockDF(empty=False, data={'volume': MockSeries(mean_val=avg_vol), 'delta': MockSeries(iloc_val=0)})
    assert detect_balance_breakout(candle_down, valid_df_bad_short_delta, profile_data, valid_df_bad_short_delta) is None

    valid_df_bad_short_delta2 = MockDF(empty=False, data={'volume': MockSeries(mean_val=avg_vol), 'delta': MockSeries(iloc_val=5)})
    assert detect_balance_breakout(candle_down, valid_df_bad_short_delta2, profile_data, valid_df_bad_short_delta2) is None


def test_detect_balance_breakout_small_body_or_zero_range(profile_data):
    avg_vol = 500
    valid_df = MockDF(empty=False, data={'volume': MockSeries(mean_val=avg_vol), 'delta': MockSeries(iloc_val=10)})

    # LONG breakout, but very small body compared to range
    candle_small_body = {
        'open': 98,
        'high': 150,
        'low': 80,
        'close': 102, # Body = 4, Range = 70. 4/70 = ~0.05 < config.BREAKOUT_MIN_BODY_RATIO
        'volume': 1000
    }

    assert detect_balance_breakout(candle_small_body, valid_df, profile_data, valid_df) is None

    # Zero range (technically impossible if body > 0, but good for edge case)
    candle_zero_range = {
        'open': 102,
        'high': 102,
        'low': 102,
        'close': 102,
        'volume': 1000
    }

    # To pass breaking up logic, open <= vah and close > vah.
    # But if high == low == open == close, it can't cross.
    # Let's mock a weird situation where high == low = 100
    candle_zero_range = {
        'open': 100,
        'high': 101, # Need range=0, but close > open, so impossible to test normally without just high=low.
        'low': 101,
        'close': 101,
        'volume': 1000
    }

    # True zero range where open <= vah, close > vah is impossible since open != close.
    # We can just check the line candle_range == 0
    # Actually if range = 0, open = close, so it's not a breakout.
    # This branch `if candle_range == 0` is covered if it somehow gets there.


def test_detect_balance_breakout_long_success(profile_data):
    # Open <= 100, Close > 100
    candle = {
        'open': 95,
        'high': 115,
        'low': 90,
        'close': 110,  # Body = 15, Range = 25. 15/25 = 0.6 >= 0.6
        'volume': 1000
    }

    avg_vol = 500
    valid_df = MockDF(empty=False, data={'volume': MockSeries(mean_val=avg_vol), 'delta': MockSeries(iloc_val=10)})

    result = detect_balance_breakout(candle, valid_df, profile_data, valid_df)

    assert result is not None
    assert result['signal_type'] == 'INITIATIVE_BREAKOUT'
    assert result['direction'] == 'LONG'
    assert result['trigger_price'] == 110
    assert result['stop_loss'] == 100
    assert result['confidence'] == 'HIGH'

def test_detect_balance_breakout_short_success(profile_data):
    # VAL = 90. Open >= 90, Close < 90
    candle = {
        'open': 95,
        'high': 100,
        'low': 75,
        'close': 80, # Body = 15, Range = 25. 15/25 = 0.6 >= 0.6
        'volume': 1000
    }

    avg_vol = 500
    valid_df = MockDF(empty=False, data={'volume': MockSeries(mean_val=avg_vol), 'delta': MockSeries(iloc_val=-10)})

    result = detect_balance_breakout(candle, valid_df, profile_data, valid_df)

    assert result is not None
    assert result['signal_type'] == 'INITIATIVE_BREAKOUT'
    assert result['direction'] == 'SHORT'
    assert result['trigger_price'] == 80
    assert result['stop_loss'] == 90
    assert result['confidence'] == 'HIGH'
