import pytest
import math

# Minimal MockSeries to substitute for pandas.Series when it's missing
class MockSeries:
    def __init__(self, data):
        self.data = list(data)

    def __len__(self):
        return len(self.data)

    @property
    def iloc(self):
        class ILoc:
            def __init__(self, data):
                self.data = data
            def __getitem__(self, key):
                if isinstance(key, slice):
                    return MockSeries(self.data[key])
                return self.data[key]
        return ILoc(self.data)

    def max(self):
        return max(self.data) if self.data else None

    def min(self):
        return min(self.data) if self.data else None

    def abs(self):
        return MockSeries([abs(x) for x in self.data])

    def mean(self):
        if not self.data:
            return 0
        return sum(self.data) / len(self.data)

    def std(self):
        if len(self.data) < 2:
            return 0
        mean = self.mean()
        variance = sum((x - mean) ** 2 for x in self.data) / (len(self.data) - 1)
        return math.sqrt(variance)

# Inject MockSeries into sys.modules if pandas is missing (to satisfy signals/volume_imbalance.py)
import sys
from unittest.mock import MagicMock

try:
    import pandas as pd
except ImportError:
    mock_pd = MagicMock()
    mock_pd.Series = MockSeries
    sys.modules["pandas"] = mock_pd
    pd = mock_pd

from signals.volume_imbalance import detect_cvd_divergence, detect_aggression_spike

def test_detect_cvd_divergence_bearish():
    # Bearish: Price makes higher high, CVD makes lower high
    # Need len >= window + 2 = 5 + 2 = 7
    # Price: [98, 99, 100, 101, 102, 104, 106]
    # CVD:   [8,  9,  10,  11,  12,  14,  12]
    # window = 5
    # current_price = 106
    # recent_prices = [99, 100, 101, 102, 104]
    # recent_high_price = 104
    # 106 > 104 * 1.0005 (104.052)
    # recent_high_cvd = 14
    # current_cvd = 12 < 14
    price = MockSeries([98, 99, 100, 101, 102, 104, 106])
    cvd = MockSeries([8, 9, 10, 11, 12, 14, 12])

    signal = detect_cvd_divergence(price, cvd, window=5)
    assert signal is not None
    assert signal['direction'] == 'SHORT'
    assert signal['signal_type'] == 'CVD_DIVERGENCE_EXHAUSTION'

def test_detect_cvd_divergence_bullish():
    # Bullish: Price makes lower low, CVD makes higher low
    # Price: [102, 101, 100, 99, 98, 96, 94]
    # CVD:   [-8,  -9,  -10, -11, -12, -14, -12]
    # current_price = 94
    # recent_prices = [101, 100, 99, 98, 96]
    # recent_low_price = 96
    # 94 < 96 * 0.9995 (95.952)
    # recent_low_cvd = -14
    # current_cvd = -12 > -14
    price = MockSeries([102, 101, 100, 99, 98, 96, 94])
    cvd = MockSeries([-8, -9, -10, -11, -12, -14, -12])

    signal = detect_cvd_divergence(price, cvd, window=5)
    assert signal is not None
    assert signal['direction'] == 'LONG'
    assert signal['signal_type'] == 'CVD_DIVERGENCE_EXHAUSTION'

def test_detect_cvd_divergence_no_signal():
    # Price and CVD both making higher highs
    price = MockSeries([98, 99, 100, 101, 102, 104, 106])
    cvd = MockSeries([8, 9, 10, 11, 12, 14, 16])

    signal = detect_cvd_divergence(price, cvd, window=5)
    assert signal is None

def test_detect_cvd_divergence_insufficient_data():
    price = MockSeries([100, 101, 102, 103, 104, 105]) # length 6, need 7
    cvd = MockSeries([10, 11, 12, 13, 14, 15])
    signal = detect_cvd_divergence(price, cvd, window=5)
    assert signal is None

def test_detect_aggression_spike_long():
    # Long: current delta > mean + 2.5 * std
    # Mean of abs [1, 1, ..., 1] (19 times) = 1, std = 0
    deltas = MockSeries([1] * 19 + [10])
    signal = detect_aggression_spike(deltas, lookback=20, std_dev_multiplier=2.5)
    assert signal is not None
    assert signal['direction'] == 'LONG'
    assert signal['signal_type'] == 'DELTA_SPIKE'

def test_detect_aggression_spike_short():
    deltas = MockSeries([1] * 19 + [-10])
    signal = detect_aggression_spike(deltas, lookback=20, std_dev_multiplier=2.5)
    assert signal is not None
    assert signal['direction'] == 'SHORT'
    assert signal['signal_type'] == 'DELTA_SPIKE'

def test_detect_aggression_spike_no_signal():
    deltas = MockSeries([1] * 20)
    signal = detect_aggression_spike(deltas, lookback=20, std_dev_multiplier=2.5)
    assert signal is None

def test_detect_aggression_spike_insufficient_data():
    deltas = MockSeries([1, 2, 3])
    signal = detect_aggression_spike(deltas, lookback=20)
    assert signal is None
