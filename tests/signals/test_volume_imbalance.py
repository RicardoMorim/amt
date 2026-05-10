import pytest
from unittest.mock import MagicMock
from signals.volume_imbalance import detect_cvd_divergence, detect_aggression_spike

def test_detect_cvd_divergence_empty_series():
    assert detect_cvd_divergence([], [], window=5) is None

def test_detect_cvd_divergence_short_series():
    # window is 5, requires len >= 7
    prices = [1, 2, 3, 4, 5, 6]
    cvds = [1, 2, 3, 4, 5, 6]
    assert detect_cvd_divergence(prices, cvds, window=5) is None

def test_detect_aggression_spike_empty_series():
    assert detect_aggression_spike([], lookback=20) is None

def test_detect_aggression_spike_short_series():
    # lookback is 20, requires len >= 20
    deltas = [1] * 19
    assert detect_aggression_spike(deltas, lookback=20) is None

def test_cvd_divergence_bearish_mocked():
    # Mocking pandas series to simulate bearish divergence
    # Price makes higher high, CVD makes lower high

    price_series = MagicMock()
    price_series.__len__.return_value = 10

    cvd_series = MagicMock()
    cvd_series.__len__.return_value = 10

    price_iloc = MagicMock()
    price_recent = MagicMock()
    price_recent.max.return_value = 100.0
    price_recent.min.return_value = 90.0
    price_iloc.__getitem__.side_effect = lambda x: 105.0 if x == -1 else price_recent
    price_series.iloc = price_iloc

    cvd_iloc = MagicMock()
    cvd_recent = MagicMock()
    cvd_recent.max.return_value = 50.0
    cvd_recent.min.return_value = 20.0
    cvd_iloc.__getitem__.side_effect = lambda x: 40.0 if x == -1 else cvd_recent
    cvd_series.iloc = cvd_iloc

    result = detect_cvd_divergence(price_series, cvd_series, window=5)

    assert result is not None
    assert result['signal_type'] == 'CVD_DIVERGENCE_EXHAUSTION'
    assert result['direction'] == 'SHORT'
    assert result['trigger_price'] == 105.0

def test_cvd_divergence_bullish_mocked():
    # Bullish: price lower low, cvd higher low
    price_series = MagicMock()
    price_series.__len__.return_value = 10

    cvd_series = MagicMock()
    cvd_series.__len__.return_value = 10

    price_iloc = MagicMock()
    price_recent = MagicMock()
    price_recent.max.return_value = 100.0
    price_recent.min.return_value = 90.0
    price_iloc.__getitem__.side_effect = lambda x: 80.0 if x == -1 else price_recent
    price_series.iloc = price_iloc

    cvd_iloc = MagicMock()
    cvd_recent = MagicMock()
    cvd_recent.max.return_value = 50.0
    cvd_recent.min.return_value = 20.0
    cvd_iloc.__getitem__.side_effect = lambda x: 30.0 if x == -1 else cvd_recent
    cvd_series.iloc = cvd_iloc

    result = detect_cvd_divergence(price_series, cvd_series, window=5)

    assert result is not None
    assert result['signal_type'] == 'CVD_DIVERGENCE_EXHAUSTION'
    assert result['direction'] == 'LONG'
    assert result['trigger_price'] == 80.0

def test_aggression_spike_long_mocked():
    delta_series = MagicMock()
    delta_series.__len__.return_value = 30

    delta_iloc = MagicMock()
    recent_deltas = MagicMock()

    recent_abs = MagicMock()
    recent_abs.mean.return_value = 10.0
    recent_abs.std.return_value = 2.0
    recent_deltas.abs.return_value = recent_abs

    delta_iloc.__getitem__.side_effect = lambda x: 20.0 if x == -1 else recent_deltas
    delta_series.iloc = delta_iloc

    result = detect_aggression_spike(delta_series, lookback=20, std_dev_multiplier=2.5)

    assert result is not None
    assert result['signal_type'] == 'DELTA_SPIKE'
    assert result['direction'] == 'LONG'
    assert result['magnitude'] == 2.0 # 20.0 / 10.0

def test_aggression_spike_short_mocked():
    delta_series = MagicMock()
    delta_series.__len__.return_value = 30

    delta_iloc = MagicMock()
    recent_deltas = MagicMock()

    recent_abs = MagicMock()
    recent_abs.mean.return_value = 10.0
    recent_abs.std.return_value = 2.0
    recent_deltas.abs.return_value = recent_abs

    delta_iloc.__getitem__.side_effect = lambda x: -20.0 if x == -1 else recent_deltas
    delta_series.iloc = delta_iloc

    result = detect_aggression_spike(delta_series, lookback=20, std_dev_multiplier=2.5)

    assert result is not None
    assert result['signal_type'] == 'DELTA_SPIKE'
    assert result['direction'] == 'SHORT'
    assert result['magnitude'] == 2.0

def test_detect_cvd_divergence_no_divergence_mocked():
    # Price makes higher high, CVD ALSO makes higher high (no divergence)
    price_series = MagicMock()
    price_series.__len__.return_value = 10
    cvd_series = MagicMock()
    cvd_series.__len__.return_value = 10

    price_iloc = MagicMock()
    price_recent = MagicMock()
    price_recent.max.return_value = 100.0
    price_recent.min.return_value = 90.0
    price_iloc.__getitem__.side_effect = lambda x: 105.0 if x == -1 else price_recent
    price_series.iloc = price_iloc

    cvd_iloc = MagicMock()
    cvd_recent = MagicMock()
    cvd_recent.max.return_value = 50.0
    cvd_recent.min.return_value = 20.0
    cvd_iloc.__getitem__.side_effect = lambda x: 60.0 if x == -1 else cvd_recent
    cvd_series.iloc = cvd_iloc

    assert detect_cvd_divergence(price_series, cvd_series, window=5) is None

def test_detect_aggression_spike_no_spike_mocked():
    delta_series = MagicMock()
    delta_series.__len__.return_value = 30

    delta_iloc = MagicMock()
    recent_deltas = MagicMock()

    recent_abs = MagicMock()
    recent_abs.mean.return_value = 10.0
    recent_abs.std.return_value = 2.0
    recent_deltas.abs.return_value = recent_abs

    # 12 is less than 10 + 2.5*2 = 15
    delta_iloc.__getitem__.side_effect = lambda x: 12.0 if x == -1 else recent_deltas
    delta_series.iloc = delta_iloc

    assert detect_aggression_spike(delta_series, lookback=20, std_dev_multiplier=2.5) is None
