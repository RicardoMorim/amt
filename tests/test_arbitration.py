"""
Unit tests for signals/arbitration.py
"""
import pytest
from signals.arbitration import SignalArbitrator

CONTEXT = {
    'candle_time': '2024-01-01T00:00:00Z',
    'asset': 'btcusdt',
    'timeframe_secs': 900,
    'trigger_price': 42000.0,
    'close_price': 42000.0,
    'session_id': '2024-01-01',
    'session_state': 'ABOVE_VA',
    'vah': 41800.0,
    'val': 41200.0,
    'poc': 41500.0,
    'distance_to_poc': 500.0,
    'distance_to_poc_pct': 0.012,
    'volume': 100.0,
    'volume_zscore': 1.5,
    'delta': 20.0,
    'delta_zscore': 1.2,
    'cvd_current': 300.0,
    'cvd_slope_short': 50.0,
    'cvd_slope_long': 200.0,
}


def make_signal(signal_type, direction):
    return {'signal_type': signal_type, 'direction': direction, 'trigger_price': 42000.0}


def test_single_signal_passthrough():
    arb = SignalArbitrator()
    sig = make_signal('INITIATIVE_BREAKOUT', 'LONG')
    composite, all_jsons = arb.arbitrate([sig], CONTEXT)
    assert composite['direction'] == 'LONG'
    assert composite['is_composite'] is False
    assert len(all_jsons) == 1


def test_confluence_same_direction():
    arb = SignalArbitrator()
    sigs = [
        make_signal('INITIATIVE_BREAKOUT', 'LONG'),
        make_signal('DELTA_SPIKE', 'LONG'),
    ]
    composite, _ = arb.arbitrate(sigs, CONTEXT)
    assert composite['direction'] == 'LONG'
    assert composite['signal_type'] == 'COMPOSITE_CONFLUENCE'
    assert composite['conflict'] is False


def test_conflict_resolved_by_weight():
    """INITIATIVE_BREAKOUT (2.0) vs DELTA_SPIKE (1.5) — ratio 2/1.5=1.33 >= threshold 1.2"""
    arb = SignalArbitrator(conflict_threshold=1.2)
    sigs = [
        make_signal('INITIATIVE_BREAKOUT', 'LONG'),   # weight 2.0
        make_signal('DELTA_SPIKE', 'SHORT'),           # weight 1.5
    ]
    composite, _ = arb.arbitrate(sigs, CONTEXT)
    assert composite['direction'] == 'LONG'
    assert composite['signal_type'] == 'RESOLVED_BY_WEIGHT'
    assert composite['conflict'] is False


def test_true_conflict_near_tie():
    """Equal weights on opposite sides — should remain CONFLICT."""
    arb = SignalArbitrator(conflict_threshold=1.2)
    sigs = [
        make_signal('FALSE_BREAKOUT', 'LONG'),   # weight 1.0
        make_signal('CVD_DIVERGENCE_EXHAUSTION', 'SHORT'),  # weight 1.0
    ]
    composite, _ = arb.arbitrate(sigs, CONTEXT)
    assert composite['direction'] == 'CONFLICT'
    assert composite['conflict'] is True


def test_empty_signals():
    arb = SignalArbitrator()
    composite, all_jsons = arb.arbitrate([], CONTEXT)
    assert composite is None
    assert all_jsons == []
