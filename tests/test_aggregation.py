"""
Unit tests for historical_runner._aggregate_by_seconds
"""
import pytest
import pandas as pd
from historical_runner import _aggregate_by_seconds


def make_tick_df(timestamps, prices, volumes, sides):
    return pd.DataFrame({
        'timestamp': pd.to_datetime(timestamps),
        'price': prices,
        'volume': volumes,
        'side': sides,
    })


def test_basic_aggregation_reduces_rows():
    df = make_tick_df(
        ['2024-01-01 00:00:00.1', '2024-01-01 00:00:00.5', '2024-01-01 00:00:01.2'],
        [100, 101, 102],
        [1, 2, 3],
        ['buy', 'sell', 'buy'],
    )
    result = _aggregate_by_seconds(df, secs=1)
    # First two ticks land in second 0, third tick in second 1
    assert len(result) == 2


def test_vwap_price():
    df = make_tick_df(
        ['2024-01-01 00:00:00.0', '2024-01-01 00:00:00.5'],
        [100.0, 200.0],
        [1.0, 3.0],
        ['buy', 'buy'],
    )
    result = _aggregate_by_seconds(df, secs=1)
    expected_vwap = (100 * 1 + 200 * 3) / (1 + 3)  # = 175.0
    assert abs(result.iloc[0]['price'] - expected_vwap) < 0.001


def test_dominant_side_buy():
    df = make_tick_df(
        ['2024-01-01 00:00:00.0', '2024-01-01 00:00:00.5'],
        [100, 100],
        [3.0, 1.0],
        ['buy', 'sell'],
    )
    result = _aggregate_by_seconds(df, secs=1)
    assert result.iloc[0]['side'] == 'buy'


def test_empty_returns_empty():
    df = pd.DataFrame(columns=['timestamp', 'price', 'volume', 'side'])
    result = _aggregate_by_seconds(df, secs=1)
    assert result.empty
