import pytest
from unittest.mock import MagicMock
from core.converters import df_to_signals
from core.contracts import AMTSignal, Direction, SignalType

def test_df_to_signals():
    # Mocking pandas DataFrame
    mock_df = MagicMock()

    # We want to mock itertuples
    class RowIter:
        def __init__(self, data):
            self.data = data
            self.index = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self.index < len(self.data):
                item = self.data[self.index]
                self.index += 1
                return item
            raise StopIteration

    class MockRow:
        def __init__(self, d):
            self.d = d
        def _asdict(self):
            return self.d

    rows = [
        MockRow({
            "symbol": "BTC/USDT",
            "direction": "LONG",
            "signal_type": "FALSE_BREAKOUT",
            "timestamp_event": "2023-01-01T00:00:00Z"
        }),
        MockRow({
            "symbol": "ETH/USDT",
            "direction": "SHORT",
            "signal_type": "CVD_DIVERGENCE",
            "timestamp_event": "2023-01-02T00:00:00Z"
        })
    ]

    mock_df.itertuples.return_value = RowIter(rows)

    signals = df_to_signals(mock_df)

    assert len(signals) == 2
    assert isinstance(signals[0], AMTSignal)
    assert signals[0].symbol == "BTC/USDT"
    assert signals[0].direction == Direction.LONG
    assert signals[0].signal_type == SignalType.FALSE_BREAKOUT

    assert signals[1].symbol == "ETH/USDT"
    assert signals[1].direction == Direction.SHORT
    assert signals[1].signal_type == SignalType.CVD_DIVERGENCE
