"""
Examples of using the shared contracts (FASE 0).

Shows:
1. Creating typed objects from scratch
2. Converting pandas DataFrames → contracts
3. Serializing / deserializing to JSON
4. Adapting legacy dicts to new contracts
5. SQLite row conversion
"""

from datetime import datetime, timezone
import json
import sqlite3
import sys
from pathlib import Path

# Add project root to path for direct execution
if Path(__file__).exists():
    sys.path.insert(0, str(Path(__file__).parent.parent))
elif "core" not in sys.path:
    sys.path.insert(0, ".")

import pandas as pd

# Import everything from core package
from core.contracts import (
    AMTSignal,
    Candle,
    FusionDecision,
    KronosPrediction,
    TradeAction,
    Direction,
    SignalType,
    SessionState,
    MarketState,
    contracts_to_json,
    contracts_from_json,
)
from core.converters import (
    df_to_candles,
    df_to_signals,
    candles_to_df,
    signals_to_df,
    adapt_signal_dict,
    sqlite_row_to_candle,
)


def example_1_create_from_scratch():
    """Create typed objects directly."""
    print("=" * 60)
    print("Example 1: Creating typed objects from scratch")
    print("=" * 60)

    # Candle
    ts = datetime(2025, 6, 15, 14, 30, 0, tzinfo=timezone.utc)
    candle = Candle(
        symbol="NQ1",
        timestamp=ts,
        open=18000.0,
        high=18050.0,
        low=17980.0,
        close=18025.0,
        volume=1500.0,
        timeframe_secs=60,
    )
    print(f"\nCandle: {candle.symbol} @ {candle.timestamp}")
    print(f"  OHLC: O={candle.open}, H={candle.high}, L={candle.low}, C={candle.close}")

    # AMTSignal
    signal = AMTSignal(
        symbol="NQ1",
        timestamp_event=ts,
        direction=Direction.LONG,
        signal_type=SignalType.FALSE_BREAKOUT,
        session_state=SessionState.NY,
        confidence=0.82,
        distance_to_poc_pct=1.23,
    )
    print(f"\nAMTSignal: {signal.symbol} {signal.direction.value}")
    print(f"  Type={signal.signal_type.value}, Confidence={signal.confidence:.2f}")

    # KronosPrediction
    pred = KronosPrediction(
        symbol="NQ1",
        timestamp=ts,
        direction=Direction.SHORT,
        confidence=0.71,
        timeframe_secs=300,
    )
    print(f"\nKronosPrediction: {pred.symbol} {pred.direction.value}")
    print(f"  Confidence={pred.confidence:.2f}, TF={pred.timeframe_secs}s")

    # FusionDecision
    fusion = FusionDecision(
        symbol="NQ1",
        timestamp=ts,
        action=TradeAction.BUY,
        direction=Direction.LONG,
        amt_score=0.65,
        kronos_score=0.48,
        fusion_score=0.57,
        confidence=0.68,
    )
    print(f"\nFusionDecision: {fusion.symbol} → {fusion.action.value}")
    print(f"  Scores: AMT={fusion.amt_score:.2f}, Kronos={fusion.kronos_score:.2f}, Fusion={fusion.fusion_score:.2f}")


def example_2_pandas_conversion():
    """Convert between pandas DataFrames and contracts."""
    print("\n" + "=" * 60)
    print("Example 2: Pandas ↔ Contracts conversion")
    print("=" * 60)

    # Create a sample OHLCV DataFrame (simulating collector output)
    df = pd.DataFrame({
        "timestamp": [
            "2025-06-15T14:00:00+00:00",
            "2025-06-15T14:01:00+00:00",
            "2025-06-15T14:02:00+00:00",
        ],
        "open": [180.0, 181.5, 180.8],
        "high": [182.0, 183.0, 181.5],
        "low": [179.0, 180.0, 179.5],
        "close": [181.5, 180.8, 182.0],
        "volume": [100, 150, 120],
    })

    print(f"\nOriginal DataFrame ({len(df)} rows):")
    print(df.head())

    # Convert to contracts
    candles = df_to_candles(df, symbol="NQ1", timeframe_secs=60)
    print(f"\nConverted to {len(candles)} Candle objects:")
    for c in candles[:2]:
        print(f"  {c.timestamp} O={c.open:.2f} C={c.close:.2f} V={c.volume}")

    # Convert back to DataFrame
    df_back = candles_to_df(candles)
    print(f"\nBack to DataFrame ({len(df_back)} rows):")
    print(df_back.head())


def example_3_json_serialization():
    """Serialize/deserialize contracts to/from JSON."""
    print("\n" + "=" * 60)
    print("Example 3: JSON Serialization / Deserialization")
    print("=" * 60)

    ts = datetime(2025, 6, 15, 14, 30, 0, tzinfo=timezone.utc)

    contracts = [
        Candle(symbol="NQ1", timestamp=ts.replace(minute=0), open=180, high=182, low=179, close=181.5, volume=100),
        AMTSignal(
            symbol="NQ1", timestamp_event=ts, direction=Direction.LONG,
            signal_type=SignalType.FALSE_BREAKOUT, confidence=0.82,
        ),
        KronosPrediction(
            symbol="NQ1", timestamp=ts, direction=Direction.SHORT,
            confidence=0.71, timeframe_secs=300,
        ),
    ]

    # Serialize to JSON
    json_str = contracts_to_json(contracts)
    print(f"\nSerialized ({len(json_str)} chars):")
    print(json_str[:200] + "...")

    # Deserialize back
    restored = contracts_from_json(json_str)
    print(f"\nDeserialized {len(restored)} contracts:")
    for c in restored:
        print(f"  {type(c).__name__}: symbol={c.symbol}")


def example_4_legacy_adaptation():
    """Adapt legacy dict-based signals to new typed contracts."""
    print("\n" + "=" * 60)
    print("Example 4: Legacy Dict → Typed Contract Adaptation")
    print("=" * 60)

    # Old format (what the existing code produces)
    legacy_signal = {
        "instrument": "ES1",       # old field name
        "side": "SHORT",            # old direction name
        "type": "VOLUME_IMBALANCE", # old signal type name
        "timestamp": "2025-06-15T14:30:00+00:00",
        "distance_to_poc_pct": 2.1,
    }

    adapted = adapt_signal_dict(legacy_signal)
    print(f"\nLegacy dict → AMTSignal:")
    print(f"  instrument={legacy_signal['instrument']} → symbol={adapted.symbol}")
    print(f"  side={legacy_signal['side']} → direction={adapted.direction.value}")
    print(f"  type={legacy_signal['type']} → signal_type={adapted.signal_type.value}")


def example_5_sqlite_conversion():
    """Convert SQLite rows to contracts."""
    print("\n" + "=" * 60)
    print("Example 5: SQLite Row → Contract Conversion")
    print("=" * 60)

    # Simulate a real SQLite connection with Row factory (dict-like access by column name)
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row  # This makes fetchone() return dict-like rows
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE ohlcv (
            symbol TEXT, timestamp TEXT, open REAL, high REAL, low REAL, close REAL, volume INTEGER
        )
    """)
    cursor.execute(
        "INSERT INTO ohlcv VALUES ('NQ1', '2025-06-15T14:30:00+00:00', 180.0, 182.0, 179.0, 181.0, 500)"
    )
    conn.commit()

    row = cursor.execute("SELECT * FROM ohlcv LIMIT 1").fetchone()
    candle = sqlite_row_to_candle(row)

    print(f"\nSQLite row → Candle:")
    if candle:
        print(f"  symbol={candle.symbol}, O={candle.open:.2f}, C={candle.close:.2f}")

    conn.close()


if __name__ == "__main__":
    example_1_create_from_scratch()
    example_2_pandas_conversion()
    example_3_json_serialization()
    example_4_legacy_adaptation()
    example_5_sqlite_conversion()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
