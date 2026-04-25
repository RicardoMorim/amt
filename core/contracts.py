"""
Shared data contracts for AMT / Kronos / Fusion Layer.

FASE 0 — Unificação de contratos de dados.
- Todos os tipos usam UTC (datetime.timezone.utc)
- timeframe_secs é sempre int
- direction normalizado para Direction enum
- Nomes consistentes: confidence, probability, score, action
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, TypeVar, cast


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Direction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"

    @classmethod
    def from_raw(cls, value: Any) -> "Direction":
        """Normaliza valores brutos para Direction."""
        if isinstance(value, cls):
            return value
        s = str(value).upper().strip()
        mapping = {
            "LONG": cls.LONG,
            "BUY": cls.LONG,
            "1": cls.LONG,
            "SHORT": cls.SHORT,
            "SELL": cls.SHORT,
            "-1": cls.SHORT,
            "FLAT": cls.FLAT,
            "HOLD": cls.FLAT,
            "0": cls.FLAT,
        }
        return mapping.get(s, cls.FLAT)


class MarketState(str, Enum):
    BALANCE = "BALANCE"
    IMBALANCE_UP = "IMBALANCE_UP"
    IMBALANCE_DOWN = "IMBALANCE_DOWN"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def from_raw(cls, value: Any) -> "MarketState":
        if isinstance(value, cls):
            return value
        s = str(value).upper().strip()
        mapping = {
            "BALANCE": cls.BALANCE,
            "IMBALANCE_UP": cls.IMBALANCE_UP,
            "IMBALANCE_DOWN": cls.IMBALANCE_DOWN,
            "UNKNOWN": cls.UNKNOWN,
        }
        return mapping.get(s, cls.UNKNOWN)


class SignalType(str, Enum):
    FALSE_BREAKOUT = "FALSE_BREAKOUT"
    VOLUME_IMBALANCE = "VOLUME_IMBALANCE"
    BALANCE_BREAKOUT = "BALANCE_BREAKOUT"
    CVD_DIVERGENCE = "CVD_DIVERGENCE"
    KRONOS_PREDICTION = "KRONOS_PREDICTION"
    FUSION_DECISION = "FUSION_DECISION"

    @classmethod
    def from_raw(cls, value: Any) -> "SignalType":
        if isinstance(value, cls):
            return value
        s = str(value).upper().strip()
        mapping = {st.value: st for st in cls}
        # Fallback: try direct match too
        return mapping.get(s, cls.FUSION_DECISION)


class SessionState(str, Enum):
    ASIA = "ASIA"
    LONDON = "LONDON"
    NY = "NY"
    OVERLAP = "OVERLAP"
    RTH = "RTH"
    ETH = "ETH"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def from_raw(cls, value: Any) -> "SessionState":
        if isinstance(value, cls):
            return value
        s = str(value).upper().strip()
        mapping = {ss.value: ss for ss in cls}
        return mapping.get(s, cls.UNKNOWN)


class TradeAction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    EXIT_LONG = "EXIT_LONG"
    EXIT_SHORT = "EXIT_SHORT"

    @classmethod
    def from_raw(cls, value: Any) -> "TradeAction":
        if isinstance(value, cls):
            return value
        s = str(value).upper().strip()
        mapping = {ta.value: ta for ta in cls}
        # Alias support
        alias_map = {
            "LONG": cls.BUY,
            "SHORT": cls.SELL,
            "FLAT": cls.HOLD,
        }
        return mapping.get(s, alias_map.get(s, cls.HOLD))


# ---------------------------------------------------------------------------
# Dataclasses — Core Contracts
# ---------------------------------------------------------------------------

@dataclass
class Candle:
    """Single OHLCV candle. All timestamps in UTC."""
    symbol: str
    timestamp: datetime  # UTC open time of the candle
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    amount: float = 0.0
    timeframe_secs: int = 60  # e.g., 60, 300, 3600

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": _dt_to_iso(self.timestamp),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "amount": self.amount,
            "timeframe_secs": self.timeframe_secs,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Candle":
        return cls(
            symbol=str(data["symbol"]),
            timestamp=_iso_to_dt(data["timestamp"]),
            open=float(data["open"]),
            high=float(data["high"]),
            low=float(data["low"]),
            close=float(data["close"]),
            volume=float(data.get("volume", 0.0)),
            amount=float(data.get("amount", 0.0)),
            timeframe_secs=int(data.get("timeframe_secs", 60)),
        )

    @classmethod
    def from_pandas_row(cls, row: Any, symbol: str = "", timeframe_secs: int = 60) -> "Candle":
        """Create a Candle from a pandas Series / dict-like row."""
        ts_raw = row.get("timestamp", row.get("time", row.get("date", None)))
        if isinstance(ts_raw, (int, float)):
            # Unix timestamp in seconds
            ts = datetime.fromtimestamp(ts_raw, tz=timezone.utc)
        elif isinstance(ts_raw, str):
            ts = _iso_to_dt(ts_raw)
        else:
            ts = datetime.now(tz=timezone.utc)

        return cls(
            symbol=symbol or str(row.get("symbol", "UNKNOWN")),
            timestamp=ts,
            open=float(row.get("open", row.get("Open", 0))),
            high=float(row.get("high", row.get("High", 0))),
            low=float(row.get("low", row.get("Low", 0))),
            close=float(row.get("close", row.get("Close", 0))),
            volume=float(row.get("volume", row.get("Volume", 0))),
            amount=float(row.get("amount", row.get("value", 0.0))),
            timeframe_secs=timeframe_secs,
        )


@dataclass
class CandleWindow:
    """A fixed-size window of consecutive candles for model input."""
    symbol: str
    timestamps: List[datetime] = field(default_factory=list)
    opens: List[float] = field(default_factory=list)
    highs: List[float] = field(default_factory=list)
    lows: List[float] = field(default_factory=list)
    closes: List[float] = field(default_factory=list)
    volumes: List[float] = field(default_factory=list)
    amounts: List[float] = field(default_factory=list)
    timeframe_secs: int = 60

    @property
    def length(self) -> int:
        return len(self.timestamps)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamps": [_dt_to_iso(t) for t in self.timestamps],
            "opens": list(self.opens),
            "highs": list(self.highs),
            "lows": list(self.lows),
            "closes": list(self.closes),
            "volumes": list(self.volumes),
            "amounts": list(self.amounts),
            "timeframe_secs": self.timeframe_secs,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CandleWindow":
        return cls(
            symbol=str(data["symbol"]),
            timestamps=[_iso_to_dt(t) for t in data.get("timestamps", [])],
            opens=[float(v) for v in data.get("opens", [])],
            highs=[float(v) for v in data.get("highs", [])],
            lows=[float(v) for v in data.get("lows", [])],
            closes=[float(v) for v in data.get("closes", [])],
            volumes=[float(v) for v in data.get("volumes", [])],
            amounts=[float(v) for v in data.get("amounts", [])],
            timeframe_secs=int(data.get("timeframe_secs", 60)),
        )

    @classmethod
    def from_candles(cls, candles: List[Candle]) -> "CandleWindow":
        if not candles:
            return cls(symbol=candles[0].symbol if candles else "")
        symbol = candles[0].symbol
        tf = candles[0].timeframe_secs
        return cls(
            symbol=symbol,
            timestamps=[c.timestamp for c in candles],
            opens=[c.open for c in candles],
            highs=[c.high for c in candles],
            lows=[c.low for c in candles],
            closes=[c.close for c in candles],
            volumes=[c.volume for c in candles],
            amounts=[c.amount for c in candles],
            timeframe_secs=tf,
        )


@dataclass
class AMTContextSnapshot:
    """Volume-profile + CVD context at a point in time."""
    symbol: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))  # UTC
    poc: float = 0.0
    vah: float = 0.0
    val: float = 0.0
    distance_to_poc_pct: float = 0.0
    volume_zscore: float = 0.0
    delta_zscore: float = 0.0
    cvd_slope_short: float = 0.0
    cvd_slope_long: float = 0.0
    market_state: MarketState = MarketState.UNKNOWN

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": _dt_to_iso(self.timestamp),
            "poc": self.poc,
            "vah": self.vah,
            "val": self.val,
            "distance_to_poc_pct": round(self.distance_to_poc_pct, 6),
            "volume_zscore": round(self.volume_zscore, 4),
            "delta_zscore": round(self.delta_zscore, 4),
            "cvd_slope_short": round(self.cvd_slope_short, 6),
            "cvd_slope_long": round(self.cvd_slope_long, 6),
            "market_state": self.market_state.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AMTContextSnapshot":
        return cls(
            symbol=str(data["symbol"]),
            timestamp=_iso_to_dt(data.get("timestamp", datetime.now(tz=timezone.utc))),
            poc=float(data.get("poc", 0.0)),
            vah=float(data.get("vah", 0.0)),
            val=float(data.get("val", 0.0)),
            distance_to_poc_pct=float(data.get("distance_to_poc_pct", 0.0)),
            volume_zscore=float(data.get("volume_zscore", 0.0)),
            delta_zscore=float(data.get("delta_zscore", 0.0)),
            cvd_slope_short=float(data.get("cvd_slope_short", 0.0)),
            cvd_slope_long=float(data.get("cvd_slope_long", 0.0)),
            market_state=MarketState.from_raw(data.get("market_state")),
        )

    @classmethod
    def from_volume_profile(cls, symbol: str, profile_data: Optional[Dict[str, Any]],
                            current_price: float = 0.0) -> "AMTContextSnapshot":
        """Adapt old volume_profile.py output into the new contract."""
        if not profile_data:
            return cls(symbol=symbol)

        poc = float(profile_data.get("poc", 0))
        vah = float(profile_data.get("vah", 0))
        val = float(profile_data.get("val", 0))

        dist_poc_pct = ((current_price - poc) / poc * 100) if poc else 0.0

        return cls(
            symbol=symbol,
            timestamp=datetime.now(tz=timezone.utc),
            poc=poc,
            vah=vah,
            val=val,
            distance_to_poc_pct=dist_poc_pct,
            market_state=_market_state_from_price(current_price, poc, vah, val),
        )


@dataclass
class AMTSignal:
    """Unified signal emitted by any AMT strategy."""
    symbol: str
    timestamp_event: datetime  # UTC
    direction: Direction
    signal_type: SignalType = SignalType.FUSION_DECISION
    session_state: SessionState = SessionState.UNKNOWN
    is_composite: bool = False

    # Confidence / probability — always [0, 1]
    confidence: float = 0.5
    probability: float = 0.0

    # Context snapshot (embedded for convenience)
    distance_to_poc_pct: float = 0.0
    volume_zscore: float = 0.0
    delta_zscore: float = 0.0
    cvd_slope_short: float = 0.0
    cvd_slope_long: float = 0.0

    # Optional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp_event": _dt_to_iso(self.timestamp_event),
            "direction": self.direction.value,
            "signal_type": self.signal_type.value,
            "session_state": self.session_state.value,
            "is_composite": self.is_composite,
            "confidence": round(self.confidence, 4),
            "probability": round(self.probability, 4),
            "distance_to_poc_pct": round(self.distance_to_poc_pct, 6),
            "volume_zscore": round(self.volume_zscore, 4),
            "delta_zscore": round(self.delta_zscore, 4),
            "cvd_slope_short": round(self.cvd_slope_short, 6),
            "cvd_slope_long": round(self.cvd_slope_long, 6),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AMTSignal":
        return cls(
            symbol=str(data["symbol"]),
            timestamp_event=_iso_to_dt(data.get("timestamp_event")),
            direction=Direction.from_raw(data.get("direction", "FLAT")),
            signal_type=SignalType.from_raw(data.get("signal_type")),
            session_state=SessionState.from_raw(data.get("session_state")),
            is_composite=bool(data.get("is_composite", False)),
            confidence=float(data.get("confidence", 0.5)),
            probability=float(data.get("probability", 0.0)),
            distance_to_poc_pct=float(data.get("distance_to_poc_pct", 0.0)),
            volume_zscore=float(data.get("volume_zscore", 0.0)),
            delta_zscore=float(data.get("delta_zscore", 0.0)),
            cvd_slope_short=float(data.get("cvd_slope_short", 0.0)),
            cvd_slope_long=float(data.get("cvd_slope_long", 0.0)),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_legacy_dict(cls, data: Dict[str, Any]) -> "AMTSignal":
        """Adapt old dict-based signals (string direction, string signal_type) to the new contract."""
        return cls(
            symbol=str(data.get("symbol", data.get("instrument", "UNKNOWN"))),
            timestamp_event=_iso_to_dt(data.get("timestamp_event", data.get("timestamp", datetime.now(tz=timezone.utc)))),
            direction=Direction.from_raw(data.get("direction", data.get("side", "FLAT"))),
            signal_type=SignalType.from_raw(data.get("signal_type", data.get("type", "FUSION_DECISION"))),
            session_state=SessionState.from_raw(data.get("session_state")),
            is_composite=bool(data.get("is_composite", False)),
            confidence=float(data.get("confidence", 0.5)),
            probability=float(data.get("probability", float(data.get("prob", 0.0)))),
            distance_to_poc_pct=float(data.get("distance_to_poc_pct", 0.0)),
            volume_zscore=float(data.get("volume_zscore", 0.0)),
            delta_zscore=float(data.get("delta_zscore", data.get("delta_z", 0.0))),
            cvd_slope_short=float(data.get("cvd_slope_short", 0.0)),
            cvd_slope_long=float(data.get("cvd_slope_long", 0.0)),
        )


@dataclass
class KronosPrediction:
    """Unified contract for Kronos model predictions."""
    symbol: str
    timestamp: datetime  # UTC — prediction target time
    direction: Direction
    confidence: float = 0.5  # [0, 1]

    # Optional: raw logits / probabilities per class
    probability_long: float = 0.0
    probability_short: float = 0.0

    # Context used for the prediction (for audit trail)
    context_length: int = 0
    timeframe_secs: int = 60

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": _dt_to_iso(self.timestamp),
            "direction": self.direction.value,
            "confidence": round(self.confidence, 4),
            "probability_long": round(self.probability_long, 4),
            "probability_short": round(self.probability_short, 4),
            "context_length": self.context_length,
            "timeframe_secs": self.timeframe_secs,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KronosPrediction":
        return cls(
            symbol=str(data["symbol"]),
            timestamp=_iso_to_dt(data.get("timestamp", data.get("target_time"))),
            direction=Direction.from_raw(data.get("direction", data.get("prediction", "FLAT"))),
            confidence=float(data.get("confidence", float(data.get("conf", 0.5)))),
            probability_long=float(data.get("probability_long", data.get("prob_long", 0.0))),
            probability_short=float(data.get("probability_short", data.get("prob_short", 0.0))),
            context_length=int(data.get("context_length", data.get("lookback", 0))),
            timeframe_secs=int(data.get("timeframe_secs", 60)),
        )


@dataclass
class FusionDecision:
    """Output of the fusion layer — final trading decision."""
    symbol: str
    timestamp: datetime  # UTC — when decision was made
    action: TradeAction
    direction: Direction = Direction.FLAT

    # Scores from each source
    amt_score: float = 0.0       # [-1, 1] or [0, 1]
    kronos_score: float = 0.0    # [-1, 1] or [0, 1]
    fusion_score: float = 0.0    # weighted composite

    confidence: float = 0.5      # [0, 1] overall confidence
    probability: float = 0.0     # [0, 1] estimated win rate

    # Which signals contributed (for audit)
    contributing_signals: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": _dt_to_iso(self.timestamp),
            "action": self.action.value,
            "direction": self.direction.value,
            "amt_score": round(self.amt_score, 4),
            "kronos_score": round(self.kronos_score, 4),
            "fusion_score": round(self.fusion_score, 4),
            "confidence": round(self.confidence, 4),
            "probability": round(self.probability, 4),
            "contributing_signals": self.contributing_signals,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FusionDecision":
        return cls(
            symbol=str(data["symbol"]),
            timestamp=_iso_to_dt(data.get("timestamp", data.get("decision_time"))),
            action=TradeAction.from_raw(data.get("action", data.get("side", "HOLD"))),
            direction=Direction.from_raw(data.get("direction", "FLAT")),
            amt_score=float(data.get("amt_score", 0.0)),
            kronos_score=float(data.get("kronos_score", float(data.get("model_score", 0.0)))),
            fusion_score=float(data.get("fusion_score", float(data.get("score", 0.0)))),
            confidence=float(data.get("confidence", 0.5)),
            probability=float(data.get("probability", 0.0)),
            contributing_signals=data.get("contributing_signals", data.get("signals", [])),
        )


@dataclass
class TradeLabel:
    """Ground-truth label for supervised training."""
    symbol: str
    timestamp: datetime  # UTC — when the trade would execute
    action: TradeAction
    entry_price: float
    exit_price: float = 0.0
    pnl_pct: float = 0.0
    holding_bars: int = 0
    stop_loss: float = 0.0
    take_profit: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": _dt_to_iso(self.timestamp),
            "action": self.action.value,
            "entry_price": round(self.entry_price, 6),
            "exit_price": round(self.exit_price, 6),
            "pnl_pct": round(self.pnl_pct, 6),
            "holding_bars": self.holding_bars,
            "stop_loss": round(self.stop_loss, 6),
            "take_profit": round(self.take_profit, 6),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TradeLabel":
        return cls(
            symbol=str(data["symbol"]),
            timestamp=_iso_to_dt(data.get("timestamp", data.get("entry_time"))),
            action=TradeAction.from_raw(data.get("action", data.get("side", "HOLD"))),
            entry_price=float(data.get("entry_price", 0.0)),
            exit_price=float(data.get("exit_price", 0.0)),
            pnl_pct=float(data.get("pnl_pct", float(data.get("return_pct", 0.0)))),
            holding_bars=int(data.get("holding_bars", data.get("bars_held", 0))),
            stop_loss=float(data.get("stop_loss", 0.0)),
            take_profit=float(data.get("take_profit", 0.0)),
        )


# ---------------------------------------------------------------------------
# Type aliases for convenience
# ---------------------------------------------------------------------------

Contract = Candle | AMTSignal | KronosPrediction | FusionDecision | TradeLabel
SignalDict = Dict[str, Any]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dt_to_iso(dt: Optional[datetime]) -> str:
    """Convert datetime to ISO-8601 UTC string."""
    if dt is None:
        return ""
    # Ensure UTC
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def _iso_to_dt(iso_str: Any) -> datetime:
    """Parse ISO-8601 string to UTC datetime."""
    if isinstance(iso_str, datetime):
        return iso_str.replace(tzinfo=timezone.utc) if iso_str.tzinfo is None else iso_str
    s = str(iso_str).strip()
    # Handle 'Z' suffix
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def _market_state_from_price(price: float, poc: float, vah: float, val: float) -> MarketState:
    """Recreate market state logic from core/market_state.py."""
    if not poc or not vah or not val:
        return MarketState.UNKNOWN
    buffer = 0.01 / 100  # 0.01%
    upper = vah * (1 + buffer)
    lower = val * (1 - buffer)
    if price > upper:
        return MarketState.IMBALANCE_UP
    elif price < lower:
        return MarketState.IMBALANCE_DOWN
    return MarketState.BALANCE


def contracts_to_json(contracts: List[Contract]) -> str:
    """Serialize a list of any contract types to JSON."""
    result = []
    for c in contracts:
        if hasattr(c, "to_dict"):
            result.append(c.to_dict())
        else:
            result.append(c)  # type: ignore[arg-type]
    return json.dumps(result, indent=2, default=str)


def contracts_from_json(json_str: str) -> List[Contract]:
    """Deserialize JSON array to contract instances (best-effort)."""
    import json as _json
    data = _json.loads(json_str)
    if not isinstance(data, list):
        data = [data]

    result: List[Contract] = []
    for item in data:
        stype = item.get("signal_type", "")
        if "action" in item and "fusion_score" in item:
            result.append(FusionDecision.from_dict(item))
        elif "probability_long" in item or "kronos_score" in item:
            result.append(KronosPrediction.from_dict(item))
        elif stype or "direction" in item and "signal_type" in item:
            result.append(AMTSignal.from_dict(item))
        elif "poc" in item and "vah" in item:
            result.append(AMTContextSnapshot.from_dict(item))
        elif "entry_price" in item and "action" in item:
            result.append(TradeLabel.from_dict(item))
        elif "closes" in item or "opens" in item:
            result.append(CandleWindow.from_dict(item))
        else:
            # Try Candle first, then fallback to dict
            try:
                result.append(Candle.from_dict(item))
            except (KeyError, TypeError):
                result.append(cast(Contract, item))  # type: ignore[assignment]
    return result
