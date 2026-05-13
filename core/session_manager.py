"""
core/session_manager.py — Automatic RTH/ETH session manager.

In AMT, different sessions (RTH, ETH, Asia, London, NY) have distinct
Value Areas that must not be mixed. This module handles automatic session
detection and profile resets based on UTC timestamps, so callers never
need to manually call SessionProfileManager.reset().

Supported session schedules:
  - Equities / Futures (CME): RTH 14:30–21:00 UTC, ETH remainder
  - Crypto: 24/7, daily sessions reset at 00:00 UTC
  - FX: Asia (00:00–08:00), London (07:00–16:00), NY (13:00–22:00) UTC
"""

from __future__ import annotations

from datetime import datetime, time, timezone
from enum import Enum
from typing import Optional

from core.volume_profile import SessionProfileManager


class SessionSchedule(str, Enum):
    CRYPTO_DAILY = "CRYPTO_DAILY"       # Reset at 00:00 UTC every day
    CME_RTH_ETH  = "CME_RTH_ETH"        # CME equities/futures RTH/ETH split
    FX_SESSIONS  = "FX_SESSIONS"        # Forex Asia / London / NY


# UTC boundary times
_CME_RTH_START = time(14, 30)   # 14:30 UTC (09:30 ET)
_CME_RTH_END   = time(21, 0)    # 21:00 UTC (16:00 ET)

_FX_ASIA_START   = time(0,  0)
_FX_ASIA_END     = time(8,  0)
_FX_LONDON_START = time(7,  0)
_FX_LONDON_END   = time(16, 0)
_FX_NY_START     = time(13, 0)
_FX_NY_END       = time(22, 0)


def _get_cme_session(t: time) -> str:
    if _CME_RTH_START <= t < _CME_RTH_END:
        return "RTH"
    return "ETH"


def _get_fx_session(t: time) -> str:
    in_london = _FX_LONDON_START <= t < _FX_LONDON_END
    in_ny     = _FX_NY_START <= t < _FX_NY_END
    in_asia   = t >= _FX_ASIA_START and t < _FX_ASIA_END

    if in_london and in_ny:
        return "OVERLAP"
    if in_ny:
        return "NY"
    if in_london:
        return "LONDON"
    if in_asia:
        return "ASIA"
    return "ETH"


class SessionManager:
    """
    Wraps SessionProfileManager with automatic session detection and reset.

    Usage:
        mgr = SessionManager(schedule=SessionSchedule.CME_RTH_ETH)
        for candle in candles:
            mgr.update(candle.timestamp, candle.close, candle.volume)
            levels = mgr.get_levels()   # always reflects current session

    Raises:
        ValueError: on invalid parameters.
        TypeError: on wrong input types.
        RuntimeError: if get_levels() is called before any data is added.
    """

    def __init__(
        self,
        schedule: SessionSchedule = SessionSchedule.CRYPTO_DAILY,
        tick_size: Optional[float] = None,
        value_area_pct: float = 0.68,
    ):
        if not isinstance(schedule, SessionSchedule):
            raise TypeError(
                f"schedule must be a SessionSchedule enum, got {type(schedule).__name__}. "
                f"Valid values: {[s.value for s in SessionSchedule]}"
            )
        if tick_size is not None and tick_size <= 0:
            raise ValueError(f"tick_size must be positive, got {tick_size}")
        if not (0 < value_area_pct < 1):
            raise ValueError(f"value_area_pct must be between 0 and 1, got {value_area_pct}")

        self.schedule       = schedule
        self.tick_size      = tick_size
        self.value_area_pct = value_area_pct

        self._profile = SessionProfileManager(
            tick_size=tick_size,
            value_area_pct=value_area_pct,
        )
        self._current_session: Optional[str] = None
        self._last_day: Optional[int] = None  # used for CRYPTO_DAILY

    @property
    def current_session(self) -> Optional[str]:
        """Returns the name of the current session (e.g. 'RTH', 'ASIA', '2026-05-13')."""
        return self._current_session

    def _detect_session(self, ts: datetime) -> str:
        """Returns a session key string for the given UTC timestamp."""
        t = ts.astimezone(timezone.utc).time()
        d = ts.astimezone(timezone.utc).date()

        if self.schedule == SessionSchedule.CRYPTO_DAILY:
            return d.isoformat()  # new session every UTC day
        elif self.schedule == SessionSchedule.CME_RTH_ETH:
            return _get_cme_session(t)
        elif self.schedule == SessionSchedule.FX_SESSIONS:
            return _get_fx_session(t)
        else:
            raise ValueError(f"Unknown SessionSchedule: {self.schedule}")

    def update(self, timestamp: datetime, price: float, volume: float) -> bool:
        """
        Feed a new price/volume tick or candle close.

        Args:
            timestamp: UTC datetime of the candle/tick.
            price: Trade or close price.
            volume: Trade or candle volume.

        Returns:
            True if a session reset occurred (new session started), False otherwise.

        Raises:
            TypeError: if timestamp is not a datetime.
            ValueError: if price or volume are invalid.
        """
        if not isinstance(timestamp, datetime):
            raise TypeError(f"timestamp must be a datetime object, got {type(timestamp).__name__}")
        if price <= 0:
            raise ValueError(f"price must be positive, got {price}")
        if volume < 0:
            raise ValueError(f"volume must be non-negative, got {volume}")

        session = self._detect_session(timestamp)
        reset_occurred = False

        if session != self._current_session:
            self._profile.reset()
            self._current_session = session
            reset_occurred = True

        self._profile.update(price, volume)
        return reset_occurred

    def get_levels(self) -> dict:
        """
        Returns POC/VAH/VAL for the current session.

        Raises:
            RuntimeError: if no data has been fed yet.
        """
        return self._profile.get_levels()

    def reset(self):
        """Manually force a session reset (e.g. for backtesting)."""
        self._profile.reset()
        self._current_session = None
