import numpy as np
import pandas as pd


def calculate_volume_profile(df, price_col='close', vol_col='volume', value_area_pct=0.68, tick_size=0.25):
    """
    Calculates Volume Profile, POC, VAH, and VAL from an OHLCV or tick dataframe.

    Args:
        df: DataFrame containing price and volume data.
        price_col: Name of the column representing price. Use 'close' for OHLCV, or 'price' for tick.
        vol_col: Name of the column representing volume.
        value_area_pct: Percentage of volume to include in the Value Area (typically 68% or 70%).
        tick_size: The minimum tick size of the instrument to bucket prices.

    Returns:
        dict containing 'poc', 'vah', 'val', and the full 'profile' Series.
    """
    if df.empty:
        return None

    df = df.copy()
    df['price_bucket'] = (df[price_col] / tick_size).round() * tick_size
    df['_profile_volume'] = pd.to_numeric(df[vol_col], errors='coerce').fillna(0.0).abs()
    profile = df.groupby('price_bucket')['_profile_volume'].sum().sort_index()

    if profile.empty:
        return None

    total_volume = profile.sum()
    poc = profile.idxmax()
    poc_vol = profile.max()

    target_vol = total_volume * value_area_pct
    current_vol = poc_vol

    poc_idx = profile.index.get_loc(poc)
    up_idx = poc_idx + 1
    down_idx = poc_idx - 1

    vah = poc
    val = poc

    prices = profile.index.values
    volumes = profile.values

    while current_vol < target_vol and (up_idx < len(prices) or down_idx >= 0):
        can_take_up = up_idx < len(prices)
        can_take_down = down_idx >= 0

        if not can_take_up and not can_take_down:
            break

        if can_take_up and (not can_take_down or volumes[up_idx] >= volumes[down_idx]):
            current_vol += float(volumes[up_idx])
            vah = prices[up_idx]
            up_idx += 1
        else:
            current_vol += float(volumes[down_idx])
            val = prices[down_idx]
            down_idx -= 1

    if vah < val:
        vah, val = val, vah

    return {
        'poc': poc,
        'vah': vah,
        'val': val,
        'profile': profile
    }


class SessionProfileManager:
    """
    Manages building a live profile and updating VAH/VAL dynamically tick-by-tick.

    Performance optimisation:
      - Uses a dirty flag: get_levels() only recomputes POC/VAH/VAL when the
        profile has changed since the last call.  In a live bot receiving thousands
        of ticks per minute this avoids redundant O(n) recomputes on every analysis.
    """

    def __init__(self, tick_size=0.25, value_area_pct=0.68):
        self.tick_size = tick_size
        self.value_area_pct = value_area_pct
        self.price_buckets: dict = {}
        self.total_volume: float = 0.0

        # Dirty flag cache
        self._dirty: bool = True
        self._cached_levels: dict | None = None

    def update(self, price: float, volume: float):
        """Add a single tick/trade to the profile and mark cache as stale."""
        bucket = round(price / self.tick_size) * self.tick_size
        self.price_buckets[bucket] = self.price_buckets.get(bucket, 0) + volume
        self.total_volume += volume
        self._dirty = True

    def reset(self):
        """Clear the profile for a new trading session (call at UTC midnight)."""
        self.price_buckets.clear()
        self.total_volume = 0.0
        self._dirty = True
        self._cached_levels = None

    def get_levels(self) -> dict | None:
        """Return POC/VAH/VAL, recomputing only if the profile changed since last call."""
        if not self._dirty and self._cached_levels is not None:
            return self._cached_levels

        if not self.price_buckets:
            return None

        profile = pd.Series(self.price_buckets).sort_index()

        poc = profile.idxmax()
        poc_vol = profile.max()

        target_vol = self.total_volume * self.value_area_pct
        current_vol = poc_vol

        prices = profile.index.values
        volumes = profile.values

        poc_idx = profile.index.get_loc(poc)
        up_idx = poc_idx + 1
        down_idx = poc_idx - 1

        vah = poc
        val = poc

        while current_vol < target_vol and (up_idx < len(prices) or down_idx >= 0):
            can_take_up = up_idx < len(prices)
            can_take_down = down_idx >= 0

            if not can_take_up and not can_take_down:
                break

            if can_take_up and (not can_take_down or volumes[up_idx] >= volumes[down_idx]):
                current_vol += float(volumes[up_idx])
                vah = prices[up_idx]
                up_idx += 1
            else:
                current_vol += float(volumes[down_idx])
                val = prices[down_idx]
                down_idx -= 1

        self._cached_levels = {'poc': poc, 'vah': vah, 'val': val}
        self._dirty = False
        return self._cached_levels
