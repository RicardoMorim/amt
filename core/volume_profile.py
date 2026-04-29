import numpy as np
import pandas as pd


def _calculate_value_area_2_line(profile, poc, poc_idx, total_volume, value_area_pct):
    target_vol = total_volume * value_area_pct
    current_vol = float(profile.iloc[poc_idx])

    prices = profile.index.values
    volumes = profile.values

    up_idx = poc_idx + 1
    down_idx = poc_idx - 1

    vah = poc
    val = poc

    while current_vol < target_vol and (up_idx < len(prices) or down_idx >= 0):
        vol_up = 0.0
        if up_idx < len(prices):
            vol_up += volumes[up_idx]
            if up_idx + 1 < len(prices):
                vol_up += volumes[up_idx + 1]

        vol_down = 0.0
        if down_idx >= 0:
            vol_down += volumes[down_idx]
            if down_idx - 1 >= 0:
                vol_down += volumes[down_idx - 1]

        if vol_up == 0 and vol_down == 0:
            break

        if vol_up >= vol_down:
            if up_idx < len(prices):
                current_vol += float(volumes[up_idx])
                vah = prices[up_idx]
                up_idx += 1
            if current_vol < target_vol and up_idx < len(prices):
                current_vol += float(volumes[up_idx])
                vah = prices[up_idx]
                up_idx += 1
        else:
            if down_idx >= 0:
                current_vol += float(volumes[down_idx])
                val = prices[down_idx]
                down_idx -= 1
            if current_vol < target_vol and down_idx >= 0:
                current_vol += float(volumes[down_idx])
                val = prices[down_idx]
                down_idx -= 1

    if vah < val:
        vah, val = val, vah

    return vah, val


def calculate_volume_profile(df, price_col='close', vol_col='volume', value_area_pct=0.68, tick_size=None):
    if df.empty:
        return None

    df = df.copy()
    
    if tick_size is None or tick_size <= 0:
        last_price = float(df[price_col].iloc[-1])
        tick_size = max(1e-8, last_price * 0.0005)

    df.loc[:, 'price_bucket'] = (df[price_col] / tick_size).round() * tick_size
    df.loc[:, '_profile_volume'] = pd.to_numeric(df[vol_col], errors='coerce').fillna(0.0).abs()
    profile = df.groupby('price_bucket')['_profile_volume'].sum().sort_index()

    if profile.empty:
        return None

    total_volume = profile.sum()
    poc = profile.idxmax()
    poc_idx = profile.index.get_loc(poc)

    vah, val = _calculate_value_area_2_line(profile, poc, poc_idx, total_volume, value_area_pct)

    return {
        'poc': poc,
        'vah': vah,
        'val': val,
        'profile': profile
    }


class SessionProfileManager:
    def __init__(self, tick_size=None, value_area_pct=0.68):
        self.tick_size = tick_size
        self.value_area_pct = value_area_pct
        self.price_buckets: dict = {}
        self.total_volume: float = 0.0

        self._dirty: bool = True
        self._cached_levels: dict | None = None

    def update(self, price: float, volume: float):
        if self.tick_size is None or self.tick_size <= 0:
            self.tick_size = max(1e-8, price * 0.0005)

        bucket = round(price / self.tick_size) * self.tick_size
        self.price_buckets[bucket] = self.price_buckets.get(bucket, 0) + volume
        self.total_volume += volume
        self._dirty = True

    def reset(self):
        self.price_buckets.clear()
        self.total_volume = 0.0
        self._dirty = True
        self._cached_levels = None

    def get_levels(self) -> dict | None:
        if not self._dirty and self._cached_levels is not None:
            return self._cached_levels

        if not self.price_buckets:
            return None

        profile = pd.Series(self.price_buckets).sort_index()

        poc = profile.idxmax()
        poc_idx = profile.index.get_loc(poc)

        vah, val = _calculate_value_area_2_line(profile, poc, poc_idx, self.total_volume, self.value_area_pct)

        self._cached_levels = {'poc': poc, 'vah': vah, 'val': val}
        self._dirty = False
        return self._cached_levels
