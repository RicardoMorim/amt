"""
Unit tests for core/volume_profile.py
"""
import pytest
import pandas as pd
from core.volume_profile import calculate_volume_profile, SessionProfileManager


# ── calculate_volume_profile ──────────────────────────────────────────────────

def test_poc_is_highest_volume_bucket():
    df = pd.DataFrame({'close': [100, 100, 100, 101, 102], 'volume': [10, 10, 10, 1, 1]})
    result = calculate_volume_profile(df, tick_size=1)
    assert result['poc'] == 100


def test_vah_above_val():
    df = pd.DataFrame({'close': [98, 99, 100, 101, 102], 'volume': [5, 10, 20, 10, 5]})
    result = calculate_volume_profile(df, tick_size=1)
    assert result['vah'] >= result['val']


def test_empty_df_returns_none():
    assert calculate_volume_profile(pd.DataFrame()) is None


def test_value_area_contains_poc():
    df = pd.DataFrame({'close': [98, 99, 100, 101, 102], 'volume': [5, 10, 20, 10, 5]})
    result = calculate_volume_profile(df, tick_size=1)
    assert result['val'] <= result['poc'] <= result['vah']


# ── SessionProfileManager ─────────────────────────────────────────────────────

def test_session_manager_basic():
    mgr = SessionProfileManager(tick_size=1.0)
    mgr.update(100, 10)
    mgr.update(101, 5)
    mgr.update(100, 3)
    levels = mgr.get_levels()
    assert levels is not None
    assert levels['poc'] == 100  # bucket 100 has 13 volume vs 5


def test_dirty_flag_cache():
    mgr = SessionProfileManager(tick_size=1.0)
    mgr.update(100, 10)
    first = mgr.get_levels()
    # Second call without update — must return same cached object
    second = mgr.get_levels()
    assert first is second


def test_dirty_flag_invalidated_on_update():
    mgr = SessionProfileManager(tick_size=1.0)
    mgr.update(100, 10)
    first = mgr.get_levels()
    mgr.update(200, 100)  # large update shifts POC
    second = mgr.get_levels()
    assert second is not first
    assert second['poc'] == 200


def test_reset_clears_profile():
    mgr = SessionProfileManager(tick_size=1.0)
    mgr.update(100, 10)
    mgr.reset()
    assert mgr.get_levels() is None
    assert mgr.total_volume == 0.0
