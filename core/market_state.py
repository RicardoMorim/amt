
def identify_market_state(price: float, profile_data: dict, balance_threshold_pct: float = 0.01) -> str:
    """
    Identifies if the market is in balance or imbalance based on current price relative to the Value Area.

    Args:
        price: Current asset price.
        profile_data: dict from calculate_volume_profile containing 'vah', 'val', 'poc'.
        balance_threshold_pct: Buffer percentage around VAH/VAL (e.g. 0.01 = 0.01%).

    Returns:
        str: 'BALANCE', 'IMBALANCE_UP', or 'IMBALANCE_DOWN'

    Raises:
        ValueError: if profile_data is None/empty or price is invalid.
        KeyError: if 'vah' or 'val' keys are missing from profile_data.
    """
    if not profile_data:
        raise ValueError("profile_data is required and cannot be None or empty")
    if price <= 0:
        raise ValueError(f"price must be positive, got {price}")

    # Raises KeyError with a clear message if keys are absent
    vah = profile_data['vah']
    val = profile_data['val']

    if vah <= 0 or val <= 0:
        raise ValueError(f"vah ({vah}) and val ({val}) must be positive prices")
    if val >= vah:
        raise ValueError(f"val ({val}) must be strictly less than vah ({vah})")

    upper_bound = vah * (1 + balance_threshold_pct / 100)
    lower_bound = val * (1 - balance_threshold_pct / 100)

    if price > upper_bound:
        return 'IMBALANCE_UP'
    elif price < lower_bound:
        return 'IMBALANCE_DOWN'
    else:
        return 'BALANCE'


def check_false_breakout(current_price: float, previous_prices, profile_data: dict,
                         cvd_data=None) -> dict | None:
    """
    Heuristic to detect a 'Look Above and Fail' or 'Look Below and Fail'.

    Raises:
        ValueError: if profile_data is None/empty, current_price is invalid,
                    or previous_prices is empty.
        KeyError: if 'vah', 'val', or 'poc' keys are missing from profile_data.
    """
    if not profile_data:
        raise ValueError("profile_data is required and cannot be None or empty")
    if current_price <= 0:
        raise ValueError(f"current_price must be positive, got {current_price}")
    if len(previous_prices) < 1:
        raise ValueError("previous_prices must contain at least 1 price")

    vah = profile_data['vah']
    val = profile_data['val']
    poc = profile_data['poc']  # raises KeyError if missing

    if vah <= 0 or val <= 0:
        raise ValueError(f"vah ({vah}) and val ({val}) must be positive prices")

    prev_price = previous_prices[-1]

    was_above_vah = prev_price > vah
    is_below_vah  = current_price < vah

    if was_above_vah and is_below_vah:
        return {'signal_type': 'FALSE_BREAKOUT', 'direction': 'SHORT', 'target': poc}

    was_below_val = prev_price < val
    is_above_val  = current_price > val

    if was_below_val and is_above_val:
        return {'signal_type': 'FALSE_BREAKOUT', 'direction': 'LONG', 'target': poc}

    return None
