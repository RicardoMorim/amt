def identify_market_state(price, profile_data, balance_threshold_pct=0.01):
    """
    Identifies if the market is in balance or imbalance based on current price relative to the Value Area.
    
    Args:
        price: Current asset price.
        profile_data: dict from calculate_volume_profile containing 'vah', 'val', 'poc'.
        balance_threshold_pct: Buffer percentage around VAH/VAL to consider still in balance.
                               Ex: 0.05% buffer means price can poke out slightly without triggering imbalance.
        
    Returns:
        str: 'BALANCE', 'IMBALANCE_UP', or 'IMBALANCE_DOWN'
    """
    if not profile_data:
        return 'UNKNOWN'
        
    vah = profile_data['vah']
    val = profile_data['val']
    
    # Adding a slight buffer so a single tick outside doesn't instantly flip state
    upper_bound = vah * (1 + balance_threshold_pct/100)
    lower_bound = val * (1 - balance_threshold_pct/100)
    
    if price > upper_bound:
        return 'IMBALANCE_UP'
    elif price < lower_bound:
        return 'IMBALANCE_DOWN'
    else:
        return 'BALANCE'

def check_false_breakout(current_price, previous_prices, profile_data, cvd_data=None):
    """
    Heuristic to detect a 'Look Above and Fail' or 'Look Below and Fail'.
    
    Args:
        current_price: The latest price in the session.
        previous_prices: List/Series of the most recent close prices (e.g. last 3 candles).
        profile_data: The active volume profile.
        
    Returns:
        dict with 'signal' and 'direction' if a false breakout is detected, else None.
    """
    if not profile_data or len(previous_prices) < 2:
        return None
        
    vah = profile_data['vah']
    val = profile_data['val']
    
    was_above_vah = any(p > vah for p in previous_prices)
    is_below_vah = current_price < vah
    
    if was_above_vah and is_below_vah:
        return {'signal_type': 'FALSE_BREAKOUT', 'direction': 'SHORT', 'target': profile_data['poc']}
        
    was_below_val = any(p < val for p in previous_prices)
    is_above_val = current_price > val
    
    if was_below_val and is_above_val:
        return {'signal_type': 'FALSE_BREAKOUT', 'direction': 'LONG', 'target': profile_data['poc']}
        
    return None
