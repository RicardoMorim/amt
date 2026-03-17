import pandas as pd
import numpy as np
import sys
sys.path.append('.')

from core.market_state import check_false_breakout
from signals.volume_imbalance import detect_cvd_divergence

def test_false_breakout():
    print("Testing check_false_breakout...")
    profile = {'vah': 102, 'val': 98, 'poc': 100}
    
    # Test 1: Previous was > 102, current is < 102 (SHORT breakout)
    res = check_false_breakout(current_price=101, previous_prices=[100, 103], profile_data=profile)
    print("Test 1 (SHORT):", res)
    assert res and res['direction'] == 'SHORT'
    
    # Test 2: Previous was < 98, current is > 98 (LONG breakout)
    res2 = check_false_breakout(current_price=99, previous_prices=[100, 97], profile_data=profile)
    print("Test 2 (LONG):", res2)
    assert res2 and res2['direction'] == 'LONG'
    
    # Test 3: Old logic bug where n-2 was above but n-1 was below, and current is below
    res3 = check_false_breakout(current_price=101, previous_prices=[100, 103, 101], profile_data=profile)
    print("Test 3 (lagging breakout should be None):", res3)
    assert res3 is None
    
    print("check_false_breakout passed!\n")

def test_cvd_divergence():
    print("Testing detect_cvd_divergence...")
    # window = 2. Looking at [102, 101, 105, 104, 103, 106]
    # Recent (window+1 to 1) means we look at the 2 candles before 106. i.e., 104 and 103.
    # So max price = 104, max cvd = 15.
    # Current price = 106, Current CVD = 18.
    # 106 > 104 (New High). 18 > 15 (New High in CVD). Result should be None.
    
    price = pd.Series([100, 102, 101, 105, 104, 103, 106])
    cvd = pd.Series([0, 10, 5, 20, 15, 10, 18])
    res = detect_cvd_divergence(price, cvd, window=2)
    print("Test 1 (No Divergence):", res)
    assert res is None
    
    # Test 2: Divergence Short
    # price = 106 (New high against 104)
    # cvd = 14 (Failed to break 15)
    cvd = pd.Series([0, 10, 5, 20, 15, 10, 14])
    res2 = detect_cvd_divergence(price, cvd, window=2)
    print("Test 2 (Short Divergence):", res2)
    assert res2 and res2['direction'] == 'SHORT'
    
    print("detect_cvd_divergence passed!\n")

if __name__ == "__main__":
    test_false_breakout()
    test_cvd_divergence()
    print("All tests passed successfully.")
