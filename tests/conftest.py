import sys
from unittest.mock import MagicMock

# Mock pandas globally for tests if it's not installed
try:
    import pandas
except ImportError:
    mock_pd = MagicMock()
    sys.modules["pandas"] = mock_pd
