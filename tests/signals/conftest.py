import sys
from unittest.mock import MagicMock

# Mock numpy locally for tests if it's not installed
try:
    import numpy
except ImportError:
    mock_np = MagicMock()
    sys.modules["numpy"] = mock_np
