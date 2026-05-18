import sys
from unittest.mock import MagicMock

# Mock pandas globally for tests if it's not installed
try:
    import pandas
except ImportError:
    mock_pd = MagicMock()
    sys.modules["pandas"] = mock_pd

# Mock joblib globally for tests if it's not installed
try:
    import joblib
except ImportError:
    mock_joblib = MagicMock()
    sys.modules["joblib"] = mock_joblib

# Mock numpy globally for tests if it's not installed
try:
    import numpy
except ImportError:
    mock_np = MagicMock()
    # Also add some common numpy attributes accessed on import
    mock_np.float32 = float
    mock_np.float64 = float
    mock_np.int32 = int
    mock_np.int64 = int
    mock_np.array = lambda x, **kwargs: x
    sys.modules["numpy"] = mock_np

# Mock torch globally for tests if it's not installed
try:
    import torch
except ImportError:
    mock_torch = MagicMock()
    mock_torch.nn = MagicMock()
    mock_torch.nn.Module = object
    mock_torch.Tensor = MagicMock
    sys.modules["torch"] = mock_torch
sys.modules["torch.nn"] = mock_torch.nn
# Mock xgboost
try:
    import xgboost
except ImportError:
    mock_xgb = MagicMock()
    sys.modules["xgboost"] = mock_xgb
