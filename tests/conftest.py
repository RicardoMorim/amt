import sys
from unittest.mock import MagicMock

# Mock pandas globally for tests if it's not installed
try:
    import pandas
except ImportError:
    mock_pd = MagicMock()
    sys.modules["pandas"] = mock_pd

try:
    import numpy
except ImportError:
    mock_np = MagicMock()
    mock_np.float32 = float
    sys.modules["numpy"] = mock_np

try:
    import torch
except ImportError:
    mock_torch = MagicMock()
    sys.modules["torch"] = mock_torch

try:
    import requests
except ImportError:
    mock_requests = MagicMock()
    sys.modules["requests"] = mock_requests

try:
    import sklearn
    from sklearn.preprocessing import LabelEncoder
except ImportError:
    mock_sklearn = MagicMock()
    sys.modules["sklearn"] = mock_sklearn
    sys.modules["sklearn.preprocessing"] = MagicMock()
    sys.modules["sklearn.metrics"] = MagicMock()

try:
    import scipy
except ImportError:
    mock_scipy = MagicMock()
    sys.modules["scipy"] = mock_scipy
try:
    import torch.nn
except ImportError:
    mock_torch_nn = MagicMock()
    mock_torch.nn = mock_torch_nn
    sys.modules["torch.nn"] = mock_torch_nn
