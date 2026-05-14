import sys
from unittest.mock import MagicMock

def _mock_module(name):
    try:
        __import__(name)
    except ImportError:
        sys.modules[name] = MagicMock()

_mock_module("pandas")
_mock_module("numpy")
_mock_module("requests")
_mock_module("sklearn")
_mock_module("sklearn.preprocessing")
_mock_module("joblib")

_mock_module("torch")
_mock_module("torch.nn")
_mock_module("xgboost")
