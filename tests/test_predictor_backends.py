import json
import pytest

import numpy as np
import torch

from ml.predictor import AMTPredictor, _TabularMLP


class DummyXGBModel:
    def predict_proba(self, X):
        # deterministic positive confidence
        arr = np.asarray(X)
        p = np.full((arr.shape[0], 2), [0.3, 0.7], dtype=float)
        return p


def _write_json(path, payload):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f)



import sys
from unittest.mock import MagicMock

try:
    import xgboost as xgb
except ImportError:
    xgb = MagicMock()
    sys.modules['xgboost'] = xgb

def test_predictor_xgb_backend(tmp_path):
    if isinstance(__import__('sys').modules.get('xgboost'), type(MagicMock())): pytest.skip('xgboost is mocked')
    model_path = tmp_path / 'xgb.json'
    enc_path = tmp_path / 'enc.json'
    meta_path = tmp_path / 'meta.json'

    # write dummy json file for xgboost model
    with open(model_path, 'w') as f:
        f.write("{}")

    _write_json(enc_path, {})
    _write_json(meta_path, {'features': []})

    p = AMTPredictor(
        backend='xgb',
        model_path=str(model_path),
        encoders_path=str(enc_path),
        meta_path=str(meta_path),
        confidence_threshold=0.6,
    )
    # mock predict_proba on the backend model
    p.model.predict_proba = MagicMock(return_value=[[0.3, 0.7]])

    out = p.should_trade({'direction': 'LONG'})
    assert out['action'] == 'BUY'
    assert out['confidence'] >= 0.6



def test_predictor_mlp_backend(tmp_path):
    # just skip this if not properly installed
    import sys
    if 'torch' in sys.modules and getattr(sys.modules['torch'], '__name__', '') == 'unittest.mock':
        pytest.skip('torch is mocked')
    if type(torch).__name__ == 'MagicMock':
        pytest.skip('torch is mocked')

    model_path = tmp_path / 'mlp.pt'
    scaler_path = tmp_path / 'scaler.json'
    enc_path = tmp_path / 'enc.json'
    meta_path = tmp_path / 'meta.json'

    input_dim = 4
    features = ['distance_to_poc_pct', 'volume_zscore', 'delta_zscore', 'cvd_slope_short']

    model = _TabularMLP(input_dim=input_dim, hidden_dims=(8, 4), dropout=0.0)
    # Force a reasonably positive output via last-layer bias
    with torch.no_grad():
        last_linear = model.net[-1]
        last_linear.bias.fill_(1.0)

    torch.save(
        {
            'state_dict': model.state_dict(),
            'input_dim': input_dim,
            'hidden_dims': [8, 4],
            'dropout': 0.0,
        },
        model_path,
    )

    # Identity-ish scaler
    _write_json(scaler_path, {'mean': [0.0] * input_dim, 'scale': [1.0] * input_dim})
    _write_json(enc_path, {})
    _write_json(meta_path, {'features': features})

    p = AMTPredictor(
        backend='mlp',
        model_path=str(model_path),
        scaler_path=str(scaler_path),
        encoders_path=str(enc_path),
        meta_path=str(meta_path),
        confidence_threshold=0.5,
    )

    signal = {
        'direction': 'SHORT',
        'distance_to_poc_pct': 0.1,
        'volume_zscore': 0.0,
        'delta_zscore': 0.0,
        'cvd_slope_short': 0.0,
    }
    out = p.should_trade(signal)
    assert out['action'] in {'SELL', 'SKIP'}
    assert 0.0 <= out['confidence'] <= 1.0
