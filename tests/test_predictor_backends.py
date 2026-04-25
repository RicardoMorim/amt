import json
import joblib
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


def test_predictor_xgb_backend(tmp_path):
    model_path = tmp_path / 'xgb.pkl'
    enc_path = tmp_path / 'enc.pkl'
    meta_path = tmp_path / 'meta.json'

    joblib.dump(DummyXGBModel(), model_path)
    joblib.dump({}, enc_path)
    _write_json(meta_path, {'features': []})

    p = AMTPredictor(
        backend='xgb',
        model_path=str(model_path),
        encoders_path=str(enc_path),
        meta_path=str(meta_path),
        confidence_threshold=0.6,
    )

    out = p.should_trade({'direction': 'LONG'})
    assert out['action'] == 'BUY'
    assert out['confidence'] >= 0.6


def test_predictor_mlp_backend(tmp_path):
    model_path = tmp_path / 'mlp.pt'
    scaler_path = tmp_path / 'scaler.pkl'
    enc_path = tmp_path / 'enc.pkl'
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
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(np.zeros((10, input_dim)))
    joblib.dump(scaler, scaler_path)
    joblib.dump({}, enc_path)
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
