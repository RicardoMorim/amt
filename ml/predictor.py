"""
ml/predictor.py
===============
Live inference — plugs into AMTSession in main.py.

B2 fix:
  Loads the LabelEncoders serialised by trainer.py so categorical
  encoding at inference time is guaranteed to match training.
  Unseen categories are mapped to -1 (XGBoost handles gracefully).

Usage:
    from ml.predictor import AMTPredictor
    predictor = AMTPredictor()
    decision  = predictor.should_trade(signal_dict)
    # → {'action': 'BUY', 'confidence': 0.73, 'skip_reason': None}
"""

from __future__ import annotations

import sys
import json
import os
import joblib
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


class _TabularMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, int] = (64, 32), dropout: float = 0.20):
        super().__init__()
        h1, h2 = hidden_dims
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class AMTPredictor:

    def _init_xgb_backend(self, model_path: str):
        self.model = joblib.load(model_path)
        self.scaler = None
        self.device = None

    def _init_mlp_backend(self, model_path: str, scaler_path: str | None):
        payload = torch.load(model_path, map_location='cpu')
        input_dim = int(payload['input_dim'])
        hidden_dims = tuple(payload.get('hidden_dims', [64, 32]))
        dropout = float(payload.get('dropout', 0.20))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _TabularMLP(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout).to(self.device)
        self.model.load_state_dict(payload['state_dict'])
        self.model.eval()
        self.scaler = joblib.load(scaler_path) if scaler_path and os.path.exists(scaler_path) else None

    def __init__(
        self,
        model_path: str | None = None,
        encoders_path: str | None = None,
        meta_path: str | None = None,
        scaler_path: str | None = None,
        backend: str = config.ML_BACKEND_DEFAULT,
        confidence_threshold: float = 0.60,
    ):
        self.backend = str(backend).lower().strip()
        if self.backend not in {"xgb", "mlp"}:
            raise ValueError(f"Unsupported backend='{backend}'. Use 'xgb' or 'mlp'.")

        if self.backend == "xgb":
            model_path = model_path or config.ML_MODEL_PATH
            encoders_path = encoders_path or config.ML_ENCODERS_PATH
            meta_path = meta_path or config.ML_META_PATH
            self._init_xgb_backend(model_path)
        else:
            model_path = model_path or config.ML_MLP_MODEL_PATH
            encoders_path = encoders_path or config.ML_MLP_ENCODERS_PATH
            meta_path = meta_path or config.ML_MLP_META_PATH
            scaler_path = scaler_path or config.ML_MLP_SCALER_PATH
            self._init_mlp_backend(model_path, scaler_path)

        self.threshold = confidence_threshold

        # B2 — load serialised encoders (same mapping as training)
        if encoders_path and os.path.exists(encoders_path):
            self.encoders = joblib.load(encoders_path)
        else:
            self.encoders = {}

        self.meta_path = meta_path
        try:
            with open(meta_path) as f:
                self.meta = json.load(f)
            self.features = self.meta.get('features', [])
        except FileNotFoundError:
            self.features = []
            self.meta = {}

    # ── Public API ─────────────────────────────────────────────────────────────

    def should_trade(self, signal: dict) -> dict:
        try:
            features = self._build_feature_vector(signal)
            conf_pos = float(self._predict_positive_proba(features))

            if conf_pos < self.threshold:
                return {
                    'action':      'SKIP',
                    'confidence':  conf_pos,
                    'skip_reason': f"confidence {conf_pos:.2%} < threshold {self.threshold:.0%}",
                }

            direction = signal.get('direction', 'LONG').upper()
            action    = 'BUY' if direction in ('LONG', 'BUY') else 'SELL'
            return {'action': action, 'confidence': conf_pos, 'skip_reason': None}

        except Exception as e:
            return {'action': 'SKIP', 'confidence': 0.0, 'skip_reason': f'predictor error: {e}'}

    def score_batch(self, signals: list) -> list:
        return [self.should_trade(s) for s in signals]

    def _predict_positive_proba(self, features: list[float]) -> float:
        if self.backend == 'xgb':
            proba = self.model.predict_proba([features])[0]
            return float(proba[1])

        x = np.array([features], dtype=np.float32)
        if self.scaler is not None:
            x = self.scaler.transform(x)
        xt = torch.tensor(x, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            p = torch.sigmoid(self.model(xt)).detach().cpu().numpy()[0]
        return float(p)

    # ── Internal ───────────────────────────────────────────────────────────────

    def _encode_cat(self, col: str, value: str) -> int:
        """Encode using saved LabelEncoder; unknown values → -1."""
        le = self.encoders.get(col)
        if le is None:
            return -1
        v = str(value)
        if v in set(le.classes_):
            return int(le.transform([v])[0])
        return -1

    def _build_feature_vector(self, s: dict) -> list:
        direction = str(s.get('direction', 'LONG')).upper()
        # direction_enc: match training (LabelEncoder on 'LONG'/'SHORT'/'CONFLICT'...)
        direction_enc = self._encode_cat('direction', direction)

        # Temporal features from candle_time if available
        hour_utc    = 0
        day_of_week = 0
        try:
            from datetime import datetime
            ct = s.get('candle_time') or s.get('timestamp_event', '')
            if ct:
                dt = datetime.fromisoformat(ct.replace('Z', ''))
                hour_utc    = dt.hour
                day_of_week = dt.weekday()
        except Exception:
            pass

        market_state = s.get('market_state', s.get('session_state', 'unknown'))
        signal_type = str(s.get('signal_type', 'unknown')).upper()

        # FASE 1 structural features (safe defaults for live mode)
        false_breakout_flag = 1 if signal_type == 'FALSE_BREAKOUT' else 0
        price_vs_poc_norm = float(s.get('price_vs_poc_norm', s.get('distance_to_poc_pct', 0.0)) or 0.0)
        price_vs_value_area = float(s.get('price_vs_value_area', 0.0) or 0.0)
        atr_20_norm = float(s.get('atr_20_norm', 0.0) or 0.0)
        dist_to_swing_high_20 = float(s.get('dist_to_swing_high_20', 0.0) or 0.0)
        dist_to_swing_low_20 = float(s.get('dist_to_swing_low_20', 0.0) or 0.0)

        base = {
            'distance_to_poc_pct': float(s.get('distance_to_poc_pct', 0.0) or 0.0),
            'volume_zscore': float(s.get('volume_zscore', 0.0) or 0.0),
            'delta_zscore': float(s.get('delta_zscore', 0.0) or 0.0),
            'cvd_slope_short': float(s.get('cvd_slope_short', 0.0) or 0.0),
            'cvd_slope_long': float(s.get('cvd_slope_long', 0.0) or 0.0),
            'signal_type_enc': self._encode_cat('signal_type', s.get('signal_type', 'unknown')),
            'direction_enc': direction_enc,
            'session_state_enc': self._encode_cat('session_state', s.get('session_state', 'unknown')),
            'market_state_enc': self._encode_cat('market_state', market_state),
            'is_composite': int(bool(s.get('is_composite', False))),
            'timeframe_secs': int(s.get('timeframe_secs', 900)),
            'hour_utc': hour_utc,
            'day_of_week': day_of_week,
            'false_breakout_flag': false_breakout_flag,
            'price_vs_poc_norm': price_vs_poc_norm,
            'price_vs_value_area': price_vs_value_area,
            'atr_20_norm': atr_20_norm,
            'dist_to_swing_high_20': dist_to_swing_high_20,
            'dist_to_swing_low_20': dist_to_swing_low_20,
        }

        # If metadata has explicit feature order, follow it for compatibility.
        if self.features:
            return [float(base.get(feat, 0.0)) for feat in self.features]

        # Fallback to legacy order when metadata is missing.
        return [
            base['distance_to_poc_pct'],
            base['volume_zscore'],
            base['delta_zscore'],
            base['cvd_slope_short'],
            base['cvd_slope_long'],
            base['signal_type_enc'],
            base['direction_enc'],
            base['session_state_enc'],
            base['is_composite'],
            base['timeframe_secs'],
            base['hour_utc'],
            base['day_of_week'],
        ]
