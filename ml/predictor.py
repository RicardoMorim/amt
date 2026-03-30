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

import json
import os
import joblib
import numpy as np


class AMTPredictor:

    def __init__(
        self,
        model_path:    str   = "ml/amt_model.pkl",
        encoders_path: str   = "ml/amt_encoders.pkl",
        confidence_threshold: float = 0.60,
    ):
        self.model     = joblib.load(model_path)
        self.threshold = confidence_threshold

        # B2 — load serialised encoders (same mapping as training)
        if os.path.exists(encoders_path):
            self.encoders = joblib.load(encoders_path)
        else:
            self.encoders = {}

        meta_path = model_path.replace('.pkl', '_meta.json').replace('amt_model', 'amt_model_meta')
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
            proba    = self.model.predict_proba([features])[0]
            conf_pos = float(proba[1])

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

        return [
            float(s.get('distance_to_poc_pct',  0.0) or 0.0),
            float(s.get('volume_zscore',         0.0) or 0.0),
            float(s.get('delta_zscore',          0.0) or 0.0),
            float(s.get('cvd_slope_short',       0.0) or 0.0),
            float(s.get('cvd_slope_long',        0.0) or 0.0),
            self._encode_cat('signal_type',   s.get('signal_type',   'unknown')),
            direction_enc,
            self._encode_cat('session_state', s.get('session_state', 'unknown')),
            int(bool(s.get('is_composite', False))),
            int(s.get('timeframe_secs', 900)),
            hour_utc,
            day_of_week,
        ]
