"""
ml/predictor.py
===============
Live inference — plugs into main.py AMTSession.

Usage in main.py:
    from ml.predictor import AMTPredictor
    predictor = AMTPredictor()
    decision  = predictor.should_trade(signal_dict)
    if decision['action'] != 'SKIP':
        send_alert(signal_dict, decision)
"""

import json
import joblib
import numpy as np
import pandas as pd

MODEL_PATH = "ml/amt_model.pkl"
META_PATH  = "ml/amt_model_meta.json"

# Encode maps — must match training encoding order
_SIGNAL_TYPE_MAP   = {}   # auto-populated from DB categories at train time
_DIRECTION_MAP     = {"buy": 0, "sell": 1}
_SESSION_STATE_MAP = {}   # auto-populated


class AMTPredictor:
    """
    Wraps the trained XGBoost model for live signal scoring.
    
    Example:
        predictor = AMTPredictor()
        result = predictor.should_trade({
            'signal_type':       'balance_breakout',
            'direction':         'buy',
            'session_state':     'balance',
            'is_composite':      True,
            'timeframe_secs':    900,
            'distance_to_poc_pct': 0.012,
            'volume_zscore':     1.8,
            'delta_zscore':      2.1,
            'cvd_slope_short':   0.003,
            'cvd_slope_long':   -0.001,
        })
        # → {'action': 'BUY', 'confidence': 0.73, 'skip_reason': None}
    """

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        confidence_threshold: float = 0.60,
    ):
        self.model      = joblib.load(model_path)
        self.threshold  = confidence_threshold

        try:
            with open(META_PATH) as f:
                self.meta = json.load(f)
            self.features = self.meta["features"]
        except FileNotFoundError:
            from dataset_builder import FEATURES
            self.features = FEATURES
            self.meta = {}

        # Category encoding caches (populated lazily)
        self._sig_enc   = {}
        self._sess_enc  = {}

    # ── Public API ─────────────────────────────────────────────────────────────

    def should_trade(self, signal: dict) -> dict:
        """
        Score a live AMT signal.
        Returns dict with keys: action, confidence, skip_reason
        """
        try:
            features = self._build_feature_vector(signal)
            proba    = self.model.predict_proba([features])[0]
            conf_pos = float(proba[1])   # probability of TRADE class

            if conf_pos < self.threshold:
                return {
                    "action":      "SKIP",
                    "confidence":  conf_pos,
                    "skip_reason": f"confidence {conf_pos:.2%} < threshold {self.threshold:.0%}",
                }

            direction = signal.get("direction", "buy").lower()
            action    = "BUY" if direction == "buy" else "SELL"

            return {
                "action":      action,
                "confidence":  conf_pos,
                "skip_reason": None,
            }

        except Exception as e:
            return {"action": "SKIP", "confidence": 0.0, "skip_reason": f"predictor error: {e}"}

    def score_batch(self, signals: list[dict]) -> list[dict]:
        """Score a list of signals at once (faster for backtesting)."""
        return [self.should_trade(s) for s in signals]

    # ── Internal ───────────────────────────────────────────────────────────────

    def _build_feature_vector(self, s: dict) -> list:
        """Map a raw signal dict to the feature vector expected by the model."""

        def cat_encode(cache: dict, value: str) -> int:
            """Encode unseen categories as -1 (model handles gracefully)."""
            v = str(value).lower()
            if v not in cache:
                cache[v] = len(cache)
            return cache[v]

        direction    = str(s.get("direction",     "buy")).lower()
        direction_enc = 0 if direction == "buy" else 1

        return [
            float(s.get("distance_to_poc_pct",  0.0) or 0.0),
            float(s.get("volume_zscore",         0.0) or 0.0),
            float(s.get("delta_zscore",          0.0) or 0.0),
            float(s.get("cvd_slope_short",       0.0) or 0.0),
            float(s.get("cvd_slope_long",        0.0) or 0.0),
            cat_encode(self._sig_enc,  s.get("signal_type",   "unknown")),
            direction_enc,
            cat_encode(self._sess_enc, s.get("session_state", "unknown")),
            int(bool(s.get("is_composite", False))),
            int(s.get("timeframe_secs", 900)),
        ]
