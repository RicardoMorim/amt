"""
ml/trainer.py
=============
Trains an XGBoost classifier on AMT signals.

B3: Optuna hyperparameter tuning (Walk-Forward, 50 trials by default).
    Expanding-window walk-forward replaces TimeSeriesSplit for realistic
    temporal splits — training window grows over time while validation
    is a fixed forward-looking period.
B4: Temporal + regime features (see dataset_builder.py).
B1+B2: LabelEncoders serialised with model.

Usage:
    python ml/trainer.py
    python ml/trainer.py path/to/amt_ml_dataset.db
"""

import os
import sys
import json
import joblib
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.dataset_builder import get_xy, FEATURES
import config

MODEL_PATH    = config.ML_MODEL_PATH
META_PATH     = config.ML_META_PATH
ENCODERS_PATH = config.ML_ENCODERS_PATH
OPTUNA_TRIALS = config.ML_OPTUNA_TRIALS

# ── Walk-Forward Validation Configuration ──────────────────────────────────────
# Expanding-window walk-forward: training window grows, validation is fixed-size.
# This mimics real trading where you retrain on all available history and validate
# on the next N bars before rolling forward.
WF_TRAIN_FRACTION = 0.70   # first 70% of data for initial training
WF_VAL_SIZE       = 50000  # fixed validation window size (rows)
WF_STEP           = 25000  # how many rows to advance the validation window each step


def _walk_forward_splits(n_samples, train_fraction, val_size, step):
    """Generate walk-forward split indices.

    Expanding-window approach:
      - Initial training window = n_samples * train_fraction
      - Validation window = fixed size (val_size)
      - Each step advances both windows by `step` rows
      - Training window grows as we move forward in time

    Yields (train_idx, val_idx) tuples.
    """
    train_start = 0
    train_end = int(n_samples * train_fraction)

    while True:
        val_start = train_end
        val_end = min(val_start + val_size, n_samples)

        if val_end - val_start < 1000:  # skip tiny validation windows
            break

        yield np.arange(train_start, train_end), np.arange(val_start, val_end)

        # Advance — expand training window and move validation forward
        train_start += step
        train_end = min(train_start + (val_end - val_start), n_samples)


def _objective(trial, X, y):
    """Optuna objective: mean AUC across walk-forward folds."""
    params = {
        'n_estimators':      trial.suggest_int('n_estimators', 100, 600),
        'max_depth':         trial.suggest_int('max_depth', 3, 8),
        'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample':         trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight':  trial.suggest_int('min_child_weight', 5, 50),
        'gamma':             trial.suggest_float('gamma', 0, 5),
        'reg_alpha':         trial.suggest_float('reg_alpha', 0, 2),
        'reg_lambda':        trial.suggest_float('reg_lambda', 0.5, 5),
        'scale_pos_weight':  (y == 0).sum() / max((y == 1).sum(), 1),
        'eval_metric':       'logloss',
        'random_state':      42,
        'verbosity':         0,
    }

    aucs = []
    for tr_idx, val_idx in _walk_forward_splits(len(X), WF_TRAIN_FRACTION, WF_VAL_SIZE, WF_STEP):
        m = xgb.XGBClassifier(**params)
        m.fit(
            X.iloc[tr_idx], y.iloc[tr_idx],
            eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
            verbose=False,
        )
        aucs.append(roc_auc_score(y.iloc[val_idx], m.predict_proba(X.iloc[val_idx])[:, 1]))

    return float(np.mean(aucs)) if aucs else 0.5


def train(db_path: str = config.DB_PATH):
    print("=" * 60)
    print("🤖 AMT ML Trainer")
    print("=" * 60)

    X, y, _, encoders = get_xy(db_path)

    if len(X) < 500:
        print("⚠️  Warning: fewer than 500 samples — model may be unreliable.")

    # ── B3: Optuna tuning with walk-forward validation ────────────────────────
    print(f"\n🔍 Optuna hyperparameter search ({OPTUNA_TRIALS} trials, walk-forward)...")
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='maximize')
        study.optimize(
            lambda trial: _objective(trial, X, y),
            n_trials=OPTUNA_TRIALS,
            show_progress_bar=True,
        )
        best_params = study.best_params
        print(f"   Best AUC: {study.best_value:.4f}")
        print(f"   Best params: {best_params}")
    except ImportError:
        print("   optuna not installed — using default params. Run: pip install optuna")
        best_params = {
            'n_estimators': 400, 'max_depth': 5, 'learning_rate': 0.04,
            'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 10,
        }

    # ── Walk-Forward Cross-Validation with best params ────────────────────────
    wf_aucs = []
    wf_val_sizes = []
    print(f"\n📊 Walk-forward validation (train={WF_TRAIN_FRACTION}, val_size={WF_VAL_SIZE}, step={WF_STEP})...")

    for fold, (tr_idx, val_idx) in enumerate(_walk_forward_splits(len(X), WF_TRAIN_FRACTION, WF_VAL_SIZE, WF_STEP), 1):
        m = xgb.XGBClassifier(
            **best_params,
            scale_pos_weight=(y.iloc[tr_idx] == 0).sum() / max((y.iloc[tr_idx] == 1).sum(), 1),
            eval_metric='logloss',
            random_state=42,
            verbosity=0,
        )
        m.fit(
            X.iloc[tr_idx], y.iloc[tr_idx],
            eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
            verbose=False,
        )
        auc = roc_auc_score(y.iloc[val_idx], m.predict_proba(X.iloc[val_idx])[:, 1])
        wf_aucs.append(auc)
        wf_val_sizes.append(len(val_idx))
        print(f"   Fold {fold}: AUC = {auc:.4f} | train={len(tr_idx):,} val={len(val_idx):,}")

    print(f"\n   Mean AUC = {np.mean(wf_aucs):.4f} ± {np.std(wf_aucs):.4f}")
    print(f"   Total folds: {len(wf_aucs)}")

    # ── Final model on all data ────────────────────────────────────────────────
    print("\n🏋️  Training final model on full dataset...")
    final_model = xgb.XGBClassifier(
        **best_params,
        scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1),
        eval_metric='logloss',
        random_state=42,
        verbosity=0,
    )
    final_model.fit(X, y)

    # ── Last fold report ───────────────────────────────────────────────────────
    _, last_val_idx = list(_walk_forward_splits(len(X), WF_TRAIN_FRACTION, WF_VAL_SIZE, WF_STEP))[-1]
    y_pred = final_model.predict(X.iloc[last_val_idx])
    print("\n📋 Classification report (last walk-forward fold):")
    print(classification_report(y.iloc[last_val_idx], y_pred, target_names=['SKIP', 'TRADE']))

    # ── Feature importance ────────────────────────────────────────────────────
    print("🔍 Feature importances:")
    importances = dict(zip(FEATURES, final_model.feature_importances_))
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
        print(f"   {feat:<30} {'█' * int(imp * 40)} {imp:.4f}")

    # ── Save model + encoders + metadata ─────────────────────────────────────
    os.makedirs('ml', exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)
    joblib.dump(encoders,    ENCODERS_PATH)

    meta = {
        'features':        FEATURES,
        'best_params':     {k: v for k, v in best_params.items() if k not in ('eval_metric',)},
        'cv_mean_auc':     float(np.mean(wf_aucs)),
        'cv_std_auc':      float(np.std(wf_aucs)),
        'n_train_samples': int(len(X)),
        'n_positive':      int(y.sum()),
        'n_negative':      int((y == 0).sum()),
        'validation_method': 'walk_forward_expanding',
        'wf_train_fraction': WF_TRAIN_FRACTION,
        'wf_val_size':       WF_VAL_SIZE,
        'wf_step':           WF_STEP,
        'wf_n_folds':        len(wf_aucs),
    }
    with open(META_PATH, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ Model    → {MODEL_PATH}")
    print(f"✅ Encoders → {ENCODERS_PATH}")
    print(f"✅ Meta     → {META_PATH}")
    print("=" * 60)
    return final_model


if __name__ == '__main__':
    db = sys.argv[1] if len(sys.argv) > 1 else config.DB_PATH
    train(db)
