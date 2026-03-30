"""
ml/trainer.py
=============
Trains an XGBoost classifier on AMT signals.

B3: Optuna hyperparameter tuning (TimeSeriesSplit, 50 trials by default).
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
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.dataset_builder import get_X_y, FEATURES
import config

MODEL_PATH    = config.ML_MODEL_PATH
META_PATH     = config.ML_META_PATH
ENCODERS_PATH = config.ML_ENCODERS_PATH
N_SPLITS      = config.ML_N_CV_SPLITS
OPTUNA_TRIALS = config.ML_OPTUNA_TRIALS


def _objective(trial, X, y, n_splits):
    """Optuna objective: mean AUC across TimeSeriesSplit folds."""
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
    tscv = TimeSeriesSplit(n_splits=n_splits)
    aucs = []
    for tr_idx, val_idx in tscv.split(X):
        m = xgb.XGBClassifier(**params)
        m.fit(
            X.iloc[tr_idx], y.iloc[tr_idx],
            eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
            verbose=False,
        )
        aucs.append(roc_auc_score(y.iloc[val_idx], m.predict_proba(X.iloc[val_idx])[:, 1]))
    return float(np.mean(aucs))


def train(db_path: str = "amt_ml_dataset.db"):
    print("=" * 60)
    print("🤖 AMT ML Trainer")
    print("=" * 60)

    X, y, df, encoders = get_X_y(db_path, ENCODERS_PATH)

    if len(X) < 500:
        print("⚠️  Warning: fewer than 500 samples — model may be unreliable.")

    # ── B3: Optuna tuning ─────────────────────────────────────────────────────
    print(f"\n🔍 Optuna hyperparameter search ({OPTUNA_TRIALS} trials)...")
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='maximize')
        study.optimize(
            lambda trial: _objective(trial, X, y, N_SPLITS),
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

    # ── Cross-validation with best params ─────────────────────────────────────
    tscv     = TimeSeriesSplit(n_splits=N_SPLITS)
    cv_aucs  = []
    print(f"\n📊 Cross-validation ({N_SPLITS}-fold TimeSeriesSplit)...")
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X), 1):
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
        cv_aucs.append(auc)
        print(f"   Fold {fold}: AUC = {auc:.4f} | val_size = {len(y.iloc[val_idx]):,}")

    print(f"\n   Mean AUC = {np.mean(cv_aucs):.4f} ± {np.std(cv_aucs):.4f}")

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
    _, val_idx = list(tscv.split(X))[-1]
    y_pred = final_model.predict(X.iloc[val_idx])
    print("\n📋 Classification report (last fold):")
    print(classification_report(y.iloc[val_idx], y_pred, target_names=['SKIP', 'TRADE']))

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
        'cv_mean_auc':     float(np.mean(cv_aucs)),
        'cv_std_auc':      float(np.std(cv_aucs)),
        'n_train_samples': int(len(X)),
        'n_positive':      int(y.sum()),
        'n_negative':      int((y == 0).sum()),
    }
    with open(META_PATH, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ Model    → {MODEL_PATH}")
    print(f"✅ Encoders → {ENCODERS_PATH}")
    print(f"✅ Meta     → {META_PATH}")
    print("=" * 60)
    return final_model


if __name__ == '__main__':
    db = sys.argv[1] if len(sys.argv) > 1 else "amt_ml_dataset.db"
    train(db)
