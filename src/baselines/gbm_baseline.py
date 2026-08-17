"""
Tuned LightGBM Baseline for the Sri Lanka Flood DNN.

This replaces the default-param GBMBaseline with separate tuned classifiers
for binary targets and tuned regressors for continuous targets, using
Optuna to search the hyperparameter space.

Usage:
    python src/baselines/gbm_baseline.py \
        --data_config configs/kaggle_data.yaml \
        --output_dir experiments/gbm_baseline \
        --n_trials 50

Results are written to:
    <output_dir>/gbm_metrics.md   — evaluation table
    <output_dir>/gbm_models.pkl   — serialised model objects
"""

import os
import sys
import argparse
import pickle
import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, roc_auc_score, brier_score_loss,
    r2_score, mean_absolute_error, mean_squared_error
)
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Make src imports work when run standalone
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ---------------------------------------------------------------------------
# Target metadata
# ---------------------------------------------------------------------------
CLS_TARGETS = [
    'target_flood_1d', 'target_flood_2d',
    'target_flood_3d', 'target_onset_1d'
]
REG_TARGETS = [
    'target_next1d_discharge', 'target_next3d_max_zscore'
]
ALL_TARGETS = CLS_TARGETS + REG_TARGETS

EXCLUDE_COLS = set(
    ALL_TARGETS + [
        'date', 'node_id', 'basin', 'zone', 'position',
        'split_temporal', 'split_basin_holdout', 'valid_sample',
        'event_id', 'flood_moderate', 'flood_high', 'flood_severe',
        'flood_state', 'label_confidence', 'thr_moderate',
        'thr_high', 'thr_severe'
    ]
)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_flat_split(panel_path: str, split: str):
    """Load the panel, filter to valid_sample and a temporal split."""
    df = pd.read_parquet(panel_path)
    if 'valid_sample' in df.columns:
        df = df[df['valid_sample'] == True]
    if 'split_temporal' in df.columns:
        df = df[df['split_temporal'] == split]
    return df


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in EXCLUDE_COLS][:33]


# ---------------------------------------------------------------------------
# Optuna objectives
# ---------------------------------------------------------------------------
def _cls_objective(trial, X_train, y_train, X_val, y_val):
    params = {
        'objective':        'binary',
        'metric':           'average_precision',
        'verbosity':        -1,
        'boosting_type':    'gbdt',
        'num_leaves':       trial.suggest_int('num_leaves', 15, 127),
        'max_depth':        trial.suggest_int('max_depth', 3, 12),
        'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators':     trial.suggest_int('n_estimators', 100, 1000),
        'min_child_samples':trial.suggest_int('min_child_samples', 5, 100),
        'subsample':        trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha':        trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda':       trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        # Handle class imbalance
        'is_unbalance':     True,
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(period=-1)]
    )
    preds = model.predict_proba(X_val)[:, 1]
    try:
        return average_precision_score(y_val, preds)
    except ValueError:
        return 0.0


def _reg_objective(trial, X_train, y_train, X_val, y_val):
    params = {
        'objective':        'regression_l1',  # MAE — more robust to outliers
        'metric':           'mae',
        'verbosity':        -1,
        'boosting_type':    'gbdt',
        'num_leaves':       trial.suggest_int('num_leaves', 15, 127),
        'max_depth':        trial.suggest_int('max_depth', 3, 12),
        'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators':     trial.suggest_int('n_estimators', 100, 1000),
        'min_child_samples':trial.suggest_int('min_child_samples', 5, 100),
        'subsample':        trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha':        trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda':       trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(period=-1)]
    )
    preds = model.predict(X_val)
    return -mean_absolute_error(y_val, preds)   # maximise → minimise MAE


# ---------------------------------------------------------------------------
# Tuned baseline class
# ---------------------------------------------------------------------------
class TunedGBMBaseline:
    """
    Per-target tuned LightGBM baseline.
    - LGBMClassifier (with is_unbalance=True) for the 4 binary targets.
    - LGBMRegressor  (L1 objective)             for the 2 continuous targets.
    """

    def __init__(self, n_trials: int = 50, seed: int = 42):
        self.n_trials = n_trials
        self.seed = seed
        self.cls_models:  dict[str, lgb.LGBMClassifier]  = {}
        self.reg_models:  dict[str, lgb.LGBMRegressor]   = {}
        self.best_params: dict[str, dict]                 = {}

    def fit(self, X_train, y_train_df, X_val, y_val_df):
        """
        Parameters
        ----------
        X_train, X_val : np.ndarray [N, features]
        y_train_df, y_val_df : pd.DataFrame with ALL_TARGETS columns
        """
        # ── Classification targets ────────────────────────────────────────
        for target in CLS_TARGETS:
            print(f"  Tuning GBM classifier for: {target}")
            y_tr = y_train_df[target].values
            y_vl = y_val_df[target].values

            study = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=self.seed)
            )
            study.optimize(
                lambda trial: _cls_objective(trial, X_train, y_tr, X_val, y_vl),
                n_trials=self.n_trials,
                gc_after_trial=True,
                show_progress_bar=False,
            )
            best = study.best_params
            self.best_params[target] = best

            # Refit on train+val with best params
            final_model = lgb.LGBMClassifier(
                **best, is_unbalance=True, verbosity=-1, random_state=self.seed
            )
            final_model.fit(
                np.vstack([X_train, X_val]),
                np.concatenate([y_tr, y_vl])
            )
            self.cls_models[target] = final_model
            print(f"    ✓ PR-AUC on val: {study.best_value:.4f}")

        # ── Regression targets ────────────────────────────────────────────
        for target in REG_TARGETS:
            print(f"  Tuning GBM regressor for: {target}")
            y_tr = y_train_df[target].values
            y_vl = y_val_df[target].values

            study = optuna.create_study(
                direction='maximize',  # we maximise negative MAE
                sampler=optuna.samplers.TPESampler(seed=self.seed)
            )
            study.optimize(
                lambda trial: _reg_objective(trial, X_train, y_tr, X_val, y_vl),
                n_trials=self.n_trials,
                gc_after_trial=True,
                show_progress_bar=False,
            )
            best = study.best_params
            self.best_params[target] = best

            final_model = lgb.LGBMRegressor(
                **best, verbosity=-1, random_state=self.seed
            )
            final_model.fit(
                np.vstack([X_train, X_val]),
                np.concatenate([y_tr, y_vl])
            )
            self.reg_models[target] = final_model
            print(f"    ✓ MAE on val: {-study.best_value:.4f}")

    def predict_proba(self, X: np.ndarray) -> dict[str, np.ndarray]:
        return {t: m.predict_proba(X)[:, 1] for t, m in self.cls_models.items()}

    def predict_reg(self, X: np.ndarray) -> dict[str, np.ndarray]:
        return {t: m.predict(X) for t, m in self.reg_models.items()}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_gbm(model: TunedGBMBaseline, X_val, y_val_df) -> str:
    cls_probs = model.predict_proba(X_val)
    reg_preds = model.predict_reg(X_val)

    lines = ["# Tuned GBM Baseline — Evaluation Results\n"]
    lines.append("## Classification (threshold-free metrics)\n")
    lines.append("| Target | PR-AUC | ROC-AUC | Brier |")
    lines.append("|---|---|---|---|")
    for t in CLS_TARGETS:
        y = y_val_df[t].values.astype(int)
        p = cls_probs[t]
        try:
            pr  = average_precision_score(y, p)
            roc = roc_auc_score(y, p)
        except ValueError:
            pr = roc = float('nan')
        brier = brier_score_loss(y, p)
        lines.append(f"| {t} | {pr:.4f} | {roc:.4f} | {brier:.5f} |")

    lines.append("\n## Regression\n")
    lines.append("| Target | R² | MAE | RMSE |")
    lines.append("|---|---|---|---|")
    for t in REG_TARGETS:
        y = y_val_df[t].values
        p = reg_preds[t]
        r2   = r2_score(y, p)
        mae  = mean_absolute_error(y, p)
        rmse = np.sqrt(mean_squared_error(y, p))
        lines.append(f"| {t} | {r2:.4f} | {mae:.4f} | {rmse:.4f} |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Tuned GBM Baseline")
    parser.add_argument('--data_config',  type=str, default='configs/kaggle_data.yaml')
    parser.add_argument('--output_dir',   type=str, default='experiments/gbm_baseline')
    parser.add_argument('--n_trials',     type=int, default=50)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    data_cfg = load_config(args.data_config)
    panel    = data_cfg['data_paths']['panel']

    print("Loading data...")
    train_df = load_flat_split(panel, 'train')
    val_df   = load_flat_split(panel, 'val')

    feat_cols = get_feature_cols(train_df)
    X_train = train_df[feat_cols].values.astype(np.float32)
    X_val   = val_df[feat_cols].values.astype(np.float32)

    print(f"Train: {X_train.shape}  Val: {X_val.shape}")

    # GBM works on flat (non-normalised) data by design — tree models are
    # invariant to monotonic feature transforms, so no StandardScaler needed.

    baseline = TunedGBMBaseline(n_trials=args.n_trials)
    print(f"\nStarting HPO ({args.n_trials} trials per target × {len(ALL_TARGETS)} targets)...")
    baseline.fit(X_train, train_df[ALL_TARGETS], X_val, val_df[ALL_TARGETS])

    # Evaluate
    report = evaluate_gbm(baseline, X_val, val_df[ALL_TARGETS])
    print("\n" + report)

    # Save
    report_path = os.path.join(args.output_dir, 'gbm_metrics.md')
    model_path  = os.path.join(args.output_dir, 'gbm_models.pkl')
    with open(report_path, 'w') as f:
        f.write(report)
    with open(model_path, 'wb') as f:
        pickle.dump(baseline, f)

    print(f"\nReport saved → {report_path}")
    print(f"Models saved → {model_path}")


if __name__ == '__main__':
    main()
