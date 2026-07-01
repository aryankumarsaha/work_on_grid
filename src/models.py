"""
Model module.
Defines the DayAheadLoadForecaster class, which handles training point and quantile regressors
for the 24-hour horizon, including hyperparameter tuning on a validation split.
"""

import time
import logging
import itertools
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Any
from lightgbm import LGBMRegressor, early_stopping
from src.config import VAL_START, TEST_START, HORIZON, QUANTILES, TUNING_GRID, TUNING_HORIZONS
from src.features import get_target_temporal_features

logger = logging.getLogger(__name__)


class DayAheadLoadForecaster:
    """
    Production wrapper class managing LightGBM point and quantile regressors.
    Provides automated hyperparameter tuning and early stopping on validation data.
    """
    def __init__(self, best_params: Dict[str, Any] = None):
        # Default hyperparameters if tuning is skipped
        self.params = best_params or {
            'learning_rate': 0.05,
            'max_depth': 6,
            'num_leaves': 31,
            'colsample_bytree': 0.8,
            'subsample': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': -1
        }
        
        # We hold separate dicts of models for point (mean) and quantiles (0.05, 0.95)
        self.point_models: Dict[int, LGBMRegressor] = {}
        self.quantile_models: Dict[float, Dict[int, LGBMRegressor]] = {q: {} for q in QUANTILES}
        
        # Feature columns list will be populated dynamically from training X
        self.feature_cols: List[str] = []

    def tune_hyperparameters(self, X: pd.DataFrame, Y: pd.DataFrame):
        """
        Runs hyperparameter optimization using Optuna and TimeSeriesSplit.
        Evaluates on a middle horizon (e.g., h=12) to optimize efficiency.
        """
        import optuna
        from sklearn.model_selection import TimeSeriesSplit
        
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        logger.info("Starting hyperparameter optimization using Optuna + TimeSeriesSplit...")
        
        train_val_mask = X.index < TEST_START
        X_tune = X.loc[train_val_mask]
        Y_tune = Y.loc[train_val_mask]
        
        def objective(trial):
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.1),
                'max_depth': trial.suggest_int('max_depth', 4, 8),
                'num_leaves': trial.suggest_int('num_leaves', 15, 63),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
                'subsample': trial.suggest_float('subsample', 0.6, 0.9),
                'random_state': 42,
                'n_jobs': -1,
                'verbosity': -1
            }
            
            # Using TimeSeriesSplit for chronologically ordered cross-validation splits
            tscv = TimeSeriesSplit(n_splits=3)
            val_scores = []
            
            h = 12
            y_tune_h = Y_tune[f"target_h{h}"]
            
            for train_idx, val_idx in tscv.split(X_tune):
                X_tr, X_va = X_tune.iloc[train_idx], X_tune.iloc[val_idx]
                y_tr, y_va = y_tune_h.iloc[train_idx], y_tune_h.iloc[val_idx]
                
                # Get temporal target features
                X_tr_cal = get_target_temporal_features(X_tr.index, h)
                X_va_cal = get_target_temporal_features(X_va.index, h)
                
                X_tr_full = pd.concat([X_tr, X_tr_cal], axis=1)
                X_va_full = pd.concat([X_va, X_va_cal], axis=1)
                
                model = LGBMRegressor(n_estimators=300, **params)
                model.fit(
                    X_tr_full, y_tr,
                    eval_set=[(X_va_full, y_va)],
                    callbacks=[early_stopping(stopping_rounds=30, verbose=False)]
                )
                
                preds = model.predict(X_va_full)
                val_scores.append(np.mean(np.abs(y_va - preds)))
                
            return float(np.mean(val_scores))
            
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=10, timeout=120)
        
        best_params = study.best_params
        self.params = {
            'learning_rate': best_params['learning_rate'],
            'max_depth': best_params['max_depth'],
            'num_leaves': best_params['num_leaves'],
            'colsample_bytree': best_params['colsample_bytree'],
            'subsample': best_params['subsample'],
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': -1
        }
        logger.info(f"Optuna Optimization complete. Best params: {self.params} (Best Score: {study.best_value:.4f})")

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame, tune: bool = True):
        """
        Trains point regressors and quantile regressors for all 24 horizons.
        Uses early stopping on the validation set.
        """
        if tune:
            self.tune_hyperparameters(X, Y)
            
        self.feature_cols = list(X.columns)
        
        train_mask = X.index < VAL_START
        val_mask = (X.index >= VAL_START) & (X.index < TEST_START)
        
        X_train_base = X.loc[train_mask]
        X_val_base = X.loc[val_mask]
        
        logger.info("Training point and interval forecast models for all 24 horizons...")
        start_time = time.time()
        
        for h in range(1, HORIZON + 1):
            y_train = Y.loc[train_mask, f"target_h{h}"]
            y_val = Y.loc[val_mask, f"target_h{h}"]
            
            # Construct target calendar features
            X_train_cal = get_target_temporal_features(X_train_base.index, h)
            X_val_cal = get_target_temporal_features(X_val_base.index, h)
            
            X_train = pd.concat([X_train_base, X_train_cal], axis=1)
            X_val = pd.concat([X_val_base, X_val_cal], axis=1)
            
            # 1. Point forecast model
            point_model = LGBMRegressor(
                n_estimators=1000,
                **self.params
            )
            point_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[early_stopping(stopping_rounds=50, verbose=False)]
            )
            self.point_models[h] = point_model
            
            # 2. Quantile forecast models (0.05 and 0.95 bounds)
            for q in QUANTILES:
                q_model = LGBMRegressor(
                    n_estimators=1000,
                    objective='quantile',
                    alpha=q,
                    **self.params
                )
                q_model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[early_stopping(stopping_rounds=50, verbose=False)]
                )
                self.quantile_models[q][h] = q_model
                
            logger.info(f"Trained Point & Quantile models for Horizon +{h}h (Best Point Iter: {point_model.best_iteration_})")
            
        logger.info(f"All models trained successfully in {time.time() - start_time:.2f} seconds.")

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates point predictions and lower/upper quantile predictions.
        Returns:
          point_preds: shape (N, 24)
          lower_preds: shape (N, 24)
          upper_preds: shape (N, 24)
        """
        N = len(X)
        point_preds = np.zeros((N, HORIZON))
        lower_preds = np.zeros((N, HORIZON))
        upper_preds = np.zeros((N, HORIZON))
        
        X_base = X[self.feature_cols]
        
        for h in range(1, HORIZON + 1):
            X_cal = get_target_temporal_features(X_base.index, h)
            X_full = pd.concat([X_base, X_cal], axis=1)
            
            point_preds[:, h - 1] = self.point_models[h].predict(X_full)
            lower_preds[:, h - 1] = self.quantile_models[0.05][h].predict(X_full)
            upper_preds[:, h - 1] = self.quantile_models[0.95][h].predict(X_full)
            
        # Guarantee that lower bound <= upper bound
        lower_preds = np.minimum(lower_preds, point_preds)
        upper_preds = np.maximum(upper_preds, point_preds)
        
        return point_preds, lower_preds, upper_preds
