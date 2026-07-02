# python main.py
# uvicorn api.app:app --host 0.0.0.0 --port 8000

# streamlit run dashboard.py
# pytest tests/



"""
Day-Ahead Electricity Load Forecaster

================================================================================
ANSWERS TO ANALYTICAL QUESTIONS (SUBMISSION REQUIREMENT)
================================================================================
Q1: What would you change if you had to forecast for hundreds of thousands of 
    meters at once instead of one?
--------------------------------------------------------------------------------
1. Global Shared Forecasting Model: Training 24 separate models for each of 
   100,000+ smart meters is completely unscalable. Instead, we train a single 
   global forecasting model (e.g., global LightGBM or deep learning model) across 
   all meters. We feed in meter-specific static features (e.g. location, customer 
   sector, contract type, historical load stats) or meter embeddings to allow 
   the model to customize predictions.
2. Horizon as an Input Feature: Instead of 24 separate models, we pass the 
   horizon h in [1, 24] as a numerical/categorical input feature, reducing model 
   count to 1.
3. Sequence-to-Sequence (MIMO) Deep Learning: Use sequence-to-sequence neural 
   networks (e.g., Temporal Fusion Transformers, DeepAR, or N-BEATS) that process 
   the historical context window and directly output the 24-hour forecast array.
4. Distributed Compute: Leverage distributed processing frameworks like Spark or 
   Ray to distribute feature engineering, model training, and batch predictions.
5. Feature Store: Use a feature store (e.g., Feast) to precompute and serve lag 
   and rolling features efficiently.
6. Batch Inference: Set up batch prediction jobs that write predictions directly 
   to a fast database (e.g., TimescaleDB or Snowflake) rather than serving single-
   meter API endpoints.

Q2: Do you think a model like this is used in practice by utilities, or would 
    something simpler win?
--------------------------------------------------------------------------------
It depends entirely on the aggregation level:
- Single Household Level: A complex model is RARELY used. Household load is 
  highly stochastic, erratic, and noisy (driven by binary choices like turning on 
  an oven). Simple baselines (Standard Load Profiles, moving averages, or historical 
  persistence) perform similarly, costing virtually nothing to compute.
- Aggregated Grid Level (Substations, Feeder lines, ISO system load): Yes, advanced 
  models like LightGBM, XGBoost, and deep neural networks are the INDUSTRY STANDARD. 
  Individual household fluctuations cancel out, producing a smooth, predictable 
  load curve. At this scale, even a 0.5% forecast accuracy improvement saves 
  utilities millions of dollars in generation and dispatch costs, easily justifying 
  the model complexity.
================================================================================
"""

import os
import time
import json
import pickle
import logging
import argparse
from datetime import datetime
from typing import Tuple, Dict, List, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import holidays
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score
from lightgbm import LGBMRegressor, early_stopping

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Constants
DEFAULT_FILE_PATH = r"individual+household+electric+power+consumption\household_power_consumption.txt"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

METRICS_PATH = os.path.join(OUTPUT_DIR, "metrics.json")
CONFIG_PATH = os.path.join(OUTPUT_DIR, "config.json")
ROBUSTNESS_PATH = os.path.join(OUTPUT_DIR, "robustness_checks.json")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

TRAIN_START = "2006-12-16 00:00:00"
VAL_START = "2009-07-01 00:00:00"
TEST_START = "2010-01-01 00:00:00"

TARGET_COL = "Global_active_power"
EXOG_COLS = [
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3"
]
ALL_COLS = [TARGET_COL] + EXOG_COLS

HORIZON = 24
QUANTILES = [0.05, 0.95]


# ==============================================================================
# DATA PREPROCESSING & RESAMPLING
# ==============================================================================
def interpolate_short_gaps(series: pd.Series, max_gap: int = 30) -> Tuple[pd.Series, pd.Series]:
    """
    Interpolates consecutive NaNs only if the gap size is <= max_gap (minutes).
    Returns the imputed series and a boolean mask indicating where imputation occurred.
    """
    is_na = series.isna()
    block_ids = (~is_na).cumsum()
    block_sizes = is_na.groupby(block_ids).transform('sum')
    
    full_interp = series.interpolate(method='linear')
    
    imputed_series = series.copy()
    mask_to_impute = is_na & (block_sizes <= max_gap)
    imputed_series = np.where(mask_to_impute, full_interp, imputed_series)
    imputed_series = pd.Series(imputed_series, index=series.index, name=series.name)
    
    return imputed_series, mask_to_impute


def seasonal_impute_hourly(df_hourly: pd.DataFrame) -> pd.DataFrame:
    """
    Imputes remaining hourly NaNs (long gaps) using seasonal profiles
    (median of dayofweek and hour from training set).
    """
    df = df_hourly.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    
    train_mask = df.index < VAL_START
    df_train = df.loc[train_mask]
    
    for col in ALL_COLS:
        profile = df_train.groupby(['dayofweek', 'hour'])[col].median().reset_index()
        profile.rename(columns={col: f"{col}_profile"}, inplace=True)
        
        df = df.reset_index().merge(profile, on=['dayofweek', 'hour'], how='left').set_index('Datetime')
        df.sort_index(inplace=True)
        
        is_missing = df[col].isna()
        df[col] = df[col].fillna(df[f"{col}_profile"])
        
        num_missing = is_missing.sum()
        if num_missing > 0:
            logger.info(f"Hourly col '{col}' had {num_missing} NaNs (long gaps), seasonally imputed using training profile.")
            
        df[f"{col}_is_profile_imputed"] = is_missing.astype(int)
        df.drop(columns=[f"{col}_profile"], inplace=True)
        
    df.drop(columns=['hour', 'dayofweek'], inplace=True)
    return df


def preprocess_data(file_path: str) -> pd.DataFrame:
    """
    Loads, cleans, and resamples raw data to hourly resolution.
    Applies split resampling (SUM for energy consumption, MEAN for electrical measurements).
    """
    logger.info(f"Loading raw dataset from {file_path}...")
    start_time = time.time()
    
    # Parse semicolon separated dataset
    df = pd.read_csv(file_path, sep=';', low_memory=False)
    logger.info(f"Loaded {len(df):,} minute-level rows in {time.time() - start_time:.2f}s")
    
    # Parse Date & Time columns
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
    df = df.dropna(subset=['Datetime'])
    df.set_index('Datetime', inplace=True)
    df.sort_index(inplace=True)
    
    # Replace ? with NaN, convert every numeric column, and cast to float32
    for col in ALL_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float32)
        
    logger.info("Imputing short gaps (<= 30 minutes) at minute-level...")
    imputed_cols = []
    is_imputed_flags = []
    
    for col in ALL_COLS:
        series_imputed, mask_imputed = interpolate_short_gaps(df[col], max_gap=30)
        imputed_cols.append(series_imputed)
        flag_series = pd.Series(mask_imputed.astype(int), index=df.index, name=f"{col}_is_short_imputed")
        is_imputed_flags.append(flag_series)
        
    df_imputed = pd.concat(imputed_cols + is_imputed_flags, axis=1)
    
    logger.info("Resampling dataset to hourly resolution (split aggregation strategy)...")
    # Consumption columns -> Hourly Sum
    sum_cols = ['Global_active_power', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    df_hourly_sums = df_imputed[sum_cols].resample('h').sum()
    
    # Electrical measurements -> Hourly Mean
    mean_cols = ['Global_reactive_power', 'Voltage', 'Global_intensity']
    df_hourly_means = df_imputed[mean_cols].resample('h').mean()
    
    df_hourly_vals = pd.concat([df_hourly_sums, df_hourly_means], axis=1)
    
    # Resample short-gap flags
    flag_names = [f"{col}_is_short_imputed" for col in ALL_COLS]
    df_hourly_flags = df_imputed[flag_names].resample('h').max().fillna(0).astype(int)
    
    df_hourly = pd.concat([df_hourly_vals, df_hourly_flags], axis=1)
    df_hourly = seasonal_impute_hourly(df_hourly)
    
    logger.info(f"Preprocessing completed. Hourly shape: {df_hourly.shape}")
    return df_hourly


# ==============================================================================
# FEATURE ENGINEERING
# ==============================================================================
def get_holiday_mask(index: pd.DatetimeIndex) -> np.ndarray:
    """
    Returns a binary array indicating whether each datetime is a France public holiday.
    """
    start_year = index.min().year
    end_year = index.max().year
    fr_holidays = holidays.France(years=list(range(start_year, end_year + 1)))
    
    is_holiday = [int(dt.date() in fr_holidays) for dt in index]
    return np.array(is_holiday)


def create_features_and_targets(df_hourly: pd.DataFrame, outlier_clipping: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Constructs multivariable lags, rolling statistics, and targets.
    Rolling features are computed strictly on shifted variables (shift 1) to prevent leakage.
    Casts features to float32 to minimize memory usage.
    """
    logger.info("Constructing feature matrix...")
    df_feat = pd.DataFrame(index=df_hourly.index)
    
    # Outlier clipping
    if outlier_clipping:
        train_mask = df_hourly.index < VAL_START
        p1 = df_hourly.loc[train_mask, TARGET_COL].quantile(0.01)
        p99 = df_hourly.loc[train_mask, TARGET_COL].quantile(0.99)
        logger.info(f"Applying outlier clipping for '{TARGET_COL}': [1st={p1:.3f}, 99th={p99:.3f}]")
        df_feat['load_cleaned'] = df_hourly[TARGET_COL].clip(lower=p1, upper=p99)
    else:
        df_feat['load_cleaned'] = df_hourly[TARGET_COL]
        
    for col in EXOG_COLS:
        if outlier_clipping:
            train_mask = df_hourly.index < VAL_START
            p1 = df_hourly.loc[train_mask, col].quantile(0.01)
            p99 = df_hourly.loc[train_mask, col].quantile(0.99)
            df_feat[f"{col}_cleaned"] = df_hourly[col].clip(lower=p1, upper=p99)
        else:
            df_feat[f"{col}_cleaned"] = df_hourly[col]

    # Historical Lags
    df_feat['load_lag_1'] = df_feat['load_cleaned'].shift(1)
    df_feat['load_lag_2'] = df_feat['load_cleaned'].shift(2)
    df_feat['load_lag_3'] = df_feat['load_cleaned'].shift(3)
    df_feat['load_lag_24'] = df_feat['load_cleaned'].shift(24)
    df_feat['load_lag_25'] = df_feat['load_cleaned'].shift(25)
    df_feat['load_lag_48'] = df_feat['load_cleaned'].shift(48)
    df_feat['load_lag_168'] = df_feat['load_cleaned'].shift(168)
    df_feat['load_lag_169'] = df_feat['load_cleaned'].shift(169)
    
    for col in EXOG_COLS:
        df_feat[f"{col}_lag_1"] = df_feat[f"{col}_cleaned"].shift(1)
        df_feat[f"{col}_lag_2"] = df_feat[f"{col}_cleaned"].shift(2)
        df_feat[f"{col}_lag_24"] = df_feat[f"{col}_cleaned"].shift(24)
        
    # Rolling Statistics (leakage-free)
    load_shifted = df_feat['load_cleaned'].shift(1)
    df_feat['rolling_mean_24'] = load_shifted.rolling(24).mean()
    df_feat['rolling_std_24'] = load_shifted.rolling(24).std()
    df_feat['rolling_min_24'] = load_shifted.rolling(24).min()
    df_feat['rolling_max_24'] = load_shifted.rolling(24).max()
    
    df_feat['rolling_mean_168'] = load_shifted.rolling(168).mean()
    df_feat['rolling_std_168'] = load_shifted.rolling(168).std()
    
    # EWMA features
    df_feat['ewm_mean_12'] = load_shifted.ewm(span=12, adjust=False).mean()
    df_feat['ewm_mean_24'] = load_shifted.ewm(span=24, adjust=False).mean()
    
    for col in EXOG_COLS:
        exog_shifted = df_feat[f"{col}_cleaned"].shift(1)
        df_feat[f"{col}_rolling_mean_24"] = exog_shifted.rolling(24).mean()
        
    # Imputation Binary Flags
    for col in ALL_COLS:
        df_feat[f"{col}_is_short_imputed"] = df_hourly[f"{col}_is_short_imputed"]
        df_feat[f"{col}_is_profile_imputed"] = df_hourly[f"{col}_is_profile_imputed"]
        
    # Target variables
    targets = {}
    for h in range(1, HORIZON + 1):
        targets[f"target_h{h}"] = df_hourly[TARGET_COL].shift(-h)
        
    df_targets = pd.DataFrame(targets, index=df_hourly.index)
    
    valid_mask = df_feat.notna().all(axis=1) & df_targets.notna().all(axis=1)
    X = df_feat.loc[valid_mask].copy()
    Y = df_targets.loc[valid_mask].copy()
    
    # Cast to float32 (Memory Optimization)
    float64_cols = X.select_dtypes(include=['float64']).columns
    X[float64_cols] = X[float64_cols].astype(np.float32)
    Y = Y.astype(np.float32)
    
    logger.info(f"Feature matrix X shape: {X.shape}, Target Y shape: {Y.shape}")
    return X, Y


def get_target_temporal_features(index: pd.DatetimeIndex, h: int) -> pd.DataFrame:
    """
    Generates cyclical calendar features for target hour (t+h).
    Adds categorical variables for LightGBM native categorical split support.
    """
    target_times = index + pd.Timedelta(hours=h)
    
    hour = target_times.hour
    dayofweek = target_times.dayofweek
    month = target_times.month
    dayofyear = target_times.dayofyear
    
    df_cal = pd.DataFrame(index=index)
    df_cal['is_weekend'] = (dayofweek >= 5).astype(int)
    df_cal['is_holiday'] = get_holiday_mask(target_times)
    
    # LightGBM Categorical fields
    df_cal['hour_cat'] = hour.astype('category')
    df_cal['dayofweek_cat'] = dayofweek.astype('category')
    df_cal['month_cat'] = month.astype('category')
    
    # Cyclical encodings
    df_cal['hour_sin'] = np.sin(2 * np.pi * hour / 24.0).astype(np.float32)
    df_cal['hour_cos'] = np.cos(2 * np.pi * hour / 24.0).astype(np.float32)
    df_cal['dayofweek_sin'] = np.sin(2 * np.pi * dayofweek / 7.0).astype(np.float32)
    df_cal['dayofweek_cos'] = np.cos(2 * np.pi * dayofweek / 7.0).astype(np.float32)
    df_cal['month_sin'] = np.sin(2 * np.pi * month / 12.0).astype(np.float32)
    df_cal['month_cos'] = np.cos(2 * np.pi * month / 12.0).astype(np.float32)
    df_cal['dayofyear_sin'] = np.sin(2 * np.pi * dayofyear / 365.25).astype(np.float32)
    df_cal['dayofyear_cos'] = np.cos(2 * np.pi * dayofyear / 365.25).astype(np.float32)
    
    return df_cal


# ==============================================================================
# MODEL CONFIGURATION & OPTUNA + TIME-SERIES TUNING
# ==============================================================================
class DayAheadLoadForecaster:
    """
    Manages point and interval forecasts for 24 horizons using LightGBM.
    Leverages Optuna + TimeSeriesSplit for optimal hyperparameter search.
    """
    def __init__(self, best_params: Dict[str, Any] = None):
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
        self.point_models: Dict[int, LGBMRegressor] = {}
        self.quantile_models: Dict[float, Dict[int, LGBMRegressor]] = {q: {} for q in QUANTILES}
        self.feature_cols: List[str] = []

    def tune_hyperparameters(self, X: pd.DataFrame, Y: pd.DataFrame):
        """
        Hyperparameter optimization using Optuna & TimeSeriesSplit to find best config.
        """
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
            
            # 3 splits time-series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            val_scores = []
            
            h = 12
            y_tune_h = Y_tune[f"target_h{h}"]
            
            for train_idx, val_idx in tscv.split(X_tune):
                X_tr, X_va = X_tune.iloc[train_idx], X_tune.iloc[val_idx]
                y_tr, y_va = y_tune_h.iloc[train_idx], y_tune_h.iloc[val_idx]
                
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
        logger.info(f"Optuna Optimization complete. Best params: {self.params} (Best Val MAE Score: {study.best_value:.4f})")

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame, tune: bool = True):
        """
        Trains point and quantile models with early stopping.
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
            
            X_train_cal = get_target_temporal_features(X_train_base.index, h)
            X_val_cal = get_target_temporal_features(X_val_base.index, h)
            
            X_train = pd.concat([X_train_base, X_train_cal], axis=1)
            X_val = pd.concat([X_val_base, X_val_cal], axis=1)
            
            # Point forecast model
            point_model = LGBMRegressor(n_estimators=1000, **self.params)
            point_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[early_stopping(stopping_rounds=50, verbose=False)])
            self.point_models[h] = point_model
            
            # Quantile models
            for q in QUANTILES:
                q_model = LGBMRegressor(n_estimators=1000, objective='quantile', alpha=q, **self.params)
                q_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[early_stopping(stopping_rounds=50, verbose=False)])
                self.quantile_models[q][h] = q_model
                
            logger.info(f"Trained Point & Quantile models for Horizon +{h}h (Best Point Iter: {point_model.best_iteration_})")
            
        logger.info(f"All models trained successfully in {time.time() - start_time:.2f} seconds.")

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates forecasts for point and lower/upper quantile ranges.
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
            
        lower_preds = np.minimum(lower_preds, point_preds)
        upper_preds = np.maximum(upper_preds, point_preds)
        
        return point_preds, lower_preds, upper_preds


# ==============================================================================
# EVALUATION & ROBUSTNESS BREAKDOWN
# ==============================================================================
def calculate_metrics(actuals: np.ndarray, predictions: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Computes MAE, RMSE, MAPE, and R2 metrics.
    """
    mae = np.mean(np.abs(actuals - predictions))
    rmse = np.sqrt(np.mean((actuals - predictions)**2))
    
    actuals_safe = np.where(actuals == 0, 1e-5, actuals)
    mape = np.mean(np.abs((actuals - predictions) / actuals_safe)) * 100
    
    r2 = r2_score(actuals.flatten(), predictions.flatten())
    
    return float(mae), float(rmse), float(mape), float(r2)


def calculate_pinball_loss(actuals: np.ndarray, predictions: np.ndarray, q: float) -> float:
    """
    Computes Pinball Loss for quantile evaluation.
    """
    diff = actuals - predictions
    loss = np.where(diff >= 0, q * diff, (q - 1) * diff)
    return float(np.mean(loss))


def calculate_coverage_rate(actuals: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """
    Computes Prediction Interval Coverage Probability (PICP).
    """
    within_bounds = (actuals >= lower) & (actuals <= upper)
    return float(np.mean(within_bounds)) * 100


def get_season(month: int) -> str:
    if month in [12, 1, 2]: return "Winter"
    elif month in [3, 4, 5]: return "Spring"
    elif month in [6, 7, 8]: return "Summer"
    else: return "Autumn"


def perform_robustness_checks(
    test_indices: pd.DatetimeIndex,
    actuals: np.ndarray,
    predictions: np.ndarray,
    train_median_load: float
) -> Dict[str, Any]:
    """
    Groups forecast errors by hour, day type, season, and load level.
    Saves outputs to robustness_checks.json.
    """
    logger.info("Performing model robustness checks (error analysis)...")
    records = []
    
    for idx, t in enumerate(test_indices):
        for h in range(1, HORIZON + 1):
            target_t = t + pd.Timedelta(hours=h)
            actual_val = actuals[idx, h - 1]
            pred_val = predictions[idx, h - 1]
            
            records.append({
                'hour': target_t.hour,
                'dayofweek': target_t.dayofweek,
                'month': target_t.month,
                'actual': actual_val,
                'ae': abs(actual_val - pred_val),
                'se': (actual_val - pred_val)**2
            })
            
    df_eval = pd.DataFrame(records)
    df_eval['day_type'] = np.where(df_eval['dayofweek'] >= 5, "Weekend", "Weekday")
    df_eval['season'] = df_eval['month'].apply(get_season)
    df_eval['load_level'] = np.where(df_eval['actual'] <= train_median_load, "Low-Load (<=Median)", "High-Load (>Median)")
    
    robustness = {}
    
    # Hour
    by_hour = {}
    for hour, group in df_eval.groupby('hour'):
        mae = group['ae'].mean()
        rmse = np.sqrt(group['se'].mean())
        mape = (group['ae'] / np.where(group['actual'] == 0, 1e-5, group['actual'])).mean() * 100
        by_hour[int(hour)] = {'MAE': float(mae), 'RMSE': float(rmse), 'MAPE': float(mape)}
    robustness['hour_of_day'] = by_hour
    
    # Day Type
    by_day_type = {}
    for dt, group in df_eval.groupby('day_type'):
        mae = group['ae'].mean()
        rmse = np.sqrt(group['se'].mean())
        mape = (group['ae'] / np.where(group['actual'] == 0, 1e-5, group['actual'])).mean() * 100
        by_day_type[str(dt)] = {'MAE': float(mae), 'RMSE': float(rmse), 'MAPE': float(mape)}
    robustness['day_type'] = by_day_type
    
    # Season
    by_season = {}
    for season, group in df_eval.groupby('season'):
        mae = group['ae'].mean()
        rmse = np.sqrt(group['se'].mean())
        mape = (group['ae'] / np.where(group['actual'] == 0, 1e-5, group['actual'])).mean() * 100
        by_season[str(season)] = {'MAE': float(mae), 'RMSE': float(rmse), 'MAPE': float(mape)}
    robustness['season'] = by_season
    
    # Load level
    by_load_level = {}
    for level, group in df_eval.groupby('load_level'):
        mae = group['ae'].mean()
        rmse = np.sqrt(group['se'].mean())
        mape = (group['ae'] / np.where(group['actual'] == 0, 1e-5, group['actual'])).mean() * 100
        by_load_level[str(level)] = {'MAE': float(mae), 'RMSE': float(rmse), 'MAPE': float(mape)}
    robustness['load_level'] = by_load_level
    
    with open(ROBUSTNESS_PATH, 'w') as f:
        json.dump(robustness, f, indent=4)
    logger.info(f"Robustness metrics saved to {ROBUSTNESS_PATH}")
    
    return robustness


# ==============================================================================
# VISUAL DIAGNOSTICS & EXPERIMENT TRACKING
# ==============================================================================
def plot_results(
    test_indices: pd.DatetimeIndex,
    actuals: np.ndarray,
    predictions: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    daily_naive: np.ndarray,
    models_dict: Dict[int, LGBMRegressor],
    robustness: Dict[str, Any]
):
    """
    Generates and saves the 5 requested diagnostic plots.
    """
    # 1. Forecast Comparison with prediction intervals
    plot_start, plot_end = "2010-01-08 00:00:00", "2010-01-14 23:00:00"
    plot_mask = (test_indices >= plot_start) & (test_indices <= plot_end)
    indices = np.where(plot_mask)[0]
    
    if len(indices) > 0:
        t_axis = test_indices[indices]
        plt.figure(figsize=(14, 7), dpi=100)
        plt.fill_between(t_axis, lower_bound[indices, 0], upper_bound[indices, 0], color='#ff7f0e', alpha=0.15, label='90% Confidence Interval')
        plt.plot(t_axis, actuals[indices, 0], label='Actual Load', color='#2b2b2b', linewidth=2)
        plt.plot(t_axis, predictions[indices, 0], label='LightGBM Forecaster (t+1)', color='#ff7f0e', linestyle='--', alpha=0.9)
        plt.plot(t_axis, daily_naive[indices, 0], label='Daily Naive Baseline (t+1)', color='#1f77b4', linestyle=':', alpha=0.8)
        
        plt.title('Day-Ahead Hourly Load Forecast & 90% Prediction Intervals', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Datetime', fontsize=12)
        plt.ylabel('Active Power (kW)', fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(frameon=True, facecolor='white', edgecolor='none', shadow=True, fontsize=11, loc='upper right')
        plt.tight_layout()
        
        plot_path = os.path.join(OUTPUT_DIR, "forecast_comparison_intervals.png")
        plt.savefig(plot_path)
        plt.close()
        
        # 2. Residuals Over Time Plot
        plt.figure(figsize=(14, 5), dpi=100)
        plt.plot(t_axis, actuals[indices, 0] - predictions[indices, 0], label='Residuals (Actual - Predicted)', color='#d62728', linewidth=1.5)
        plt.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)
        plt.title('Forecast Residuals Over Time (t+1 Horizon)', fontsize=12, fontweight='bold', pad=15)
        plt.xlabel('Datetime', fontsize=11)
        plt.ylabel('Error (kW)', fontsize=11)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(frameon=True, facecolor='white', edgecolor='none')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "residuals_vs_time.png"))
        plt.close()
        
    # 3. Error Distribution Histogram
    residuals_flat = (actuals - predictions).flatten()
    plt.figure(figsize=(10, 6), dpi=100)
    plt.hist(residuals_flat, bins=60, density=True, color='#9467bd', alpha=0.75, edgecolor='white')
    mu, sigma = np.mean(residuals_flat), np.std(residuals_flat)
    bins = np.linspace(np.min(residuals_flat), np.max(residuals_flat), 100)
    plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
             linewidth=2, color='#2b2b2b', label=f'Normal Fit (μ={mu:.3f}, σ={sigma:.3f})')
    plt.title('Distribution of Prediction Errors (Residuals)', fontsize=12, fontweight='bold', pad=15)
    plt.xlabel('Prediction Error (Actual - Predicted) in kW', fontsize=11)
    plt.ylabel('Density', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(frameon=True, facecolor='white', edgecolor='none')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "error_distribution.png"))
    plt.close()
        
    # 4. Feature Importance Plot (Top 20 Features)
    feat_imp_df = pd.DataFrame()
    for h, model in models_dict.items():
        df_temp = pd.DataFrame({'feature': model.feature_name_, f'importance_h{h}': model.feature_importances_})
        feat_imp_df = df_temp if feat_imp_df.empty else feat_imp_df.merge(df_temp, on='feature', how='outer')
        
    imp_cols = [c for c in feat_imp_df.columns if c != 'feature']
    feat_imp_df['mean_importance'] = feat_imp_df[imp_cols].mean(axis=1)
    feat_imp_df.sort_values('mean_importance', ascending=True, inplace=True)
    top_n = feat_imp_df.tail(20)
    
    plt.figure(figsize=(10, 7), dpi=100)
    plt.barh(top_n['feature'], top_n['mean_importance'], color='#ff7f0e', alpha=0.85)
    plt.title('Averaged LightGBM Feature Importance (Top 20 Features)', fontsize=12, fontweight='bold', pad=15)
    plt.xlabel('Importance Value (Split Count)', fontsize=11)
    plt.grid(True, axis='x', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"))
    plt.close()
    
    # 5. Robustness check error breakdowns plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=100)
    
    hour_data = robustness['hour_of_day']
    hours = sorted(list(hour_data.keys()))
    axes[0, 0].plot(hours, [hour_data[h]['MAE'] for h in hours], marker='o', color='#ff7f0e', linewidth=2)
    axes[0, 0].set_title('MAE by Hour of Day', fontsize=11, fontweight='bold')
    axes[0, 0].set_xticks(range(0, 24, 2))
    axes[0, 0].grid(True, linestyle=':', alpha=0.6)
    
    day_data = robustness['day_type']
    axes[0, 1].bar(list(day_data.keys()), [day_data[d]['MAE'] for d in day_data.keys()], color=['#1f77b4', '#ff7f0e'], alpha=0.8, width=0.4)
    axes[0, 1].set_title('MAE by Day Type', fontsize=11, fontweight='bold')
    axes[0, 1].grid(True, axis='y', linestyle=':', alpha=0.6)
    
    season_data = robustness['season']
    axes[1, 0].bar(list(season_data.keys()), [season_data[s]['MAE'] for s in season_data.keys()], color='#2ca02c', alpha=0.8, width=0.4)
    axes[1, 0].set_title('MAE by Season', fontsize=11, fontweight='bold')
    axes[1, 0].grid(True, axis='y', linestyle=':', alpha=0.6)
    
    load_data = robustness['load_level']
    axes[1, 1].bar(list(load_data.keys()), [load_data[l]['MAE'] for l in load_data.keys()], color='#9467bd', alpha=0.8, width=0.4)
    axes[1, 1].set_title('MAE by Load Level', fontsize=11, fontweight='bold')
    axes[1, 1].grid(True, axis='y', linestyle=':', alpha=0.6)
    
    plt.suptitle('Model Robustness Error Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "robustness_analysis.png"))
    plt.close()


def track_experiment(metrics: Dict[str, Any], hyperparameters: Dict[str, Any], features_used: int, training_time: float):
    config_record = {
        'timestamp': datetime.now().isoformat(),
        'hyperparameters': hyperparameters,
        'num_features': features_used,
        'training_time_seconds': training_time
    }
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config_record, f, indent=4)
        
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    exp_dir = os.path.join(OUTPUT_DIR, "experiments")
    os.makedirs(exp_dir, exist_ok=True)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(exp_dir, f"run_{run_timestamp}.json"), 'w') as f:
        json.dump({'config': config_record, 'metrics': metrics}, f, indent=4)


# ==============================================================================
# PIPELINE ORCHESTRATION
# ==============================================================================
def compute_baselines(df_hourly: pd.DataFrame, test_indices: pd.DatetimeIndex) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes Daily Naive, Weekly Naive, and 7-day Rolling Average baseline forecasts.
    """
    n_test = len(test_indices)
    daily_naive = np.zeros((n_test, HORIZON))
    weekly_naive = np.zeros((n_test, HORIZON))
    rolling_avg = np.zeros((n_test, HORIZON))
    
    load_series = df_hourly[TARGET_COL]
    
    for idx, t in enumerate(test_indices):
        # Daily Naive: same hour yesterday (t-23 to t)
        daily_naive[idx, :] = load_series.loc[t - pd.Timedelta(hours=23) : t].values
        # Weekly Naive: same hour last week (t-167 to t-144)
        weekly_naive[idx, :] = load_series.loc[t - pd.Timedelta(hours=167) : t - pd.Timedelta(hours=144)].values
        # Rolling Average: mean of past 168 hours (7 days) ending at t
        rolling_avg[idx, :] = load_series.loc[t - pd.Timedelta(hours=167) : t].mean()
        
    return daily_naive, weekly_naive, rolling_avg


def main():
    parser = argparse.ArgumentParser(description="Run the day-ahead electricity load forecaster pipeline.")
    parser.add_argument("--data", type=str, default=DEFAULT_FILE_PATH, help="Path to household_power_consumption.txt")
    args = parser.parse_args()
    
    logger.info("Initializing day-ahead forecaster pipeline...")
    total_start = time.time()
    
    # 1. Preprocess
    df_hourly = preprocess_data(args.data)
    
    # 2. Features
    X, Y = create_features_and_targets(df_hourly, outlier_clipping=True)
    
    train_median_load = float(df_hourly.loc[df_hourly.index < VAL_START, TARGET_COL].median())
    logger.info(f"Training median load: {train_median_load:.3f} kW")
    
    # 3. Model Fit (Optuna + TimeSeriesSplit tuning inside)
    forecaster = DayAheadLoadForecaster()
    train_start = time.time()
    forecaster.fit(X, Y, tune=True)
    training_time = time.time() - train_start
    
    # Save model
    with open(os.path.join(MODEL_DIR, "forecaster_lgb.pkl"), 'wb') as f:
        pickle.dump(forecaster, f)
    logger.info("Model serialized and saved.")
    
    # 4. Predict
    test_mask = X.index >= TEST_START
    X_test = X.loc[test_mask]
    Y_test = Y.loc[test_mask]
    
    logger.info("Generating predictions on the test set...")
    pred_start = time.time()
    point_preds, lower_preds, upper_preds = forecaster.predict(X_test)
    prediction_time = time.time() - pred_start
    
    actuals = Y_test.values
    
    # 5. Baselines
    daily_naive, weekly_naive, rolling_avg = compute_baselines(df_hourly, X_test.index)
    
    # 6. Evaluate
    mae_lgbm, rmse_lgbm, mape_lgbm, r2_lgbm = calculate_metrics(actuals, point_preds)
    mae_daily, rmse_daily, mape_daily, r2_daily = calculate_metrics(actuals, daily_naive)
    mae_weekly, rmse_weekly, mape_weekly, r2_weekly = calculate_metrics(actuals, weekly_naive)
    mae_roll, rmse_roll, mape_roll, r2_roll = calculate_metrics(actuals, rolling_avg)
    
    pinball_05 = calculate_pinball_loss(actuals, lower_preds, 0.05)
    pinball_95 = calculate_pinball_loss(actuals, upper_preds, 0.95)
    coverage = calculate_coverage_rate(actuals, lower_preds, upper_preds)
    
    metrics = {
        'model_metrics': {'MAE': mae_lgbm, 'RMSE': rmse_lgbm, 'MAPE': mape_lgbm, 'R2': r2_lgbm, 'training_time_s': training_time, 'prediction_time_s': prediction_time},
        'daily_naive': {'MAE': mae_daily, 'RMSE': rmse_daily, 'MAPE': mape_daily, 'R2': r2_daily},
        'weekly_naive': {'MAE': mae_weekly, 'RMSE': rmse_weekly, 'MAPE': mape_weekly, 'R2': r2_weekly},
        'rolling_avg_7d': {'MAE': mae_roll, 'RMSE': rmse_roll, 'MAPE': mape_roll, 'R2': r2_roll},
        'quantiles': {
            'pinball_0.05': pinball_05,
            'pinball_0.95': pinball_95,
            'coverage_rate_90pct': coverage
        }
    }
    
    robustness = perform_robustness_checks(X_test.index, actuals, point_preds, train_median_load)
    plot_results(X_test.index, actuals, point_preds, lower_preds, upper_preds, daily_naive, forecaster.point_models, robustness)
    track_experiment(metrics, forecaster.params, len(forecaster.feature_cols), training_time)
    
    # 7. Print Output Table
    print("\n" + "="*90)
    print("DAY-AHEAD ELECTRICITY LOAD FORECASTER EVALUATION RESULTS")
    print("="*90)
    print(f"{'Model/Baseline':<35} | {'MAE (kW)':<10} | {'RMSE (kW)':<10} | {'MAPE (%)':<10} | {'R² Score':<10}")
    print("-"*90)
    print(f"{'Weekly Naive (Same hour last week)':<35} | {mae_weekly:<10.4f} | {rmse_weekly:<10.4f} | {mape_weekly:<10.2f} | {r2_weekly:<10.4f}")
    print(f"{'Daily Naive (Tomorrow looks like today)':<35} | {mae_daily:<10.4f} | {rmse_daily:<10.4f} | {mape_daily:<10.2f} | {r2_daily:<10.4f}")
    print(f"{'7-Day Rolling Average Persistence':<35} | {mae_roll:<10.4f} | {rmse_roll:<10.4f} | {mape_roll:<10.2f} | {r2_roll:<10.4f}")
    print(f"{'LightGBM Direct Forecaster (Ours)':<35} | {mae_lgbm:<10.4f} | {rmse_lgbm:<10.4f} | {mape_lgbm:<10.2f} | {r2_lgbm:<10.4f}")
    print("="*90)
    print("INTERVAL FORECAST METRICS (90% CONFIDENCE INTERVALS)")
    print("-"*90)
    print(f"Pinball Loss (q=0.05): {pinball_05:.4f}")
    print(f"Pinball Loss (q=0.95): {pinball_95:.4f}")
    print(f"Nominal 90% Coverage Rate: {coverage:.2f}% (Expected: ~90.00%)")
    print("="*90)
    print("RUN TIMES")
    print("-"*90)
    print(f"Model Training Time (72 models): {training_time:.2f} seconds")
    print(f"Model Prediction Time ({len(X_test)} samples): {prediction_time:.4f} seconds")
    print("="*90)
    
    imp_mae = ((mae_daily - mae_lgbm) / mae_daily) * 100
    imp_rmse = ((rmse_daily - rmse_lgbm) / rmse_daily) * 100
    print(f"Point Forecast Improvement over Daily Naive: MAE: {imp_mae:+.2f}%, RMSE: {imp_rmse:+.2f}%")
    print(f"Full execution finished in {time.time() - total_start:.2f} seconds.")
    print("="*90 + "\n")


if __name__ == "__main__":
    main()
