"""
Feature engineering module.
Constructs lags, rolling statistics (leakage-free), cyclical temporal features,
and holiday flags for the load forecasting model.
"""

import logging
import numpy as np
import pandas as pd
import holidays
from typing import Tuple
from src.config import VAL_START, TARGET_COL, EXOG_COLS, ALL_COLS, HORIZON

logger = logging.getLogger(__name__)


def get_holiday_mask(index: pd.DatetimeIndex) -> np.ndarray:
    """
    Returns a binary array indicating whether each datetime in index is a French holiday.
    """
    start_year = index.min().year
    end_year = index.max().year
    fr_holidays = holidays.France(years=list(range(start_year, end_year + 1)))
    
    # Check each date (ignoring time) in the holidays set
    is_holiday = [int(dt.date() in fr_holidays) for dt in index]
    return np.array(is_holiday)


def create_features_and_targets(df_hourly: pd.DataFrame, outlier_clipping: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Builds the feature matrix X and target matrix Y.
    All historical rolling stats are computed on shifted columns to ensure zero leakage.
    """
    logger.info("Constructing feature matrix...")
    
    df_feat = pd.DataFrame(index=df_hourly.index)
    
    # 1. Outlier clipping on active power based on training set percentiles
    if outlier_clipping:
        train_mask = df_hourly.index < VAL_START
        p1 = df_hourly.loc[train_mask, TARGET_COL].quantile(0.01)
        p99 = df_hourly.loc[train_mask, TARGET_COL].quantile(0.99)
        logger.info(f"Applying outlier clipping for '{TARGET_COL}': [1st={p1:.3f}, 99th={p99:.3f}]")
        df_feat['load_cleaned'] = df_hourly[TARGET_COL].clip(lower=p1, upper=p99)
    else:
        df_feat['load_cleaned'] = df_hourly[TARGET_COL]
        
    # Also clip exogenous features to limit influence of sensor spikes
    for col in EXOG_COLS:
        if outlier_clipping:
            train_mask = df_hourly.index < VAL_START
            p1 = df_hourly.loc[train_mask, col].quantile(0.01)
            p99 = df_hourly.loc[train_mask, col].quantile(0.99)
            df_feat[f"{col}_cleaned"] = df_hourly[col].clip(lower=p1, upper=p99)
        else:
            df_feat[f"{col}_cleaned"] = df_hourly[col]

    # 2. Historical Lags (strictly available at time t, meaning lag >= 1 hour relative to target)
    # We define lag_k as the value at t - k + 1.
    # To follow Point 3 (rolling calculations on shifted load), all history features use shift(1) or higher.
    # This prevents any concurrent value leakage.
    df_feat['load_lag_1'] = df_feat['load_cleaned'].shift(1)
    df_feat['load_lag_2'] = df_feat['load_cleaned'].shift(2)
    df_feat['load_lag_3'] = df_feat['load_cleaned'].shift(3)
    df_feat['load_lag_24'] = df_feat['load_cleaned'].shift(24)
    df_feat['load_lag_25'] = df_feat['load_cleaned'].shift(25)
    df_feat['load_lag_48'] = df_feat['load_cleaned'].shift(48)
    df_feat['load_lag_168'] = df_feat['load_cleaned'].shift(168)
    df_feat['load_lag_169'] = df_feat['load_cleaned'].shift(169)
    
    # Exogenous Lags
    for col in EXOG_COLS:
        df_feat[f"{col}_lag_1"] = df_feat[f"{col}_cleaned"].shift(1)
        df_feat[f"{col}_lag_2"] = df_feat[f"{col}_cleaned"].shift(2)
        df_feat[f"{col}_lag_24"] = df_feat[f"{col}_cleaned"].shift(24)
        
    # 3. Rolling Statistics (computed on shift(1) to guarantee zero leakage)
    load_shifted = df_feat['load_cleaned'].shift(1)
    df_feat['rolling_mean_24'] = load_shifted.rolling(24).mean()
    df_feat['rolling_std_24'] = load_shifted.rolling(24).std()
    df_feat['rolling_min_24'] = load_shifted.rolling(24).min()
    df_feat['rolling_max_24'] = load_shifted.rolling(24).max()
    
    df_feat['rolling_mean_168'] = load_shifted.rolling(168).mean()
    df_feat['rolling_std_168'] = load_shifted.rolling(168).std()
    
    # EWMA features on shifted series
    df_feat['ewm_mean_12'] = load_shifted.ewm(span=12, adjust=False).mean()
    df_feat['ewm_mean_24'] = load_shifted.ewm(span=24, adjust=False).mean()
    
    # Exogenous rolling features
    for col in EXOG_COLS:
        exog_shifted = df_feat[f"{col}_cleaned"].shift(1)
        df_feat[f"{col}_rolling_mean_24"] = exog_shifted.rolling(24).mean()
        
    # 4. Imputation Binary Indicators (passed to model as features)
    for col in ALL_COLS:
        df_feat[f"{col}_is_short_imputed"] = df_hourly[f"{col}_is_short_imputed"]
        df_feat[f"{col}_is_profile_imputed"] = df_hourly[f"{col}_is_profile_imputed"]
        
    # 5. Target Variables (horizon t+1 to t+24)
    # The actual unclipped Global_active_power values
    targets = {}
    for h in range(1, HORIZON + 1):
        targets[f"target_h{h}"] = df_hourly[TARGET_COL].shift(-h)
        
    df_targets = pd.DataFrame(targets, index=df_hourly.index)
    
    # Drop rows that contain NaNs in features (due to shifts and rollings) or targets
    valid_mask = df_feat.notna().all(axis=1) & df_targets.notna().all(axis=1)
    X = df_feat.loc[valid_mask].copy()
    Y = df_targets.loc[valid_mask].copy()
    
    # Cast all float64 columns to float32 to reduce memory usage (Industry Standard)
    float64_cols = X.select_dtypes(include=['float64']).columns
    X[float64_cols] = X[float64_cols].astype(np.float32)
    Y = Y.astype(np.float32)
    
    logger.info(f"Feature matrix X shape: {X.shape}, Target Y shape: {Y.shape}")
    return X, Y


def get_target_temporal_features(index: pd.DatetimeIndex, h: int) -> pd.DataFrame:
    """
    Generates cyclical and calendar features for target hour (t+h).
    Enriches features with France public holidays.
    """
    target_times = index + pd.Timedelta(hours=h)
    
    hour = target_times.hour
    dayofweek = target_times.dayofweek
    month = target_times.month
    dayofyear = target_times.dayofyear
    
    df_cal = pd.DataFrame(index=index)
    df_cal['is_weekend'] = (dayofweek >= 5).astype(int)
    df_cal['is_holiday'] = get_holiday_mask(target_times)
    
    # Add raw calendar categories for LightGBM native categoricals
    df_cal['hour_cat'] = hour.astype('category')
    df_cal['dayofweek_cat'] = dayofweek.astype('category')
    df_cal['month_cat'] = month.astype('category')
    
    # Cyclical Encodings
    df_cal['hour_sin'] = np.sin(2 * np.pi * hour / 24.0).astype(np.float32)
    df_cal['hour_cos'] = np.cos(2 * np.pi * hour / 24.0).astype(np.float32)
    df_cal['dayofweek_sin'] = np.sin(2 * np.pi * dayofweek / 7.0).astype(np.float32)
    df_cal['dayofweek_cos'] = np.cos(2 * np.pi * dayofweek / 7.0).astype(np.float32)
    df_cal['month_sin'] = np.sin(2 * np.pi * month / 12.0).astype(np.float32)
    df_cal['month_cos'] = np.cos(2 * np.pi * month / 12.0).astype(np.float32)
    df_cal['dayofyear_sin'] = np.sin(2 * np.pi * dayofyear / 365.25).astype(np.float32)
    df_cal['dayofyear_cos'] = np.cos(2 * np.pi * dayofyear / 365.25).astype(np.float32)
    
    return df_cal
