"""
Preprocessing module for raw data ingestion, cleaning, and imputation.
Handles short gaps with linear interpolation, long gaps with seasonal median profiles,
and adds binary imputation flags.
"""

import time
import logging
import numpy as np
import pandas as pd
from typing import Tuple, List
from src.config import VAL_START, ALL_COLS

logger = logging.getLogger(__name__)


def interpolate_short_gaps(series: pd.Series, max_gap: int = 30) -> Tuple[pd.Series, pd.Series]:
    """
    Interpolates consecutive NaNs only if the gap size is <= max_gap (minutes).
    Returns the imputed series and a boolean mask indicating where imputation occurred.
    """
    is_na = series.isna()
    # Create block IDs for contiguous NaN groups
    block_ids = (~is_na).cumsum()
    # Count sizes of each block
    block_sizes = is_na.groupby(block_ids).transform('sum')
    
    # Interpolate all, then restore NaNs for long blocks
    full_interp = series.interpolate(method='linear')
    
    imputed_series = series.copy()
    # Use full interpolation where NaN and block_size <= max_gap
    mask_to_impute = is_na & (block_sizes <= max_gap)
    imputed_series = np.where(mask_to_impute, full_interp, imputed_series)
    imputed_series = pd.Series(imputed_series, index=series.index, name=series.name)
    
    return imputed_series, mask_to_impute


def seasonal_impute_hourly(df_hourly: pd.DataFrame) -> pd.DataFrame:
    """
    Imputes remaining hourly NaNs (from long gaps) using seasonal profiles
    (median of dayofweek and hour from training set).
    """
    df = df_hourly.copy()
    
    # Create temporary features for grouping
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    
    # Separate training set to calculate profiles (to avoid leakage)
    train_mask = df.index < VAL_START
    df_train = df.loc[train_mask]
    
    for col in ALL_COLS:
        # Calculate seasonal profiles (median)
        profile = df_train.groupby(['dayofweek', 'hour'])[col].median().reset_index()
        profile.rename(columns={col: f"{col}_profile"}, inplace=True)
        
        # Merge profile back
        df = df.reset_index().merge(profile, on=['dayofweek', 'hour'], how='left').set_index('Datetime')
        df.sort_index(inplace=True)
        
        # Impute
        is_missing = df[col].isna()
        df[col] = df[col].fillna(df[f"{col}_profile"])
        
        # Log missingness
        num_missing = is_missing.sum()
        if num_missing > 0:
            logger.info(f"Hourly col '{col}' had {num_missing} NaNs (long gaps), seasonally imputed using training profile.")
            
        # Add profile imputation flag
        df[f"{col}_is_profile_imputed"] = is_missing.astype(int)
        df.drop(columns=[f"{col}_profile"], inplace=True)
        
    df.drop(columns=['hour', 'dayofweek'], inplace=True)
    return df


def preprocess_data(file_path: str) -> pd.DataFrame:
    """
    Main preprocessor pipeline:
    1. Loads semicolon separated file
    2. Parses dates & times
    3. Handles minute-level short gaps
    4. Resamples to hourly mean
    5. Imputes long hourly gaps seasonally
    6. Returns clean hourly dataframe with binary imputation flags.
    """
    logger.info(f"Loading raw dataset from {file_path}...")
    start_time = time.time()
    
    # Load dataset
    df = pd.read_csv(file_path, sep=';', low_memory=False)
    logger.info(f"Loaded {len(df):,} minute-level rows in {time.time() - start_time:.2f}s")
    
    # Parse Datetime
    logger.info("Parsing Date and Time columns...")
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
    df = df.dropna(subset=['Datetime'])
    df.set_index('Datetime', inplace=True)
    df.sort_index(inplace=True)
    
    # Convert active/reactive power, voltage, intensity, sub_meterings to numeric and cast to float32
    for col in ALL_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float32)
        
    # Apply minute-level short gap imputation
    logger.info("Imputing short gaps (<= 30 minutes) at minute-level...")
    imputed_cols = []
    is_imputed_flags = []
    
    for col in ALL_COLS:
        series_imputed, mask_imputed = interpolate_short_gaps(df[col], max_gap=30)
        imputed_cols.append(series_imputed)
        # Store indicator column
        flag_series = pd.Series(mask_imputed.astype(int), index=df.index, name=f"{col}_is_short_imputed")
        is_imputed_flags.append(flag_series)
        
    df_imputed = pd.concat(imputed_cols + is_imputed_flags, axis=1)
    
    # Resample to hourly (SUM for energy consumption, MEAN for electrical measurements)
    logger.info("Resampling dataset to hourly resolution...")
    
    sum_cols = ['Global_active_power', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    df_hourly_sums = df_imputed[sum_cols].resample('h').sum()
    
    mean_cols = ['Global_reactive_power', 'Voltage', 'Global_intensity']
    df_hourly_means = df_imputed[mean_cols].resample('h').mean()
    
    df_hourly_vals = pd.concat([df_hourly_sums, df_hourly_means], axis=1)
    
    # Resample short-gap flags (if any minute in the hour was imputed, mark hour as imputed)
    flag_names = [f"{col}_is_short_imputed" for col in ALL_COLS]
    df_hourly_flags = df_imputed[flag_names].resample('h').max().fillna(0).astype(int)
    
    df_hourly = pd.concat([df_hourly_vals, df_hourly_flags], axis=1)
    
    # Seasonal Imputation for hourly long gaps (NaNs)
    df_hourly = seasonal_impute_hourly(df_hourly)
    
    logger.info(f"Preprocessing completed. Hourly shape: {df_hourly.shape}")
    return df_hourly
