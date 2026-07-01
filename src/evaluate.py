"""
Evaluation module.
Computes point and quantile forecast metrics.
Implements detailed error analysis and robustness checks grouped by temporal and load dimensions.
"""

import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any
from sklearn.metrics import r2_score
from src.config import HORIZON, ROBUSTNESS_PATH

logger = logging.getLogger(__name__)


def calculate_metrics(actuals: np.ndarray, predictions: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Computes MAE, RMSE, MAPE, and R2 metrics.
    """
    mae = np.mean(np.abs(actuals - predictions))
    rmse = np.sqrt(np.mean((actuals - predictions)**2))
    
    # Avoid division by zero
    actuals_safe = np.where(actuals == 0, 1e-5, actuals)
    mape = np.mean(np.abs((actuals - predictions) / actuals_safe)) * 100
    
    r2 = r2_score(actuals.flatten(), predictions.flatten())
    
    return float(mae), float(rmse), float(mape), float(r2)


def calculate_pinball_loss(actuals: np.ndarray, predictions: np.ndarray, q: float) -> float:
    """
    Computes Pinball Loss for a given quantile q.
    """
    diff = actuals - predictions
    loss = np.where(diff >= 0, q * diff, (q - 1) * diff)
    return float(np.mean(loss))


def calculate_coverage_rate(actuals: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """
    Computes the percentage of actuals falling within the lower and upper bounds.
    """
    within_bounds = (actuals >= lower) & (actuals <= upper)
    return float(np.mean(within_bounds)) * 100


def get_season(month: int) -> str:
    """
    Maps month integer to season string.
    """
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"


def perform_robustness_checks(
    test_indices: pd.DatetimeIndex,
    actuals: np.ndarray,
    predictions: np.ndarray,
    train_median_load: float
) -> Dict[str, Any]:
    """
    Groups forecast errors (MAE, RMSE, MAPE) by:
    - Hour of day (0-23)
    - Day type (Weekday vs. Weekend)
    - Season (Winter, Spring, Summer, Autumn)
    - Load level (Low-load vs. High-load based on train_median_load)
    Saves results to robustness_checks.json.
    """
    logger.info("Performing model robustness checks (error analysis)...")
    
    records = []
    
    # Flatten horizons to create a flat dataframe of prediction timestamps, actuals, and predictions
    # This represents evaluations step-by-step
    for idx, t in enumerate(test_indices):
        for h in range(1, HORIZON + 1):
            target_t = t + pd.Timedelta(hours=h)
            actual_val = actuals[idx, h - 1]
            pred_val = predictions[idx, h - 1]
            
            records.append({
                'target_time': target_t,
                'hour': target_t.hour,
                'dayofweek': target_t.dayofweek,
                'month': target_t.month,
                'actual': actual_val,
                'pred': pred_val,
                'ae': abs(actual_val - pred_val),
                'se': (actual_val - pred_val)**2
            })
            
    df_eval = pd.DataFrame(records)
    
    # Grouping helpers
    df_eval['day_type'] = np.where(df_eval['dayofweek'] >= 5, "Weekend", "Weekday")
    df_eval['season'] = df_eval['month'].apply(get_season)
    df_eval['load_level'] = np.where(df_eval['actual'] <= train_median_load, "Low-Load (<=Median)", "High-Load (>Median)")
    
    robustness = {}
    
    # 1. Hour of Day Breakdown
    by_hour = {}
    for hour, group in df_eval.groupby('hour'):
        mae = group['ae'].mean()
        rmse = np.sqrt(group['se'].mean())
        mape = (group['ae'] / np.where(group['actual'] == 0, 1e-5, group['actual'])).mean() * 100
        by_hour[int(hour)] = {'MAE': float(mae), 'RMSE': float(rmse), 'MAPE': float(mape)}
    robustness['hour_of_day'] = by_hour
    
    # 2. Day Type Breakdown
    by_day_type = {}
    for dt, group in df_eval.groupby('day_type'):
        mae = group['ae'].mean()
        rmse = np.sqrt(group['se'].mean())
        mape = (group['ae'] / np.where(group['actual'] == 0, 1e-5, group['actual'])).mean() * 100
        by_day_type[str(dt)] = {'MAE': float(mae), 'RMSE': float(rmse), 'MAPE': float(mape)}
    robustness['day_type'] = by_day_type
    
    # 3. Season Breakdown
    by_season = {}
    for season, group in df_eval.groupby('season'):
        mae = group['ae'].mean()
        rmse = np.sqrt(group['se'].mean())
        mape = (group['ae'] / np.where(group['actual'] == 0, 1e-5, group['actual'])).mean() * 100
        by_season[str(season)] = {'MAE': float(mae), 'RMSE': float(rmse), 'MAPE': float(mape)}
    robustness['season'] = by_season
    
    # 4. Load Level Breakdown
    by_load_level = {}
    for level, group in df_eval.groupby('load_level'):
        mae = group['ae'].mean()
        rmse = np.sqrt(group['se'].mean())
        mape = (group['ae'] / np.where(group['actual'] == 0, 1e-5, group['actual'])).mean() * 100
        by_load_level[str(level)] = {'MAE': float(mae), 'RMSE': float(rmse), 'MAPE': float(mape)}
    robustness['load_level'] = by_load_level
    
    # Write to file
    with open(ROBUSTNESS_PATH, 'w') as f:
        json.dump(robustness, f, indent=4)
    logger.info(f"Robustness metrics saved to {ROBUSTNESS_PATH}")
    
    return robustness
