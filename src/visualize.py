"""
Visualization module.
Generates and saves diagnostic plots for forecasting results:
1. Forecast comparison with prediction intervals.
2. Averaged feature importances across the 24 horizons.
3. Robustness error analysis plots.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List
from src.config import OUTPUT_DIR

logger = logging.getLogger(__name__)


def plot_forecast_comparison(
    test_indices: pd.DatetimeIndex,
    actuals: np.ndarray,
    predictions: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    daily_naive: np.ndarray,
    plot_start: str = "2010-01-08 00:00:00",
    plot_end: str = "2010-01-14 23:00:00"
):
    """
    Plots actual load, LightGBM forecasts, Daily Naive baseline, and the 90% confidence interval band
    over a selected out-of-sample test week.
    """
    logger.info(f"Generating comparison plot from {plot_start} to {plot_end}...")
    
    # Identify indices falling inside the plot window
    plot_mask = (test_indices >= plot_start) & (test_indices <= plot_end)
    indices = np.where(plot_mask)[0]
    
    if len(indices) == 0:
        logger.warning("No test samples found in the requested plot timeframe. Skipping forecast plot.")
        return
        
    t_axis = test_indices[indices]
    
    # We plot the 1-hour-ahead forecasts (h=1) for visibility
    y_true = actuals[indices, 0]
    y_pred = predictions[indices, 0]
    y_lower = lower_bound[indices, 0]
    y_upper = upper_bound[indices, 0]
    y_naive = daily_naive[indices, 0]
    
    plt.figure(figsize=(14, 7), dpi=100)
    
    # Plot Confidence Band
    plt.fill_between(t_axis, y_lower, y_upper, color='#ff7f0e', alpha=0.15, label='90% Confidence Interval')
    
    # Plot curves
    plt.plot(t_axis, y_true, label='Actual Load', color='#2b2b2b', linewidth=2)
    plt.plot(t_axis, y_pred, label='LightGBM Forecaster (t+1)', color='#ff7f0e', linestyle='--', alpha=0.9)
    plt.plot(t_axis, y_naive, label='Daily Naive Baseline (t+1)', color='#1f77b4', linestyle=':', alpha=0.8)
    
    plt.title('Day-Ahead Hourly Load Forecast & 90% Prediction Intervals', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Datetime', fontsize=12)
    plt.ylabel('Active Power (kW)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(frameon=True, facecolor='white', edgecolor='none', shadow=True, fontsize=11, loc='upper right')
    plt.tight_layout()
    
    plot_path = os.path.join(OUTPUT_DIR, "forecast_comparison_intervals.png")
    plt.savefig(plot_path)
    logger.info(f"Forecast comparison plot saved to {plot_path}")
    plt.close()


def plot_feature_importance(models: Dict[int, Any], feature_names: List[str]):
    """
    Computes and plots the average feature importance (by gain/split) across all 24 point models.
    """
    logger.info("Computing and plotting averaged feature importances...")
    
    # Alternate clean approach: Aggregate feature importances into a pandas Series
    feat_imp_df = pd.DataFrame()
    for h, model in models.items():
        df_temp = pd.DataFrame({
            'feature': model.feature_name_,
            f'importance_h{h}': model.feature_importances_
        })
        if feat_imp_df.empty:
            feat_imp_df = df_temp
        else:
            feat_imp_df = feat_imp_df.merge(df_temp, on='feature', how='outer')
            
    # Calculate mean importance across all horizons
    imp_cols = [c for c in feat_imp_df.columns if c != 'feature']
    feat_imp_df['mean_importance'] = feat_imp_df[imp_cols].mean(axis=1)
    feat_imp_df.sort_values('mean_importance', ascending=True, inplace=True)
    
    # Plot top 20 features (Industry Standard request)
    top_n = feat_imp_df.tail(20)
    
    plt.figure(figsize=(10, 7), dpi=100)
    plt.barh(top_n['feature'], top_n['mean_importance'], color='#ff7f0e', alpha=0.85, edgecolor='none')
    plt.title('Averaged LightGBM Feature Importance (Across 24 Horizons)', fontsize=12, fontweight='bold', pad=15)
    plt.xlabel('Importance Value (Split Count)', fontsize=11)
    plt.ylabel('Features', fontsize=11)
    plt.grid(True, axis='x', linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    plot_path = os.path.join(OUTPUT_DIR, "feature_importance.png")
    plt.savefig(plot_path)
    logger.info(f"Feature importance plot saved to {plot_path}")
    plt.close()


def plot_residuals(
    test_indices: pd.DatetimeIndex,
    actuals: np.ndarray,
    predictions: np.ndarray,
    plot_start: str = "2010-01-08 00:00:00",
    plot_end: str = "2010-01-14 23:00:00"
):
    """
    Plots the forecast residuals (Error = Actual - Predicted) over time for a sample test week.
    """
    logger.info(f"Generating residuals plot from {plot_start} to {plot_end}...")
    plot_mask = (test_indices >= plot_start) & (test_indices <= plot_end)
    indices = np.where(plot_mask)[0]
    
    if len(indices) == 0:
        return
        
    t_axis = test_indices[indices]
    y_true = actuals[indices, 0]
    y_pred = predictions[indices, 0]
    residuals = y_true - y_pred
    
    plt.figure(figsize=(14, 5), dpi=100)
    plt.plot(t_axis, residuals, label='Residuals (Actual - Predicted)', color='#d62728', linewidth=1.5)
    plt.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)
    
    plt.title('Forecast Residuals Over Time (t+1 Horizon)', fontsize=12, fontweight='bold', pad=15)
    plt.xlabel('Datetime', fontsize=11)
    plt.ylabel('Error (kW)', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(frameon=True, facecolor='white', edgecolor='none')
    plt.tight_layout()
    
    plot_path = os.path.join(OUTPUT_DIR, "residuals_vs_time.png")
    plt.savefig(plot_path)
    logger.info(f"Residuals plot saved to {plot_path}")
    plt.close()


def plot_error_distribution(actuals: np.ndarray, predictions: np.ndarray):
    """
    Plots a histogram of the forecasting errors (residuals) across the entire test set.
    """
    logger.info("Generating prediction error distribution histogram...")
    # Flatten both matrices to evaluate errors across all prediction points
    residuals = (actuals - predictions).flatten()
    
    plt.figure(figsize=(10, 6), dpi=100)
    # Histogram of errors
    count, bins, ignored = plt.hist(residuals, bins=60, density=True, color='#9467bd', alpha=0.75, edgecolor='white')
    
    # Kernel density estimation/normal approximation trace
    mu = np.mean(residuals)
    sigma = np.std(residuals)
    plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
             linewidth=2, color='#2b2b2b', label=f'Normal Fit (μ={mu:.3f}, σ={sigma:.3f})')
             
    plt.title('Distribution of Prediction Errors (Residuals)', fontsize=12, fontweight='bold', pad=15)
    plt.xlabel('Prediction Error (Actual - Predicted) in kW', fontsize=11)
    plt.ylabel('Density', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(frameon=True, facecolor='white', edgecolor='none')
    plt.tight_layout()
    
    plot_path = os.path.join(OUTPUT_DIR, "error_distribution.png")
    plt.savefig(plot_path)
    logger.info(f"Error distribution plot saved to {plot_path}")
    plt.close()


def plot_robustness_analysis(robustness: Dict[str, Any]):
    """
    Plots the error analysis breakdown by hour of day, season, and load level.
    """
    logger.info("Plotting robustness analysis charts...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=100)
    
    # 1. Hour of Day MAE
    hour_data = robustness['hour_of_day']
    hours = sorted(list(hour_data.keys()))
    hour_mae = [hour_data[h]['MAE'] for h in hours]
    axes[0, 0].plot(hours, hour_mae, marker='o', color='#ff7f0e', linewidth=2)
    axes[0, 0].set_title('MAE by Hour of Day', fontsize=11, fontweight='bold')
    axes[0, 0].set_xlabel('Hour (0-23)')
    axes[0, 0].set_ylabel('MAE (kW)')
    axes[0, 0].grid(True, linestyle=':', alpha=0.6)
    axes[0, 0].set_xticks(range(0, 24, 2))
    
    # 2. Day Type MAE
    day_data = robustness['day_type']
    day_types = list(day_data.keys())
    day_mae = [day_data[d]['MAE'] for d in day_types]
    axes[0, 1].bar(day_types, day_mae, color=['#1f77b4', '#ff7f0e'], alpha=0.8, width=0.4)
    axes[0, 1].set_title('MAE by Day Type', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('MAE (kW)')
    axes[0, 1].grid(True, axis='y', linestyle=':', alpha=0.6)
    
    # 3. Season MAE
    season_data = robustness['season']
    seasons = list(season_data.keys())
    season_mae = [season_data[s]['MAE'] for s in seasons]
    axes[1, 0].bar(seasons, season_mae, color='#2ca02c', alpha=0.8, width=0.4)
    axes[1, 0].set_title('MAE by Season', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('MAE (kW)')
    axes[1, 0].grid(True, axis='y', linestyle=':', alpha=0.6)
    
    # 4. Load Level MAE
    load_data = robustness['load_level']
    levels = list(load_data.keys())
    level_mae = [load_data[l]['MAE'] for l in levels]
    axes[1, 1].bar(levels, level_mae, color='#9467bd', alpha=0.8, width=0.4)
    axes[1, 1].set_title('MAE by Load Level', fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel('MAE (kW)')
    axes[1, 1].grid(True, axis='y', linestyle=':', alpha=0.6)
    
    plt.suptitle('Model Robustness Error Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plot_path = os.path.join(OUTPUT_DIR, "robustness_analysis.png")
    plt.savefig(plot_path)
    logger.info(f"Robustness analysis plot saved to {plot_path}")
    plt.close()
