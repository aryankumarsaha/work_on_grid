"""
Master orchestration script for the day-ahead electricity load forecaster.
Coordinates data cleaning, feature building, hyperparameter tuning, model training,
metric evaluations, robustness tests, and visualizations.
"""

import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from src.config import DATA_PATH, VAL_START, TEST_START, TARGET_COL, HORIZON
from src.preprocess import preprocess_data
from src.features import create_features_and_targets
from src.models import DayAheadLoadForecaster
from src.evaluate import calculate_metrics, calculate_pinball_loss, calculate_coverage_rate, perform_robustness_checks
from src.visualize import plot_forecast_comparison, plot_feature_importance, plot_robustness_analysis, plot_residuals, plot_error_distribution
from src.utils import setup_project_logging, save_serialized_model, track_experiment

logger = logging.getLogger(__name__)


def compute_baselines(df_hourly: pd.DataFrame, test_indices: pd.DatetimeIndex) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes Daily Naive, Weekly Naive, and 7-day Rolling Average baseline forecasts.
    Daily Naive: y_{t+h} = y_{t+h-24}
    Weekly Naive: y_{t+h} = y_{t+h-168}
    Rolling Average: y_{t+h} = mean(y_{t-167} to y_t)
    """
    logger.info("Computing baseline forecasts for test set...")
    n_test = len(test_indices)
    daily_naive = np.zeros((n_test, HORIZON))
    weekly_naive = np.zeros((n_test, HORIZON))
    rolling_avg = np.zeros((n_test, HORIZON))
    
    load_series = df_hourly[TARGET_COL]
    
    for idx, t in enumerate(test_indices):
        # Daily: same hour yesterday (t-23 to t)
        daily_naive[idx, :] = load_series.loc[t - pd.Timedelta(hours=23) : t].values
        # Weekly: same hour last week (t-167 to t-144)
        weekly_naive[idx, :] = load_series.loc[t - pd.Timedelta(hours=167) : t - pd.Timedelta(hours=144)].values
        # Rolling Average: mean of past 168 hours (7 days) ending at t
        rolling_avg[idx, :] = load_series.loc[t - pd.Timedelta(hours=167) : t].mean()
        
    return daily_naive, weekly_naive, rolling_avg


def main():
    # 1. Initialize logging
    setup_project_logging()
    logger.info("Initializing day-ahead forecaster pipeline...")
    total_start = time.time()
    
    # 2. Preprocess raw data
    df_hourly = preprocess_data(DATA_PATH)
    
    # 3. Create features and target variables
    X, Y = create_features_and_targets(df_hourly, outlier_clipping=True)
    
    # Get training median for load breakdown checks
    train_mask = X.index < VAL_START
    train_median_load = float(df_hourly.loc[df_hourly.index < VAL_START, TARGET_COL].median())
    logger.info(f"Training median load: {train_median_load:.3f} kW")
    
    # 4. Initialize forecaster
    forecaster = DayAheadLoadForecaster()
    
    # 5. Fit point and quantile regressors
    logger.info("Fitting models...")
    train_start = time.time()
    forecaster.fit(X, Y, tune=True)
    training_time = time.time() - train_start
    
    # 6. Save model object
    save_serialized_model(forecaster, "forecaster_lgb.pkl")
    
    # 7. Generate predictions on out-of-sample Test Set
    test_mask = X.index >= TEST_START
    X_test = X.loc[test_mask]
    Y_test = Y.loc[test_mask]
    
    logger.info("Generating predictions on the test set...")
    pred_start = time.time()
    point_preds, lower_preds, upper_preds = forecaster.predict(X_test)
    prediction_time = time.time() - pred_start
    
    actuals = Y_test.values
    
    # 8. Compute baselines
    daily_naive, weekly_naive, rolling_avg = compute_baselines(df_hourly, X_test.index)
    
    # 9. Calculate point forecasting metrics (including R2 score)
    mae_lgbm, rmse_lgbm, mape_lgbm, r2_lgbm = calculate_metrics(actuals, point_preds)
    mae_daily, rmse_daily, mape_daily, r2_daily = calculate_metrics(actuals, daily_naive)
    mae_weekly, rmse_weekly, mape_weekly, r2_weekly = calculate_metrics(actuals, weekly_naive)
    mae_roll, rmse_roll, mape_roll, r2_roll = calculate_metrics(actuals, rolling_avg)
    
    # 10. Calculate interval and loss metrics
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
    
    # 11. Run Robustness Checks (Error breakdown)
    robustness = perform_robustness_checks(X_test.index, actuals, point_preds, train_median_load)
    
    # 12. Create Visualization plots
    plot_forecast_comparison(X_test.index, actuals, point_preds, lower_preds, upper_preds, daily_naive)
    plot_feature_importance(forecaster.point_models, X_test.columns.tolist())
    plot_residuals(X_test.index, actuals, point_preds)
    plot_error_distribution(actuals, point_preds)
    plot_robustness_analysis(robustness)
    
    # 13. Track experiment config and metrics in JSON files
    track_experiment(metrics, forecaster.params, len(forecaster.feature_cols), training_time)
    
    # 14. Output formatted results table
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
