"""
Configuration module for the day-ahead electricity load forecaster.
Defines folder structure, splits, features, and model settings.
"""

import os

# Root directories
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.dirname(PROJECT_DIR)

# File Paths
DATA_PATH = os.path.join(WORKSPACE_DIR, "individual+household+electric+power+consumption", "household_power_consumption.txt")
OUTPUT_DIR = os.path.join(WORKSPACE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

METRICS_PATH = os.path.join(OUTPUT_DIR, "metrics.json")
CONFIG_PATH = os.path.join(OUTPUT_DIR, "config.json")
ROBUSTNESS_PATH = os.path.join(OUTPUT_DIR, "robustness_checks.json")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Datetime splits
TRAIN_START = "2006-12-16 00:00:00"
VAL_START = "2009-07-01 00:00:00"
TEST_START = "2010-01-01 00:00:00"

# Target and Exogenous columns
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

# Forecast settings
HORIZON = 24
QUANTILES = [0.05, 0.95]

# Hyperparameter Tuning Settings
TUNING_GRID = {
    'learning_rate': [0.03, 0.05, 0.1],
    'max_depth': [4, 6],
    'num_leaves': [15, 31],
    'colsample_bytree': [0.8],
    'subsample': [0.8]
}
TUNING_HORIZONS = [1, 12, 24]  # Tune hyperparameters on a subset of horizons to save time
