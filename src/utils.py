"""
Utility module.
Handles logging configuration, experiment tracking exports (JSON), and model serialization (pickle).
"""

import os
import json
import pickle
import logging
from datetime import datetime
from typing import Dict, Any
from src.config import OUTPUT_DIR, METRICS_PATH, CONFIG_PATH

logger = logging.getLogger(__name__)


def setup_project_logging():
    """
    Sets up logging to write to both stdout and outputs/training.log.
    """
    log_file = os.path.join(OUTPUT_DIR, "training.log")
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers
    if not root_logger.handlers:
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
    logger.info(f"Logging initialized. Log file: {log_file}")


def save_serialized_model(model_obj: Any, filename: str):
    """
    Saves a python object using pickle.
    """
    model_path = os.path.join(OUTPUT_DIR, "models", filename)
    with open(model_path, 'wb') as f:
        pickle.dump(model_obj, f)
    logger.info(f"Serialized model saved to {model_path}")


def load_serialized_model(filename: str) -> Any:
    """
    Loads a python object using pickle.
    """
    model_path = os.path.join(OUTPUT_DIR, "models", filename)
    with open(model_path, 'rb') as f:
        obj = pickle.load(f)
    logger.info(f"Loaded serialized model from {model_path}")
    return obj


def track_experiment(
    metrics: Dict[str, Any],
    hyperparameters: Dict[str, Any],
    features_used: int,
    training_time: float
):
    """
    Saves experiment configurations and metrics to metrics.json and config.json.
    Also creates a detailed run-specific record for experiment tracking history.
    """
    logger.info("Recording experiment configuration and results...")
    
    # 1. Save config.json
    config_record = {
        'timestamp': datetime.now().isoformat(),
        'hyperparameters': hyperparameters,
        'num_features': features_used,
        'training_time_seconds': training_time
    }
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config_record, f, indent=4)
    logger.info(f"Configuration details saved to {CONFIG_PATH}")
    
    # 2. Save metrics.json
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Evaluation metrics saved to {METRICS_PATH}")
    
    # 3. Create run history folder and save detailed experiment record
    exp_dir = os.path.join(OUTPUT_DIR, "experiments")
    os.makedirs(exp_dir, exist_ok=True)
    
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_path = os.path.join(exp_dir, f"run_{run_timestamp}.json")
    
    run_record = {
        'config': config_record,
        'metrics': metrics
    }
    with open(run_path, 'w') as f:
        json.dump(run_record, f, indent=4)
    logger.info(f"Historical run trace saved to {run_path}")
