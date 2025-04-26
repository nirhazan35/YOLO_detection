"""
Training Pipeline Configuration

Configuration parameters for the LSTM model training pipeline.
"""

import os
import torch

# Base paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
RESULTS_PATH = os.path.join(PROJECT_ROOT, "results")
LOGS_PATH = os.path.join(PROJECT_ROOT, "logs")

# Create directories if they don't exist
os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(LOGS_PATH, exist_ok=True)

# Training configuration
TRAIN_CONFIG = {
    # Data parameters
    "batch_size": 32,
    "test_size": 0.2,
    "val_size": 0.15,
    "sequence_length": 16,
    "random_state": 42,
    "weighted_sampling": True,
    
    # Training parameters
    "num_epochs": 50,
    "early_stopping_patience": 7,
    "learning_rate": 0.001,
    "weight_decay": 0.001,
    "pos_weight": 3.0,  # Weight for positive class in loss function
    "clip_grad_norm": 5.0,  # For gradient clipping
    
    # Scheduler parameters
    "scheduler_factor": 0.5,
    "scheduler_patience": 5,
    "scheduler_min_lr": 1e-6,
    
    # Device configuration
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "num_workers": 4,
    "pin_memory": True,
    
    # Logging
    "log_interval": 10,  # Log every N batches
    "tensorboard": True,
}

# Paths for saving models and results
SAVE_PATHS = {
    "model_dir": os.path.join(PROJECT_ROOT, "models"),
    "best_model": os.path.join(PROJECT_ROOT, "models", "best_model.pth"),
    "last_model": os.path.join(PROJECT_ROOT, "models", "last_model.pth"),
    "results": os.path.join(RESULTS_PATH, "results.json"),
    "tensorboard_dir": os.path.join(LOGS_PATH, "tensorboard"),
}

# Create directories if they don't exist
for path in SAVE_PATHS.values():
    if path.endswith('.pth') or path.endswith('.json'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    else:
        os.makedirs(path, exist_ok=True)

# Feature paths
FEATURE_PATHS = {
    "combined": os.path.join(PROJECT_ROOT, "data", "features", "combined"),
    "rgb": os.path.join(PROJECT_ROOT, "data", "features", "rgb"),
    "flow": os.path.join(PROJECT_ROOT, "data", "features", "flow"),
} 