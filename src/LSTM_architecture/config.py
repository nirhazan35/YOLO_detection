"""
LSTM Architecture Configuration

Configuration parameters for the LSTM model architecture.
"""

import os

# Base paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_PATH = os.path.join(PROJECT_ROOT, "data")
MODELS_PATH = os.path.join(PROJECT_ROOT, "models")

# Create models directory if it doesn't exist
os.makedirs(MODELS_PATH, exist_ok=True)

# Model configuration
MODEL_CONFIG = {
    # Feature dimensions
    "rgb_feature_dim": 281,  # 25 (object features) + 256 (spatial features)
    "flow_feature_dim": 128,
    "combined_feature_dim": 409,  # 281 + 128
    
    # LSTM parameters
    "hidden_dim": 256,
    "num_layers": 2,
    "bidirectional": True,
    "dropout": 0.3,
    
    # Attention parameters
    "num_attention_heads": 4,
    
    # Sequence parameters
    "sequence_length": 16,  # Number of frames per sequence
    
    # Output parameters
    "output_dim": 1,  # Binary classification (accident/non-accident)
    
    # Model type
    "model_type": "Enhanced",  # "Basic" or "Enhanced"
}

# Model file paths
MODEL_PATHS = {
    "basic": os.path.join(MODELS_PATH, "accident_lstm_basic.pth"),
    "enhanced": os.path.join(MODELS_PATH, "accident_lstm_enhanced.pth"),
    "best_model": os.path.join(MODELS_PATH, "accident_lstm_best.pth"),
}

# Input feature paths
FEATURE_PATHS = {
    "combined": os.path.join(DATA_PATH, "features", "combined"),
    "rgb": os.path.join(DATA_PATH, "features", "rgb"),
    "flow": os.path.join(DATA_PATH, "features", "flow"),
}

# Visualization paths
VIS_PATH = os.path.join(PROJECT_ROOT, "visualizations", "lstm")
os.makedirs(VIS_PATH, exist_ok=True) 