"""
Feature Extraction Configuration

Configuration parameters for the feature extraction module.
"""

import os
from typing import Dict, Any

# Base paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_PATH = os.path.join(PROJECT_ROOT, "data")

# YOLO11 model configuration
YOLO_CONFIG = {
    "model_path": os.path.join(PROJECT_ROOT, "yolo11m.pt"),
    "max_objects": 5,
    "confidence_threshold": 0.25,  # Default confidence threshold
    "iou_threshold": 0.45,        # Default IoU threshold for NMS
    "device": None,               # None means auto-select (uses CUDA if available)
}

# Input data paths (already processed frames)
INPUT_PATHS = {
    "accidents": os.path.join(DATA_PATH, "processed", "accidents"),
    "non_accidents": os.path.join(DATA_PATH, "processed", "non_accidents"),
    "test": os.path.join(DATA_PATH, "processed", "test"),
}

# Feature extraction configuration
FEATURE_CONFIG = {
    "object_feature_dim": 5,      # x, y, w, h, confidence per object
    "max_objects": 5,             # Maximum objects per frame
    "spatial_feature_dim": 256,   # Spatial feature dimension from YOLO backbone
    "flow_feature_dim": 128,      # Dimension for optical flow features
}

# Output configuration
OUTPUT_CONFIG = {
    "features_dir": os.path.join(DATA_PATH, "features"),
    "rgb_features_dir": os.path.join(DATA_PATH, "features", "rgb"),
    "flow_features_dir": os.path.join(DATA_PATH, "features", "flow"),
    "combined_features_dir": os.path.join(DATA_PATH, "features", "combined"),
    "visualizations_dir": os.path.join(DATA_PATH, "visualizations"),
    "feature_format": "npz",      # Format to save features (npz, npy, or pkl)
}

# Create necessary directories
for path in [
    OUTPUT_CONFIG["features_dir"], 
    OUTPUT_CONFIG["rgb_features_dir"],
    OUTPUT_CONFIG["flow_features_dir"],
    OUTPUT_CONFIG["combined_features_dir"],
    OUTPUT_CONFIG["visualizations_dir"]
]:
    os.makedirs(path, exist_ok=True)

# Categories for road-related objects (COCO dataset classes)
ROAD_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    9: "traffic light",
}

# Directory patterns for the processed data
DIR_PATTERNS = {
    "frames": "*_frames",         # Pattern for frame directories
    "flow": "*_flow"              # Pattern for optical flow directories
}

# Frame patterns for reading
FRAME_PATTERNS = {
    "rgb": "frame_*.jpg",         # Pattern for RGB frames
    "flow": "*.jpg"               # Pattern for optical flow frames
}

# Batch processing settings
BATCH_PROCESSING = {
    "batch_size": 16,             # Number of frames to process at once
    "num_workers": None,          # None for auto-detection based on CPU count
}

# Derived values
FEATURE_DIM = {
    "rgb": (FEATURE_CONFIG["object_feature_dim"] * FEATURE_CONFIG["max_objects"] + 
           FEATURE_CONFIG["spatial_feature_dim"]),
    "flow": FEATURE_CONFIG["flow_feature_dim"],
    "combined": (FEATURE_CONFIG["object_feature_dim"] * FEATURE_CONFIG["max_objects"] + 
                FEATURE_CONFIG["spatial_feature_dim"] + FEATURE_CONFIG["flow_feature_dim"])
} 