# Paths
YOLO_CONFIG = {
    'model_path': 'yolo11n.pt',  # Path to YOLO11 weights
    'processed_data': './data/processed',  # Processed video frames
    'features_output': './data/features'   # Output directory for features
}

# Feature extraction parameters
FEATURE_CONFIG = {
    'object_classes': ['car', 'person', 'truck', 'motorcycle', 'bus'],
    'max_objects': 5,  # Maximum number of objects per frame
    'spatial_dim': 256  # Dimension of spatial features
}

# GPU settings
GPU_CONFIG = {
    'use_gpu': True,
    'gpu_id': 0,
    'mixed_precision': True
} 