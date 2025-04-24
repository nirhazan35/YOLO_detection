"""
Data Processing Configuration File

This file contains all configuration parameters for the data processing pipeline.
Modify the parameters here instead of using command-line arguments.
"""

# Paths
DATA_PATH = {
    # Input data paths
    'accidents': './data/accidents',
    'non_accidents': './data/non_accidents',
    
    # Output data paths - parent directory for processed data
    'processed_parent': './data/processed',
    
    # Automatically derived output paths (don't change these)
    'processed_accidents': './data/processed/accidents',
    'processed_non_accidents': './data/processed/non_accidents',
    
    # Temporary directory for intermediate processing
    'temp_dir': './data/processed/temp'
}

# Video processing parameters
VIDEO_CONFIG = {
    'target_resolution': (640, 480),  # Width, height
    'target_fps': 15,                 # Frames per second
    'clip_duration': 4,               # seconds
    'num_frames': 16                  # Extract 16 frames per video
}

# Data split
SPLIT_RATIO = {
    'train': 0.7,
    'val': 0.2,
    'test': 0.1
}

# Processing parallelization
PARALLEL_PROCESSING = {
    'num_workers': None,              # None for auto-detection based on system
    'checkpoint_interval': 20,        # Save checkpoint after this many videos
    'batch_size': 4                   # Process frames in batches of this size
}

# GPU settings
GPU_CONFIG = {
    'use_gpu': None,                 # None for auto-detection, True to force GPU, False to force CPU
    'gpu_id': 0                      # Used if multiple GPUs are available
}

# Validation thresholds
VALIDATION = {
    'frame_variance_threshold': 10,  # Minimum variance for a valid frame
    'sequence_validity_ratio': 0.75  # Minimum ratio of valid frames for a valid sequence
}

# Random seed for reproducibility
RANDOM_SEED = 42

# Keep original preprocessed videos (useful for debugging)
KEEP_ORIGINALS = True