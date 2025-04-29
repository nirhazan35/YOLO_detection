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
    
    # Organized output directories for frames and flow
    'accidents_frames': './data/processed/accidents/frames',
    'accidents_flow': './data/processed/accidents/flow',
    'non_accidents_frames': './data/processed/non_accidents/frames',
    'non_accidents_flow': './data/processed/non_accidents/flow',
    
    # Temporary directory for intermediate processing
    'temp_dir': './data/processed/temp'
}

# Video processing parameters
VIDEO_CONFIG = {
    'target_resolution': (640, 480),  # Width, height
    'target_fps': 15,                 # Frames per second
    'clip_duration': 4,               # seconds
    'num_frames': 16,                 # Extract 16 frames per video
    'max_video_length': 60,           # Maximum video length to process in seconds
    'temporal_subsampling': True,     # Whether to apply temporal subsampling for long videos
    'content_aware_sampling': True,   # Whether to use content-aware frame selection
    'motion_threshold': 0.05          # Minimum normalized motion for content-aware selection
}

# Processing options
PROCESSING_OPTIONS = {
    'process_frames': True,          # Whether to process and save frames
    'process_flow': True,            # Whether to process and save optical flow
    'save_metadata': True            # Whether to save metadata
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
    'gpu_id': 0,                     # Used if multiple GPUs are available
    'prefer_raft': True              # Whether to prefer RAFT optical flow when available
}

# Optical flow settings
OPTICAL_FLOW = {
    'method': 'auto',                # 'auto', 'raft', 'dis', or 'farneback'
    'farneback_levels': 3,           # Number of pyramid levels for Farneback
    'farneback_winsize': 15,         # Window size for Farneback
    'dis_preset': 'medium',          # DIS preset: 'fast', 'medium', or 'accurate'
    'visualize_flow': True           # Whether to save flow visualization images
}

# Validation thresholds
VALIDATION = {
    'frame_variance_threshold': 10,  # Minimum variance for a valid frame
    'sequence_validity_ratio': 0.75, # Minimum ratio of valid frames for a valid sequence
    'motion_consistency_check': True, # Check consistency between RGB and flow frames
    'frozen_frame_threshold': 3.0,   # Threshold for detecting frozen frames (L2 difference)
    'max_consecutive_frozen': 3      # Maximum allowed consecutive frozen frames
}

# Random seed for reproducibility
RANDOM_SEED = 42

# Keep original preprocessed videos (useful for debugging)
KEEP_ORIGINALS = True