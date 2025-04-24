# Data Processing Module

This module handles the preprocessing and preparation of video data for road accident detection.

## Overview

The data processing pipeline takes raw accident and non-accident videos and processes them into a standardized format suitable for feature extraction and model training. The pipeline includes:

1. Video preprocessing (resolution standardization, FPS normalization)
2. Frame extraction
3. Optical flow computation
4. Data validation

## Components

- **dataset_creator.py**: Main entry point that orchestrates the entire data processing pipeline
- **data_preprocessing.py**: Handles video standardization and frame extraction
- **optical_flow.py**: Computes optical flow between consecutive frames to capture motion information
- **data_validation.py**: Validates processed data for quality and integrity
- **config.py**: Configuration parameters for the data processing pipeline
- **data_augmentation.py**: Data augmentation functionality (currently disabled)

## Usage

The processing pipeline can be run using the main runner script:

```bash
python src/run_data_processing.py
```

## Configuration

Modify parameters in `config.py` to adjust the behavior of the data processing:

- **DATA_PATH**: Input and output data paths
- **VIDEO_CONFIG**: Video processing parameters (resolution, FPS, etc.)
- **RANDOM_SEED**: Seed for reproducibility
- **KEEP_ORIGINALS**: Whether to keep original preprocessed videos

## Output Structure

For each processed video, the following outputs are generated:

- `{video_name}_frames/`: Directory containing extracted video frames
- `{video_name}_flow/`: Directory containing optical flow frames
- `{video_name}_metadata.json`: Metadata about the processed video

## Notes

- Videos under 4 seconds maintain their original length
- The pipeline processes accident and non-accident videos separately
- Processing progress is saved periodically and can be resumed if interrupted 