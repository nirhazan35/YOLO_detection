# Road Accident Detection

This project implements a pipeline for detecting road accidents from video data using computer vision and deep learning techniques.

## Project Structure

The project consists of three main stages:

1. **Data Processing** - Preprocessing and validating video data
2. **Feature Extraction** - Extracting features from videos using YOLO11
3. **Accident Detection** - Training and evaluating models to detect accidents

## Data Processing

The data processing pipeline handles:
- Video preprocessing (resizing, FPS normalization)
- Frame extraction
- Optical flow computation
- Validation of processed data

To run the data processing:

```bash
python src/run_data_processing.py
```

## Feature Extraction

The feature extraction module uses YOLO11 to extract object detection and spatial features from the processed video frames.

Key features:
- Object detection features (cars, pedestrians, trucks, motorcycles, buses)
- Spatial features from YOLO's backbone
- Combined feature representation for each frame

All parameters are managed through the `config.py` file, eliminating the need for command-line arguments.

To run the feature extraction:

```bash
./src/feature_extraction/setup_and_run.sh
```

Alternatively, you can run specific components:

```bash
# Test feature extraction on a sample frame
python src/feature_extraction/test_feature_extraction.py

# Run full feature extraction pipeline
python src/feature_extraction/run_feature_extraction.py
```

See [Feature Extraction README](src/feature_extraction/README.md) for more details.

## Directory Structure

```
.
├── data/                       # Data directory
│   ├── accidents/              # Raw accident videos
│   ├── non_accidents/          # Raw non-accident videos
│   ├── processed/              # Processed video data
│   └── features/               # Extracted features
├── src/                        # Source code
│   ├── data_processing/        # Data processing modules
│   ├── feature_extraction/     # Feature extraction modules
│   │   ├── config.py           # Configuration parameters
│   │   ├── feature_extraction.py # Main feature extraction code
│   │   ├── run_feature_extraction.py # Execution script
│   │   └── test_feature_extraction.py # Testing utility
│   └── run_data_processing.py  # Data processing script
├── venv/                       # Virtual environment
└── README.md                   # Main README
```

## Getting Started

1. Clone the repository
2. Set up a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. Run the data processing:
```bash
python src/run_data_processing.py
```
4. Run the feature extraction:
```bash
./src/feature_extraction/setup_and_run.sh
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- OpenCV
- Ultralytics (for YOLO11)
- CUDA-capable GPU (recommended)
