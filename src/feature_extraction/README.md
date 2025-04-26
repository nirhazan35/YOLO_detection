# Feature Extraction Module

This module provides functionality for extracting features from pre-processed video frames for road accident detection.

## Overview

The feature extraction module extracts two main types of features:

1. **RGB Frame Features** (using YOLO11):
   - **Object Detection Features**: Bounding boxes and confidence scores for road-related objects (cars, pedestrians, trucks, bicycles, motorcycles, buses, and traffic lights).
   - **Spatial Features**: High-dimensional feature vectors extracted from YOLO11's backbone network.

2. **Optical Flow Features**:
   - **Motion Features**: Statistical and CNN-based features extracted from optical flow frames.

## Components

- `feature_extractor.py`: YOLO11-based feature extractor for RGB frames
- `flow_feature_extractor.py`: Feature extractors for optical flow frames
- `utils.py`: Helper functions for visualization and data handling
- `config.py`: Configuration settings for the feature extraction

## Architecture

The system uses a hybrid approach as follows:

1. **YOLO Branch** (Objects + Spatial Context):
   - Extracts object detection features (bounding boxes, class probabilities)
   - Gets backbone features for spatial context information

2. **Optical Flow Branch** (Motion Patterns):
   - Processes pre-computed optical flow frames
   - Extracts motion features using a simple CNN and statistical methods

3. **Feature Combination**:
   - Combines RGB features and flow features for a complete representation
   - Features can be used separately or combined for downstream tasks

## Usage

### Working with Processed Frames

```python
from src.feature_extraction import YOLO11FeatureExtractor, SimpleFlowFeatureExtractor
import cv2

# Initialize extractors
yolo_extractor = YOLO11FeatureExtractor(model_path="yolo11m.pt")
flow_extractor = SimpleFlowFeatureExtractor()

# Extract features from RGB frame
rgb_frame = cv2.imread("frame_00.jpg")
rgb_features = yolo_extractor.extract_features(rgb_frame)

# Extract features from optical flow frame
flow_frame = cv2.imread("flow_00.jpg")
flow_features = flow_extractor.extract_features(flow_frame)

# Combine features
combined_features = np.concatenate([rgb_features, flow_features])
```

### Saving and Loading Features

```python
from src.feature_extraction.utils import save_features, load_features

# Save features to file
metadata = {"video_name": "example.mp4", "category": "accident"}
save_features(features, "features.npz", metadata)

# Load features from file
loaded_features, loaded_metadata = load_features("features.npz")
```

### Visualization

```python
from src.feature_extraction.utils import visualize_detection

# Visualize detections on a frame
object_features = rgb_features[:25]  # First 25 values are object features
visualize_detection(frame, object_features, save_path="detection.jpg")

# Visualize feature statistics
from src.feature_extraction.utils import visualize_feature_statistics
visualize_feature_statistics(features, save_dir="visualizations")
```

## Feature Format

Each video produces three types of feature files:

1. **RGB Features**: 
   - Object detection features (5 objects × 5 values per object = 25 values)
   - Spatial features (256-dimensional vector)
   - Total: 281 dimensions per frame

2. **Flow Features**:
   - Motion statistics and patterns (128-dimensional vector)

3. **Combined Features**:
   - Concatenation of RGB and flow features (409 dimensions per frame)

## Running the Feature Extraction

To extract features from pre-processed frames, run:

```
python src/run_feature_extraction.py
```

This will:
- Extract features from RGB frames in `data/processed/{category}/{video}/frames/`
- Extract features from flow frames in `data/processed/{category}/{video}/flow/`
- Save separate feature files for RGB, flow, and combined features
- Generate visualizations for sample frames

## Configuration

You can modify settings in `config.py`:

- YOLO model settings (confidence threshold, IoU threshold)
- Feature output settings (directories, file formats)
- Feature dimensions

## Requirements

- Python 3.8+
- OpenCV
- NumPy
- PyTorch
- Ultralytics (for YOLO11) 