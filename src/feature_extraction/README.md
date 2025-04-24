# YOLO11 Feature Extraction

This module extracts features from processed video frames using the YOLO11 model. The features include both object detection features (bounding boxes and confidence scores) and spatial features extracted from the YOLO backbone.

## Features Extracted

The feature extractor outputs two types of features for each video frame:

1. **Object Detection Features**: 
   - Top 5 objects detected (prioritizing cars, pedestrians, trucks, motorcycles, and buses)
   - For each object: normalized bounding box coordinates (x, y, width, height) and confidence score
   - Shape: (N, 5, 5) where N is the number of frames

2. **Spatial Features**:
   - 256-dimensional feature vector extracted from the YOLO backbone
   - Represents spatial and semantic information about the entire scene
   - Shape: (N, 256) where N is the number of frames

## Configuration

All parameters are managed through the `config.py` file:

- `YOLO_CONFIG`: Paths to model weights, processed data, and output directory
- `FEATURE_CONFIG`: Classes to detect, maximum objects per frame, and spatial feature dimension
- `GPU_CONFIG`: GPU settings including whether to use GPU and mixed precision

To modify any settings, simply edit the `config.py` file.

## Usage

### Setup and Run

The easiest way to run the feature extraction is using the setup script:

```bash
./src/feature_extraction/setup_and_run.sh
```

This script will:
1. Create a virtual environment if it doesn't exist
2. Install all necessary dependencies
3. Download the YOLO11n model
4. Run a test to verify the feature extraction works
5. Run the full feature extraction pipeline on the processed data

### Manual Execution

If you prefer to run the steps manually:

1. Install dependencies:
```bash
pip install torch torchvision ultralytics opencv-python numpy
```

2. Run the feature extraction:
```bash
python src/feature_extraction/run_feature_extraction.py
```

## Output

The extracted features are saved as NumPy arrays (.npy files) in the output directory specified in the config file, with the following structure:

```
data/features/
├── train/
│   ├── accident/
│   │   ├── video1_features.npy
│   │   └── ...
│   └── non_accident/
│       ├── video1_features.npy
│       └── ...
├── val/
└── test/
```

Each .npy file contains a numpy array of shape (N, 281), where:
- N is the number of frames in the video
- 281 = 25 (object features: 5 objects × 5 features) + 256 (spatial features)

## Testing

To test the feature extraction without running the full pipeline:

```bash
python src/feature_extraction/test_feature_extraction.py
```

This will:
1. Download the YOLO11n model if it doesn't exist
2. Test feature extraction on a sample frame
3. Save visualization and extracted features to the test_results directory 