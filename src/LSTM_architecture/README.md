# LSTM Architecture for Road Accident Detection

This module contains the LSTM-based architecture for detecting road accidents in video sequences.

## Overview

The architecture processes sequences of features extracted from video frames, including:
- Object detection features from YOLOv11
- Spatial features from the YOLO backbone
- Optical flow features for motion analysis

## Model Architecture

### Basic Model: `AccidentLSTM`
- Bidirectional LSTM with multiple layers
- Multi-head attention mechanism
- Temporal pooling
- Fully connected classification layers

### Enhanced Model: `EnhancedAccidentLSTM`
- Separate processing paths for RGB and flow features
- Custom temporal attention mechanisms
- Feature fusion through a gated mechanism
- Improved classification layers

## Input Structure

The model expects inputs with the following shape:
```
(batch_size, sequence_length, feature_dim)
```

Where:
- `batch_size` is the number of video clips in a batch
- `sequence_length` is the number of frames per clip (typically 16)
- `feature_dim` is the dimensionality of features per frame (409)

## Feature Composition

Each frame's feature vector (409 dimensions) consists of:
1. Object features (25 dimensions)
   - 5 objects × 5 values (x, y, width, height, confidence)
2. Spatial features (256 dimensions)
   - Features from YOLO backbone
3. Flow features (128 dimensions)
   - Features from optical flow processing

## Configuration

Model configuration can be customized in `config.py`, including:
- Network architecture parameters
- Feature dimensions
- Model file paths
- Training parameters 