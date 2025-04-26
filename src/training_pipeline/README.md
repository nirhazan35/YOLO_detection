# Training Pipeline for Road Accident Detection

This module contains the training pipeline for LSTM-based road accident detection models.

## Overview

The training pipeline includes components for:
- Data loading and preprocessing
- Training LSTM models
- Model evaluation
- Result visualization

## Components

### Data Loader

The `data_loader.py` module handles:
- Loading feature files from disk
- Creating training/validation/test splits
- Normalizing features
- Creating PyTorch DataLoaders
- Handling class imbalance with weighted sampling

### Trainer

The `trainer.py` module provides:
- The `AccidentLSTMTrainer` class for model training
- Training and validation loops
- Learning rate scheduling
- Early stopping
- Checkpoint saving
- TensorBoard integration
- Evaluation metrics calculation

### Evaluation

The `evaluate.py` module includes:
- Model loading utilities
- Time-to-detection (TTD) evaluation
- Visualization functions for:
  - ROC and Precision-Recall curves
  - Confusion matrices
  - Training history plots
  - Test result summaries

### Configuration

The `config.py` file contains:
- Training hyperparameters
- Data split ratios
- Learning rate scheduler settings
- File paths for model and result saving

## Usage

To train a model with default settings:

```bash
python src/training_pipeline/train.py
```

Command-line options:

- `--model_type`: Type of model to train (`Basic` or `Enhanced`)
- `--epochs`: Number of training epochs
- `--batch_size`: Training batch size
- `--feature_dir`: Custom feature directory
- `--model_dir`: Custom model output directory
- `--no_early_stopping`: Disable early stopping

## Output

The training pipeline creates an output directory for each run with:
- Model checkpoints (best and last)
- Training history
- Test metrics
- Visualizations
- Configuration files 