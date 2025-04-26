"""
Training Pipeline Module for Road Accident Detection

This module contains the training pipeline for the LSTM-based
accident detection model, including data loading, training,
evaluation, and visualization components.
"""

from src.training_pipeline.data_loader import create_data_splits, create_data_loaders, AccidentDataset
from src.training_pipeline.trainer import AccidentLSTMTrainer
from src.training_pipeline.evaluate import load_model, evaluate_time_to_detection, plot_detection_curves, plot_confusion_matrix
from src.training_pipeline.config import TRAIN_CONFIG, SAVE_PATHS 