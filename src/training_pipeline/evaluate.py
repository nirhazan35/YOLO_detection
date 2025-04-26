"""
LSTM Model Evaluation for Road Accident Detection

This module provides functions to evaluate LSTM models for accident detection
and visualize their performance.
"""

import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix
from typing import Dict, Tuple, List, Optional, Union, Any
import seaborn as sns
import logging
from src.LSTM_architecture.model import AccidentLSTM, EnhancedAccidentLSTM
from src.training_pipeline.trainer import AccidentLSTMTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model(model_path: str, model_config: Dict[str, Any], 
               model_type: str = "Enhanced") -> torch.nn.Module:
    """
    Load a trained model from disk.
    
    Args:
        model_path: Path to model checkpoint
        model_config: Model configuration dictionary
        model_type: Type of model to load ("Basic" or "Enhanced")
        
    Returns:
        Loaded PyTorch model
    """
    # Create appropriate model
    if model_type == "Basic":
        model = AccidentLSTM(
            input_dim=model_config['combined_feature_dim'],
            hidden_dim=model_config['hidden_dim'],
            output_dim=model_config['output_dim'],
            num_layers=model_config['num_layers']
        )
    else:  # Enhanced
        model = EnhancedAccidentLSTM(
            rgb_dim=model_config['rgb_feature_dim'],
            flow_dim=model_config['flow_feature_dim'],
            hidden_dim=model_config['hidden_dim'],
            output_dim=model_config['output_dim'],
            num_layers=model_config['num_layers']
        )
    
    # Load weights
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    logger.info(f"Loaded {model_type} model from {model_path}")
    return model


def evaluate_time_to_detection(model: torch.nn.Module, test_sequences: np.ndarray, 
                              test_labels: np.ndarray, threshold: float = 0.5, 
                              device: torch.device = None) -> Dict[str, float]:
    """
    Evaluate model's time-to-detection (TTD) performance.
    
    Args:
        model: Trained accident detection model
        test_sequences: Test sequences of shape (N, seq_len, features)
        test_labels: Test labels of shape (N)
        threshold: Decision threshold
        device: Device to run inference on
        
    Returns:
        Dictionary with TTD metrics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    model.eval()
    
    # Only select accident sequences
    accident_indices = np.where(test_labels == 1)[0]
    accident_sequences = test_sequences[accident_indices]
    
    # Track detection metrics
    detection_frames = []
    
    # Process each accident sequence
    for sequence in accident_sequences:
        detected = False
        detection_frame = -1
        
        # Process frames one by one to simulate online detection
        for i in range(1, len(sequence) + 1):
            # Extract partial sequence
            partial_sequence = sequence[:i]
            
            # Pad if necessary
            if i < len(sequence):
                pad_length = len(sequence) - i
                padding = np.zeros((pad_length, sequence.shape[1]))
                partial_sequence = np.vstack([partial_sequence, padding])
            
            # Convert to tensor
            input_tensor = torch.FloatTensor(partial_sequence).unsqueeze(0).to(device)
            
            # Get prediction
            with torch.no_grad():
                output = model(input_tensor)
                prediction = torch.sigmoid(output).item()
            
            # Check if accident detected
            if prediction > threshold and not detected:
                detected = True
                detection_frame = i
                break
        
        # Store detection frame
        if detected:
            detection_frames.append(detection_frame)
        else:
            # If not detected by the end, use sequence length + 1
            detection_frames.append(len(sequence) + 1)
    
    # Calculate metrics
    avg_detection_frame = np.mean(detection_frames)
    median_detection_frame = np.median(detection_frames)
    
    # Calculate early detection rate (detect before 50% of sequence)
    early_detection_threshold = len(sequence) // 2
    early_detections = [f for f in detection_frames if f <= early_detection_threshold]
    early_detection_rate = len(early_detections) / len(detection_frames)
    
    # Calculate miss rate (never detected)
    miss_rate = detection_frames.count(len(sequence) + 1) / len(detection_frames)
    
    # Return metrics
    return {
        'avg_detection_frame': float(avg_detection_frame),
        'median_detection_frame': float(median_detection_frame),
        'early_detection_rate': float(early_detection_rate),
        'miss_rate': float(miss_rate)
    }


def plot_detection_curves(y_true: np.ndarray, y_scores: np.ndarray, save_dir: str) -> None:
    """
    Plot ROC and Precision-Recall curves.
    
    Args:
        y_true: Ground truth labels
        y_scores: Model predictions (probabilities)
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    ax1.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax1.plot([0, 1], [0, 1], 'k--', lw=2)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Receiver Operating Characteristic')
    ax1.legend(loc="lower right")
    
    # Plot Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    
    ax2.plot(recall, precision, lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(loc="lower left")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'detection_curves.png'), dpi=300)
    plt.close()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_dir: str) -> None:
    """
    Plot confusion matrix.
    
    Args:
        y_true: Ground truth labels
        y_pred: Model predictions (binary)
        save_dir: Directory to save plot
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Accident', 'Accident'],
                yticklabels=['Non-Accident', 'Accident'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()


def plot_training_history(history: Dict[str, List[float]], save_dir: str) -> None:
    """
    Plot training history.
    
    Args:
        history: Dictionary with training history
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract training data
    epochs = history['epochs']
    train_loss = history['train_loss']
    val_loss = history['val_loss']
    learning_rates = history['learning_rates']
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot training and validation loss
    ax1.plot(epochs, train_loss, 'b-', label='Training Loss')
    ax1.plot(epochs, val_loss, 'r-', label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot learning rate
    ax2.plot(epochs, learning_rates, 'g-')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate Schedule')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=300)
    plt.close()


def visualize_test_results(test_results: Dict[str, Any], save_dir: str) -> None:
    """
    Create a visualization of test results.
    
    Args:
        test_results: Dictionary with test results
        save_dir: Directory to save visualization
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create a summary figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract metrics
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc']
    values = [test_results.get(m, 0) for m in metrics]
    
    # Create bar chart
    bars = ax.bar(metrics, values, color='skyblue')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    ax.set_ylim(0, 1.1)
    ax.set_title('Test Performance Metrics')
    ax.set_ylabel('Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_dir, 'test_metrics.png'), dpi=300)
    plt.close()
    
    # Save as text file too
    with open(os.path.join(save_dir, 'test_metrics.txt'), 'w') as f:
        for metric, value in test_results.items():
            f.write(f"{metric}: {value}\n") 