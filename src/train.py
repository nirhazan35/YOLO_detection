"""
LSTM Training Script for Road Accident Detection

This script trains an LSTM model for road accident detection using
pre-extracted features from video frames.

Usage:
    python src/train.py
"""

import os
import sys
import torch
import numpy as np
import logging
import json
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("train_lstm.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    # Add project root directory to path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
        logger.info(f"Added project root to path: {parent_dir}")
    
    # Use hardcoded default values
    model_type = "Enhanced"  # "Basic" or "Enhanced"
    epochs = 50
    batch_size = 32
    feature_dir = None  # Will use default from config
    model_dir = None    # Will use default from config
    early_stopping = True
    
    # Import after path setup to avoid import errors
    try:
        from src.LSTM_architecture.model import AccidentLSTM, EnhancedAccidentLSTM
        from src.LSTM_architecture.config import MODEL_CONFIG, MODEL_PATHS, FEATURE_PATHS
        from src.training_pipeline.config import TRAIN_CONFIG, SAVE_PATHS
        from src.training_pipeline.data_loader import create_data_splits, create_data_loaders
        from src.training_pipeline.trainer import AccidentLSTMTrainer
        from src.training_pipeline.evaluate import plot_training_history, plot_detection_curves, plot_confusion_matrix, visualize_test_results
        
        logger.info("Successfully imported required modules")
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        return 1
    
    # Update configuration with hardcoded values
    if feature_dir:
        logger.info(f"Using custom feature directory: {feature_dir}")
    else:
        feature_dir = FEATURE_PATHS["combined"]
        logger.info(f"Using default feature directory: {feature_dir}")
    
    if model_dir:
        logger.info(f"Using custom model directory: {model_dir}")
    else:
        model_dir = SAVE_PATHS["model_dir"]
        logger.info(f"Using default model directory: {model_dir}")
    
    TRAIN_CONFIG["num_epochs"] = epochs
    TRAIN_CONFIG["batch_size"] = batch_size
    logger.info(f"Training for {epochs} epochs with batch size {batch_size}")
    
    if not early_stopping:
        TRAIN_CONFIG["early_stopping_patience"] = float('inf')
        logger.info("Early stopping disabled")
    
    # Set up timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(model_dir, f"{model_type.lower()}_run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Update save paths for this run
    run_save_paths = {
        "model_dir": run_dir,
        "best_model": os.path.join(run_dir, "best_model.pth"),
        "last_model": os.path.join(run_dir, "last_model.pth"),
        "results": os.path.join(run_dir, "results.json"),
        "tensorboard_dir": os.path.join(SAVE_PATHS["tensorboard_dir"], f"{model_type.lower()}_run_{timestamp}"),
        "vis_dir": os.path.join(run_dir, "visualizations")
    }
    
    # Create visualization directory
    os.makedirs(run_save_paths["vis_dir"], exist_ok=True)
    
    # Create and initialize model
    logger.info(f"Creating {model_type} LSTM model")
    if model_type == "Basic":
        model = AccidentLSTM(
            input_dim=MODEL_CONFIG["combined_feature_dim"],
            hidden_dim=MODEL_CONFIG["hidden_dim"],
            output_dim=MODEL_CONFIG["output_dim"],
            num_layers=MODEL_CONFIG["num_layers"]
        )
    else:  # Enhanced
        model = EnhancedAccidentLSTM(
            rgb_dim=MODEL_CONFIG["rgb_feature_dim"],
            flow_dim=MODEL_CONFIG["flow_feature_dim"],
            hidden_dim=MODEL_CONFIG["hidden_dim"],
            output_dim=MODEL_CONFIG["output_dim"],
            num_layers=MODEL_CONFIG["num_layers"]
        )
    
    # Load data
    logger.info(f"Loading and preparing data from {feature_dir}")
    try:
        data_dict = create_data_splits(
            feature_dir,
            test_size=TRAIN_CONFIG["test_size"],
            val_size=TRAIN_CONFIG["val_size"],
            seq_length=MODEL_CONFIG["sequence_length"],
            random_state=TRAIN_CONFIG["random_state"]
        )
        
        logger.info(f"Data loaded: {len(data_dict['X_train'])} training, "
                  f"{len(data_dict['X_val'])} validation, {len(data_dict['X_test'])} test sequences")
        
        # Create data loaders
        data_loaders = create_data_loaders(
            data_dict,
            batch_size=TRAIN_CONFIG["batch_size"],
            weighted_sampling=TRAIN_CONFIG["weighted_sampling"]
        )
        
        logger.info("Created data loaders")
        
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return 1
    
    # Initialize trainer
    logger.info("Initializing trainer")
    trainer = AccidentLSTMTrainer(
        model=model,
        config=TRAIN_CONFIG,
        save_paths=run_save_paths
    )
    
    # Save configurations for reference
    with open(os.path.join(run_dir, "model_config.json"), 'w') as f:
        json.dump(MODEL_CONFIG, f, indent=4)
    
    with open(os.path.join(run_dir, "train_config.json"), 'w') as f:
        json.dump({k: str(v) if isinstance(v, torch.device) else v 
                   for k, v in TRAIN_CONFIG.items()}, f, indent=4)
    
    # Train model
    logger.info("Starting training")
    try:
        training_history = trainer.train(
            data_loaders["train_loader"],
            data_loaders["val_loader"]
        )
        
        # Log training completion
        logger.info(f"Training completed for {len(training_history['epochs'])} epochs")
        logger.info(f"Best validation loss: {trainer.best_val_loss:.4f} at epoch {trainer.best_epoch}")
        
        # Load best model for testing
        trainer.load_best_model()
        
        # Evaluate on test set
        logger.info("Evaluating on test set")
        test_metrics = trainer.test(data_loaders["test_loader"])
        logger.info(f"Test metrics: {test_metrics}")
        
        # Save results
        trainer.save_results(test_metrics)
        
        # Create visualizations
        logger.info("Creating visualizations")
        
        # Training history plot
        plot_training_history(training_history, run_save_paths["vis_dir"])
        
        # Get test predictions for curves
        model.eval()
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for data, targets in data_loaders["test_loader"]:
                data = data.to(TRAIN_CONFIG["device"])
                outputs = model(data)
                all_outputs.append(outputs.cpu())
                all_targets.append(targets.cpu())
        
        all_outputs = torch.sigmoid(torch.cat(all_outputs, dim=0)).numpy()
        all_targets = torch.cat(all_targets, dim=0).numpy()
        
        # Plot ROC and PR curves
        plot_detection_curves(all_targets, all_outputs, run_save_paths["vis_dir"])
        
        # Plot confusion matrix
        pred_binary = (all_outputs > 0.5).astype(int)
        plot_confusion_matrix(all_targets, pred_binary, run_save_paths["vis_dir"])
        
        # Visualize test results
        visualize_test_results(test_metrics, run_save_paths["vis_dir"])
        
        logger.info(f"All results and visualizations saved to {run_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.exception("Stack trace:")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 