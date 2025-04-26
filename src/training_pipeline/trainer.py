"""
LSTM Trainer Module for Road Accident Detection

This module handles the training, validation, and testing of LSTM models
for road accident detection.
"""

import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from typing import Dict, Tuple, List, Optional, Union, Any
import logging
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_recall_curve, roc_curve, auc, f1_score, precision_score, recall_score, accuracy_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AccidentLSTMTrainer:
    """
    Trainer class for Accident LSTM models.
    """
    def __init__(self, model: nn.Module, config: Dict[str, Any], save_paths: Dict[str, str]):
        """
        Initialize the trainer.
        
        Args:
            model: LSTM model to train
            config: Training configuration dictionary
            save_paths: Dictionary with paths for saving models and results
        """
        self.model = model
        self.config = config
        self.save_paths = save_paths
        self.device = config['device']
        
        # Move model to device
        self.model.to(self.device)
        
        # Create loss function with pos_weight for class imbalance
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([config['pos_weight']]).to(self.device)
        )
        
        # Create optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Create learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config['scheduler_factor'],
            patience=config['scheduler_patience'],
            min_lr=config['scheduler_min_lr']
        )
        
        # Initialize tensorboard if enabled
        self.writer = None
        if config['tensorboard']:
            self.writer = SummaryWriter(save_paths['tensorboard_dir'])
        
        # Initialize tracking variables
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'epochs': [],
            'learning_rates': []
        }
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train model for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        epoch_loss = 0.0
        batches_processed = 0
        
        # Use tqdm for progress bar
        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (data, targets) in enumerate(pbar):
            # Move data to device
            data = data.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config['clip_grad_norm'])
            
            # Update weights
            self.optimizer.step()
            
            # Accumulate batch loss
            epoch_loss += loss.item()
            batches_processed += 1
            
            # Update progress bar
            pbar.set_postfix({"loss": loss.item()})
            
            # Log to tensorboard
            if self.writer and batch_idx % self.config['log_interval'] == 0:
                global_step = len(train_loader) * self.training_history['epochs'][-1] if self.training_history['epochs'] else 0
                global_step += batch_idx
                self.writer.add_scalar('training/batch_loss', loss.item(), global_step)
        
        # Calculate average epoch loss
        avg_epoch_loss = epoch_loss / batches_processed if batches_processed > 0 else 0
        return avg_epoch_loss
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        val_loss = 0.0
        batches_processed = 0
        
        all_outputs = []
        all_targets = []
        
        # No gradient computation for validation
        with torch.no_grad():
            for data, targets in val_loader:
                # Move data to device
                data = data.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                
                # Accumulate batch loss
                val_loss += loss.item()
                batches_processed += 1
                
                # Store predictions and targets for metrics
                all_outputs.append(outputs.cpu())
                all_targets.append(targets.cpu())
        
        # Calculate average validation loss
        avg_val_loss = val_loss / batches_processed if batches_processed > 0 else 0
        
        # Concatenate all predictions and targets
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Convert to numpy for metric calculation
        outputs_np = torch.sigmoid(all_outputs).numpy()
        targets_np = all_targets.numpy()
        
        # Calculate metrics
        metrics = self.calculate_metrics(outputs_np, targets_np)
        metrics['val_loss'] = avg_val_loss
        
        return metrics
    
    def test(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Test the model.
        
        Args:
            test_loader: DataLoader for test data
            
        Returns:
            Dictionary with test metrics
        """
        self.model.eval()
        test_loss = 0.0
        batches_processed = 0
        
        all_outputs = []
        all_targets = []
        
        # No gradient computation for testing
        with torch.no_grad():
            for data, targets in tqdm(test_loader, desc="Testing"):
                # Move data to device
                data = data.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                
                # Accumulate batch loss
                test_loss += loss.item()
                batches_processed += 1
                
                # Store predictions and targets for metrics
                all_outputs.append(outputs.cpu())
                all_targets.append(targets.cpu())
        
        # Calculate average test loss
        avg_test_loss = test_loss / batches_processed if batches_processed > 0 else 0
        
        # Concatenate all predictions and targets
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Convert to numpy for metric calculation
        outputs_np = torch.sigmoid(all_outputs).numpy()
        targets_np = all_targets.numpy()
        
        # Calculate metrics
        metrics = self.calculate_metrics(outputs_np, targets_np)
        metrics['test_loss'] = avg_test_loss
        
        return metrics
    
    def calculate_metrics(self, outputs: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """
        Calculate evaluation metrics.
        
        Args:
            outputs: Model predictions (after sigmoid)
            targets: Ground truth labels
            
        Returns:
            Dictionary with metrics
        """
        # Binary predictions
        pred_binary = (outputs > 0.5).astype(int)
        
        # Basic metrics
        accuracy = accuracy_score(targets, pred_binary)
        precision = precision_score(targets, pred_binary, zero_division=0)
        recall = recall_score(targets, pred_binary, zero_division=0)
        f1 = f1_score(targets, pred_binary, zero_division=0)
        
        # Precision-Recall curve and AUC
        precision_curve, recall_curve, _ = precision_recall_curve(targets, outputs)
        pr_auc = auc(recall_curve, precision_curve)
        
        # ROC curve and AUC
        fpr, tpr, _ = roc_curve(targets, outputs)
        roc_auc = auc(fpr, tpr)
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'pr_auc': float(pr_auc),
            'roc_auc': float(roc_auc)
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, List[float]]:
        """
        Train the model for the specified number of epochs.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            
        Returns:
            Dictionary with training history
        """
        # Initialize lists for tracking metrics
        train_losses = []
        val_losses = []
        epochs = []
        learning_rates = []
        
        # Track early stopping patience
        patience_counter = 0
        
        # Training loop
        for epoch in range(1, self.config['num_epochs'] + 1):
            logger.info(f"Epoch {epoch}/{self.config['num_epochs']}")
            
            # Track epoch start time
            start_time = time.time()
            
            # Train for one epoch
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            val_loss = val_metrics['val_loss']
            
            # Calculate epoch duration
            epoch_time = time.time() - start_time
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Print progress
            logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s")
            logger.info(f"Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
            logger.info(f"Validation Metrics: {val_metrics}")
            
            # Update learning rate scheduler
            self.scheduler.step(val_loss)
            
            # Track metrics
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            epochs.append(epoch)
            learning_rates.append(current_lr)
            
            # Log to tensorboard
            if self.writer:
                self.writer.add_scalar('training/epoch_loss', train_loss, epoch)
                self.writer.add_scalar('validation/loss', val_loss, epoch)
                for key, value in val_metrics.items():
                    if key != 'val_loss':
                        self.writer.add_scalar(f'validation/{key}', value, epoch)
                self.writer.add_scalar('training/learning_rate', current_lr, epoch)
            
            # Check if this is the best model so far
            if val_loss < self.best_val_loss:
                logger.info(f"Validation loss improved from {self.best_val_loss:.4f} to {val_loss:.4f}")
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                
                # Save best model
                torch.save(self.model.state_dict(), self.save_paths['best_model'])
                logger.info(f"Saved best model to {self.save_paths['best_model']}")
                
                # Reset patience counter
                patience_counter = 0
            else:
                # Increment patience counter
                patience_counter += 1
                logger.info(f"Validation loss did not improve. Patience: {patience_counter}/{self.config['early_stopping_patience']}")
                
                # Check for early stopping
                if patience_counter >= self.config['early_stopping_patience']:
                    logger.info(f"Early stopping triggered after {epoch} epochs")
                    break
        
        # Save final model
        torch.save(self.model.state_dict(), self.save_paths['last_model'])
        logger.info(f"Saved final model to {self.save_paths['last_model']}")
        
        # Update training history
        self.training_history = {
            'train_loss': train_losses,
            'val_loss': val_losses,
            'epochs': epochs,
            'learning_rates': learning_rates
        }
        
        # Close tensorboard writer
        if self.writer:
            self.writer.close()
        
        return self.training_history
    
    def load_best_model(self):
        """
        Load the best model from checkpoint.
        """
        if os.path.exists(self.save_paths['best_model']):
            self.model.load_state_dict(torch.load(self.save_paths['best_model']))
            logger.info(f"Loaded best model from {self.save_paths['best_model']}")
        else:
            logger.warning(f"Best model checkpoint not found at {self.save_paths['best_model']}")
    
    def save_results(self, test_metrics: Dict[str, float]):
        """
        Save training history and test metrics to JSON.
        
        Args:
            test_metrics: Dictionary with test metrics
        """
        # Prepare results dictionary
        results = {
            'training_history': {
                'train_loss': [float(x) for x in self.training_history['train_loss']],
                'val_loss': [float(x) for x in self.training_history['val_loss']],
                'epochs': self.training_history['epochs'],
                'learning_rates': [float(x) for x in self.training_history['learning_rates']]
            },
            'best_model': {
                'epoch': self.best_epoch,
                'val_loss': float(self.best_val_loss)
            },
            'test_metrics': test_metrics,
            'config': {k: str(v) if isinstance(v, torch.device) else v for k, v in self.config.items()}
        }
        
        # Save to JSON
        with open(self.save_paths['results'], 'w') as f:
            json.dump(results, f, indent=4)
        
        logger.info(f"Saved results to {self.save_paths['results']}")


def process_stream(frame_buffer: List[np.ndarray], model: nn.Module, 
                  yolo_extractor, flow_extractor, device: torch.device, 
                  threshold: float = 0.7) -> bool:
    """
    Process a stream of frames for real-time accident detection.
    
    Args:
        frame_buffer: List of frames (oldest to newest)
        model: Loaded accident detection model
        yolo_extractor: YOLO feature extractor
        flow_extractor: Optical flow feature extractor
        device: Device to run inference on
        threshold: Decision threshold
        
    Returns:
        True if accident detected, False otherwise
    """
    # Extract features for all frames
    features = []
    prev_frame = None
    
    for i, frame in enumerate(frame_buffer):
        # Extract RGB features
        rgb_feat = yolo_extractor.extract_features(frame)
        
        # Extract flow features (skip first frame)
        if i > 0 and prev_frame is not None:
            flow_feat = flow_extractor.extract_features(prev_frame, frame)
        else:
            # Use zeros for the first frame
            flow_feat = np.zeros(128)
        
        # Combine features
        combined_feat = np.concatenate([rgb_feat, flow_feat])
        features.append(combined_feat)
        
        # Update previous frame
        prev_frame = frame
    
    # Convert to tensor
    input_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.sigmoid(output).item()
    
    # Return decision
    return prediction > threshold 