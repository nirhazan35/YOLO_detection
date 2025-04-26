"""
Data Loading Module for Accident Detection LSTM

This module handles loading, preprocessing, and preparing training data
for the accident detection model.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple, List, Optional, Union


def load_features(feature_path: str) -> Tuple[np.ndarray, Dict]:
    """
    Load features from a saved feature file.
    
    Args:
        feature_path: Path to the feature file (.npz format)
        
    Returns:
        Tuple of (feature_array, metadata)
    """
    try:
        feature_data = np.load(feature_path, allow_pickle=True)
        
        # Extract features and metadata
        features = feature_data['features'] if 'features' in feature_data else feature_data['arr_0']
        metadata = feature_data['metadata'] if 'metadata' in feature_data else {}
        
        return features, metadata
    except Exception as e:
        print(f"Error loading features from {feature_path}: {e}")
        return np.array([]), {}


def load_sequences(feature_dir: str, seq_length: int = 16, overlap: int = 8) -> Tuple[List[np.ndarray], List[int], List[str]]:
    """
    Load feature sequences from the feature directory, splitting into overlapping sequences.
    Track video source for each sequence to ensure proper data splitting.
    
    Args:
        feature_dir: Base directory containing feature files organized by category
        seq_length: Length of sequences to extract
        overlap: Overlap between consecutive sequences (for data augmentation)
        
    Returns:
        Tuple of (sequences, labels, video_ids) as lists
    """
    sequences = []
    labels = []
    video_ids = []  # Track which video each sequence belongs to
    
    # Assuming directory structure: features/{category}/{video_features.npz}
    for category in ["accidents", "non_accidents"]:
        category_path = os.path.join(feature_dir, category)
        if not os.path.exists(category_path):
            print(f"Warning: Category path {category_path} does not exist.")
            continue
            
        label = 1 if category == "accidents" else 0
        
        for feature_file in os.listdir(category_path):
            if not feature_file.endswith('.npz'):
                continue
                
            video_id = f"{category}_{feature_file}"  # Create a unique ID for each video
            feature_path = os.path.join(category_path, feature_file)
            features, metadata = load_features(feature_path)
            
            if len(features) == 0:
                continue
                
            # Split into sequences of length seq_length with overlap
            num_frames = features.shape[0]
            for i in range(0, max(1, num_frames - seq_length + 1), max(1, seq_length - overlap)):
                end_idx = min(i + seq_length, num_frames)
                sequence = features[i:end_idx]
                
                # Pad if necessary
                if sequence.shape[0] < seq_length:
                    pad_length = seq_length - sequence.shape[0]
                    padding = np.zeros((pad_length, sequence.shape[1]))
                    sequence = np.vstack([sequence, padding])
                
                sequences.append(sequence)
                labels.append(label)
                video_ids.append(video_id)  # Store video ID for each sequence
                
    return sequences, labels, video_ids


def create_data_splits(features_dir: str, test_size: float = 0.2, val_size: float = 0.15, 
                      seq_length: int = 16, random_state: int = 42) -> Dict[str, np.ndarray]:
    """
    Load features and create train/validation/test splits, ensuring videos
    are properly split by class and sequences from the same video stay together.
    
    Args:
        features_dir: Directory containing feature files
        test_size: Proportion of data to use for testing
        val_size: Proportion of data to use for validation
        seq_length: Length of sequences to extract
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary containing X_train, y_train, X_val, y_val, X_test, y_test
    """
    # Load sequences with video IDs
    sequences, labels, video_ids = load_sequences(features_dir, seq_length=seq_length)
    
    if len(sequences) == 0:
        raise ValueError("No sequences could be loaded from the features directory")
    
    # Group sequences by video and category
    videos_by_category = {}
    for seq, label, vid_id in zip(sequences, labels, video_ids):
        category = "accident" if label == 1 else "non_accident"
        if category not in videos_by_category:
            videos_by_category[category] = {}
        
        if vid_id not in videos_by_category[category]:
            videos_by_category[category][vid_id] = []
        
        videos_by_category[category][vid_id].append(seq)
    
    # Split video IDs by category to maintain class balance
    train_videos, val_videos, test_videos = {}, {}, {}
    
    for category, videos in videos_by_category.items():
        video_ids_list = list(videos.keys())
        
        # First split to separate test set
        train_val_ids, test_ids = train_test_split(
            video_ids_list,
            test_size=test_size,
            random_state=random_state
        )
        
        # Then split remaining into train and validation
        adjusted_val_size = val_size / (1 - test_size)
        train_ids, val_ids = train_test_split(
            train_val_ids,
            test_size=adjusted_val_size,
            random_state=random_state
        )
        
        # Store the split video IDs
        train_videos[category] = train_ids
        val_videos[category] = val_ids
        test_videos[category] = test_ids
    
    # Create train, val, test datasets by collecting sequences from their respective videos
    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []
    
    for category, videos in videos_by_category.items():
        label = 1 if category == "accident" else 0
        
        for vid_id, sequences in videos.items():
            # Determine which split this video belongs to
            if vid_id in train_videos.get(category, []):
                X_train.extend(sequences)
                y_train.extend([label] * len(sequences))
            elif vid_id in val_videos.get(category, []):
                X_val.extend(sequences)
                y_val.extend([label] * len(sequences))
            elif vid_id in test_videos.get(category, []):
                X_test.extend(sequences)
                y_test.extend([label] * len(sequences))
    
    # Convert to numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    # Print statistics about the split
    print(f"Training set: {len(X_train)} sequences ({np.sum(y_train)} accident, {len(y_train) - np.sum(y_train)} non-accident)")
    print(f"Validation set: {len(X_val)} sequences ({np.sum(y_val)} accident, {len(y_val) - np.sum(y_val)} non-accident)")
    print(f"Test set: {len(X_test)} sequences ({np.sum(y_test)} accident, {len(y_test) - np.sum(y_test)} non-accident)")
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test
    }


def normalize_sequences(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize sequences based on training data statistics.
    
    Args:
        X_train, X_val, X_test: Input feature arrays
        
    Returns:
        Normalized versions of input arrays
    """
    # Calculate mean and standard deviation from training data
    means = np.mean(X_train, axis=(0, 1), keepdims=True)
    stds = np.std(X_train, axis=(0, 1), keepdims=True)
    
    # Replace zeros in stds to avoid division by zero
    stds[stds == 0] = 1.0
    
    # Normalize using computed statistics
    X_train_norm = (X_train - means) / stds
    X_val_norm = (X_val - means) / stds
    X_test_norm = (X_test - means) / stds
    
    return X_train_norm, X_val_norm, X_test_norm


class AccidentDataset(Dataset):
    """
    PyTorch Dataset for accident detection data.
    """
    def __init__(self, features: np.ndarray, labels: np.ndarray, normalize: bool = True):
        """
        Initialize the dataset.
        
        Args:
            features: Feature sequences (num_sequences, seq_length, feature_dim)
            labels: Labels for each sequence (num_sequences,)
            normalize: Whether to normalize features
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        
        if normalize:
            # Normalize per channel
            means = torch.mean(self.features, dim=(0,1))
            stds = torch.std(self.features, dim=(0,1))
            stds[stds == 0] = 1.0  # Prevent division by zero
            self.features = (self.features - means) / stds
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def create_data_loaders(data_dict: Dict[str, np.ndarray], batch_size: int = 32,
                       weighted_sampling: bool = True) -> Dict[str, DataLoader]:
    """
    Create PyTorch DataLoaders from data dictionary.
    
    Args:
        data_dict: Dictionary with X_train, y_train, X_val, y_val, X_test, y_test
        batch_size: Batch size for training
        weighted_sampling: Whether to use weighted sampling to handle class imbalance
        
    Returns:
        Dictionary with train_loader, val_loader, test_loader
    """
    # Normalize the data
    X_train_norm, X_val_norm, X_test_norm = normalize_sequences(
        data_dict['X_train'], data_dict['X_val'], data_dict['X_test']
    )
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_norm)
    y_train_tensor = torch.FloatTensor(data_dict['y_train'])
    X_val_tensor = torch.FloatTensor(X_val_norm)
    y_val_tensor = torch.FloatTensor(data_dict['y_val'])
    X_test_tensor = torch.FloatTensor(X_test_norm)
    y_test_tensor = torch.FloatTensor(data_dict['y_test'])
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor.unsqueeze(1))
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor.unsqueeze(1))
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor.unsqueeze(1))
    
    # Create samplers for handling class imbalance
    train_sampler = None
    if weighted_sampling:
        class_counts = np.bincount(data_dict['y_train'].astype(int))
        weights = 1. / class_counts
        samples_weights = weights[data_dict['y_train'].astype(int)]
        train_sampler = WeightedRandomSampler(samples_weights, len(samples_weights))
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler if train_sampler else None,
        shuffle=True if train_sampler is None else False,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader
    } 