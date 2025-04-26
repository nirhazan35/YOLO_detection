"""
Feature Extraction Utilities

Utility functions for feature extraction, visualization, and data handling.
"""

import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union
import json
import pickle


def save_features(features: np.ndarray, save_path: str, metadata: Optional[Dict] = None):
    """
    Save extracted features to a file.
    
    Args:
        features: Extracted features as numpy array
        save_path: Path to save the features
        metadata: Optional metadata to save with the features
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Determine file format from extension
    _, ext = os.path.splitext(save_path)
    
    if ext.lower() == '.npy':
        # Save as numpy array
        np.save(save_path, features)
        
        # Save metadata separately if provided
        if metadata is not None:
            metadata_path = save_path.replace('.npy', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
    elif ext.lower() == '.npz':
        # Save as compressed numpy array with metadata
        if metadata is not None:
            np.savez(save_path, features=features, **metadata)
        else:
            np.savez(save_path, features=features)
            
    elif ext.lower() == '.pkl' or ext.lower() == '.pickle':
        # Save as pickle
        with open(save_path, 'wb') as f:
            if metadata is not None:
                pickle.dump({'features': features, 'metadata': metadata}, f)
            else:
                pickle.dump({'features': features}, f)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use .npy, .npz, or .pkl")


def load_features(load_path: str) -> Tuple[np.ndarray, Optional[Dict]]:
    """
    Load features from a file.
    
    Args:
        load_path: Path to load the features from
        
    Returns:
        Tuple of (features, metadata)
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Feature file not found: {load_path}")
        
    # Determine file format from extension
    _, ext = os.path.splitext(load_path)
    
    if ext.lower() == '.npy':
        # Load numpy array
        features = np.load(load_path)
        
        # Try to load metadata if it exists
        metadata = None
        metadata_path = load_path.replace('.npy', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                
        return features, metadata
        
    elif ext.lower() == '.npz':
        # Load compressed numpy array
        data = np.load(load_path)
        features = data['features']
        
        # Extract metadata if available
        metadata = {k: data[k] for k in data.files if k != 'features'}
        if not metadata:
            metadata = None
            
        return features, metadata
        
    elif ext.lower() == '.pkl' or ext.lower() == '.pickle':
        # Load pickle
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
            
        if isinstance(data, dict):
            features = data.get('features', data)
            metadata = data.get('metadata', None)
        else:
            features = data
            metadata = None
            
        return features, metadata
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use .npy, .npz, or .pkl")


def visualize_detection(frame: np.ndarray, detections: np.ndarray, save_path: Optional[str] = None):
    """
    Visualize object detections on a frame.
    
    Args:
        frame: Original frame as numpy array
        detections: Detection features (x, y, w, h, conf)
        save_path: Optional path to save the visualization
    """
    # Make a copy of the frame to avoid modifying the original
    vis_frame = frame.copy()
    
    # Define colors for different object classes (BGR format)
    colors = {
        0: (0, 255, 0),    # person (green)
        1: (255, 0, 0),    # bicycle (blue)
        2: (0, 0, 255),    # car (red)
        3: (255, 255, 0),  # motorcycle (cyan)
        7: (255, 0, 255),  # truck (magenta)
    }
    
    # Draw bounding boxes
    h, w = frame.shape[:2]
    
    for i in range(0, len(detections), 5):
        # Skip if all zeros (padding)
        if np.all(detections[i:i+5] == 0):
            continue
            
        # Extract detection
        x, y, width, height, conf = detections[i:i+5]
        
        # Convert normalized coordinates to pixel coordinates
        x1 = int(x * w - width * w / 2)
        y1 = int(y * h - height * h / 2)
        x2 = int(x * w + width * w / 2)
        y2 = int(y * h + height * h / 2)
        
        # Ensure coordinates are within frame bounds
        x1 = max(0, min(w-1, x1))
        y1 = max(0, min(h-1, y1))
        x2 = max(0, min(w-1, x2))
        y2 = max(0, min(h-1, y2))
        
        # Assume class based on index (simplified for visualization)
        # In a real implementation, you'd need the actual class information
        class_id = (i // 5) % len(colors)
        color = colors.get(class_id, (0, 255, 255))  # default to yellow
        
        # Draw the bounding box
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw confidence
        label = f"{conf:.2f}"
        cv2.putText(vis_frame, label, (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Display or save
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, vis_frame)
    else:
        # Convert to RGB for display
        vis_frame_rgb = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(12, 8))
        plt.imshow(vis_frame_rgb)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    return vis_frame


def visualize_features_pca(features: np.ndarray, labels: Optional[np.ndarray] = None, 
                          save_path: Optional[str] = None):
    """
    Visualize feature vectors using PCA for dimensionality reduction.
    
    Args:
        features: Feature vectors to visualize
        labels: Optional labels for coloring
        save_path: Optional path to save the visualization
    """
    try:
        from sklearn.decomposition import PCA
        
        # Apply PCA to reduce to 2 dimensions
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        
        if labels is not None:
            # Plot with color by label
            unique_labels = np.unique(labels)
            for label in unique_labels:
                mask = labels == label
                plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                           label=f"Class {label}", alpha=0.6)
            plt.legend()
        else:
            # Plot all points in same color
            plt.scatter(features_2d[:, 0], features_2d[:, 1], alpha=0.6)
            
        plt.title("PCA Visualization of Features")
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
        plt.grid(alpha=0.3)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.tight_layout()
            plt.show()
            
    except ImportError:
        print("scikit-learn is required for PCA visualization. Install with 'pip install scikit-learn'")


def visualize_feature_statistics(features: np.ndarray, save_dir: Optional[str] = None):
    """
    Visualize statistics of feature vectors.
    
    Args:
        features: Feature vectors to analyze
        save_dir: Optional directory to save visualizations
    """
    # Create figures directory if saving
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Calculate basic statistics
    feature_means = np.mean(features, axis=0)
    feature_stds = np.std(features, axis=0)
    feature_mins = np.min(features, axis=0)
    feature_maxs = np.max(features, axis=0)
    
    # Plot mean and standard deviation
    plt.figure(figsize=(14, 6))
    plt.plot(feature_means, 'b-', label='Mean')
    plt.fill_between(np.arange(len(feature_means)), 
                    feature_means - feature_stds, 
                    feature_means + feature_stds, 
                    alpha=0.3, color='b', label='±1 Std Dev')
    plt.title('Feature Means and Standard Deviations')
    plt.xlabel('Feature Index')
    plt.ylabel('Value')
    plt.grid(alpha=0.3)
    plt.legend()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'feature_mean_std.png'), 
                   bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.tight_layout()
        plt.show()
    
    # Plot min and max
    plt.figure(figsize=(14, 6))
    plt.plot(feature_maxs, 'r-', label='Max')
    plt.plot(feature_mins, 'g-', label='Min')
    plt.fill_between(np.arange(len(feature_mins)), 
                    feature_mins, feature_maxs, 
                    alpha=0.2, color='y', label='Range')
    plt.title('Feature Min and Max Values')
    plt.xlabel('Feature Index')
    plt.ylabel('Value')
    plt.grid(alpha=0.3)
    plt.legend()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'feature_min_max.png'), 
                   bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.tight_layout()
        plt.show()
        
    # Plot histogram of feature means
    plt.figure(figsize=(10, 6))
    plt.hist(feature_means, bins=30, alpha=0.7)
    plt.title('Distribution of Feature Means')
    plt.xlabel('Mean Value')
    plt.ylabel('Frequency')
    plt.grid(alpha=0.3)
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'feature_means_histogram.png'), 
                   bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.tight_layout()
        plt.show() 