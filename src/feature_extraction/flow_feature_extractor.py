"""
Optical Flow Feature Extractor Module

This module extracts features from optical flow frames for motion analysis
in road accident detection.
"""

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Union, Optional


class SimpleFlowFeatureExtractor:
    """
    A simple feature extractor for optical flow frames that uses
    traditional computer vision techniques.
    """
    
    def __init__(self, feature_dim: int = 128):
        """
        Initialize the flow feature extractor.
        
        Args:
            feature_dim: Dimension of output feature vector
        """
        self.feature_dim = feature_dim
    
    def extract_features(self, flow_frame: np.ndarray) -> np.ndarray:
        """
        Extract features from a single optical flow frame.
        
        Args:
            flow_frame: Optical flow frame as numpy array (visualized as BGR image)
            
        Returns:
            Feature vector as numpy array
        """
        # Convert BGR flow visualization to HSV
        # In HSV, Hue represents direction and Value represents magnitude
        hsv = cv2.cvtColor(flow_frame, cv2.COLOR_BGR2HSV)
        
        # Split into channels
        h, s, v = cv2.split(hsv)
        
        # Extract statistics from different regions
        features = self._extract_regional_statistics(h, s, v)
        
        # Pad or truncate to match desired feature dimension
        if len(features) > self.feature_dim:
            features = features[:self.feature_dim]
        elif len(features) < self.feature_dim:
            features = np.pad(features, (0, self.feature_dim - len(features)))
            
        return features
    
    def _extract_regional_statistics(self, h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Extract statistical features from different regions of the flow frame.
        
        Args:
            h: Hue channel (direction)
            s: Saturation channel
            v: Value channel (magnitude)
            
        Returns:
            Feature vector
        """
        height, width = h.shape
        
        # Define regions (center, corners, edges)
        regions = [
            (slice(0, height//3), slice(0, width//3)),                # Top-left
            (slice(0, height//3), slice(width//3, 2*width//3)),       # Top-center
            (slice(0, height//3), slice(2*width//3, width)),          # Top-right
            (slice(height//3, 2*height//3), slice(0, width//3)),      # Middle-left
            (slice(height//3, 2*height//3), slice(width//3, 2*width//3)), # Center
            (slice(height//3, 2*height//3), slice(2*width//3, width)),# Middle-right
            (slice(2*height//3, height), slice(0, width//3)),         # Bottom-left
            (slice(2*height//3, height), slice(width//3, 2*width//3)),# Bottom-center
            (slice(2*height//3, height), slice(2*width//3, width))    # Bottom-right
        ]
        
        features = []
        
        # Global features
        v_mean = np.mean(v)
        v_std = np.std(v)
        v_max = np.max(v)
        h_mean = np.mean(h)
        h_std = np.std(h)
        
        features.extend([v_mean, v_std, v_max, h_mean, h_std])
        
        # Regional features
        for region in regions:
            region_h = h[region]
            region_v = v[region]
            
            # Magnitude statistics
            v_mean = np.mean(region_v)
            v_std = np.std(region_v)
            v_max = np.max(region_v)
            
            # Direction statistics (using circular mean for angles)
            h_sin = np.mean(np.sin(region_h * 2 * np.pi / 180))
            h_cos = np.mean(np.cos(region_h * 2 * np.pi / 180))
            h_mean = np.arctan2(h_sin, h_cos) * 180 / (2 * np.pi)
            if h_mean < 0:
                h_mean += 180
            
            # Add to features
            features.extend([v_mean, v_std, v_max, h_mean])
        
        # Compute histogram of gradients (simplified)
        gx = cv2.Sobel(v, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(v, cv2.CV_32F, 0, 1, ksize=3)
        mag, ang = cv2.cartToPolar(gx, gy)
        
        # Bin angles into 8 directions
        nbins = 8
        bin_range = 180 // nbins
        hist = np.zeros(nbins)
        
        for i in range(nbins):
            mask = np.logical_and(
                ang >= i * bin_range,
                ang < (i + 1) * bin_range
            )
            hist[i] = np.sum(mag[mask])
        
        # Normalize histogram
        if np.sum(hist) > 0:
            hist = hist / np.sum(hist)
        
        features.extend(hist)
        
        # Add more complex features: flow entropy
        flow_entropy = self._compute_entropy(v)
        features.append(flow_entropy)
        
        # Add flow complexity measure (variation of direction)
        direction_complexity = self._compute_entropy(h)
        features.append(direction_complexity)
        
        # Add ratio of moving pixels (pixels with significant motion)
        motion_ratio = np.sum(v > 20) / (height * width)
        features.append(motion_ratio)
        
        return np.array(features)
    
    def _compute_entropy(self, img: np.ndarray, bins: int = 32) -> float:
        """
        Compute entropy of an image channel.
        
        Args:
            img: Image channel
            bins: Number of bins for histogram
            
        Returns:
            Entropy value
        """
        hist = np.histogram(img, bins=bins, range=(0, 255))[0]
        hist = hist / np.sum(hist)
        
        # Remove zeros to avoid log(0)
        hist = hist[hist > 0]
        
        return -np.sum(hist * np.log2(hist))


class FlowCNN(nn.Module):
    """
    A small 2D CNN model for extracting features from optical flow frames.
    """
    
    def __init__(self, output_dim: int = 128):
        """
        Initialize the Flow CNN model.
        
        Args:
            output_dim: Dimension of output feature vector
        """
        super().__init__()
        
        # Define the CNN architecture
        self.conv1 = nn.Conv2d(3, 16, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        # Global average pooling and linear layer for the desired output dimension
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, output_dim)
        
        self.output_dim = output_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CNN.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Output feature vector of shape (B, output_dim)
        """
        # Convolutional layers with ReLU and batch normalization
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Global average pooling
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        
        # Final fully connected layer
        x = self.fc(x)
        
        return x


class TorchFlowFeatureExtractor:
    """
    PyTorch-based feature extractor for optical flow frames.
    """
    
    def __init__(
        self,
        feature_dim: int = 128,
        use_gpu: Optional[bool] = None,
        pretrained: bool = False,
        pretrained_path: Optional[str] = None
    ):
        """
        Initialize the flow feature extractor with a CNN model.
        
        Args:
            feature_dim: Dimension of output feature vector
            use_gpu: Whether to use GPU for inference (None for auto-detection)
            pretrained: Whether to load pretrained weights
            pretrained_path: Path to pretrained weights file
        """
        # Determine device
        self.device = torch.device("cuda" if torch.cuda.is_available() and 
                                  (use_gpu is None or use_gpu) else "cpu")
        
        # Initialize model
        self.model = FlowCNN(output_dim=feature_dim)
        self.feature_dim = feature_dim
        
        # Load pretrained weights if specified
        if pretrained and pretrained_path:
            self.model.load_state_dict(torch.load(pretrained_path, map_location=self.device))
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
    
    def extract_features(self, flow_frame: np.ndarray) -> np.ndarray:
        """
        Extract features from a single optical flow frame using the CNN.
        
        Args:
            flow_frame: Optical flow frame as numpy array (visualized as BGR image)
            
        Returns:
            Feature vector as numpy array
        """
        # Preprocess the frame
        processed_frame = self._preprocess_frame(flow_frame)
        
        # Extract features
        with torch.no_grad():
            features = self.model(processed_frame)
        
        # Convert to numpy
        features_np = features.cpu().numpy().flatten()
        
        return features_np
    
    def extract_features_batch(self, flow_frames: List[np.ndarray]) -> np.ndarray:
        """
        Extract features from a batch of optical flow frames.
        
        Args:
            flow_frames: List of optical flow frames
            
        Returns:
            Array of features with shape (num_frames, feature_dim)
        """
        # Preprocess all frames
        processed_frames = []
        for frame in flow_frames:
            processed = self._preprocess_frame(frame, add_batch=False)
            processed_frames.append(processed)
        
        # Stack frames into a batch
        batch = torch.stack(processed_frames)
        
        # Extract features
        with torch.no_grad():
            features = self.model(batch)
        
        # Convert to numpy
        features_np = features.cpu().numpy()
        
        return features_np
    
    def _preprocess_frame(self, frame: np.ndarray, add_batch: bool = True) -> torch.Tensor:
        """
        Preprocess a frame for the CNN.
        
        Args:
            frame: Input frame as numpy array
            add_batch: Whether to add a batch dimension
            
        Returns:
            Preprocessed frame as torch tensor
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to 224x224 if needed
        if frame_rgb.shape[:2] != (224, 224):
            frame_rgb = cv2.resize(frame_rgb, (224, 224))
        
        # Convert to float and normalize
        frame_float = frame_rgb.astype(np.float32) / 255.0
        
        # Convert to torch tensor and permute dimensions
        frame_tensor = torch.from_numpy(frame_float).permute(2, 0, 1)
        
        # Add batch dimension if needed
        if add_batch:
            frame_tensor = frame_tensor.unsqueeze(0)
        
        # Move to device
        frame_tensor = frame_tensor.to(self.device)
        
        return frame_tensor 