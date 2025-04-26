"""
Flow feature extraction module for road accident detection.

This module contains feature extractors for optical flow frames.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict, Any
import os
import logging
from typing import Union
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleFlowFeatureExtractor:
    """
    Simple feature extractor for optical flow frames based on statistical measures.
    """
    
    def __init__(self, feature_dim: int = 128):
        """
        Initialize the feature extractor.
        
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

    def extract_features_batch(self, flow_frames: List[np.ndarray]) -> np.ndarray:
        """
        Extract features from a batch of optical flow frames.
        
        Args:
            flow_frames: List of optical flow frames
            
        Returns:
            Array of feature vectors, one per frame
        """
        features = []
        for frame in flow_frames:
            features.append(self.extract_features(frame))
        return np.array(features)


class I3D_Block(nn.Module):
    """
    Inflated 3D ConvNet block based on I3D architecture.
    """
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)):
        super().__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm3d(out_channels)
        
    def forward(self, x):
        x = self.conv3d(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        return x


class I3DFlowNet(nn.Module):
    """
    I3D-inspired network for optical flow feature extraction.
    """
    def __init__(self, output_dim=128):
        super().__init__()
        # Input is expected to be (B, C, T, H, W)
        # where C=3 for RGB flow visualization (or 2 for raw flow)
        self.conv1 = I3D_Block(3, 16, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        
        self.conv2 = I3D_Block(16, 32, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2))
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        
        self.conv3a = I3D_Block(32, 64)
        self.conv3b = I3D_Block(64, 64)
        self.pool3 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        
        self.conv4a = I3D_Block(64, 128)
        self.conv4b = I3D_Block(128, 128)
        
        # Global average pooling across spatial and temporal dimensions
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Feature output
        self.fc = nn.Linear(128, output_dim)
        
        self.output_dim = output_dim
        
    def forward(self, x):
        # Forward pass through I3D network
        x = self.conv1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.pool2(x)
        
        x = self.conv3a(x)
        x = self.conv3b(x)
        x = self.pool3(x)
        
        x = self.conv4a(x)
        x = self.conv4b(x)
        
        # Global pooling
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        
        # Feature output
        x = self.fc(x)
        
        return x


class I3DFlowFeatureExtractor:
    """
    Advanced feature extractor for optical flow using I3D-based network.
    """
    def __init__(
        self,
        feature_dim: int = 128,
        use_gpu: Optional[bool] = None,
        pretrained: bool = False,
        pretrained_path: Optional[str] = None,
        temporal_window: int = 8
    ):
        """
        Initialize the I3D flow feature extractor.
        
        Args:
            feature_dim: Dimension of the feature vector
            use_gpu: Whether to use GPU if available
            pretrained: Whether to load pretrained weights
            pretrained_path: Path to pretrained weights
            temporal_window: Number of frames to use for temporal context
        """
        self.feature_dim = feature_dim
        self.temporal_window = temporal_window
        
        # Set up device
        if use_gpu is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        
        # Create I3D network
        self.model = I3DFlowNet(output_dim=feature_dim)
        
        # Load pretrained weights if specified
        if pretrained and pretrained_path:
            try:
                state_dict = torch.load(pretrained_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                logger.info(f"Loaded pretrained I3D flow model from {pretrained_path}")
            except Exception as e:
                logger.warning(f"Could not load pretrained model: {e}")
        
        # Move model to device and set to evaluation mode
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize buffer for temporal context
        self.frame_buffer = []
        
        logger.info(f"Initialized I3D flow feature extractor (feature_dim={feature_dim}, device={self.device})")
    
    def extract_features(self, flow_frame: np.ndarray) -> np.ndarray:
        """
        Extract features from a single optical flow frame.
        Will use temporal context from previously processed frames.
        
        Args:
            flow_frame: Optical flow frame as numpy array (BGR color format)
            
        Returns:
            Feature vector as numpy array
        """
        # Add frame to buffer
        self.frame_buffer.append(flow_frame)
        
        # Keep only the latest frames based on temporal window
        if len(self.frame_buffer) > self.temporal_window:
            self.frame_buffer.pop(0)
        
        # Calculate temporal differences between consecutive frames
        temporal_diffs = []
        for i in range(1, len(self.frame_buffer)):
            prev = cv2.cvtColor(self.frame_buffer[i-1], cv2.COLOR_BGR2GRAY)
            curr = cv2.cvtColor(self.frame_buffer[i], cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(prev, curr)
            temporal_diffs.append(diff)
        
        # If we don't have enough frames for I3D, use statistical features
        if len(self.frame_buffer) < 2:
            # Use simple extractor as fallback
            simple_extractor = SimpleFlowFeatureExtractor(feature_dim=self.feature_dim)
            return simple_extractor.extract_features(flow_frame)
        
        # Prepare input tensor for I3D network
        # If we don't have enough frames yet, duplicate the current ones
        frames_to_use = self.frame_buffer.copy()
        while len(frames_to_use) < self.temporal_window:
            frames_to_use.append(frames_to_use[-1])
        
        # Convert frames to tensor
        tensor_input = self._prepare_i3d_input(frames_to_use)
        
        # Extract features using I3D model
        with torch.no_grad():
            features = self.model(tensor_input)
            features = features.cpu().numpy().flatten()
        
        # Augment with temporal difference features
        if temporal_diffs:
            # Extract statistical features from temporal differences
            temp_features = []
            for diff in temporal_diffs:
                mean_diff = np.mean(diff)
                std_diff = np.std(diff)
                max_diff = np.max(diff)
                temp_features.extend([mean_diff, std_diff, max_diff])
            
            # Combine I3D features with temporal difference features
            combined_features = np.concatenate([features, np.array(temp_features)])
            
            # Resize to match desired feature dimension
            if len(combined_features) > self.feature_dim:
                combined_features = combined_features[:self.feature_dim]
            elif len(combined_features) < self.feature_dim:
                combined_features = np.pad(combined_features, (0, self.feature_dim - len(combined_features)))
            
            return combined_features
        
        return features
    
    def extract_features_batch(self, flow_frames: List[np.ndarray]) -> np.ndarray:
        """
        Extract features from a batch of optical flow frames.
        
        Args:
            flow_frames: List of optical flow frames
            
        Returns:
            Array of feature vectors, one per frame
        """
        # Reset frame buffer
        self.frame_buffer = []
        
        # Extract features for each frame
        features = []
        for frame in flow_frames:
            features.append(self.extract_features(frame))
        
        return np.array(features)
    
    def _prepare_i3d_input(self, frames: List[np.ndarray]) -> torch.Tensor:
        """
        Prepare input tensor for I3D network.
        
        Args:
            frames: List of frames
            
        Returns:
            Tensor of shape (1, 3, T, H, W)
        """
        # Ensure all frames have the same shape
        height, width = frames[0].shape[:2]
        processed_frames = []
        
        for frame in frames:
            # Resize if necessary
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            
            # Convert to RGB and normalize
            if frame.shape[2] == 3:  # If BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Normalize to [0, 1]
            frame = frame.astype(np.float32) / 255.0
            
            # Transpose to (C, H, W)
            frame = frame.transpose(2, 0, 1)
            
            processed_frames.append(frame)
        
        # Stack frames along time dimension to create tensor of shape (C, T, H, W)
        tensor = np.stack(processed_frames, axis=1)
        
        # Add batch dimension
        tensor = tensor[np.newaxis, ...]
        
        # Convert to tensor and move to device
        tensor = torch.from_numpy(tensor).to(self.device)
        
        return tensor

class TemporalDifferenceExtractor:
    """
    Feature extractor that focuses on temporal differences between consecutive flow frames.
    """
    def __init__(self, feature_dim: int = 64, window_size: int = 5):
        """
        Initialize the temporal difference extractor.
        
        Args:
            feature_dim: Dimension of output feature vector
            window_size: Number of frames to consider for temporal context
        """
        self.feature_dim = feature_dim
        self.window_size = window_size
        self.frame_buffer = []
    
    def extract_features(self, flow_frame: np.ndarray) -> np.ndarray:
        """
        Extract temporal difference features from a flow frame.
        
        Args:
            flow_frame: Optical flow frame (BGR visualization)
            
        Returns:
            Feature vector
        """
        # Add frame to buffer
        self.frame_buffer.append(flow_frame)
        
        # Keep only the latest frames
        if len(self.frame_buffer) > self.window_size:
            self.frame_buffer.pop(0)
        
        # If we don't have enough frames yet, return zeros
        if len(self.frame_buffer) < 2:
            return np.zeros(self.feature_dim)
        
        # Calculate features from temporal differences
        features = []
        
        # Convert frames to HSV
        hsv_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) for frame in self.frame_buffer]
        
        # Get magnitude channels (V in HSV)
        magnitude_frames = [hsv[:,:,2] for hsv in hsv_frames]
        
        # Calculate different temporal differences
        for i in range(1, len(magnitude_frames)):
            curr = magnitude_frames[i]
            prev = magnitude_frames[i-1]
            
            # Simple difference
            diff = cv2.absdiff(curr, prev)
            
            # Statistical features
            mean_diff = np.mean(diff)
            std_diff = np.std(diff)
            max_diff = np.max(diff)
            
            # Calculate regional differences
            h, w = diff.shape
            regions = [
                diff[:h//2, :w//2],              # Top-left
                diff[:h//2, w//2:],              # Top-right
                diff[h//2:, :w//2],              # Bottom-left
                diff[h//2:, w//2:],              # Bottom-right
                diff[h//4:3*h//4, w//4:3*w//4]   # Center
            ]
            
            region_means = [np.mean(region) for region in regions]
            
            # Add features
            features.extend([mean_diff, std_diff, max_diff])
            features.extend(region_means)
            
            # Add more advanced features: directional flow changes
            # For this, use the Hue channel which represents direction
            curr_h = hsv_frames[i][:,:,0]
            prev_h = hsv_frames[i-1][:,:,0]
            
            # Calculate mean direction change (handle circular nature)
            h_diff = np.abs(curr_h.astype(np.float32) - prev_h.astype(np.float32))
            h_diff = np.minimum(h_diff, 180 - h_diff)  # Consider circular difference
            mean_h_diff = np.mean(h_diff)
            std_h_diff = np.std(h_diff)
            
            features.extend([mean_h_diff, std_h_diff])
        
        # Resize features to the desired dimension
        features = np.array(features)
        if len(features) > self.feature_dim:
            features = features[:self.feature_dim]
        elif len(features) < self.feature_dim:
            features = np.pad(features, (0, self.feature_dim - len(features)))
        
        return features


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
    Feature extractor for optical flow using a PyTorch CNN model.
    """
    
    def __init__(
        self,
        feature_dim: int = 128,
        use_gpu: Optional[bool] = None,
        pretrained: bool = False,
        pretrained_path: Optional[str] = None
    ):
        """
        Initialize the feature extractor.
        
        Args:
            feature_dim: Dimension of output feature vector
            use_gpu: Whether to use GPU if available (None for auto-detection)
            pretrained: Whether to use a pretrained model
            pretrained_path: Path to pretrained model weights
        """
        self.feature_dim = feature_dim
        
        # Set device
        if use_gpu is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
            
        # Create the model
        self.model = FlowCNN(output_dim=feature_dim)
        
        # Load pretrained weights if requested
        if pretrained and pretrained_path:
            try:
                state_dict = torch.load(pretrained_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                logger.info(f"Loaded pretrained weights from {pretrained_path}")
            except Exception as e:
                logger.warning(f"Failed to load pretrained weights: {e}")
        
        # Move model to device and set to evaluation mode
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Initialized flow feature extractor (feature_dim={feature_dim}, device={self.device})")
    
    def extract_features(self, flow_frame: np.ndarray) -> np.ndarray:
        """
        Extract features from a single optical flow frame.
        
        Args:
            flow_frame: Optical flow frame as numpy array
            
        Returns:
            Feature vector as numpy array
        """
        # Preprocess the frame
        tensor = self._preprocess_frame(flow_frame)
        
        # Forward pass through the model
        with torch.no_grad():
            features = self.model(tensor)
            features = features.cpu().numpy().flatten()
        
        return features
    
    def extract_features_batch(self, flow_frames: List[np.ndarray]) -> np.ndarray:
        """
        Extract features from a batch of optical flow frames.
        
        Args:
            flow_frames: List of optical flow frames
            
        Returns:
            Array of feature vectors, one per frame
        """
        # Preprocess all frames
        tensors = []
        for frame in flow_frames:
            tensor = self._preprocess_frame(frame, add_batch=False)
            tensors.append(tensor)
        
        # Stack tensors along batch dimension
        batch_tensor = torch.stack(tensors)
        
        # Forward pass through the model
        with torch.no_grad():
            features = self.model(batch_tensor)
            features = features.cpu().numpy()
        
        return features
    
    def _preprocess_frame(self, frame: np.ndarray, add_batch: bool = True) -> torch.Tensor:
        """
        Preprocess a frame for the CNN model.
        
        Args:
            frame: Input frame as numpy array
            add_batch: Whether to add a batch dimension
            
        Returns:
            Preprocessed frame as PyTorch tensor
        """
        # Convert to RGB if necessary
        if frame.shape[2] == 3:  # If BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize if necessary (model expects 224x224)
        if frame.shape[0] != 224 or frame.shape[1] != 224:
            frame = cv2.resize(frame, (224, 224))
        
        # Normalize to [0, 1]
        frame = frame.astype(np.float32) / 255.0
        
        # Transpose to channel-first format (C, H, W)
        frame = frame.transpose(2, 0, 1)
        
        # Convert to tensor
        tensor = torch.from_numpy(frame)
        
        # Add batch dimension if requested
        if add_batch:
            tensor = tensor.unsqueeze(0)
        
        # Move to device
        tensor = tensor.to(self.device)
        
        return tensor 