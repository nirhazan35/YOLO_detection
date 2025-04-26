"""
Feature Extractor Module

This module contains classes and functions for extracting features from video frames
using the YOLO11 model for road accident detection.
"""

import numpy as np
import torch
from ultralytics import YOLO
from typing import List, Tuple, Dict, Union, Optional

class YOLO11FeatureExtractor:
    """
    Feature extractor using YOLO11 model to detect road-related objects and extract
    both object detection features and spatial features from frames.
    """
    
    # Classes related to road accidents that we're interested in
    ROAD_CLASSES = {
        2: 'car',
        0: 'person',  # pedestrian
        7: 'truck',
        1: 'bicycle',
        3: 'motorcycle',
        5: 'bus',
        9: 'traffic light'
    }
    
    def __init__(
        self, 
        model_path: str = "yolo11m.pt",
        device: Optional[str] = None,
        max_objects: int = 5
    ):
        """
        Initialize the YOLO11 feature extractor.
        
        Args:
            model_path: Path to the YOLO11 model weights
            device: Device to run inference on ('cuda', 'cpu', etc.). If None, automatically selected.
            max_objects: Maximum number of objects to include in features
        """
        self.model = YOLO(model_path)
        self.device = device
        self.max_objects = max_objects
        self.obj_feature_dim = 5  # x, y, w, h, class_prob
        self.spatial_feature_dim = 256
        self.feature_dim = self._calculate_feature_dim()
        
        # Verify model loaded successfully
        if self.model is None:
            raise RuntimeError(f"Failed to load YOLO11 model from {model_path}")
    
    def _calculate_feature_dim(self) -> int:
        """Calculate the dimension of the output feature vector"""
        # Object features: 5 values per object (x, y, w, h, class_prob)
        obj_dim = self.obj_feature_dim * self.max_objects
        
        # Spatial features dimension (backbone output)
        spatial_dim = self.spatial_feature_dim
        
        return obj_dim + spatial_dim
    
    def extract_features(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract features from a single frame.
        
        Args:
            frame: Input frame as numpy array (H, W, C)
            
        Returns:
            Combined object and spatial features as a numpy array
        """
        # Run inference on the frame
        results = self.model(frame, verbose=False)
        
        # Extract object detection features
        object_features = self._extract_object_features(results[0])
        
        # Extract spatial features from backbone
        spatial_features = self._extract_spatial_features(results[0])
        
        # Concatenate both feature types
        combined_features = np.concatenate([object_features, spatial_features])
        
        return combined_features
    
    def _extract_object_features(self, result) -> np.ndarray:
        """
        Extract object detection features from a single result.
        
        Returns:
            Object features as numpy array with shape (max_objects * 5,)
            Each object has 5 values: [x, y, width, height, class_probability]
        """
        # Get normalized bounding boxes (xywh format, values 0-1)
        if hasattr(result.boxes, 'xywhn') and len(result.boxes) > 0:
            boxes = result.boxes.xywhn.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
        else:
            # No detections
            boxes = np.zeros((0, 4))
            classes = np.zeros(0)
            confidences = np.zeros(0)
        
        # Filter for only road-related classes we're interested in
        road_class_indices = []
        for i, cls in enumerate(classes):
            if int(cls) in self.ROAD_CLASSES:
                road_class_indices.append(i)
        
        filtered_boxes = boxes[road_class_indices] if road_class_indices else np.zeros((0, 4))
        filtered_classes = classes[road_class_indices] if road_class_indices else np.zeros(0)
        filtered_confidences = confidences[road_class_indices] if road_class_indices else np.zeros(0)
        
        # Sort by confidence and take top max_objects
        if len(filtered_confidences) > 0:
            sort_idx = np.argsort(-filtered_confidences)
            sorted_boxes = filtered_boxes[sort_idx]
            sorted_classes = filtered_classes[sort_idx]
            sorted_confidences = filtered_confidences[sort_idx]
            
            # Take top max_objects
            sorted_boxes = sorted_boxes[:self.max_objects]
            sorted_classes = sorted_classes[:self.max_objects]
            sorted_confidences = sorted_confidences[:self.max_objects]
        else:
            sorted_boxes = np.zeros((0, 4))
            sorted_classes = np.zeros(0)
            sorted_confidences = np.zeros(0)
        
        # Initialize feature array with zeros
        # Each object has 5 values: x, y, w, h, confidence
        object_features = np.zeros(self.max_objects * self.obj_feature_dim)
        
        # Fill in detected objects
        for i in range(min(len(sorted_boxes), self.max_objects)):
            # Get box coordinates
            x, y, w, h = sorted_boxes[i]
            conf = sorted_confidences[i]
            
            # Fill in the object's features
            start_idx = i * self.obj_feature_dim
            object_features[start_idx:start_idx+4] = [x, y, w, h]
            object_features[start_idx+4] = conf
        
        return object_features
    
    def _extract_spatial_features(self, result) -> np.ndarray:
        """
        Extract spatial features from the model's backbone.
        
        Returns:
            Spatial features as numpy array with shape (256,)
        """
        try:
            # Method 1: Try to access backbone features directly from YOLO results
            if hasattr(result, 'features') and result.features is not None:
                features = result.features
                if isinstance(features, torch.Tensor):
                    # Global average pooling and flatten
                    features = features.mean(dim=[2, 3]).squeeze().cpu().numpy()
                    return self._resize_features(features)
                
            # Method 2: Extract from the model's probs vector (classification output)
            if hasattr(result, 'probs') and result.probs is not None:
                probs = result.probs.cpu().numpy()
                if len(probs) > 0:
                    return self._resize_features(probs)
                    
            # Method 3: Use the original image with PyTorch hooks to get backbone features
            # This is more complex but reliable
            features = self._extract_features_with_hooks(result.orig_img)
            if features is not None:
                return features
            
            # Fallback: Use a simple CNN feature extraction from the image
            # Convert the detection results into a feature map
            width, height = result.orig_img.shape[1], result.orig_img.shape[0]
            detection_map = np.zeros((height, width), dtype=np.float32)
            
            # Add detection confidences to the map
            if hasattr(result.boxes, 'xywh') and len(result.boxes) > 0:
                boxes = result.boxes.xywh.cpu().numpy()  # center_x, center_y, width, height
                scores = result.boxes.conf.cpu().numpy()
                
                for i, box in enumerate(boxes):
                    x, y, w, h = box
                    x, y, w, h = int(x), int(y), int(w), int(h)
                    x1, y1 = max(0, x - w//2), max(0, y - h//2)
                    x2, y2 = min(width, x + w//2), min(height, y + h//2)
                    detection_map[y1:y2, x1:x2] = np.maximum(detection_map[y1:y2, x1:x2], scores[i])
            
            # Extract statistics from the detection map
            if np.sum(detection_map) > 0:
                # Compute histograms and statistics
                flat_map = detection_map.flatten()
                hist, _ = np.histogram(flat_map[flat_map > 0], bins=20, range=(0, 1))
                hist = hist / np.sum(hist) if np.sum(hist) > 0 else hist
                
                # Regional statistics
                h, w = detection_map.shape
                regions = [
                    detection_map[:h//2, :w//2],              # Top-left
                    detection_map[:h//2, w//2:],              # Top-right
                    detection_map[h//2:, :w//2],              # Bottom-left
                    detection_map[h//2:, w//2:],              # Bottom-right
                    detection_map[h//4:3*h//4, w//4:3*w//4]   # Center
                ]
                
                region_stats = []
                for region in regions:
                    if np.sum(region) > 0:
                        region_stats.extend([
                            np.mean(region),
                            np.std(region),
                            np.max(region),
                            np.sum(region > 0) / region.size  # Coverage
                        ])
                    else:
                        region_stats.extend([0, 0, 0, 0])
                
                # Additional global statistics
                global_stats = [
                    np.mean(detection_map),
                    np.std(detection_map),
                    np.max(detection_map),
                    np.sum(detection_map > 0) / detection_map.size,  # Coverage
                    len(boxes) if hasattr(result.boxes, 'xywh') else 0  # Number of detections
                ]
                
                # Combine features
                combined = np.concatenate([hist, region_stats, global_stats])
                return self._resize_features(combined)
            
            # Last resort: return zeros
            return np.zeros(self.spatial_feature_dim)
                
        except (AttributeError, IndexError) as e:
            # If we can't access backbone features, return zeros
            print(f"Warning: Could not extract spatial features: {e}")
            return np.zeros(self.spatial_feature_dim)
    
    def _resize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Resize features to match the desired spatial feature dimension.
        
        Args:
            features: Input feature vector
            
        Returns:
            Resized feature vector
        """
        # Ensure we have the right dimension (pad or truncate if needed)
        if len(features) > self.spatial_feature_dim:
            features = features[:self.spatial_feature_dim]
        elif len(features) < self.spatial_feature_dim:
            features = np.pad(features, (0, self.spatial_feature_dim - len(features)))
        
        return features
    
    def _extract_features_with_hooks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract backbone features using PyTorch hooks.
        
        Args:
            image: Input image
            
        Returns:
            Feature vector or None if extraction failed
        """
        try:
            # Access the PyTorch model inside the YOLO wrapper
            model = self.model.model
            
            # Register a forward hook
            features = []
            
            def hook_fn(module, input, output):
                # Extract and store the output
                features.append(output)
            
            # Try to find the backbone's last layer
            if hasattr(model, 'backbone') and model.backbone is not None:
                # Find a good layer to extract from
                target_module = None
                for name, module in model.backbone.named_modules():
                    # Look for the last convolutional or bottleneck layer
                    if isinstance(module, torch.nn.Conv2d) or 'Bottleneck' in module.__class__.__name__:
                        target_module = module
                
                if target_module is not None:
                    # Register hook
                    hook = target_module.register_forward_hook(hook_fn)
                    
                    # Convert image to tensor
                    if isinstance(image, np.ndarray):
                        # Convert HWC to CHW
                        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
                        # Add batch dimension
                        image_tensor = image_tensor.unsqueeze(0)
                        # Normalize
                        image_tensor = image_tensor / 255.0
                    else:
                        image_tensor = image
                    
                    # Forward pass
                    with torch.no_grad():
                        _ = model(image_tensor)
                    
                    # Remove hook
                    hook.remove()
                    
                    # Process features
                    if features:
                        feature_tensor = features[0]
                        # Global average pooling
                        pooled = torch.nn.functional.adaptive_avg_pool2d(feature_tensor, (1, 1))
                        # Flatten
                        flat = pooled.view(pooled.size(0), -1)
                        # Convert to numpy
                        feature_vector = flat.cpu().numpy()[0]
                        return self._resize_features(feature_vector)
            
            return None
            
        except Exception as e:
            print(f"Error extracting features with hooks: {e}")
            return None
    
    def extract_features_batch(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Extract features from a batch of frames.
        
        Args:
            frames: List of frames as numpy arrays
            
        Returns:
            Array of features with shape (num_frames, feature_dim)
        """
        features = []
        for frame in frames:
            frame_features = self.extract_features(frame)
            features.append(frame_features)
        
        return np.array(features) 