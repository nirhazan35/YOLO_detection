"""
Feature Extractor Module

This module contains classes and functions for extracting features from video frames
using the YOLO11 model for road accident detection.
"""

import numpy as np
import torch
from ultralytics import YOLO
from typing import List, Tuple, Dict, Union, Optional
import torch.nn as nn

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
            Plus additional relative positioning features between objects
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
        # Base object features: x, y, w, h, confidence
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
        
        # Calculate relative positioning features if we have more than one object
        if len(sorted_boxes) > 1:
            rel_pos_features = self._calculate_relative_positions(sorted_boxes)
            
            # Append these to the object_features array if there's room
            # We need to ensure we're still returning exactly max_objects * obj_feature_dim
            # So we'll replace some of the empty padding with useful relational features
            if len(sorted_boxes) < self.max_objects:
                # Calculate how many unused entries we have in the feature vector
                unused_entries = (self.max_objects - len(sorted_boxes)) * self.obj_feature_dim
                
                # Use up to that many entries for relational features
                rel_features_to_use = min(unused_entries, len(rel_pos_features))
                if rel_features_to_use > 0:
                    # Start filling from the end of the actual object features
                    start_idx = len(sorted_boxes) * self.obj_feature_dim
                    object_features[start_idx:start_idx+rel_features_to_use] = rel_pos_features[:rel_features_to_use]
        
        return object_features
    
    def _calculate_relative_positions(self, boxes: np.ndarray) -> np.ndarray:
        """
        Calculate relative positioning features between objects.
        
        Args:
            boxes: Array of bounding boxes in xywh format (normalized)
            
        Returns:
            Array of relative positioning features
        """
        num_boxes = len(boxes)
        features = []
        
        # Skip if less than 2 boxes
        if num_boxes < 2:
            return np.array([])
        
        for i in range(num_boxes):
            box1_x, box1_y, box1_w, box1_h = boxes[i]
            box1_center = np.array([box1_x, box1_y])
            
            for j in range(i + 1, num_boxes):
                box2_x, box2_y, box2_w, box2_h = boxes[j]
                box2_center = np.array([box2_x, box2_y])
                
                # Calculate distance between centers (Euclidean)
                distance = np.linalg.norm(box1_center - box2_center)
                
                # Calculate angle between centers
                angle = np.arctan2(box2_y - box1_y, box2_x - box1_x)
                
                # Calculate size ratio
                area1 = box1_w * box1_h
                area2 = box2_w * box2_h
                size_ratio = area1 / (area2 + 1e-6)  # Avoid division by zero
                
                # Calculate overlap ratio
                x_overlap = max(0, min(box1_x + box1_w/2, box2_x + box2_w/2) - max(box1_x - box1_w/2, box2_x - box2_w/2))
                y_overlap = max(0, min(box1_y + box1_h/2, box2_y + box2_h/2) - max(box1_y - box1_h/2, box2_y - box2_h/2))
                intersection = x_overlap * y_overlap
                union = area1 + area2 - intersection
                iou = intersection / (union + 1e-6)  # Intersection over Union
                
                # Add these features
                features.extend([distance, angle, size_ratio, iou])
        
        return np.array(features)
    
    def extract_trajectory_features(self, frames_results, window_size=5):
        """
        Extract trajectory features from a sequence of frame results.
        
        Args:
            frames_results: List of YOLO detection results from consecutive frames
            window_size: Number of frames to consider for trajectory
            
        Returns:
            Trajectory features as numpy array
        """
        num_frames = len(frames_results)
        if num_frames < 2:
            return np.zeros(self.max_objects * 4)  # 4 values per object: dx, dy, speed, angle
        
        # We'll focus on the window_size most recent frames
        start_idx = max(0, num_frames - window_size)
        recent_results = frames_results[start_idx:]
        
        # Track objects across frames using simple IoU matching
        trajectories = []
        
        # Extract boxes from all frames first
        all_boxes = []
        all_classes = []
        all_confidences = []
        
        for result in recent_results:
            if hasattr(result.boxes, 'xywhn') and len(result.boxes) > 0:
                boxes = result.boxes.xywhn.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                
                # Filter for road-related classes
                road_indices = [i for i, cls in enumerate(classes) if int(cls) in self.ROAD_CLASSES]
                
                filtered_boxes = boxes[road_indices] if road_indices else np.zeros((0, 4))
                filtered_classes = classes[road_indices] if road_indices else np.zeros(0)
                filtered_confidences = confidences[road_indices] if road_indices else np.zeros(0)
                
                # Sort by confidence
                if len(filtered_confidences) > 0:
                    sort_idx = np.argsort(-filtered_confidences)
                    sorted_boxes = filtered_boxes[sort_idx][:self.max_objects]
                    sorted_classes = filtered_classes[sort_idx][:self.max_objects]
                else:
                    sorted_boxes = np.zeros((0, 4))
                    sorted_classes = np.zeros(0)
                
                all_boxes.append(sorted_boxes)
                all_classes.append(sorted_classes)
            else:
                all_boxes.append(np.zeros((0, 4)))
                all_classes.append(np.zeros(0))
        
        # Calculate trajectory features for each object
        trajectory_features = np.zeros(self.max_objects * 4)  # dx, dy, speed, angle
        
        # For simplicity, we'll just look at displacement between first and last frame
        if len(all_boxes) >= 2 and len(all_boxes[0]) > 0 and len(all_boxes[-1]) > 0:
            first_boxes = all_boxes[0]
            last_boxes = all_boxes[-1]
            
            # Match objects between first and last frame using IoU
            for i, first_box in enumerate(first_boxes):
                if i >= self.max_objects:
                    break
                    
                best_match = -1
                best_iou = 0.3  # IoU threshold
                
                for j, last_box in enumerate(last_boxes):
                    # Calculate IoU between boxes
                    x_overlap = max(0, min(first_box[0] + first_box[2]/2, last_box[0] + last_box[2]/2) - 
                                    max(first_box[0] - first_box[2]/2, last_box[0] - last_box[2]/2))
                    y_overlap = max(0, min(first_box[1] + first_box[3]/2, last_box[1] + last_box[3]/2) - 
                                    max(first_box[1] - first_box[3]/2, last_box[1] - last_box[3]/2))
                    
                    intersection = x_overlap * y_overlap
                    area1 = first_box[2] * first_box[3]
                    area2 = last_box[2] * last_box[3]
                    union = area1 + area2 - intersection
                    iou = intersection / (union + 1e-6)
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_match = j
                
                # If we found a match, calculate trajectory features
                if best_match != -1:
                    matched_box = last_boxes[best_match]
                    
                    # Calculate displacement
                    dx = matched_box[0] - first_box[0]
                    dy = matched_box[1] - first_box[1]
                    
                    # Calculate speed and angle
                    speed = np.sqrt(dx*dx + dy*dy)
                    angle = np.arctan2(dy, dx)
                    
                    # Store features
                    feature_idx = i * 4
                    trajectory_features[feature_idx:feature_idx+4] = [dx, dy, speed, angle]
        
        return trajectory_features
    
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

class CrossModalFusion:
    """
    Cross-modal fusion between RGB and flow features with attention mechanism.
    """
    def __init__(self, rgb_dim=281, flow_dim=128, fused_dim=409, use_gpu=None):
        """
        Initialize the cross-modal fusion module.
        
        Args:
            rgb_dim: Dimension of RGB features
            flow_dim: Dimension of flow features
            fused_dim: Dimension of fused features
            use_gpu: Whether to use GPU if available
        """
        self.rgb_dim = rgb_dim
        self.flow_dim = flow_dim
        self.fused_dim = fused_dim
        
        # Determine device
        if use_gpu is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        
        # Attention networks
        self.rgb_attention = nn.Sequential(
            nn.Linear(rgb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, flow_dim),
            nn.Sigmoid()
        ).to(self.device)
        
        self.flow_attention = nn.Sequential(
            nn.Linear(flow_dim, 128),
            nn.ReLU(),
            nn.Linear(128, rgb_dim),
            nn.Sigmoid()
        ).to(self.device)
        
        # Feature normalization layers
        self.rgb_norm = nn.LayerNorm(rgb_dim).to(self.device)
        self.flow_norm = nn.LayerNorm(flow_dim).to(self.device)
        
        # Final fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(rgb_dim + flow_dim, fused_dim),
            nn.ReLU()
        ).to(self.device)
        
        # Set to evaluation mode
        self.rgb_attention.eval()
        self.flow_attention.eval()
        self.rgb_norm.eval()
        self.flow_norm.eval()
        self.fusion_layer.eval()
    
    def fuse_features(self, rgb_features, flow_features):
        """
        Fuse RGB and flow features using cross-modal attention.
        
        Args:
            rgb_features: RGB features tensor or numpy array
            flow_features: Flow features tensor or numpy array
            
        Returns:
            Fused features
        """
        # Convert to tensors if needed
        if isinstance(rgb_features, np.ndarray):
            rgb_features = torch.from_numpy(rgb_features).float().to(self.device)
        if isinstance(flow_features, np.ndarray):
            flow_features = torch.from_numpy(flow_features).float().to(self.device)
        
        # Make sure dimensions match
        if rgb_features.dim() == 1:
            rgb_features = rgb_features.unsqueeze(0)
        if flow_features.dim() == 1:
            flow_features = flow_features.unsqueeze(0)
        
        # Apply normalization (standardizes feature scales)
        rgb_features = self.rgb_norm(rgb_features)
        flow_features = self.flow_norm(flow_features)
        
        # Apply attention mechanisms
        with torch.no_grad():
            # Flow-guided attention for RGB
            flow_attn_weights = self.flow_attention(flow_features)
            attended_rgb = rgb_features * flow_attn_weights
            
            # RGB-guided attention for flow
            rgb_attn_weights = self.rgb_attention(rgb_features)
            attended_flow = flow_features * rgb_attn_weights
            
            # Concatenate attended features
            combined_features = torch.cat([attended_rgb, attended_flow], dim=1)
            
            # Apply fusion layer
            fused_features = self.fusion_layer(combined_features)
        
        # Return as numpy array
        return fused_features.cpu().numpy()
    
    def fuse_features_batch(self, rgb_features_batch, flow_features_batch):
        """
        Fuse batches of RGB and flow features.
        
        Args:
            rgb_features_batch: Batch of RGB features (numpy array or tensor)
            flow_features_batch: Batch of flow features (numpy array or tensor)
            
        Returns:
            Batch of fused features
        """
        # Convert to tensors if needed
        if isinstance(rgb_features_batch, np.ndarray):
            rgb_features_batch = torch.from_numpy(rgb_features_batch).float().to(self.device)
        if isinstance(flow_features_batch, np.ndarray):
            flow_features_batch = torch.from_numpy(flow_features_batch).float().to(self.device)
        
        # Apply normalization
        rgb_features_batch = self.rgb_norm(rgb_features_batch)
        flow_features_batch = self.flow_norm(flow_features_batch)
        
        # Apply attention and fusion
        with torch.no_grad():
            # Flow-guided attention for RGB
            flow_attn_weights = self.flow_attention(flow_features_batch)
            attended_rgb = rgb_features_batch * flow_attn_weights
            
            # RGB-guided attention for flow
            rgb_attn_weights = self.rgb_attention(rgb_features_batch)
            attended_flow = flow_features_batch * rgb_attn_weights
            
            # Concatenate attended features
            combined_features = torch.cat([attended_rgb, attended_flow], dim=1)
            
            # Apply fusion layer
            fused_features = self.fusion_layer(combined_features)
        
        # Return as numpy array
        return fused_features.cpu().numpy() 