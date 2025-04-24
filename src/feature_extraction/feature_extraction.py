import os
import sys
import numpy as np
import torch
import logging
import json
from tqdm import tqdm
import cv2
from pathlib import Path

# Add parent directory to path if necessary
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import config
from feature_extraction.config import YOLO_CONFIG, FEATURE_CONFIG, GPU_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("feature_extraction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class YOLOFeatureExtractor:
    """Extract features from video frames using YOLO11."""
    
    def __init__(self, model_path=None, device=None):
        """
        Initialize the YOLO feature extractor.
        
        Args:
            model_path: Path to YOLO11 weights (if None, use config value)
            device: Computation device ('cuda' or 'cpu')
        """
        # Use config value if parameter is not specified
        self.model_path = model_path if model_path else YOLO_CONFIG['model_path']
        
        # Set device based on config if not specified
        if device is None:
            if GPU_CONFIG['use_gpu'] and torch.cuda.is_available():
                self.device = torch.device(f"cuda:{GPU_CONFIG['gpu_id']}")
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
            
        logger.info(f"Using device: {self.device}")
        
        # Load YOLO model
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            logger.info(f"Loaded YOLO model from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            sys.exit(1)
            
        # Classes we're interested in (from config)
        self.target_classes = FEATURE_CONFIG['object_classes']
        
        # Map class names to class IDs in YOLO
        self.class_ids = {
            'person': 0,
            'bicycle': 1,
            'car': 2,
            'motorcycle': 3,
            'bus': 5,
            'truck': 7
        }
    
    def extract_features(self, frame):
        """
        Extract features from a single frame.
        
        Args:
            frame: Input image frame
            
        Returns:
            Dictionary containing object and spatial features
        """
        try:
            # Run inference with YOLO
            with torch.no_grad():
                results = self.model(frame, verbose=False)
                
            # Extract object detection features
            object_features = self._extract_object_features(results[0])
            
            # Extract spatial features
            spatial_features = self._extract_spatial_features(results[0])
            
            return {
                'object_features': object_features,
                'spatial_features': spatial_features
            }
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            # Return empty features if extraction fails
            return {
                'object_features': np.zeros((FEATURE_CONFIG['max_objects'], 5)),  # max_objects, 5 features each
                'spatial_features': np.zeros(FEATURE_CONFIG['spatial_dim'])       # spatial_dim features
            }
    
    def _extract_object_features(self, result):
        """
        Extract object detection features (bounding boxes and classes).
        
        Args:
            result: YOLO detection result for a single frame
            
        Returns:
            Numpy array of object features
        """
        # Initialize empty array for object features
        # Each object: [x, y, width, height, class_probability]
        max_objects = FEATURE_CONFIG['max_objects']
        object_features = np.zeros((max_objects, 5))  # max_objects, 5 features each
        
        try:
            # Get detections
            boxes = result.boxes
            
            if len(boxes) == 0:
                return object_features
            
            # Get normalized bounding box coordinates [x, y, width, height]
            xywhn = boxes.xywhn.cpu().numpy()
            
            # Get class IDs and probabilities
            cls = boxes.cls.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            
            # Filter for target classes and sort by confidence
            valid_detections = []
            
            for i in range(len(boxes)):
                class_id = int(cls[i])
                class_name = result.names[class_id]
                
                if class_name in self.target_classes:
                    valid_detections.append({
                        'bbox': xywhn[i],
                        'class_id': class_id,
                        'confidence': conf[i]
                    })
            
            # Sort by confidence (highest first)
            valid_detections.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Take top N detections (where N is max_objects)
            for i in range(min(max_objects, len(valid_detections))):
                detection = valid_detections[i]
                
                # Store bbox coordinates and class probability
                object_features[i, 0:4] = detection['bbox']
                object_features[i, 4] = detection['confidence']
                
            return object_features
            
        except Exception as e:
            logger.error(f"Error extracting object features: {e}")
            return object_features
    
    def _extract_spatial_features(self, result):
        """
        Extract spatial features from YOLO's backbone.
        
        Args:
            result: YOLO detection result for a single frame
            
        Returns:
            Numpy array of spatial features
        """
        spatial_dim = FEATURE_CONFIG['spatial_dim']
        
        try:
            # Get feature maps from the backbone directly with YOLO11
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'backbone'):
                # Get the input image tensor
                with torch.no_grad():
                    img_tensor = torch.from_numpy(result.orig_img).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
                    img_tensor = img_tensor.to(self.device)
                    
                    # Pass through model backbone
                    features = self.model.model.backbone(img_tensor)
                    
                    # Get the feature map from the last layer (YOLO11 specific)
                    if isinstance(features, list) or isinstance(features, tuple):
                        features = features[-1]  # Take the deepest feature map
                    
                    # Apply global average pooling to get a fixed-size feature vector
                    features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
                    features = features.reshape(features.size(0), -1)  # Flatten
                    
                    # Convert to numpy and ensure spatial_dim dimensions
                    features_np = features.cpu().numpy()[0]
                    
                    if features_np.size >= spatial_dim:
                        return features_np[:spatial_dim]
                    else:
                        # Pad with zeros if smaller
                        padded = np.zeros(spatial_dim)
                        padded[:features_np.size] = features_np
                        return padded
            
            # Alternative: extract prototypes/feature vectors if available in YOLO11
            elif hasattr(result, 'probs') and result.probs is not None:
                features = result.probs.cpu().numpy()
                
                if features.size >= spatial_dim:
                    return features.flatten()[:spatial_dim]
                else:
                    padded = np.zeros(spatial_dim)
                    padded[:features.size] = features.flatten()
                    return padded
            
            # Another approach: try to access internal tensors from proto/feature outputs
            elif hasattr(result, 'proto') and result.proto is not None:
                proto = result.proto.cpu().numpy()
                # Reshape and process proto features
                proto_flat = proto.reshape(proto.shape[0], -1)
                
                if proto_flat.size >= spatial_dim:
                    return proto_flat[:spatial_dim]
                else:
                    padded = np.zeros(spatial_dim)
                    padded[:proto_flat.size] = proto_flat
                    return padded
            
            # If all else fails, return zeros
            else:
                logger.warning("Could not extract YOLO11 features, returning zeros")
                return np.zeros(spatial_dim)
                
        except Exception as e:
            logger.error(f"Error extracting spatial features: {e}")
            return np.zeros(spatial_dim)
    
    def extract_features_from_frames(self, frames):
        """
        Extract features from a sequence of frames.
        
        Args:
            frames: List of video frames
            
        Returns:
            Dictionary of features for all frames
        """
        all_features = {
            'object_features': [],
            'spatial_features': []
        }
        
        for frame in frames:
            features = self.extract_features(frame)
            all_features['object_features'].append(features['object_features'])
            all_features['spatial_features'].append(features['spatial_features'])
        
        # Convert to numpy arrays
        all_features['object_features'] = np.array(all_features['object_features'])
        all_features['spatial_features'] = np.array(all_features['spatial_features'])
        
        return all_features
    
    def get_concatenated_features(self, frames):
        """
        Extract and concatenate all features from frames.
        
        Args:
            frames: List of video frames
            
        Returns:
            Array of concatenated features for each frame
        """
        features = self.extract_features_from_frames(frames)
        
        num_frames = len(frames)
        concatenated_features = []
        
        for i in range(num_frames):
            # Get features for this frame
            object_feat = features['object_features'][i]
            spatial_feat = features['spatial_features'][i]
            
            # Flatten object features (max_objects * 5 features)
            flat_object_feat = object_feat.flatten()
            
            # Concatenate object and spatial features
            frame_features = np.concatenate([flat_object_feat, spatial_feat])
            concatenated_features.append(frame_features)
        
        return np.array(concatenated_features)

def load_frames(frame_dir):
    """
    Load frames from directory.
    
    Args:
        frame_dir: Directory containing frame images
        
    Returns:
        List of frames as numpy arrays
    """
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])
    frames = []
    
    for f in frame_files:
        frame_path = os.path.join(frame_dir, f)
        frame = cv2.imread(frame_path)
        if frame is not None:
            frames.append(frame)
    
    return frames

def process_split(split_dir, extractor, output_dir):
    """
    Process all videos in a dataset split.
    
    Args:
        split_dir: Directory containing processed video frames
        extractor: YOLOFeatureExtractor instance
        output_dir: Directory to save extracted features
        
    Returns:
        Number of processed videos
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all frame directories ending with _frames
    frame_dirs = []
    for item in os.listdir(split_dir):
        item_path = os.path.join(split_dir, item)
        if os.path.isdir(item_path) and item.endswith('_frames'):
            frame_dirs.append(item_path)
    
    logger.info(f"Found {len(frame_dirs)} videos to process in {split_dir}")
    
    # Process each video
    count = 0
    for frame_dir in tqdm(frame_dirs, desc=f"Processing {os.path.basename(split_dir)}"):
        video_name = os.path.basename(frame_dir).replace('_frames', '')
        
        # Load frames
        frames = load_frames(frame_dir)
        
        if not frames:
            logger.warning(f"No frames found in {frame_dir}")
            continue
        
        # Extract features
        try:
            features = extractor.get_concatenated_features(frames)
            
            # Save features
            output_path = os.path.join(output_dir, f"{video_name}_features.npy")
            np.save(output_path, features)
            
            count += 1
        except Exception as e:
            logger.error(f"Error processing {frame_dir}: {e}")
    
    return count

def extract_features_from_dataset(data_dir=None, output_dir=None, model_path=None, device=None):
    """
    Extract features from the entire dataset.
    
    Args:
        data_dir: Directory containing processed data (if None, use config value)
        output_dir: Directory to save extracted features (if None, use config value)
        model_path: Path to YOLO model weights (if None, use config value)
        device: Computation device ('cuda' or 'cpu') (if None, use GPU if available)
    """
    # Use config values if parameters are not specified
    data_dir = data_dir if data_dir else YOLO_CONFIG['processed_data']
    output_dir = output_dir if output_dir else YOLO_CONFIG['features_output']
    model_path = model_path if model_path else YOLO_CONFIG['model_path']
    
    # Set device based on config if not specified
    if device is None:
        if GPU_CONFIG['use_gpu'] and torch.cuda.is_available():
            device = f"cuda:{GPU_CONFIG['gpu_id']}"
        else:
            device = 'cpu'
            
    logger.info(f"Extracting features from {data_dir} using {model_path}")
    logger.info(f"Saving features to {output_dir}")
    logger.info(f"Using device: {device}")
    
    # Create output directories for accident and non-accident data
    os.makedirs(os.path.join(output_dir, 'accidents'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'non_accidents'), exist_ok=True)

    # Initialize feature extractor
    extractor = YOLOFeatureExtractor(model_path=model_path, device=device)
    
    # Process each category (accidents and non-accidents)
    total_processed = 0
    
    # Process accident videos
    accident_dir = os.path.join(data_dir, 'accidents')
    accident_output_dir = os.path.join(output_dir, 'accidents')
    
    if os.path.exists(accident_dir):
        logger.info(f"Processing accident videos")
        accident_count = process_split(accident_dir, extractor, accident_output_dir)
        logger.info(f"Processed {accident_count} accident videos")
        total_processed += accident_count
    else:
        logger.warning(f"Directory not found: {accident_dir}")
    
    # Process non-accident videos
    non_accident_dir = os.path.join(data_dir, 'non_accidents')
    non_accident_output_dir = os.path.join(output_dir, 'non_accidents')
    
    if os.path.exists(non_accident_dir):
        logger.info(f"Processing non-accident videos")
        non_accident_count = process_split(non_accident_dir, extractor, non_accident_output_dir)
        logger.info(f"Processed {non_accident_count} non-accident videos")
        total_processed += non_accident_count
    else:
        logger.warning(f"Directory not found: {non_accident_dir}")
    
    logger.info(f"Feature extraction completed. Processed {total_processed} videos.")

def main():
    """Main function to execute the feature extraction using config values."""
    # Extract features using config values
    extract_features_from_dataset()

if __name__ == "__main__":
    main()