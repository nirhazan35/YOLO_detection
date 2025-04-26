import os
import sys
import numpy as np
import torch
import logging
import json
import traceback
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
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(f"cuda:{config['gpu_id']}" if config['use_gpu'] and torch.cuda.is_available() else "cpu")
        
        # Load YOLO model
        try:
            self.model = self.load_model()
            self.classes = self.model.names
            
            # Create class mapping using model names
            self.class_map = {}
            for i, name in self.classes.items():
                if name in self.config['object_classes']:
                    self.class_map[name] = i
            
            logging.info(f"YOLO model loaded successfully on {self.device}")
            logging.info(f"Available classes: {self.classes}")
            logging.info(f"Class mapping for configured classes: {self.class_map}")
            self.model_loaded = True
        except Exception as e:
            logging.error(f"Failed to load YOLO model: {str(e)}")
            logging.error(traceback.format_exc())
            self.model_loaded = False
    
    def load_model(self):
        """Load the YOLO model using the ultralyticsplus API"""
        try:
            import torch
            from ultralytics import YOLO
            
            model_path = self.config['model_path']
            if not os.path.exists(model_path):
                logging.error(f"Model path does not exist: {model_path}")
                raise FileNotFoundError(f"Model file not found at {model_path}")
            
            # Load the model
            logging.info(f"Loading YOLO model from {model_path}")
            model = YOLO(model_path)
            
            # Move model to the appropriate device
            model.to(self.device)
            return model
        except ImportError as e:
            logging.error(f"Required packages not installed: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Error loading YOLO model: {str(e)}")
            raise
    
    def extract_object_features(self, frame):
        """
        Extract object detection features from a video frame
        Returns a vector of detected objects with their class, confidence, and bounding box
        """
        if not self.model_loaded:
            logging.warning("Could not extract YOLO11 features, returning zeros")
            # Return zeros with the expected shape
            # [class_id, confidence, x1, y1, width, height] for max_objects
            return np.zeros((self.config['max_objects'], 5))
        
        try:
            # Convert frame to RGB if it's BGR (OpenCV default)
            if frame.shape[2] == 3:  # If it has 3 channels
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame
            
            # Run inference
            results = self.model(frame_rgb, verbose=False)
            
            # Extract detections
            detections = []
            
            if len(results) > 0:
                # Get predictions from first result
                result = results[0]
                boxes = result.boxes
                
                # Convert boxes to desired format
                for i, box in enumerate(boxes):
                    cls_id = int(box.cls[0].item())
                    cls_name = self.model.names[cls_id]
                    
                    # Filter by configured classes
                    if cls_name in self.config['object_classes']:
                        conf = box.conf[0].item()
                        xyxy = box.xyxy[0].cpu().numpy()  # x1, y1, x2, y2
                        
                        # Calculate width and height from coordinates
                        width = xyxy[2] - xyxy[0]
                        height = xyxy[3] - xyxy[1]
                        
                        # Store as [class_id, confidence, x1, y1, width, height]
                        detections.append([cls_id, conf, xyxy[0], xyxy[1], width, height])
            
            # Sort by confidence (highest first)
            detections = sorted(detections, key=lambda x: x[1], reverse=True)
            
            # Take only the top max_objects
            detections = detections[:self.config['max_objects']]
            
            # Pad with zeros if needed
            if len(detections) < self.config['max_objects']:
                padding = np.zeros((self.config['max_objects'] - len(detections), 5))
                detections = np.vstack([detections, padding]) if detections else padding
            
            return np.array(detections)
        
        except Exception as e:
            logging.error(f"Error extracting object features: {str(e)}")
            logging.error(traceback.format_exc())
            return np.zeros((self.config['max_objects'], 5))
    
    def _extract_spatial_features(self, frame):
        """
        Extract spatial features from a video frame using YOLO11's internal feature maps
        Returns a feature vector representing the spatial information
        """
        if not self.model_loaded:
            logging.warning("Could not extract YOLO11 features, returning zeros")
            return np.zeros(self.config['spatial_dim'])
        
        try:
            # Convert frame to RGB if it's BGR (OpenCV default)
            if frame.shape[2] == 3:  # If it has 3 channels
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame
            
            # Convert to tensor and normalize
            img = torch.from_numpy(frame_rgb).to(self.device)
            img = img.permute(2, 0, 1).float() / 255.0  # HWC -> CHW
            
            # Add batch dimension
            if len(img.shape) == 3:
                img = img.unsqueeze(0)
            
            # Set model to evaluation mode and disable gradients
            self.model.model.eval()
            with torch.no_grad():
                # Forward pass through the model's backbone and neck
                try:
                    # Try getting the features through the model API
                    # Get the neck (FPN) features which are richer for spatial understanding
                    features = self.model.model.forward_backbone(img)
                    logging.info(f"Extracted features with shape: {[f.shape for f in features]}")
                    
                    # Process the feature maps
                    # Take the first feature map which has the highest resolution
                    feature_map = features[0]  # Usually [1, C, H, W]
                    
                    # Global average pooling to reduce spatial dimensions
                    pooled_features = torch.nn.functional.adaptive_avg_pool2d(feature_map, (1, 1))
                    pooled_features = pooled_features.view(pooled_features.size(0), -1)  # Flatten
                    
                    # Extract feature vector
                    feature_vector = pooled_features[0].cpu().numpy()  # Get the first batch item
                    
                    # If needed, adapt dimensions to match spatial_dim
                    if len(feature_vector) > self.config['spatial_dim']:
                        # Use PCA-like approach: take first N components
                        feature_vector = feature_vector[:self.config['spatial_dim']]
                    elif len(feature_vector) < self.config['spatial_dim']:
                        # Pad with zeros
                        feature_vector = np.pad(feature_vector, (0, self.config['spatial_dim'] - len(feature_vector)))
                    
                    return feature_vector
                    
                except (AttributeError, TypeError) as e:
                    logging.warning(f"Could not access model backbone directly: {e}. Trying alternative method.")
                    
                    # Alternative approach: Use YOLO results for features
                    results = self.model(img, verbose=False)
                    if not hasattr(results[0], 'features') or results[0].features is None:
                        raise AttributeError("YOLO results don't contain features")
                        
                    # Get features from YOLO results
                    feature_vector = results[0].features.cpu().numpy()
                    
                    # Adapt to required dimension
                    if len(feature_vector) > self.config['spatial_dim']:
                        feature_vector = feature_vector[:self.config['spatial_dim']]
                    elif len(feature_vector) < self.config['spatial_dim']:
                        feature_vector = np.pad(feature_vector, (0, self.config['spatial_dim'] - len(feature_vector)))
                    
                    return feature_vector
                    
        except Exception as e:
            logging.error(f"Error extracting spatial features: {str(e)}")
            logging.error(traceback.format_exc())
            return np.zeros(self.config['spatial_dim'])
    
    def extract_spatial_features(self, frame):
        """
        Extract spatial features from a video frame
        Returns a feature vector representing the spatial information
        """
        # Try using YOLO's internal features first
        try:
            features = self._extract_spatial_features(frame)
            if np.any(features != 0):  # Check if features are non-zero
                return features
            else:
                logging.warning("YOLO11 feature extraction returned zeros, falling back to basic method")
        except Exception as e:
            logging.warning(f"Failed to extract YOLO11 features: {str(e)}, falling back to basic method")
        
        # Fallback method if YOLO features can't be extracted
        try:
            # Resize to a smaller dimension
            target_size = (self.config['spatial_dim'] // 16, self.config['spatial_dim'] // 16)
            resized_frame = cv2.resize(frame, target_size)
            
            # Convert to grayscale
            gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
            
            # Normalize
            normalized = gray_frame.astype(np.float32) / 255.0
            
            # Flatten
            features = normalized.flatten()
            
            # If needed, use PCA or other dimensionality reduction to get to spatial_dim
            if len(features) > self.config['spatial_dim']:
                # Simple approach: subsample
                indices = np.linspace(0, len(features)-1, self.config['spatial_dim']).astype(int)
                features = features[indices]
            elif len(features) < self.config['spatial_dim']:
                # Pad with zeros
                features = np.pad(features, (0, self.config['spatial_dim'] - len(features)))
            
            return features
        except Exception as e:
            logging.error(f"Error in fallback spatial feature extraction: {str(e)}")
            logging.error(traceback.format_exc())
            return np.zeros(self.config['spatial_dim'])

    def extract_features(self, frame):
        """
        Extract features from a single frame.
        
        Args:
            frame: Input image frame (BGR format from OpenCV)
            
        Returns:
            Dictionary containing object and spatial features
        """
        try:
            # Convert frame to RGB for YOLO processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Extract object detection features
            object_features = self.extract_object_features(frame_rgb)
            
            # Extract spatial features
            spatial_features = self.extract_spatial_features(frame_rgb)
            
            return {
                'object_features': object_features,
                'spatial_features': spatial_features
            }
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            logger.error(traceback.format_exc())
            # Return empty features if extraction fails
            return {
                'object_features': np.zeros((self.config['max_objects'], 5)),  # max_objects, 5 features each
                'spatial_features': np.zeros(self.config['spatial_dim'])       # spatial_dim features
            }
    
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
        
        for i, frame in enumerate(frames):
            try:
                features = self.extract_features(frame)
                all_features['object_features'].append(features['object_features'])
                all_features['spatial_features'].append(features['spatial_features'])
            except Exception as e:
                logger.error(f"Error processing frame {i}: {str(e)}")
                logger.error(traceback.format_exc())
                # Use zeros for this frame
                all_features['object_features'].append(np.zeros((FEATURE_CONFIG['max_objects'], 5)))
                all_features['spatial_features'].append(np.zeros(FEATURE_CONFIG['spatial_dim']))
        
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
            try:
                # Get features for this frame
                object_feat = features['object_features'][i]
                spatial_feat = features['spatial_features'][i]
                
                # Flatten object features (max_objects * 5 features)
                flat_object_feat = object_feat.flatten()
                
                # Concatenate object and spatial features
                frame_features = np.concatenate([flat_object_feat, spatial_feat])
                concatenated_features.append(frame_features)
                
                # Verify feature dimensions
                expected_dim = (FEATURE_CONFIG['max_objects'] * 5) + FEATURE_CONFIG['spatial_dim']
                if frame_features.shape[0] != expected_dim:
                    logger.warning(f"Feature dimension mismatch! Expected {expected_dim}, got {frame_features.shape[0]}")
            except Exception as e:
                logger.error(f"Error concatenating features for frame {i}: {str(e)}")
                logger.error(traceback.format_exc())
                # Use zeros for this frame
                expected_dim = (FEATURE_CONFIG['max_objects'] * 5) + FEATURE_CONFIG['spatial_dim']
                concatenated_features.append(np.zeros(expected_dim))
        
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
        else:
            logger.warning(f"Could not read frame: {frame_path}")
    
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
    errors = 0
    
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
            
            # Verify the saved file
            if os.path.exists(output_path):
                feature_shape = features.shape
                logger.info(f"Saved features for {video_name} with shape {feature_shape}")
                count += 1
            else:
                logger.error(f"Failed to save features for {video_name}")
                errors += 1
                
        except Exception as e:
            logger.error(f"Error processing {frame_dir}: {e}")
            logger.error(traceback.format_exc())
            errors += 1
    
    if errors > 0:
        logger.warning(f"Completed with {errors} errors out of {len(frame_dirs)} videos")
        
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
    extractor = YOLOFeatureExtractor(config=GPU_CONFIG)
    
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