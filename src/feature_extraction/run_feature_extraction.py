import os
import sys
import time
import logging
import json
import traceback
import datetime
import torch
import glob
import pickle
import cv2

# Add parent directory to path if necessary
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from feature_extraction.config import YOLO_CONFIG, GPU_CONFIG, DATASET_CONFIG, FEATURE_CONFIG
from feature_extraction.feature_extraction import extract_features_from_dataset
from feature_extraction.feature_extraction import YOLOFeatureExtractor

# Create logs directory and feature_extraction_logs subdirectory
logs_dir = os.path.join(os.path.dirname(parent_dir), "logs")
feature_extraction_logs_dir = os.path.join(logs_dir, "feature_extraction_logs")
os.makedirs(feature_extraction_logs_dir, exist_ok=True)

# Generate log filename with timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join(feature_extraction_logs_dir, f"run_feature_extraction_{timestamp}.log")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"Logging to {log_filename}")

def main():
    configure_logging()
    
    logging.info("Starting feature extraction process...")
    
    # Measure timing
    start_time = time.time()
    
    try:
        # Check required directories exist
        check_directories()
        
        # Create output directories
        create_output_dirs()
        
        # Create feature configuration to pass to the model
        feature_config = {
            'model_path': YOLO_CONFIG['model_path'],
            'object_classes': FEATURE_CONFIG['object_classes'],
            'max_objects': FEATURE_CONFIG['max_objects'],
            'spatial_dim': FEATURE_CONFIG['spatial_dim'],
            'use_gpu': GPU_CONFIG['use_gpu'],
            'gpu_id': GPU_CONFIG['gpu_id']
        }
        
        # Extract features from accident videos
        logging.info("Extracting features from accident videos...")
        extract_features_from_category(
            input_dir=DATASET_CONFIG['accident_dir'],
            output_file=os.path.join(DATASET_CONFIG['feature_dir'], 'accident_features.pkl'),
            feature_config=feature_config
        )
        
        # Extract features from non-accident videos
        logging.info("Extracting features from non-accident videos...")
        extract_features_from_category(
            input_dir=DATASET_CONFIG['non_accident_dir'],
            output_file=os.path.join(DATASET_CONFIG['feature_dir'], 'non_accident_features.pkl'),
            feature_config=feature_config
        )
        
        end_time = time.time()
        logging.info(f"Feature extraction completed in {end_time - start_time:.2f} seconds")
        
    except Exception as e:
        logging.error(f"Error during feature extraction: {str(e)}")
        logging.error(traceback.format_exc())
        sys.exit(1)

def extract_features_from_category(input_dir, output_file, feature_config):
    """
    Extract features from all videos in a category.
    
    Args:
        input_dir: Directory containing processed video clips
        output_file: Output file to save extracted features
        feature_config: Configuration for the feature extractor
    """
    # Get list of video files
    video_files = glob.glob(os.path.join(input_dir, '*.mp4'))
    logging.info(f"Found {len(video_files)} videos in {input_dir}")
    
    # Create feature extractor
    feature_extractor = YOLOFeatureExtractor(config=feature_config)
    
    # Extract features from each video
    all_video_features = {}
    
    for video_path in video_files:
        try:
            video_filename = os.path.basename(video_path)
            logging.info(f"Processing {video_filename}...")
            
            # Extract frames
            frames = extract_frames_from_video(video_path)
            if not frames:
                logging.warning(f"No frames extracted from {video_filename}")
                continue
                
            logging.info(f"Extracted {len(frames)} frames from {video_filename}")
            
            # Extract features from frames
            video_features = feature_extractor.extract_features_from_frames(frames)
            
            # Concatenate features for each frame
            concatenated_features = feature_extractor.get_concatenated_features(frames)
            
            # Verify feature dimensions
            expected_dim = (feature_config['max_objects'] * 5) + feature_config['spatial_dim']
            if concatenated_features.shape[1] != expected_dim:
                logging.warning(f"Feature dimension mismatch in {video_filename}! Expected {expected_dim}, got {concatenated_features.shape[1]}")
            
            # Store features
            all_video_features[video_filename] = concatenated_features
            
        except Exception as e:
            logging.error(f"Error processing {video_path}: {str(e)}")
            logging.error(traceback.format_exc())
    
    # Save features to file
    with open(output_file, 'wb') as f:
        pickle.dump(all_video_features, f)
    
    logging.info(f"Saved features for {len(all_video_features)} videos to {output_file}")
    return len(all_video_features)

def extract_frames_from_video(video_path):
    """
    Extract frames from a video file.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        List of frames as numpy arrays
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        logging.error(f"Could not open video: {video_path}")
        return frames
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    return frames

def configure_logging():
    """Configure logging for feature extraction"""
    logging.info("Feature extraction logging configured")

def check_directories():
    """Check that all required directories exist"""
    # Check input directories
    if not os.path.exists(DATASET_CONFIG['accident_dir']):
        raise FileNotFoundError(f"Accident videos directory not found: {DATASET_CONFIG['accident_dir']}")
    
    if not os.path.exists(DATASET_CONFIG['non_accident_dir']):
        raise FileNotFoundError(f"Non-accident videos directory not found: {DATASET_CONFIG['non_accident_dir']}")

def create_output_dirs():
    """Create output directories if they don't exist"""
    # Create main feature directory
    os.makedirs(DATASET_CONFIG['feature_dir'], exist_ok=True)
    
    # Create nested feature directories for different splits
    os.makedirs(os.path.join(DATASET_CONFIG['feature_dir'], 'train'), exist_ok=True)
    os.makedirs(os.path.join(DATASET_CONFIG['feature_dir'], 'val'), exist_ok=True)
    os.makedirs(os.path.join(DATASET_CONFIG['feature_dir'], 'test'), exist_ok=True)
    
    logging.info(f"Ensured feature output directories exist: {DATASET_CONFIG['feature_dir']}")

if __name__ == "__main__":
    main()