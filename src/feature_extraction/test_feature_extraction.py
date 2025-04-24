import os
import sys
import torch
import numpy as np
import cv2
import logging
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Add the parent directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from feature_extraction.config import YOLO_CONFIG, GPU_CONFIG
from feature_extraction.feature_extraction import YOLOFeatureExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def download_yolo11n():
    """
    Download YOLO11n model if not already present
    """
    model_path = YOLO_CONFIG['model_path']
    if not os.path.exists(model_path):
        logger.info(f"Downloading {model_path} model...")
        try:
            # This will download the model automatically
            YOLO(model_path)
            logger.info("Model downloaded successfully")
        except Exception as e:
            logger.error(f"Error downloading model: {e}")
            sys.exit(1)
    else:
        logger.info(f"{model_path} model already exists")
    
    return model_path

def test_on_sample_frame(model_path):
    """
    Test feature extraction on a sample frame
    """
    # Find a sample frame from the processed data
    data_dir = YOLO_CONFIG['processed_data']
    sample_frame = None
    
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".jpg"):
                sample_frame = os.path.join(root, file)
                break
        if sample_frame:
            break
    
    if not sample_frame:
        logger.error("No sample frames found in processed data")
        # Create a test frame with some objects
        logger.info("Creating a dummy test frame")
        sample_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw some rectangles to simulate objects
        cv2.rectangle(sample_frame, (100, 100), (200, 200), (0, 255, 0), -1)
        cv2.rectangle(sample_frame, (300, 150), (400, 250), (0, 0, 255), -1)
        # Save the test frame
        os.makedirs("test_data", exist_ok=True)
        test_frame_path = "test_data/test_frame.jpg"
        cv2.imwrite(test_frame_path, sample_frame)
        sample_frame = test_frame_path
    
    # Load the sample frame
    logger.info(f"Testing feature extraction on {sample_frame}")
    frame = cv2.imread(sample_frame)
    
    if frame is None:
        logger.error(f"Could not load frame from {sample_frame}")
        sys.exit(1)
    
    # Initialize the feature extractor
    extractor = YOLOFeatureExtractor(model_path=model_path)
    
    # Extract features
    features = extractor.extract_features(frame)
    
    logger.info("Feature extraction successful")
    logger.info(f"Object features shape: {features['object_features'].shape}")
    logger.info(f"Spatial features shape: {features['spatial_features'].shape}")
    
    # Display results
    object_features = features['object_features']
    
    # Check for valid detections by looking at confidence scores in column 4
    valid_detections = object_features[:, 4] > 0
    num_detections = np.sum(valid_detections)
    
    logger.info(f"Number of valid detections: {num_detections}")
    
    # Visualize the frame with detections
    visualization = frame.copy()
    
    h, w = frame.shape[:2]
    for i in range(len(object_features)):
        if object_features[i, 4] > 0:  # If confidence > 0
            # Denormalize bounding box coordinates
            x_center, y_center = int(object_features[i, 0] * w), int(object_features[i, 1] * h)
            width, height = int(object_features[i, 2] * w), int(object_features[i, 3] * h)
            
            # Calculate top-left and bottom-right corners
            x1 = max(0, x_center - width // 2)
            y1 = max(0, y_center - height // 2)
            x2 = min(w, x_center + width // 2)
            y2 = min(h, y_center + height // 2)
            
            # Draw rectangle and confidence score
            cv2.rectangle(visualization, (x1, y1), (x2, y2), (0, 255, 0), 2)
            conf_text = f"{object_features[i, 4]:.2f}"
            cv2.putText(visualization, conf_text, (x1, y1-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Save the visualization
    os.makedirs("test_results", exist_ok=True)
    viz_path = "test_results/detection_visualization.jpg"
    cv2.imwrite(viz_path, visualization)
    logger.info(f"Visualization saved to {viz_path}")
    
    # Save extracted features for inspection
    features_path = "test_results/sample_features.npz"
    np.savez(
        features_path, 
        object_features=features['object_features'],
        spatial_features=features['spatial_features']
    )
    logger.info(f"Features saved to {features_path}")
    
    return True

def main():
    """Main test function"""
    # Check CUDA availability
    if torch.cuda.is_available() and GPU_CONFIG['use_gpu']:
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        device = "cuda"
    else:
        logger.info("CUDA not available or disabled in config, using CPU")
        device = "cpu"
    
    # Download model if needed
    model_path = download_yolo11n()
    
    # Test on a sample frame
    test_on_sample_frame(model_path)
    
    logger.info("All tests completed successfully")

if __name__ == "__main__":
    main() 