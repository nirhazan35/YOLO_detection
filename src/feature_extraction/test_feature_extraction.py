#!/usr/bin/env python
# Test script for YOLO feature extraction

import os
import sys
import argparse
import logging
import numpy as np
import torch
import traceback
import cv2

# Add parent directory to path if necessary
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from feature_extraction.config import YOLO_CONFIG, FEATURE_CONFIG, GPU_CONFIG
from feature_extraction.feature_extraction import YOLOFeatureExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("feature_extraction_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def verify_model_capabilities(model_path=None):
    """
    Check if the YOLO model has the necessary capabilities for feature extraction.
    
    Args:
        model_path: Path to YOLO model weights
        
    Returns:
        Boolean indicating if the model is compatible
    """
    try:
        # Use config if no path provided
        if model_path is None:
            model_path = YOLO_CONFIG['model_path']
            
        logger.info(f"Verifying model capabilities for {model_path}")
        
        # Import and load model
        from ultralytics import YOLO
        model = YOLO(model_path)
        
        # Check for names attribute
        if not hasattr(model, 'names'):
            logger.error("Model does not have 'names' attribute")
            return False
            
        logger.info(f"Model has {len(model.names)} classes")
        
        # Check if target classes exist in model
        target_classes = FEATURE_CONFIG['object_classes']
        found_classes = [cls for cls in target_classes if cls in model.names.values()]
        
        if len(found_classes) < len(target_classes):
            missing = set(target_classes) - set(found_classes)
            logger.warning(f"Some target classes not found in model: {missing}")
        
        # Check model structure
        if not hasattr(model, 'model'):
            logger.error("Model does not have expected structure (missing 'model' attribute)")
            return False
            
        # Check for backbone
        if not hasattr(model.model, 'backbone'):
            logger.warning("Model does not have 'backbone' attribute")
        else:
            logger.info("Model has backbone")
            
        # Check for neck
        if not hasattr(model.model, 'neck'):
            logger.warning("Model does not have 'neck' attribute")
        else:
            logger.info("Model has neck")
        
        # Create a test image and run prediction
        test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Run inference
        results = model(test_img, verbose=False)
        
        # Check if result has expected attributes
        if not hasattr(results[0], 'boxes'):
            logger.error("Results missing 'boxes' attribute")
            return False
            
        # Check for features attribute or method
        features_available = False
        
        # Try direct features access
        if hasattr(results[0], 'features'):
            logger.info("Results have 'features' attribute")
            features_available = True
            
        # Try predict with save_features
        try:
            results = model.predict(test_img, verbose=False, save_features=True)
            if hasattr(results[0], 'features'):
                logger.info("Model supports save_features")
                features_available = True
        except Exception as e:
            logger.warning(f"Model does not support save_features: {e}")
            
        if not features_available:
            logger.warning("No direct way to access features - will need to use hooks")
        
        return True
    except Exception as e:
        logger.error(f"Error verifying model: {e}")
        logger.error(traceback.format_exc())
        return False
        
def validate_feature_dimensions():
    """
    Verify that the feature extractor outputs the expected dimensions.
    
    Returns:
        Boolean indicating if dimensions are correct
    """
    try:
        logger.info("Validating feature dimensions...")
        
        # Create feature configuration
        feature_config = {
            'model_path': YOLO_CONFIG['model_path'],
            'object_classes': FEATURE_CONFIG['object_classes'],
            'max_objects': FEATURE_CONFIG['max_objects'],
            'spatial_dim': FEATURE_CONFIG['spatial_dim'],
            'use_gpu': GPU_CONFIG['use_gpu'],
            'gpu_id': GPU_CONFIG['gpu_id']
        }
        
        # Create feature extractor
        extractor = YOLOFeatureExtractor(config=feature_config)
        
        # Create dummy frame
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Extract features from single frame
        features = extractor.extract_features(dummy_frame)
        
        # Check object features shape
        obj_shape = features['object_features'].shape
        expected_obj_shape = (FEATURE_CONFIG['max_objects'], 5)
        
        if obj_shape != expected_obj_shape:
            logger.error(f"Object features shape mismatch! Expected {expected_obj_shape}, got {obj_shape}")
            return False
            
        # Check spatial features shape
        spatial_shape = features['spatial_features'].shape
        expected_spatial_shape = (FEATURE_CONFIG['spatial_dim'],)
        
        if spatial_shape != expected_spatial_shape:
            logger.error(f"Spatial features shape mismatch! Expected {expected_spatial_shape}, got {spatial_shape}")
            return False
            
        # Extract concatenated features
        concat_features = extractor.get_concatenated_features([dummy_frame])
        
        # Check concatenated shape
        concat_shape = concat_features.shape
        expected_dim = (FEATURE_CONFIG['max_objects'] * 5) + FEATURE_CONFIG['spatial_dim']
        expected_concat_shape = (1, expected_dim)
        
        if concat_shape != expected_concat_shape:
            logger.error(f"Concatenated features shape mismatch! Expected {expected_concat_shape}, got {concat_shape}")
            return False
            
        logger.info(f"Feature dimensions validated successfully: {expected_dim} total features per frame")
        return True
    except Exception as e:
        logger.error(f"Error validating dimensions: {e}")
        logger.error(traceback.format_exc())
        return False

def test_feature_extraction_pipeline(test_dir=None):
    """
    Test the full feature extraction pipeline on a small set of test frames.
    
    Args:
        test_dir: Directory with test frames (if None, creates random frames)
        
    Returns:
        Boolean indicating if test passed
    """
    try:
        # Create feature configuration
        feature_config = {
            'model_path': YOLO_CONFIG['model_path'],
            'object_classes': FEATURE_CONFIG['object_classes'],
            'max_objects': FEATURE_CONFIG['max_objects'],
            'spatial_dim': FEATURE_CONFIG['spatial_dim'],
            'use_gpu': GPU_CONFIG['use_gpu'],
            'gpu_id': GPU_CONFIG['gpu_id']
        }
        
        # Create feature extractor
        extractor = YOLOFeatureExtractor(config=feature_config)
        
        # Generate test frames if no directory provided
        if test_dir is None or not os.path.exists(test_dir):
            logger.info("Creating synthetic test frames...")
            frames = []
            for i in range(5):  # Create 5 test frames
                frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                # Add some shapes to make it more realistic
                cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 2)
                cv2.circle(frame, (400, 240), 50, (0, 0, 255), -1)
                frames.append(frame)
        else:
            # Load frames from directory
            logger.info(f"Loading test frames from {test_dir}")
            frame_files = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png'))]
            frames = []
            for f in sorted(frame_files):
                frame_path = os.path.join(test_dir, f)
                frame = cv2.imread(frame_path)
                if frame is not None:
                    frames.append(frame)
            
            if not frames:
                logger.error(f"No frames found in {test_dir}")
                return False
                
        # Extract features
        logger.info(f"Extracting features from {len(frames)} frames")
        features = extractor.get_concatenated_features(frames)
        
        # Check shapes
        if features.shape[0] != len(frames):
            logger.error(f"Feature extraction failed! Expected {len(frames)} feature sets, got {features.shape[0]}")
            return False
            
        # Check for all zeros (potential failure)
        if np.all(features == 0):
            logger.error("All features are zero! Extraction may have failed.")
            return False
            
        # Check for NaN values
        if np.isnan(features).any():
            logger.error("Feature array contains NaN values!")
            return False
            
        # Test saving features
        output_dir = os.path.join("tests", "test_results")
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, "test_features.npy")
        np.save(output_path, features)
        
        # Verify saved file
        if not os.path.exists(output_path):
            logger.error(f"Failed to save features to {output_path}")
            return False
            
        # Load saved features and compare
        loaded_features = np.load(output_path)
        if not np.array_equal(features, loaded_features):
            logger.error("Loaded features do not match original features!")
            return False
            
        logger.info(f"Feature extraction pipeline test completed successfully!")
        logger.info(f"Feature array shape: {features.shape}")
        
        # Print sample of feature values
        logger.info(f"Feature sample (first frame, first 10 values): {features[0, :10]}")
        
        # Calculate some statistics
        feature_mean = np.mean(features)
        feature_std = np.std(features)
        feature_nonzero = np.count_nonzero(features) / features.size * 100
        
        logger.info(f"Feature statistics - Mean: {feature_mean:.6f}, Std: {feature_std:.6f}, Non-zero: {feature_nonzero:.2f}%")
        
        return True
    except Exception as e:
        logger.error(f"Error testing pipeline: {e}")
        logger.error(traceback.format_exc())
        return False

def main():
    """Main function to run tests."""
    parser = argparse.ArgumentParser(description="Test YOLO feature extraction")
    parser.add_argument("--test-all", action="store_true", help="Run all tests")
    parser.add_argument("--check-model", action="store_true", help="Check model capabilities")
    parser.add_argument("--validate-dims", action="store_true", help="Validate feature dimensions")
    parser.add_argument("--test-pipeline", action="store_true", help="Test full pipeline")
    parser.add_argument("--test-dir", type=str, help="Directory with test frames")
    
    args = parser.parse_args()
    
    # If no arguments provided, run all tests
    if not (args.check_model or args.validate_dims or args.test_pipeline or args.test_all):
        args.test_all = True
        
    success = True
    
    # Run tests based on arguments
    if args.check_model or args.test_all:
        logger.info("=== Checking Model Capabilities ===")
        model_ok = verify_model_capabilities()
        logger.info(f"Model check {'PASSED' if model_ok else 'FAILED'}")
        success = success and model_ok
        
    if args.validate_dims or args.test_all:
        logger.info("=== Validating Feature Dimensions ===")
        dims_ok = validate_feature_dimensions()
        logger.info(f"Dimension validation {'PASSED' if dims_ok else 'FAILED'}")
        success = success and dims_ok
        
    if args.test_pipeline or args.test_all:
        logger.info("=== Testing Feature Extraction Pipeline ===")
        pipeline_ok = test_feature_extraction_pipeline(args.test_dir)
        logger.info(f"Pipeline test {'PASSED' if pipeline_ok else 'FAILED'}")
        success = success and pipeline_ok
        
    # Final result
    if success:
        logger.info("All tests PASSED!")
        return 0
    else:
        logger.error("Some tests FAILED!")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 