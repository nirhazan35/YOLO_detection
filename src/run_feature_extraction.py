"""
Road Accident Detection - Feature Extraction Pipeline

This script extracts features from pre-processed video frames using YOLO11 for
object detection and spatial features, and a custom CNN for optical flow features.

Usage:
    python src/run_feature_extraction.py
"""

import os
import sys
import time
import logging
import traceback
import glob
import numpy as np
import cv2
from pathlib import Path
import json
import torch
import re

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

def main():
    # Add the project root directory to the Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
        logger.info(f"Added project root to path: {parent_dir}")
    
    # Import after modifying path
    try:
        from src.feature_extraction.config import INPUT_PATHS, OUTPUT_CONFIG, FRAME_PATTERNS, BATCH_PROCESSING, YOLO_CONFIG, FEATURE_CONFIG
        from src.feature_extraction.feature_extractor import YOLO11FeatureExtractor
        from src.feature_extraction.flow_feature_extractor import SimpleFlowFeatureExtractor, TorchFlowFeatureExtractor
        from src.feature_extraction.utils import save_features, visualize_detection
        logger.info("Successfully imported required modules")
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        logger.error(f"Import path details: {traceback.format_exc()}")
        logger.error(f"Python path: {sys.path}")
        logger.error("Make sure you're running this script from the project root directory.")
        return 1
    
    # Check GPU availability
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("Using CPU for feature extraction")
    
    # Initialize feature extractors
    try:
        # YOLO feature extractor for RGB frames
        yolo_extractor = YOLO11FeatureExtractor(
            model_path=YOLO_CONFIG["model_path"],
            device=YOLO_CONFIG["device"],
            max_objects=YOLO_CONFIG["max_objects"]
        )
        
        # Flow feature extractor (simple CV-based)
        simple_flow_extractor = SimpleFlowFeatureExtractor(
            feature_dim=FEATURE_CONFIG["flow_feature_dim"]
        )
        
        # PyTorch-based flow extractor for better features
        torch_flow_extractor = TorchFlowFeatureExtractor(
            feature_dim=FEATURE_CONFIG["flow_feature_dim"],
            use_gpu=use_gpu
        )
        
        logger.info(f"Initialized feature extractors")
    except Exception as e:
        logger.error(f"Failed to initialize feature extractors: {e}")
        logger.error(traceback.format_exc())
        return 1
    
    # Create output directories
    for output_dir in [OUTPUT_CONFIG["rgb_features_dir"], OUTPUT_CONFIG["flow_features_dir"], OUTPUT_CONFIG["combined_features_dir"]]:
        # Create category-specific subdirectories
        for category in ["accidents", "non_accidents", "test"]:
            os.makedirs(os.path.join(output_dir, category), exist_ok=True)
    
    # Process each category (accidents, non-accidents, test)
    total_videos = 0
    total_frames = 0
    start_time = time.time()
    
    # Create organized video list by matching frame and flow directories
    for category, input_dir in INPUT_PATHS.items():
        if not os.path.exists(input_dir):
            logger.warning(f"Input directory not found: {input_dir}")
            continue
        
        # Get all directories for frames and flow
        frame_dirs = []
        flow_dirs = []
        
        all_dirs = glob.glob(os.path.join(input_dir, "*"))
        for dir_path in all_dirs:
            if not os.path.isdir(dir_path):
                continue
                
            dir_name = os.path.basename(dir_path)
            if '_frames' in dir_name:
                frame_dirs.append(dir_path)
            elif '_flow' in dir_name:
                flow_dirs.append(dir_path)
        
        logger.info(f"Found {len(frame_dirs)} frame directories and {len(flow_dirs)} flow directories in {category}")
        
        # Extract base video names from directories
        video_names = set()
        for dir_path in frame_dirs + flow_dirs:
            dir_name = os.path.basename(dir_path)
            video_name = dir_name.replace('_frames', '').replace('_flow', '')
            video_names.add(video_name)
        
        logger.info(f"Found {len(video_names)} unique videos in {category}")
        
        # Process each video
        for video_idx, video_name in enumerate(sorted(video_names)):
            logger.info(f"Processing video {video_idx+1}/{len(video_names)}: {video_name}")
            
            # Find corresponding frame and flow directories
            frame_dir = None
            flow_dir = None
            
            for dir_path in frame_dirs:
                if os.path.basename(dir_path).startswith(f"{video_name}_frames"):
                    frame_dir = dir_path
                    break
                    
            for dir_path in flow_dirs:
                if os.path.basename(dir_path).startswith(f"{video_name}_flow"):
                    flow_dir = dir_path
                    break
            
            if not frame_dir and not flow_dir:
                logger.warning(f"No frame or flow directories found for {video_name}")
                continue
                
            try:
                # Extract features from frame directory
                rgb_features = []
                if frame_dir:
                    logger.info(f"Processing frames from {os.path.basename(frame_dir)}")
                    
                    # Get RGB frame paths
                    rgb_frame_pattern = os.path.join(frame_dir, "frame_*.jpg")
                    rgb_frame_paths = sorted(glob.glob(rgb_frame_pattern))
                    
                    if not rgb_frame_paths:
                        logger.warning(f"No frames found in {frame_dir}")
                    else:
                        logger.info(f"Found {len(rgb_frame_paths)} RGB frames")
                        
                        # Process in batches
                        batch_size = BATCH_PROCESSING["batch_size"]
                        
                        for i in range(0, len(rgb_frame_paths), batch_size):
                            batch_paths = rgb_frame_paths[i:i+batch_size]
                            batch_frames = [cv2.imread(path) for path in batch_paths]
                            
                            # Skip invalid frames
                            batch_frames = [frame for frame in batch_frames if frame is not None]
                            if not batch_frames:
                                continue
                            
                            # Extract RGB features
                            batch_rgb_features = []
                            for frame in batch_frames:
                                features = yolo_extractor.extract_features(frame)
                                batch_rgb_features.append(features)
                            
                            rgb_features.extend(batch_rgb_features)
                            
                            # Log progress
                            logger.info(f"Extracted features from {len(rgb_features)}/{len(rgb_frame_paths)} RGB frames")
                
                # Extract features from flow directory
                flow_features = []
                if flow_dir:
                    logger.info(f"Processing optical flow from {os.path.basename(flow_dir)}")
                    
                    # Get flow frame paths
                    flow_frame_pattern = os.path.join(flow_dir, "*.jpg")
                    flow_frame_paths = sorted(glob.glob(flow_frame_pattern))
                    
                    if not flow_frame_paths:
                        logger.warning(f"No flow frames found in {flow_dir}")
                    else:
                        logger.info(f"Found {len(flow_frame_paths)} flow frames")
                        
                        # Process in batches
                        batch_size = BATCH_PROCESSING["batch_size"]
                        
                        for i in range(0, len(flow_frame_paths), batch_size):
                            batch_paths = flow_frame_paths[i:i+batch_size]
                            batch_frames = [cv2.imread(path) for path in batch_paths]
                            
                            # Skip invalid frames
                            batch_frames = [frame for frame in batch_frames if frame is not None]
                            if not batch_frames:
                                continue
                            
                            # Extract flow features using the PyTorch extractor
                            try:
                                batch_flow_features = torch_flow_extractor.extract_features_batch(batch_frames)
                            except Exception as e:
                                logger.warning(f"PyTorch flow extractor failed: {e}. Using simple extractor instead.")
                                batch_flow_features = []
                                for frame in batch_frames:
                                    features = simple_flow_extractor.extract_features(frame)
                                    batch_flow_features.append(features)
                                batch_flow_features = np.array(batch_flow_features)
                            
                            flow_features.extend(batch_flow_features)
                            
                            # Log progress
                            logger.info(f"Extracted features from {len(flow_features)}/{len(flow_frame_paths)} flow frames")
                
                # Convert to numpy arrays
                rgb_features = np.array(rgb_features) if rgb_features else np.array([])
                flow_features = np.array(flow_features) if flow_features else np.array([])
                
                if len(rgb_features) == 0 and len(flow_features) == 0:
                    logger.warning(f"No features extracted for {video_name}")
                    continue
                
                # Create combined features if we have both RGB and flow
                combined_features = np.array([])
                if len(rgb_features) > 0 and len(flow_features) > 0:
                    # Make sure dimensions match
                    min_frames = min(len(rgb_features), len(flow_features))
                    rgb_features_aligned = rgb_features[:min_frames]
                    flow_features_aligned = flow_features[:min_frames]
                    
                    # Combine features
                    combined_features = np.concatenate([rgb_features_aligned, flow_features_aligned], axis=1)
                    logger.info(f"Created combined features with shape {combined_features.shape}")
                
                # Prepare output paths
                rgb_output_path = os.path.join(OUTPUT_CONFIG["rgb_features_dir"], category, f"{video_name}.{OUTPUT_CONFIG['feature_format']}")
                flow_output_path = os.path.join(OUTPUT_CONFIG["flow_features_dir"], category, f"{video_name}.{OUTPUT_CONFIG['feature_format']}")
                combined_output_path = os.path.join(OUTPUT_CONFIG["combined_features_dir"], category, f"{video_name}.{OUTPUT_CONFIG['feature_format']}")
                
                # Create metadata
                output_metadata = {
                    "video_name": video_name,
                    "category": category,
                    "rgb_frames": len(rgb_features),
                    "flow_frames": len(flow_features),
                    "combined_frames": len(combined_features),
                    "extraction_time": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
                # Save features
                if len(rgb_features) > 0:
                    save_features(rgb_features, rgb_output_path, output_metadata)
                    logger.info(f"Saved RGB features to {rgb_output_path}")
                
                if len(flow_features) > 0:
                    save_features(flow_features, flow_output_path, output_metadata)
                    logger.info(f"Saved flow features to {flow_output_path}")
                
                if len(combined_features) > 0:
                    save_features(combined_features, combined_output_path, output_metadata)
                    logger.info(f"Saved combined features to {combined_output_path}")
                
                # Update counters
                total_frames += max(len(rgb_features), len(flow_features))
                total_videos += 1
                
                # Create visualizations for a sample frame
                if len(rgb_features) > 0 and len(rgb_frame_paths) > 0:
                    vis_dir = os.path.join(OUTPUT_CONFIG["visualizations_dir"], category)
                    os.makedirs(vis_dir, exist_ok=True)
                    
                    # Sample frames (first, middle, and last)
                    sample_indices = [0, len(rgb_frame_paths)//2, -1]
                    for idx in sample_indices:
                        if idx < 0:
                            idx = len(rgb_frame_paths) + idx
                        
                        if idx < len(rgb_frame_paths):
                            frame_path = rgb_frame_paths[idx]
                            frame = cv2.imread(frame_path)
                            
                            if frame is not None and idx < len(rgb_features):
                                # Get object detection features (first 25 values)
                                object_features = rgb_features[idx][:yolo_extractor.obj_feature_dim * yolo_extractor.max_objects]
                                
                                # Create visualization
                                vis_path = os.path.join(vis_dir, f"{video_name}_frame{idx}_vis.jpg")
                                visualize_detection(frame, object_features, save_path=vis_path)
                
            except Exception as e:
                logger.error(f"Error processing video {video_name}: {e}")
                logger.error(traceback.format_exc())
    
    # Log summary
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logger.info(f"Feature extraction completed!")
    logger.info(f"Processed {total_videos} videos with {total_frames} frames")
    logger.info(f"Total processing time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    logger.info(f"Features saved to: {OUTPUT_CONFIG['features_dir']}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 