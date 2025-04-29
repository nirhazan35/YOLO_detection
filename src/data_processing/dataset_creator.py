import os
import numpy as np
import random
import shutil
import logging
import time
import json
from pathlib import Path
import cv2
import torch
import gc
import argparse

# Import project modules
from data_processing.config import (
    DATA_PATH, VIDEO_CONFIG, RANDOM_SEED, KEEP_ORIGINALS
)
from data_processing.data_preprocessing import (
    preprocess_video_all, load_video_frames,
    check_gpu_availability, normalize_path
)
from data_processing.optical_flow import compute_optical_flow, save_optical_flow_frames
from data_processing.data_validation import validate_processed_data, validate_rgb_flow_pair

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dataset_creation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def set_random_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")

def get_video_paths(accident_dir, non_accident_dir):
    """
    Get paths to accident and non-accident videos.
    
    Args:
        accident_dir: Directory containing accident videos
        non_accident_dir: Directory containing non-accident videos
        
    Returns:
        Dictionary with video paths and labels
    """
    # Video extensions to look for
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    
    # Get accident videos
    accident_videos = []
    for root, _, files in os.walk(accident_dir):
        for file in files:
            if file.lower().endswith(video_extensions):
                video_path = os.path.join(root, file)
                # Normalize path for current OS
                video_path = normalize_path(video_path)
                accident_videos.append(video_path)
    
    # Get non-accident videos
    non_accident_videos = []
    for root, _, files in os.walk(non_accident_dir):
        for file in files:
            if file.lower().endswith(video_extensions):
                video_path = os.path.join(root, file)
                # Normalize path for current OS
                video_path = normalize_path(video_path)
                non_accident_videos.append(video_path)
    
    logger.info(f"Found {len(accident_videos)} accident videos and {len(non_accident_videos)} non-accident videos")
    
    return {
        'accident': accident_videos,
        'non_accident': non_accident_videos
    }

def process_video(video_path, label, video_idx, config, use_gpu=None):
    """
    Process a single video through the entire pipeline.
    
    Args:
        video_path: Path to the input video
        label: Video label (accident, non_accident)
        video_idx: Video index for unique naming
        config: Processing configuration
        use_gpu: Whether to use GPU for processing (None for auto-detection)
        
    Returns:
        Boolean indicating success or failure
    """
    try:
        # Normalize video path
        video_path = normalize_path(video_path)
        
        # Determine output base directory based on label
        if label == 'accident':
            output_base_dir = DATA_PATH['processed_accidents']
        else:
            output_base_dir = DATA_PATH['processed_non_accidents']
        
        # Create a unique video name
        video_name = f"{label}_{video_idx:04d}"
        
        # Define output directories - directly in the accident/non-accident folder
        output_dir = output_base_dir
        frames_output_dir = os.path.join(output_dir, f"{video_name}_frames")
        flow_output_dir = os.path.join(output_dir, f"{video_name}_flow")
        metadata_path = os.path.join(output_dir, f"{video_name}_metadata.json")
        
        # Skip if already processed and metadata exists (for resuming)
        if os.path.exists(metadata_path):
            # Validate the processed data
            is_valid, error = validate_processed_data(output_dir, metadata_path)
            if is_valid:
                logger.info(f"Skipping already processed video: {video_name}")
                return True
            else:
                logger.warning(f"Previously processed data invalid: {error}. Reprocessing...")
        
        # Create directories
        os.makedirs(frames_output_dir, exist_ok=True)
        os.makedirs(flow_output_dir, exist_ok=True)
        
        # Step 1: Preprocess video and extract frames
        processed_data = preprocess_video_all(
            video_path, 
            output_dir,  # Use the output directory directly
            video_name, 
            config
        )
        
        if processed_data is None:
            logger.error(f"Failed to process video: {video_path}")
            return False
        
        # Load frames
        frames = load_video_frames(processed_data['frames_dir'], num_frames=config['num_frames'])
        
        if not frames:
            logger.error(f"No frames extracted from: {video_path}")
            return False
        
        # Step 2: Compute optical flow (auto-detect GPU if not specified)
        flow_frames = compute_optical_flow(
            frames, 
            use_gpu=use_gpu
        )
        
        # Validate consistency between RGB frames and optical flow
        is_consistent, error_msg = validate_rgb_flow_pair(frames, flow_frames)
        if not is_consistent:
            logger.warning(f"RGB-flow consistency check failed: {error_msg}. Will try to continue anyway.")
        
        # Step 3: Save frames and flow
        for i, frame in enumerate(frames):
            cv2.imwrite(os.path.join(frames_output_dir, f"frame_{i:02d}.jpg"), frame)
        
        save_optical_flow_frames(
            flow_frames, 
            flow_output_dir, 
            video_name
        )
        
        # Save metadata including processing parameters
        metadata = {
            'original_path': video_path,
            'label': label,
            'frames_count': len(frames),
            'flow_frames_count': len(flow_frames),
            'processed_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'processing_config': {
                'resolution': config['target_resolution'],
                'fps': config['target_fps'],
                'content_aware': config['content_aware_sampling'],
                'flow_method': 'raft' if 'raft' in str(flow_frames).lower() else 'standard'
            },
            'split': None  # Will be assigned later in the pipeline
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Clean up temporary files
        if 'preprocessed_video' in processed_data and os.path.exists(processed_data['preprocessed_video']):
            os.remove(processed_data['preprocessed_video'])
            logger.info(f"Removed temporary preprocessed video: {processed_data['preprocessed_video']}")
            
        # Clean up the temporary frames directory
        if 'frames_dir' in processed_data and os.path.exists(processed_data['frames_dir']):
            shutil.rmtree(processed_data['frames_dir'])
            logger.info(f"Removed temporary frames directory: {processed_data['frames_dir']}")
        
        # Force garbage collection to free memory
        gc.collect()
        
        return True
    
    except Exception as e:
        logger.error(f"Error processing video {video_path}: {str(e)}")
        return False

def save_progress(output_dir, processed_videos, failed_videos):
    """
    Save processing progress.
    
    Args:
        output_dir: Base output directory
        processed_videos: List of successfully processed videos
        failed_videos: List of failed videos
    """
    progress = {
        'processed_videos': processed_videos,
        'failed_videos': failed_videos,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    progress_file = os.path.join(output_dir, 'progress.json')
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=4)
    
    logger.info(f"Saved progress to {progress_file}")

def load_progress(output_dir, reset_failed=False):
    """
    Load processing progress.
    
    Args:
        output_dir: Base output directory
        reset_failed: Whether to reset the failed videos list and retry processing them
        
    Returns:
        Tuple of (processed_videos, failed_videos)
    """
    progress_file = os.path.join(output_dir, 'progress.json')
    
    if not os.path.exists(progress_file):
        logger.info("No progress file found, starting from scratch")
        return [], []
    
    try:
        with open(progress_file, 'r') as f:
            progress = json.load(f)
        
        processed_videos = progress.get('processed_videos', [])
        failed_videos = progress.get('failed_videos', []) if not reset_failed else []
        
        if reset_failed:
            logger.info(f"Reset failed videos list. Will attempt to process {len(progress.get('failed_videos', []))} previously failed videos.")
            # Also save the updated progress file with empty failed videos
            save_progress(output_dir, processed_videos, failed_videos)
        else:
            logger.info(f"Loaded progress: {len(processed_videos)} processed, {len(failed_videos)} failed")
        
        return processed_videos, failed_videos
    except Exception as e:
        logger.error(f"Error loading progress: {e}")
        return [], []

def create_dataset(reset_failed=False):
    """
    Main function to create the dataset.
    
    Args:
        reset_failed: Whether to retry processing previously failed videos
    """
    # Set random seed for reproducibility
    set_random_seed(RANDOM_SEED)
    
    # Create output directories
    os.makedirs(DATA_PATH['processed_parent'], exist_ok=True)
    os.makedirs(DATA_PATH['processed_accidents'], exist_ok=True)
    os.makedirs(DATA_PATH['processed_non_accidents'], exist_ok=True)
    
    # Get video paths
    video_paths = get_video_paths(DATA_PATH['accidents'], DATA_PATH['non_accidents'])
    
    # Load progress if exists
    processed_videos, failed_videos = load_progress(DATA_PATH['processed_parent'], reset_failed)
    
    # Check if GPU is available
    device = check_gpu_availability()
    use_gpu = device.type == 'cuda'
    
    # Process videos
    for label, paths in video_paths.items():
        for i, video_path in enumerate(paths):
            # Skip if already processed
            if video_path in processed_videos:
                logger.info(f"Skipping already processed video: {video_path}")
                continue
            
            # Skip if already failed and reset_failed is False
            if video_path in failed_videos:
                logger.info(f"Skipping previously failed video: {video_path}")
                continue
            
            # Process the video
            logger.info(f"Processing {label} video {i+1}/{len(paths)}: {video_path}")
            success = process_video(
                video_path, 
                label, 
                i, 
                VIDEO_CONFIG,
                use_gpu=use_gpu
            )
            
            if success:
                processed_videos.append(video_path)
                logger.info(f"Successfully processed video: {video_path}")
            else:
                failed_videos.append(video_path)
                logger.error(f"Failed to process video: {video_path}")
            
            # Save progress periodically
            if (i + 1) % 10 == 0:
                save_progress(DATA_PATH['processed_parent'], processed_videos, failed_videos)
    
    # Save final progress
    save_progress(DATA_PATH['processed_parent'], processed_videos, failed_videos)
    
    # Print summary
    logger.info(f"Dataset creation complete. Processed {len(processed_videos)} videos, failed {len(failed_videos)} videos.")
    
    return processed_videos, failed_videos

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset creation for road accident detection")
    parser.add_argument("--reset-failed", action="store_true", help="Reset the list of failed videos and attempt to process them again")
    args = parser.parse_args()
    
    create_dataset(reset_failed=args.reset_failed)