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

# Import project modules
from data_processing.config import (
    DATA_PATH, VIDEO_CONFIG, RANDOM_SEED, KEEP_ORIGINALS
)
from data_processing.data_preprocessing import (
    preprocess_video_all, load_video_frames,
    check_gpu_availability
)
from data_processing.optical_flow import compute_optical_flow, save_optical_flow_frames
from data_processing.data_validation import validate_processed_data

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
                accident_videos.append(os.path.join(root, file))
    
    # Get non-accident videos
    non_accident_videos = []
    for root, _, files in os.walk(non_accident_dir):
        for file in files:
            if file.lower().endswith(video_extensions):
                non_accident_videos.append(os.path.join(root, file))
    
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
        
        # Step 3: Save frames and flow
        for i, frame in enumerate(frames):
            cv2.imwrite(os.path.join(frames_output_dir, f"frame_{i:02d}.jpg"), frame)
        
        save_optical_flow_frames(
            flow_frames, 
            flow_output_dir, 
            video_name
        )
        
        # Save metadata
        metadata = {
            'original_path': video_path,
            'label': label,
            'frames_count': len(frames),
            'flow_frames_count': len(flow_frames),
            'processed_date': time.strftime('%Y-%m-%d %H:%M:%S')
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
    
    progress_path = os.path.join(output_dir, 'progress.json')
    with open(progress_path, 'w') as f:
        json.dump(progress, f, indent=2)
    
    logger.info(f"Saved progress: {len(processed_videos)} processed, {len(failed_videos)} failed")

def load_progress(output_dir):
    """
    Load processing progress.
    
    Args:
        output_dir: Base output directory
        
    Returns:
        Progress dictionary or None if not found
    """
    progress_path = os.path.join(output_dir, 'progress.json')
    if os.path.exists(progress_path):
        try:
            with open(progress_path, 'r') as f:
                progress = json.load(f)
            logger.info(f"Loaded progress from {progress['timestamp']}")
            return progress
        except Exception as e:
            logger.error(f"Error loading progress: {e}")
    
    return None

def create_dataset():
    """
    Process all videos sequentially to create the dataset.
    Using configuration parameters from config.py.
    """
    logger.info("Starting dataset creation...")
    
    # Set random seed
    set_random_seed(RANDOM_SEED)
    
    # Check GPU availability
    device = check_gpu_availability()
    use_gpu = (device.type == 'cuda')
    
    # Configure GPU device if specified
    if use_gpu and torch.cuda.is_available():
        try:
            torch.cuda.set_device(0)
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        except Exception as e:
            logger.warning(f"Failed to set GPU device: {e}")
            use_gpu = False
    
    # Create output directories
    os.makedirs(DATA_PATH['processed_parent'], exist_ok=True)
    os.makedirs(DATA_PATH['processed_accidents'], exist_ok=True)
    os.makedirs(DATA_PATH['processed_non_accidents'], exist_ok=True)
    
    # Get video paths
    video_paths = get_video_paths(DATA_PATH['accidents'], DATA_PATH['non_accidents'])
    
    # Combine all videos
    all_videos = {
        'accident': video_paths['accident'],
        'non_accident': video_paths['non_accident']
    }
    
    logger.info(f"Total videos to process: {len(all_videos['accident'])} accident, {len(all_videos['non_accident'])} non-accident")
    
    # Save dataset information
    with open(os.path.join(DATA_PATH['processed_parent'], 'dataset_info.json'), 'w') as f:
        json.dump({
            'accident': len(all_videos['accident']),
            'non_accident': len(all_videos['non_accident']),
            'total': len(all_videos['accident']) + len(all_videos['non_accident'])
        }, f, indent=2)
    
    # Create processing queue
    processing_queue = []
    for label in all_videos:
        for i, video_path in enumerate(all_videos[label]):
            video_idx = i + 1  # 1-based index
            processing_queue.append((video_path, label, video_idx, VIDEO_CONFIG, use_gpu))
    
    # Load progress if exists
    progress = load_progress(DATA_PATH['processed_parent'])
    processed_videos = []
    failed_videos = []
    
    if progress:
        processed_videos = progress['processed_videos']
        failed_videos = progress['failed_videos']
        logger.info(f"Resuming from previous progress: {len(processed_videos)} already processed")
    
    # Process videos sequentially
    for idx, (video_path, label, video_idx, config, use_gpu) in enumerate(processing_queue):
        # Skip already processed videos
        if video_path in processed_videos:
            logger.info(f"Skipping already processed video: {video_path}")
            continue
        
        logger.info(f"Processing video {idx+1}/{len(processing_queue)}: {video_path}")
        
        success = process_video(video_path, label, video_idx, config, use_gpu)
        if success:
            processed_videos.append(video_path)
            logger.info(f"Successfully processed: {video_path}")
        else:
            failed_videos.append(video_path)
            logger.error(f"Failed to process: {video_path}")
        
        # Save progress every 5 videos
        if (idx + 1) % 5 == 0:
            save_progress(DATA_PATH['processed_parent'], processed_videos, failed_videos)
    
    # Final report
    logger.info(f"Dataset creation completed. Processed {len(processed_videos)} videos successfully.")
    if failed_videos:
        logger.warning(f"Failed to process {len(failed_videos)} videos.")
        with open(os.path.join(DATA_PATH['processed_parent'], 'failed_videos.json'), 'w') as f:
            json.dump(failed_videos, f, indent=2)
    
    # Create final success report
    success_report = {
        'total_videos': len(processing_queue),
        'processed_videos': len(processed_videos),
        'failed_videos': len(failed_videos),
        'success_rate': len(processed_videos) / len(processing_queue) if processing_queue else 0,
        'processing_time': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(os.path.join(DATA_PATH['processed_parent'], 'processing_report.json'), 'w') as f:
        json.dump(success_report, f, indent=2)
    
    logger.info(f"Dataset creation report saved to {os.path.join(DATA_PATH['processed_parent'], 'processing_report.json')}")

if __name__ == '__main__':
    create_dataset()