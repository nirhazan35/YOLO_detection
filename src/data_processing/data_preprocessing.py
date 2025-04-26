import os
import cv2
import numpy as np
import subprocess
import shutil
import gc
import torch
from tqdm import tqdm
import logging

# Import config and validation module from data_processing package
from data_processing.config import PARALLEL_PROCESSING, GPU_CONFIG, VIDEO_CONFIG, KEEP_ORIGINALS
from data_processing.data_validation import validate_frame, validate_frames_sequence, repair_frame_sequence, detect_frozen_frames

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check if CUDA is available
def check_gpu_availability():
    """Check if CUDA is available and which device to use."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("CUDA not available, using CPU")
    return device

def create_directories(base_path):
    """Create necessary directories for processed data."""
    os.makedirs(base_path, exist_ok=True)
    
    # Create accident and non-accident directories
    os.makedirs(os.path.join(base_path, 'accidents'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'non_accidents'), exist_ok=True)
    
    logger.info(f"Created directory structure in {base_path}")

def calculate_frame_motion(frame1, frame2):
    """
    Calculate motion between two consecutive frames.
    
    Args:
        frame1: First frame
        frame2: Second frame
        
    Returns:
        Motion score (higher means more motion)
    """
    if frame1 is None or frame2 is None:
        return 0
    
    # Convert to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Calculate absolute difference
    diff = cv2.absdiff(gray1, gray2)
    
    # Normalize by image size
    motion_score = np.mean(diff) / 255.0
    
    return motion_score

def content_aware_frame_selection(frames, num_frames, motion_threshold=None):
    """
    Select frames based on their content (amount of motion).
    Prioritizes frames with significant motion changes.
    
    Args:
        frames: List of all video frames
        num_frames: Target number of frames to select
        motion_threshold: Threshold for considering significant motion
        
    Returns:
        List of selected frames
    """
    if len(frames) <= num_frames:
        return frames
    
    if motion_threshold is None:
        motion_threshold = VIDEO_CONFIG['motion_threshold']
    
    # Calculate motion scores between consecutive frames
    motion_scores = []
    for i in range(1, len(frames)):
        score = calculate_frame_motion(frames[i-1], frames[i])
        motion_scores.append((i, score))
    
    # Sort frames by motion score
    sorted_frames = sorted(motion_scores, key=lambda x: x[1], reverse=True)
    
    # Get indices of frames with highest motion
    high_motion_indices = [idx for idx, score in sorted_frames if score > motion_threshold]
    
    # If we have enough high motion frames, use them
    if len(high_motion_indices) >= num_frames:
        # Sort indices to maintain temporal order
        selected_indices = sorted(high_motion_indices[:num_frames])
        logger.info(f"Selected {len(selected_indices)} frames based on high motion content")
    else:
        # Otherwise, use high motion frames + evenly spaced frames for the rest
        remaining = num_frames - len(high_motion_indices)
        even_indices = np.linspace(0, len(frames) - 1, remaining + 2, dtype=int)[1:-1]
        
        # Combine and remove duplicates
        selected_indices = sorted(list(set(high_motion_indices + even_indices.tolist())))
        
        # If still too many, take evenly spaced ones from the selection
        if len(selected_indices) > num_frames:
            selected_indices = sorted(np.linspace(0, len(selected_indices) - 1, num_frames, dtype=int))
        
        logger.info(f"Selected {len(high_motion_indices)} high motion frames + {len(selected_indices) - len(high_motion_indices)} evenly spaced frames")
    
    # Return selected frames
    return [frames[i] for i in selected_indices[:num_frames]]

def temporal_subsample_video(cap, max_length=None):
    """
    Temporally subsample a long video to a maximum length.
    
    Args:
        cap: OpenCV VideoCapture object
        max_length: Maximum video length in seconds
        
    Returns:
        Subsampled frame indices
    """
    if max_length is None:
        max_length = VIDEO_CONFIG['max_video_length']
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    if duration <= max_length:
        # Video is already shorter than max_length, no subsampling needed
        return list(range(total_frames))
    
    # Calculate subsampling ratio
    ratio = max_length / duration
    
    # Generate indices
    indices = []
    for i in range(total_frames):
        if i % int(1/ratio) == 0:
            indices.append(i)
    
    logger.info(f"Temporally subsampled video from {duration:.2f}s to {max_length}s ({len(indices)} frames)")
    return indices

def preprocess_video_ffmpeg(input_path, output_path, target_resolution=None, target_fps=None, duration=None):
    """
    Preprocess video using FFmpeg:
    - Resize to target resolution
    - Convert to target FPS
    - Trim to specified duration only if video is longer than 4 seconds
    """
    # Use config values if parameters are not specified
    if target_resolution is None:
        target_resolution = VIDEO_CONFIG['target_resolution']
    if target_fps is None:
        target_fps = VIDEO_CONFIG['target_fps']
    if duration is None:
        duration = VIDEO_CONFIG['clip_duration']
        
    width, height = target_resolution
    
    # First get video duration using ffprobe
    duration_cmd = [
        'ffprobe', 
        '-v', 'error', 
        '-show_entries', 'format=duration', 
        '-of', 'default=noprint_wrappers=1:nokey=1', 
        input_path
    ]
    
    try:
        video_duration = float(subprocess.check_output(duration_cmd).decode('utf-8').strip())
        logger.info(f"Video duration: {video_duration:.2f} seconds")
        
        # Only use duration parameter if video is longer than 4 seconds
        trim_option = []
        if video_duration > 4.0:
            trim_option = ['-t', str(duration)]
            logger.info(f"Trimming video to {duration} seconds")
        else:
            logger.info(f"Video is under 4 seconds ({video_duration:.2f}s), keeping original length")
    except Exception as e:
        logger.warning(f"Could not determine video duration: {e}. Will apply default trimming.")
        trim_option = ['-t', str(duration)]
    
    # Build FFmpeg command
    cmd = [
        'ffmpeg', '-y', '-i', input_path,
        '-vf', f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
        '-r', str(target_fps),
        '-c:v', 'libx264', '-crf', '18',
        '-an',  # Remove audio
    ]
    
    # Add trim option only if needed
    if trim_option:
        cmd.extend(trim_option)
    
    # Add output path
    cmd.append(output_path)
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Verify the output file exists and has a non-zero size
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return True
        else:
            logger.error(f"FFmpeg processed the file but output is empty or missing: {output_path}")
            return False
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg error: {e.stderr.decode() if hasattr(e, 'stderr') else str(e)}")
        return False

def extract_frames(video_path, num_frames=None, batch_size=None):
    """
    Extract evenly spaced frames from a video with batched processing for memory efficiency.
    
    Args:
        video_path: Path to the video file
        num_frames: Number of frames to extract (None to use config value)
        batch_size: Number of frames to process at once for memory efficiency (None to use config value)
        
    Returns:
        List of extracted frames as numpy arrays or None if failed
    """
    # Use config values if parameters are not specified
    if num_frames is None:
        num_frames = VIDEO_CONFIG['num_frames']
    if batch_size is None:
        batch_size = PARALLEL_PROCESSING['batch_size']
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return None
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        logger.error(f"Video has no frames: {video_path}")
        cap.release()
        return None
    
    # Apply temporal subsampling for long videos if enabled
    if VIDEO_CONFIG['temporal_subsampling'] and total_frames > num_frames * 2:
        frame_indices = temporal_subsample_video(cap)
    else:
        # Calculate frame indices to extract evenly
        frame_indices = np.linspace(0, total_frames - 1, min(total_frames, num_frames * 2), dtype=int)
    
    # Extract frames in batches
    all_frames = []
    for batch_start in range(0, len(frame_indices), batch_size):
        batch_indices = frame_indices[batch_start:batch_start + batch_size]
        batch_frames = []
        
        for idx in batch_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Failed to read frame {idx} from {video_path}")
                # If frame extraction fails, use black frame
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Validate the frame
            is_valid, error = validate_frame(frame)
            if not is_valid:
                logger.warning(f"Invalid frame at index {idx}: {error}")
                # Use black frame for invalid frames
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                
            batch_frames.append(frame)
        
        all_frames.extend(batch_frames)
        
        # Force garbage collection to free memory
        batch_frames = None
        gc.collect()
    
    cap.release()
    
    # Apply content-aware frame selection if enabled and we have more frames than needed
    if VIDEO_CONFIG['content_aware_sampling'] and len(all_frames) > num_frames:
        frames = content_aware_frame_selection(all_frames, num_frames)
        logger.info(f"Applied content-aware frame selection, selected {len(frames)} frames from {len(all_frames)}")
    else:
        # Otherwise just use evenly spaced frames
        if len(all_frames) > num_frames:
            indices = np.linspace(0, len(all_frames) - 1, num_frames, dtype=int)
            frames = [all_frames[i] for i in indices]
        else:
            frames = all_frames
    
    # Validate the entire frame sequence
    is_valid, valid_count, error = validate_frames_sequence(frames)
    if not is_valid:
        logger.warning(f"Frame sequence validation failed: {error}")
        # Try to repair the frame sequence
        frames = repair_frame_sequence(frames, num_frames)
    
    # Check for frozen frames
    frozen_indices, frozen_count = detect_frozen_frames(frames)
    if frozen_count > 0:
        logger.warning(f"Detected {frozen_count} frozen frames after extraction")
    
    return frames

def extract_frames_opencv(video_path, output_dir, video_name, num_frames=None):
    """
    Extract frames from video and save to disk.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        video_name: Base name for the extracted frames
        num_frames: Number of frames to extract (None to use config value)
    
    Returns:
        Boolean indicating success or failure
    """
    # Use config value if parameter is not specified
    if num_frames is None:
        num_frames = VIDEO_CONFIG['num_frames']
    
    # Extract frames from the video
    frames = extract_frames(video_path, num_frames)
    
    if frames is None or len(frames) == 0:
        logger.error(f"No frames extracted from: {video_path}")
        return False
    
    # Create output directory
    frames_dir = os.path.join(output_dir, f"{video_name}_frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    # Save frames to disk
    for i, frame in enumerate(frames):
        output_path = os.path.join(frames_dir, f"frame_{i:04d}.jpg")
        cv2.imwrite(output_path, frame)
    
    return True

def preprocess_video_all(input_path, output_dir, video_name, config=None):
    """
    Full preprocessing pipeline for a video:
    - Standardize resolution, FPS, and duration
    - Extract frames
    
    Args:
        input_path: Path to input video
        output_dir: Output directory
        video_name: Base name for processed files
        config: Processing configuration
        
    Returns:
        Dictionary with processing results or None if failed
    """
    # Use default config if not specified
    if config is None:
        config = VIDEO_CONFIG
    
    try:
        # Create temp directory if needed
        temp_dir = os.path.join(output_dir, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Preprocessed video path
        preprocessed_video = os.path.join(temp_dir, f"{video_name}_preprocessed.mp4")
        
        # Step 1: Preprocess video with FFmpeg
        success = preprocess_video_ffmpeg(
            input_path, 
            preprocessed_video,
            target_resolution=config.get('target_resolution'),
            target_fps=config.get('target_fps'),
            duration=config.get('clip_duration')
        )
        
        if not success:
            logger.error(f"Failed to preprocess video: {input_path}")
            return None
        
        # Create frames directory
        frames_dir = os.path.join(temp_dir, f"{video_name}_frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        # Step 2: Extract frames using OpenCV (this will also apply 
        # content-aware selection if enabled)
        frames = extract_frames(
            preprocessed_video, 
            num_frames=config.get('num_frames')
        )
        
        if frames is None or len(frames) == 0:
            logger.error(f"Failed to extract frames from: {preprocessed_video}")
            return None
        
        # Save frames to disk
        for i, frame in enumerate(frames):
            output_path = os.path.join(frames_dir, f"frame_{i:04d}.jpg")
            cv2.imwrite(output_path, frame)
        
        # Return processing results
        return {
            'preprocessed_video': preprocessed_video,
            'frames_dir': frames_dir,
            'num_frames': len(frames)
        }
        
    except Exception as e:
        logger.error(f"Error in preprocessing pipeline: {str(e)}")
        return None

def load_video_frames(frames_dir, num_frames=None):
    """
    Load video frames from disk.
    
    Args:
        frames_dir: Directory containing frame images
        num_frames: Number of frames to load (None to load all)
        
    Returns:
        List of frames as numpy arrays
    """
    # Check if directory exists
    if not os.path.exists(frames_dir):
        logger.error(f"Frames directory does not exist: {frames_dir}")
        return None
    
    # Get frame files
    frame_files = [f for f in os.listdir(frames_dir) if f.endswith('.jpg')]
    
    # Sort by frame number
    frame_files = sorted(frame_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    # Limit number of frames if specified
    if num_frames is not None and len(frame_files) > num_frames:
        # Use evenly spaced frames
        indices = np.linspace(0, len(frame_files) - 1, num_frames, dtype=int)
        frame_files = [frame_files[i] for i in indices]
    
    # Load frames
    frames = []
    for file in frame_files:
        file_path = os.path.join(frames_dir, file)
        frame = cv2.imread(file_path)
        
        # Validate frame
        is_valid, error = validate_frame(frame)
        if not is_valid:
            logger.warning(f"Invalid frame: {file_path}, {error}")
            # Use black frame for invalid frames
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
        frames.append(frame)
    
    return frames

def batch_process_frames(frame_processor, frames, batch_size=None):
    """
    Process frames in batches to manage memory usage.
    
    Args:
        frame_processor: Function to process each frame
        frames: List of input frames
        batch_size: Number of frames to process at once (None to use config value)
        
    Returns:
        List of processed frames
    """
    # Use batch size from config if not specified
    if batch_size is None:
        batch_size = PARALLEL_PROCESSING['batch_size']
        
    processed_frames = []
    
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i+batch_size]
        
        # Process batch
        processed_batch = [frame_processor(frame) for frame in batch]
        processed_frames.extend(processed_batch)
        
        # Force garbage collection to free memory
        processed_batch = None
        gc.collect()
    
    return processed_frames