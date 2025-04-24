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
from data_processing.data_validation import validate_frame, validate_frames_sequence, repair_frame_sequence

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
    
    # Calculate frame indices to extract
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    # Extract frames in batches
    frames = []
    for batch_start in range(0, len(indices), batch_size):
        batch_indices = indices[batch_start:batch_start + batch_size]
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
        
        frames.extend(batch_frames)
        
        # Force garbage collection to free memory
        batch_frames = None
        gc.collect()
    
    cap.release()
    
    # Validate the entire frame sequence
    is_valid, valid_count, error = validate_frames_sequence(frames)
    if not is_valid:
        logger.warning(f"Frame sequence validation failed: {error}")
        # Try to repair the frame sequence
        frames = repair_frame_sequence(frames, num_frames)
    
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
        
    frames = extract_frames(video_path, num_frames)
    if frames is None:
        return False
    
    # Save frames
    os.makedirs(output_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        output_path = os.path.join(output_dir, f"{video_name}_frame_{i:02d}.jpg")
        cv2.imwrite(output_path, frame)
        
        # Verify the saved frame
        saved_frame = cv2.imread(output_path)
        is_valid, error = validate_frame(saved_frame)
        if not is_valid:
            logger.error(f"Saved frame validation failed: {output_path}, {error}")
            return False
    
    return True

def preprocess_video_all(input_path, output_dir, video_name, config=None):
    """
    Complete video preprocessing pipeline.
    
    Args:
        input_path: Path to the input video
        output_dir: Directory to save processed data
        video_name: Base name for processed files
        config: Dictionary with processing parameters (None to use global config)
    
    Returns:
        Dictionary with paths to processed data or None if failed
    """
    # Use global config if not specified
    if config is None:
        config = VIDEO_CONFIG
        
    # Create paths directly in the output directory
    preprocessed_video = os.path.join(output_dir, f"{video_name}_preprocessed.mp4")
    frames_dir = os.path.join(output_dir, f"{video_name}_temp_frames")
    
    # Make sure the frames directory exists
    os.makedirs(frames_dir, exist_ok=True)
    
    # Step 1: Preprocess video with FFmpeg
    success = preprocess_video_ffmpeg(
        input_path, 
        preprocessed_video,
        target_resolution=config['target_resolution'],
        target_fps=config['target_fps'],
        duration=config['clip_duration']
    )
    
    if not success:
        logger.error(f"Failed to preprocess video: {input_path}")
        return None
    
    # Step 2: Extract frames
    success = extract_frames_opencv(
        preprocessed_video,
        frames_dir,
        video_name,
        num_frames=config['num_frames']
    )
    
    if not success:
        logger.error(f"Failed to extract frames from: {preprocessed_video}")
        return None
    
    # Return paths to processed data
    return {
        'preprocessed_video': preprocessed_video,
        'frames_dir': frames_dir
    }

def load_video_frames(frames_dir, num_frames=None):
    """
    Load frames from directory.
    
    Args:
        frames_dir: Directory containing frame images
        num_frames: Expected number of frames (None to use config value)
        
    Returns:
        List of frames as numpy arrays
    """
    # Use config value if parameter is not specified
    if num_frames is None:
        num_frames = VIDEO_CONFIG['num_frames']
        
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    frames = []
    
    for f in frame_files[:num_frames]:
        frame_path = os.path.join(frames_dir, f)
        frame = cv2.imread(frame_path)
        
        # Validate frame
        is_valid, error = validate_frame(frame)
        if not is_valid:
            logger.warning(f"Invalid frame loaded from {frame_path}: {error}")
            # If this is the first frame and it's invalid, create a black frame
            if not frames:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # Otherwise use the last valid frame
            else:
                frame = frames[-1].copy()
        
        frames.append(frame)
    
    # If we don't have enough frames, pad with last frame
    if frames and len(frames) < num_frames:
        while len(frames) < num_frames:
            frames.append(frames[-1].copy())
    
    # Validate the sequence
    is_valid, valid_count, error = validate_frames_sequence(frames)
    if not is_valid:
        logger.warning(f"Frame sequence validation failed during loading: {error}")
        # Try to repair the frame sequence
        frames = repair_frame_sequence(frames, num_frames)
    
    return frames

# Memory-efficient batch processing function
def batch_process_frames(frame_processor, frames, batch_size=None):
    """
    Process frames in batches to optimize memory usage.
    
    Args:
        frame_processor: Function that processes frames
        frames: List of frames to process
        batch_size: Number of frames to process at once (None to use config value)
        
    Returns:
        List of processed frames
    """
    # Use config value if parameter is not specified
    if batch_size is None:
        batch_size = PARALLEL_PROCESSING['batch_size']
        
    processed_frames = []
    
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i+batch_size]
        
        # Process batch
        processed_batch = [frame_processor(frame) for frame in batch]
        processed_frames.extend(processed_batch)
        
        # Clear batch from memory
        batch = None
        processed_batch = None
        gc.collect()
    
    return processed_frames