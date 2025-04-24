import os
import cv2
import numpy as np
import logging
from pathlib import Path
import json

# Import config
from data_processing.config import VALIDATION

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def validate_frame(frame):
    """
    Validate a single frame.
    
    Args:
        frame: Input frame as numpy array
        
    Returns:
        Boolean indicating if frame is valid and an error message if invalid
    """
    # Check if frame is None
    if frame is None:
        return False, "Frame is None"
    
    # Check if frame is empty
    if frame.size == 0:
        return False, "Frame is empty"
        
    # Check if frame has correct dimensions
    if len(frame.shape) != 3:
        return False, f"Frame has incorrect dimensions: {frame.shape}"
    
    # Check if frame has correct channels (should be BGR)
    if frame.shape[2] != 3:
        return False, f"Frame has incorrect number of channels: {frame.shape[2]}"
    
    # Check if frame contains only zeros or has very low variance (could be black/corrupt)
    # Use the variance threshold from the config file
    if np.var(frame) < VALIDATION['frame_variance_threshold']:
        return False, "Frame has very low variance, might be corrupted or black"
    
    return True, ""

def validate_frames_sequence(frames):
    """
    Validate a sequence of frames.
    
    Args:
        frames: List of frames
        
    Returns:
        Boolean indicating if sequence is valid, number of valid frames, and error message
    """
    if not frames:
        return False, 0, "Empty frame sequence"
    
    valid_count = 0
    error_frames = []
    
    for i, frame in enumerate(frames):
        is_valid, error = validate_frame(frame)
        if is_valid:
            valid_count += 1
        else:
            error_frames.append((i, error))
    
    # If less than the threshold ratio of frames are valid, consider the sequence invalid
    # Use the validity ratio from the config file
    validity_threshold = VALIDATION['sequence_validity_ratio']
    is_sequence_valid = valid_count / len(frames) >= validity_threshold
    
    error_msg = ""
    if not is_sequence_valid:
        error_msg = f"Only {valid_count}/{len(frames)} frames are valid. Errors: {error_frames}"
    
    return is_sequence_valid, valid_count, error_msg

def validate_optical_flow(flow_frames):
    """
    Validate optical flow frames.
    
    Args:
        flow_frames: List of optical flow visualization frames
        
    Returns:
        Boolean indicating if flow is valid and an error message if invalid
    """
    if not flow_frames:
        return False, "Empty optical flow sequence"
    
    # Check if all frames have the same shape
    shapes = [frame.shape for frame in flow_frames]
    if len(set(tuple(shape) for shape in shapes)) > 1:
        return False, "Inconsistent shapes in optical flow frames"
    
    # Check if flow frames have reasonable motion info (non-zero variance)
    motion_detected = False
    for frame in flow_frames:
        # Convert to HSV where V channel represents magnitude
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        magnitude = hsv[:, :, 2]
        
        # If any frame has significant motion, consider valid
        if np.mean(magnitude) > 5:
            motion_detected = True
            break
    
    if not motion_detected:
        return False, "No significant motion detected in optical flow frames"
    
    return True, ""

def validate_processed_data(data_dir, metadata_file):
    """
    Validate processed data by checking file integrity and metadata.
    
    Args:
        data_dir: Directory containing processed frames and flow
        metadata_file: Path to metadata JSON file
        
    Returns:
        Boolean indicating if data is valid and an error message if invalid
    """
    try:
        # Check if directory exists
        if not os.path.exists(data_dir):
            return False, f"Directory does not exist: {data_dir}"
        
        # Check metadata file
        if not os.path.exists(metadata_file):
            return False, f"Metadata file not found: {metadata_file}"
        
        # Load metadata
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Check if metadata contains required fields
        required_fields = ['frames_count', 'flow_frames_count', 'label', 'split']
        for field in required_fields:
            if field not in metadata:
                return False, f"Missing field in metadata: {field}"
        
        # Check if frames exist
        frames_dir = os.path.join(data_dir, f"{os.path.basename(metadata_file).split('_metadata.json')[0]}_frames")
        if not os.path.exists(frames_dir):
            return False, f"Frames directory not found: {frames_dir}"
        
        # Count frames
        frame_files = [f for f in os.listdir(frames_dir) if f.endswith('.jpg')]
        if len(frame_files) != metadata['frames_count']:
            return False, f"Mismatch in frame count: expected {metadata['frames_count']}, found {len(frame_files)}"
        
        # Check if optical flow exists
        flow_dir = os.path.join(data_dir, f"{os.path.basename(metadata_file).split('_metadata.json')[0]}_flow")
        if not os.path.exists(flow_dir):
            return False, f"Flow directory not found: {flow_dir}"
        
        # Count flow frames
        flow_files = [f for f in os.listdir(flow_dir) if f.endswith('.jpg')]
        if len(flow_files) != metadata['flow_frames_count']:
            return False, f"Mismatch in flow frame count: expected {metadata['flow_frames_count']}, found {len(flow_files)}"
        
        return True, ""
    
    except Exception as e:
        return False, f"Error validating processed data: {str(e)}"

def repair_frame_sequence(frames, target_num_frames):
    """
    Attempt to repair a frame sequence by replacing invalid frames.
    
    Args:
        frames: List of frames
        target_num_frames: Target number of frames
        
    Returns:
        Repaired list of frames
    """
    if not frames:
        return frames
    
    repaired_frames = []
    valid_frames = []
    
    # First identify all valid frames
    for frame in frames:
        is_valid, _ = validate_frame(frame)
        if is_valid:
            valid_frames.append(frame)
    
    # If no valid frames, return empty list
    if not valid_frames:
        logger.error("No valid frames to use for repair")
        return []
    
    # Use valid frames to replace invalid ones
    for i, frame in enumerate(frames):
        is_valid, _ = validate_frame(frame)
        if is_valid:
            repaired_frames.append(frame)
        else:
            # Replace with a valid frame (use nearest valid frame)
            if valid_frames:
                replacement = valid_frames[min(i, len(valid_frames)-1)]
                repaired_frames.append(replacement)
                logger.warning(f"Replaced invalid frame at position {i}")
    
    # Ensure we have the correct number of frames
    if len(repaired_frames) < target_num_frames:
        # Duplicate last frame to reach target count
        while len(repaired_frames) < target_num_frames:
            repaired_frames.append(repaired_frames[-1].copy())
        logger.warning(f"Padded frame sequence from {len(frames)} to {target_num_frames}")
    elif len(repaired_frames) > target_num_frames:
        # Truncate excess frames
        repaired_frames = repaired_frames[:target_num_frames]
        logger.warning(f"Truncated frame sequence from {len(frames)} to {target_num_frames}")
    
    return repaired_frames 