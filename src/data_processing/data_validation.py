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

def detect_frozen_frames(frames, threshold=None):
    """
    Detect frozen frames in a sequence by calculating differences between consecutive frames.
    
    Args:
        frames: List of video frames
        threshold: L2 difference threshold below which frames are considered frozen
        
    Returns:
        List of indices of frozen frames, total number of frozen frames
    """
    if not frames or len(frames) < 2:
        return [], 0
    
    if threshold is None:
        threshold = VALIDATION['frozen_frame_threshold']
    
    frozen_indices = []
    consecutive_frozen = 0
    max_consecutive = VALIDATION['max_consecutive_frozen']
    
    for i in range(1, len(frames)):
        # Convert to grayscale for comparison
        prev_gray = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        
        # Calculate L2 difference (normalized by image size)
        diff = np.sqrt(np.sum((prev_gray.astype(float) - curr_gray.astype(float))**2))
        diff_norm = diff / (prev_gray.shape[0] * prev_gray.shape[1])
        
        if diff_norm < threshold:
            frozen_indices.append(i)
            consecutive_frozen += 1
            
            # Check for too many consecutive frozen frames
            if consecutive_frozen > max_consecutive:
                logger.warning(f"Too many consecutive frozen frames detected: {consecutive_frozen}")
        else:
            consecutive_frozen = 0
    
    return frozen_indices, len(frozen_indices)

def check_motion_consistency(rgb_frames, flow_frames):
    """
    Check consistency between RGB frame changes and optical flow magnitude.
    Detects cases where RGB frames change but flow shows no motion or vice versa.
    
    Args:
        rgb_frames: List of RGB frames
        flow_frames: List of optical flow visualization frames
        
    Returns:
        Boolean indicating consistency, list of inconsistent frame indices
    """
    if len(rgb_frames) != len(flow_frames) or len(rgb_frames) < 2:
        return False, [], "Frame count mismatch or insufficient frames"
    
    inconsistent_indices = []
    
    for i in range(1, len(rgb_frames)):
        # Measure RGB frame difference
        prev_rgb = cv2.cvtColor(rgb_frames[i-1], cv2.COLOR_BGR2GRAY)
        curr_rgb = cv2.cvtColor(rgb_frames[i], cv2.COLOR_BGR2GRAY)
        rgb_diff = np.mean(np.abs(prev_rgb.astype(float) - curr_rgb.astype(float)))
        
        # Measure flow magnitude
        flow = flow_frames[i]
        hsv = cv2.cvtColor(flow, cv2.COLOR_BGR2HSV)
        flow_magnitude = np.mean(hsv[:,:,2])  # V channel contains magnitude
        
        # Check for inconsistency: significant RGB change but minimal flow,
        # or minimal RGB change but significant flow
        if (rgb_diff > 10 and flow_magnitude < 5) or (rgb_diff < 2 and flow_magnitude > 20):
            inconsistent_indices.append(i)
    
    # Consider consistent if less than 25% of frames are inconsistent
    is_consistent = len(inconsistent_indices) < len(rgb_frames) * 0.25
    
    if not is_consistent:
        msg = f"Motion inconsistency detected in {len(inconsistent_indices)} frames"
        logger.warning(msg)
        return False, inconsistent_indices, msg
    
    return True, inconsistent_indices, ""

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
    
    # Check for frozen frames if enabled in config
    frozen_indices, frozen_count = detect_frozen_frames(frames)
    if frozen_count > 0:
        logger.warning(f"Detected {frozen_count} frozen frames at indices: {frozen_indices}")
        # We don't consider frozen frames invalid, just log them
    
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

def validate_rgb_flow_pair(rgb_frames, flow_frames):
    """
    Validate the consistency between RGB frames and optical flow frames.
    
    Args:
        rgb_frames: List of RGB frames
        flow_frames: List of optical flow frames
        
    Returns:
        Boolean indicating if pair is valid and an error message if invalid
    """
    # Check if counts match (flow should have same count as RGB)
    if len(rgb_frames) != len(flow_frames):
        return False, f"Frame count mismatch: RGB={len(rgb_frames)}, Flow={len(flow_frames)}"
    
    # Check individual validity
    rgb_valid, _, rgb_error = validate_frames_sequence(rgb_frames)
    if not rgb_valid:
        return False, f"RGB frames invalid: {rgb_error}"
    
    flow_valid, flow_error = validate_optical_flow(flow_frames)
    if not flow_valid:
        return False, f"Flow frames invalid: {flow_error}"
    
    # Check motion consistency if enabled
    if VALIDATION['motion_consistency_check']:
        consistent, inconsistent_indices, msg = check_motion_consistency(rgb_frames, flow_frames)
        if not consistent:
            return False, msg
    
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
    
    # Repair frozen frames by interpolation if possible
    frozen_indices, _ = detect_frozen_frames(repaired_frames)
    if frozen_indices and len(frozen_indices) < len(repaired_frames) / 2:
        for idx in frozen_indices:
            # Find nearest non-frozen frame before and after
            before_idx = idx - 1
            while before_idx in frozen_indices and before_idx >= 0:
                before_idx -= 1
                
            after_idx = idx + 1
            while after_idx in frozen_indices and after_idx < len(repaired_frames):
                after_idx += 1
                
            # If valid frames found, interpolate
            if 0 <= before_idx < len(repaired_frames) and 0 <= after_idx < len(repaired_frames):
                before_frame = repaired_frames[before_idx].copy().astype(float)
                after_frame = repaired_frames[after_idx].copy().astype(float)
                weight = (idx - before_idx) / (after_idx - before_idx)
                
                # Linear interpolation
                interpolated = ((1 - weight) * before_frame + weight * after_frame).astype(np.uint8)
                repaired_frames[idx] = interpolated
                logger.info(f"Interpolated frozen frame at position {idx}")
    
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