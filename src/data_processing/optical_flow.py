import cv2
import numpy as np
import logging
import torch
import gc
from tqdm import tqdm
from data_processing.data_validation import validate_frame, validate_optical_flow

# Import config
from data_processing.config import PARALLEL_PROCESSING, GPU_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_optical_flow(frames, use_gpu=None, batch_size=None):
    """
    Compute optical flow between consecutive frames.
    
    Args:
        frames: List of video frames
        use_gpu: Whether to use GPU acceleration (None for auto-detection)
        batch_size: Number of frame pairs to process at once (None to use config value)
        
    Returns:
        List of optical flow visualization frames
    """
    if len(frames) <= 1:
        logger.warning("Not enough frames to compute optical flow")
        return [np.zeros_like(frames[0]) for _ in range(len(frames))]
    
    # Use batch size from config if not specified
    if batch_size is None:
        batch_size = PARALLEL_PROCESSING['batch_size']
    
    # Auto-detect GPU if not specified
    if use_gpu is None:
        use_gpu = GPU_CONFIG['use_gpu']
        # If still None, auto-detect
        if use_gpu is None:
            use_gpu = torch.cuda.is_available()
    
    # Use CUDA if available and requested
    if use_gpu and torch.cuda.is_available():
        try:
            # Try to use GPU-based flow algorithm (DIS or DenseRLOF)
            flow_algorithm = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
            logger.info("Using GPU-accelerated optical flow with DIS algorithm")
        except:
            logger.warning("GPU-accelerated optical flow not available, falling back to CPU")
            use_gpu = False
    else:
        use_gpu = False
        
    if not use_gpu:
        logger.info("Using CPU-based optical flow with Farneback algorithm")
    
    flow_frames = []
    
    # Add a zero flow for the first frame
    flow_frames.append(np.zeros_like(frames[0]))
    
    # Convert all frames to grayscale first to save memory
    gray_frames = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frames.append(gray)
    
    # Process in batches to manage memory
    for batch_start in range(0, len(frames)-1, batch_size):
        batch_end = min(batch_start + batch_size, len(frames)-1)
        batch_flow_frames = []
        
        for i in range(batch_start, batch_end):
            prev_gray = gray_frames[i]
            curr_gray = gray_frames[i+1]
            
            try:
                if use_gpu:
                    # Use GPU algorithm
                    flow = flow_algorithm.calc(prev_gray, curr_gray, None)
                else:
                    # Use Farneback on CPU
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray, curr_gray, None, 
                        pyr_scale=0.5, levels=3, winsize=15, 
                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                    )
                
                # Convert flow to RGB visualization
                magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                hsv = np.zeros_like(frames[i+1])
                hsv[..., 1] = 255
                hsv[..., 0] = angle * 180 / np.pi / 2
                hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
                flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                
                batch_flow_frames.append(flow_rgb)
            except Exception as e:
                logger.error(f"Error computing optical flow: {e}")
                # Use zero flow if calculation fails
                batch_flow_frames.append(np.zeros_like(frames[i+1]))
        
        flow_frames.extend(batch_flow_frames)
        
        # Force memory cleanup
        batch_flow_frames = None
        gc.collect()
    
    # Validate optical flow results
    is_valid, error = validate_optical_flow(flow_frames)
    if not is_valid:
        logger.warning(f"Optical flow validation failed: {error}")
        # Return the frames anyway, but log the warning
    
    return flow_frames

def save_optical_flow_frames(flow_frames, output_dir, video_name, batch_size=None):
    """
    Save optical flow frames to disk with batched processing.
    
    Args:
        flow_frames: List of optical flow visualization frames
        output_dir: Directory to save flow frames
        video_name: Base name for the saved frames
        batch_size: Number of frames to save at once (None to use config value)
        
    Returns:
        Boolean indicating success or failure
    """
    # Use batch size from config if not specified
    if batch_size is None:
        batch_size = PARALLEL_PROCESSING['batch_size']
        
    try:
        for i in range(0, len(flow_frames), batch_size):
            batch_frames = flow_frames[i:i+batch_size]
            
            for j, frame in enumerate(batch_frames):
                frame_idx = i + j
                output_path = f"{output_dir}/{video_name}_flow_{frame_idx:02d}.jpg"
                cv2.imwrite(output_path, frame)
                
                # Verify the saved frame
                saved_frame = cv2.imread(output_path)
                is_valid, error = validate_frame(saved_frame)
                if not is_valid:
                    logger.error(f"Invalid saved flow frame: {output_path}, {error}")
            
            # Clear batch from memory
            batch_frames = None
            gc.collect()
            
        return True
    except Exception as e:
        logger.error(f"Error saving optical flow frames: {e}")
        return False

def compute_flow_features(flow_frames, batch_size=None):
    """
    Extract simple features from optical flow frames with batched processing.
    
    Args:
        flow_frames: List of optical flow visualization frames
        batch_size: Number of frames to process at once (None to use config value)
        
    Returns:
        Array of features
    """
    # Use batch size from config if not specified
    if batch_size is None:
        batch_size = PARALLEL_PROCESSING['batch_size']
        
    all_features = []
    
    for i in range(0, len(flow_frames), batch_size):
        batch_frames = flow_frames[i:i+batch_size]
        batch_features = []
        
        for frame in batch_frames:
            # Convert to HSV to separate magnitude and angle
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Extract magnitude (V channel)
            magnitude = hsv[:, :, 2]
            
            # Calculate simple statistics
            mean_magnitude = np.mean(magnitude)
            std_magnitude = np.std(magnitude)
            max_magnitude = np.max(magnitude)
            
            # Calculate magnitude in different regions (simple motion distribution)
            h, w = magnitude.shape
            regions = [
                magnitude[:h//2, :w//2],              # Top-left
                magnitude[:h//2, w//2:],              # Top-right
                magnitude[h//2:, :w//2],              # Bottom-left
                magnitude[h//2:, w//2:],              # Bottom-right
                magnitude[h//4:3*h//4, w//4:3*w//4]   # Center
            ]
            
            region_means = [np.mean(region) for region in regions]
            
            # Combine features
            frame_features = [mean_magnitude, std_magnitude, max_magnitude] + region_means
            batch_features.append(frame_features)
        
        all_features.extend(batch_features)
        
        # Clear batch from memory
        batch_frames = None
        batch_features = None
        gc.collect()
    
    return np.array(all_features)

def compute_flow_gpu(prev, curr):
    """
    Attempt to compute optical flow using GPU-accelerated algorithms.
    
    Args:
        prev: Previous frame (grayscale)
        curr: Current frame (grayscale)
        
    Returns:
        Optical flow field
    """
    try:
        # Try to use GPU-based optical flow if CUDA is available
        if torch.cuda.is_available():
            try:
                # DIS optical flow (faster)
                flow_algorithm = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
                flow = flow_algorithm.calc(prev, curr, None)
                return flow
            except:
                # Fall back to DenseRLOF
                try:
                    rlof = cv2.optflow.DenseRLOFOpticalFlow_create()
                    flow = rlof.calc(prev, curr, None)
                    return flow
                except:
                    logger.warning("GPU-based optical flow failed, falling back to CPU")
                    
        # Fallback to Farneback
        flow = cv2.calcOpticalFlowFarneback(
            prev, curr, None, 
            pyr_scale=0.5, levels=3, winsize=15, 
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        return flow
    except Exception as e:
        logger.error(f"Error computing optical flow: {e}")
        return None