import os
import sys
import time
import logging
import torch
from datetime import datetime

# Add parent directory to path if necessary
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from feature_extraction.config import YOLO_CONFIG, GPU_CONFIG
from feature_extraction.feature_extraction import extract_features_from_dataset

# Configure logging with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"feature_extraction_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Execute the YOLO feature extraction pipeline using config parameters."""
    start_time = time.time()
    
    # Get parameters from config
    data_dir = YOLO_CONFIG['processed_data']
    output_dir = YOLO_CONFIG['features_output']
    model_path = YOLO_CONFIG['model_path']
    device = 'cuda' if GPU_CONFIG['use_gpu'] and torch.cuda.is_available() else 'cpu'
    
    logger.info("Starting YOLO11 feature extraction")
    logger.info(f"Processing data from: {data_dir}")
    logger.info(f"Saving features to: {output_dir}")
    logger.info(f"Using model: {model_path}")
    logger.info(f"Using device: {device}")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract features
    extract_features_from_dataset(
        data_dir=data_dir,
        output_dir=output_dir,
        model_path=model_path,
        device=device
    )
    
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info(f"Feature extraction completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")

if __name__ == "__main__":
    main()