"""
Road Accident Detection - Data Processing Pipeline

This script runs the data processing pipeline using the configuration in data_processing/config.py.
Simply run this script to start the entire pipeline:

python src/run_data_processing.py

To retry processing previously failed videos:
python src/run_data_processing.py --reset-failed

To process only video frames (no optical flow):
python src/run_data_processing.py --frames-only

To process only optical flow (no video frames):
python src/run_data_processing.py --flow-only
"""

import os
import sys
import time
import logging
import traceback
import argparse
import subprocess
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if required dependencies are installed."""
    
    # Check FFmpeg
    ffmpeg_exists = shutil.which("ffmpeg") is not None
    if not ffmpeg_exists:
        logger.error("FFmpeg is not installed or not in the system PATH.")
        logger.error("Please install FFmpeg to continue. Instructions:")
        logger.error("  - Windows: Download from https://ffmpeg.org/download.html and add to PATH")
        logger.error("  - macOS: Install using Homebrew with 'brew install ffmpeg'")
        logger.error("  - Linux: Install using 'apt-get install ffmpeg' or equivalent for your distro")
        return False
    
    try:
        # Check FFmpeg version
        ffmpeg_version = subprocess.check_output(["ffmpeg", "-version"], stderr=subprocess.STDOUT).decode("utf-8")
        logger.info(f"Found FFmpeg: {ffmpeg_version.split('\\n')[0]}")
    except Exception as e:
        logger.warning(f"FFmpeg found but couldn't get version: {e}")
    
    # Check OpenCV
    try:
        import cv2
        logger.info(f"Found OpenCV version: {cv2.__version__}")
    except ImportError:
        logger.error("OpenCV (cv2) is not installed.")
        logger.error("Please install it using 'pip install opencv-python'")
        return False
    
    # Check PyTorch
    try:
        import torch
        logger.info(f"Found PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            logger.info(f"CUDA is available with {torch.cuda.device_count()} device(s)")
            logger.info(f"Current CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("CUDA is not available, will use CPU for processing (slower)")
    except ImportError:
        logger.error("PyTorch is not installed.")
        logger.error("Please install it following the instructions at https://pytorch.org/get-started/locally/")
        return False
    
    return True

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Road Accident Detection Data Processing Pipeline")
    parser.add_argument("--reset-failed", action="store_true", help="Reset the list of failed videos and attempt to process them again")
    parser.add_argument("--frames-only", action="store_true", help="Process and save only frames (no optical flow)")
    parser.add_argument("--flow-only", action="store_true", help="Process and save only optical flow (no frames)")
    args = parser.parse_args()
    
    # Validate arguments
    if args.frames_only and args.flow_only:
        logger.error("Cannot use both --frames-only and --flow-only flags together.")
        return 1
    
    # Check dependencies
    if not check_dependencies():
        logger.error("Missing required dependencies. Aborting.")
        return 1
    
    # Add the project root directory to the Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
        logger.info(f"Added project root to path: {parent_dir}")
    
    # Import after modifying path
    try:
        from src.data_processing.config import DATA_PATH, PROCESSING_OPTIONS
        from src.data_processing.dataset_creator import create_dataset
        logger.info("Successfully imported required modules")
        
        # Configure processing options
        processing_options = {
            'process_frames': not args.flow_only,
            'process_flow': not args.frames_only,
            'save_metadata': True
        }
        
        if args.frames_only:
            logger.info("Processing only frames (no optical flow) as requested by --frames-only flag")
        elif args.flow_only:
            logger.info("Processing only optical flow (no frames) as requested by --flow-only flag")
        else:
            logger.info("Processing both frames and optical flow")
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        logger.error(f"Import path details: {traceback.format_exc()}")
        logger.error(f"Python path: {sys.path}")
        logger.error("Make sure you're running this script from the project root directory.")
        return 1
    
    # Check if input directories exist
    if not os.path.exists(DATA_PATH['accidents']):
        logger.error(f"Accident videos directory not found: {DATA_PATH['accidents']}")
        logger.error("Please update the path in data_processing/config.py or create the directory.")
        return 1
        
    if not os.path.exists(DATA_PATH['non_accidents']):
        logger.error(f"Non-accident videos directory not found: {DATA_PATH['non_accidents']}")
        logger.error("Please update the path in data_processing/config.py or create the directory.")
        return 1
    
    # Create output directories if they don't exist
    os.makedirs(DATA_PATH['processed_parent'], exist_ok=True)
    
    # Run the data processing pipeline
    options_str = []
    if args.reset_failed:
        options_str.append("reset_failed=True")
    if args.frames_only:
        options_str.append("frames_only")
    if args.flow_only:
        options_str.append("flow_only")
    
    logger.info(f"Starting data processing pipeline{' with ' + ', '.join(options_str) if options_str else ''}...")
    
    start_time = time.time()
    
    try:
        create_dataset(reset_failed=args.reset_failed, processing_options=processing_options)
        
        elapsed_time = time.time() - start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        logger.info(f"Data processing completed successfully!")
        logger.info(f"Total processing time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        return 0
    except Exception as e:
        logger.error(f"Error during data processing: {e}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 