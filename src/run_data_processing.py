"""
Road Accident Detection - Data Processing Pipeline

This script runs the data processing pipeline using the configuration in data_processing/config.py.
Simply run this script to start the entire pipeline:

python src/run_data_processing.py
"""

import os
import sys
import time
import logging
import traceback

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

def main():
    # Add the project root directory to the Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
        logger.info(f"Added project root to path: {parent_dir}")
    
    # Import after modifying path
    try:
        from src.data_processing.config import DATA_PATH
        from src.data_processing.dataset_creator import create_dataset
        logger.info("Successfully imported required modules")
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
    logger.info("Starting data processing pipeline...")
    start_time = time.time()
    
    try:
        create_dataset()
        
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