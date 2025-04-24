#!/bin/bash

# Setup script for YOLO11 feature extraction
set -e  # Exit on any error

# Navigate to project root
echo "Navigating to project root..."
cd "$(dirname "$0")/../.."
PROJECT_ROOT=$(pwd)
echo "Project root: $PROJECT_ROOT"

# Check if a virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics opencv-python matplotlib numpy tqdm

# Ensure directories exist
echo "Creating necessary directories..."
mkdir -p data/features
mkdir -p test_results

# Download YOLO11n model and test feature extraction
echo "Testing YOLO11 feature extraction..."
python src/feature_extraction/test_feature_extraction.py

# Check if test was successful
if [ $? -ne 0 ]; then
    echo "Test failed. Please check the error messages."
    exit 1
fi

# Run feature extraction
echo "Running full feature extraction pipeline..."
python src/feature_extraction/run_feature_extraction.py

echo "Feature extraction completed."
echo "Features saved to: $PROJECT_ROOT/data/features"
echo "---------------------------------------------"
echo "Thank you for using YOLO11 feature extraction!" 