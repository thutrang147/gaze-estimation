#!/bin/bash

# Script to run the WebCam Gaze Estimation application
# Usage: ./run.sh [main|pl|app]

# Activate conda environment
source /opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh
conda activate gaze_estimation

# Get the directory where this script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# Check which mode to run
MODE=${1:-main}

case $MODE in
  main)
    echo "Running OpenVINO-based gaze estimation..."
    python src/main.py
    ;;
  pl)
    echo "Running PyTorch pl_gaze model..."
    python src/main_pl.py
    ;;
  app)
    echo "Running Flask web interface..."
    echo "Open http://localhost:5000 in your browser"
    python src/app.py
    ;;
  *)
    echo "Usage: ./run.sh [main|pl|app]"
    echo "  main - Run OpenVINO-based model (default)"
    echo "  pl   - Run PyTorch pl_gaze model"
    echo "  app  - Run Flask web interface"
    exit 1
    ;;
esac
