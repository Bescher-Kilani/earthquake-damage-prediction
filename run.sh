#!/bin/bash
# Run script for Earthquake Damage Prediction Pipeline — Team 2
# Activates the virtual environment and runs the pipeline in one step.
# Usage:
#   bash run.sh                  → generate submission file (default)
#   bash run.sh --compare        → model comparison with CV
#   bash run.sh --sampling       → sampling strategy comparison
#   bash run.sh --two-stage      → two-stage classifier experiment
#   bash run.sh --help           → show all available options

set -e  # stop on first error

# Check that venv exists
if [ ! -d "venv" ]; then
    echo "ERROR: Virtual environment not found."
    echo "Please run 'bash setup.sh' first."
    exit 1
fi

# Check that data files exist
if [ ! -f "data/train_values.csv" ] || [ ! -f "data/train_labels.csv" ] || [ ! -f "data/test_values.csv" ]; then
    echo "ERROR: Data files not found in data/."
    echo "Please copy train_values.csv, train_labels.csv, test_values.csv into data/."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Run pipeline — pass all arguments through
python3 src/pipeline_sklearn.py "$@"
