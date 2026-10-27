#!/bin/bash
# Runs the pipeline inside the virtual environment. All arguments are passed through.
# Usage:
#   bash run.sh                  generate submission file (default)
#   bash run.sh --compare        model comparison with 5-fold CV
#   bash run.sh --sampling       sampling strategy comparison
#   bash run.sh --two-stage      two-stage classifier experiment
#   bash run.sh --tune-optuna    hyperparameter tuning with Optuna

set -e

if [ ! -d "venv" ]; then
    echo "ERROR: virtual environment not found. Run 'bash setup.sh' first."
    exit 1
fi

if [ ! -f "data/train_values.csv" ] || [ ! -f "data/train_labels.csv" ] || [ ! -f "data/test_values.csv" ]; then
    echo "ERROR: data files missing in data/. See data/README.md for the download link."
    exit 1
fi

source venv/bin/activate
python3 src/pipeline_sklearn.py "$@"