#!/bin/bash
# Setup script for Earthquake Damage Prediction Pipeline — Team 2
# Run once after cloning the repository.
# Usage: bash setup.sh

set -e  # stop on first error

echo "=== Setting up virtual environment ==="
python3 -m venv venv
source venv/bin/activate

echo "=== Installing dependencies ==="
pip install --upgrade pip
pip install -r requirements.txt

echo "=== Creating output directory ==="
mkdir -p outputs
mkdir -p data

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Copy data files into data/:"
echo "       data/train_values.csv"
echo "       data/train_labels.csv"
echo "       data/test_values.csv"
echo ""
echo "  2. Activate the virtual environment (required before every session):"
echo "       source venv/bin/activate"
echo ""
echo "  3. Run the pipeline:"
echo "       python3 src/pipeline_sklearn.py"