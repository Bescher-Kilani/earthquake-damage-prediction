#!/bin/bash
# One-time setup: creates a virtual environment and installs dependencies.
# Usage: bash setup.sh

set -e

echo "=== Creating virtual environment ==="
python3 -m venv venv
source venv/bin/activate

echo "=== Installing dependencies ==="
pip install --upgrade pip
pip install -r requirements.txt

mkdir -p data outputs

echo ""
echo "=== Setup complete ==="
echo "Next: download the data (see data/README.md), then run: bash run.sh"