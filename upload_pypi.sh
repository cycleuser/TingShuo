#!/bin/bash
# Upload TingShuo to PyPI
# Prerequisites: pip install build twine
set -e

echo "=== TingShuo PyPI Upload ==="

# Clean previous builds
echo "Cleaning old build artifacts ..."
rm -rf dist/ build/ *.egg-info

# Build
echo "Building package ..."
python -m build

# Check
echo "Checking package ..."
python -m twine check dist/*

# Upload
echo "Uploading to PyPI ..."
python -m twine upload dist/*

echo ""
echo "=== Upload complete! ==="
echo "Install with: pip install tingshuo"
