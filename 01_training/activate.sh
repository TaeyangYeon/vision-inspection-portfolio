#!/bin/bash
# Virtual environment activation script for vision-inspection-portfolio training module

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate the virtual environment
source "$SCRIPT_DIR/.venv/bin/activate"

echo "Virtual environment activated! Python version: $(python --version)"