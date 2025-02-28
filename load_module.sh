#!/bin/bash

# This script loads the necessary modules or dependencies for project prj_smp
#
# Make it executable:
# $ chmod +x load_modules.sh
#
# Usage:
#    $ . load_modules.sh
# OR $ source load_modules.sh  (This ensures that environment changes persist in your terminal session)
# 
# Run it under the home directory of repo `prj_smp`
#
# Last updated: Feb 27, 2025

# Load a Python module
echo "Loading Python module..."
module load python/3.12.4 || { echo "Failed to load Python module"; exit 1; }

# Load opencv-python module
echo "Loading OpenCV module..."
module load opencv/4.10.0 || { echo "Failed to load OpenCV module"; exit 1; }

# Load Git LFS module
echo "Loading Git LFS module..."
module load git-lfs || { echo "Failed to load Git LFS module"; exit 1; }

# Activate the virtual environment
echo "Activating virtual environment..."
source env/bin/activate || { echo "Failed to activate virtual environment"; exit 1; }

# Upgrade pip in the virtual environment
# (--no-index is to download the available packages from the Alliance instead of from PyPI)
echo "Upgrading pip..."
pip install --no-index --upgrade pip || { echo "Failed to upgrade pip"; exit 1; }

# Install the packages listed on the requirements.txt
echo "Installing dependencies..."
pip install --no-index -r requirements.txt || { echo "Failed to install dependencies"; exit 1; }