#!/bin/bash

# Setup script for Vast AI deployment
# Run this script on your Vast AI instance

echo "Setting up environment for Atari checkpoint processing..."

# Update system packages
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev

# Install Python dependencies
pip3 install --upgrade pip
pip3 install -r requirements.txt

# Verify installations
echo "Verifying installations..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
python3 -c "import tensorflow_datasets as tfds; print(f'TFDS version: {tfds.__version__}')"

echo "Setup complete! You can now run: python3 pull_checkpoints.py"
