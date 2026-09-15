#!/bin/bash

set -e  # Exit on error

echo "🔄 Initializing submodules..."
git submodule update --init --recursive

echo "✅ Installing submodule dependencies..."

# ---- robocasa ----
echo "➡️  Setting up robocasa..."
pip install robosuite==1.5.0
pip install -e external/robocasa
(
  cd external/robocasa
  pip install pre-commit
  pre-commit install
  echo "y" | python robocasa/scripts/download_kitchen_assets.py
  python robocasa/scripts/setup_macros.py
)

# ---- install torch ----
echo "🔥 Installing pytorch..."

# Desired PyTorch version
TORCH_VERSION="2.2.0"
TORCHVISION_VERSION="0.17.0"
TORCHAUDIO_VERSION="2.2.0"

# Function to install torch with appropriate CUDA version
install_torch() {
  if command -v nvidia-smi &> /dev/null; then
    CUDA_MAIN_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+' | head -1)

    echo "Detected CUDA major version: $CUDA_MAIN_VERSION"

    if [[ "$CUDA_MAIN_VERSION" -ge 12 ]]; then
      echo "Installing torch with CUDA 12.1"
      pip install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION} --index-url https://download.pytorch.org/whl/cu121
    elif [[ "$CUDA_MAIN_VERSION" -eq 11 ]]; then
      CUDA_MINOR_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: 11\.\K[0-9]+' | head -1)
      if [[ "$CUDA_MINOR_VERSION" -ge 8 ]]; then
        echo "Installing torch with CUDA 11.8"
        pip install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION} --index-url https://download.pytorch.org/whl/cu118
      else
        echo "CUDA 11 version too old, falling back to CPU"
        pip install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION}
      fi
    else
      echo "Unsupported CUDA version, falling back to CPU"
      pip install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION}
    fi
  else
    echo "CUDA not detected, installing CPU version"
    pip install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION}
  fi
}

install_torch

# ---- lelan ----
echo "➡️  Installing lelan dependencies..."
pip install tqdm==4.64.0
pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git
pip install opencv-python==4.6.0.66
pip install h5py==3.6.0
pip install wandb==0.12.18
pip install prettytable efficientnet-pytorch warmup-scheduler
pip install diffusers==0.11.1
pip install openai-clip==1.0.1
pip install scikit-video==1.1.11
pip install open3d==0.19.0
pip install lmdb vit-pytorch positional-encodings
pip install -e external/lelan/train

# ---- Other submodules (basic install only) ----
echo "➡️  Installing remaining submodules..."
pip install -e external/diffusion_policy
pip install -e external/mimicgen
pip install -e external/robomimic

# ---- Main package ----
echo "🎯 Installing main package..."
pip install -e .

# ---- Correct a few package versions ----
pip install nerfstudio==1.1.5 scikit-optimize cma sentencepiece peft==0.10.0 transformers==4.36.0 huggingface-hub==0.25.0 numpy==1.23.3
pip install --upgrade --no-deps timm==1.0.12

echo "✅ All done!"
