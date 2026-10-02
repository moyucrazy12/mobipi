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

# ---- CUDA toolkit for gsplat ----
# gsplat (used by nerfstudio's splatfacto) JIT-compiles its CUDA kernels on first use,
# which needs nvcc matching torch's CUDA version. Without it: "gsplat: No CUDA toolkit found".
setup_gsplat_cuda() {
  if ! command -v nvidia-smi &> /dev/null; then
    echo "No NVIDIA GPU detected, skipping CUDA toolkit setup for gsplat"
    return
  fi
  if [[ -z "$CONDA_PREFIX" ]]; then
    echo "⚠️  No active conda env, skipping CUDA toolkit setup for gsplat"
    return
  fi

  # Check torch's CUDA version at the end, since later pip installs may change the torch build
  TORCH_CUDA=$(python -c "import torch; print(torch.version.cuda or '')")
  case "$TORCH_CUDA" in
    11.7) CUDA_LABEL="cuda-11.7.1" ;;
    11.8) CUDA_LABEL="cuda-11.8.0" ;;
    12.1) CUDA_LABEL="cuda-12.1.1" ;;
    *)
      echo "⚠️  No CUDA toolkit mapping for torch CUDA '$TORCH_CUDA', skipping gsplat setup"
      return
      ;;
  esac

  echo "🧩 Installing CUDA toolkit $CUDA_LABEL for gsplat (torch CUDA $TORCH_CUDA)..."
  conda install -y -c "nvidia/label/$CUDA_LABEL" cuda-toolkit

  # Conda puts libcudart in lib/, but torch's extension builder links against $CUDA_HOME/lib64
  export CUDA_HOME="$CONDA_PREFIX"
  export LIBRARY_PATH="$CONDA_PREFIX/lib"

  # Older nvcc may not support the GPU's architecture (e.g. CUDA 11.7 vs Ada sm_89).
  # Build for the newest arch <= the GPU's that both nvcc and torch support, plus PTX,
  # which the driver JIT-compiles for the actual GPU.
  NVCC_ARCHS=$("$CONDA_PREFIX/bin/nvcc" --list-gpu-arch | grep -oP 'compute_\K[0-9]+' | tr '\n' ' ')
  export TORCH_CUDA_ARCH_LIST=$(python -c "
import inspect, re, torch
from torch.utils import cpp_extension
gpu = torch.cuda.get_device_capability()
torch_archs = set(re.findall(r\"'(\d+\.\d+)'\", inspect.getsource(cpp_extension._get_cuda_arch_flags)))
archs = [(int(a[:-1]), int(a[-1])) for a in '$NVCC_ARCHS'.split()]
archs = [a for a in archs if a <= gpu and '%d.%d' % a in torch_archs]
print('%d.%d+PTX' % max(archs))
")

  # Persist the variables in the conda env (applied on every `conda activate`)
  conda env config vars set CUDA_HOME="$CUDA_HOME" LIBRARY_PATH="$LIBRARY_PATH" TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST"

  echo "🔨 Compiling gsplat CUDA kernels (TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST), this takes a few minutes..."
  python -c "from gsplat.cuda._backend import _C; assert _C is not None, 'gsplat CUDA build failed'"
  echo "ℹ️  Re-activate the env to load the new variables: conda deactivate && conda activate ${CONDA_DEFAULT_ENV}"
}

setup_gsplat_cuda

echo "✅ All done!"
