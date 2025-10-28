#!/bin/bash
# install_git_deps.sh
# Script to install git-based dependencies for SAM 3D Body

set -e  # Exit on error

echo "=========================================="
echo "SAM 3D Body - Git Dependencies Installer"
echo "=========================================="
echo ""

# Check if pip is available
if ! command -v pip &> /dev/null; then
    echo "Error: pip is not installed or not in PATH"
    exit 1
fi

echo "Installing git-based dependencies..."
echo ""

# Install PyMomentum (required)
echo "[1/5] Installing PyMomentum..."
pip install "git+https://github.com/facebookresearch/momentum@77c3994"
echo "✓ PyMomentum installed"
echo ""

# Install Detectron2 (required for human detection)
echo "[2/5] Installing Detectron2..."
pip install "git+https://github.com/facebookresearch/detectron2.git@a1ce2f9"
echo "✓ Detectron2 installed"
echo ""

# Install PyTorch3D (optional)
echo "[3/5] Installing PyTorch3D (optional)..."
read -p "Install PyTorch3D? This may take several minutes. [y/N] " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip install "git+https://github.com/facebookresearch/pytorch3d.git@75ebeea"
    echo "✓ PyTorch3D installed"
else
    echo "⊘ Skipped PyTorch3D"
fi
echo ""

# Install MoGe (optional)
echo "[4/5] Installing MoGe (optional, for FOV estimation)..."
read -p "Install MoGe? [y/N] " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip install "git+https://github.com/microsoft/MoGe.git"
    echo "✓ MoGe installed"
else
    echo "⊘ Skipped MoGe"
fi
echo ""

# Install Flash Attention (optional)
echo "[5/5] Installing Flash Attention (optional, requires CUDA)..."
read -p "Install Flash Attention 2.7.3? Requires CUDA and proper compiler setup. [y/N] " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip install flash-attn==2.7.3
    echo "✓ Flash Attention installed"
else
    echo "⊘ Skipped Flash Attention"
fi
echo ""

echo "=========================================="
echo "Git dependencies installation complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Download model checkpoints from HuggingFace"
echo "  2. See examples/README.md for usage instructions"
echo ""
