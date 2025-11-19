# Installation Guide for SAM 3D Body

## Step-by-Step Installation

### 1. Create and Activate Environment

```bash
conda create -n sam_3d_body python=3.12 -y
conda activate sam_3d_body
```

### 2. Install PyTorch (select a version)

```bash
# CUDA 11.8
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
# CUDA 12.4
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

### 3. Install Python Dependencies

```bash
pip install pytorch-lightning pyrender opencv-python yacs scikit-image einops timm dill pandas rich hydra-core hydra-submitit-launcher hydra-colorlog pyrootutils webdataset chump networkx==3.2.1 roma joblib seaborn wandb appdirs appnope ffmpeg cython jsonlines pytest xtcocotools loguru optree
```

### 4. Install Detectron2

```bash
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9'
```

### 5. Install MoGe (Optional)

```bash
pip install git+https://github.com/microsoft/MoGe.git
```
