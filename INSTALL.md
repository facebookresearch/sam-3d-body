# Installation Guide for SAM 3D Body

## Step-by-Step Installation

### 1. Create and Activate Environment

```bash
conda create -n sam_3d_body python=3.12 -y
conda activate sam_3d_body
```

### 2. Install CUDA 11.8

```bash
conda install -c nvidia -c conda-forge cuda=11.8 cuda-nvcc=11.8 cuda-compiler=11.8 cuda-gdb=11.8 cuda-libraries=11.8 cuda-cudart=11.8  cuda-cudart-dev=11.8 cuda-libraries-dev=11.8 cuda-nsight-compute=11.8 cuda-nsight=11.8 cuda-nvdisasm=11.8 cuda-nvprof=11.8 cuda-nvvp=11.8 cuda-nvprune=11.8 cuda-profiler-api=11.8 cuda-cccl=11.8 cuda-driver-dev=11.8 cuda-cuobjdump=11.8 cuda-nvml-dev=11.8 cuda-sanitizer-api=11.8 cuda-cuxxfilt=11.8 --strict -y
```

### 3. Install PyMomentum and PyTorch 2.5.1

```bash
conda install pymomentum=0.1.20="cuda118*" pytorch==2.5.1="cuda118*" torchvision==0.20.1="cuda118*" torchaudio==2.5.1="cuda_118*" -c nvidia -y
```

### 4. Install CMake, GCC and GXX

```bash
conda install cmake==3.25 gcc=11.4.0 gxx=11.4.0 -y
```

### 5. Export Environment Variables

```bash
export CC="$CONDA_PREFIX/bin/gcc"
export CXX="$CONDA_PREFIX/bin/g++"
export CUDAHOSTCXX="$CONDA_PREFIX/bin/g++"

export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.9;9.0"
export CUDA_HOME=$CONDA_PREFIX
export FORCE_CUDA=1
```

### 6. Install Python Dependencies

```bash
pip install pytorch-lightning smplx pyrender opencv-python yacs scikit-image einops timm dill pandas rich hydra-core hydra-submitit-launcher hydra-colorlog pyrootutils webdataset chumpy

pip install networkx==3.2.1 roma joblib seaborn wandb appdirs appnope ffmpeg cython jsonlines pytest xtcocotools loguru optree
```

### 7. Install Ninja

```bash
pip install ninja
```

### 8. Install Detectron2

```bash
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9'
```

### 9. Install PyTorch3D

```bash
pip install "git+https://github.com/facebookresearch/pytorch3d.git@75ebeea"
```

### 10. Install FlashAttn (Optional)

```bash
pip install flash-attn==2.7.3
```

### 11. Install MoGe (Optional)

```bash
pip install git+https://github.com/microsoft/MoGe.git
```

### 12. Install xFormers (Optional)

```bash
conda install xformers=="0.0.29.post1"="cuda_118*" -y
```