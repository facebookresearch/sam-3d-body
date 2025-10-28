# Installation Guide for SAM 3D Body

## Quick Installation from PyPI

For most users, the simplest way to install SAM 3D Body is via PyPI:

```bash
# Basic installation
pip install sam-3d-body

# With visualization support
pip install sam-3d-body[vis]

# With all optional features
pip install sam-3d-body[full]
```

After installing from PyPI, you'll need to install the git-based dependencies (see [Git Dependencies](#git-dependencies) below).

## Development Installation from Source

If you want to develop or modify SAM 3D Body, follow the step-by-step installation below.

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
conda install pytorch==2.5.1="cuda118*" torchvision==0.20.1="cuda118*" torchaudio==2.5.1="cuda_118*" -c nvidia -y
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

### 6. Install Ninja

```bash
pip install ninja
```

### 7. Install PyMomentum

```bash
conda install -c conda-forge ceres-solver=2.2.0 cli11=2.5.0 dispenso=1.4.0 -y
conda install -c conda-forge eigen=3.4.0 ezc3d=1.5.19 drjit=1.2.0 fmt=11.2.0 -y
conda install -c conda-forge fx-gltf=2.0.0 indicators=2.3 kokkos=4.6.2 ms-gsl=4.2.0 openfbx=0.9 spdlog=1.15.3 -y
conda install -c conda-forge urdfdom=4.0.1 nlohmann_json=3.12.0 librerun-sdk=0.17.0 re2 -y
conda install boost=1.84.0 -y

export CFLAGS="-I$CONDA_PREFIX/include ${CFLAGS} -fPIC"
export CXXFLAGS="-I$CONDA_PREFIX/include ${CXXFLAGS} -fPIC -std=c++20"
export CMAKE_ARGS="-DMOMENTUM_BUILD_TESTING=OFF -DMOMENTUM_BUILD_RENDERER=OFF -DMOMENTUM_ENABLE_SIMD=OFF"
export CMAKE_PREFIX_PATH="$CONDA_PREFIX"
export CMAKE_SYSTEM_PREFIX_PATH="$CONDA_PREFIX"
export CMAKE_FIND_ROOT_PATH="$CONDA_PREFIX"
export CMAKE_FIND_ROOT_PATH_MODE_PROGRAM=NEVER
export CMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY
export CMAKE_FIND_ROOT_PATH_MODE_INCLUDE=ONLY
export CMAKE_FIND_ROOT_PATH_MODE_PACKAGE=ONLY
export CMAKE_FIND_USE_PACKAGE_REGISTRY=OFF
export CMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=OFF
export PKG_CONFIG_PATH="$CONDA_PREFIX/lib/pkgconfig:$CONDA_PREFIX/share/pkgconfig"

pip install "git+https://github.com/facebookresearch/momentum@77c3994"
```

### 8. Install Python Dependencies

```bash
pip install pytorch-lightning smplx pyrender opencv-python yacs scikit-image einops timm dill pandas rich hydra-core hydra-submitit-launcher hydra-colorlog pyrootutils webdataset chump

pip install networkx==3.2.1 roma joblib seaborn wandb appdirs appnope ffmpeg cython jsonlines pytest xtcocotools loguru optree
```

### 9. Install Detectron2

```bash
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9'
```

### 10. Install PyTorch3D

```bash
pip install "git+https://github.com/facebookresearch/pytorch3d.git@75ebeea"
```

### 11. Install FlashAttn (Optional)

```bash
pip install flash-attn==2.7.3
```

### 12. Install MoGe (Optional)

```bash
pip install git+https://github.com/microsoft/MoGe.git
```

### 13. Install xFormers (Optional)

```bash
conda install xformers=="0.0.29.post1"="cuda_118*" -y
```

### 14. Install SAM 3D Body Package

After installing all dependencies, install the SAM 3D Body package itself:

```bash
# For development (editable install)
pip install -e .

# Or for regular installation
pip install .
```

## Git Dependencies

The following dependencies require git installation and cannot be installed via PyPI. These are already covered in the step-by-step guide above, but for reference:

```bash
# PyMomentum (required for model architecture)
pip install git+https://github.com/facebookresearch/momentum@77c3994

# Detectron2 (required for human detection)
pip install git+https://github.com/facebookresearch/detectron2.git@a1ce2f9

# PyTorch3D (optional, for advanced 3D operations)
pip install git+https://github.com/facebookresearch/pytorch3d.git@75ebeea

# MoGe (optional, for FOV estimation)
pip install git+https://github.com/microsoft/MoGe.git

# Flash Attention (optional, for faster inference)
pip install flash-attn==2.7.3
```

For a convenience script that installs all git dependencies, see [`install_git_deps.sh`](./install_git_deps.sh).