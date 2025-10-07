conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1  pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c nvidia -c conda-forge cuda=11.8 cuda-nvcc=11.8 cuda-compiler=11.8 cuda-gdb=11.8 cuda-libraries=11.8 cuda-cudart=11.8  cuda-cudart-dev=11.8 cuda-libraries-dev=11.8 cuda-nsight-compute=11.8 cuda-nsight=11.8 cuda-nvdisasm=11.8 cuda-nvprof=11.8 cuda-nvvp=11.8 cuda-nvprune=11.8 cuda-profiler-api=11.8 cuda-cccl=11.8 cuda-driver-dev=11.8 cuda-cuobjdump=11.8 cuda-nvml-dev=11.8 cuda-sanitizer-api=11.8 cuda-cuxxfilt=11.8 --strict
conda install pymomentum=0.1.20="cuda118*" pytorch==2.5.1="cuda118*" torchvision==0.20.1="cuda118*" torchaudio==2.5.1="cuda_118*" -c nvidia
conda install cmake==3.25
conda install gcc=11.4.0 gxx=11.4.0
conda install xformers=="0.0.29.post1"="cuda_118*"
pip install ninja
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.9;9.0"
export CUDA_HOME=$CONDA_PREFIX
export FORCE_CUDA=1
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

export FORCE_CUDA=1
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
pip install pytorch-lightning smplx pyrender opencv-python yacs scikit-image einops timm dill pandas rich hydra-core hydra-submitit-launcher hydra-colorlog pyrootutils webdataset chumpy
pip install networkx==3.2.1 roma joblib seaborn wandb appdirs appnope ffmpeg cython jsonlines pytest xtcocotools loguru optree
pip install flash-attn==2.6.3