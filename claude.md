# SAM 3D Body - AI Assistant Guide

This document provides essential context for AI assistants working with the SAM 3D Body codebase.

## Project Overview

**SAM 3D Body (3DB)** is a promptable foundation model for single-image 3D human mesh recovery (HMR). Developed by Meta AI Research (FAIR), it emphasizes data quality and diversity to maximize performance.

### Key Features
- Encoder-decoder architecture with auxiliary prompt support (2D keypoints, masks)
- XR Body (XRB) parametric mesh representation that decouples skeletal pose and body shape
- High-quality annotations from multi-stage pipeline
- Robust performance on challenging scenarios (occlusions, rare poses)
- SAM-family inspired promptable interface

### Technology Stack
- **PyTorch 2.5.1** with CUDA 11.8
- **PyTorch Lightning** for training orchestration
- **PyMomentum** for mesh representation (git dependency)
- **Detectron2** for human detection (git dependency)
- **PyTorch3D** for 3D operations (git dependency, optional)
- **Hydra** for configuration management
- **HuggingFace Hub** for model distribution (built-in)

### Recent Major Changes
- **Package renamed**: `core/` → `sam_3d_body/` (October 2024)
- **Demo relocated**: `demo.py` → `examples/demo.py`
- **Builder API added**: `build_sam_3d_body_model()`, `build_sam_3d_body_hf()`
- **PyPI preparation**: Added pyproject.toml, packaging infrastructure
- **Import namespace**: All imports now use `sam_3d_body.*` instead of `core.*`

## Codebase Structure

```
sam-3d-body/
├── sam_3d_body/              # Main package (formerly 'core/')
│   ├── __init__.py           # Package exports, version
│   ├── estimator.py          # SAM3DBodyEstimator class (main interface)
│   ├── build_sam.py          # Builder API functions
│   │
│   ├── models/               # Model architectures
│   │   ├── meta_arch/        # High-level model classes
│   │   │   ├── sam3d_body.py         # Main SAM3DBody model
│   │   │   ├── sam3d_body_triplet.py # Triplet variant with hand processing
│   │   │   └── base_model.py         # Base classes
│   │   ├── backbones/        # Encoder architectures
│   │   │   └── vit_hmr2.py   # Vision transformer backbone
│   │   ├── decoders/         # Decoder components
│   │   │   ├── promptable_decoder.py # Main decoder
│   │   │   ├── prompt_encoder.py     # Prompt encoding
│   │   │   └── keypoint_prompt_sampler.py
│   │   ├── heads/            # Prediction heads
│   │   │   ├── atlas_head.py # Main mesh prediction head
│   │   │   └── camera_head.py # Camera parameter prediction
│   │   └── modules/          # Shared components
│   │       ├── atlas46.py    # Atlas mesh utilities
│   │       ├── transformer.py
│   │       └── geometry_utils.py
│   │
│   ├── data/                 # Data processing
│   │   ├── transforms/       # Image/data transforms
│   │   │   ├── common.py
│   │   │   └── bbox_utils.py
│   │   └── utils/
│   │       └── io.py         # Loading utilities
│   │
│   ├── metadata/             # Keypoint definitions
│   │   ├── atlas70.py        # 70-keypoint system
│   │   ├── coco.py           # COCO keypoints
│   │   ├── h36m.py           # Human3.6M keypoints
│   │   ├── openpose.py       # OpenPose keypoints
│   │   └── joint14.py
│   │
│   ├── visualization/        # Rendering and visualization
│   │   ├── renderer.py       # Main renderer
│   │   ├── mesh_renderer.py
│   │   ├── skeleton_renderer.py
│   │   ├── skeleton_visualizer.py
│   │   └── pytorch3d_renderer.py
│   │
│   └── utils/               # General utilities
│       ├── checkpoint.py    # Model loading/saving
│       ├── config.py        # Configuration utilities
│       ├── logging.py
│       └── rotation_utils.py
│
├── examples/                # Usage examples
│   ├── demo.py              # Main demo script (moved from root)
│   └── README.md            # Example documentation
│
├── assets/                  # Images, diagrams for README
├── pyproject.toml          # Package configuration
├── requirements.txt        # Python dependencies
├── MANIFEST.in             # Package manifest
├── PUBLISHING.md           # PyPI publishing guide
├── README.md               # Project documentation
├── INSTALL.md              # Installation guide
└── LICENSE                 # Apache 2.0
```

## Domain Knowledge

### Human Mesh Recovery (HMR)
The task of estimating a 3D human body mesh from a 2D image. Outputs typically include:
- **3D vertices**: Points defining the body surface mesh
- **3D keypoints**: Anatomical landmarks (joints, body parts)
- **2D keypoints**: Projected positions on the image
- **Pose parameters**: Skeletal pose representation
- **Shape parameters**: Body shape variations
- **Camera parameters**: Perspective projection information

### XR Body (XRB) Model
Our parametric mesh representation that decouples:
- **Skeletal pose**: Joint rotations and positions
- **Body shape**: Morphological variations
- Provides better accuracy and interpretability than SMPL-based models

### Atlas70 Keypoint System
70 keypoints covering:
- Full body joints (shoulders, elbows, wrists, hips, knees, ankles)
- Face landmarks
- Hand keypoints (fingers, palm)
- Torso markers

Reference: `sam_3d_body/metadata/atlas70.py`

### Promptable Architecture
Inspired by SAM (Segment Anything Model), accepts auxiliary prompts:
- **2D keypoint prompts**: User-provided or detected keypoints
- **Mask prompts**: Segmentation masks for person isolation
- **Bounding box prompts**: Person detection boxes (always used)

### Model Variants
- **SAM3DBody**: Base model with standard processing
- **SAM3DBodyTriplet**: Three-stage variant with dedicated hand refinement
  - Stage 1: Full body estimation
  - Stage 2: Hand region cropping
  - Stage 3: Hand-specific refinement

### Model Inputs (Batch Format)
```python
batch = {
    'img': torch.Tensor,              # [B, N, 3, H, W] - input images
    'bbox': torch.Tensor,             # [B, N, 4] - bounding boxes
    'bbox_center': torch.Tensor,      # [B, N, 2] - bbox centers
    'bbox_scale': torch.Tensor,       # [B, N, 2] - bbox scales
    'mask': torch.Tensor,             # [B, N, 1, H, W] - person masks
    'mask_score': torch.Tensor,       # [B, N] - mask confidence
    'cam_int': torch.Tensor,          # [B, 3, 3] - camera intrinsics
    'person_valid': torch.Tensor,     # [B, N] - valid person flags
    # Additional for triplet model
    'lhand_img': torch.Tensor,        # Left hand crops
    'rhand_img': torch.Tensor,        # Right hand crops
}
```
Where `B` = batch size, `N` = max persons per image

### Model Outputs
```python
outputs = {
    'pred_vertices': np.ndarray,      # [N, V, 3] - 3D mesh vertices
    'pred_keypoints_3d': np.ndarray,  # [N, 70, 3] - 3D keypoints
    'pred_keypoints_2d': np.ndarray,  # [N, 70, 2] - 2D projections
    'pred_cam_t': np.ndarray,         # [N, 3] - camera translation
    'focal_length': np.ndarray,       # [N,] - focal lengths
    'global_rot': np.ndarray,         # [N, 3, 3] - global rotation
    'body_pose_params': np.ndarray,   # Body pose parameters
    'hand_pose_params': np.ndarray,   # Hand pose parameters
    'shape_params': np.ndarray,       # Shape parameters
    'scale_params': np.ndarray,       # Scale parameters
    'expr_params': np.ndarray,        # Facial expression params
}
```

## Development Guidelines

### Import Conventions

**Always use the sam_3d_body namespace:**
```python
from sam_3d_body import SAM3DBodyEstimator, build_sam_3d_body_model
from sam_3d_body.models.meta_arch import SAM3DBody
from sam_3d_body.visualization import Renderer
```

### Code Patterns

**1. Script Entry Points**
```python
import pyrootutils

# Setup project root and add to Python path
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
```

**2. Model Building**
```python
# Option 1: Direct instantiation
from sam_3d_body import SAM3DBodyEstimator

model = SAM3DBodyEstimator(
    checkpoint_path="path/to/checkpoint.ckpt",
    proto_path="path/to/assets",
    detector_path="path/to/detector",  # optional
    moge_path="path/to/moge.pt"       # optional
)

# Option 2: Builder function
from sam_3d_body import build_sam_3d_body_model

model = build_sam_3d_body_model(
    checkpoint_path="path/to/checkpoint.ckpt",
    proto_path="path/to/assets",
)

# Option 3: HuggingFace Hub
from sam_3d_body import build_sam_3d_body_hf

model = build_sam_3d_body_hf("facebook/sam-3d-body-large")
```

**3. Inference**
```python
# Process single image
outputs = model.process_one_image("path/to/image.jpg")

# Process with bounding boxes
outputs = model.process_one_image(
    img="path/to/image.jpg",
    bboxes=np.array([[x1, y1, x2, y2]])
)

# Outputs is a list of dicts, one per detected person
for person in outputs:
    vertices = person["pred_vertices"]
    keypoints_3d = person["pred_keypoints_3d"]
```

### Testing Workflow

**Before making changes:**
```bash
# 1. Install in editable mode
pip install -e .

# 2. Test imports
python -c "import sam_3d_body; print(sam_3d_body.__version__)"

# 3. Test builder API
python -c "from sam_3d_body import build_sam_3d_body_model, SAM3DBodyEstimator"

# 4. Run demo
python examples/demo.py --help
```

### Adding New Features

1. **Add code** to appropriate module in `sam_3d_body/`
2. **Update exports** in `sam_3d_body/__init__.py` if public API
3. **Add example** to `examples/` if user-facing
4. **Update docs** in README.md or examples/README.md
5. **Test** with `pip install -e .`

### Files to Handle Carefully

**Never modify without consultation:**
- `sam_3d_body/models/meta_arch/sam3d_body*.py` - Core model architecture
- `sam_3d_body/models/modules/atlas46.py` - Mesh representation
- Model checkpoint files (*.pt, *.ckpt)

**Modify with validation:**
- `sam_3d_body/__init__.py` - Keep exports minimal, validate imports
- `pyproject.toml` - Validate with: `python -c "import tomllib; tomllib.load(open('pyproject.toml', 'rb'))"`

**Safe to modify:**
- Examples, demos, visualization code
- Documentation files (README.md, INSTALL.md, etc.)
- Utility functions in `sam_3d_body/utils/`

## Package & Distribution

### PyPI Package Information
- **Package name**: `sam-3d-body` (with hyphens for PyPI)
- **Import name**: `sam_3d_body` (with underscores for Python)
- **Version**: 1.0.0 (defined in `sam_3d_body/__init__.py`)
- **License**: Apache 2.0

### Installation Methods

**From PyPI (when published):**
```bash
pip install sam-3d-body              # Basic
pip install sam-3d-body[vis]         # With visualization
pip install sam-3d-body[full]        # All optional features
```

**Local development:**
```bash
cd /path/to/sam-3d-body
pip install -e .                     # Editable install
pip install -e ".[vis]"              # With optional deps
```

**From source:**
```bash
git clone https://github.com/facebookresearch/sam-3d-body.git
cd sam-3d-body
pip install .
```

### Git Dependencies

These **cannot** be installed via PyPI and require manual installation:

```bash
# Required
pip install git+https://github.com/facebookresearch/momentum@77c3994
pip install git+https://github.com/facebookresearch/detectron2.git@a1ce2f9

# Optional
pip install git+https://github.com/facebookresearch/pytorch3d.git@75ebeea
pip install git+https://github.com/microsoft/MoGe.git
pip install flash-attn==2.7.3  # Requires CUDA and proper compiler setup
```

Or use the convenience script:
```bash
bash install_git_deps.sh
```

### Building & Publishing

**Build the package:**
```bash
# Clean previous builds
rm -rf build/ dist/ *.egg-info

# Build wheel and source distribution
python -m build
```

**Test locally:**
```bash
pip install dist/sam_3d_body-1.0.0-py3-none-any.whl
```

**Publish to PyPI:**
See `PUBLISHING.md` for complete guide.

## Common Tasks

### Running Demo Script
```bash
python examples/demo.py \
    --image_folder /path/to/images \
    --checkpoint_path /path/to/model.ckpt \
    --proto_path /path/to/assets \
    --detector_path /path/to/detector \
    --moge_path /path/to/moge.pt
```

With environment variables:
```bash
export SAM3D_PROTO_PATH=/path/to/assets
export SAM3D_DETECTOR_PATH=/path/to/detector
export SAM3D_MOGE_PATH=/path/to/moge.pt

python examples/demo.py \
    --image_folder /path/to/images \
    --checkpoint_path /path/to/model.ckpt
```

### Debugging Common Issues

**Import errors:**
```bash
# Check if package is installed
pip list | grep sam-3d-body

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"

# Verify dependencies
pip check
```

**Model loading errors:**
- Verify checkpoint path exists and is accessible
- Check proto_path points to assets directory
- Ensure PyMomentum is installed (git dependency)

**Detector errors:**
- Ensure Detectron2 is installed: `pip list | grep detectron2`
- Check detector_path contains required files
- Verify CUDA is available if using GPU

**FOV estimation errors:**
- Check if MoGe is installed (optional): `pip list | grep moge`
- Verify moge_path points to model checkpoint
- Model works without MoGe (falls back to default FOV)

## Important Constraints

### System Requirements
- **Python**: 3.10, 3.11, or 3.12
- **CUDA**: 11.8 (for GPU training/inference)
- **OS**: Linux (primary), macOS (limited support)

### Large Files Not in Repo
- Model checkpoints (several GB each)
- Atlas model assets
- Detector model files
- Test images and datasets

Download separately from HuggingFace or use `build_sam_3d_body_hf()`.

### Configuration Paths
All paths should be configurable (no hardcoded paths):
- Accept via command-line arguments
- Fall back to environment variables
- Provide clear error messages when missing

### Do Not Commit
- Model weights (*.pt, *.ckpt, *.pth)
- Generated outputs (images, videos)
- Log files
- Virtual environments
- IDE settings (except shared configs)

## Quick Reference

### Key Commands
```bash
# Install package
pip install -e .

# Test imports
python -c "import sam_3d_body; print(sam_3d_body.__version__)"

# Run demo
python examples/demo.py --image_folder images/ --checkpoint_path model.ckpt

# Build package
python -m build

# Validate config
python -c "import tomllib; tomllib.load(open('pyproject.toml', 'rb'))"
```

### Important Paths
- **Main package**: `sam_3d_body/`
- **Model code**: `sam_3d_body/models/meta_arch/sam3d_body.py`
- **Inference interface**: `sam_3d_body/estimator.py`
- **Examples**: `examples/demo.py`
- **Package config**: `pyproject.toml`

### Documentation Links
- **Installation**: `INSTALL.md`
- **Publishing**: `PUBLISHING.md`
- **Examples**: `examples/README.md`
- **Main docs**: `README.md`
- **License**: `LICENSE`

### Useful Patterns
```python
# Get package version
import sam_3d_body
print(sam_3d_body.__version__)

# List all exports
from sam_3d_body import __all__
print(__all__)

# Check if dependencies installed
import importlib
has_pytorch3d = importlib.util.find_spec("pytorch3d") is not None
```

---

*This guide was last updated: October 2024*
*For questions or updates, see repository maintainers.*
