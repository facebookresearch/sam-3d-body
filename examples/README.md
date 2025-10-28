# SAM 3D Body Examples

This directory contains example scripts demonstrating how to use SAM 3D Body.

## Demo Script

The `demo.py` script processes a folder of images and generates 3D human mesh reconstructions.

### Requirements

Before running the examples, make sure you have installed SAM 3D Body and its dependencies:

```bash
# Install from PyPI
pip install sam-3d-body

# Or install from source
pip install -e .

# Install optional dependencies for visualization
pip install sam-3d-body[vis]

# Install git-based dependencies (see ../INSTALL.md for details)
bash ../install_git_deps.sh
```

### Download Model Checkpoints

Download the SAM 3D Body model checkpoints from HuggingFace:

```bash
# Using the builder API in Python
from sam_3d_body import build_sam_3d_body_hf
model = build_sam_3d_body_hf("facebook/sam-3d-body-large")
```

Or download manually from the [HuggingFace repository](https://huggingface.co/facebook/sam-3d-body).

### Running the Demo

Basic usage:

```bash
python demo.py \
    --image_folder /path/to/images \
    --checkpoint_path /path/to/model.ckpt \
    --proto_path /path/to/assets \
    --detector_path /path/to/detector \
    --moge_path /path/to/moge/model.pt
```

Using environment variables:

```bash
export SAM3D_PROTO_PATH=/path/to/assets
export SAM3D_DETECTOR_PATH=/path/to/detector
export SAM3D_MOGE_PATH=/path/to/moge/model.pt

python demo.py \
    --image_folder /path/to/images \
    --checkpoint_path /path/to/model.ckpt
```

### Command-line Arguments

- `--image_folder` (required): Path to folder containing input images
- `--checkpoint_path` (required): Path to SAM 3D Body model checkpoint
- `--output_folder` (optional): Path to output folder (default: `./output/<image_folder_name>`)
- `--detector_path` (optional): Path to detector model folder
- `--proto_path` (optional): Path to proto/assets folder
- `--moge_path` (optional): Path to MoGe model checkpoint for FOV estimation
- `--bbox_thresh` (optional): Bounding box detection threshold (default: 0.8)
- `--use_mask` (optional): Enable SAM2 mask prediction

### Output

The demo generates visualizations for each detected person in the input images:
- Original image
- 2D keypoint overlay
- 3D mesh overlay on image
- 3D mesh side view

All outputs are saved to the specified output folder.

## Programmatic Usage

You can also use SAM 3D Body programmatically:

```python
from sam_3d_body import build_sam_3d_body_model, SAM3DBodyEstimator

# Method 1: Using the builder function
model = build_sam_3d_body_model(
    checkpoint_path="/path/to/checkpoint.ckpt",
    proto_path="/path/to/assets",
    detector_path="/path/to/detector",
    moge_path="/path/to/moge.pt"
)

# Method 2: Direct instantiation
estimator = SAM3DBodyEstimator(
    checkpoint_path="/path/to/checkpoint.ckpt",
    proto_path="/path/to/assets"
)

# Process an image
outputs = estimator.process_one_image("/path/to/image.jpg")

# Access results
for person_output in outputs:
    vertices = person_output["pred_vertices"]
    keypoints_3d = person_output["pred_keypoints_3d"]
    keypoints_2d = person_output["pred_keypoints_2d"]
    # ... use the outputs
```

## Additional Examples

More examples coming soon!
