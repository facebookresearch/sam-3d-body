This repository provides setup instructions and utilities for preparing and processing datasets used in SAM-3D-Body, including downloading annotations and creating WebDataset archives.

## 🧩 Environment Setup

Create and configure the conda environment for dataset preparation:

```bash
conda create --name sam_3d_body_data python=3.9 -y
conda activate sam_3d_body_data
conda install pytorch=2.4.0 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install hdbscan pyntcloud==0.3.1 pykalman torchgeometry colour==0.1.5 flask==3.1.1 trimesh==4.7.3
pip install pandas==2.0.3 numpy==1.26 pycolmap==0.3.0 yacs==0.1.8 projectaria-tools==1.3.3 opencv-python==4.7.0.72
pip install datasets huggingface_hub webdataset pycocotools
```

## 📂 Image Preparation

Prepare dataset images required for SAM-3D-Body by following the individual setup guides linked below:

- [3DPW](scripts/3dpw/README.md)
- [AI Challenger](scripts/aic/README.md)
- [COCO](scripts/coco/README.md)
- [EgoExo4D](scripts/egoexo4d/README.md)
- [EgoHumans](scripts/egohumans/README.md)
- [Harmony4D](scripts/harmony4d/README.md)
- [MPII](scripts/mpii/README.md)
- [SA1B](scripts/sa1b/README.md)

Each dataset guide describes how to download, extract, and structure the raw data for WebDataset creation.

## ⚙️ Environment Variables

Set up environment variables for annotation download and WebDataset output directories:

```bash
  export SAM3D_BODY_ANN_DIR=/path/to/sam3d/body/annotations
  export SAM3D_BODY_WDS_DIR=/path/to/sam3d/body/webdatasets
```

## ⬇️ Download Annotations

You can download all annotation splits or only a specific split from SAM-3D-Body.

### Download all splits

  ```bash
    python scripts/download.py \
      --save_dir $SAM3D_BODY_ANN_DIR
  ```

### Download a specific split

  Replace `DATA_SPLIT` with the desired split name (e.g., `coco_train`, `sa1b_train`, `harmony4d_test`, etc.):

  ```bash
    python scripts/download.py \
      --save_dir $SAM3D_BODY_ANN_DIR \
      --splits DATA_SPLIT 
  ```

## 🗜️ Create WebDatasets

Convert a dataset split into WebDataset format for efficient distributed training and evaluation.

```bash
python scripts/create_webdataset.py \                              
    --annotation_dir $SAM3D_BODY_ANN_DIR/DATA_SPLIT \
    --webdataset_dir $SAM3D_BODY_WDS_DIR/DATA_SPLIT \
    --image_dir DATASET_IMG_DIR 
```

where

- `--annotation_dir`: Path to the annotation directory for the target split (e.g., `$SAM3D_BODY_ANN_DIR/coco_train`).
- `--webdataset_dir`: Output directory for the generated WebDataset files (e.g., `$SAM3D_BODY_WDS_DIR/coco_train`).
- `--image_dir`: Path to the corresponding image directory for the dataset (e.g., `$COCO_IMG_DIR`).
