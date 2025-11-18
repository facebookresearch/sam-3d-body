# 3DPW Dataset Preparation

Follow the steps below to prepare the **[3DPW Dataset](https://virtualhumans.mpi-inf.mpg.de/3DPW/evaluation.html)** for **[SAM-3D-Body Data](https://huggingface.co/datasets/facebook/sam-3d-body-dataset)**.

---

- Set the following environment variables to simplify directory references:

    ```bash
    export ThreeDPW_IMG_DIR=/path/to/3dpw/images
    ```

- Download `imageFiles.zip` from 🔗 [3DPW Dataset](https://virtualhumans.mpi-inf.mpg.de/3DPW/license.html) and extract the images to `$ThreeDPW_IMG_DIR`.

- `$ThreeDPW_IMG_DIR` should follow the directory structure below.

    ```plaintext
    $ThreeDPW_IMG_DIR
    ├── courtyard_arguing_00
    ├── courtyard_backpack_00
    ├── ...
    └── outdoors_slalom_01
    ```

<!-- - Download 🔗 [SAM-3D-Body Data](https://huggingface.co/datasets/facebook/sam-3d-body-dataset) to `$SAM3D_BODY_ANN_DIR`.

    ```bash
    python scripts/download.py \
        --save_dir $SAM3D_BODY_ANN_DIR \
        --splits 3dpw_train
    ```

- Create WebDataset shards with the following command:

    ```bash
    python scripts/create_webdataset.py \
        --annotation_dir $SAM3D_BODY_ANN_DIR/3dpw_train \
        --image_dir $ThreeDPW_IMG_DIR \
        --webdataset_dir $SAM3D_BODY_WDS_DIR/3dpw_train
    ``` -->
