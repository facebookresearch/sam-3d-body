# MPII Dataset Preparation

Follow the steps below to prepare the **[MPII Dataset](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/software-and-datasets/mpii-human-pose-dataset)** for **[SAM-3D-Body Data](https://huggingface.co/datasets/facebook/sam-3d-body-dataset)**.

---

- Set the following environment variables to simplify directory references:

    ```bash
    export COCO_IMG_DIR=/path/to/coco/images
    ```

- Download and extract the `COCO2014` dataset.

    ```bash
    wget https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz
    tar zxvf mpii_human_pose_v1.tar.gz -C $MPII_IMG_DIR 
    rm mpii_human_pose_v1.tar.gz 
    ```

- `$COCO_IMG_DIR` should follow the directory structure below.

    ```plaintext
    $MPII_IMG_DIR
    └── images
    ```

<!-- - Download 🔗 [SAM-3D-Body Data](https://huggingface.co/datasets/facebook/sam-3d-body-dataset) to `$SAM3D_BODY_ANN_DIR`.

    ```bash
    python scripts/download.py \
        --save_dir $SAM3D_BODY_ANN_DIR \
        --splits coco_train
    ```

- Create WebDataset shards with the following command:

    ```bash
    python scripts/create_webdataset.py \
        --annotation_dir $SAM3D_BODY_ANN_DIR/coco_train \
        --image_dir $COCO_IMG_DIR \
        --webdataset_dir $SAM3D_BODY_WDS_DIR/coco_train
    ``` -->
