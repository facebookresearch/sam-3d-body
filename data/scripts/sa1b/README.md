# SA1B Dataset Preparation

Follow the steps below to prepare the **[SA1B Dataset](https://ai.meta.com/datasets/segment-anything/)** for **[SAM-3D-Body Data](https://huggingface.co/datasets/facebook/sam-3d-body-dataset)**.

---

- Set the following environment variables to simplify directory references:

    ```bash
    export SA1B_IMG_DIR=/path/to/sa1b/images
    ```

- Download and extract the `SA1B` dataset.

- `$SA1B_IMG_DIR` should follow the directory structure below.

    ```plaintext
    $SA1B_IMG_DIR
    ├── sa_xxxxxxx.jpg
    └── .......
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
