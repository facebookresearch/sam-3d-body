# AI Challenger Dataset Preparation

Follow the steps below to prepare the **[AI Challenger Keypoint Dataset](https://github.com/AIChallenger/AI_Challenger_2017)** for **[SAM-3D-Body Data](https://huggingface.co/datasets/facebook/sam-3d-body-dataset)**.

---

- Set the following environment variables to simplify directory references:

    ```bash
    export AIC_IMG_DIR=/path/to/ai/challenger/images
    ```

- Download the `AI Challenger Keypoint` images to `$AIC_IMG_DIR`.

- `$AIC_IMG_DIR` should follow the directory structure below.

    ```plaintext
    $AIC_IMG_DIR
    ├── test
    │   └── images
    └── train
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
