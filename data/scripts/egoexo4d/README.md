# COCO Dataset Preparation

Follow the steps below to prepare the **[EgoExo4D Dataset](https://ego-exo4d-data.org)** for **[SAM-3D-Body Data](https://huggingface.co/datasets/facebook/sam-3d-body-dataset)**.

---

- [Sign the Ego-Exo4D Licenses](https://ego4ddataset.com/egoexo-license/), this will take 2 days to be approved.

- Setup the CLI downloader. [Click here for docs](https://docs.ego-exo4d-data.org/download/).

- Set the following environment variables to simplify directory references:

    ```bash
    export EE4D_IMG_DIR=/path/to/egoexo4d/images
    ```

- Download `EgoExo4D` undistorted images.

    ```bash
    egoexo -o ./$EE4D_IMG_DIR --parts sam_3d_body
    ```

- `$EE4D_IMG_DIR` should follow the directory structure below.

    ```plaintext
    $EE4D_IMG_DIR
    ├── cmu_bike01_2
    ├── cmu_bike01_7
    └── ...
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
