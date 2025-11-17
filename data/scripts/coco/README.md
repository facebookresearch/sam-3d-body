# COCO Dataset Preparation

Follow the steps below to prepare the **[COCO Dataset](https://cocodataset.org/#home)** for **[SAM-3D-Body Data](https://huggingface.co/datasets/facebook/sam-3d-body-dataset)**.

---

- Set the following environment variables to simplify directory references:

    ```bash
    export COCO_IMG_DIR=/path/to/coco/images
    ```

- Download and extract the `COCO2014` dataset.

    ```bash
    cd $COCO_IMG_DIR
    wget http://images.cocodataset.org/zips/train2014.zip
    wget http://images.cocodataset.org/zips/val2014.zip
    wget http://images.cocodataset.org/zips/test2014.zip
    unzip train2014.zip 
    unzip val2014.zip 
    unzip test2014.zip 
    rm train2014.zip val2014.zip test2014.zip 
    ```

- `$COCO_IMG_DIR` should follow the directory structure below.

    ```plaintext
    $COCO_IMG_DIR
    ├── test2014
    ├── train2014
    └── val2014
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
