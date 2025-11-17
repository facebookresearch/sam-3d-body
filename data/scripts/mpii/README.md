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
