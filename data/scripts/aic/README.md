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
