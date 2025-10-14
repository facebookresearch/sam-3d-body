# SAM 3D: Segment Anything 

**[AI at Meta, FAIR](https://ai.meta.com/research/)**

Xitong Yang*, Devansh Kukreja*, Don Pinkus*, Taosha Fan, David Park, Soyong Shin, Jinkun Cao, Jiawei Liu, Nicolas Ugrinovic, Anushka Sagar†, Jitendra Malik†, Piotr Dollar†, Kris Kitani†

*Core contributors, †Project leads

[[`<REPLACE ME Paper>`](https://ai.meta.com/research/publications/sam-2-segment-anything-in-images-and-videos/)] [[`<REPLACE ME Project>`](https://ai.meta.com/sam2)] [[`<REPLACE ME Demo>`](https://sam2.metademolab.com/)] [[`<REPLACE ME Dataset>`](https://ai.meta.com/datasets/segment-anything-video)] [[`<REPLACE ME Blog>`](https://ai.meta.com/blog/segment-anything-2)] [[`<REPLACE ME BibTeX>`](#citing-sam-2)]

![SAM 3D Body Model Architecture](assets/model_diagram.png?raw=true)

**SAM 3D Body** is a foundation model for estimating 3D human pose and shape from single images. We address key limitations in existing approaches by focusing on data quality and diversity. Unlike previous methods that rely on noise-prone pseudo-ground-truth annotations, SAM 3D Body leverages high-quality supervision from multi-view capture systems, synthetic data, and a scalable data engine for mining challenging scenarios. 

The model employs an encoder-decoder architecture to regress parametric body model parameters, with explicit separation of pose and shape for better interpretability. Following the promptable design paradigm of SAM, inference can be guided by lightweight prompts such as 2D keypoints or masks. The model estimates camera intrinsics to handle perspective distortion, making it robust for close-range images with diverse viewpoints, poses, and clothing.

## Latest updates

**10/20/2025 -- Checkpoints Launched, Dataset Released, Web Demo and Paper are out**
- < MORE DETAILS HERE >

## Installation

< INSTALLATION INSTRUCTIONS HERE >

## Getting Started

### Download Checkpoints

First, we need to download a model checkpoint. All the model checkpoints can be downloaded by running:

```bash
cd checkpoints && \
./download_ckpts.sh && \
cd ..
```

or individually from:

- [sam3d_hiera_tiny.pt](https://huggingface.co/facebook/sam3d-hiera-tiny) (placeholder link - to be updated)
- [sam3d_hiera_small.pt](https://huggingface.co/facebook/sam3d-hiera-small) (placeholder link - to be updated)  
- [sam3d_hiera_base_plus.pt](https://huggingface.co/facebook/sam3d-hiera-base-plus) (placeholder link - to be updated)
- [sam3d_hiera_large.pt](https://huggingface.co/facebook/sam3d-hiera-large) (placeholder link - to be updated)

Then SAM 3D can be used in a few lines as follows for image prediction.

### Image prediction

< MODELS DESCRIPTION HERE >

< SIMPLE CODE TO RUN INFERENCE HERE >

< Link to Colab Notebook >

### Video prediction

< Optional: Do we want to add tooling to run inference on videos? >

< Link to Colab Notebook >

## Load from 🤗 Hugging Face

Alternatively, models can also be loaded from [Hugging Face](https://huggingface.co/models?search=facebook/sam3d) (requires `pip install huggingface_hub`).

For image prediction:

```python
import torch
from sam3d.sam3d_image_predictor import SAM3DImagePredictor

predictor = SAM3DImagePredictor.from_pretrained("facebook/sam3d-hiera-large")

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(<your_image>)
    poses, meshes = predictor.predict(<input_prompts>)
```

## Model Description

### SAM 3D checkpoints

The table below shows the SAM 3D checkpoints released on September 23, 2025.

|      **Model**       | **Size (M)** |    **Speed (FPS)**     | **H36M test (MPJPE)** | **3DPW test (MPJPE)** |
| :------------------: | :----------: | :--------------------: | :-----------------: | :----------------: |
|   sam3d_hiera_tiny <br /> ([config](configs/sam3d/sam3d_hiera_t.yaml), [checkpoint](https://huggingface.co/facebook/sam3d-hiera-tiny))    |     TBD     |          TBD          |        TBD         |        TBD        |
|   sam3d_hiera_small <br /> ([config](configs/sam3d/sam3d_hiera_s.yaml), [checkpoint](https://huggingface.co/facebook/sam3d-hiera-small))   |      TBD      |          TBD          |        TBD         |        TBD        |
| sam3d_hiera_base_plus <br /> ([config](configs/sam3d/sam3d_hiera_b+.yaml), [checkpoint](https://huggingface.co/facebook/sam3d-hiera-base-plus)) |     TBD     |        TBD          |        TBD         |        TBD        |
|   sam3d_hiera_large <br /> ([config](configs/sam3d/sam3d_hiera_l.yaml), [checkpoint](https://huggingface.co/facebook/sam3d-hiera-large))   |    TBD     |          TBD          |        TBD         |        TBD        |


< TODO: Update when we run speedtests >
Speed measured on an A100 with `torch 2.5.1, cuda 12.4`. See `benchmark.py` for an example on benchmarking (compiling all the model components). Compiling only the image encoder can be more flexible and also provide (a smaller) speed-up (set `compile_image_encoder: True` in the config).

## Segment Anything 3D Dataset

< Info on the 3D annotations we're releasing >

## Training SAM 3D

You can train or fine-tune SAM 3D on custom datasets of images, videos, or both.

< Link to training README >

## Web demo for SAM 3D

< Link to Web Demo >

## License

The SAM 3D Body model checkpoints and code are licensed under [Apache 2.0](./LICENSE).

## Contributing

See [contributing](CONTRIBUTING.md) and the [code of conduct](CODE_OF_CONDUCT.md).

## Contributors

The SAM 3D Body project was made possible with the help of many contributors (alphabetical):

Third-party code: we acknowledge the use of open-source libraries and frameworks that made this work possible.

## Citing SAM 3D Body

If you use SAM 3D Body or the SAM 3D Body dataset in your research, please use the following BibTeX entry.

```bibtex
@article{yang2025sam3dbody,
  title={SAM 3D Body: Single Image Human Mesh Recovery},
  author={Yang, Xitong and Kukreja, Devansh and Pinkus, Don and Fan, Taosha and Park, David and Shin, Soyong and Cao, Jinkun and Liu, Jiawei and Ugrinovic, Nicolas and Sagar, Anushka and Malik, Jitendra and Dollar, Piotr and Kitani, Kris},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```
