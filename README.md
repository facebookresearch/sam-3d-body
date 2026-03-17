# SAM 3D

SAM 3D Body is one part of SAM 3D, a pair of models for object and human mesh reconstruction. If you’re looking for SAM 3D Objects, [click here](https://github.com/facebookresearch/sam-3d-objects).

# SAM 3D Body: Robust Full-Body Human Mesh Recovery

<p align="left">
<a href="https://arxiv.org/abs/2602.15989"><img src="https://img.shields.io/badge/arXiv-2602.15989-b31b1b.svg" alt="arXiv"></a>
<a href="https://ai.meta.com/research/publications/sam-3d-body-robust-full-body-human-mesh-recovery/"><img src='https://img.shields.io/badge/Meta_AI-Paper-4A90E2?logo=meta&logoColor=white' alt='Paper'></a>
<a href="https://ai.meta.com/blog/sam-3d/"><img src='https://img.shields.io/badge/Project_Page-Blog-9B72F0?logo=googledocs&logoColor=white' alt='Blog'></a>
<a href="https://huggingface.co/datasets/facebook/sam-3d-body-dataset"><img src='https://img.shields.io/badge/🤗_Hugging_Face-Dataset-F59500?logoColor=white' alt='Dataset'></a>
<a href="https://www.aidemos.meta.com/segment-anything/editor/convert-body-to-3d"><img src='https://img.shields.io/badge/🤸_Playground-Live_Demo-E85D5D?logoColor=white' alt='Live Demo'></a>
</p>

[Xitong Yang](https://scholar.google.com/citations?user=k0qC-7AAAAAJ&hl=en)\*, [Devansh Kukreja](https://www.linkedin.com/in/devanshkukreja)\*, [Don Pinkus](https://www.linkedin.com/in/don-pinkus-9140702a)\*, [Anushka Sagar](https://www.linkedin.com/in/anushkasagar), [Taosha Fan](https://scholar.google.com/citations?user=3PJeg1wAAAAJ&hl=en), [Jinhyung Park](https://jindapark.github.io/)⚬, [Soyong Shin](https://yohanshin.github.io/)⚬, [Jinkun Cao](https://www.jinkuncao.com/), [Jiawei Liu](https://jia-wei-liu.github.io/), [Nicolas Ugrinovic](https://www.iri.upc.edu/people/nugrinovic/), [Matt Feiszli](https://scholar.google.com/citations?user=A-wA73gAAAAJ&hl=en&oi=ao)†, [Jitendra Malik](https://people.eecs.berkeley.edu/~malik/)†, [Piotr Dollar](https://pdollar.github.io/)†, [Kris Kitani](https://kriskitani.github.io/)†

***Meta Superintelligence Labs***

*Core Contributor,  ⚬Intern, †Project Lead

![SAM 3D Body Model Architecture](assets/model_diagram.png?raw=true)

**SAM 3D Body (3DB)** is a promptable model for single-image full-body 3D human mesh recovery (HMR). Our method demonstrates state-of-the-art performance, with strong generalization and consistent accuracy in diverse in-the-wild conditions. 3DB estimates the human pose of the body, feet, and hands based on the [Momentum Human Rig](https://github.com/facebookresearch/MHR) (MHR), a new parametric mesh representation that decouples skeletal structure and surface shape for improved accuracy and interpretability.

3DB employs an encoder-decoder architecture and supports auxiliary prompts, including 2D keypoints and masks, enabling user-guided inference similar to the SAM family of models. Our model is trained on high-quality annotations from a multi-stage annotation pipeline using differentiable optimization, multi-view geometry, dense keypoint detection, and a data engine to collect and annotated data covering both common and rare poses across a wide range of viewpoints.

## Qualitative Results

<table>
<thead>
<tr>
<th align="center">Input</th>
<th align="center"><strong>SAM 3D Body</strong></th>
<th align="center">CameraHMR</th>
<th align="center">NLF</th>
<th align="center">HMR2.0b</th>
</tr>
</thead>
<tbody>
<tr>
<td align="center"><img src="assets/qualitative_comparisons/sample1/input_bbox.png" alt="Sample 1 Input" width="160"></td>
<td align="center"><img src="assets/qualitative_comparisons/sample1/SAM 3D Body.png" alt="Sample 1 - SAM 3D Body" width="160"></td>
<td align="center"><img src="assets/qualitative_comparisons/sample1/camerahmr.png" alt="Sample 1 - CameraHMR" width="160"></td>
<td align="center"><img src="assets/qualitative_comparisons/sample1/nlf.png" alt="Sample 1 - NLF" width="160"></td>
<td align="center"><img src="assets/qualitative_comparisons/sample1/4dhumans.png" alt="Sample 1 - 4DHumans (HMR2.0b)" width="160"></td>
</tr>
<tr>
<td align="center"><img src="assets/qualitative_comparisons/sample2/input_bbox.png" alt="Sample 2 Input" width="160"></td>
<td align="center"><img src="assets/qualitative_comparisons/sample2/SAM 3D Body.png" alt="Sample 2 - SAM 3D Body" width="160"></td>
<td align="center"><img src="assets/qualitative_comparisons/sample2/camerahmr.png" alt="Sample 2 - CameraHMR" width="160"></td>
<td align="center"><img src="assets/qualitative_comparisons/sample2/nlf.png" alt="Sample 2 - NLF" width="160"></td>
<td align="center"><img src="assets/qualitative_comparisons/sample2/4dhumans.png" alt="Sample 2 - 4DHumans (HMR2.0b)" width="160"></td>
</tr>
<tr>
<td align="center"><img src="assets/qualitative_comparisons/sample3/input_bbox.png" alt="Sample 3 Input" width="160"></td>
<td align="center"><img src="assets/qualitative_comparisons/sample3/SAM 3D Body.png" alt="Sample 3 - SAM 3D Body" width="160"></td>
<td align="center"><img src="assets/qualitative_comparisons/sample3/camerahmr.png" alt="Sample 3 - CameraHMR" width="160"></td>
<td align="center"><img src="assets/qualitative_comparisons/sample3/nlf.png" alt="Sample 3 - NLF" width="160"></td>
<td align="center"><img src="assets/qualitative_comparisons/sample3/4dhumans.png" alt="Sample 3 - 4DHumans (HMR2.0b)" width="160"></td>
</tr>
<tr>
<td align="center"><img src="assets/qualitative_comparisons/sample4/input_bbox.png" alt="Sample 4 Input" width="160"></td>
<td align="center"><img src="assets/qualitative_comparisons/sample4/SAM 3D Body.png" alt="Sample 4 - SAM 3D Body" width="160"></td>
<td align="center"><img src="assets/qualitative_comparisons/sample4/camerahmr.png" alt="Sample 4 - CameraHMR" width="160"></td>
<td align="center"><img src="assets/qualitative_comparisons/sample4/nlf.png" alt="Sample 4 - NLF" width="160"></td>
<td align="center"><img src="assets/qualitative_comparisons/sample4/4dhumans.png" alt="Sample 4 - 4DHumans (HMR2.0b)" width="160"></td>
</tr>
</tbody>
</table>

*Our SAM 3D Body demonstrates superior reconstruction quality with more accurate pose estimation, better shape recovery, and improved handling of occlusions and challenging viewpoints compared to existing approaches.*

## Latest updates

**11/19/2025** -- Checkpoints Launched, Dataset Released, Web Demo and Paper are out!

## Installation
See [INSTALL.md](INSTALL.md) for instructions for python environment setup and model checkpoint access.

## Getting Started

3DB can reconstruct 3D full-body human mesh from a single image, optionally with keypoint/mask prompts and/or hand refinement from the hand decoder.

For a quick start, run our demo script for model inference and visualization with models from [Hugging Face](https://huggingface.co/facebook) (please make sure to follow [INSTALL.md](INSTALL.md) to request access to our checkpoints.).

```bash
# Download assets from HuggingFace
hf download facebook/sam-3d-body-dinov3 --local-dir checkpoints/sam-3d-body-dinov3

# Run demo script with default ViTdet detector and MoGe2 FOV model
python demo.py \
    --image_folder <path_to_images> \
    --output_folder <path_to_output> \
    --checkpoint_path ./checkpoints/sam-3d-body-dinov3/model.ckpt \
    --mhr_path ./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt

# To use SAM3 as the detector to align with online playground of SAM3D
python demo.py \
    --image_folder <path_to_images> \
    --output_folder <path_to_output> \
    --checkpoint_path ./checkpoints/sam-3d-body-dinov3/model.ckpt \
    --mhr_path ./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt \
    --detector_name sam3
```

You can also try the following lines of code with models loaded directly from [Hugging Face](https://huggingface.co/facebook)

```python
import cv2
import numpy as np
from notebook.utils import setup_sam_3d_body
from tools.vis_utils import visualize_sample_together

# Set up the estimator
estimator = setup_sam_3d_body(hf_repo_id="facebook/sam-3d-body-dinov3")

# Load and process image
img_bgr = cv2.imread("path/to/image.jpg")
outputs = estimator.process_one_image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

# Visualize and save results
rend_img = visualize_sample_together(img_bgr, outputs, estimator.faces)
cv2.imwrite("output.jpg", rend_img.astype(np.uint8))
```

For a complete demo with visualization, see [notebook/demo_human.ipynb](notebook/demo_human.ipynb).


## Model Description

### SAM 3D Body checkpoints

The table below shows the performance of SAM 3D Body checkpoints released on 11/19/2025.

|      **Backbone (size)**       | **3DPW (MPJPE)** |    **EMDB (MPJPE)**     | **RICH (PVE)** | **COCO (PCK@.05)** |  **LSPET (PCK@.05)** | **Freihand (PA-MPJPE)**
| :------------------: | :----------: | :--------------------: | :-----------------: | :----------------: | :----------------: | :----------------: |
|  DINOv3-H+ (840M) <br /> ([config](https://huggingface.co/facebook/sam-3d-body-dinov3/blob/main/model_config.yaml), [checkpoint](https://huggingface.co/facebook/sam-3d-body-dinov3/blob/main/model.ckpt))   |      54.8      |          61.7         |       60.3        |       86.5        | 68.0 | 5.5
|   ViT-H  (631M) <br /> ([config](https://huggingface.co/facebook/sam-3d-body-vith/blob/main/model_config.yaml), [checkpoint](https://huggingface.co/facebook/sam-3d-body-vith/blob/main/model.ckpt))    |     54.8   |         62.9         |       61.7        |        86.8       | 68.9 |  5.5


## SAM 3D Body Dataset
The SAM 3D Body data is released on [Hugging Face](https://huggingface.co/datasets/facebook/sam-3d-body-dataset). Please follow the [instructions](./data/README.md) to download and process the data.

## SAM 3D Objects

[SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects) is a foundation model that reconstructs full 3D shape geometry, texture, and layout from a single image.

As a way to combine the strengths of both **SAM 3D Objects** and **SAM 3D Body**, we provide an example notebook that demonstrates how to combine the results of both models such that they are aligned in the same frame of reference. Check it out [here](https://github.com/facebookresearch/sam-3d-objects/blob/main/notebook/demo_3db_mesh_alignment.ipynb).

## License

The SAM 3D Body model checkpoints and code are licensed under [SAM License](./LICENSE).

## Contributing

See [contributing](CONTRIBUTING.md) and the [code of conduct](CODE_OF_CONDUCT.md).

## Contributors

The SAM 3D Body project was made possible with the help of many contributors:
Vivian Lee, George Orlin, Nikhila Ravi, Andrew Westbury, Jyun-Ting Song, Zejia Weng, Xizi Zhang, Yuting Ye, Federica Bogo, Ronald Mallet, Ahmed Osman, Rawal Khirodkar, Javier Romero, Carsten Stoll, Jean-Charles Bazin, Sofien Bouaziz, Yuan Dong, Su Zhaoen, Fabian Prada, Alexander Richard, Michael Zollhoefer, Roman Rädle, Sasha Mitts, Michelle Chan, Yael Yungster, Azita Shokrpour, Helen Klein, Mallika Malhotra, Ida Cheng, Eva Galper.

## Citing SAM 3D Body

If you use SAM 3D Body or the SAM 3D Body dataset in your research, please use the following BibTeX entry.

```bibtex
@article{yang2026sam3dbody,
  title={SAM 3D Body: Robust Full-Body Human Mesh Recovery},
  author={Yang, Xitong and Kukreja, Devansh and Pinkus, Don and Sagar, Anushka and Fan, Taosha and Park, Jinhyung and Shin, Soyong and Cao, Jinkun and Liu, Jiawei and Ugrinovic, Nicolas and Feiszli, Matt and Malik, Jitendra and Dollar, Piotr and Kitani, Kris},
  journal={arXiv preprint arXiv:2602.15989},
  year={2026}
}
```
