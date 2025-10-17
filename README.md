# SAM 3D Body: Single Image Human Mesh Recovery 

**[AI at Meta, FAIR](https://ai.meta.com/research/)**

[Xitong Yang](https://scholar.google.com/citations?user=k0qC-7AAAAAJ&hl=en)\*, [Devansh Kukreja](https://www.linkedin.com/in/devanshkukreja)\*, [Don Pinkus](https://www.linkedin.com/in/don-pinkus-9140702a)\*, [Taosha Fan](https://scholar.google.com/citations?user=3PJeg1wAAAAJ&hl=en), [David Park](https://jindapark.github.io/), [Soyong Shin](https://yohanshin.github.io/), [Jinkun Cao](https://www.jinkuncao.com/), [Jiawei Liu](https://jia-wei-liu.github.io/), [Nicolas Ugrinovic](https://www.iri.upc.edu/people/nugrinovic/), [Anushka Sagar](https://www.linkedin.com/in/anushkasagar)†, [Jitendra Malik](https://people.eecs.berkeley.edu/~malik/)†, [Piotr Dollar](https://pdollar.github.io/)†, [Kris Kitani](https://kriskitani.github.io/)†

*Core contributors, †Project leads

[[`<REPLACE ME Paper>`](https://ai.meta.com/research/publications/sam-2-segment-anything-in-images-and-videos/)] [[`<REPLACE ME Project>`](https://ai.meta.com/sam2)] [[`<REPLACE ME Demo>`](https://sam2.metademolab.com/)] [[`<Dataset>`](https://huggingface.co/datasets/facebook/sam-3d-body-dataset)] [[`<REPLACE ME Blog>`](https://ai.meta.com/blog/segment-anything-2)] [[`<REPLACE ME BibTeX>`](#citing-sam-2)]

![SAM 3D Body Model Architecture](assets/model_diagram.png?raw=true)

**SAM 3D Body (3DB)** is a promptable foundation model for single-image 3D human mesh recovery (HMR). Our method emphasizes data quality and diversity to maximize performance, addressing key problems with noisy pseudo-ground-truth meshes commonly used in public datasets. We introduce the XR Body model (XRB), a new parametric mesh representation that decouples skeletal pose and body shape for improved accuracy and interpretability.

3DB employs an encoder-decoder architecture and supports auxiliary prompts, including 2D keypoints and masks, enabling user-guided inference similar to the SAM family of models. We derive high-quality annotations from a multi-stage annotation pipeline using differentiable optimization, multi-view geometry, dense keypoint detection, and a data engine to collect and annotated data covering both common and rare poses across a wide range of viewpoints. Our experiments demonstrate substantial improvements over prior methods, with robust performance on challenging scenarios such as occlusions and rare poses.

## Visual Comparisons

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

**10/20/2025 -- Checkpoints Launched, Dataset Released, Web Demo and Paper are out**
- < MORE DETAILS HERE >

## Installation

Please see [`INSTALL.md`](./INSTALL.md) for environment installation instructions of SAM 3D Body codebase.

## Getting Started

### Download Checkpoints [TODO: Update this]

First, we need to download a model checkpoint. All the model checkpoints can be downloaded by running:

```bash
cd checkpoints && \
./download_ckpts.sh && \
cd ..
```

or individually from:

- [TODO_Update_this_tiny.pt](https://huggingface.co/facebook/TODO_Update_this_tiny) (placeholder link - TODO: Update this)
- [TODO_Update_this_small.pt](https://huggingface.co/facebook/TODO_Update_this_small) (placeholder link - TODO: Update this)  
- [TODO_Update_this_base_plus.pt](https://huggingface.co/facebook/TODO_Update_this_base_plus) (placeholder link - TODO: Update this)
- [TODO_Update_this_large.pt](https://huggingface.co/facebook/TODO_Update_this_large) (placeholder link - TODO: Update this)

Then SAM 3D Body can be used in a few lines as follows for image prediction.

### Image prediction [TODO: Update this]

< MODELS DESCRIPTION HERE >

< SIMPLE CODE TO RUN INFERENCE HERE >

< Link to Colab Notebook >

### Video prediction [TODO: Update this]

< Optional: Do we want to add tooling to run inference on videos? >

< Link to Colab Notebook >

## Load from 🤗 Hugging Face [TODO: Update this]

Alternatively, models can also be loaded from [Hugging Face](https://huggingface.co/models?search=facebook/sam3d) (requires `pip install huggingface_hub`).

For image prediction [TODO: Update this]:

```python
import torch
from sam3d.sam3d_image_predictor import SAM3DImagePredictor

predictor = SAM3DImagePredictor.from_pretrained("facebook/sam3d-hiera-large")

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(<your_image>)
    poses, meshes = predictor.predict(<input_prompts>)
```

## Model Description

### SAM 3D Body checkpoints [TODO: Update this]

The table below shows the SAM 3D Body checkpoints released on [TODO: Update this].

|      **Model**       | **Size (M)** |    **Speed (FPS)**     | **H36M test (MPJPE)** | **3DPW test (MPJPE)** |
| :------------------: | :----------: | :--------------------: | :-----------------: | :----------------: |
|   [TODO: Update this]_tiny <br /> ([config](configs/sam3d/TODO_Update_this_t.yaml), [checkpoint](https://huggingface.co/facebook/TODO_Update_this-tiny))    |     TBD     |          TBD          |        TBD         |        TBD        |
|   [TODO: Update this]_small <br /> ([config](configs/sam3d/TODO_Update_this_s.yaml), [checkpoint](https://huggingface.co/facebook/TODO_Update_this-small))   |      TBD      |          TBD          |        TBD         |        TBD        |
| [TODO: Update this]_base_plus <br /> ([config](configs/sam3d/TODO_Update_this_b.yaml), [checkpoint](https://huggingface.co/facebook/TODO_Update_this-base-plus)) |     TBD     |        TBD          |        TBD         |        TBD        |
|   [TODO: Update this]_large <br /> ([config](configs/sam3d/TODO_Update_this_l.yaml), [checkpoint](https://huggingface.co/facebook/TODO_Update_this-large))   |    TBD     |          TBD          |        TBD         |        TBD        |


< TODO: Update when we run speedtests >
Speed measured on an A100 with `torch 2.5.1, cuda 12.4`. See `benchmark.py` for an example on benchmarking (compiling all the model components). Compiling only the image encoder can be more flexible and also provide (a smaller) speed-up (set `compile_image_encoder: True` in the config).

## SAM 3D Body Dataset [TODO: Update this]

< Info on the 3D annotations we're releasing >

## Training SAM 3D Body [TODO: Update this]

You can train or fine-tune SAM 3D Body on custom datasets of images, videos, or both.

< Link to training README >

## Web demo for SAM 3D Body [TODO: Update this]

< Link to Web Demo >

## License

The SAM 3D Body model checkpoints and code are licensed under [Apache 2.0](./LICENSE).

## Contributing

See [contributing](CONTRIBUTING.md) and the [code of conduct](CODE_OF_CONDUCT.md).

## Contributors

The SAM 3D Body project was made possible with the help of many contributors:
Vivian Lee, George Orlin, Matt Feiszli, Nikhila Ravi, Andrew Westbury, Jyun-Ting Song, Zejia Weng, Xizi Zhang, Yuting Ye, Federica Bogo, Ronald Mallet, Ahmed Osman, Rawal Khirodkar, Javier Romero, Carsten Stoll, Juan Carlos Guzman, Sofien Bouaziz, Yuan Dong, Su Zhaoen, Fabian Prada, Alexander Richard, Michael Zollhoefer, Roman Rädle, Sasha Mitts, Michelle Chan, Yael Yungster, Azita Shokrpour, Helen Klein, Mallika Malhotra, Ida Cheng, Eva Galper.

Third-party code: [TODO: Update this]

## Citing SAM 3D Body [TODO: Update this]

If you use SAM 3D Body or the SAM 3D Body dataset in your research, please use the following BibTeX entry.

```bibtex
@article{yang2025sam3dbody,
  title={SAM 3D Body: Single Image Human Mesh Recovery},
  author={Yang, Xitong and Kukreja, Devansh and Pinkus, Don and Fan, Taosha and Park, David and Shin, Soyong and Cao, Jinkun and Liu, Jiawei and Ugrinovic, Nicolas and Sagar, Anushka and Malik, Jitendra and Dollar, Piotr and Kitani, Kris},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```
