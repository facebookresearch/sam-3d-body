# SAM 3D: Segment Anything 

**[AI at Meta, FAIR](https://ai.meta.com/research/)**

< Authors go here >

[[`<REPLACE ME Paper>`](https://ai.meta.com/research/publications/sam-2-segment-anything-in-images-and-videos/)] [[`<REPLACE ME Project>`](https://ai.meta.com/sam2)] [[`<REPLACE ME Demo>`](https://sam2.metademolab.com/)] [[`<REPLACE ME Dataset>`](https://ai.meta.com/datasets/segment-anything-video)] [[`<REPLACE ME Blog>`](https://ai.meta.com/blog/segment-anything-2)] [[`<REPLACE ME BibTeX>`](#citing-sam-2)]

< Architecture Diagram goes here >

**Segment Anything 3D Models (SAM 3D)** is a pair of foundation models towards solving 3D perception from monocular views. < MORE DETAILS HERE>

## Latest updates

**10/15/2025 -- Checkpoints Launched, Dataset Released, Web Demo and Paper are out**
- < MORE DETAILS HERE >

## Installation

< INSTALLATION INSTRUCTIONS HERE >

## Getting Started

### Download Checkpoints

First, we need to download a model checkpoint. All the model checkpoints can be downloaded by running:

< DOWNLOAD CLI INSTRUCTIONS HERE >

Then SAM 3D can be used in a few lines as follows for image prediction.

### Image prediction

< MODELS DESCRIPTION HERE >

< SIMPLE CODE TO RUN INFERENCE HERE >

< Link to Colab Notebook >

### Video prediction

< Optional: Do we want to add tooling to run inference on videos? >

< Link to Colab Notebook >

## Load from 🤗 Hugging Face

Alternatively, models can also be loaded from [Hugging Face](https://huggingface.co/models?search=facebook/sam2) (requires `pip install huggingface_hub`).

For image prediction:

< Code sample to load and run inference with huggingface >

## Model Description

### SAM 3D checkpoints

The table below shows the SAM 3D checkpoints released on September 23, 2025.

< Table with model, size, speed, and MPJPE >

|      **Model**       | **Size (M)** |    **Speed (FPS)**     | **<Dataset> test (PCK @ 0.05)** | **<Dataset> test (MPJPE)** |
| :------------------: | :----------: | :--------------------: | :-----------------: | :----------------: |


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

< TODO: Add License Type once it's decided >

## Contributing

< TODO: Add contributing.md and code_of_conduct.md >

See [contributing](CONTRIBUTING.md) and the [code of conduct](CODE_OF_CONDUCT.md).

## Contributors

The SAM 3D project was made possible with the help of many contributors (alphabetical):

< List of collaborators here >

Third-party code: < Credit third party code here >

## Citing SAM 3D

If you use SAM 3D or the SAM3D dataset in your research, please use the following BibTeX entry.

< TODO: Add bibtex here >
