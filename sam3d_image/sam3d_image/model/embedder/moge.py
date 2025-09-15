import torch
import warnings
from moge.model.v1 import MoGeModel
from sam3d_image.model.backbone.dit.embedder.dino import Dino
import os
from loguru import logger


# MoGe feature extractor; inherit from DINO since mostly the same model
class MoGe(Dino):
    def __init__(
        self,
        input_size: int = 518,
        moge_model: str = "Ruicheng/moge-vitl",
        normalize_images: bool = True,
        # for backward compatible
        prenorm_features: bool = False,
        use_head_features: bool = False,
        freeze_backbone: bool = True,
        prune_network: bool = False,
    ):
        torch.nn.Module.__init__(self)
        logger.info(os.path.dirname(__file__))
        logger.info(moge_model)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            moge_model = MoGeModel.from_pretrained(moge_model)
            self.backbone = moge_model.backbone
            if use_head_features:
                self.head_proj = moge_model.head.projects
                self.intermediate_layers = moge_model.intermediate_layers
            else:
                # Don't need the parameters
                self.head_proj = None
                self.intermediate_layers = None

        self.resize_input_size = (input_size, input_size)
        self.use_head_features = use_head_features
        if self.use_head_features:
            self.embed_dim = self.head_proj[0].weight.shape[0]
        else:
            self.embed_dim = self.backbone.embed_dim
        self.input_size = input_size
        self.input_channels = 3
        self.normalize_images = normalize_images
        self.prenorm_features = prenorm_features

        # freeze
        if freeze_backbone:
            self.requires_grad_(False)
            self.eval()
        elif not prune_network:
            logger.warning(
                "Unfreeze encoder w/o prune parameter may lead to error in ddp/fp16 training"
            )

        if prune_network:
            self._prune_network()

    def _forward_heads(self, hidden_states, patch_h, patch_w):
        # it's strange, MoGe does not use the cls tokens in hidden_states
        summed_tokens = torch.stack(
            [
                proj(
                    feat.permute(0, 2, 1).unflatten(2, (patch_h, patch_w)).contiguous()
                )
                for proj, (feat, _) in zip(self.head_proj, hidden_states)
            ],
            dim=1,
        ).sum(dim=1)
        return summed_tokens.flatten(2).permute(0, 2, 1)

    def forward(self, x, **kwargs):
        _resized_images = self._preprocess_input(x)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if self.use_head_features:
                # we don't have to do this given we preprocess the input; but keep here for future needs
                img_h, img_w = _resized_images.shape[-2:]
                patch_h, patch_w = img_h // 14, img_w // 14

                hidden_states = self._forward_intermediate_layers(
                    _resized_images,
                    intermediate_layers=self.intermediate_layers,
                )
                tokens = self._forward_heads(hidden_states, patch_h, patch_w)
            else:
                tokens = self._forward_last_layer(_resized_images)
        return tokens.to(x.dtype)
