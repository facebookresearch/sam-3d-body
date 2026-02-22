# Copyright (c) Meta Platforms, Inc. and affiliates.

import importlib
import os
import sys

import torch
from torch import nn


def _load_dinov3_backbone(name: str, **kwargs):
    """Load a DINOv3 backbone without importing hubconf.py.

    ``torch.hub.load`` executes ``hubconf.py`` which unconditionally imports
    the detection module.  That module uses ``@dataclass(kw_only=True)``
    (Python 3.10+), so the hub load crashes on Python 3.9 even though the
    backbone itself is fully compatible.

    This helper ensures the repo is cached, then imports only
    ``dinov3.hub.backbones`` — avoiding the problematic detection import.
    """
    # Ensure the repo is downloaded / cached (no hubconf import here).
    repo_dir = torch.hub._get_cache_or_reload(
        "facebookresearch/dinov3",
        force_reload=False,
        trust_repo=True,
        calling_fn="load",
        verbose=False,
    )

    # Temporarily add the repo to sys.path so its packages are importable.
    need_path = repo_dir not in sys.path
    if need_path:
        sys.path.insert(0, repo_dir)
    try:
        backbones_mod = importlib.import_module("dinov3.hub.backbones")
        fn = getattr(backbones_mod, name, None)
        if fn is None:
            raise ValueError(
                f"Unknown DINOv3 backbone '{name}'. "
                f"Available: {[n for n in dir(backbones_mod) if n.startswith('dinov3_')]}"
            )
        return fn(**kwargs)
    finally:
        if need_path:
            sys.path.remove(repo_dir)


class Dinov3Backbone(nn.Module):
    def __init__(
        self, name="dinov2_vitb14", pretrained_weight=None, cfg=None, *args, **kwargs
    ):
        super().__init__()
        self.name = name
        self.cfg = cfg

        self.encoder = _load_dinov3_backbone(
            self.name,
            pretrained=False,
            drop_path=self.cfg.MODEL.BACKBONE.DROP_PATH_RATE,
        )
        self.patch_size = self.encoder.patch_size
        self.embed_dim = self.embed_dims = self.encoder.embed_dim

    def forward(self, x, extra_embed=None):
        """
        Encode a RGB image using a ViT-backbone
        Args:
            - x: torch.Tensor of shape [bs,3,w,h]
        Return:
            - y: torch.Tensor of shape [bs,k,d] - image in patchified mode
        """
        assert extra_embed is None, "Not Implemented Yet"

        y = self.encoder.get_intermediate_layers(x, n=1, reshape=True, norm=True)[-1]

        return y

    def get_layer_depth(self, param_name: str, prefix: str = "encoder."):
        """Get the layer-wise depth of a parameter.
        Args:
            param_name (str): The name of the parameter.
            prefix (str): The prefix for the parameter.
                Defaults to an empty string.
        Returns:
            Tuple[int, int]: The layer-wise depth and the num of layers.
        Note:
            The first depth is the stem module (``layer_depth=0``), and the
            last depth is the subsequent module (``layer_depth=num_layers-1``)
        """
        num_layers = self.encoder.n_blocks + 2

        if not param_name.startswith(prefix):
            # For subsequent module like head
            return num_layers - 1, num_layers

        param_name = param_name[len(prefix) :]

        if param_name in ("cls_token", "pos_embed", "storage_tokens"):
            layer_depth = 0
        elif param_name.startswith("patch_embed"):
            layer_depth = 0
        elif param_name.startswith("blocks"):
            layer_id = int(param_name.split(".")[1])
            layer_depth = layer_id + 1
        else:
            layer_depth = num_layers - 1

        return layer_depth, num_layers
