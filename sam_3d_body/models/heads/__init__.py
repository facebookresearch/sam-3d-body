# Copyright (c) Meta Platforms, Inc. and affiliates.

from ..modules import to_2tuple
from .camera_head import PerspectiveHead
from .mhr_head import MHRHead


def build_head(cfg, head_type="mhr", enable_hand_model=False, default_scale_factor=1.0):
    if head_type == "mhr":
        return MHRHead(
            input_dim=cfg.MODEL.DECODER.DIM,
            mlp_depth=cfg.MODEL.MHR_HEAD.get("MLP_DEPTH", 1),
            mhr_model_path=cfg.MODEL.MHR_HEAD.MHR_MODEL_PATH,
            mlp_channel_div_factor=cfg.MODEL.MHR_HEAD.get("MLP_CHANNEL_DIV_FACTOR", 1),
            enable_hand_model=enable_hand_model,
        )
    elif head_type == "perspective":
        return PerspectiveHead(
            input_dim=cfg.MODEL.DECODER.DIM,
            img_size=to_2tuple(cfg.MODEL.IMAGE_SIZE),
            mlp_depth=cfg.MODEL.get("CAMERA_HEAD", dict()).get("MLP_DEPTH", 1),
            mlp_channel_div_factor=cfg.MODEL.get("CAMERA_HEAD", dict()).get(
                "MLP_CHANNEL_DIV_FACTOR", 1
            ),
            default_scale_factor=default_scale_factor,
        )
    else:
        raise ValueError("Invalid head type: ", head_type)
