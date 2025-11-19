# Copyright (c) Meta Platforms, Inc. and affiliates.


def create_backbone(name, cfg=None):
    if name in ["vit_hmr"]:
        from .vit import vit

        backbone = vit(cfg)
    elif name in ["vit_hmr_512_384"]:
        from .vit import vit512_384

        backbone = vit512_384(cfg)
    elif name in ["vit_l"]:
        from .vit import vit_l

        backbone = vit_l(cfg)
    elif name in ["vit_b"]:
        from .vit import vit_b

        backbone = vit_b(cfg)
    elif name in [
        "dinov3_vit7b",
        "dinov3_vith16plus",
        "dinov3_vits16",
        "dinov3_vits16plus",
        "dinov3_vitb16",
        "dinov3_vitl16",
    ]:
        from .dinov3 import Dinov3Backbone

        backbone = Dinov3Backbone(name, cfg=cfg)
    else:
        raise NotImplementedError("Backbone type is not implemented")

    return backbone
