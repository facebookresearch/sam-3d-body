from core.metadata import with_modelroot

import torch

PRETRAINED_WEIGHT_PATH = {
    "vit_hmr": with_modelroot("hmr2/vitpose_backbone.pth"),
    "vit_hmr_256": with_modelroot("hmr2/vitpose_backbone.pth"),
    "vit_hmr_512_384": with_modelroot("hmr2/vitpose_backbone_interpPosEmb32x24.pth"),
    "vit_l": with_modelroot("hmr2/vitpose_l_backbone.pth"),
    "vit_b": with_modelroot("hmr2/vitpose_b_backbone.pth"),
    # "sapiens_0.3b": with_modelroot("sapiens/sapiens_0.3b/body/checkpoint.pth"),
    # "sapiens_0.6b": with_modelroot("sapiens/sapiens_0.6b/body/checkpoint.pth"),
    # "sapiens_1b": with_modelroot("sapiens/sapiens_1b/body/checkpoint.pth"),
    # "sapiens_2b": with_modelroot("sapiens/sapiens_2b/body/checkpoint.pth"),
    # "sapiens_clean_0.3b": with_modelroot("sapiens/sapiens_0.3b/clean/checkpoint.pth"),
    # "sapiens_clean_0.6b": with_modelroot("sapiens/sapiens_0.6b/clean/checkpoint.pth"),
    # "sapiens_clean_1b": with_modelroot("sapiens/sapiens_1b/clean/checkpoint.pth"),
    # "sapiens_clean_2b": with_modelroot("sapiens/sapiens_2b/clean/checkpoint.pth"),
    # "pe_core_b": with_modelroot("perception_encoder/PE-Core-B16-224.pt"),
    # "pe_core_l": with_modelroot("perception_encoder/PE-Core-L14-336.pt"),
    # "pe_core_g": with_modelroot("perception_encoder/PE-Core-G14-448.pt"),
    # "pe_spatial_g": with_modelroot("perception_encoder/PE-Spatial-G14-448.pt"),
    # "pe_spatial_l": with_modelroot("perception_encoder/PE-Spatial-L14-448.pt"),
    # "pe_spatial_b": with_modelroot("perception_encoder/PE-Spatial-B16-512.pt"),
    # "pe_spatial_s": with_modelroot("perception_encoder/PE-Spatial-S16-512.pt"),
    # "pe_spatial_t": with_modelroot("perception_encoder/PE-Spatial-T16-512.pt"),
    "dinov3_vits16": with_modelroot("dinov3/dinov3_vits16"),
    "dinov3_vits16plus": with_modelroot("dinov3/dinov3_vits16plus"),
    "dinov3_vitb16": with_modelroot("dinov3/dinov3_vitb16"),
    "dinov3_vitl16": with_modelroot("dinov3/dinov3_vitl16"),
    "dinov3_vith16plus": with_modelroot("dinov3/dinov3_vith16plus"),
    "dinov3_vit7b": with_modelroot("dinov3/dinov3_vit7b"),
}


def create_backbone(name, cfg=None, pretrained=False, drop_path=0.0):
    if name in ["vit_hmr", "hmr2", "vit"]:
        from .vit_hmr2 import vit

        backbone = vit(cfg)
    elif name in ["vit_hmr_256"]:
        from .vit_hmr2 import vit256

        backbone = vit256(cfg)
    elif name in ["vit_hmr_512_384"]:
        from .vit_hmr2 import vit512_384

        backbone = vit512_384(cfg)
    elif name in ["vit_l"]:
        from .vit_hmr2 import vit_l

        backbone = vit_l(cfg)
    elif name in ["vit_b"]:
        from .vit_hmr2 import vit_b

        backbone = vit_b(cfg)
    elif name in [
        "sapiens_0.3b",
        "sapiens_0.6b",
        "sapiens_1b",
        "sapiens_2b",
        "sapiens_clean_0.3b",
        "sapiens_clean_0.6b",
        "sapiens_clean_1b",
        "sapiens_clean_2b",
    ]:
        from .vit_sapiens import build_vit

        raise NotImplementedError("Sapiens models are not supported anymore")
        backbone = build_vit(name, cfg)
    elif name in [
        "dinov2_vitb14",
        "dinov2_vitl14",
        "dinov2_vits14",
        "dinov2_vitl14_reg",
        "dinov2_vitb14_reg",
    ]:
        from .dinov2 import Dinov2Backbone

        backbone = Dinov2Backbone(name, pretrained=True)
    elif name in [
        "pe_core_b",
        "pe_core_l",
        "pe_core_g",
        "pe_spatial_g",
        "pe_spatial_l",
        "pe_spatial_b",
        "pe_spatial_s",
        "pe_spatial_t",
    ]:
        from .perception_encoder import PEBackbone

        pe = PEBackbone(name=name, cfg=cfg)
        for p in pe.parameters():
            p.requires_grad = True
        backbone = pe
    elif name in [
        "dinov3_vit7b",
        "dinov3_vith16plus",
        "dinov3_vits16",
        "dinov3_vits16plus",
        "dinov3_vitb16",
        "dinov3_vitl16",
    ]:
        from .dinov3 import Dinov3Backbone

        backbone = Dinov3Backbone(
            name, pretrained_weight=PRETRAINED_WEIGHT_PATH[name], drop_path=drop_path
        )
    else:
        raise NotImplementedError("Backbone type is not implemented")

    if pretrained:
        from core.utils.checkpoint import load_state_dict
        from core.utils.logging import get_pylogger

        logger = get_pylogger(__name__)

        if not name in PRETRAINED_WEIGHT_PATH:
            raise ValueError(f"Unknown pretraiend weights for backbone type {name}")
        weight_path = PRETRAINED_WEIGHT_PATH[name]

        logger.info(f"Loading default backbone weights from {weight_path}")
        checkpoint = torch.load(
            weight_path,
            map_location="cpu",
            weights_only=True,
        )
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        if name in [
            "vit_b",
            "vit_l",
            "vit_hmr",
            "vit",
            "vit_hmr_256",
            "vit_hmr_512_384",
        ]:  # FIXME
            load_state_dict(backbone, state_dict)

        elif name in [
            "dinov3_vit7b",
            "dinov3_vith16plus",
            "dinov3_vits16",
            "dinov3_vits16plus",
            "dinov3_vitb16",
            "dinov3_vitl16",
        ]:
            load_state_dict(backbone.encoder, state_dict, strict=False)

    return backbone
