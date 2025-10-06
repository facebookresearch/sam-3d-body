
def create_backbone(name, cfg=None, pretrained=False, drop_path=0.0):
    if name in ["vit_hmr"]:
        from .vit_hmr2 import vit

        backbone = vit(cfg)
    elif name in ["vit_hmr_512_384"]:
        from .vit_hmr2 import vit512_384

        backbone = vit512_384(cfg)
    elif name in ["vit_l"]:
        from .vit_hmr2 import vit_l

        backbone = vit_l(cfg)
    elif name in ["vit_b"]:
        from .vit_hmr2 import vit_b

        backbone = vit_b(cfg)
    else:
        raise NotImplementedError("Backbone type is not implemented")

    return backbone
