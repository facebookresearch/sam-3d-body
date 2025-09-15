import torch

from core.metadata import with_modelroot
from torch import nn

PRETRAIN_PATH = {
    "dinov2_vitb14_reg": "/private/home/jinhyun1/codes/dinov2/dinov2_vitb14_reg4_pretrain.pth",
    "dinov2_vitl14_reg": with_modelroot("dinov2/dinov2_vitl14_reg4_pretrain.pth"),
}


class Dinov2Backbone(nn.Module):
    def __init__(self, name="dinov2_vitb14", pretrained=False, *args, **kwargs):
        super().__init__()
        self.name = name
        self.encoder = torch.hub.load(
            with_modelroot("dinov2/dinov2"), self.name, source="local", pretrained=False
        )
        if pretrained:
            ckpt = torch.load(PRETRAIN_PATH[self.name], map_location="cpu")
            self.encoder.load_state_dict(
                ckpt.get("state_dict", ckpt),
                strict=False,
            )
        self.patch_size = self.encoder.patch_size
        self.embed_dim = self.embed_dims = self.encoder.embed_dim

        del self.encoder.mask_token  # Unused, and it causes issues.

    def forward(self, x, extra_embed=None):
        """
        Encode a RGB image using a ViT-backbone
        Args:
            - x: torch.Tensor of shape [bs,3,w,h]
        Return:
            - y: torch.Tensor of shape [bs,k,d] - image in patchified mode
        """
        assert extra_embed is None, "Not Implemented Yet"
        assert len(x.shape) == 4
        # with torch.autocast(device_type="cuda", dtype=torch.float16): y = self.encoder.get_intermediate_layers(x)[0] # ViT-L+896x896: [bs,4096,1024] - [bs,nb_patches,emb]
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            y = self.encoder.get_intermediate_layers(x, reshape=True)[
                0
            ]  # ViT-L+896x896: [bs,4096,1024] - [bs,nb_patches,emb]
        # print(x.shape, torch.is_autocast_enabled(), y.requires_grad, y.dtype)

        return y
