# Copyright (c) Meta Platforms, Inc. and affiliates.
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import einops


class CameraEncoder(nn.Module):
    def __init__(self, embed_dim, patch_size=14):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.camera = FourierPositionEncoding(n=3,
                                            num_bands=16,
                                            max_resolution=64)

        self.conv = nn.Conv2d(embed_dim+99, embed_dim, kernel_size=1, bias=False)
        self.norm = LayerNorm2d(embed_dim)

    def forward(self, img_embeddings, rays):
        B, D, _h, _w = img_embeddings.shape

        with torch.no_grad():
            scale = 1 / self.patch_size
            rays = F.interpolate(rays, scale_factor=(scale, scale), mode='bilinear', align_corners=False, antialias=True)
            rays = rays.permute(0, 2, 3, 1).contiguous() # [b, h, w, 2]
            rays = torch.cat([rays, torch.ones_like(rays[..., :1])], dim=-1)
            rays_embeddings = self.camera(pos=rays.reshape(B, -1, 3))                           # (bs, N, 99): rays fourier embedding
            rays_embeddings = einops.rearrange(rays_embeddings, 'b (h w) c -> b c h w', h=_h, w=_w).contiguous()

        z = torch.concat([img_embeddings, rays_embeddings], dim=1)
        z = self.norm(self.conv(z))

        return z


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class FourierPositionEncoding(nn.Module):
    def __init__(self, n, num_bands, max_resolution):
        """
        Module that generate Fourier encoding - no learning involved
        """
        super().__init__()

        self.num_bands = num_bands
        self.max_resolution = [max_resolution] * n

    @property
    def channels(self):
        """
        Return the output dimension
        """
        num_dims = len(self.max_resolution)
        encoding_size = self.num_bands * num_dims
        encoding_size *= 2 # sin-cos
        encoding_size += num_dims # concat

        return encoding_size

    def forward(self, pos):
        """
        Forward pass that take rays as input and generate Fourier positional encodings
        """
        fourier_pos_enc = _generate_fourier_features(pos, num_bands=self.num_bands, max_resolution=self.max_resolution)
        return fourier_pos_enc


def _generate_fourier_features(pos, num_bands, max_resolution):
    """Generate fourier features from a given set of positions and frequencies"""
    b, n = pos.shape[:2]
    device = pos.device

    # Linear frequency sampling
    min_freq = 1.0
    freq_bands = torch.stack([torch.linspace(start=min_freq, end=res / 2, steps=num_bands, device=device) for res in max_resolution], dim=0)

    # Stacking
    per_pos_features = torch.stack([pos[i, :, :][:, :, None] * freq_bands[None, :, :] for i in range(b)], 0)
    per_pos_features = per_pos_features.reshape(b, n, -1)

    # Sin-Cos
    per_pos_features = torch.cat([torch.sin(np.pi * per_pos_features), torch.cos(np.pi * per_pos_features)], dim=-1)

    # Concat with initial pos
    per_pos_features = torch.cat([pos, per_pos_features], dim=-1)

    return per_pos_features
