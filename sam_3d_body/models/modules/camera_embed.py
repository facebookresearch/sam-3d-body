# Copyright (c) Meta Platforms, Inc. and affiliates.

import einops
import numpy as np
import torch
import torch.nn.functional as F

from sam_3d_body.models.modules.transformer import LayerNorm2d
from torch import nn


class CameraEncoder(nn.Module):
    def __init__(self, embed_dim, patch_size=14):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.camera = FourierPositionEncoding(n=3, num_bands=16, max_resolution=64)

        self.conv = nn.Conv2d(embed_dim + 99, embed_dim, kernel_size=1, bias=False)
        self.norm = LayerNorm2d(embed_dim)

    def forward(self, img_embeddings, rays):
        B, D, _h, _w = img_embeddings.shape

        with torch.no_grad():
            scale = 1 / self.patch_size
            rays = F.interpolate(
                rays,
                scale_factor=(scale, scale),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            rays = rays.permute(0, 2, 3, 1).contiguous()  # [b, h, w, 2]
            rays = torch.cat([rays, torch.ones_like(rays[..., :1])], dim=-1)
            rays_embeddings = self.camera(
                pos=rays.reshape(B, -1, 3)
            )  # (bs, N, 99): rays fourier embedding
            rays_embeddings = einops.rearrange(
                rays_embeddings, "b (h w) c -> b c h w", h=_h, w=_w
            ).contiguous()

        z = torch.concat([img_embeddings, rays_embeddings], dim=1)
        z = self.norm(self.conv(z))

        return z


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
        encoding_size *= 2  # sin-cos
        encoding_size += num_dims  # concat

        return encoding_size

    def forward(self, pos):
        """
        Forward pass that take rays as input and generate Fourier positional encodings
        """
        fourier_pos_enc = _generate_fourier_features(
            pos, num_bands=self.num_bands, max_resolution=self.max_resolution
        )
        return fourier_pos_enc


def _generate_fourier_features(pos, num_bands, max_resolution):
    """Generate fourier features from a given set of positions and frequencies"""
    b, n = pos.shape[:2]
    device = pos.device

    # Linear frequency sampling
    min_freq = 1.0
    freq_bands = torch.stack(
        [
            torch.linspace(start=min_freq, end=res / 2, steps=num_bands, device=device)
            for res in max_resolution
        ],
        dim=0,
    )

    # Stacking
    per_pos_features = torch.stack(
        [pos[i, :, :][:, :, None] * freq_bands[None, :, :] for i in range(b)], 0
    )
    per_pos_features = per_pos_features.reshape(b, n, -1)

    # Sin-Cos
    per_pos_features = torch.cat(
        [torch.sin(np.pi * per_pos_features), torch.cos(np.pi * per_pos_features)],
        dim=-1,
    )

    # Concat with initial pos
    per_pos_features = torch.cat([pos, per_pos_features], dim=-1)

    return per_pos_features
