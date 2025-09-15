# Multi-HMR
# Copyright (c) 2024-present NAVER Corp.
# CC BY-NC-SA 4.0 license

import torch
from torch import nn
import numpy as np

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