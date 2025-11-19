# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
import torch.nn as nn

# FP16_TYPE = torch.float16

FP16_MODULES = [
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
    nn.Linear,
]

FP16_MODULES = tuple(FP16_MODULES)


def convert_to_fp16_safe(module, dtype=torch.float16):
    for child in module.children():
        convert_to_fp16_safe(child, dtype)
    if not isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
        module.to(dtype)


def convert_module_to_f16(l, dtype=torch.float16):
    """
    Convert primitive modules to float16.
    """
    if isinstance(l, FP16_MODULES):
        for p in l.parameters():
            # p.data = p.data.half()
            p.data = p.data.to(dtype)


def convert_module_to_f32(l):
    """
    Convert primitive modules to float32, undoing convert_module_to_f16().
    """
    if isinstance(l, FP16_MODULES):
        for p in l.parameters():
            p.data = p.data.float()


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
