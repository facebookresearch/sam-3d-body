import copy
import math
import random
from collections import OrderedDict

from dataclasses import asdict, dataclass, replace
from functools import partial
from logging import getLogger
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np

import torch
import torch.nn as nn

from einops import rearrange, repeat
try:
    from flash_attn.flash_attn_interface import flash_attn_qkvpacked_func
except:
    print("No Flash Attention!")
from timm.layers import DropPath
from torch import broadcast_tensors, einsum, nn, Tensor
from torch.amp import autocast
from torch.nn import functional as F, Module, ModuleList
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.parameter import Parameter
from torch.utils.checkpoint import checkpoint


logger = getLogger()


def fetch_pe_checkpoint(name: str, path: Optional[str] = None):
    path = path or f"hf://facebook/{name}:{name}.pt"

    if path.startswith("hf://"):
        # Load from huggingface
        path = path[len("hf://") :]
        repo, file = path.split(":")

        # To count the download, config.yaml is empty
        hf_hub_download(repo_id=repo, filename="config.yaml")
        return hf_hub_download(repo_id=repo, filename=file)
    else:
        return path


@dataclass
class PEConfig:
    """Vision Tower Config."""

    patch_size: int
    width: int
    layers: int
    heads: int
    mlp_ratio: float
    output_dim: Optional[int]

    ls_init_value: float = None
    drop_path: float = 0.0

    image_size: int = (224,)
    use_abs_posemb: bool = True
    use_cls_token: bool = False
    use_rope2d: bool = True

    pool_type: str = "attn"
    attn_pooler_heads: int = 8

    use_ln_pre: bool = True
    use_ln_post: bool = True


@dataclass
class PETextConfig:
    """Text Tower Config."""

    context_length: int
    width: int
    heads: int
    layers: int

    output_dim: int

    mlp_ratio: float = 4.0
    vocab_size: int = 49408


PE_VISION_CONFIG = {}
PE_TEXT_CONFIG = {}


#########################################
#                PE CORE                #
#########################################

# PE_VISION_CONFIG["PE-Core-G14-448"] = PEConfig(
#     image_size=448,
#     patch_size=14,
#     width=1536,
#     layers=50,
#     heads=16,
#     mlp_ratio=8960 / 1536,
#     pool_type="attn",
#     output_dim=1280,
#     use_cls_token=False,
# )
PE_VISION_CONFIG["PE-Core-G14-448"] = PEConfig(
    image_size=448,
    patch_size=14,
    width=1536,
    layers=50,
    heads=16,
    mlp_ratio=8960 / 1536,
    pool_type="none",
    use_ln_post=False,
    output_dim=None,
    use_cls_token=False,
)
PE_TEXT_CONFIG["PE-Core-G14-448"] = PETextConfig(
    context_length=72, width=1280, heads=20, layers=24, output_dim=1280
)


# PE_VISION_CONFIG["PE-Core-L14-336"] = PEConfig(
#     image_size=336,
#     patch_size=14,
#     width=1024,
#     layers=24,
#     heads=16,
#     mlp_ratio=4.0,
#     pool_type="attn",
#     output_dim=1024,
#     use_cls_token=True,
# )
PE_VISION_CONFIG["PE-Core-L14-336"] = PEConfig(
    image_size=336,
    patch_size=14,
    width=1024,
    layers=24,
    heads=16,
    mlp_ratio=4.0,
    pool_type="none",
    use_ln_post=False,
    output_dim=None,
    use_cls_token=True,
)
PE_TEXT_CONFIG["PE-Core-L14-336"] = PETextConfig(
    context_length=32, width=1024, heads=16, layers=24, output_dim=1024
)


# PE_VISION_CONFIG["PE-Core-B16-224"] = PEConfig(
#     image_size=224,
#     patch_size=16,
#     width=768,
#     layers=12,
#     heads=12,
#     mlp_ratio=4.0,
#     pool_type="attn",
#     output_dim=1024,
#     use_cls_token=True,
# )
PE_VISION_CONFIG["PE-Core-B16-224"] = PEConfig(
    image_size=224,
    patch_size=16,
    width=768,
    layers=12,
    heads=12,
    mlp_ratio=4.0,
    pool_type="none",
    use_ln_post=False,
    output_dim=None,
    use_cls_token=True,
)
PE_TEXT_CONFIG["PE-Core-B16-224"] = PE_TEXT_CONFIG["PE-Core-L14-336"]

# PE_VISION_CONFIG["PE-Core-S16-384"] = PEConfig(
#     image_size=384,
#     patch_size=16,
#     width=384,
#     layers=12,
#     heads=6,
#     mlp_ratio=4.0,
#     pool_type="attn",
#     output_dim=512,
#     use_cls_token=True,
# )
PE_VISION_CONFIG["PE-Core-S16-384"] = PEConfig(
    image_size=384,
    patch_size=16,
    width=384,
    layers=12,
    heads=6,
    mlp_ratio=4.0,
    pool_type="none",
    use_ln_post=False,
    output_dim=None,
    use_cls_token=True,
)
PE_TEXT_CONFIG["PE-Core-S16-384"] = PETextConfig(
    context_length=32, width=512, heads=8, layers=12, output_dim=512
)


# PE_VISION_CONFIG["PE-Core-T16-384"] = PEConfig(
#     image_size=384,
#     patch_size=16,
#     width=192,
#     layers=12,
#     heads=3,
#     mlp_ratio=4.0,
#     pool_type="attn",
#     output_dim=512,
#     use_cls_token=True,
# )
PE_VISION_CONFIG["PE-Core-T16-384"] = PEConfig(
    image_size=384,
    patch_size=16,
    width=192,
    layers=12,
    heads=3,
    mlp_ratio=4.0,
    pool_type="none",
    use_ln_post=False,
    output_dim=None,
    use_cls_token=True,
)
PE_TEXT_CONFIG["PE-Core-T16-384"] = PE_TEXT_CONFIG["PE-Core-S16-384"]
#########################################
#                PE Lang                #
#########################################

PE_VISION_CONFIG["PE-Lang-G14-448"] = replace(
    PE_VISION_CONFIG["PE-Core-G14-448"],
    image_size=448,
    pool_type="none",
    use_ln_post=False,
    output_dim=None,
    ls_init_value=0.1,
    layers=47,
)

PE_VISION_CONFIG["PE-Lang-L14-448"] = replace(
    PE_VISION_CONFIG["PE-Core-L14-336"],
    image_size=448,
    pool_type="none",
    use_ln_post=False,
    output_dim=None,
    ls_init_value=0.1,
    layers=23,
)


#########################################
#               PE Spatial              #
#########################################

PE_VISION_CONFIG["PE-Spatial-G14-448"] = replace(
    PE_VISION_CONFIG["PE-Core-G14-448"],
    image_size=448,
    pool_type="attn",
    use_ln_post=False,
    output_dim=None,
    ls_init_value=0.1,
)

PE_VISION_CONFIG["PE-Spatial-L14-448"] = replace(
    PE_VISION_CONFIG["PE-Core-L14-336"],
    image_size=448,
    pool_type="none",
    use_ln_post=False,
    output_dim=None,
)


PE_VISION_CONFIG["PE-Spatial-B16-512"] = replace(
    PE_VISION_CONFIG["PE-Core-B16-224"],
    image_size=512,
    pool_type="none",
    use_ln_post=False,
    output_dim=None,
)


PE_VISION_CONFIG["PE-Spatial-S16-512"] = replace(
    PE_VISION_CONFIG["PE-Core-S16-384"],
    image_size=512,
    pool_type="none",
    use_ln_post=False,
    output_dim=None,
)


PE_VISION_CONFIG["PE-Spatial-T16-512"] = replace(
    PE_VISION_CONFIG["PE-Core-T16-384"],
    image_size=512,
    pool_type="none",
    use_ln_post=False,
    output_dim=None,
)


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


# broadcat, as tortoise-tts was using it


def broadcat(tensors, dim=-1):
    broadcasted_tensors = broadcast_tensors(*tensors)
    return torch.cat(broadcasted_tensors, dim=dim)


# rotary embedding helper functions


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


@autocast("cuda", enabled=False)
def apply_rotary_emb(freqs, t, start_index=0, scale=1.0, seq_dim=-2):
    dtype = t.dtype

    if t.ndim == 3:
        seq_len = t.shape[seq_dim]
        freqs = freqs[-seq_len:]

    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim

    assert (
        rot_dim <= t.shape[-1]
    ), f"feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}"

    t_left, t, t_right = (
        t[..., :start_index],
        t[..., start_index:end_index],
        t[..., end_index:],
    )
    # breakpoint()
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    out = torch.cat((t_left, t, t_right), dim=-1)

    return out.type(dtype)


# learned rotation helpers


def apply_learned_rotations(rotations, t, start_index=0, freq_ranges=None):
    if exists(freq_ranges):
        rotations = einsum("..., f -> ... f", rotations, freq_ranges)
        rotations = rearrange(rotations, "... r f -> ... (r f)")

    rotations = repeat(rotations, "... n -> ... (n r)", r=2)
    return apply_rotary_emb(rotations, t, start_index=start_index)


class RotaryEmbedding(Module):
    def __init__(
        self,
        dim,
        custom_freqs: Optional[Tensor] = None,
        freqs_for: Union[
            Literal["lang"], Literal["pixel"], Literal["constant"]
        ] = "lang",
        theta=10000,
        max_freq=10,
        num_freqs=1,
        learned_freq=False,
        use_xpos=False,
        xpos_scale_base=512,
        interpolate_factor=1.0,
        theta_rescale_factor=1.0,
        seq_before_head_dim=False,
        cache_if_possible=True,
    ):
        super().__init__()
        # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
        # has some connection to NTK literature
        # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/

        theta *= theta_rescale_factor ** (dim / (dim - 2))

        self.freqs_for = freqs_for

        if exists(custom_freqs):
            freqs = custom_freqs
        elif freqs_for == "lang":
            freqs = 1.0 / (
                theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
            )
        elif freqs_for == "pixel":
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * pi
        elif freqs_for == "constant":
            freqs = torch.ones(num_freqs).float()

        self.cache_if_possible = cache_if_possible

        self.tmp_store("cached_freqs", None)
        self.tmp_store("cached_scales", None)

        self.freqs = nn.Parameter(freqs, requires_grad=learned_freq)

        self.learned_freq = learned_freq

        # dummy for device

        self.tmp_store("dummy", torch.tensor(0))

        # default sequence dimension

        self.seq_before_head_dim = seq_before_head_dim
        self.default_seq_dim = -3 if seq_before_head_dim else -2

        # interpolation factors

        assert interpolate_factor >= 1.0
        self.interpolate_factor = interpolate_factor

        # xpos

        self.use_xpos = use_xpos
        if not use_xpos:
            self.tmp_store("scale", None)
            return

        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)

        self.scale_base = xpos_scale_base
        self.tmp_store("scale", scale)

        # add apply_rotary_emb as static method

        self.apply_rotary_emb = staticmethod(apply_rotary_emb)

    @property
    def device(self):
        return self.dummy.device

    def tmp_store(self, key, value):
        self.register_buffer(key, value, persistent=False)

    def get_seq_pos(self, seq_len, device, dtype, offset=0):
        return (
            torch.arange(seq_len, device=device, dtype=dtype) + offset
        ) / self.interpolate_factor

    def rotate_queries_or_keys(self, t, seq_dim=None, offset=0):
        seq_dim = default(seq_dim, self.default_seq_dim)

        assert (
            not self.use_xpos
        ), "you must use `.rotate_queries_and_keys` method instead and pass in both queries and keys, for length extrapolatable rotary embeddings"

        device, dtype, seq_len = t.device, t.dtype, t.shape[seq_dim]

        freqs = self.forward(
            self.get_seq_pos(seq_len, device=device, dtype=dtype, offset=offset),
            seq_len=seq_len,
            offset=offset,
        )

        if seq_dim == -3:
            freqs = rearrange(freqs, "n d -> n 1 d")

        return apply_rotary_emb(freqs, t, seq_dim=seq_dim)

    def rotate_queries_with_cached_keys(self, q, k, seq_dim=None, offset=0):
        seq_dim = default(seq_dim, self.default_seq_dim)

        q_len, k_len = q.shape[seq_dim], k.shape[seq_dim]
        assert q_len <= k_len

        rotated_q = self.rotate_queries_or_keys(
            q, seq_dim=seq_dim, offset=k_len - q_len + offset
        )
        rotated_k = self.rotate_queries_or_keys(k, seq_dim=seq_dim, offset=offset)

        rotated_q = rotated_q.type(q.dtype)
        rotated_k = rotated_k.type(k.dtype)

        return rotated_q, rotated_k

    def rotate_queries_and_keys(self, q, k, seq_dim=None):
        seq_dim = default(seq_dim, self.default_seq_dim)

        assert self.use_xpos
        device, dtype, seq_len = q.device, q.dtype, q.shape[seq_dim]

        seq = self.get_seq_pos(seq_len, dtype=dtype, device=device)

        freqs = self.forward(seq, seq_len=seq_len)
        scale = self.get_scale(seq, seq_len=seq_len).to(dtype)

        if seq_dim == -3:
            freqs = rearrange(freqs, "n d -> n 1 d")
            scale = rearrange(scale, "n d -> n 1 d")

        rotated_q = apply_rotary_emb(freqs, q, scale=scale, seq_dim=seq_dim)
        rotated_k = apply_rotary_emb(freqs, k, scale=scale**-1, seq_dim=seq_dim)

        rotated_q = rotated_q.type(q.dtype)
        rotated_k = rotated_k.type(k.dtype)

        return rotated_q, rotated_k

    def get_scale(self, t: Tensor, seq_len: Optional[int] = None, offset=0):
        assert self.use_xpos

        should_cache = self.cache_if_possible and exists(seq_len)

        if (
            should_cache
            and exists(self.cached_scales)
            and (seq_len + offset) <= self.cached_scales.shape[0]
        ):
            return self.cached_scales[offset : (offset + seq_len)]

        scale = 1.0
        if self.use_xpos:
            power = (t - len(t) // 2) / self.scale_base
            scale = self.scale ** rearrange(power, "n -> n 1")
            scale = torch.cat((scale, scale), dim=-1)

        if should_cache:
            self.tmp_store("cached_scales", scale)

        return scale

    def get_axial_freqs(self, *dims):
        Colon = slice(None)
        all_freqs = []

        for ind, dim in enumerate(dims):
            if self.freqs_for == "pixel":
                pos = torch.linspace(-1, 1, steps=dim, device=self.device)
            else:
                pos = torch.arange(dim, device=self.device)

            freqs = self.forward(pos, seq_len=dim)

            all_axis = [None] * len(dims)
            all_axis[ind] = Colon

            new_axis_slice = (Ellipsis, *all_axis, Colon)
            all_freqs.append(freqs[new_axis_slice])

        all_freqs = broadcast_tensors(*all_freqs)
        return torch.cat(all_freqs, dim=-1)

    @autocast("cuda", enabled=False)
    def forward(self, t: Tensor, seq_len=None, offset=0):
        should_cache = (
            self.cache_if_possible
            and not self.learned_freq
            and exists(seq_len)
            and self.freqs_for != "pixel"
        )

        if (
            should_cache
            and exists(self.cached_freqs)
            and (offset + seq_len) <= self.cached_freqs.shape[0]
        ):
            return self.cached_freqs[offset : (offset + seq_len)].detach()

        freqs = self.freqs

        freqs = einsum("..., f -> ... f", t.type(freqs.dtype), freqs)
        freqs = repeat(freqs, "... n -> ... (n r)", r=2)

        if should_cache:
            self.tmp_store("cached_freqs", freqs.detach())

        return freqs


class Rope2D:
    """Helper class to apply RoPE2D as well as interpolate on the fly."""

    def __init__(self, dim, use_cls_token=False):
        self.dim = dim
        self.use_cls_token = use_cls_token
        self.grid_size = None
        self.freq = None

    def init_tensors(self):
        self.rope = RotaryEmbedding(self.dim // 2)

    def update_grid(self, device, grid_h, grid_w):
        if self.grid_size != (grid_h, grid_w):
            self.grid_size = (grid_h, grid_w)

            self.rope = self.rope.to(device)

            if self.use_cls_token:
                # +1 to leave space for the cls token to be (0, 0)
                grid_y_range = torch.arange(grid_h, device=device) + 1
                grid_x_range = torch.arange(grid_w, device=device) + 1
            else:
                grid_y_range = torch.arange(grid_h, device=device)
                grid_x_range = torch.arange(grid_w, device=device)

            freqs_y = self.rope(grid_y_range)[:, None].expand(grid_h, grid_w, -1)
            freqs_x = self.rope(grid_x_range)[None, :].expand(grid_h, grid_w, -1)
            freq = torch.cat([freqs_x, freqs_y], dim=-1).reshape(grid_h * grid_w, -1)

            if self.use_cls_token:
                freq = torch.cat(
                    [torch.zeros(1, freq.shape[-1], device=device), freq], dim=0
                )

            self.freq = freq[None, ...]

        self.freq = self.freq.to(device)

    def __call__(self, q, k):
        # batch, heads, seq, dim = q.shape
        q = apply_rotary_emb(self.freq[:, None, :, :], q)
        k = apply_rotary_emb(self.freq[:, None, :, :], k)

        return q, k


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.dim = dim
        self.init_values = init_values

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

    def init_tensors(self):
        self.gamma = nn.Parameter(self.init_values * torch.ones(self.dim))


class AttentionPooling(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_probe: int = 1,
        mlp_ratio: int = 4,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        assert (
            self.embed_dim % num_heads == 0
        ), "embed_dim must be divisible by num_heads"

        self.probe = nn.Parameter(torch.randn(1, num_probe, self.embed_dim))
        self.attn = nn.MultiheadAttention(
            self.embed_dim, self.num_heads, batch_first=True
        )

        self.layernorm = norm_layer(embed_dim)
        self.mlp_width = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(self.embed_dim, self.mlp_width)),
                    ("gelu", act_layer()),
                    ("c_proj", nn.Linear(self.mlp_width, self.embed_dim)),
                ]
            )
        )

    def forward(self, x: torch.Tensor):
        batch, _, _ = x.shape

        q = self.probe.repeat((batch, 1, 1)).to(x.dtype)
        x = self.attn(q, x, x, need_weights=False)[0]
        x = x + self.mlp(self.layernorm(x))

        return x


# class SelfAttention(nn.Module):
#     r"""
#     Implements sequence packed attention and RoPe
#     """

#     def __init__(
#         self,
#         embed_dim: int,
#         num_heads: int,
#         rope: Optional[nn.Module] = None,
#     ):
#         super(SelfAttention, self).__init__()
#         self.embed_dim = embed_dim

#         self.num_heads = num_heads
#         self.head_dim = embed_dim // num_heads
#         assert (
#             self.head_dim * num_heads == self.embed_dim
#         ), "embed_dim must be divisible by num_heads"

#         # To make this compatibile with nn.MultiHeadAttention
#         self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))
#         self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
#         self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

#         self.rope = rope
#         self.scale = self.head_dim ** (-0.5)

#     def init_tensors(self):
#         xavier_uniform_(self.in_proj_weight)
#         constant_(self.in_proj_bias, 0.0)
#         constant_(self.out_proj.bias, 0.0)

#     def forward(self, x, attn_mask=None):
#         batch, seq, embed_dim = x.shape
#         proj = F.linear(x, self.in_proj_weight, self.in_proj_bias)

#         # reshape to 3, E and not E, 3 is deliberate for better memory coalescing and keeping same order as chunk()
#         proj = (
#             proj.unflatten(-1, (3, embed_dim))
#             .unsqueeze(0)
#             .transpose(0, -2)
#             .squeeze(-2)
#             .contiguous()
#         )
#         q, k, v = proj[0], proj[1], proj[2]

#         # Use "q_" so that we don't accidentally quit in pdb :)
#         q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads)
#         k = rearrange(k, "b s (h d) -> b h s d", h=self.num_heads)
#         v = rearrange(v, "b s (h d) -> b h s d", h=self.num_heads)

#         if self.rope:
#             q, k = self.rope(q, k)

#         attn = self.scaled_dot_product_attention(
#             q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=self.scale
#         )

#         attn = rearrange(attn, "b h s d -> b s (h d)")

#         return F.linear(attn, self.out_proj.weight, self.out_proj.bias)

#     def scaled_dot_product_attention(
#         self,
#         query,
#         key,
#         value,
#         attn_mask=None,
#         dropout_p=0.0,
#         is_causal=False,
#         scale=None,
#         enable_gqa=False,
#     ) -> torch.Tensor:
#         L, S = query.size(-2), key.size(-2)
#         scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
#         attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
#         if is_causal:
#             assert attn_mask is None
#             temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
#             attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
#             attn_bias.to(query.dtype)

#         if attn_mask is not None:
#             if attn_mask.dtype == torch.bool:
#                 attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
#             else:
#                 attn_bias = attn_mask + attn_bias

#         if enable_gqa:
#             key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
#             value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)

#         attn_weight = query @ key.transpose(-2, -1) * scale_factor
#         attn_weight += attn_bias
#         attn_weight = torch.softmax(attn_weight, dim=-1)
#         attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
#         return attn_weight @ value


class SelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        rope: Optional[nn.Module] = None,
    ):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.rope = rope
        self.init_tensors()

    def init_tensors(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.constant_(self.in_proj_bias, 0.0)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x, attn_mask=None):
        # breakpoint()
        B, S, _ = x.shape

        # Project input to QKV
        qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias)
        qkv = qkv.view(B, S, 3, self.num_heads, self.head_dim)

        # Unpack q/k for RoPE if used
        if self.rope is not None:
            # Extract Q and K: [B, S, H, D]
            q, k = qkv[:, :, 0], qkv[:, :, 1]

            # Permute to [B, H, S, D] for RoPE
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)

            # Apply RoPE2D (self.freq: [S, D], expand as needed)
            q, k = self.rope(q, k)

            # Permute back to [B, S, H, D]
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)

            # Replace in qkv
            qkv[:, :, 0] = q
            qkv[:, :, 1] = k

            # q, k = qkv[:, :, 0], qkv[:, :, 1]
            # q, k = self.rope(q, k)
            # qkv[:, :, 0] = q
            # qkv[:, :, 1] = k

        # Apply FlashAttention
        attn_output = flash_attn_qkvpacked_func(
            qkv, dropout_p=0.0, causal=False
        )  # Output: [B, S, H * D]
        attn_output = attn_output.reshape(B, S, self.embed_dim)
        # breakpoint()
        return self.out_proj(attn_output)


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        drop_path: float = 0.0,
        rope: Optional[nn.Module] = None,
    ):
        super().__init__()

        if rope:
            self.attn = SelfAttention(d_model, n_head, rope=rope)
        else:
            self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)

        self.ls_1 = (
            LayerScale(d_model, ls_init_value)
            if ls_init_value is not None
            else nn.Identity()
        )
        self.ls_2 = (
            LayerScale(d_model, ls_init_value)
            if ls_init_value is not None
            else nn.Identity()
        )

        self.ln_1 = norm_layer(d_model)
        self.ln_2 = norm_layer(d_model)

        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, mlp_width)),
                    ("gelu", act_layer()),
                    ("c_proj", nn.Linear(mlp_width, d_model)),
                ]
            )
        )

    def _call_attn(
        self,
        q_x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ):

        if attn_mask is not None:
            # Leave boolean masks as is
            if not attn_mask.dtype == torch.bool:
                attn_mask = attn_mask.to(q_x.dtype)

        if isinstance(self.attn, SelfAttention):
            return self.attn(q_x, attn_mask=attn_mask)
        else:
            return self.attn(q_x, q_x, q_x, attn_mask=attn_mask, need_weights=False)[0]

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        x = x + self.drop_path1(
            self.ls_1(self._call_attn(self.ln_1(x), attn_mask=attn_mask))
        )
        x = x + self.drop_path2(self.ls_2(self.mlp(self.ln_2(x))))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        drop_path: float = 0.0,
        rope: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False

        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    width,
                    heads,
                    mlp_ratio,
                    ls_init_value=ls_init_value,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    drop_path=drop_path,
                    rope=rope,
                )
                for _ in range(layers)
            ]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def truncate(self, layer_idx: int):
        """Delete layers so the last layer is the given layer index."""
        self.layers = ((self.layers + layer_idx) % self.layers) + 1
        self.resblocks = nn.ModuleList(self.resblocks[: self.layers])

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        layer_idx: int = -1,
    ):
        stop_idx = (self.layers + layer_idx) % self.layers

        for i, r in enumerate(self.resblocks):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                x = checkpoint(r, x, None, None, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)

            if i == stop_idx:
                break

        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        mlp_ratio: float,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = partial(nn.LayerNorm, eps=1e-5),
        use_ln_pre: bool = True,
        use_ln_post: bool = True,
        ls_init_value: float = None,
        drop_path: float = 0.0,
        image_size: int = 448,  # Pretrain image size only; you can pass in any image size
        use_abs_posemb: bool = True,
        use_rope2d: bool = True,
        use_cls_token: bool = False,
        output_dim: Optional[int] = 1280,
        attn_pooler_heads: int = 8,
        pool_type: Literal["attn", "tok", "avg", "none"] = "attn",
    ):
        super().__init__()
        assert pool_type in ("attn", "tok", "avg", "none")
        self.pool_type = pool_type
        self.patch_size = patch_size

        self.output_dim = output_dim or width
        self.proj_dim = output_dim
        self.heads = heads
        self.width = width
        self.layers = layers

        self.use_abs_posemb = use_abs_posemb
        self.use_cls_token = use_cls_token
        self.use_rope2d = use_rope2d
        self.image_size = image_size
        self.embed_dims = width
        # breakpoint()
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        self.rope = (
            Rope2D(
                dim=width // heads,
                use_cls_token=self.use_cls_token,
            )
            if self.use_rope2d
            else None
        )

        self.ln_pre = norm_layer(width) if use_ln_pre else nn.Identity()
        self.ln_post = norm_layer(self.width) if use_ln_post else nn.Identity()

        self.transformer = Transformer(
            width,
            layers,
            heads,
            mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
            drop_path=drop_path,
            rope=self.rope,
        )

        if pool_type == "attn":
            self.attn_pool = AttentionPooling(
                embed_dim=width,
                num_heads=attn_pooler_heads,
                act_layer=act_layer,
                norm_layer=norm_layer,
            )
        else:
            self.attn_pool = None

        self.init_tensors()

    def init_tensors(self):
        def init_submodule_tensors(module):
            for name, child in module.named_children():
                if hasattr(child, "init_tensors"):
                    logger.debug(f"Initializing tensors for submodule: {name}")
                    child.init_tensors()
                init_submodule_tensors(child)

        init_submodule_tensors(self)
        self.rope.init_tensors()

        # class embeddings and positional embeddings
        init_scale = self.width**-0.5

        if self.use_cls_token:
            self.class_embedding = nn.Parameter(init_scale * torch.randn(self.width))

        if self.use_abs_posemb:
            self.posemb_grid_size = self.image_size // self.patch_size
            self.positional_embedding = nn.Parameter(
                init_scale
                * torch.randn(
                    int(self.use_cls_token) + self.posemb_grid_size**2, self.width
                )
            )

        if self.proj_dim is not None:
            self.proj = nn.Parameter(
                init_scale * torch.randn(self.width, self.proj_dim)
            )

    def load_ckpt(self, ckpt_path: str):
        _sd = torch.load(ckpt_path, weights_only=True)
        if "state_dict" in _sd:
            _sd = _sd["state_dict"]
        elif "weights" in _sd:
            _sd = _sd["weights"]

        # for backwards compatibility
        _sd = {k.replace("module.", ""): v for k, v in _sd.items()}
        if any(k.startswith("visual.") for k in _sd):
            _sd = {k.replace("visual.", ""): v for k, v in _sd.items() if "visual" in k}

        m, u = self.load_state_dict(_sd, strict=False)
        logger.info(f"Missing keys for loading vision encoder: {m}")
        logger.info(f"Unexpected keys for loading vision encoder: {u}")
        print(f"Missing keys for loading vision encoder: {m}")
        print(f"Unexpected keys for loading vision encoder: {u}")

    def truncate(self, layer_idx: int):
        """Delete layers so the last layer is the given layer index."""
        self.transformer.truncate(layer_idx)
        self.layers = self.transformer.layers

    @classmethod
    def from_config(
        cls,
        name: str,
        pretrained: bool = False,
        checkpoint_path: Optional[str] = None,
        **kwdargs,
    ):
        if name not in PE_VISION_CONFIG:
            raise RuntimeError(f"{name} not found in configs.")

        args = asdict(PE_VISION_CONFIG[name])
        args.update(kwdargs)

        model = cls(**args)
        if pretrained:
            model.load_ckpt(fetch_pe_checkpoint(name, checkpoint_path))

        return model

    @classmethod
    def available_configs(cls):
        return list(PE_VISION_CONFIG.keys())

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.set_grad_checkpointing(enable=enable)

    def _sample_abs_posemb(self, grid_h: int, grid_w: int):
        """Interpolates the absolute position embedding if necessary."""
        if self.posemb_grid_size == grid_h and self.posemb_grid_size == grid_w:
            return self.positional_embedding[None, ...]

        pos_embed = self.positional_embedding
        if self.use_cls_token:
            cls_token_embed, pos_embed = pos_embed[:1], pos_embed[1:]

        pos_embed = (
            pos_embed.reshape(1, self.posemb_grid_size, self.posemb_grid_size, -1)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        pos_embed = F.interpolate(
            pos_embed.float(),
            size=(grid_h, grid_w),
            mode="bilinear",
            align_corners=False,
        ).to(pos_embed.dtype)
        pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(-1, self.width).contiguous()

        if self.use_cls_token:
            pos_embed = torch.cat([cls_token_embed, pos_embed], dim=0)

        return pos_embed[None, ...]

    def _pool(self, x: torch.Tensor):
        if self.pool_type == "tok":
            return x[:, 0]
        elif self.pool_type == "avg":
            return x.mean(dim=1)
        elif self.pool_type == "attn":
            return self.attn_pool(x).squeeze(1)
        elif self.pool_type == "none":
            return x
        else:
            raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor,
        norm: bool = False,
        layer_idx: int = -1,
        strip_cls_token: bool = False,
        extra_embed=None,
    ):
        batch, _, h, w = x.shape
        grid_h, grid_w = h // self.patch_size, w // self.patch_size

        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1).reshape(batch, -1, self.width)

        if self.use_cls_token:
            x = torch.cat(
                [self.class_embedding.view(1, 1, -1).expand(batch, -1, -1), x],
                dim=1,
            )

        if self.use_abs_posemb:
            x = x + self._sample_abs_posemb(grid_h, grid_w)

        if self.use_rope2d:
            self.rope.update_grid(x.device, grid_h, grid_w)

        if extra_embed is not None:
            x = x + extra_embed.flatten(2).transpose(1, 2).to(x)

        x = self.ln_pre(x)
        x = self.transformer(x, layer_idx=layer_idx)

        if norm:
            x = self.ln_post(x)

        if strip_cls_token and self.use_cls_token:
            x = x[:, 1:, :]

        return x

    # def forward(self, x: torch.Tensor, **kwargs):
    #     x = self.forward_features(x, norm=True, **kwargs)
    #     x = self._pool(x)

    #     if self.proj_dim is not None:
    #         x = x @ self.proj

    #     return x


class TextTransformer(nn.Module):
    def __init__(
        self,
        context_length: int = 72,
        vocab_size: int = 49408,
        width: int = 512,
        heads: int = 8,
        layers: int = 12,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,
        output_dim: int = 1280,
        no_causal_mask: bool = False,
        pad_id: int = 0,
        pool_type: str = "argmax",
        proj_bias: bool = False,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = partial(nn.LayerNorm, eps=1e-5),
        output_tokens: bool = False,
        use_ln_post: bool = True,
    ):
        super().__init__()
        assert pool_type in ("first", "last", "argmax", "none")
        self.pool_type = pool_type
        self.output_tokens = output_tokens
        self.num_pos = self.context_length = context_length
        self.vocab_size = vocab_size
        self.width = width
        self.output_dim = output_dim
        self.heads = heads
        self.pad_id = pad_id
        self.layers = layers

        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = nn.Parameter(torch.empty(self.num_pos, width))

        self.transformer = Transformer(
            width=width,
            layers=layers,
            heads=heads,
            mlp_ratio=mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

        self.ln_final = norm_layer(width) if use_ln_post else nn.Identity()

        if no_causal_mask:
            self.attn_mask = None
        else:
            self.register_buffer(
                "attn_mask", self.build_causal_mask(), persistent=False
            )

        if pool_type == "attn" or pool_type == "attn_eos":
            self.attn_pool = AttentionPooling(
                embed_dim=width,
                num_heads=heads,
                act_layer=act_layer,
                norm_layer=norm_layer,
            )
        else:  # argmax
            self.attn_pool = None

        if proj_bias:
            self.text_projection = nn.Linear(width, output_dim)
        else:
            self.text_projection = nn.Parameter(torch.empty(width, output_dim))

    def build_causal_mask(self):
        # lazily create causal attention mask, with full attention between the tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.num_pos, self.num_pos)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def load_ckpt(self, ckpt_path: str):
        _sd = torch.load(ckpt_path, weights_only=True)
        if "state_dict" in _sd:
            _sd = _sd["state_dict"]
        elif "weights" in _sd:
            _sd = _sd["weights"]

        _sd = {k.replace("module.", ""): v for k, v in _sd.items()}

        m, u = self.load_state_dict(_sd, strict=False)

        logger.info(f"Missing keys for loading model: {m}")
        logger.info(f"Unexpected keys for loading model: {u}")
        print(f"Missing keys for loading model: {m}")
        print(f"Unexpected keys for loading model: {u}")

    def build_cls_mask(self, text):
        cls_mask = (text != self.pad_id).unsqueeze(1)
        cls_mask = F.pad(cls_mask, (1, 0, cls_mask.shape[2], 0), value=True)
        additive_mask = torch.empty(cls_mask.shape, device=cls_mask.device)
        additive_mask.fill_(0)
        additive_mask.masked_fill_(~cls_mask, float("-inf"))
        additive_mask = torch.repeat_interleave(additive_mask, self.heads, 0)
        return additive_mask

    def text_global_pool(
        self, x, text: Optional[torch.Tensor] = None, pool_type: str = "argmax"
    ):
        if pool_type == "first":
            pooled, tokens = x[:, 0], x[:, 1:]
        elif pool_type == "last":
            pooled, tokens = x[:, -1], x[:, :-1]
        elif pool_type == "argmax":
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            assert text is not None
            pooled, tokens = x[torch.arange(x.shape[0]), text.argmax(dim=-1)], x
        else:
            pooled = tokens = x

        return pooled, tokens

    def forward(self, text):
        seq_len = text.shape[1]
        x = self.token_embedding(text)
        attn_mask = self.attn_mask
        if attn_mask is not None:
            attn_mask = attn_mask[:seq_len, :seq_len]

        x = x + self.positional_embedding[:seq_len]
        x = self.transformer(x, attn_mask=attn_mask)

        x = self.ln_final(x)
        pooled, tokens = self.text_global_pool(x, text, pool_type=self.pool_type)

        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                pooled = self.text_projection(pooled)
            else:
                pooled = pooled @ self.text_projection

        if self.output_tokens:
            return pooled, tokens

        return pooled


class CLIP(TextTransformer):
    def __init__(
        self,
        vision_cfg: PEConfig,
        text_cfg: PETextConfig,
        init_logit_scale: float = np.log(1 / 0.07),
    ):
        super(CLIP, self).__init__(**asdict(text_cfg))
        self.visual = VisionTransformer(**asdict(vision_cfg))
        self.image_size = self.visual.image_size  # For ease of use
        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)

    def encode_image(self, image, normalize: bool = False):
        x = self.visual(image)
        return F.normalize(x, dim=-1) if normalize else x

    def encode_video(self, video, normalize: bool = False):  # b n c h w
        b, n, c, h, w = video.shape
        frms = video.reshape(b * n, c, h, w)
        frm_feats = self.encode_image(frms, normalize=normalize)
        video_feats = frm_feats.reshape(b, n, -1)
        video_feats = video_feats.mean(dim=1)
        return video_feats

    def encode_text(self, text, normalize: bool = False):
        x = super().forward(text)
        return F.normalize(x, dim=-1) if normalize else x

    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
    ):
        image_features = (
            self.encode_image(image, normalize=True) if image is not None else None
        )
        text_features = (
            self.encode_text(text, normalize=True) if text is not None else None
        )
        return image_features, text_features, self.logit_scale.exp()

    @classmethod
    def from_config(
        cls,
        name: str,
        pretrained: bool = False,
        checkpoint_path: Optional[str] = None,  # To load your own
    ):
        if name not in PE_VISION_CONFIG or name not in PE_TEXT_CONFIG:
            raise RuntimeError(f"{name} not found in configs.")

        model = cls(PE_VISION_CONFIG[name], PE_TEXT_CONFIG[name])
        if pretrained:
            model.load_ckpt(fetch_pe_checkpoint(name, checkpoint_path))

        return model

    @classmethod
    def available_configs(cls):
        return [k for k in PE_VISION_CONFIG if k in PE_TEXT_CONFIG]


def PEBackbone(name, cfg):
    model_name = {
        "pe_core_b": "PE-Core-B16-224",
        "pe_core_l": "PE-Core-L14-336",
        "pe_core_g": "PE-Core-G14-448",
        "pe_spatial_g": "PE-Spatial-G14-448",
        "pe_spatial_l": "PE-Spatial-L14-448",
        "pe_spatial_b": "PE-Spatial-B16-512",
        "pe_spatial_s": "PE-Spatial-S16-512",
        "pe_spatial_t": "PE-Spatial-T16-512",
    }
    if name not in model_name:
        raise RuntimeError(f"{name} not found in configs.")

    return VisionTransformer.from_config(
        model_name[name],
        pretrained=True,
        checkpoint_path=f"/checkpoint/3po/model/perception_encoder/{model_name[name]}.pt",
    )
