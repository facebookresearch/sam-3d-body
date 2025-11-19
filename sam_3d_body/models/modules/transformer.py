# Copyright (c) Meta Platforms, Inc. and affiliates.

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .drop_path import DropPath

from .layer_scale import LayerScale
from .swiglu_ffn import SwiGLUFFNFused


class MLP(nn.Module):
    # borrowed from DET R
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class LayerNorm32(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)


def build_norm_layer(cfg: Dict, num_features: int):
    """Build normalization layer.

    Args:
        cfg (dict): The norm layer config, which should contain:

            - type (str): Layer type.
            - layer args: Args needed to instantiate a norm layer.
            - requires_grad (bool, optional): Whether stop gradient updates.
        num_features (int): Number of input channels.
        postfix (int | str): The postfix to be appended into norm abbreviation
            to create named layer.

    Returns:
        tuple[str, nn.Module]: The first element is the layer name consisting
        of abbreviation and postfix, e.g., bn1, gn. The second element is the
        created norm layer.
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be a dict")
    if "type" not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')
    cfg_ = cfg.copy()

    layer_type = cfg_.pop("type")
    if layer_type == "LN":
        norm_layer = LayerNorm32
    else:
        raise ValueError("Unsupported norm layer: ", layer_type)

    requires_grad = cfg_.pop("requires_grad", True)
    cfg_.setdefault("eps", 1e-5)
    if norm_layer is not nn.GroupNorm:
        layer = norm_layer(num_features, **cfg_)
        if layer_type == "SyncBN" and hasattr(layer, "_specify_ddp_gpu_num"):
            layer._specify_ddp_gpu_num(1)
    else:
        assert "num_groups" in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)

    for param in layer.parameters():
        param.requires_grad = requires_grad

    return layer


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


class FFN(nn.Module):
    """Implements feed-forward networks (FFNs) with identity connection.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`. Defaults: 256.
        feedforward_channels (int): The hidden dimension of FFNs.
            Defaults: 1024.
        num_fcs (int, optional): The number of fully-connected layers in
            FFNs. Default: 2.
        act_layer (nn.Module, optional): The activation layer for FFNs.
            Default: nn.ReLU
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
        add_identity (bool, optional): Whether to add the
            identity connection. Default: `True`.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        layer_scale_init_value (float): Initial value of scale factor in
            LayerScale. Default: 1.0
    """

    # @deprecated_api_warning(
    #     {
    #         'dropout': 'ffn_drop',
    #         'add_residual': 'add_identity'
    #     },
    #     cls_name='FFN')
    def __init__(
        self,
        embed_dims=256,
        feedforward_channels=1024,
        output_dims=None,
        num_fcs=2,
        act_layer=nn.ReLU,
        ffn_drop=0.0,
        drop_path_rate=0.0,
        add_identity=True,
        layer_scale_init_value=0.0,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.output_dims = output_dims or embed_dims
        self.num_fcs = num_fcs

        layers = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(
                nn.Sequential(
                    nn.Linear(in_channels, feedforward_channels),
                    act_layer(),
                    nn.Dropout(ffn_drop),
                )
            )
            in_channels = feedforward_channels
        layers.append(nn.Linear(in_channels, self.output_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = nn.Sequential(*layers)
        self.dropout_layer = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else torch.nn.Identity()
        )
        self.add_identity = add_identity

        if layer_scale_init_value > 0:
            self.gamma2 = LayerScale(embed_dims, scale=layer_scale_init_value)
        else:
            self.gamma2 = nn.Identity()

    # @deprecated_api_warning({'residual': 'identity'}, cls_name='FFN')
    def forward(self, x, identity=None):
        """Forward function for `FFN`.

        The function would add x to the output tensor if residue is None.
        """
        out = self.layers(x)
        out = self.gamma2(out)
        if not self.add_identity:
            return self.dropout_layer(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)


class MultiheadAttention(nn.Module):
    """Multi-head Attention Module.

    This module implements multi-head attention that supports different input
    dims and embed dims. And it also supports a shortcut from ``value``, which
    is useful if input dims is not the same with embed dims.

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        input_dims (int, optional): The input dimension, and if None,
            use ``embed_dims``. Defaults to None.
        attn_drop (float): Dropout rate of the dropout layer after the
            attention calculation of query and key. Defaults to 0.
        proj_drop (float): Dropout rate of the dropout layer after the
            output projection. Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        qkv_bias (bool): If True, add a learnable bias to q, k, v.
            Defaults to True.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        proj_bias (bool) If True, add a learnable bias to output projection.
            Defaults to True.
        v_shortcut (bool): Add a shortcut from value to output. It's usually
            used if ``input_dims`` is different from ``embed_dims``.
            Defaults to False.
        use_layer_scale (bool): Whether to use layer scale. Defaults to False.
        layer_scale_init_value (float or torch.Tensor): Init value of layer
            scale. Defaults to 0.
    """

    def __init__(
        self,
        embed_dims,
        num_heads,
        input_dims=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path_rate=0.0,
        qkv_bias=True,
        proj_bias=True,
        v_shortcut=False,
        layer_scale_init_value=0.0,
    ):
        super().__init__()

        self.input_dims = input_dims or embed_dims
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.v_shortcut = v_shortcut

        self.head_dims = embed_dims // num_heads

        self.qkv = nn.Linear(self.input_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(embed_dims, embed_dims, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.out_drop = DropPath(drop_path_rate)

        if layer_scale_init_value > 0:
            layer_scale_init_value = layer_scale_init_value or 1e-5
            self.gamma1 = LayerScale(
                embed_dims, layer_scale_init_value=layer_scale_init_value
            )
        else:
            self.gamma1 = nn.Identity()

    def forward(self, x):
        B, N, _ = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dims)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_drop = self.attn_drop if self.training else 0.0
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=attn_drop)
        x = x.transpose(1, 2).reshape(B, N, self.embed_dims)

        x = self.proj(x)
        x = self.out_drop(self.gamma1(self.proj_drop(x)))

        if self.v_shortcut:
            x = v.squeeze(1) + x
        return x


class Attention(nn.Module):
    """Multi-head Attention Module for both self and cross attention.

    Support masking invalid elements for attention.

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        input_dims (int, optional): The input dimension, and if None,
            use ``embed_dims``. Defaults to None.
        attn_drop (float): Dropout rate of the dropout layer after the
            attention calculation of query and key. Defaults to 0.
        proj_drop (float): Dropout rate of the dropout layer after the
            output projection. Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        qkv_bias (bool): If True, add a learnable bias to q, k, v.
            Defaults to True.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        proj_bias (bool) If True, add a learnable bias to output projection.
            Defaults to True.
        v_shortcut (bool): Add a shortcut from value to output. It's usually
            used if ``input_dims`` is different from ``embed_dims``.
            Defaults to False.
        use_layer_scale (bool): Whether to use layer scale. Defaults to False.
        layer_scale_init_value (float or torch.Tensor): Init value of layer
            scale. Defaults to 0.
    """

    def __init__(
        self,
        embed_dims,
        num_heads,
        query_dims=None,
        key_dims=None,
        value_dims=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path_rate=0.0,
        qkv_bias=True,
        proj_bias=True,
        v_shortcut=False,
        layer_scale_init_value=0.0,
    ):
        super().__init__()

        self.query_dims = query_dims or embed_dims
        self.key_dims = key_dims or embed_dims
        self.value_dims = value_dims or embed_dims
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.v_shortcut = v_shortcut

        self.head_dims = embed_dims // num_heads

        self.q_proj = nn.Linear(self.query_dims, embed_dims, bias=qkv_bias)
        self.k_proj = nn.Linear(self.key_dims, embed_dims, bias=qkv_bias)
        self.v_proj = nn.Linear(self.value_dims, embed_dims, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(embed_dims, self.query_dims, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.out_drop = DropPath(drop_path_rate)

        if layer_scale_init_value > 0:
            layer_scale_init_value = layer_scale_init_value or 1e-5
            self.gamma1 = LayerScale(
                embed_dims, layer_scale_init_value=layer_scale_init_value
            )
        else:
            self.gamma1 = nn.Identity()

    def _separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        x = x.reshape(b, n, self.num_heads, self.head_dims)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        B, N, _ = q.shape
        q = self._separate_heads(self.q_proj(q))
        k = self._separate_heads(self.k_proj(k))
        v = self._separate_heads(self.v_proj(v))

        attn_drop = self.attn_drop if self.training else 0.0
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=attn_drop
        )
        x = x.transpose(1, 2).reshape(B, N, self.embed_dims)

        x = self.proj(x)
        x = self.out_drop(self.gamma1(self.proj_drop(x)))

        if self.v_shortcut:
            x = v.squeeze(1) + x
        return x


class TransformerEncoderLayer(nn.Module):
    """Implements one encoder layer in Vision Transformer.

    Args:
        embed_dims (int): The feature dimension
        num_heads (int): Parallel attention heads
        feedforward_channels (int): The hidden dimension for FFNs
        layer_scale_init_value (float or torch.Tensor): Init value of layer
            scale. Defaults to 0.
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Defaults to 0.
        attn_drop_rate (float): The drop out rate for attention output weights.
            Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Defaults to 2.
        qkv_bias (bool): enable bias for qkv if True. Defaults to True.
        ffn_type (str): Select the type of ffn layers. Defaults to 'origin'.
        act_layer (nn.Module, optional): The activation layer for FFNs.
            Default: nn.GELU
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
    """

    def __init__(
        self,
        embed_dims,
        num_heads,
        feedforward_channels,
        layer_scale_init_value=0.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        num_fcs=2,
        qkv_bias=True,
        ffn_type="origin",
        act_layer=nn.GELU,
        norm_cfg=dict(type="LN", eps=1e-6),
    ):
        super().__init__()

        self.embed_dims = embed_dims

        self.ln1 = build_norm_layer(norm_cfg, self.embed_dims)

        self.attn = MultiheadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            drop_path_rate=drop_path_rate,
            qkv_bias=qkv_bias,
            layer_scale_init_value=layer_scale_init_value,
        )

        self.ln2 = build_norm_layer(norm_cfg, self.embed_dims)

        if ffn_type == "origin":
            self.ffn = FFN(
                embed_dims=embed_dims,
                feedforward_channels=feedforward_channels,
                num_fcs=num_fcs,
                ffn_drop=drop_rate,
                drop_path_rate=drop_path_rate,
                act_layer=act_layer,
                layer_scale_init_value=layer_scale_init_value,
            )
        elif ffn_type == "swiglu_fused":
            self.ffn = SwiGLUFFNFused(
                embed_dims=embed_dims,
                feedforward_channels=feedforward_channels,
                layer_scale_init_value=layer_scale_init_value,
            )
        else:
            raise NotImplementedError

    @property
    def norm1(self):
        return self.ln1

    @property
    def norm2(self):
        return self.ln2

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = self.ffn(self.ln2(x), identity=x)
        return x


class TransformerDecoderLayer(nn.Module):
    """Implements one decoder layer in cross-attention Transformer.

    Adapted from Segment Anything Model (SAM) implementation.

    Args:
        embed_dims (int): The feature dimension
        num_heads (int): Parallel attention heads
        feedforward_channels (int): The hidden dimension for FFNs
        layer_scale_init_value (float or torch.Tensor): Init value of layer
            scale. Defaults to 0.
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Defaults to 0.
        attn_drop_rate (float): The drop out rate for attention output weights.
            Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Defaults to 2.
        qkv_bias (bool): enable bias for qkv if True. Defaults to True.
        ffn_type (str): Select the type of ffn layers. Defaults to 'origin'.
        act_layer (nn.Module, optional): The activation layer for FFNs.
            Default: nn.GELU
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        enable_twoway (bool): Whether to enable two-way Transformer (used in SAM).
        repeat_pe (bool): Whether to re-add PE at each layer (used in SAM)
        skip_first_pe (bool)
    """

    def __init__(
        self,
        token_dims: int,
        context_dims: int,
        num_heads: int = 8,
        head_dims: int = 64,
        mlp_dims: int = 1024,
        layer_scale_init_value: float = 0.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        ffn_type: str = "origin",
        act_layer: nn.Module = nn.GELU,
        norm_cfg: Dict = dict(type="LN", eps=1e-6),
        enable_twoway: bool = False,
        repeat_pe: bool = False,
        skip_first_pe: bool = False,
    ):
        super().__init__()
        self.repeat_pe = repeat_pe
        self.skip_first_pe = skip_first_pe
        if self.repeat_pe:
            self.ln_pe_1 = build_norm_layer(norm_cfg, token_dims)
            self.ln_pe_2 = build_norm_layer(norm_cfg, context_dims)

        self.ln1 = build_norm_layer(norm_cfg, token_dims)

        self.self_attn = Attention(
            embed_dims=num_heads * head_dims,
            num_heads=num_heads,
            query_dims=token_dims,
            key_dims=token_dims,
            value_dims=token_dims,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
        )

        self.ln2_1 = build_norm_layer(norm_cfg, token_dims)
        self.ln2_2 = build_norm_layer(norm_cfg, context_dims)

        self.cross_attn = Attention(
            embed_dims=num_heads * head_dims,
            num_heads=num_heads,
            query_dims=token_dims,
            key_dims=context_dims,
            value_dims=context_dims,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
        )

        self.ln3 = build_norm_layer(norm_cfg, token_dims)

        if ffn_type == "origin":
            self.ffn = FFN(
                embed_dims=token_dims,
                feedforward_channels=mlp_dims,
                ffn_drop=drop_rate,
                drop_path_rate=drop_path_rate,
                act_layer=act_layer,
                layer_scale_init_value=layer_scale_init_value,
            )
        elif ffn_type == "swiglu_fused":
            self.ffn = SwiGLUFFNFused(
                embed_dims=token_dims,
                feedforward_channels=mlp_dims,
                layer_scale_init_value=layer_scale_init_value,
            )
        else:
            raise NotImplementedError

        self.enable_twoway = enable_twoway
        if self.enable_twoway:
            self.ln4_1 = build_norm_layer(norm_cfg, context_dims)
            self.ln4_2 = build_norm_layer(norm_cfg, token_dims)

            self.cross_attn_2 = Attention(
                embed_dims=num_heads * head_dims,
                num_heads=num_heads,
                query_dims=context_dims,
                key_dims=token_dims,
                value_dims=token_dims,
                attn_drop=attn_drop_rate,
                proj_drop=drop_rate,
                drop_path_rate=drop_path_rate,
                layer_scale_init_value=layer_scale_init_value,
            )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        x_pe: Optional[torch.Tensor] = None,
        context_pe: Optional[torch.Tensor] = None,
        x_mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            x: shape [B, N, C]
            context: shape [B, N, C]
            x_mask: shape [B, N]
        """
        if self.repeat_pe and context_pe is not None:
            # LaPE: https://openaccess.thecvf.com/content/ICCV2023/papers/Yu_LaPE_Layer-adaptive_Position_Embedding_for_Vision_Transformers_with_Independent_Layer_ICCV_2023_paper.pdf
            x_pe = self.ln_pe_1(x_pe)
            context_pe = self.ln_pe_2(context_pe)

        # Self attention block for tokens
        if self.repeat_pe and not self.skip_first_pe and x_pe is not None:
            q = k = self.ln1(x) + x_pe
            v = self.ln1(x)
        else:
            q = k = v = self.ln1(x)

        attn_mask = None
        if x_mask is not None:
            attn_mask = x_mask[:, :, None] @ x_mask[:, None, :]
            # Set diagonal to 1 to prevent nan output
            attn_mask.diagonal(dim1=1, dim2=2).fill_(1)
            attn_mask = attn_mask > 0
        x = x + self.self_attn(q=q, k=k, v=v, attn_mask=attn_mask)

        # Cross attention block, tokens attending to image embedding
        if self.repeat_pe and context_pe is not None:
            q = self.ln2_1(x) + x_pe
            k = self.ln2_2(context) + context_pe
            v = self.ln2_2(context)
        else:
            q = self.ln2_1(x)
            k = v = self.ln2_2(context)
        x = x + self.cross_attn(q=q, k=k, v=v)

        # MLP block
        x = self.ffn(self.ln3(x), identity=x)

        # (Optional) Cross attention block, image embeddings attending to tokens
        if self.enable_twoway:
            if self.repeat_pe and context_pe is not None:
                q = self.ln4_1(context) + context_pe
                k = self.ln4_2(x) + x_pe
                v = self.ln4_2(x)
            else:
                q = self.ln4_1(context)
                k = v = self.ln4_2(x)
            attn_mask = (
                (x_mask[:, None, :].repeat(1, context.shape[1], 1)) > 0
                if x_mask is not None
                else None
            )
            context = context + self.cross_attn_2(q=q, k=k, v=v, attn_mask=attn_mask)

        return x, context
