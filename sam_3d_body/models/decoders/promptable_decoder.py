# Copyright (c) Meta Platforms, Inc. and affiliates.

import pickle
from typing import Dict, Optional

import torch
import torch.nn as nn

from ..modules.transformer import build_norm_layer, TransformerDecoderLayer


class PromptableDecoder(nn.Module):
    """Cross-attention based Transformer decoder with prompts input.

    Args:
        token_dims (int): The dimension of input pose tokens.
        prompt_dims (int): The dimension of input prompt tokens.
        context_dims (int): The dimension of image context features.
        dims (int): The projected dimension of all tokens in the decoder.
        depth (int): The number of layers for Transformer decoder.
        num_heads (int): The number of heads for multi-head attention.
        head_dims (int): The dimension of each head.
        mlp_dims (int): The dimension of hidden layers in MLP.
        layer_scale_init_value (float or torch.Tensor): Init value of layer
            scale. Defaults to 0.
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Defaults to 0.
        attn_drop_rate (float): The drop out rate for attention output weights.
            Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        ffn_type (str): Select the type of ffn layers. Defaults to 'origin'.
        act_layer (nn.Module, optional): The activation layer for FFNs.
            Default: nn.GELU
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        enable_twoway (bool): Whether to enable two-way Transformer (used in SAM).
        repeat_pe (bool): Whether to re-add PE at each layer (used in SAM)
    """

    def __init__(
        self,
        dims: int,
        context_dims: int,
        depth: int,
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
        frozen: bool = False,
        do_interm_preds: bool = False,
        do_keypoint_tokens: bool = False,
        keypoint_token_update: bool = False,
    ):
        super().__init__()

        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(
                TransformerDecoderLayer(
                    token_dims=dims,
                    context_dims=context_dims,
                    num_heads=num_heads,
                    head_dims=head_dims,
                    mlp_dims=mlp_dims,
                    layer_scale_init_value=layer_scale_init_value,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=drop_path_rate,
                    ffn_type=ffn_type,
                    act_layer=act_layer,
                    norm_cfg=norm_cfg,
                    enable_twoway=enable_twoway,
                    repeat_pe=repeat_pe,
                    skip_first_pe=(i == 0),
                )
            )

        self.norm_final = build_norm_layer(norm_cfg, dims)
        self.do_interm_preds = do_interm_preds
        self.do_keypoint_tokens = do_keypoint_tokens
        self.keypoint_token_update = keypoint_token_update

        self.frozen = frozen
        self._freeze_stages()

    def forward(
        self,
        token_embedding: torch.Tensor,
        image_embedding: torch.Tensor,
        token_augment: Optional[torch.Tensor] = None,
        image_augment: Optional[torch.Tensor] = None,
        token_mask: Optional[torch.Tensor] = None,
        channel_first: bool = True,
        token_to_pose_output_fn=None,
        keypoint_token_update_fn=None,
        hand_embeddings=None,
        hand_augment=None,
    ):
        """
        Args:
            token_embedding: [B, N, C]
            image_embedding: [B, C, H, W]
        """
        if channel_first:
            image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
            if image_augment is not None:
                image_augment = image_augment.flatten(2).permute(0, 2, 1)
            if hand_embeddings is not None:
                hand_embeddings = hand_embeddings.flatten(2).permute(0, 2, 1)
                hand_augment = hand_augment.flatten(2).permute(0, 2, 1)
                if len(hand_augment) == 1:
                    # inflate batch dimension
                    assert len(hand_augment.shape) == 3
                    hand_augment = hand_augment.repeat(len(hand_embeddings), 1, 1)

        if self.do_interm_preds:
            assert token_to_pose_output_fn is not None
            all_pose_outputs = []

        for layer_idx, layer in enumerate(self.layers):
            if hand_embeddings is None:
                token_embedding, image_embedding = layer(
                    token_embedding,
                    image_embedding,
                    token_augment,
                    image_augment,
                    token_mask,
                )
            else:
                token_embedding, image_embedding = layer(
                    token_embedding,
                    torch.cat([image_embedding, hand_embeddings], dim=1),
                    token_augment,
                    torch.cat([image_augment, hand_augment], dim=1),
                    token_mask,
                )
                image_embedding = image_embedding[:, : image_augment.shape[1]]

            if self.do_interm_preds and layer_idx < len(self.layers) - 1:
                curr_pose_output = token_to_pose_output_fn(
                    self.norm_final(token_embedding),
                    prev_pose_output=(
                        all_pose_outputs[-1] if len(all_pose_outputs) > 0 else None
                    ),
                    layer_idx=layer_idx,
                )
                all_pose_outputs.append(curr_pose_output)

                if self.keypoint_token_update:
                    assert keypoint_token_update_fn is not None
                    token_embedding, token_augment, _, _ = keypoint_token_update_fn(
                        token_embedding, token_augment, curr_pose_output, layer_idx
                    )

        out = self.norm_final(token_embedding)

        if self.do_interm_preds:
            curr_pose_output = token_to_pose_output_fn(
                out,
                prev_pose_output=(
                    all_pose_outputs[-1] if len(all_pose_outputs) > 0 else None
                ),
                layer_idx=layer_idx,
            )
            all_pose_outputs.append(curr_pose_output)

            return out, all_pose_outputs
        else:
            return out

    def _freeze_stages(self):
        """Freeze parameters."""
        if self.frozen:
            for layer in self.layers:
                layer.eval()
            self.norm_final.eval()
            for param in self.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        """
        Convert the model into training mode.
        (not called by lightning in trainer.fit() actually)
        """
        super().train(mode)
        self._freeze_stages()
