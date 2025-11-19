# Copyright (c) Meta Platforms, Inc. and affiliates.

from typing import Any, Optional, Tuple

import numpy as np

import torch
import torch.nn as nn

from sam_3d_body.models.modules.transformer import LayerNorm2d


class PromptEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_body_joints: int,
        # img_size: Tuple[int, int],
        # patch_resolution: Tuple[int, int],
        frozen: bool = False,
        mask_embed_type: Optional[str] = None,
    ) -> None:
        """
        Encodes prompts for input to SAM's mask decoder.

        Arguments:
            embed_dim (int): The prompts' embedding dimension
            num_body_joints (int): The number of body joints
            img_size (Tuple): The padded size of the image as input
                to the image encoder, as (H, W).
            patch_resolution (Tuple): image path size, as (H, W)
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_body_joints = num_body_joints
        # self.img_size = img_size
        # self.patch_resolution = patch_resolution

        # Keypoint prompts
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)
        self.point_embeddings = nn.ModuleList(
            [nn.Embedding(1, embed_dim) for _ in range(self.num_body_joints)]
        )
        self.not_a_point_embed = nn.Embedding(1, embed_dim)
        self.invalid_point_embed = nn.Embedding(1, embed_dim)

        # Mask prompt
        if mask_embed_type in ["v1"]:
            mask_in_chans = 16  # SAM2
            self.mask_downscaling = nn.Sequential(
                nn.Conv2d(1, mask_in_chans // 4, kernel_size=4, stride=4),
                LayerNorm2d(mask_in_chans // 4),
                nn.GELU(),
                nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=4, stride=4),
                LayerNorm2d(mask_in_chans),
                nn.GELU(),
                nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
            )
        elif mask_embed_type in ["v2"]:
            mask_in_chans = 256
            self.mask_downscaling = nn.Sequential(
                nn.Conv2d(1, mask_in_chans // 64, kernel_size=2, stride=2),
                LayerNorm2d(mask_in_chans // 64),
                nn.GELU(),
                nn.Conv2d(
                    mask_in_chans // 64,
                    mask_in_chans // 16,
                    kernel_size=2,
                    stride=2,
                ),
                LayerNorm2d(mask_in_chans // 16),
                nn.GELU(),
                nn.Conv2d(
                    mask_in_chans // 16, mask_in_chans // 4, kernel_size=2, stride=2
                ),
                LayerNorm2d(mask_in_chans // 4),
                nn.GELU(),
                nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
                LayerNorm2d(mask_in_chans),
                nn.GELU(),
                nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
            )
        else:
            assert mask_embed_type is None

        if mask_embed_type is not None:
            # Zero-initialize the last conv layer as gating
            nn.init.zeros_(self.mask_downscaling[-1].weight)
            nn.init.zeros_(self.mask_downscaling[-1].bias)

            self.no_mask_embed = nn.Embedding(1, embed_dim)
            nn.init.zeros_(self.no_mask_embed.weight)

        self.frozen = frozen
        self._freeze_stages()

    def get_dense_pe(self, size: Tuple[int, int]) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(size).unsqueeze(0)

    def _embed_keypoints(
        self,
        points: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Embeds point prompts.
        Assuming points have been normalized to [0, 1].

        Output shape [B, N, C], mask shape [B, N]
        """
        assert points.min() >= 0 and points.max() <= 1
        point_embedding = self.pe_layer._pe_encoding(points.to(torch.float))
        point_embedding[labels == -2] = 0.0  # invalid points
        point_embedding[labels == -2] += self.invalid_point_embed.weight
        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        for i in range(self.num_body_joints):
            point_embedding[labels == i] += self.point_embeddings[i].weight

        point_mask = labels > -2
        return point_embedding, point_mask

    def _get_batch_size(
        self,
        keypoints: Optional[torch.Tensor],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> int:
        """
        Gets the batch size of the output given the batch size of the input prompts.
        """
        if keypoints is not None:
            return keypoints.shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        else:
            return 1

    def _get_device(self) -> torch.device:
        return self.point_embeddings[0].weight.device

    def forward(
        self,
        keypoints: Optional[torch.Tensor],
        boxes: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embeds different types of prompts, returning both sparse and dense
        embeddings.

        Arguments:
          keypoints (torchTensor or none): point coordinates and labels to embed.
          boxes (torch.Tensor or none): boxes to embed
          masks (torch.Tensor or none): masks to embed

        Returns:
          torch.Tensor: sparse embeddings for the points and boxes, with shape
            BxNx(embed_dim), where N is determined by the number of input points
            and boxes.
          torch.Tensor: dense embeddings for the masks, in the shape
            Bx(embed_dim)x(embed_H)x(embed_W)
        """
        bs = self._get_batch_size(keypoints, boxes, masks)
        sparse_embeddings = torch.empty(
            (bs, 0, self.embed_dim), device=self._get_device()
        )
        sparse_masks = torch.empty((bs, 0), device=self._get_device())
        if keypoints is not None:
            coords = keypoints[:, :, :2]
            labels = keypoints[:, :, -1]
            point_embeddings, point_mask = self._embed_keypoints(
                coords, labels
            )  # pad=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
            sparse_masks = torch.cat([sparse_masks, point_mask], dim=1)

        return sparse_embeddings, sparse_masks

    def get_mask_embeddings(
        self,
        masks: Optional[torch.Tensor] = None,
        bs: int = 1,
        size: Tuple[int, int] = (16, 16),  # [H, W]
    ) -> torch.Tensor:
        """Embeds mask inputs."""
        no_mask_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, size[0], size[1]
        )
        if masks is not None:
            mask_embeddings = self.mask_downscaling(masks)
        else:
            mask_embeddings = no_mask_embeddings
        return mask_embeddings, no_mask_embeddings

    def _freeze_stages(self):
        """Freeze parameters."""
        if self.frozen:
            for param in self.parameters():
                param.requires_grad = False


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C
