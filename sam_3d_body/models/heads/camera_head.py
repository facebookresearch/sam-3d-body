# Copyright (c) Meta Platforms, Inc. and affiliates.

from typing import Optional, Tuple

import torch
import torch.nn as nn

from sam_3d_body.models.modules.geometry_utils import perspective_projection

from ..modules import get_intrinsic_matrix, to_2tuple
from ..modules.transformer import FFN


class PerspectiveHead(nn.Module):
    """
    Predict camera translation (s, tx, ty) and perform full-perspective
    2D reprojection (CLIFF/CameraHMR setup).
    """

    def __init__(
        self,
        input_dim: int,
        img_size: Tuple[int, int],  # model input size (W, H)
        mlp_depth: int = 1,
        drop_ratio: float = 0.0,
        mlp_channel_div_factor: int = 8,
        default_scale_factor: float = 1,
    ):
        super().__init__()

        # Metadata to compute 3D skeleton and 2D reprojection
        self.img_size = to_2tuple(img_size)
        self.ncam = 3  # (s, tx, ty)
        self.default_scale_factor = default_scale_factor

        self.proj = FFN(
            embed_dims=input_dim,
            feedforward_channels=input_dim // mlp_channel_div_factor,
            output_dims=self.ncam,
            num_fcs=mlp_depth,
            ffn_drop=drop_ratio,
            add_identity=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        init_estimate: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            x: pose token with shape [B, C], usually C=DECODER.DIM
            init_estimate: [B, self.ncam]
        """
        pred_cam = self.proj(x)
        if init_estimate is not None:
            pred_cam = pred_cam + init_estimate

        return pred_cam

    def perspective_projection(
        self,
        points_3d: torch.Tensor,
        pred_cam: torch.Tensor,
        bbox_center: torch.Tensor,
        bbox_size: torch.Tensor,
        img_size: torch.Tensor,
        cam_int: torch.Tensor,
        use_intrin_center: bool = False,
    ):
        """
        Args:
            bbox_center / img_size: shape [N, 2], in original image space (w, h)
            bbox_size: shape [N,], in original image space
            cam_int: shape [N, 3, 3]
        """
        batch_size = points_3d.shape[0]
        pred_cam = pred_cam.clone()
        pred_cam[..., [0, 2]] *= -1  # Camera system difference

        # Compute camera translation: (scale, x, y) --> (x, y, depth)
        # depth ~= f / s
        # Note that f is in the NDC space (see Zolly section 3.1)
        s, tx, ty = pred_cam[:, 0], pred_cam[:, 1], pred_cam[:, 2]
        bs = bbox_size * s * self.default_scale_factor + 1e-8
        focal_length = cam_int[:, 0, 0]
        tz = 2 * focal_length / bs

        if not use_intrin_center:
            cx = 2 * (bbox_center[:, 0] - (img_size[:, 0] / 2)) / bs
            cy = 2 * (bbox_center[:, 1] - (img_size[:, 1] / 2)) / bs
        else:
            cx = 2 * (bbox_center[:, 0] - (cam_int[:, 0, 2])) / bs
            cy = 2 * (bbox_center[:, 1] - (cam_int[:, 1, 2])) / bs

        pred_cam_t = torch.stack([tx + cx, ty + cy, tz], dim=-1)

        # Compute camera translation
        j3d_cam = points_3d + pred_cam_t.unsqueeze(1)

        # Projection to the image plane.
        # Note that the projection output is in *original* image space now.
        j2d = perspective_projection(j3d_cam, cam_int)

        return {
            "pred_keypoints_2d": j2d.reshape(batch_size, -1, 2),
            "pred_cam_t": pred_cam_t,
            "focal_length": focal_length,
            "pred_keypoints_2d_depth": j3d_cam.reshape(batch_size, -1, 3)[:, :, 2],
        }
