from collections import namedtuple
import math
import torch

from pytorch3d.transforms import (
    Transform3d,
    axis_angle_to_quaternion,
)

DecomposedTransform = namedtuple(
    "DecomposedTransform", ["scale", "rotation", "translation"]
)


def compose_transform(
    scale: torch.Tensor, rotation: torch.Tensor, translation: torch.Tensor
) -> Transform3d:
    """
    Args:
        scale: (..., 3) tensor of scale factors
        rotation: (..., 3, 3) tensor of rotation matrices
        translation: (..., 3) tensor of translation vectors
    """
    tfm = Transform3d(dtype=scale.dtype, device=scale.device)
    return tfm.scale(scale).rotate(rotation).translate(translation)


def decompose_transform(transform: Transform3d) -> DecomposedTransform:
    """
    Returns:
        scale: (..., 3) tensor of scale factors
        rotation: (..., 3, 3) tensor of rotation matrices
        translation: (..., 3) tensor of translation vectors
    """
    matrices = transform.get_matrix()
    scale = torch.norm(matrices[:, :3, :3], dim=-1)
    rotation = matrices[:, :3, :3] / scale.unsqueeze(-1)  # Normalize rotation matrix
    translation = matrices[:, 3, :3]  # Extract translation vector
    return DecomposedTransform(scale, rotation, translation)


def get_rotation_about_x_axis(angle: float = math.pi / 2) -> torch.Tensor:
    axis = torch.tensor([1.0, 0.0, 0.0])
    axis_angle = axis * angle
    return axis_angle_to_quaternion(axis_angle)
