# Copyright (c) Meta Platforms, Inc. and affiliates.

from typing import Optional

import cv2

import numpy as np
import torch
from torch.nn import functional as F


def cam_crop_to_full(cam_bbox, box_center, box_size, img_size, focal_length=5000.0):
    # Convert cam_bbox to full image
    img_w, img_h = img_size[:, 0], img_size[:, 1]
    cx, cy, b = box_center[:, 0], box_center[:, 1], box_size
    w_2, h_2 = img_w / 2.0, img_h / 2.0
    bs = b * cam_bbox[:, 0] + 1e-9
    if type(focal_length) is float:
        focal_length = torch.ones_like(cam_bbox[:, 0]) * focal_length
    tz = 2 * focal_length / bs
    tx = (2 * (cx - w_2) / bs) + cam_bbox[:, 1]
    ty = (2 * (cy - h_2) / bs) + cam_bbox[:, 2]
    full_cam = torch.stack([tx, ty, tz], dim=-1)
    return full_cam


def aa_to_rotmat(theta: torch.Tensor):
    """
    Convert axis-angle representation to rotation matrix.
    Works by first converting it to a quaternion.
    Args:
        theta (torch.Tensor): Tensor of shape (B, 3) containing axis-angle representations.
    Returns:
        torch.Tensor: Corresponding rotation matrices with shape (B, 3, 3).

    Alternatives:
        import roma
        y = roma.rotvec_to_rotmat(x)
    """
    norm = torch.norm(theta + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim=1)
    return _quat_to_rotmat(quat)


def _quat_to_rotmat(quat: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion representation to rotation matrix.
    Args:
        quat (torch.Tensor) of shape (B, 4); 4 <===> (w, x, y, z).
    Returns:
        torch.Tensor: Corresponding rotation matrices with shape (B, 3, 3).
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack(
        [
            w2 + x2 - y2 - z2,
            2 * xy - 2 * wz,
            2 * wy + 2 * xz,
            2 * wz + 2 * xy,
            w2 - x2 + y2 - z2,
            2 * yz - 2 * wx,
            2 * xz - 2 * wy,
            2 * wx + 2 * yz,
            w2 - x2 - y2 + z2,
        ],
        dim=1,
    ).view(B, 3, 3)
    return rotMat


def rot6d_to_rotmat(x: torch.Tensor) -> torch.Tensor:
    """
    Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Args:
        x (torch.Tensor): (B,6) Batch of 6-D rotation representations.
    Returns:
        torch.Tensor: Batch of corresponding rotation matrices with shape (B,3,3).

    Alternatives:
        import roma
        x = x.reshape(-1,2,3).permute(0, 2, 1).contiguous()
        y = roma.special_gramschmidt(x)
    """
    x = x.reshape(-1, 2, 3).permute(0, 2, 1).contiguous()
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum("bi,bi->b", b1, a2).unsqueeze(-1) * b1)
    b3 = torch.linalg.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)


def rotmat_to_rot6d(x: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        x: batch of rotation matrices of size (B, 3, 3)

    Returns:
        6D rotation representation, of size (B, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    batch_dim = x.size()[:-2]
    return x[..., :2, :].clone().reshape(batch_dim + (6,))


def rot_aa(aa: np.array, rot: float) -> np.array:
    """
    Rotate axis angle parameters.
    Args:
        aa (np.array): Axis-angle vector of shape (3,).
        rot (np.array): Rotation angle in degrees.
    Returns:
        np.array: Rotated axis-angle vector.
    """
    # pose parameters
    R = np.array(
        [
            [np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0],
            [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
            [0, 0, 1],
        ]
    )
    # find the rotation of the body in camera frame
    per_rdg, _ = cv2.Rodrigues(aa)
    # apply the global rotation to the global orientation
    resrot, _ = cv2.Rodrigues(np.dot(R, per_rdg))
    aa = (resrot.T)[0]
    return aa.astype(np.float32)


def transform_points(
    points: torch.Tensor,
    translation: Optional[torch.Tensor] = None,
    rotation: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Transform a set of 3D points given translation and rotation.
    Args:
        points (torch.Tensor): Tensor of shape (B, N, 3) containing the input 3D points.
        translation (torch.Tensor): Tensor of shape (B, 3) containing the 3D camera translation.
        rotation (torch.Tensor): Tensor of shape (B, 3, 3) containing the camera rotation.
    Returns:
        torch.Tensor: Tensor of shape (B, N, 3) containing the transformed points.
    """
    if rotation is not None:
        points = torch.einsum("bij,bkj->bki", rotation, points)

    if translation is not None:
        points = points + translation.unsqueeze(1)

    return points


def get_intrinsic_matrix(
    focal_length: torch.Tensor, principle: torch.Tensor
) -> torch.Tensor:
    """
    Populate intrinsic camera matrix K given focal length and principle point.
    Args:
        focal_length: Tensor of shape (2,)
        principle: Tensor of shape (2,)
    Returns:
        Tensor of shape (3, 3)
    """
    if isinstance(focal_length, float):
        fl_x = fl_y = focal_length
    elif len(focal_length) == 1:
        fl_x = fl_y = focal_length[0]
    else:
        fl_x, fl_y = focal_length[0], focal_length[1]
    K = torch.eye(3)
    K[0, 0] = fl_x
    K[1, 1] = fl_y
    K[0, -1] = principle[0]
    K[1, -1] = principle[1]

    return K


def perspective_projection(x, K):
    """
    Computes the perspective projection of a set of points assuming the extrinsinc params have already been applied
    Args:
        - x [bs,N,3]: 3D points
        - K [bs,3,3]: Camera instrincs params
    """
    # Apply perspective distortion
    y = x / x[:, :, -1].unsqueeze(-1)  # (bs, N, 3)

    # Apply camera intrinsics
    y = torch.einsum("bij,bkj->bki", K, y)  # (bs, N, 3)

    return y[:, :, :2]


def inverse_perspective_projection(points, K, distance):
    """
    Computes the inverse perspective projection of a set of points given an estimated distance.
    Input:
        points (bs, N, 2): 2D points
        K (bs,3,3): camera intrinsics params
        distance (bs, N, 1): distance in the 3D world
    Similar to:
        - pts_l_norm = cv2.undistortPoints(np.expand_dims(pts_l, axis=1), cameraMatrix=K_l, distCoeffs=None)
    """
    # Apply camera intrinsics
    points = torch.cat([points, torch.ones_like(points[..., :1])], -1)
    points = torch.einsum("bij,bkj->bki", torch.inverse(K), points)

    # Apply perspective distortion
    if distance == None:
        return points
    points = points * distance
    return points


def get_cam_intrinsics(img_size, fov=55, p_x=None, p_y=None):
    """Given image size, fov and principal point coordinates, return K the camera parameter matrix"""
    K = np.eye(3)
    # Get focal length.
    focal = get_focalLength_from_fieldOfView(fov=fov, img_size=img_size)
    K[0, 0], K[1, 1] = focal, focal

    # Set principal point
    if p_x is not None and p_y is not None:
        K[0, -1], K[1, -1] = p_x * img_size, p_y * img_size
    else:
        K[0, -1], K[1, -1] = img_size // 2, img_size // 2

    return K


def get_focalLength_from_fieldOfView(fov=60, img_size=512):
    """
    Compute the focal length of the camera lens by assuming a certain FOV for the entire image
    Args:
        - fov: float, expressed in degree
        - img_size: int
    Return:
        focal: float
    """
    focal = img_size / (2 * np.tan(np.radians(fov) / 2))
    return focal


def focal_length_normalization(x, f, fovn=60, img_size=448):
    """
    Section 3.1 of https://arxiv.org/pdf/1904.02028.pdf
    E = (fn/f) * E' where E is 1/d
    """
    fn = get_focalLength_from_fieldOfView(fov=fovn, img_size=img_size)
    y = x * (fn / f)
    return y


def undo_focal_length_normalization(y, f, fovn=60, img_size=448):
    """
    Undo focal_length_normalization()
    """
    fn = get_focalLength_from_fieldOfView(fov=fovn, img_size=img_size)
    x = y * (f / fn)
    return x


EPS_LOG = 1e-10


def log_depth(x, eps=EPS_LOG):
    """
    Move depth to log space
    """
    return torch.log(x + eps)


def undo_log_depth(y, eps=EPS_LOG):
    """
    Undo log_depth()
    """
    return torch.exp(y) - eps
