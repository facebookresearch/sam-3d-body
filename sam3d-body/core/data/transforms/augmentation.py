import random
from typing import Dict, List, Tuple

import numpy as np
from core.models.modules.geometry_utils import rot_aa
from yacs.config import CfgNode

from .crop_utils import (
    extreme_cropping,
    extreme_cropping_aggressive,
    extreme_cropping_around_hand,
    extreme_cropping_halfbody,
)
from .flip_utils import atlas_index_flip, atlas_index_ori


def get_augmentation_params(aug_config: CfgNode) -> Tuple:
    """
    Compute random augmentation parameters.
    Args:
        aug_config (CfgNode): Config containing augmentation parameters.
    Returns:
        scale (float): Box rescaling factor.
        rot (float): Random image rotation.
        do_flip (bool): Whether to flip image or not.
        do_extreme_crop (bool): Whether to apply extreme cropping (as proposed in EFT).
        color_scale (List): Color rescaling factor
        tx (float): Random translation along the x axis.
        ty (float): Random translation along the y axis.
    """

    tx = np.clip(np.random.randn(), -1.0, 1.0) * aug_config.TRANS_FACTOR
    ty = np.clip(np.random.randn(), -1.0, 1.0) * aug_config.TRANS_FACTOR
    scale = np.clip(np.random.randn(), -1.0, 1.0) * aug_config.SCALE_FACTOR + 1.0
    assert (
        aug_config.ROT_AUG_RATE == 0
    ), "Careful - as of 6/17/25, kps rotation is around base, mesh around root. As such, mesh may not be aligned w/ image."
    rot = (
        np.clip(np.random.randn(), -2.0, 2.0) * aug_config.ROT_FACTOR
        if random.random() <= aug_config.ROT_AUG_RATE
        else 0
    )
    do_flip = aug_config.DO_FLIP and random.random() <= aug_config.FLIP_AUG_RATE
    do_extreme_crop = random.random() <= aug_config.EXTREME_CROP_AUG_RATE
    do_hand_centric_crop = random.random() <= aug_config.AROUND_HAND_CROP_AUG_RATE
    extreme_crop_lvl = aug_config.get("EXTREME_CROP_AUG_LEVEL", 0)
    color_jitter_scale = aug_config.get("COLOR_JITTER_SCALE", 0.0)
    if color_jitter_scale == 0.0:
        c_up = 1.0 + aug_config.COLOR_SCALE
        c_low = 1.0 - aug_config.COLOR_SCALE
        color_scale = [
            random.uniform(c_low, c_up),
            random.uniform(c_low, c_up),
            random.uniform(c_low, c_up),
        ]
    else:
        color_scale = [1, 1, 1]

    return {
        "tx": tx,
        "ty": ty,
        "scale": scale,
        "rot": rot,
        "do_flip": do_flip,
        "do_extreme_crop": do_extreme_crop,
        "do_hand_centric_crop": do_hand_centric_crop,
        "extreme_crop_lvl": extreme_crop_lvl,
        "color_scale": color_scale,
    }


def get_image_augmentation(aug_config: CfgNode) -> Tuple:
    """
    Compute random augmentation parameters.
    Args:
        aug_config (CfgNode): Config containing augmentation parameters.
    Returns:
        rot (float): Random image rotation.
        do_flip (bool): Whether to flip image or not.
    """
    rot = (
        np.clip(np.random.randn(), -2.0, 2.0) * aug_config.ROT_FACTOR
        if random.random() <= aug_config.ROT_AUG_RATE
        else 0
    )
    do_flip = aug_config.DO_FLIP and random.random() <= aug_config.FLIP_AUG_RATE
    do_vflip = aug_config.DO_VFLIP and random.random() <= aug_config.VFLIP_AUG_RATE

    # if vertical flip is enabled, disable horizontal flip
    if do_vflip:
        do_flip = False

    # For FOV augmentation
    do_center_crop = (
        aug_config.get("DO_CENTER_CROP", False)
        and random.random() <= aug_config.get("CENTER_CROP_RATE", 0.0)
    )

    return {
        "rot": rot,
        "do_flip": do_flip,
        "do_vflip": do_vflip,
        "do_center_crop": do_center_crop,
    }


def get_mask_augmentation(aug_config: CfgNode) -> Tuple:
    """
    Compute random augmentation parameters.
    Args:
        aug_config (CfgNode): Config containing augmentation parameters.
    Returns:
        mask_prompt_rate (float): Mask prompt rate.
        mask_sp_factor (float): Single person dataset mask prompt rate attenuation factor.
    """

    # do_mask_prompt = random.random() < aug_config.get("MASK_PROMPT_RATE", 0.0)
    # now `do_mask_prompt` is decided in loader
    mask_prompt_rate = aug_config.get("MASK_PROMPT_RATE", 0.0)
    # if it's an MP dataset use MASK_PROMPT_RATE otherwise attenuate the frequency
    # by a factor
    mask_sp_factor = aug_config.get("MASK_PROMPT_SP_FACTOR", 1.0)

    return {
        # "do_mask_prompt": do_mask_prompt,
        "mask_prompt_rate": mask_prompt_rate,
        "mask_sp_factor": mask_sp_factor,
    }


def get_person_augmentation(
    aug_config: CfgNode, dataset_name: str = None, mask_jitter_ctrl: float = False
) -> Tuple:
    """
    Compute random augmentation parameters.
    Args:
        aug_config (CfgNode): Config containing augmentation parameters.
        dataset_name (str): Dataset name.
        mask_jitter_ctrl (float): Whether to control mask jittering.
    Returns:
        scale (float): Box rescaling factor.
        do_extreme_crop (bool): Whether to apply extreme cropping (as proposed in EFT).
        color_scale (List): Color rescaling factor
        tx (float): Random translation along the x axis.
        ty (float): Random translation along the y axis.
    """

    jitter_factor = mask_jitter_ctrl * aug_config.TRANS_FACTOR
    tx = np.clip(np.random.randn(), -1.0, 1.0) * jitter_factor
    ty = np.clip(np.random.randn(), -1.0, 1.0) * jitter_factor
    scale = np.clip(np.random.randn(), -1.0, 1.0) * aug_config.SCALE_FACTOR + 1.0
    do_extreme_crop = random.random() <= aug_config.EXTREME_CROP_AUG_RATE
    extreme_crop_lvl = aug_config.get("EXTREME_CROP_AUG_LEVEL", 0)
    do_hand_centric_crop = random.random() <= aug_config.AROUND_HAND_CROP_AUG_RATE
    color_jitter_scale = aug_config.get("COLOR_JITTER_SCALE", 0.0)
    if color_jitter_scale == 0.0:
        c_up = 1.0 + aug_config.COLOR_SCALE
        c_low = 1.0 - aug_config.COLOR_SCALE
        color_scale = [
            random.uniform(c_low, c_up),
            random.uniform(c_low, c_up),
            random.uniform(c_low, c_up),
        ]
    else:
        color_scale = [1, 1, 1]

    # only do flying object if not doing extreme cropping
    fo_rate, fo_scale = aug_config.FLYING_OBJECT_AUG_RATE, 0.0
    if dataset_name is not None:
        if "harmony4d" in dataset_name:
            fo_rate = 0
        elif any([name in dataset_name for name in ["goliath", "metasim", "egoexo4d"]]):
            fo_rate *= 2
    do_flying_object = not do_extreme_crop and random.random() <= fo_rate

    if do_flying_object:
        fo_scale = (
            np.random.normal(loc=aug_config.FLYING_OBJECT_SCALE * 10, scale=1) / 10
        )

        sc_clip = 0.6
        if aug_config.get("FLYING_OBJECT_SCALE_CLIP", None) is not None:
            sc_clip = aug_config.get("FLYING_OBJECT_SCALE_CLIP", None)
        fo_scale = np.clip(fo_scale, 0.05, sc_clip)

    return {
        "tx": tx,
        "ty": ty,
        "scale": scale,
        "do_extreme_crop": do_extreme_crop,
        "do_hand_centric_crop": do_hand_centric_crop,
        "extreme_crop_lvl": extreme_crop_lvl,
        "color_scale": color_scale,
        "do_flying_object": do_flying_object,
        "fo_scale": fo_scale,
    }


def transform_smpl_pose(body_pose: np.ndarray, augmentation_params: Dict) -> np.ndarray:
    """
    Apply random augmentations to the SMPL parameters.
    Args:
        body_pose (np.ndarray): shape [72,]
    """
    if augmentation_params["do_flip"]:
        # fmt: off
        body_pose_permutation = [6, 7, 8, 3, 4, 5, 9, 10, 11, 15, 16, 17, 12, 13,
                                14 ,18, 19, 20, 24, 25, 26, 21, 22, 23, 27, 28, 29, 33,
                                34, 35, 30, 31, 32, 36, 37, 38, 42, 43, 44, 39, 40, 41,
                                45, 46, 47, 51, 52, 53, 48, 49, 50, 57, 58, 59, 54, 55,
                                56, 63, 64, 65, 60, 61, 62, 69, 70, 71, 66, 67, 68]
        # fmt: on
        body_pose[3:] = body_pose[body_pose_permutation]
        body_pose[1::3] *= -1
        body_pose[2::3] *= -1

    if not augmentation_params["rot"] == 0:
        # Only rotate global_orient
        body_pose[:3] = rot_aa(body_pose[:3], augmentation_params["rot"])

    return body_pose


def transform_atlas_pose(
    global_rot: np.ndarray,
    body_pose: np.ndarray,
    hand_pose: np.ndarray,
    augmentation_params: Dict,
) -> np.ndarray:
    """
    Apply random augmentations to the ATLAS parameters.
    Args:
        global_rot
        body_pose (np.ndarray): shape [,],
        hand_pose
    """
    if augmentation_params["do_flip"]:
        global_rot[[0, 1]] *= -1

        ori_pose = body_pose.copy()
        body_pose[3::6] *= -1
        body_pose[4::6] *= -1
        body_pose[atlas_index_ori] = ori_pose[atlas_index_flip]

        hand_pose = np.concatenate([hand_pose[32:], hand_pose[:32]], axis=0)

    if augmentation_params["do_vflip"]:
        global_rot[[0, 1]] *= -1
        ori_pose = body_pose.copy()
        body_pose[3::6] *= -1
        body_pose[4::6] *= -1
        body_pose[atlas_index_ori] = ori_pose[atlas_index_flip]

        hand_pose = np.concatenate([hand_pose[32:], hand_pose[:32]], axis=0)
        global_rot[[1]] += float(np.pi)

    return global_rot, body_pose, hand_pose


def transform_atlas46_pose(
    global_rot: np.ndarray,
    body_pose: np.ndarray,
    hand_pose: np.ndarray,
    scale_params: np.ndarray,
    shape_params: np.ndarray,
    augmentation_params: Dict,
    cfg,
) -> np.ndarray:
    if augmentation_params["do_flip"] or augmentation_params["do_vflip"]:
        # fmt: off
        to_flip_idxs = [0, 1, 2, 3, 131, 132, 6, 7, 8, 9, 12, 13, 15, 16, 18, 19, 21, 22]
        atlas46_index_ori = [24,34,25,35,26,36,27,37,28,38,29,39,30,40,31,41,32,42,33,43,44,53,45,54,46,55,47,56,48,57,49,58,50,59,51,60,52,61,62,89,63,90,64,91,65,92,66,93,67,94,68,95,69,96,70,97,71,98,72,99,73,100,74,101,75,102,76,103,77,104,78,105,79,106,80,107,81,108,82,109,83,110,84,111,85,112,86,113,87,114,88,115,120,116,121,117,122,118,123,119]
        atlas46_index_flip = [34,24,35,25,36,26,37,27,38,28,39,29,40,30,41,31,42,32,43,33,53,44,54,45,55,46,56,47,57,48,58,49,59,50,60,51,61,52,89,62,90,63,91,64,92,65,93,66,94,67,95,68,96,69,97,70,98,71,99,72,100,73,101,74,102,75,103,76,104,77,105,78,106,79,107,80,108,81,109,82,110,83,111,84,112,85,113,86,114,87,115,88,116,120,117,121,118,122,119,123]
        # fmt: on

        global_rot[[1, 2]] *= -1

        ori_pose = body_pose.copy()
        body_pose[to_flip_idxs] *= -1
        body_pose[atlas46_index_ori] = ori_pose[atlas46_index_flip]

        num_hand_comps = cfg.MODEL.ATLAS_HEAD.NUM_HAND_COMPS
        hand_pose = np.concatenate(
            [hand_pose[num_hand_comps:], hand_pose[:num_hand_comps]], axis=0
        )

        # Flip left and right scales
        scale_param_num_per_hand = scale_params.shape[0] - 18
        assert scale_param_num_per_hand % 2 == 0, str(scale_param_num_per_hand)
        scale_param_num_per_hand = scale_param_num_per_hand // 2

        right_hand_scale_params = scale_params[
            -2 * scale_param_num_per_hand : -scale_param_num_per_hand
        ].copy()
        left_hand_scale_params = scale_params[-scale_param_num_per_hand:].copy()
        scale_params[-2 * scale_param_num_per_hand : -scale_param_num_per_hand] = (
            left_hand_scale_params
        )
        scale_params[-scale_param_num_per_hand:] = right_hand_scale_params

        # Flip kneeknock
        scale_params[15] *= -1

        # Flip hand shape
        shape_param_num_per_hand = cfg.MODEL.ATLAS_HEAD.NUM_HAND_SHAPE_COMPS
        right_hand_shape_params = shape_params[
            -2 * shape_param_num_per_hand : -shape_param_num_per_hand
        ].copy()
        left_hand_shape_params = shape_params[-shape_param_num_per_hand:].copy()
        shape_params[-2 * shape_param_num_per_hand : -shape_param_num_per_hand] = (
            left_hand_shape_params
        )
        shape_params[-shape_param_num_per_hand:] = right_hand_shape_params

    if augmentation_params["do_vflip"]:
        global_rot[[2]] += float(np.pi)

    return global_rot, body_pose, hand_pose, scale_params, shape_params


def fliplr_keypoints(
    joints: np.ndarray, width: float, flip_permutation: List[int]
) -> np.ndarray:
    """
    Flip 2D or 3D keypoints.
    Args:
        joints (np.array): Array of shape (N, 3) or (N, 4) containing 2D or 3D keypoint locations and confidence.
        flip_permutation (List): Permutation to apply after flipping.
    Returns:
        np.array: Flipped 2D or 3D keypoints with shape (N, 3) or (N, 4) respectively.
    """
    joints = joints.copy()
    # Flip horizontal
    joints[:, 0] = width - joints[:, 0] - 1
    joints = joints[flip_permutation, :]

    return joints


def flipud_keypoints(
    joints: np.ndarray, height: float, flip_permutation: List[int]
) -> np.ndarray:
    """
    Flip 2D or 3D keypoints.
    Args:
        joints (np.array): Array of shape (N, 3) or (N, 4) containing 2D or 3D keypoint locations and confidence.
        flip_permutation (List): Permutation to apply after flipping.
    Returns:
        np.array: Flipped 2D or 3D keypoints with shape (N, 3) or (N, 4) respectively.
    """
    joints = joints.copy()
    # Flip vertical
    joints[:, 1] = height - joints[:, 1] - 1
    joints = joints[flip_permutation, :]

    return joints


def transform_keypoints_3d(
    keypoints_3d: np.ndarray, flip_permutation: List, augmentation_params: Dict
) -> np.ndarray:
    """
    Process 3D keypoints (rotation/flipping).
    Args:
        keypoints_3d (np.array): Input array of shape (N, 4) containing the 3D keypoints and confidence.
            Assuming keypoints_3d are within range [-1, 1]
        flip_permutation (List): Permutation to apply after flipping.
    Returns:
        np.array: Transformed 3D keypoints with shape (N, 4).
    """
    if augmentation_params["do_flip"]:
        keypoints_3d = fliplr_keypoints(keypoints_3d, 1, flip_permutation)
    elif augmentation_params["do_vflip"]:
        return transform_keypoints_3d(
            keypoints_3d, flip_permutation, {"do_flip": True, "rot": 180}
        )

    if not augmentation_params["rot"] == 0:
        # in-plane rotation
        rot_rad = -augmentation_params["rot"] * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat = np.eye(3)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        keypoints_3d[:, :-1] = np.einsum("ij,kj->ki", rot_mat, keypoints_3d[:, :-1])

    return keypoints_3d


def transform_keypoints_3d_rot_around_root(
    keypoints_3d: np.ndarray,
    flip_permutation: List,
    augmentation_params: Dict,
    world_to_root_meters=0.9239869713783264,
) -> np.ndarray:
    """
    Process 3D keypoints (rotation/flipping).
    Args:
        keypoints_3d (np.array): Input array of shape (N, 4) containing the 3D keypoints and confidence.
            Assuming keypoints_3d are within range [-1, 1]
        flip_permutation (List): Permutation to apply after flipping.
    Returns:
        np.array: Transformed 3D keypoints with shape (N, 4).
    """
    if augmentation_params["do_flip"]:
        keypoints_3d = fliplr_keypoints(keypoints_3d, 1, flip_permutation)
    elif augmentation_params["do_vflip"]:
        return transform_keypoints_3d(
            keypoints_3d, flip_permutation, {"do_flip": True, "rot": 180}
        )

    if not augmentation_params["rot"] == 0:
        # Center at root
        keypoints_3d[:, 1] -= world_to_root_meters
        # in-plane rotation
        rot_rad = -augmentation_params["rot"] * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat = np.eye(3)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        keypoints_3d[:, :-1] = np.einsum("ij,kj->ki", rot_mat, keypoints_3d[:, :-1])
        keypoints_3d[:, 1] += world_to_root_meters

    return keypoints_3d


def get_extreme_cropping(
    center: np.ndarray,
    scale: np.ndarray,
    keypoints_2d: np.ndarray,
    augmentation_params: Dict,
    min_size: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    if not augmentation_params["do_extreme_crop"]:
        return center, scale

    center_x, center_y = center[0], center[1]
    width, height = scale[0], scale[1]

    if augmentation_params["extreme_crop_lvl"] == 0:
        center_x1, center_y1, width1, height1 = extreme_cropping(
            center_x, center_y, width, height, keypoints_2d
        )
    elif augmentation_params["extreme_crop_lvl"] == 1:
        center_x1, center_y1, width1, height1 = extreme_cropping_aggressive(
            center_x, center_y, width, height, keypoints_2d
        )
    elif augmentation_params["extreme_crop_lvl"] == 2:
        center_x1, center_y1, width1, height1 = extreme_cropping_halfbody(
            center_x, center_y, width, height, keypoints_2d
        )
    else:
        raise ValueError(
            "Invalid extreme_crop_lvl", augmentation_params["extreme_crop_lvl"]
        )

    if augmentation_params["do_hand_centric_crop"]:
        # additional croppoing strategy to enhance hand-centric crops
        center_x1, center_y1, width1, height1 = extreme_cropping_around_hand(
            center_x1, center_y1, width1, height1, keypoints_2d
        )

    # Resize to min_size if too small
    if width1 < min_size or height1 < min_size:
        # width1 == height1 returned from extreme_crop()
        width1 = height1 = min_size

    center[0] = center_x1
    center[1] = center_y1
    scale[0] = width1
    scale[1] = height1
    return center, scale
