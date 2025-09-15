import torch
import numpy as np
from typing import Tuple

left_hand_joint_idx = torch.arange(42, 63)
right_hand_joint_idx = torch.arange(21, 42)
HAND_JOINTS_IDX = torch.cat([left_hand_joint_idx, right_hand_joint_idx]).tolist()

def crop_to_hand(center_x: float, center_y: float, width: float, height: float, keypoints_2d: np.ndarray, padding: float = 1.1) -> Tuple:
    ori_keypoints_2d = keypoints_2d.copy()
    keypoints_2d = np.zeros_like(keypoints_2d)
    if keypoints_2d.shape[0] == 61:  # openpose (25) + HMR2_J19 (19) + WHAM_coco (17)
        assert False, "Not implemented for 61 keypoints"
    elif keypoints_2d.shape[0] == 70:  # ATLAS70
        # lower_body_keypoints = list(range(11, 21))
        keybody_keypoints = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 41, 62]
        keypoints_2d[HAND_JOINTS_IDX, :] = ori_keypoints_2d[HAND_JOINTS_IDX]
    else:
        raise ValueError("Invalid keypoints_2d shape: ", keypoints_2d.shape[0])

    p = torch.rand(1).item()
    if keypoints_2d[:, -1].sum() > 1:
        center, scale = get_bbox(keypoints_2d, rescale=1.2 + p)
        center_x = center[0]
        center_y = center[1]
        width = padding * scale[0]
        height = padding * scale[1]
    return center_x, center_y, width, height


def crop_to_hips(center_x: float, center_y: float, width: float, height: float, keypoints_2d: np.ndarray, padding: float = 1.1) -> Tuple:
    """
    Extreme cropping: Crop the box up to the hip locations.
    Args:
        center_x (float): x coordinate of the bounding box center.
        center_y (float): y coordinate of the bounding box center.
        width (float): Bounding box width.
        height (float): Bounding box height.
        keypoints_2d (np.ndarray): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        center_x (float): x coordinate of the new bounding box center.
        center_y (float): y coordinate of the new bounding box center.
        width (float): New bounding box width.
        height (float): New bounding box height.
    """
    keypoints_2d = keypoints_2d.copy()
    if keypoints_2d.shape[0] == 61:  # openpose (25) + HMR2_J19 (19) + WHAM_coco (17)
        lower_body_keypoints_openpose = [10, 11, 13, 14, 19, 20, 21, 22, 23, 24]
        lower_body_keypoints_j19 = [25 + i for i in [1, 0, 4, 5]]
        lower_body_keypoints_coco = [44 + i for i in [14, 16, 13, 15]]
        lower_body_keypoints = lower_body_keypoints_openpose + lower_body_keypoints_j19 + lower_body_keypoints_coco
        keybody_keypoints = list(range(61))
    elif keypoints_2d.shape[0] == 70:  # ATLAS70
        lower_body_keypoints = list(range(11, 21))
        keybody_keypoints = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 41, 62]
    else:
        raise ValueError("Invalid keypoints_2d shape: ", keypoints_2d.shape[0])

    keypoints_2d[lower_body_keypoints, :] = 0
    if keypoints_2d[keybody_keypoints, -1].sum() > 2:
        center, scale = get_bbox(keypoints_2d)
        center_x = center[0]
        center_y = center[1]
        width = padding * scale[0]
        height = padding * scale[1]
    return center_x, center_y, width, height


def crop_to_lower(center_x: float, center_y: float, width: float, height: float, keypoints_2d: np.ndarray, padding: float=1.5) -> Tuple:
    """
    Extreme cropping: Crop the box lower than  the hip locations.
    Args:
        center_x (float): x coordinate of the bounding box center.
        center_y (float): y coordinate of the bounding box center.
        width (float): Bounding box width.
        height (float): Bounding box height.
        keypoints_2d (np.ndarray): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        center_x (float): x coordinate of the new bounding box center.
        center_y (float): y coordinate of the new bounding box center.
        width (float): New bounding box width.
        height (float): New bounding box height.
    """
    ori_keypoints_2d = keypoints_2d.copy()
    keypoints_2d = np.zeros_like(keypoints_2d)
    if keypoints_2d.shape[0] == 61:  # openpose (25) + HMR2_J19 (19) + WHAM_coco (17)
        lower_body_keypoints_openpose = [10, 11, 13, 14, 19, 20, 21, 22, 23, 24]
        lower_body_keypoints_j19 = [25 + i for i in [1, 0, 4, 5]]
        lower_body_keypoints_coco = [44 + i for i in [14, 16, 13, 15]]
        lower_body_keypoints = lower_body_keypoints_openpose + lower_body_keypoints_j19 + lower_body_keypoints_coco
    else:
        raise ValueError("Invalid keypoints_2d shape: ", keypoints_2d.shape[0])

    keypoints_2d[lower_body_keypoints] = ori_keypoints_2d[lower_body_keypoints]
    if keypoints_2d[:, -1].sum() > 1:
        center, scale = get_bbox(keypoints_2d)
        center_x = center[0]
        center_y = center[1]
        width = padding * scale[0]
        height = padding * scale[1]
    return center_x, center_y, width, height


def crop_to_shoulders(center_x: float, center_y: float, width: float, height: float, keypoints_2d: np.ndarray):
    """
    Extreme cropping: Crop the box up to the shoulder locations.
    Args:
        center_x (float): x coordinate of the bounding box center.
        center_y (float): y coordinate of the bounding box center.
        width (float): Bounding box width.
        height (float): Bounding box height.
        keypoints_2d (np.ndarray): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        center_x (float): x coordinate of the new bounding box center.
        center_y (float): y coordinate of the new bounding box center.
        width (float): New bounding box width.
        height (float): New bounding box height.
    """
    keypoints_2d = keypoints_2d.copy()
    if keypoints_2d.shape[0] == 61:  # openpose (25) + HMR2_J19 (19) + WHAM_coco (17)
        lower_body_keypoints_openpose = [3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 19, 20, 21, 22, 23, 24]
        lower_body_keypoints_j19 = [25 + i for i in [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 14, 15, 16]]
        lower_body_keypoints_coco = [44 + i for i in [7, 8, 9, 10, 11, 12, 14, 16, 13, 15]]
        lower_body_keypoints = lower_body_keypoints_openpose + lower_body_keypoints_j19 + lower_body_keypoints_coco
    elif keypoints_2d.shape[0] == 70:  # ATLAS70
        lower_body_keypoints = list(range(7, 67))
    else:
        raise ValueError("Invalid keypoints_2d shape: ", keypoints_2d.shape[0])
    
    keypoints_2d[lower_body_keypoints, :] = 0
    if keypoints_2d[:, -1].sum() > 1:
        center, scale = get_bbox(keypoints_2d)
        center_x = center[0]
        center_y = center[1]
        width = 1.2 * scale[0]
        height = 1.2 * scale[1]
    return center_x, center_y, width, height

def crop_to_head(center_x: float, center_y: float, width: float, height: float, keypoints_2d: np.ndarray):
    """
    Extreme cropping: Crop the box and keep on only the head.
    Args:
        center_x (float): x coordinate of the bounding box center.
        center_y (float): y coordinate of the bounding box center.
        width (float): Bounding box width.
        height (float): Bounding box height.
        keypoints_2d (np.ndarray): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        center_x (float): x coordinate of the new bounding box center.
        center_y (float): y coordinate of the new bounding box center.
        width (float): New bounding box width.
        height (float): New bounding box height.
    """
    keypoints_2d = keypoints_2d.copy()
    if keypoints_2d.shape[0] == 61:  # openpose (25) + HMR2_J19 (19) + WHAM_coco (17)
        lower_body_keypoints_openpose = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 19, 20, 21, 22, 23, 24]
        lower_body_keypoints_j19 = [25 + i for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16]]
        lower_body_keypoints_coco = [44 + i for i in [5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 13, 15]]
        lower_body_keypoints = lower_body_keypoints_openpose + lower_body_keypoints_j19 + lower_body_keypoints_coco
    elif keypoints_2d.shape[0] == 70:  # ATLAS70
        lower_body_keypoints = list(range(5, 69))
    else:
        raise ValueError("Invalid keypoints_2d shape: ", keypoints_2d.shape[0])
    
    keypoints_2d[lower_body_keypoints, :] = 0
    if keypoints_2d[:, -1].sum() > 1:
        center, scale = get_bbox(keypoints_2d)
        center_x = center[0]
        center_y = center[1]
        width = 1.3 * scale[0]
        height = 1.3 * scale[1]
    return center_x, center_y, width, height

def crop_torso_only(center_x: float, center_y: float, width: float, height: float, keypoints_2d: np.ndarray):
    """
    Extreme cropping: Crop the box and keep on only the torso.
    Args:
        center_x (float): x coordinate of the bounding box center.
        center_y (float): y coordinate of the bounding box center.
        width (float): Bounding box width.
        height (float): Bounding box height.
        keypoints_2d (np.ndarray): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        center_x (float): x coordinate of the new bounding box center.
        center_y (float): y coordinate of the new bounding box center.
        width (float): New bounding box width.
        height (float): New bounding box height.
    """
    ori_keypoints_2d = keypoints_2d.copy()
    keypoints_2d = np.zeros_like(keypoints_2d)
    if keypoints_2d.shape[0] == 61:  # openpose (25) + HMR2_J19 (19) + WHAM_coco (17)
        torso_body_keypoints_openpose = [1, 2, 5, 8, 9, 12]
        torso_body_keypoints_j19 = [25 + i for i in [2, 3, 8, 9, 12, 14, 15, 16]]
        torso_body_keypoints_coco = [44 + i for i in [5, 6, 11, 12]]
        torso_body_keypoints = torso_body_keypoints_openpose + torso_body_keypoints_j19 + torso_body_keypoints_coco
    elif keypoints_2d.shape[0] == 70:  # ATLAS70
        torso_body_keypoints = [5, 6, 9, 10, 67, 68, 69]
    else:
        raise ValueError("Invalid keypoints_2d shape: ", keypoints_2d.shape[0])
    
    keypoints_2d[torso_body_keypoints] = ori_keypoints_2d[torso_body_keypoints]
    if keypoints_2d[:, -1].sum() > 1:
        center, scale = get_bbox(keypoints_2d)
        center_x = center[0]
        center_y = center[1]
        width = 1.1 * scale[0]
        height = 1.1 * scale[1]
    return center_x, center_y, width, height

def crop_rightarm_only(center_x: float, center_y: float, width: float, height: float, keypoints_2d: np.ndarray):
    """
    Extreme cropping: Crop the box and keep on only the right arm.
    Args:
        center_x (float): x coordinate of the bounding box center.
        center_y (float): y coordinate of the bounding box center.
        width (float): Bounding box width.
        height (float): Bounding box height.
        keypoints_2d (np.ndarray): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        center_x (float): x coordinate of the new bounding box center.
        center_y (float): y coordinate of the new bounding box center.
        width (float): New bounding box width.
        height (float): New bounding box height.
    """
    ori_keypoints_2d = keypoints_2d.copy()
    keypoints_2d = np.zeros_like(keypoints_2d)
    if keypoints_2d.shape[0] == 61:  # openpose (25) + HMR2_J19 (19) + WHAM_coco (17)
        rightarm_body_keypoints_openpose = [2, 3, 4]
        rightarm_body_keypoints_j19 = [25 + i for i in [6, 7, 8]]
        rightarm_body_keypoints_coco = [44 + i for i in [6, 8, 10]]
        rightarm_body_keypoints = rightarm_body_keypoints_openpose + rightarm_body_keypoints_j19 + rightarm_body_keypoints_coco
    elif keypoints_2d.shape[0] == 70:  # ATLAS70
        rightarm_body_keypoints = [6, 8, 10, 64, 66] + list(range(21, 42))
    else:
        raise ValueError("Invalid keypoints_2d shape: ", keypoints_2d.shape[0])
    
    keypoints_2d[rightarm_body_keypoints] = ori_keypoints_2d[rightarm_body_keypoints]
    if keypoints_2d[:, -1].sum() > 1:
        center, scale = get_bbox(keypoints_2d)
        center_x = center[0]
        center_y = center[1]
        width = 1.1 * scale[0]
        height = 1.1 * scale[1]
    return center_x, center_y, width, height

def crop_leftarm_only(center_x: float, center_y: float, width: float, height: float, keypoints_2d: np.ndarray):
    """
    Extreme cropping: Crop the box and keep on only the left arm.
    Args:
        center_x (float): x coordinate of the bounding box center.
        center_y (float): y coordinate of the bounding box center.
        width (float): Bounding box width.
        height (float): Bounding box height.
        keypoints_2d (np.ndarray): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        center_x (float): x coordinate of the new bounding box center.
        center_y (float): y coordinate of the new bounding box center.
        width (float): New bounding box width.
        height (float): New bounding box height.
    """
    ori_keypoints_2d = keypoints_2d.copy()
    keypoints_2d = np.zeros_like(keypoints_2d)
    if keypoints_2d.shape[0] == 61:  # openpose (25) + HMR2_J19 (19) + WHAM_coco (17)
        leftarm_body_keypoints_openpose = [5, 6, 7]
        leftarm_body_keypoints_j19 = [25 + i for i in [9, 10, 11]]
        leftarm_body_keypoints_coco = [44 + i for i in [5, 7, 9]]
        leftarm_body_keypoints = leftarm_body_keypoints_openpose + leftarm_body_keypoints_j19 + leftarm_body_keypoints_coco
    elif keypoints_2d.shape[0] == 70:  # ATLAS70
        leftarm_body_keypoints = [5, 7, 9, 63, 65] + list(range(42, 63))
    else:
        raise ValueError("Invalid keypoints_2d shape: ", keypoints_2d.shape[0])
    
    keypoints_2d[leftarm_body_keypoints] = ori_keypoints_2d[leftarm_body_keypoints]
    if keypoints_2d[:, -1].sum() > 1:
        center, scale = get_bbox(keypoints_2d)
        center_x = center[0]
        center_y = center[1]
        width = 1.1 * scale[0]
        height = 1.1 * scale[1]
    return center_x, center_y, width, height

def crop_legs_only(center_x: float, center_y: float, width: float, height: float, keypoints_2d: np.ndarray):
    """
    Extreme cropping: Crop the box and keep on only the legs.
    Args:
        center_x (float): x coordinate of the bounding box center.
        center_y (float): y coordinate of the bounding box center.
        width (float): Bounding box width.
        height (float): Bounding box height.
        keypoints_2d (np.ndarray): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        center_x (float): x coordinate of the new bounding box center.
        center_y (float): y coordinate of the new bounding box center.
        width (float): New bounding box width.
        height (float): New bounding box height.
    """
    keypoints_2d = keypoints_2d.copy()
    if keypoints_2d.shape[0] == 61:  # openpose (25) + HMR2_J19 (19) + WHAM_coco (17)
        nonlegs_body_keypoints_openpose = [0, 1, 2, 3, 4, 5, 6, 7, 15, 16, 17, 18]
        nonlegs_body_keypoints_j19 = [25 + i for i in [6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18]]
        nonlegs_body_keypoints_coco = [44 + i for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
        nonlegs_body_keypoints = nonlegs_body_keypoints_openpose + nonlegs_body_keypoints_j19 + nonlegs_body_keypoints_coco
    elif keypoints_2d.shape[0] == 70:  # ATLAS70
        nonlegs_body_keypoints = list(range(0, 9)) + list(range(21, 69))
    else:
        raise ValueError("Invalid keypoints_2d shape: ", keypoints_2d.shape[0])
    
    keypoints_2d[nonlegs_body_keypoints, :] = 0
    if keypoints_2d[:, -1].sum() > 2:
        center, scale = get_bbox(keypoints_2d)
        center_x = center[0]
        center_y = center[1]
        width = 1.1 * scale[0]
        height = 1.1 * scale[1]
    return center_x, center_y, width, height

def crop_rightleg_only(center_x: float, center_y: float, width: float, height: float, keypoints_2d: np.ndarray):
    """
    Extreme cropping: Crop the box and keep on only the right leg.
    Args:
        center_x (float): x coordinate of the bounding box center.
        center_y (float): y coordinate of the bounding box center.
        width (float): Bounding box width.
        height (float): Bounding box height.
        keypoints_2d (np.ndarray): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        center_x (float): x coordinate of the new bounding box center.
        center_y (float): y coordinate of the new bounding box center.
        width (float): New bounding box width.
        height (float): New bounding box height.
    """
    ori_keypoints_2d = keypoints_2d.copy()
    keypoints_2d = np.zeros_like(keypoints_2d)
    if keypoints_2d.shape[0] == 61:  # openpose (25) + HMR2_J19 (19) + WHAM_coco (17)
        rightleg_body_keypoints_openpose = [9, 10, 11, 22, 23, 24]
        rightleg_body_keypoints_j19 = [25 + i for i in [0, 1, 2]]
        rightleg_body_keypoints_coco = [44 + i for i in [12, 14, 16]]
        rightleg_body_keypoints = rightleg_body_keypoints_openpose + rightleg_body_keypoints_j19 + rightleg_body_keypoints_coco
    elif keypoints_2d.shape[0] == 70:  # ATLAS70
        rightleg_body_keypoints = [10, 12, 14, 18, 19, 20]
    else:
        raise ValueError("Invalid keypoints_2d shape: ", keypoints_2d.shape[0])
    
    keypoints_2d[rightleg_body_keypoints] = ori_keypoints_2d[rightleg_body_keypoints]
    if keypoints_2d[:, -1].sum() > 1:
        center, scale = get_bbox(keypoints_2d)
        center_x = center[0]
        center_y = center[1]
        width = 1.1 * scale[0]
        height = 1.1 * scale[1]
    return center_x, center_y, width, height

def crop_leftleg_only(center_x: float, center_y: float, width: float, height: float, keypoints_2d: np.ndarray):
    """
    Extreme cropping: Crop the box and keep on only the left leg.
    Args:
        center_x (float): x coordinate of the bounding box center.
        center_y (float): y coordinate of the bounding box center.
        width (float): Bounding box width.
        height (float): Bounding box height.
        keypoints_2d (np.ndarray): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        center_x (float): x coordinate of the new bounding box center.
        center_y (float): y coordinate of the new bounding box center.
        width (float): New bounding box width.
        height (float): New bounding box height.
    """
    ori_keypoints_2d = keypoints_2d.copy()
    keypoints_2d = np.zeros_like(keypoints_2d)
    if keypoints_2d.shape[0] == 61:  # openpose (25) + HMR2_J19 (19) + WHAM_coco (17)
        leftleg_body_keypoints_openpose = [12, 13, 14, 19, 20, 21]
        leftleg_body_keypoints_j19 = [25 + i for i in [3, 4, 5]]
        leftleg_body_keypoints_coco = [44 + i for i in [11, 13, 15]]
        leftleg_body_keypoints = leftleg_body_keypoints_openpose + leftleg_body_keypoints_j19 + leftleg_body_keypoints_coco
    elif keypoints_2d.shape[0] == 70:  # ATLAS70
        leftleg_body_keypoints = [9, 11, 13, 15, 16, 17]
    else:
        raise ValueError("Invalid keypoints_2d shape: ", keypoints_2d.shape[0])
    
    keypoints_2d[leftleg_body_keypoints] = ori_keypoints_2d[leftleg_body_keypoints]
    if keypoints_2d[:, -1].sum() > 1:
        center, scale = get_bbox(keypoints_2d)
        center_x = center[0]
        center_y = center[1]
        width = 1.1 * scale[0]
        height = 1.1 * scale[1]
    return center_x, center_y, width, height


def has_hand(keypoints_2d: np.ndarray) -> bool:
    """
    Check if the sample has hand kps annotation
    Args:
        keypoints_2d (np.ndarray): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        bool: True if all main body joints are visible.
    """
    if keypoints_2d.shape[0] == 61:  # openpose (25) + HMR2_J19 (19) + WHAM_coco (17)
        assert False, "Not implemented for 61 keypoints"
        # body_keypoints_openpose = [2, 3, 4, 5, 6, 7, 10, 11, 13, 14]
        # body_keypoints_j19 = [25 + i for i in [8, 7, 6, 9, 10, 11, 1, 0, 4, 5]]
        # body_keypoints_coco = [44 + i for i in [6, 8, 10, 5, 7, 9, 14, 16, 13, 15]]

        # return ((keypoints_2d[body_keypoints_openpose, -1] + keypoints_2d[body_keypoints_j19, -1] + keypoints_2d[body_keypoints_coco, -1]) > 0).sum() == len(body_keypoints_openpose)
    elif keypoints_2d.shape[0] == 70:  # ATLAS70
        # body_keypoints = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 41, 62]
        return ((keypoints_2d[HAND_JOINTS_IDX, -1]) >= 0.5).sum() > len(HAND_JOINTS_IDX) * 0.3
    else:
        raise ValueError("Invalid keypoints_2d shape: ", keypoints_2d.shape[0])


def full_body(keypoints_2d: np.ndarray) -> bool:
    """
    Check if all main body joints are visible.
    Args:
        keypoints_2d (np.ndarray): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        bool: True if all main body joints are visible.
    """
    if keypoints_2d.shape[0] == 61:  # openpose (25) + HMR2_J19 (19) + WHAM_coco (17)
        body_keypoints_openpose = [2, 3, 4, 5, 6, 7, 10, 11, 13, 14]
        body_keypoints_j19 = [25 + i for i in [8, 7, 6, 9, 10, 11, 1, 0, 4, 5]]
        body_keypoints_coco = [44 + i for i in [6, 8, 10, 5, 7, 9, 14, 16, 13, 15]]

        return ((keypoints_2d[body_keypoints_openpose, -1] + keypoints_2d[body_keypoints_j19, -1] + keypoints_2d[body_keypoints_coco, -1]) > 0).sum() == len(body_keypoints_openpose)
    elif keypoints_2d.shape[0] == 70:  # ATLAS70
        body_keypoints = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 41, 62]
        return ((keypoints_2d[body_keypoints, -1]) >= 0.5).sum() > len(body_keypoints) * 0.8
    else:
        raise ValueError("Invalid keypoints_2d shape: ", keypoints_2d.shape[0])

def upper_body(keypoints_2d: np.ndarray):
    """
    Check if only upper body joints are visible (no lower body joints).
    Args:
        keypoints_2d (np.ndarray): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        bool: True if only main upper-body joints are visible.
    """
    if keypoints_2d.shape[0] == 61:  # openpose (25) + HMR2_J19 (19) + WHAM_coco (17)
        lower_body_keypoints_openpose = [10, 11, 13, 14]
        lower_body_keypoints_j19 = [25 + i for i in [1, 0, 4, 5]]
        lower_body_keypoints_coco = [44 + i for i in [14, 16, 13, 15]]
        lower_body_keypoints = lower_body_keypoints_openpose + lower_body_keypoints_j19 + lower_body_keypoints_coco

        upper_body_keypoints_openpose = [0, 1, 2, 5, 15, 16, 17, 18]
        upper_body_keypoints_j19 = [15 + i for i in [8, 9, 12, 13, 17, 18]]
        upper_body_keypoints_coco = [44 + i for i in [0, 1, 2, 3, 4, 5, 6]]
        upper_body_keypoints = upper_body_keypoints_openpose + upper_body_keypoints_j19 + upper_body_keypoints_coco

        return (keypoints_2d[lower_body_keypoints, -1] > 0).sum() == 0 and (keypoints_2d[upper_body_keypoints, -1] > 0).sum() > 2
    elif keypoints_2d.shape[0] == 70:  # ATLAS70
        lower_body_keypoints = list(range(11, 21))
        upper_body_keypoints = list(range(0, 11)) + [41, 62] + list(range(63, 70))

        return (keypoints_2d[upper_body_keypoints, -1] >= 0.5).sum() > 4
    else:
        raise ValueError("Invalid keypoints_2d shape: ", keypoints_2d.shape[0])

def get_bbox(keypoints_2d: np.ndarray, rescale: float = 1.2) -> Tuple:
    """
    Get center and scale for bounding box from keypoint detections.
    Args:
        keypoints_2d (np.ndarray): Array of shape (N, 3) containing 2D keypoint locations.
        rescale (float): Scale factor to rescale bounding boxes computed from the keypoints.
    Returns:
        center (np.ndarray): Array of shape (2,) containing the new bounding box center.
        scale (float): New bounding box scale.
    """
    valid = keypoints_2d[:,-1] > 0
    valid_keypoints = keypoints_2d[valid][:,:-1]
    center = 0.5 * (valid_keypoints.max(axis=0) + valid_keypoints.min(axis=0))
    bbox_size = (valid_keypoints.max(axis=0) - valid_keypoints.min(axis=0))
    # adjust bounding box tightness
    scale = bbox_size
    scale *= rescale
    return center, scale

def extreme_cropping(center_x: float, center_y: float, width: float, height: float, keypoints_2d: np.ndarray) -> Tuple:
    """
    Perform extreme cropping
    Args:
        center_x (float): x coordinate of bounding box center.
        center_y (float): y coordinate of bounding box center.
        width (float): bounding box width.
        height (float): bounding box height.
        keypoints_2d (np.ndarray): Array of shape (N, 3) containing 2D keypoint locations.
        rescale (float): Scale factor to rescale bounding boxes computed from the keypoints.
    Returns:
        center_x (float): x coordinate of bounding box center.
        center_y (float): y coordinate of bounding box center.
        width (float): bounding box width.
        height (float): bounding box height.
    """
    p = torch.rand(1).item()
    if full_body(keypoints_2d):
        if p < 0.7:
            center_x, center_y, width, height = crop_to_hips(center_x, center_y, width, height, keypoints_2d)
        elif p < 0.9:
            center_x, center_y, width, height = crop_to_shoulders(center_x, center_y, width, height, keypoints_2d)
        else:
            center_x, center_y, width, height = crop_to_head(center_x, center_y, width, height, keypoints_2d)
    elif upper_body(keypoints_2d):
        if p < 0.9:
            center_x, center_y, width, height = crop_to_shoulders(center_x, center_y, width, height, keypoints_2d)
        else:
            center_x, center_y, width, height = crop_to_head(center_x, center_y, width, height, keypoints_2d)

    return center_x, center_y, max(width, height), max(width, height)


def extreme_cropping_aggressive(center_x: float, center_y: float, width: float, height: float, keypoints_2d: np.ndarray) -> Tuple:
    """
    Perform aggressive extreme cropping
    Args:
        center_x (float): x coordinate of bounding box center.
        center_y (float): y coordinate of bounding box center.
        width (float): bounding box width.
        height (float): bounding box height.
        keypoints_2d (np.ndarray): Array of shape (N, 3) containing 2D keypoint locations.
        rescale (float): Scale factor to rescale bounding boxes computed from the keypoints.
    Returns:
        center_x (float): x coordinate of bounding box center.
        center_y (float): y coordinate of bounding box center.
        width (float): bounding box width.
        height (float): bounding box height.
    """
    p = torch.rand(1).item()
    if full_body(keypoints_2d):
        if p < 0.3:
            center_x, center_y, width, height = crop_to_hips(center_x, center_y, width, height, keypoints_2d)
        elif p < 0.5:
            center_x, center_y, width, height = crop_to_shoulders(center_x, center_y, width, height, keypoints_2d)
        elif p < 0.55:
            center_x, center_y, width, height = crop_to_head(center_x, center_y, width, height, keypoints_2d)
        elif p < 0.7:
            center_x, center_y, width, height = crop_torso_only(center_x, center_y, width, height, keypoints_2d)
        elif p < 0.75:
            center_x, center_y, width, height = crop_rightarm_only(center_x, center_y, width, height, keypoints_2d)
        elif p < 0.8:
            center_x, center_y, width, height = crop_leftarm_only(center_x, center_y, width, height, keypoints_2d)
        elif p < 0.9:
            center_x, center_y, width, height = crop_legs_only(center_x, center_y, width, height, keypoints_2d)
        elif p < 0.95:
            center_x, center_y, width, height = crop_rightleg_only(center_x, center_y, width, height, keypoints_2d)
        else:
            center_x, center_y, width, height = crop_leftleg_only(center_x, center_y, width, height, keypoints_2d)
    elif upper_body(keypoints_2d):
        if p < 0.2:
            center_x, center_y, width, height = crop_to_hips(center_x, center_y, width, height, keypoints_2d)
        elif p < 0.4:
            center_x, center_y, width, height = crop_to_shoulders(center_x, center_y, width, height, keypoints_2d)
        elif p < 0.5:
            center_x, center_y, width, height = crop_to_head(center_x, center_y, width, height, keypoints_2d)
        elif p < 0.8:
            center_x, center_y, width, height = crop_torso_only(center_x, center_y, width, height, keypoints_2d)
        elif p < 0.9:
            center_x, center_y, width, height = crop_rightarm_only(center_x, center_y, width, height, keypoints_2d)
        else:
            center_x, center_y, width, height = crop_leftarm_only(center_x, center_y, width, height, keypoints_2d)

    return center_x, center_y, max(width, height), max(width, height)


def extreme_cropping_halfbody(center_x: float, center_y: float, width: float, height: float, keypoints_2d: np.ndarray) -> Tuple:
    """
    Perform extreme cropping
    Args:
        center_x (float): x coordinate of bounding box center.
        center_y (float): y coordinate of bounding box center.
        width (float): bounding box width.
        height (float): bounding box height.
        keypoints_2d (np.ndarray): Array of shape (N, 3) containing 2D keypoint locations.
        rescale (float): Scale factor to rescale bounding boxes computed from the keypoints.
    Returns:
        center_x (float): x coordinate of bounding box center.
        center_y (float): y coordinate of bounding box center.
        width (float): bounding box width.
        height (float): bounding box height.
    """
    p = torch.rand(1).item()
    if full_body(keypoints_2d):
        if p < 0.7:
            center_x, center_y, width, height = crop_to_hips(center_x, center_y, width, height, keypoints_2d, padding=1.1)
        else:
            center_x, center_y, width, height = crop_to_lower(center_x, center_y, width, height, keypoints_2d, padding=1.1)
    return center_x, center_y, max(width, height), max(width, height)


def extreme_cropping_around_hand(center_x: float, center_y: float, width: float, height: float, keypoints_2d: np.ndarray) -> Tuple:
    """
    Perform extreme cropping
    Args:
        center_x (float): x coordinate of bounding box center.
        center_y (float): y coordinate of bounding box center.
        width (float): bounding box width.
        height (float): bounding box height.
        keypoints_2d (np.ndarray): Array of shape (N, 3) containing 2D keypoint locations.
        rescale (float): Scale factor to rescale bounding boxes computed from the keypoints.
    Returns:
        center_x (float): x coordinate of bounding box center.
        center_y (float): y coordinate of bounding box center.
        width (float): bounding box width.
        height (float): bounding box height.
    """
    p = torch.rand(1).item()
    if has_hand(keypoints_2d):
        center_x, center_y, width, height = crop_to_hand(center_x, center_y, width, height, keypoints_2d, padding=1.1)
    return center_x, center_y, max(width, height), max(width, height)
