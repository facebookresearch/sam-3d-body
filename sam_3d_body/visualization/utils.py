# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
from typing import Dict, Optional, Union

import cv2
import numpy as np
from detectron2.config import LazyConfig
from omegaconf import OmegaConf


def draw_text(
    image: np.ndarray,
    texts: str,
    positions: np.ndarray,
    image_size: Optional[tuple] = None,
    font_size: Optional[int] = None,
    color: Union[str, tuple] = "g",
    vertical_alignment: str = "top",
    horizontal_alignment: str = "left",
):
    """Draw single or multiple text boxes.

    Args:
        texts (Union[str, List[str]]): Texts to draw.
        positions (np.ndarray: The position to draw
            the texts, which should have the same length with texts and
            each dim contain x and y.
        image_size (Optional[tuple]): image size to bound text drawing.
            (width, height)
        font_size (Union[int, List[int]], optional): The font size of
            texts.  Defaults to None.
        color (Union[str, tuple): The colors of texts.
        vertical_alignment (str): The verticalalignment
            of texts. verticalalignment controls whether the y positional
            argument for the text indicates the bottom, center or top side
            of the text bounding box.
        horizontal_alignment (str): The
            horizontalalignment of texts. Horizontalalignment controls
            whether the x positional argument for the text indicates the
            left, center or right side of the text bounding box.
    """
    font_scale = max(0.1, font_size / 30)
    thickness = max(1, font_size // 15)

    text_size, text_baseline = cv2.getTextSize(
        texts, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness
    )

    x = int(positions[0])
    if horizontal_alignment == "right":
        x = max(0, x - text_size[0])
    y = int(positions[1])
    if vertical_alignment == "top":
        y = y + text_size[1]
        if image_size is not None:
            y = min(image_size[1], y)

    return cv2.putText(
        image, texts, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness - 1
    )


def draw_box(
    img,
    bbox=[],
    text="",
    box_color=(0, 255, 0),
    text_color=(0, 255, 0),
    font_scale=0.7,
    font_thickness=1,
):
    # BOX_MODE is XYXY_ABS for cv2.rectangle.
    pt1 = (int(bbox[0]), int(bbox[1]))
    pt2 = (int(bbox[2]), int(bbox[3]))
    img = cv2.rectangle(
        img,
        pt1,
        pt2,
        box_color,
        2,
    )
    if text:
        y, dy = int(bbox[1]) + 30, 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_origin = (pt1[0] + 2, pt1[1] + text_size[1] + 2)
        for line in text.split("\n"):
            img = cv2.putText(
                img,
                str(line),
                text_origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,  # FontScale.
                text_color,  # Color.
                font_thickness,  # Thickness.
                cv2.LINE_AA,
            )
            y += dy

    return img


def parse_pose_metainfo(metainfo: Union[str, Dict]):
    """Load meta information of pose dataset and check its integrity.

    Args:
        metainfo (dict): Raw data of pose meta information, which should
            contain following contents:

            - "pose_format" (str): The name of the pose format
            - "keypoint_info" (dict): The keypoint-related meta information,
                e.g., name, upper/lower body, and symmetry
            - "skeleton_info" (dict): The skeleton-related meta information,
                e.g., start/end keypoint of limbs
            - "joint_weights" (list[float]): The loss weights of keypoints
            - "sigmas" (list[float]): The keypoint distribution parameters
                to calculate OKS score. See `COCO keypoint evaluation
                <https://cocodataset.org/#keypoints-eval>`__.

            An example of metainfo is shown as follows.

            .. code-block:: none
                {
                    "pose_format": "coco",
                    "keypoint_info":
                    {
                        0:
                        {
                            "name": "nose",
                            "type": "upper",
                            "swap": "",
                            "color": [51, 153, 255],
                        },
                        1:
                        {
                            "name": "right_eye",
                            "type": "upper",
                            "swap": "left_eye",
                            "color": [51, 153, 255],
                        },
                        ...
                    },
                    "skeleton_info":
                    {
                        0:
                        {
                            "link": ("left_ankle", "left_knee"),
                            "color": [0, 255, 0],
                        },
                        ...
                    },
                    "joint_weights": [1., 1., ...],
                    "sigmas": [0.026, 0.025, ...],
                }


            A special case is that `metainfo` can have the key "from_file",
            which should be the path of a config file. In this case, the
            actual metainfo will be loaded by:

            .. code-block:: python
                metainfo = mmengine.Config.fromfile(metainfo['from_file'])

    Returns:
        Dict: pose meta information that contains following contents:

        - "dataset_name" (str): Same as ``"dataset_name"`` in the input
        - "num_keypoints" (int): Number of keypoints
        - "keypoint_id2name" (dict): Mapping from keypoint id to name
        - "keypoint_name2id" (dict): Mapping from keypoint name to id
        - "upper_body_ids" (list): Ids of upper-body keypoint
        - "lower_body_ids" (list): Ids of lower-body keypoint
        - "flip_indices" (list): The Id of each keypoint's symmetric keypoint
        - "flip_pairs" (list): The Ids of symmetric keypoint pairs
        - "keypoint_colors" (numpy.ndarray): The keypoint color matrix of
            shape [K, 3], where each row is the color of one keypint in bgr
        - "num_skeleton_links" (int): The number of links
        - "skeleton_links" (list): The links represented by Id pairs of start
             and end points
        - "skeleton_link_colors" (numpy.ndarray): The link color matrix
        - "dataset_keypoint_weights" (numpy.ndarray): Same as the
            ``"joint_weights"`` in the input
        - "sigmas" (numpy.ndarray): Same as the ``"sigmas"`` in the input
    """

    if type(metainfo) == str:
        if not os.path.isfile(metainfo):
            raise ValueError("Invalid metainfo file path: ", metainfo)
        metainfo = OmegaConf.to_container(LazyConfig.load(metainfo).pose_info)

    # check data integrity
    assert "pose_format" in metainfo
    assert "keypoint_info" in metainfo
    assert "skeleton_info" in metainfo
    assert "joint_weights" in metainfo
    assert "sigmas" in metainfo

    # parse metainfo
    parsed = dict(
        pose_format=None,
        num_keypoints=None,
        keypoint_id2name={},
        keypoint_name2id={},
        upper_body_ids=[],
        lower_body_ids=[],
        flip_indices=[],
        flip_pairs=[],
        keypoint_colors=[],
        num_skeleton_links=None,
        skeleton_links=[],
        skeleton_link_colors=[],
        dataset_keypoint_weights=None,
        sigmas=None,
    )

    parsed["pose_format"] = metainfo["pose_format"]

    if "remove_teeth" in metainfo:
        parsed["remove_teeth"] = metainfo["remove_teeth"]

    if "min_visible_keypoints" in metainfo:
        parsed["min_visible_keypoints"] = metainfo["min_visible_keypoints"]

    if "teeth_keypoint_ids" in metainfo:
        parsed["teeth_keypoint_ids"] = metainfo["teeth_keypoint_ids"]

    if "coco_wholebody_to_goliath_mapping" in metainfo:
        parsed["coco_wholebody_to_goliath_mapping"] = metainfo[
            "coco_wholebody_to_goliath_mapping"
        ]

    if "coco_wholebody_to_goliath_keypoint_info" in metainfo:
        parsed["coco_wholebody_to_goliath_keypoint_info"] = metainfo[
            "coco_wholebody_to_goliath_keypoint_info"
        ]

    # parse keypoint information
    parsed["num_keypoints"] = len(metainfo["keypoint_info"])

    for kpt_id, kpt in metainfo["keypoint_info"].items():
        kpt_name = kpt["name"]
        parsed["keypoint_id2name"][kpt_id] = kpt_name
        parsed["keypoint_name2id"][kpt_name] = kpt_id
        parsed["keypoint_colors"].append(kpt.get("color", [255, 128, 0]))

        kpt_type = kpt.get("type", "")
        if kpt_type == "upper":
            parsed["upper_body_ids"].append(kpt_id)
        elif kpt_type == "lower":
            parsed["lower_body_ids"].append(kpt_id)

        swap_kpt = kpt.get("swap", "")
        if swap_kpt == kpt_name or swap_kpt == "":
            parsed["flip_indices"].append(kpt_name)
        else:
            parsed["flip_indices"].append(swap_kpt)
            pair = (swap_kpt, kpt_name)
            if pair not in parsed["flip_pairs"]:
                parsed["flip_pairs"].append(pair)

    # parse skeleton information
    parsed["num_skeleton_links"] = len(metainfo["skeleton_info"])
    for _, sk in metainfo["skeleton_info"].items():
        parsed["skeleton_links"].append(sk["link"])
        parsed["skeleton_link_colors"].append(sk.get("color", [96, 96, 255]))

    # parse extra information
    parsed["dataset_keypoint_weights"] = np.array(
        metainfo["joint_weights"], dtype=np.float32
    )
    parsed["sigmas"] = np.array(metainfo["sigmas"], dtype=np.float32)

    if "stats_info" in metainfo:
        parsed["stats_info"] = {}
        for name, val in metainfo["stats_info"].items():
            parsed["stats_info"][name] = np.array(val, dtype=np.float32)

    # formatting
    def _map(src, mapping: dict):
        if isinstance(src, (list, tuple)):
            cls = type(src)
            return cls(_map(s, mapping) for s in src)
        else:
            return mapping[src]

    parsed["flip_pairs"] = _map(
        parsed["flip_pairs"], mapping=parsed["keypoint_name2id"]
    )
    parsed["flip_indices"] = _map(
        parsed["flip_indices"], mapping=parsed["keypoint_name2id"]
    )
    parsed["skeleton_links"] = _map(
        parsed["skeleton_links"], mapping=parsed["keypoint_name2id"]
    )

    parsed["keypoint_colors"] = np.array(parsed["keypoint_colors"], dtype=np.uint8)
    parsed["skeleton_link_colors"] = np.array(
        parsed["skeleton_link_colors"], dtype=np.uint8
    )

    return parsed
