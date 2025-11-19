# Copyright (c) Meta Platforms, Inc. and affiliates.

from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np

from .utils import draw_text, parse_pose_metainfo


class SkeletonVisualizer:
    def __init__(
        self,
        bbox_color: Optional[Union[str, Tuple[int]]] = "green",
        kpt_color: Optional[Union[str, Tuple[Tuple[int]]]] = "red",
        link_color: Optional[Union[str, Tuple[Tuple[int]]]] = None,
        text_color: Optional[Union[str, Tuple[int]]] = (255, 255, 255),
        line_width: Union[int, float] = 1,
        radius: Union[int, float] = 3,
        alpha: float = 1.0,
        show_keypoint_weight: bool = False,
    ):
        self.bbox_color = bbox_color
        self.kpt_color = kpt_color
        self.link_color = link_color
        self.line_width = line_width
        self.text_color = text_color
        self.radius = radius
        self.alpha = alpha
        self.show_keypoint_weight = show_keypoint_weight

        # Pose specific meta info if available.
        self.pose_meta = {}
        self.skeleton = None

    def set_pose_meta(self, pose_meta: Dict):
        parsed_meta = parse_pose_metainfo(pose_meta)

        self.pose_meta = parsed_meta.copy()
        self.bbox_color = parsed_meta.get("bbox_color", self.bbox_color)
        self.kpt_color = parsed_meta.get("keypoint_colors", self.kpt_color)
        self.link_color = parsed_meta.get("skeleton_link_colors", self.link_color)
        self.skeleton = parsed_meta.get("skeleton_links", self.skeleton)

    def draw_skeleton(
        self,
        image: np.ndarray,
        keypoints: np.ndarray,
        kpt_thr: float = 0.3,
        show_kpt_idx: bool = False,
    ):
        """Draw keypoints and skeletons (optional) of GT or prediction.

        Args:
            image (np.ndarray): The image to draw.
            keypoints (np.ndarray): B x N x 3
            kpt_thr (float, optional): Minimum threshold of keypoints
                to be shown. Default: 0.3.
            show_kpt_idx (bool): Whether to show the index of keypoints.
                Defaults to ``False``

        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """

        image = image.copy()
        img_h, img_w, _ = image.shape
        if len(keypoints.shape) == 2:
            keypoints = keypoints[None, :, :]

        # loop for each person
        for cur_keypoints in keypoints:
            kpts = cur_keypoints[:, :-1]
            score = cur_keypoints[:, -1]

            if self.kpt_color is None or isinstance(self.kpt_color, str):
                kpt_color = [self.kpt_color] * len(kpts)
            elif len(self.kpt_color) == len(kpts):
                kpt_color = self.kpt_color
            else:
                raise ValueError(
                    f"the length of kpt_color "
                    f"({len(self.kpt_color)}) does not matches "
                    f"that of keypoints ({len(kpts)})"
                )

            # draw links
            if self.skeleton is not None and self.link_color is not None:
                if self.link_color is None or isinstance(self.link_color, str):
                    link_color = [self.link_color] * len(self.skeleton)
                elif len(self.link_color) == len(self.skeleton):
                    link_color = self.link_color
                else:
                    raise ValueError(
                        f"the length of link_color "
                        f"({len(self.link_color)}) does not matches "
                        f"that of skeleton ({len(self.skeleton)})"
                    )

                for sk_id, sk in enumerate(self.skeleton):
                    pos1 = (int(kpts[sk[0], 0]), int(kpts[sk[0], 1]))
                    pos2 = (int(kpts[sk[1], 0]), int(kpts[sk[1], 1]))
                    if (
                        pos1[0] <= 0
                        or pos1[0] >= img_w
                        or pos1[1] <= 0
                        or pos1[1] >= img_h
                        or pos2[0] <= 0
                        or pos2[0] >= img_w
                        or pos2[1] <= 0
                        or pos2[1] >= img_h
                        or score[sk[0]] < kpt_thr
                        or score[sk[1]] < kpt_thr
                        or link_color[sk_id] is None
                    ):
                        # skip the link that should not be drawn
                        continue

                    color = link_color[sk_id]
                    if not isinstance(color, str):
                        color = tuple(int(c) for c in color)
                    transparency = self.alpha
                    if self.show_keypoint_weight:
                        transparency *= max(
                            0, min(1, 0.5 * (score[sk[0]] + score[sk[1]]))
                        )

                    image = cv2.line(
                        image,
                        pos1,
                        pos2,
                        color,
                        thickness=self.line_width,
                    )

            # draw each point on image
            for kid, kpt in enumerate(kpts):
                if score[kid] < kpt_thr or kpt_color[kid] is None:
                    # skip the point that should not be drawn
                    continue

                color = kpt_color[kid]
                if not isinstance(color, str):
                    color = tuple(int(c) for c in color)
                transparency = self.alpha
                if self.show_keypoint_weight:
                    transparency *= max(0, min(1, score[kid]))

                if transparency == 1.0:
                    image = cv2.circle(
                        image,
                        (int(kpt[0]), int(kpt[1])),
                        int(self.radius),
                        color,
                        -1,
                    )
                else:
                    temp = image = cv2.circle(
                        image.copy(),
                        (int(kpt[0]), int(kpt[1])),
                        int(self.radius),
                        color,
                        -1,
                    )
                    image = cv2.addWeighted(
                        image, 1 - transparency, temp, transparency, 0
                    )

                if show_kpt_idx:
                    kpt[0] += self.radius
                    kpt[1] -= self.radius
                    image = draw_text(
                        image,
                        str(kid),
                        kpt,
                        image_size=(img_w, img_h),
                        color=color,
                        font_size=self.radius * 3,
                        vertical_alignment="bottom",
                        horizontal_alignment="center",
                    )

        return image

    def draw_skeleton_analysis(
        self,
        image: np.ndarray,
        keypoints: np.ndarray,
        kpt_thr: float = 0.3,
        show_kpt_idx: bool = False,
    ):
        """Draw keypoints and skeletons (optional) of prediction.
        The color is determined by whether the keypoint is correctly predicted.

        Args:
            image (np.ndarray): The image to draw.
            keypoints (np.ndarray): B x N x 4
            kpt_thr (float, optional): Minimum threshold of keypoints
                to be shown. Default: 0.3.
            show_kpt_idx (bool): Whether to show the index of keypoints.
                Defaults to ``False``

        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """

        image = image.copy()
        img_h, img_w, _ = image.shape
        if len(keypoints.shape) == 2:
            keypoints = keypoints[None, :, :]

        # loop for each person
        for cur_keypoints in keypoints:
            kpts = cur_keypoints[:, :-2]
            score = cur_keypoints[:, -2]
            correct = cur_keypoints[:, -1]

            kpt_color = [
                [0, 255, 0] if correct[kid] else [0, 0, 255] for kid in range(len(kpts))
            ]
            kpt_color = np.array(kpt_color, dtype=np.uint8)

            # draw links
            if self.skeleton is not None and self.link_color is not None:
                if self.link_color is None or isinstance(self.link_color, str):
                    link_color = [self.link_color] * len(self.skeleton)
                elif len(self.link_color) == len(self.skeleton):
                    link_color = self.link_color
                else:
                    raise ValueError(
                        f"the length of link_color "
                        f"({len(self.link_color)}) does not matches "
                        f"that of skeleton ({len(self.skeleton)})"
                    )

                for sk_id, sk in enumerate(self.skeleton):
                    pos1 = (int(kpts[sk[0], 0]), int(kpts[sk[0], 1]))
                    pos2 = (int(kpts[sk[1], 0]), int(kpts[sk[1], 1]))
                    if (
                        pos1[0] <= 0
                        or pos1[0] >= img_w
                        or pos1[1] <= 0
                        or pos1[1] >= img_h
                        or pos2[0] <= 0
                        or pos2[0] >= img_w
                        or pos2[1] <= 0
                        or pos2[1] >= img_h
                        or score[sk[0]] < kpt_thr
                        or score[sk[1]] < kpt_thr
                        or link_color[sk_id] is None
                    ):
                        # skip the link that should not be drawn
                        continue

                    color = link_color[sk_id]
                    if not isinstance(color, str):
                        color = tuple(int(c) for c in color)
                    transparency = self.alpha
                    if self.show_keypoint_weight:
                        transparency *= max(
                            0, min(1, 0.5 * (score[sk[0]] + score[sk[1]]))
                        )

                    image = cv2.line(
                        image,
                        pos1,
                        pos2,
                        color,
                        thickness=self.line_width,
                    )

            # draw each point on image
            for kid, kpt in enumerate(kpts):
                if score[kid] < kpt_thr or kpt_color[kid] is None:
                    # skip the point that should not be drawn
                    continue

                color = kpt_color[kid]
                if not isinstance(color, str):
                    color = tuple(int(c) for c in color)
                transparency = self.alpha
                if self.show_keypoint_weight:
                    transparency *= max(0, min(1, score[kid]))

                if transparency == 1.0:
                    image = cv2.circle(
                        image,
                        (int(kpt[0]), int(kpt[1])),
                        int(self.radius),
                        color,
                        -1,
                    )
                else:
                    temp = image = cv2.circle(
                        image.copy(),
                        (int(kpt[0]), int(kpt[1])),
                        int(self.radius),
                        color,
                        -1,
                    )
                    image = cv2.addWeighted(
                        image, 1 - transparency, temp, transparency, 0
                    )

                if show_kpt_idx:
                    kpt[0] += self.radius
                    kpt[1] -= self.radius
                    image = draw_text(
                        image,
                        str(kid),
                        kpt,
                        image_size=(img_w, img_h),
                        color=color,
                        font_size=self.radius * 3,
                        vertical_alignment="bottom",
                        horizontal_alignment="center",
                    )

        return image
