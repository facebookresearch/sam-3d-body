# Copyright (c) Meta Platforms, Inc. and affiliates.

from typing import Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch.nn as nn
import torchvision.transforms.functional as F
from PIL import Image, ImageOps
from sam_3d_body.models.modules import to_2tuple

from .bbox_utils import (
    bbox_cs2xyxy,
    bbox_xywh2cs,
    bbox_xyxy2cs,
    fix_aspect_ratio,
    get_udp_warp_matrix,
    get_warp_matrix,
)


class Compose:
    """Compose multiple transforms sequentially.

    Args:
        transforms (Sequence[dict, callable], optional): Sequence of transform
            object or config dict to be composed.
    """

    def __init__(self, transforms: Optional[List[Callable]] = None):
        if transforms is None:
            transforms = []
        else:
            self.transforms = transforms

    def __call__(self, data: dict) -> Optional[dict]:
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
           dict: Transformed data.
        """
        for t in self.transforms:
            data = t(data)
            # The transform will return None when it failed to load images or
            # cannot find suitable augmentation parameters to augment the data.
            # Here we simply return None if the transform returns None and the
            # dataset will handle it by randomly selecting another data sample.
            if data is None:
                return None
        return data

    def __repr__(self):
        """Print ``self.transforms`` in sequence.

        Returns:
            str: Formatted string.
        """
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


class VisionTransformWrapper:
    """A wrapper to use torchvision transform functions in this codebase."""

    def __init__(self, transform: Callable):
        self.transform = transform

    def __call__(self, results: Dict) -> Optional[dict]:
        results["img"] = self.transform(results["img"])
        return results

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.transform.__class__.__name__
        return repr_str


class GetBBoxCenterScale(nn.Module):
    """Convert bboxes to center and scale.

    The center is the coordinates of the bbox center, and the scale is the
    bbox width and height normalized by a scale factor.

    Required Keys:

        - bbox
        - bbox_format

    Added Keys:

        - bbox_center
        - bbox_scale

    Args:
        padding (float): The bbox padding scale that will be multilied to
            `bbox_scale`. Defaults to 1.25
    """

    def __init__(self, padding: float = 1.25) -> None:
        super().__init__()

        self.padding = padding

    def forward(self, results: Dict) -> Optional[dict]:
        """The transform function of :class:`GetBBoxCenterScale`.

        Args:
            results (dict): The result dict

        Returns:
            dict: The result dict.
        """
        if "bbox_center" in results and "bbox_scale" in results:
            results["bbox_scale"] *= self.padding
        else:
            bbox = results["bbox"]
            bbox_format = results.get("bbox_format", "none")
            if bbox_format == "xywh":
                center, scale = bbox_xywh2cs(bbox, padding=self.padding)
            elif bbox_format == "xyxy":
                center, scale = bbox_xyxy2cs(bbox, padding=self.padding)
            else:
                raise ValueError(
                    "Invalid bbox format: {}".format(results["bbox_format"])
                )

            results["bbox_center"] = center
            results["bbox_scale"] = scale
        return results

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__ + f"(padding={self.padding})"
        return repr_str


class SquarePad:
    def __call__(self, results: Dict) -> Optional[dict]:
        assert isinstance(results["img"], Image.Image)
        w, h = results["img"].size

        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, max_wh - w - hp, max_wh - h - vp)

        results["img"] = F.pad(results["img"], padding, 0, "constant")
        return results

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        return repr_str


class ToPIL:
    def __call__(self, results: Dict) -> Optional[dict]:
        if isinstance(results["img"], list):
            if isinstance(results["img"][0], np.ndarray):
                results["img"] = [Image.fromarray(img) for img in results["img"]]
        elif isinstance(results["img"], np.ndarray):
            results["img"] = Image.fromarray(results["img"])


class ToCv2:
    def __call__(self, results: Dict) -> Optional[dict]:
        if isinstance(results["img"], list):
            if isinstance(results["img"][0], Image.Image):
                results["img"] = [np.array(img) for img in results["img"]]
        elif isinstance(results["img"], Image.Image):
            results["img"] = np.array(results["img"])


class TopdownAffine(nn.Module):
    """Get the bbox image as the model input by affine transform.

    Required Keys:
        - img
        - bbox_center
        - bbox_scale
        - bbox_rotation (optional)
        - keypoints_2d (optional)
        - mask (optional)

    Modified Keys:
        - img
        - bbox_scale

    Added Keys:
        - input_size
        - transformed_keypoints

    Args:
        input_size (Tuple[int, int]): The input image size of the model in
            [w, h]. The bbox region will be cropped and resize to `input_size`
        use_udp (bool): Whether use unbiased data processing. See
            `UDP (CVPR 2020)`_ for details. Defaults to ``False``
        aspect_ratio (float): both HMR2.0 and Sapiens will expand input bbox to
            a fixed ratio (width/height = 192/256), then expand to the ratio of
            the model input size. E.g., HMR2.0 will eventually expand to 1:1, while
            Sapiens will be 768:1024.

    .. _`UDP (CVPR 2020)`: https://arxiv.org/abs/1911.07524
    """

    def __init__(
        self,
        input_size: Union[int, Tuple[int, int]],
        use_udp: bool = False,
        aspect_ratio: float = 0.75,
        fix_square: bool = False,
    ) -> None:
        super().__init__()

        self.input_size = to_2tuple(input_size)
        self.use_udp = use_udp
        self.aspect_ratio = aspect_ratio
        self.fix_square = fix_square

    def forward(self, results: Dict) -> Optional[dict]:
        """The transform function of :class:`TopdownAffine`.

        See ``transform()`` method of :class:`BaseTransform` for details.

        Args:
            results (dict): The result dict

        Returns:
            dict: The result dict.
        """
        # # Debug only
        # import copy
        # results['ori_img'] = np.zeros((2000, 2000, 3), dtype=np.uint8)
        # results['ori_img'][:results['img'].shape[0], :results['img'].shape[1]] = copy.deepcopy(results['img'])

        w, h = self.input_size
        warp_size = (int(w), int(h))

        # expand bbox to fixed aspect ratio
        results["orig_bbox_scale"] = results["bbox_scale"].copy()
        if self.fix_square and results["bbox_scale"][0] == results["bbox_scale"][1]:
            # In HMR2.0 etc, no fexpand_aspect_ratio for square bbox
            bbox_scale = fix_aspect_ratio(results["bbox_scale"], aspect_ratio=w / h)
        else:
            # first to a prior aspect ratio, then reshape to model input size
            bbox_scale = fix_aspect_ratio(
                results["bbox_scale"], aspect_ratio=self.aspect_ratio
            )
            results["bbox_scale"] = fix_aspect_ratio(bbox_scale, aspect_ratio=w / h)
        results["bbox_expand_factor"] = (
            results["bbox_scale"].max() / results["orig_bbox_scale"].max()
        )
        rot = 0.0
        if results["bbox_center"].ndim == 2:
            assert results["bbox_center"].shape[0] == 1, (
                "Only support cropping one instance at a time. Got invalid "
                f'shape of bbox_center {results["bbox_center"].shape}.'
            )
            center = results["bbox_center"][0]
            scale = results["bbox_scale"][0]
            if "bbox_rotation" in results:
                rot = results["bbox_rotation"][0]
        else:
            center = results["bbox_center"]
            scale = results["bbox_scale"]
            if "bbox_rotation" in results:
                rot = results["bbox_rotation"]

        if self.use_udp:
            warp_mat = get_udp_warp_matrix(center, scale, rot, output_size=(w, h))
        else:
            warp_mat = get_warp_matrix(center, scale, rot, output_size=(w, h))

        if "img" not in results:
            pass
        elif isinstance(results["img"], list):
            results["img"] = [
                cv2.warpAffine(img, warp_mat, warp_size, flags=cv2.INTER_LINEAR)
                for img in results["img"]
            ]
            height, width = results["img"][0].shape[:2]
            results["ori_img_size"] = np.array([width, height])
        else:
            height, width = results["img"].shape[:2]
            results["ori_img_size"] = np.array([width, height])
            results["img"] = cv2.warpAffine(
                results["img"], warp_mat, warp_size, flags=cv2.INTER_LINEAR
            )

        if results.get("keypoints_2d", None) is not None:
            results["orig_keypoints_2d"] = results["keypoints_2d"].copy()
            transformed_keypoints = results["keypoints_2d"].copy()
            # Only transform (x, y) coordinates
            # cv2 expect the input to be [[[x1, y1], [x2, y2]]]
            transformed_keypoints[:, :2] = cv2.transform(
                results["keypoints_2d"][None, :, :2], warp_mat
            )[0]
            results["keypoints_2d"] = transformed_keypoints

        if results.get("mask", None) is not None:
            results["mask"] = cv2.warpAffine(
                results["mask"], warp_mat, warp_size, flags=cv2.INTER_LINEAR
            )

        results["img_size"] = np.array([w, h])
        results["input_size"] = np.array([w, h])
        results["affine_trans"] = warp_mat
        return results

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        repr_str += f"(input_size={self.input_size}, "
        repr_str += f"use_udp={self.use_udp})"
        return repr_str


class NormalizeKeypoint(nn.Module):
    """
    Normalize 2D keypoints to range [-0.5, 0.5].

    Required Keys:
        - keypoints_2d
        - img_size

    Modified Keys:
        - keypoints_2d
    """

    def forward(self, results: Dict) -> Optional[dict]:
        if "keypoints_2d" in results:
            img_size = results.get("img_size", results["input_size"])

            results["keypoints_2d"][:, :2] = (
                results["keypoints_2d"][:, :2] / np.array(img_size).reshape(1, 2) - 0.5
            )
        return results
