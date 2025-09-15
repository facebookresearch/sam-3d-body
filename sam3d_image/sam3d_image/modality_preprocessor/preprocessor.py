import warnings
import torch

from dataclasses import dataclass
from typing import Callable
from dataclasses import dataclass
from .img_and_mask_transforms import (
    normalize_pointmap_ssi,
)


@dataclass
class PreProcessor:
    """
    Preprocessor configuration for image, mask, and pointmap transforms.

    Transform application order:
    1. Pointmap normalization (if normalize_pointmap=True)
    2. Joint transforms (img_mask_pointmap_joint_transform or img_mask_joint_transform)
    3. Individual transforms (img_transform, mask_transform, pointmap_transform)

    For backward compatibility, img_mask_joint_transform is preserved. When both
    img_mask_pointmap_joint_transform and img_mask_joint_transform are present,
    img_mask_pointmap_joint_transform takes priority.
    """

    img_transform: Callable = (None,)
    mask_transform: Callable = (None,)
    img_mask_joint_transform: list[Callable] = (None,)

    # New fields for pointmap support
    pointmap_transform: Callable = (None,)
    img_mask_pointmap_joint_transform: list[Callable] = (None,)

    # Pointmap normalization option
    normalize_pointmap: bool = False

    def __call__(
        self, rgb_image, rgb_image_mask, pointmap=None
    ):
        """Extended version that handles pointmaps"""

        # TODO(Sasha): fix this it's so ugly.
        # Apply pointmap normalization if enabled
        pointmap_scale = None
        pointmap_shift = None
        if pointmap is not None and self.normalize_pointmap:
            pointmap, pointmap_scale, pointmap_shift = normalize_pointmap_ssi(pointmap)

        # These two are typically used for getting cropped images of the object
        #   : first apply joint transforms
        processed_rgb_image, processed_mask, processed_pointmap = (
            self._preprocess_image_mask_pointmap(rgb_image, rgb_image_mask, pointmap)
        )
        #   : then apply individual transforms on top of the joint transforms
        processed_rgb_image = self._apply_transform(
            processed_rgb_image, self.img_transform
        )
        processed_mask = self._apply_transform(processed_mask, self.mask_transform)
        if processed_pointmap is not None:
            processed_pointmap = self._apply_transform(
                processed_pointmap, self.pointmap_transform
            )

        # This version is typically the full version of the image
        #   : apply individual transforms only
        rgb_image = self._apply_transform(rgb_image, self.img_transform)
        rgb_image_mask = self._apply_transform(rgb_image_mask, self.mask_transform)
        full_pointmap = None
        if pointmap is not None:
            full_pointmap = self._apply_transform(pointmap, self.pointmap_transform)

        result = {
            "mask": processed_mask,
            "image": processed_rgb_image,
            "rgb_image": rgb_image,
            "rgb_image_mask": rgb_image_mask,
        }

        # Add pointmap results if available
        if processed_pointmap is not None:
            result.update(
                {
                    "pointmap": processed_pointmap,
                    "rgb_pointmap": full_pointmap,
                }
            )

        # Add normalization parameters if normalization was applied
        if pointmap_scale is not None and pointmap_shift is not None:
            result.update(
                {
                    "pointmap_scale": pointmap_scale,
                    "pointmap_shift": pointmap_shift,
                }
            )

        return result

    def _process_image_and_mask(self, rgb_image, rgb_image_mask):
        """Original method - calls extended version without pointmap"""
        return self.__call__(rgb_image, rgb_image_mask, None)

    def _preprocess_image_mask_pointmap(self, rgb_image, mask_image, pointmap=None):
        """Apply joint transforms with priority: triple transforms > dual transforms."""
        # Priority: img_mask_pointmap_joint_transform when pointmap is provided
        if (
            self.img_mask_pointmap_joint_transform != (None,)
            and self.img_mask_pointmap_joint_transform is not None
            and pointmap is not None
        ):
            for trans in self.img_mask_pointmap_joint_transform:
                rgb_image, mask_image, pointmap = trans(
                    rgb_image, mask_image, pointmap=pointmap
                )
            return rgb_image, mask_image, pointmap

        # Fallback: img_mask_joint_transform (existing behavior)
        elif (
            self.img_mask_joint_transform != (None,)
            and self.img_mask_joint_transform is not None
        ):
            for trans in self.img_mask_joint_transform:
                rgb_image, mask_image = trans(rgb_image, mask_image)
            return rgb_image, mask_image, pointmap

        return rgb_image, mask_image, pointmap

    def _preprocess_image_and_mask(self, rgb_image, mask_image):
        """Backward compatibility wrapper - only applies dual transforms"""
        rgb_image, mask_image, _ = self._preprocess_image_mask_pointmap(
            rgb_image, mask_image, None
        )
        return rgb_image, mask_image

    # keep here for backward compatibility
    def _preprocess_image_and_mask_inference(self, rgb_image, mask_image):
        warnings.warn(
            "The _preprocess_image_and_mask_inference is deprecated! Please use _preprocess_image_and_mask",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return self._preprocess_image_and_mask(rgb_image, mask_image)

    def _apply_transform(self, input: torch.Tensor, transform):
        if input is not None and transform is not None and transform != (None,):
            input = transform(input)

        return input

