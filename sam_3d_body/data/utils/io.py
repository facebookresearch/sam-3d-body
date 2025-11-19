# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import time
from typing import Any, List

import braceexpand
import cv2
import numpy as np

from PIL import Image


def expand(s):
    return os.path.expanduser(os.path.expandvars(s))


def expand_urls(urls: str | List[str]):
    if isinstance(urls, str):
        urls = [urls]
    urls = [u for url in urls for u in braceexpand.braceexpand(expand(url))]
    return urls


def load_image_from_file(
    data_info: dict,
    backend: str = "cv2",
    image_format: str = "rgb",
    retry: int = 10,
) -> dict:
    img = load_image(data_info["img_path"], backend, image_format, retry)
    data_info["img"] = img
    data_info["img_shape"] = img.shape[:2]
    data_info["ori_shape"] = img.shape[:2]
    return data_info


def _pil_load(path: str, image_format: str) -> Image.Image:
    with Image.open(path) as img:
        if img is not None and image_format.lower() == "rgb":
            img = img.convert("RGB")
    return img


def _cv2_load(path: str, image_format: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is not None and image_format.lower() == "rgb":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def load_image(
    path: str,
    backend: str = "pil",
    image_format: str = "rgb",
    retry: int = 10,
) -> Any:
    for i_try in range(retry):
        if backend == "pil":
            img = _pil_load(path, image_format)
        elif backend == "cv2":
            img = _cv2_load(path, image_format)
        else:
            raise ValueError("Invalid backend {} for loading image.".format(backend))

        if img is not None:
            return img
        else:
            print("Reading {} failed. Will retry.".format(path))
            time.sleep(1.0)
        if i_try == retry - 1:
            raise Exception("Failed to load image {}".format(path))


def resize_image(img, target_size, center=None, scale=None):
    height, width = img.shape[:2]
    aspect_ratio = width / height

    # Calculate the new size while maintaining the aspect ratio
    if aspect_ratio > 1:
        new_width = target_size
        new_height = int(target_size / aspect_ratio)
    else:
        new_width = int(target_size * aspect_ratio)
        new_height = target_size

    # Resize the image using OpenCV
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Create a new blank image with the target size
    final_img = np.ones((target_size, target_size, 3), dtype=np.uint8) * 255

    # Paste the resized image onto the blank image, centering it
    start_x = (target_size - new_width) // 2
    start_y = (target_size - new_height) // 2
    final_img[start_y : start_y + new_height, start_x : start_x + new_width] = (
        resized_img
    )

    if center is not None and scale is not None:
        ratio_width = new_width / width
        ratio_height = new_height / height

        new_scale = np.stack(
            [scale[:, 0] * ratio_width, scale[:, 1] * ratio_height], axis=1
        )
        new_center = np.stack(
            [center[:, 0] * ratio_width, center[:, 1] * ratio_height], axis=1
        )
        new_center[:, 0] += start_x
        new_center[:, 1] += start_y
    else:
        new_center, new_scale = None, None
    return aspect_ratio, final_img, new_center, new_scale
