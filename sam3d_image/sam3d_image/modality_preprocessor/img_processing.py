import math

import random

import torch
import torch.nn.functional as F

from torchvision import transforms
from torchvision.transforms import functional as tv_F


def pad_to_square(image, value=0):
    _, _, h, w = image.shape  # Assuming image is in (B, C, H, W) format
    if h == w:
        return image  # The image is already square

    # Calculate the padding
    diff = abs(h - w)
    pad2 = diff

    # Pad the image to make it square
    if h > w:
        padding = (0, pad2, 0, 0)  # Pad width (left, right, top, bottom)
    else:
        padding = (0, 0, 0, pad2)  # Pad height
    # Apply padding
    padded_image = torch.nn.functional.pad(image, padding, mode="constant", value=value)
    return padded_image


def preprocess_img(
    x,
    mask=None,
    img_target_shape=224,
    mask_target_shape=256,
    normalize=False,
):
    if x.shape[1] != x.shape[2]:
        x = pad_to_square(x)
    if mask is not None and mask.shape[1] != mask.shape[2]:
        mask = pad_to_square(mask)
    if x.shape[2] != img_target_shape:
        x = F.interpolate(
            x,
            size=(img_target_shape, img_target_shape),
            # scale_factor=float(img_target_shape)/x.shape[2],
            mode="bilinear",
        )
    if mask is not None and mask.shape[2] != mask_target_shape:
        if mask is not None:
            mask = F.interpolate(
                mask,
                size=(mask_target_shape, mask_target_shape),
                # scale_factor=float(mask_target_shape)/mask.shape[2],
                mode="nearest",
            )
    if normalize:
        imgs_normed = resnet_img_normalization(x)
    else:
        imgs_normed = x
    return imgs_normed, mask


def resnet_img_normalization(x):
    resnet_mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).reshape(
        (3, 1, 1)
    )
    resnet_std = torch.tensor([0.229, 0.224, 0.225], device=x.device).reshape((3, 1, 1))
    if x.ndim == 4:
        resnet_mean = resnet_mean[None]
        resnet_std = resnet_std[None]
    x = (x - resnet_mean) / resnet_std
    return x


# pad image to be centered for unprojecting depth
def pad_to_square_centered(image, value=0, pointmap=None):
    h, w = image.shape[-2], image.shape[-1]  # Assuming image is in (B, C, H, W) format
    if h == w:
        if pointmap is not None:
            return image, pointmap
        return image  # The image is already square

    # Calculate the padding
    diff = abs(h - w)
    pad1 = diff // 2
    pad2 = diff - pad1

    # Pad the image to make it square
    if h > w:
        padding = (pad1, pad2, 0, 0)  # Pad width (left, right, top, bottom)
    else:
        padding = (0, 0, pad1, pad2)  # Pad height
    # Apply padding to image
    padded_image = F.pad(image, padding, mode="constant", value=value)

    # Apply padding to pointmap if provided
    if pointmap is not None:
        # Pad pointmap using torch functional with NaN fill value
        padded_pointmap = F.pad(pointmap, padding, mode="constant", value=float("nan"))

        return padded_image, padded_pointmap
    return padded_image


def crop_img_to_obj(mask, context_size):
    nonzeros = torch.nonzero(mask)
    if len(nonzeros) > 0:
        r_max, c_max = nonzeros.max(dim=0)[0]
        r_min, c_min = nonzeros.min(dim=0)[0]
        box_h = max(1, r_max - r_min)
        box_w = max(1, c_max - c_min)
        left = max(0, c_min - int(box_w * context_size))
        right = min(mask.shape[-1], c_max + int(box_w * context_size))
        top = max(0, r_min - int(box_h * context_size))
        bot = min(mask.shape[-2], r_max + int(box_h * context_size))
        return left, right, top, bot
    return None, None, None, None


def random_pad(img, mask=None, max_ratio=0.0, pointmap=None):
    max_size = int(max(img.shape) * max_ratio)
    padding = tuple([random.randint(0, max_size) for _ in range(4)])
    img = F.pad(img, padding)
    if mask is not None:
        mask = F.pad(mask, padding)

    if pointmap is not None:
        pointmap = F.pad(pointmap, padding, mode="constant", value=float("nan"))
        return img, mask, pointmap
    return img, mask


def get_img_color_augmentation(
    color_jit_prob=0.5,
    gaussian_blur_prob=0.1,
):
    transform = transforms.Compose(
        [
            # (a) Random Color Jitter
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                    )
                ],
                p=color_jit_prob,
            ),
            # (b) Randomly apply GaussianBlur
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))],
                p=gaussian_blur_prob,
            ),
        ]
    )
    return transform
