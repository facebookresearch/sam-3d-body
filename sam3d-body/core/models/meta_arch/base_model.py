"""Define an abstract base model for consistent format input / processing / output."""

from abc import abstractmethod
from functools import partial
from typing import Dict, Optional

import torch
from yacs.config import CfgNode

from ..optim.fp16_utils import convert_module_to_f16, convert_to_fp16_safe

from .base_lightning_module import BaseLightningModule


def get_cam_intrinsics(model, batch):
    batch_size = batch["img_full"].shape[0]

    # Initialize camera intrinsics
    img_h, img_w = batch["ori_img_size"][:, 0, 1], batch["ori_img_size"][:, 0, 0]
    cam_intrinsics = torch.zeros((batch_size, 3, 3)).to(batch["img"])
    cam_intrinsics[:, 0, 2] = img_w / 2
    cam_intrinsics[:, 1, 2] = img_h / 2
    cam_intrinsics[:, 2, 2] = 1

    # Get Camera intrinsics using HumanFoV Model
    img_full_resized = model.normalize_img(batch["img_full"])
    model.cam_model.eval()
    with torch.no_grad():
        estimated_fov, _ = model.cam_model(img_full_resized)
    vfov = estimated_fov[:, 1]
    fl_h = img_h / (2 * torch.tan(vfov / 2))
    cam_intrinsics[:, 0, 0] = fl_h
    cam_intrinsics[:, 1, 1] = fl_h

    return cam_intrinsics


class BaseModel(BaseLightningModule):
    def __init__(self, cfg: Optional[CfgNode], **kwargs):
        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters(logger=False)
        self.cfg = cfg

        self._initialze_model(**kwargs)

        # Initialize attributes for image-based batch format
        self._max_num_person = None
        self._person_valid = None

    @abstractmethod
    def _initialze_model(self, **kwargs) -> None:
        pass

    def data_preprocess(
        self,
        inputs: torch.Tensor,
        crop_width: bool = False,
        is_full: bool = False,  # whether for full_branch
    ) -> torch.Tensor:
        image_mean = self.image_mean if not is_full else self.full_image_mean
        image_std = self.image_std if not is_full else self.full_image_std

        if inputs.max() > 1 and image_mean.max() <= 1.0:
            inputs = inputs / 255.0
        elif inputs.max() <= 1.0 and image_mean.max() > 1:
            inputs = inputs * 255.0
        batch_inputs = (inputs - image_mean) / image_std

        if crop_width:
            if self.cfg.MODEL.BACKBONE.TYPE in ["vit_hmr"]:
                # HMR2.0 backbone (ViTPose) assumes a different aspect ratio as input size
                batch_inputs = batch_inputs[:, :, :, 32:-32]
            elif self.cfg.MODEL.BACKBONE.TYPE in ["vit_hmr_512_384"]:
                batch_inputs = batch_inputs[:, :, :, 64:-64]
            else:
                raise Exception

        return batch_inputs

    def _initialize_batch(self, batch: Dict) -> None:
        # Check whether the input batch is with format
        # [batch_size, num_person, ...]
        if batch["img"].dim() == 5:
            self._batch_size, self._max_num_person = batch["img"].shape[:2]
            self._person_valid = self._flatten_person(batch["person_valid"]) > 0
        else:
            self._batch_size = batch["img"].shape[0]
            self._max_num_person = 0
            self._person_valid = None

    def _flatten_person(self, x: torch.Tensor) -> torch.Tensor:
        assert self._max_num_person is not None, "No max_num_person initialized"

        if self._max_num_person:
            # Merge person crops to batch dimension
            shape = x.shape
            x = x.view(self._batch_size * self._max_num_person, *shape[2:])
        return x

    def _unflatten_person(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        if self._max_num_person:
            x = x.view(self._batch_size, self._max_num_person, *shape[1:])
        return x

    def _get_valid(self, x: torch.Tensor) -> torch.Tensor:
        assert self._max_num_person is not None, "No max_num_person initialized"

        if self._person_valid is not None:
            x = x[self._person_valid]
        return x

    def _full_to_crop(
        self, batch: Dict, pred_keypoints_2d: torch.Tensor
    ) -> torch.Tensor:
        """Convert full-image keypoints coordinates to crop and normalize to [-0.5. 0.5]"""
        pred_keypoints_2d_cropped = torch.cat(
            [pred_keypoints_2d, torch.ones_like(pred_keypoints_2d[:, :, [-1]])], dim=-1
        )
        affine_trans = self._flatten_person(batch["affine_trans"]).to(
            pred_keypoints_2d_cropped
        )
        img_size = self._flatten_person(batch["img_size"]).unsqueeze(1)
        pred_keypoints_2d_cropped = pred_keypoints_2d_cropped @ affine_trans.mT
        pred_keypoints_2d_cropped = pred_keypoints_2d_cropped[..., :2] / img_size - 0.5

        return pred_keypoints_2d_cropped

    def _cam_full_to_crop(
        self, batch: Dict, pred_cam_t: torch.Tensor, focal_length: torch.Tensor = None
    ) -> torch.Tensor:
        """Revert the camera translation from full to crop image space"""
        num_person = batch["img"].shape[1]
        cam_int = self._flatten_person(
            batch["cam_int"].unsqueeze(1).expand(-1, num_person, -1, -1).contiguous()
        )
        bbox_center = self._flatten_person(batch["bbox_center"])
        bbox_size = self._flatten_person(batch["bbox_scale"])[:, 0]
        img_size = self._flatten_person(batch["ori_img_size"])
        input_size = self._flatten_person(batch["img_size"])[:, 0]

        tx, ty, tz = pred_cam_t[:, 0], pred_cam_t[:, 1], pred_cam_t[:, 2]
        if focal_length is None:
            focal_length = cam_int[:, 0, 0]
        bs = 2 * focal_length / (tz + 1e-8)

        if not self.cfg.MODEL.DECODER.get("USE_INTRIN_CENTER", False):
            cx = 2 * (bbox_center[:, 0] - (img_size[:, 0] / 2)) / bs
            cy = 2 * (bbox_center[:, 1] - (img_size[:, 1] / 2)) / bs
        else:
            cx = 2 * (bbox_center[:, 0] - (cam_int[:, 0, 2])) / bs
            cy = 2 * (bbox_center[:, 1] - (cam_int[:, 1, 2])) / bs

        crop_cam_t = torch.stack(
            [tx - cx, ty - cy, tz * bbox_size / input_size], dim=-1
        )
        return crop_cam_t

    def convert_to_fp16(self) -> torch.dtype:
        """
        Convert the torso of the model to float16.
        """

        # fp16_type = torch.float16

        fp16_type = (
            torch.float16
            if self.cfg.TRAIN.get("FP16_TYPE", "float16") == "float16"
            else torch.bfloat16
        )

        if hasattr(self, "backbone"):
            self._set_fp16(self.backbone, fp16_type)
        if hasattr(self, "full_encoder"):
            self._set_fp16(self.full_encoder, fp16_type)

        return fp16_type

    def _set_fp16(self, module, fp16_type):
        if hasattr(module, "pos_embed"):
            module.apply(partial(convert_module_to_f16, dtype=fp16_type))
            module.pos_embed.data = module.pos_embed.data.to(fp16_type)
        elif hasattr(module, "positional_embedding"):
            # Perception Encoder Only
            module.apply(partial(convert_to_fp16_safe, dtype=fp16_type))
            module.positional_embedding.data = module.positional_embedding.data.to(
                fp16_type
            )
        elif hasattr(module.encoder, "rope_embed"):
            # DINOv3
            module.encoder.apply(partial(convert_to_fp16_safe, dtype=fp16_type))
            module.encoder.rope_embed = module.encoder.rope_embed.to(fp16_type)
        else:
            # DINOv2
            module.encoder.pos_embed.data = module.encoder.pos_embed.data.to(fp16_type)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        raise NotImplementedError
