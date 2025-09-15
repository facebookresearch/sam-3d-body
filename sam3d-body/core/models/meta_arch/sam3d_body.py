import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.metadata import ATLAS70_TO_OPENPOSE, KEY_BODY, PROMPT_KEYPOINTS
from core.utils.checkpoint import load_state_dict
from core.utils.config import get_config

# from core.utils.config import get_config
from core.utils.logging import get_pylogger
from torchvision.ops import roi_align

from ..backbones import create_backbone
from ..decoders import build_decoder, build_keypoint_sampler, PromptEncoder
from ..heads import build_head
from ..modules import to_2tuple
from ..modules.transformer import FFN

from .base_model import BaseModel


logger = get_pylogger(__name__)


# fmt: off
PROMPT_KEYPOINTS = {  # keypoint_idx: prompt_idx
    "atlas70": {
        i: i for i in range(70)
    },  # all 70 keypoints are supported for prompting
    "atlas70_smpl": {  # for SMPL evaluation only, using the 61 keypoint format ("wds_1223")
        **ATLAS70_TO_OPENPOSE,
        **{
            33: 6, 32: 8, 31: 41, 34: 5, 35: 7, 36: 62, 26: 12, 25: 14, 29: 11, 30: 13, 50: 6,
            52: 8, 54: 41, 49: 5, 51: 7, 53: 62, 58: 12, 60: 14, 57: 11, 59: 13, 44: 0, 55: 9, 56: 10,
        },
    },
}
KEY_BODY = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 41, 62]  # key body joints for prompting
# fmt: on


def load_sam3d_body(checkpoint_path, estimate_cam_int=True):
    # Check the current directory, and if not present check the parent dir.
    model_cfg = os.path.join(os.path.dirname(checkpoint_path), "model_config.yaml")
    if not os.path.exists(model_cfg):
        # Looks at parent dir
        model_cfg = os.path.join(
            os.path.dirname(os.path.dirname(checkpoint_path)), "model_config.yaml"
        )

    model_cfg = get_config(model_cfg)

    # Disable face for inference
    model_cfg.defrost()
    model_cfg.MODEL.ATLAS_HEAD.ZERO_FACE = True
    model_cfg.freeze()

    model = SAM3DBody(model_cfg, estimate_cam_int=estimate_cam_int)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    load_state_dict(model, state_dict, strict=False)
    return model, model_cfg


class SAM3DBody(BaseModel):

    pelvis_idx = [9, 10]  # left_hip, right_hip

    def _initialze_model(self, estimate_cam_int=False):
        self.estimate_cam_int = estimate_cam_int
        self.register_buffer(
            "image_mean", torch.tensor(self.cfg.MODEL.IMAGE_MEAN).view(-1, 1, 1), False
        )
        self.register_buffer(
            "image_std", torch.tensor(self.cfg.MODEL.IMAGE_STD).view(-1, 1, 1), False
        )

        # Create the full-image encoder (frozen)
        if self.cfg.MODEL.FULL_ENCODER.ENABLE:
            self.full_encoder = create_backbone(self.cfg.MODEL.FULL_ENCODER.TYPE)
            for param in self.full_encoder.parameters():
                param.requires_grad = False
            self.register_buffer(
                "full_image_mean",
                torch.tensor(self.cfg.MODEL.FULL_ENCODER.IMAGE_MEAN).view(-1, 1, 1),
                False,
            )
            self.register_buffer(
                "full_image_std",
                torch.tensor(self.cfg.MODEL.FULL_ENCODER.IMAGE_STD).view(-1, 1, 1),
                False,
            )

            # Create the full-image decoder
            if self.cfg.MODEL.FULL_DECODER.ENABLE:
                if self.cfg.MODEL.FULL_HEAD.CAMERA_ENABLE:
                    self.full_camera = build_head(
                        self.cfg, self.cfg.MODEL.FULL_HEAD.CAMERA_TYPE
                    )
                    self.full_init_camera = nn.Embedding(1, self.full_camera.ncam)
                    nn.init.zeros_(self.full_init_camera.weight)
                if self.cfg.MODEL.FULL_HEAD.FOCAL_LENGTH_ENABLE:
                    raise NotImplementedError

                init_dim = (
                    self.full_encoder.embed_dims
                    if not self.cfg.MODEL.FULL_HEAD.CAMERA_ENABLE
                    else self.full_encoder.embed_dims + self.full_camera.ncam
                )
                self.init_to_token_full = nn.Linear(
                    init_dim, self.cfg.MODEL.FULL_DECODER.DIM
                )
                self.full_decoder = build_decoder(
                    self.cfg.MODEL.FULL_DECODER,
                    context_dim=self.full_encoder.embed_dims,
                )

        # Create backbone feature extractor for human crops
        self.backbone = create_backbone(
            self.cfg.MODEL.BACKBONE.TYPE,
            self.cfg,
            pretrained=False,
        )

        # Create header for pose estimation output
        self.head_pose = build_head(self.cfg, self.cfg.MODEL.PERSON_HEAD.POSE_TYPE)
        # Initialize pose token with learnable params (not mean pose in SMPL)
        # Note: bias/initial value should be zero-pose in cont, not all-zeros
        self.init_pose = nn.Embedding(1, self.head_pose.npose)
        if self.cfg.MODEL.PERSON_HEAD.ZERO_POSE_INIT:
            self.init_pose.weight.data = self.head_pose.get_zero_pose_init(
                self.cfg.MODEL.PERSON_HEAD.get("ZERO_POSE_INIT_BODY_FACTOR", 1)
            )
        else:
            raise NotImplementedError

        if self.cfg.MODEL.PERSON_HEAD.CAMERA_ENABLE:
            self.head_camera = build_head(
                self.cfg, self.cfg.MODEL.PERSON_HEAD.CAMERA_TYPE
            )
            self.init_camera = nn.Embedding(1, self.head_camera.ncam)
            nn.init.zeros_(self.init_camera.weight)

        self.camera_type = "perspective"

        # Support conditioned information for decoder
        if self.cfg.MODEL.DECODER.CONDITION_TYPE == "cliff":
            cond_dim = 3
        elif self.cfg.MODEL.DECODER.CONDITION_TYPE == "none":
            cond_dim = 0
        else:
            raise ValueError(
                "Invalid condition type", self.cfg.MODEL.DECODER.CONDITION_TYPE
            )
        init_dim = (
            self.head_pose.npose + cond_dim
            if not self.cfg.MODEL.PERSON_HEAD.CAMERA_ENABLE
            else self.head_pose.npose + self.head_camera.ncam + cond_dim
        )
        self.init_to_token_atlas = nn.Linear(init_dim, self.cfg.MODEL.DECODER.DIM)
        self.prev_to_token_atlas = nn.Linear(
            init_dim - cond_dim, self.cfg.MODEL.DECODER.DIM
        )

        # Create prompt encoder
        self.max_num_clicks = 0
        if self.cfg.MODEL.PROMPT_ENCODER.ENABLE:
            self.max_num_clicks = self.cfg.MODEL.PROMPT_ENCODER.MAX_NUM_CLICKS
            self.prompt_keypoints = PROMPT_KEYPOINTS[
                self.cfg.MODEL.PROMPT_ENCODER.PROMPT_KEYPOINTS
            ]

            self.prompt_encoder = PromptEncoder(
                embed_dim=self.backbone.embed_dims,  # need to match backbone dims for PE
                num_body_joints=len(set(self.prompt_keypoints.values())),
                frozen=self.cfg.MODEL.PROMPT_ENCODER.get("frozen", False),
                mask_embed_type=self.cfg.MODEL.PROMPT_ENCODER.get(
                    "MASK_EMBED_TYPE", None
                ),
            )
            self.prompt_to_token = nn.Linear(
                self.backbone.embed_dims, self.cfg.MODEL.DECODER.DIM
            )

            self.keypoint_prompt_sampler = build_keypoint_sampler(
                self.cfg.MODEL.PROMPT_ENCODER.get("KEYPOINT_SAMPLER", {}),
                prompt_keypoints=self.prompt_keypoints,
                keybody_idx=KEY_BODY,
            )
            # To keep track of prompting history
            self.prompt_hist = np.zeros(
                (len(set(self.prompt_keypoints.values())) + 2, self.max_num_clicks),
                dtype=np.float32,
            )

            if self.cfg.MODEL.DECODER.FROZEN:
                for param in self.prompt_to_token.parameters():
                    param.requires_grad = False

        # Create promptable decoder
        self.decoder = build_decoder(
            self.cfg.MODEL.DECODER, context_dim=self.backbone.embed_dims
        )

        # Manually convert the torso of the model to fp16.
        if self.cfg.TRAIN.USE_FP16:
            self.convert_to_fp16()
            if self.cfg.TRAIN.get("FP16_TYPE", "float16") == "float16":
                self.backbone_dtype = torch.float16
            else:
                self.backbone_dtype = torch.bfloat16
        else:
            self.convert_to_fp32()
            self.backbone_dtype = torch.float32

        if self.cfg.MODEL.get("RAY_CONDITION_TYPE", None) is not None:
            if self.cfg.MODEL.BACKBONE.TYPE in [
                "vit_b",
                "vit_l",
                "vit_hmr",
                "hmr2",
                "vit",
                "vit_hmr_256",
                "vit_hmr_512_384",
            ]:
                self.ray_cond_emb = nn.Conv2d(
                    2,
                    self.backbone.embed_dim,
                    self.backbone.patch_size,
                    stride=self.backbone.patch_size,
                )

            elif self.cfg.MODEL.BACKBONE.TYPE in [
                "pe_core_b",
                "pe_core_l",
                "pe_core_g",
                "pe_spatial_g",
                "pe_spatial_l",
                "pe_spatial_b",
                "pe_spatial_s",
                "pe_spatial_t",
            ]:
                self.ray_cond_emb = nn.Conv2d(
                    2,
                    self.backbone.embed_dims,
                    self.backbone.patch_size,
                    stride=self.backbone.patch_size,
                )

            elif self.cfg.MODEL.BACKBONE.TYPE in [
                "dinov3_vit7b",
                "dinov3_vith16plus",
                "dinov3_vits16",
                "dinov3_vits16plus",
                "dinov3_vitb16",
                "dinov3_vitl16",
            ]:
                self.ray_cond_emb = nn.Conv2d(
                    2,
                    self.backbone.encoder.embed_dim,
                    self.backbone.encoder.patch_size,
                    stride=self.backbone.encoder.patch_size,
                )

            else:
                assert False, "Not implemented yet"

            # Zero conv
            torch.nn.init.zeros_(self.ray_cond_emb.weight)
            torch.nn.init.zeros_(self.ray_cond_emb.bias)

        if self.cfg.MODEL.DECODER.get("DO_KEYPOINT_TOKENS", False):
            # TODO: Tie with prompt encoder?
            if not self.cfg.MODEL.DECODER.get("DO_KEYPOINT_TOKENS_BODY_ONLY", False):
                # Do all KPS
                self.keypoint_embedding_idxs = list(range(70))
                self.keypoint_embedding = nn.Embedding(
                    len(self.keypoint_embedding_idxs), self.cfg.MODEL.DECODER.DIM
                )
            else:
                # Just body (including wrist)
                # fmt: off
                self.keypoint_embedding_idxs = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,41,62,63,64,65,66,67,68,69]
                # fmt: on
                self.keypoint_embedding = nn.Embedding(
                    len(self.keypoint_embedding_idxs), self.cfg.MODEL.DECODER.DIM
                )

            if self.cfg.MODEL.DECODER.get("DO_KEYPOINT_TOKENS_2D_KPS_PRED", False):
                # These tokens also predict 2D KPS
                self.keypoint_head = FFN(
                    embed_dims=self.cfg.MODEL.DECODER.DIM,
                    feedforward_channels=self.cfg.MODEL.DECODER.DIM,
                    output_dims=2,
                    num_fcs=2,
                    add_identity=False,
                )

            if self.cfg.MODEL.DECODER.get("SEPARATE_AUX_TOKENS_2D", False):
                # Sorry, we have to be *very* careful here.
                # This means we're going to have a _separate_ set of auxiliary tokens.
                # If this is the case, DO_KEYPOINT_TOKENS_2D_KPS_PRED *must* be true,
                # and "DO_KEYPOINT_TOKENS_2D_KPS_PRED" is applied to these separate aux tokens
                # instead of the above atlas-tied-keypoint tokens.
                self.keypoint_embedding_aux = nn.Embedding(
                    len(self.keypoint_embedding_idxs), self.cfg.MODEL.DECODER.DIM
                )
                assert self.cfg.MODEL.DECODER.get(
                    "DO_KEYPOINT_TOKENS_2D_KPS_PRED", False
                )

        if self.cfg.MODEL.DECODER.get("KEYPOINT_TOKEN_UPDATE", None) in [
            "v1",
            "v2",
            "v3",
        ]:
            if not self.cfg.MODEL.DECODER.get(
                "KEYPOINT_TOKEN_UPDATE_COORD_EMB_USE_MLP", False
            ):
                # For some reason that escapes me, the backbone's pos embs are 1280, while the decoder is 1024.
                self.keypoint_posemb_linear = nn.Linear(
                    self.backbone.embed_dims, self.cfg.MODEL.DECODER.DIM
                )
            else:
                # TODO (jinhyun1): Shared for now
                self.keypoint_posemb_linear = FFN(
                    embed_dims=2,
                    feedforward_channels=self.cfg.MODEL.DECODER.DIM,
                    output_dims=self.cfg.MODEL.DECODER.DIM,
                    num_fcs=2,
                    add_identity=False,
                )
            if self.cfg.MODEL.DECODER.KEYPOINT_TOKEN_UPDATE == "v2":
                self.keypoint_feat_linear = nn.Linear(
                    self.backbone.embed_dims, self.cfg.MODEL.DECODER.DIM
                )
            elif self.cfg.MODEL.DECODER.KEYPOINT_TOKEN_UPDATE == "v3":
                from ..modules import MSDeformAttn

                # Different params for each layer
                # TODO (jinhyun1): Tie? Norm? This goes 1280 -> 1024 -> 1024.
                # Maybe it should be 1280 -> 256 -> 1024?
                # How many heads? Default is 8, because 256/8 = 32, so should it be more heads?
                self.keypoint_feat_sampling_layers = nn.ModuleList(
                    [
                        MSDeformAttn(
                            d_model=self.cfg.MODEL.DECODER.DIM,
                            n_levels=1,
                            n_heads=16,
                            n_points=4,
                            input_dim=self.backbone.embed_dims,
                        )
                        for _ in range(len(self.decoder.layers) - 1)
                    ]
                )
        else:
            assert self.cfg.MODEL.DECODER.get("KEYPOINT_TOKEN_UPDATE", None) is None

        if self.cfg.MODEL.DECODER.get("DO_KEYPOINT3D_TOKENS", False):
            # TODO: Tie with prompt encoder? Tie with 2D? There's nothing linking 2D and 3D together - maybe same embedding, then 2D vs 3D repated embedding?
            # TODO: Maybe these should also predict 3D heatmaps
            if not self.cfg.MODEL.DECODER.get("DO_KEYPOINT3D_TOKENS_BODY_ONLY", False):
                # Do all KPS
                self.keypoint3d_embedding_idxs = list(range(70))
                self.keypoint3d_embedding = nn.Embedding(
                    len(self.keypoint3d_embedding_idxs), self.cfg.MODEL.DECODER.DIM
                )
            else:
                # Just body (including wrist)
                # fmt: off
                self.keypoint3d_embedding_idxs = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,41,62,63,64,65,66,67,68,69]
                # fmt: on
                self.keypoint3d_embedding = nn.Embedding(
                    len(self.keypoint3d_embedding_idxs), self.cfg.MODEL.DECODER.DIM
                )

            if self.cfg.MODEL.DECODER.get("DO_KEYPOINT_TOKENS_3D_KPS_PRED", False):
                # These tokens also predict 2D KPS
                self.keypoint3d_head = FFN(
                    embed_dims=self.cfg.MODEL.DECODER.DIM,
                    feedforward_channels=self.cfg.MODEL.DECODER.DIM,
                    output_dims=3,
                    num_fcs=2,
                    add_identity=False,
                )

            if self.cfg.MODEL.DECODER.get("SEPARATE_AUX_TOKENS_3D", False):
                # Sorry, we have to be *very* careful here.
                # This means we're going to have a _separate_ set of auxiliary tokens.
                # If this is the case, DO_KEYPOINT_TOKENS_3D_KPS_PRED *must* be true,
                # and "DO_KEYPOINT_TOKENS_3D_KPS_PRED" is applied to these separate aux tokens
                # instead of the above atlas-tied-keypoint tokens.
                self.keypoint3d_embedding_aux = nn.Embedding(
                    len(self.keypoint3d_embedding_idxs), self.cfg.MODEL.DECODER.DIM
                )
                assert self.cfg.MODEL.DECODER.get(
                    "DO_KEYPOINT_TOKENS_3D_KPS_PRED", False
                )

        if self.cfg.MODEL.DECODER.get("KEYPOINT3D_TOKEN_UPDATE", None) in ["v1"]:
            # v1 is to use an MLP embedding, root normalized (TODO: or pelvis normalized?)
            # TODO (jinhyun1): Shared for now
            self.keypoint3d_posemb_linear = FFN(
                embed_dims=3,
                feedforward_channels=self.cfg.MODEL.DECODER.DIM,
                output_dims=self.cfg.MODEL.DECODER.DIM,
                num_fcs=2,
                add_identity=False,
            )

        if self.cfg.MODEL.DECODER.get("LEARNABLE_KPS2D_UNCERTAINTY", None) in [
            "v1",
            "v3",
        ]:
            # v1 is to use a learnable uncertainty for each keypoint.
            self.keypoint2d_loss_logsigma = nn.Embedding(1, 70)
            torch.nn.init.zeros_(self.keypoint2d_loss_logsigma.weight)

            if self.cfg.MODEL.DECODER.get("LEARNABLE_KPS2D_UNCERTAINTY", None) in [
                "v3"
            ]:
                # This takes the keypoint2d tokens. Note that this does not handle SMPL24 or potentially body tokens.
                assert self.cfg.MODEL.DECODER.get("DO_KEYPOINT_TOKENS", False)
                self.keypoint2d_loss_logsigma_layer = FFN(
                    embed_dims=self.cfg.MODEL.DECODER.DIM,
                    feedforward_channels=self.cfg.MODEL.DECODER.DIM,
                    output_dims=1,
                    num_fcs=2,
                    add_identity=False,
                )
        elif self.cfg.MODEL.DECODER.get("LEARNABLE_KPS2D_UNCERTAINTY", None) in ["v2"]:
            self.keypoint2d_loss_logsigma = FFN(
                embed_dims=self.cfg.MODEL.DECODER.DIM,
                feedforward_channels=self.cfg.MODEL.DECODER.DIM,
                output_dims=70,
                num_fcs=2,
                add_identity=False,
            )

        if self.cfg.MODEL.DECODER.get("LEARNABLE_KPS3D_UNCERTAINTY", None) in [
            "v1",
            "v3",
        ]:
            # v1 is to use a learnable uncertainty for each keypoint.
            self.keypoint3d_loss_logsigma = nn.Embedding(1, 70)
            torch.nn.init.zeros_(self.keypoint3d_loss_logsigma.weight)
            if self.cfg.LOSS_WEIGHTS.get("SMPL_24_KEYPOINTS_3D", 0) != 0:
                self.keypoint3d_smpl24_loss_logsigma = nn.Embedding(1, 24)
                torch.nn.init.zeros_(self.keypoint3d_smpl24_loss_logsigma.weight)

            if self.cfg.MODEL.DECODER.get("LEARNABLE_KPS3D_UNCERTAINTY", None) in [
                "v3"
            ]:
                # This takes the keypoint3d tokens. Note that this does not predict SMPL24 or potentially body tokens.
                # So, we'll handle those separately with global learnable
                assert self.cfg.MODEL.DECODER.get("DO_KEYPOINT3D_TOKENS", False)
                self.keypoint3d_loss_logsigma_layer = FFN(
                    embed_dims=self.cfg.MODEL.DECODER.DIM,
                    feedforward_channels=self.cfg.MODEL.DECODER.DIM,
                    output_dims=1,
                    num_fcs=2,
                    add_identity=False,
                )
        elif self.cfg.MODEL.DECODER.get("LEARNABLE_KPS3D_UNCERTAINTY", None) in ["v2"]:
            # This directly processes pose token.
            self.keypoint3d_loss_logsigma = FFN(
                embed_dims=self.cfg.MODEL.DECODER.DIM,
                feedforward_channels=self.cfg.MODEL.DECODER.DIM,
                output_dims=70
                + (
                    24
                    if (self.cfg.LOSS_WEIGHTS.get("SMPL_24_KEYPOINTS_3D", 0) != 0)
                    else 0
                ),
                num_fcs=2,
                add_identity=False,
            )

    def forward_full_branch(self, batch: Dict) -> Dict:
        """Run a forward pass for the full-image branch."""
        batch_size, num_person = batch["img"].shape[:2]

        self.full_encoder.eval()
        x = self.data_preprocess(batch["img_full"], is_full=True)
        with torch.no_grad(), torch.autocast(
            device_type="cuda",
            dtype=self.backbone_dtype,
            enabled=self.cfg.TRAIN.USE_FP16,
        ):
            full_embeddings = self.full_encoder(x.type(self.backbone_dtype))
        full_embeddings = full_embeddings.type(x.dtype)

        # Get person-tokens via ROI pooling
        batch_idx = (
            torch.arange(batch_size)
            .view(-1, 1, 1)
            .repeat(1, num_person, 1)
            .to(batch["full_bbox"])
        )
        rois = torch.cat([batch_idx, batch["full_bbox"]], dim=-1).view(-1, 5)
        if self.cfg.MODEL.FULL_ENCODER.IMAGE_SIZE == 518:
            full_map = full_embeddings.view(batch_size, 37, 37, -1).permute(0, 3, 1, 2)
        else:
            raise NotImplementedError
        person_embeddings = roi_align(
            input=full_map,
            boxes=rois,
            output_size=1,
            spatial_scale=1.0 / 14.0,
            sampling_ratio=-1,
            aligned=True,
        )
        person_embeddings = person_embeddings.view(batch_size, num_person, -1)

        # Concatenate person tokens with a full-token (for focal length) as full-decoder input
        full_token_input = torch.cat(
            [
                full_embeddings.mean(dim=1, keepdim=True),
                person_embeddings,
            ],
            dim=1,
        )
        if hasattr(self, "full_camera"):
            init_camera = self.full_init_camera.weight.unsqueeze(dim=1).expand(
                batch_size, num_person + 1, -1
            )
            full_token_input = torch.cat([full_token_input, init_camera], dim=-1)
        full_token_embeddings = self.init_to_token_full(full_token_input)

        full_mask = torch.cat(
            [torch.ones_like(batch["person_valid"][:, [0]]), batch["person_valid"]],
            dim=1,
        )
        full_tokens = self.full_decoder(
            full_token_embeddings,
            full_embeddings,
            token_mask=full_mask,
            channel_first=False,
        )  # (B, N+1, C)

        img_token = full_tokens[:, 0]
        person_tokens = full_tokens[:, 1:]

        full_output = None
        if hasattr(self, "full_camera"):
            pred_cam = self.full_camera(person_tokens, init_camera[:, 1:])
            full_output = {"pred_cam": pred_cam}

        return {
            "full_tokens": full_tokens,
            "full_mask": full_mask,
            "full_output": full_output,
        }

    def _get_decoder_condition(self, batch: Dict) -> Optional[torch.Tensor]:
        num_person = batch["img"].shape[1]

        if self.cfg.MODEL.DECODER.CONDITION_TYPE == "cliff":
            # CLIFF-style condition info (cx/f, cy/f, b/f)
            cx, cy = torch.chunk(
                self._flatten_person(batch["bbox_center"]), chunks=2, dim=-1
            )
            img_w, img_h = torch.chunk(
                self._flatten_person(batch["ori_img_size"]), chunks=2, dim=-1
            )
            b = self._flatten_person(batch["bbox_scale"])[:, [0]]

            focal_length = self._flatten_person(
                batch["cam_int"]
                .unsqueeze(1)
                .expand(-1, num_person, -1, -1)
                .contiguous()
            )[:, 0, 0]
            # condition_info = torch.cat([cx - img_w / 2.0, cy - img_h / 2.0, b], dim=-1
            if not self.cfg.MODEL.DECODER.get("USE_INTRIN_CENTER", False):
                condition_info = torch.cat(
                    [cx - img_w / 2.0, cy - img_h / 2.0, b], dim=-1
                )
            else:
                full_img_cxy = self._flatten_person(
                    batch["cam_int"]
                    .unsqueeze(1)
                    .expand(-1, num_person, -1, -1)
                    .contiguous()
                )[:, [0, 1], [2, 2]]
                condition_info = torch.cat(
                    [cx - full_img_cxy[:, [0]], cy - full_img_cxy[:, [1]], b], dim=-1
                )
            condition_info[:, :2] = condition_info[:, :2] / focal_length.unsqueeze(
                -1
            )  # [-1, 1]
            condition_info[:, 2] = condition_info[:, 2] / focal_length  # [-1, 1]
        elif self.cfg.MODEL.DECODER.CONDITION_TYPE == "none":
            return None
        else:
            raise NotImplementedError

        return condition_info.type(batch["img"].dtype)

    def forward_decoder(
        self,
        image_embeddings: torch.Tensor,
        init_estimate: Optional[torch.Tensor] = None,
        keypoints: Optional[torch.Tensor] = None,
        prev_estimate: Optional[torch.Tensor] = None,
        condition_info: Optional[torch.Tensor] = None,
        batch=None,
        full_output=None,
    ):
        """
        Args:
            image_embeddings: image features from the backbone, shape (B, C, H, W)
            init_estimate: initial estimate to be refined on, shape (B, 1, C)
            keypoints: optional prompt input, shape (B, N, 3),
                3 for coordinates (x,y) + label.
                (x, y) should be normalized to range [0, 1].
                label==-1 indicates incorrect points,
                label==-2 indicates invalid points
            prev_estimate: optional prompt input, shape (B, 1, C),
                previous estimate for pose refinement.
            condition_info: optional condition information that is concatenated with
                the input tokens, shape (B, c)
        """
        batch_size = image_embeddings.shape[0]

        # Initial estimation for residual prediction.
        if init_estimate is None:
            init_pose = self.init_pose.weight.expand(batch_size, -1).unsqueeze(dim=1)
            if hasattr(self, "init_camera"):
                init_camera = self.init_camera.weight.expand(batch_size, -1).unsqueeze(
                    dim=1
                )

            init_estimate = (
                init_pose
                if not hasattr(self, "init_camera")
                else torch.cat([init_pose, init_camera], dim=-1)
            )  # This is basically pose & camera translation at the end. B x 1 x (404 + 3)

        if condition_info is not None:
            init_input = torch.cat(
                [condition_info.view(batch_size, 1, -1), init_estimate], dim=-1
            )  # B x 1 x 410 (this is with the CLIFF condition)
        else:
            init_input = init_estimate
        token_embeddings = self.init_to_token_atlas(init_input).view(
            batch_size, 1, -1
        )  # B x 1 x 1024 (linear layered)
        num_pose_token = token_embeddings.shape[1]
        assert num_pose_token == 1  # TODO: extend to multiple pose token (multi-hmr)

        image_augment, token_augment, token_mask = None, None, None
        if hasattr(self, "prompt_encoder") and keypoints is not None:
            if prev_estimate is None:
                # Use initial embedding if no previous embedding
                prev_estimate = init_estimate
            # Previous estimate w/o the CLIFF condition.
            prev_embeddings = self.prev_to_token_atlas(prev_estimate).view(
                batch_size, 1, -1
            )  # 407 -> B x 1 x 1024; linear layer-ed

            if self.cfg.MODEL.BACKBONE.TYPE in [
                "vit_hmr",
                "hmr2",
                "vit",
                "vit_b",
                "vit_l",
            ]:
                # HMR2.0 backbone (ViTPose) assumes a different aspect ratio as input size
                image_augment = self.prompt_encoder.get_dense_pe((16, 16))
                image_augment = image_augment[:, :, :, 2:-2]
            elif self.cfg.MODEL.BACKBONE.TYPE in ["vit_hmr_512_384"]:
                # HMR2.0 backbone (ViTPose) assumes a different aspect ratio as input size
                image_augment = self.prompt_encoder.get_dense_pe((32, 32))
                image_augment = image_augment[:, :, :, 4:-4]
            else:
                image_augment = self.prompt_encoder.get_dense_pe(
                    image_embeddings.shape[-2:]
                )  # (1, C, H, W)

            if self.cfg.MODEL.get("RAY_CONDITION_TYPE", None) in [
                "decoder_v1",
                "decoder_v2",
            ]:
                assert "ray_cond" in batch
                image_augment = image_augment + batch["ray_cond"].to(image_augment)

            # To start, keypoints is all [0, 0, -2]. The points get sent into self.pe_layer._pe_encoding,
            # the labels determine the embedding weight (special one for -2, -1, then each of joint.)
            prompt_embeddings, prompt_mask = self.prompt_encoder(
                keypoints=keypoints
            )  # B x 1 x 1280
            prompt_embeddings = self.prompt_to_token(
                prompt_embeddings
            )  # Linear layered: B x 1 x 1024

            # Concatenate pose tokens and prompt embeddings as decoder input
            token_embeddings = torch.cat(
                [
                    token_embeddings,
                    prev_embeddings,
                    prompt_embeddings,
                ],
                dim=1,
            )

            token_augment = torch.zeros_like(token_embeddings)
            token_augment[:, [num_pose_token]] = (
                prev_embeddings  # No embedding for current? ig it has its self.init_to_token_atlas?
            )
            token_augment[:, (num_pose_token + 1) :] = prompt_embeddings
            # token_mask = torch.ones(batch_size, token_embeddings.shape[1]).to(prompt_mask)
            # token_mask[:, num_pose_token:] = prompt_mask
            # token_mask = token_mask.to(token_embeddings.dtype)  # bool to float
            # token_mask.requires_grad = False
            token_mask = None

            if self.cfg.MODEL.DECODER.get("DO_KEYPOINT_TOKENS", False):
                # Put in a token for each keypoint
                kps_emb_start_idx = token_embeddings.shape[1]
                token_embeddings = torch.cat(
                    [
                        token_embeddings,
                        self.keypoint_embedding.weight[None, :, :].repeat(
                            batch_size, 1, 1
                        ),
                    ],
                    dim=1,
                )  # B x 3 + 70 x 1024
                # No positional embeddings
                token_augment = torch.cat(
                    [
                        token_augment,
                        torch.zeros_like(
                            token_embeddings[:, token_augment.shape[1] :, :]
                        ),
                    ],
                    dim=1,
                )  # B x 3 + 70 x 1024

            if self.cfg.MODEL.DECODER.get("DO_KEYPOINT3D_TOKENS", False):
                # Put in a token for each keypoint
                kps3d_emb_start_idx = token_embeddings.shape[1]
                token_embeddings = torch.cat(
                    [
                        token_embeddings,
                        self.keypoint3d_embedding.weight[None, :, :].repeat(
                            batch_size, 1, 1
                        ),
                    ],
                    dim=1,
                )  # B x 3 + 70 + 70 x 1024
                # No positional embeddings
                token_augment = torch.cat(
                    [
                        token_augment,
                        torch.zeros_like(
                            token_embeddings[:, token_augment.shape[1] :, :]
                        ),
                    ],
                    dim=1,
                )  # B x 3 + 70 + 70 x 1024

            if self.cfg.MODEL.DECODER.get("SEPARATE_AUX_TOKENS_2D", False):
                # Put in a token for each keypoint
                kps_emb_aux_start_idx = token_embeddings.shape[1]
                token_embeddings = torch.cat(
                    [
                        token_embeddings,
                        self.keypoint_embedding_aux.weight[None, :, :].repeat(
                            batch_size, 1, 1
                        ),
                    ],
                    dim=1,
                )
                # No positional embeddings
                token_augment = torch.cat(
                    [
                        token_augment,
                        torch.zeros_like(
                            token_embeddings[:, token_augment.shape[1] :, :]
                        ),
                    ],
                    dim=1,
                )

            if self.cfg.MODEL.DECODER.get("SEPARATE_AUX_TOKENS_3D", False):
                # Put in a token for each keypoint
                kps3d_emb_aux_start_idx = token_embeddings.shape[1]
                token_embeddings = torch.cat(
                    [
                        token_embeddings,
                        self.keypoint3d_embedding_aux.weight[None, :, :].repeat(
                            batch_size, 1, 1
                        ),
                    ],
                    dim=1,
                )
                # No positional embeddings
                token_augment = torch.cat(
                    [
                        token_augment,
                        torch.zeros_like(
                            token_embeddings[:, token_augment.shape[1] :, :]
                        ),
                    ],
                    dim=1,
                )

        if not self.cfg.MODEL.DECODER.get("DO_INTERM_PREDS", False):
            tokens = self.decoder(
                token_embeddings,
                image_embeddings,
                token_augment,
                image_augment,
                token_mask,
            )
            pose_token = tokens[:, 0]

            # Get pose outputs (atlas parameters)
            pose_output = self.head_pose(pose_token, init_pose.view(batch_size, -1))
            if hasattr(self, "head_camera"):
                pred_cam = self.head_camera(
                    pose_token, init_camera.view(batch_size, -1)
                )
                pose_output["pred_cam"] = pred_cam

            pose_output = self.camera_project(pose_output, batch, full_output)
        else:
            # We're doing intermediate model predictions
            # Yes, there's an easier way to do this, which is just to get the output features
            # and decode them here, but we may want to use predictions inside the decoder, so..

            def token_to_pose_output_fn(tokens, prev_pose_output, layer_idx):
                # Get the pose token
                pose_token = tokens[:, 0]

                # Pose & camera are done residually. Optionally, we may choose to make their predictions
                # residual through the decoder layers.
                if (
                    self.cfg.MODEL.DECODER.get("DO_INTERM_RESIDUAL_PRED", False)
                    and prev_pose_output is not None
                ):
                    # TODO (jinhyun1): Detach or not detach? Also, probably shouldn't .clone(), but I'm terrified of in-place
                    prev_pose = torch.cat(
                        [
                            prev_pose_output["pred_pose_raw"],
                            prev_pose_output["shape"],
                            prev_pose_output["scale"],
                            prev_pose_output["hand"],
                            prev_pose_output["face"],
                        ],
                        dim=1,
                    ).clone()
                    prev_camera = prev_pose_output["pred_cam"].clone()
                else:
                    prev_pose = init_pose.view(batch_size, -1)
                    prev_camera = init_camera.view(batch_size, -1)

                # Get pose outputs (atlas parameters)
                if not self.cfg.MODEL.DECODER.get(
                    "DO_INTERM_SLIM_KEYPOINTS", False
                ) or (layer_idx == len(self.decoder.layers) - 1):
                    # If we're not doing slim keypoints (original default) or it's the last decoder layer...
                    pose_output = self.head_pose(pose_token, prev_pose)
                else:
                    # Otherwise, we don't need vertices, so...
                    pose_output = self.head_pose(
                        pose_token, prev_pose, slim_keypoints=True
                    )
                # Get Camera Translation
                if hasattr(self, "head_camera"):
                    pred_cam = self.head_camera(pose_token, prev_camera)
                    pose_output["pred_cam"] = pred_cam
                # Run camera projection
                pose_output = self.camera_project(pose_output, batch, full_output)

                # Get 2D KPS in crop (we don't need it usually, but it's cheap to compute anyway so why not)
                pose_output["pred_keypoints_2d_cropped"] = self._full_to_crop(
                    batch, pose_output["pred_keypoints_2d"]
                )

                # Optionally, do auxiliary 2D keypoint predictions
                if self.cfg.MODEL.DECODER.get("DO_KEYPOINT_TOKENS_2D_KPS_PRED", False):
                    num_keypoints = self.keypoint_embedding.weight.shape[0]
                    if not self.cfg.MODEL.DECODER.get("SEPARATE_AUX_TOKENS_2D", False):
                        # Then, the atlas-tied 2d kps
                        pose_output["pred_keypoints_2d_cropped_aux"] = (
                            self.keypoint_head(
                                tokens[
                                    :,
                                    kps_emb_start_idx : kps_emb_start_idx
                                    + num_keypoints,
                                    :,
                                ]
                            )
                        )
                    else:
                        # If separate, it's the auxiliary 2d kps
                        pose_output["pred_keypoints_2d_cropped_aux"] = (
                            self.keypoint_head(
                                tokens[
                                    :,
                                    kps_emb_aux_start_idx : kps_emb_aux_start_idx
                                    + num_keypoints,
                                    :,
                                ]
                            )
                        )

                # Optionally, do auxiliary 3D keypoint predictions
                if self.cfg.MODEL.DECODER.get("DO_KEYPOINT_TOKENS_3D_KPS_PRED", False):
                    num_keypoints3d = self.keypoint3d_embedding.weight.shape[0]
                    if not self.cfg.MODEL.DECODER.get("SEPARATE_AUX_TOKENS_3D", False):
                        pose_output["pred_keypoints_3d_aux"] = self.keypoint3d_head(
                            tokens[
                                :,
                                kps3d_emb_start_idx : kps3d_emb_start_idx
                                + num_keypoints3d,
                                :,
                            ]
                        )
                    else:
                        # If separate, it's the auxiliary 3d kps
                        pose_output["pred_keypoints_3d_aux"] = self.keypoint3d_head(
                            tokens[
                                :,
                                kps3d_emb_aux_start_idx : kps3d_emb_aux_start_idx
                                + num_keypoints3d,
                                :,
                            ]
                        )

                # Optionally, do uncertainty prediction for 2D
                if self.cfg.MODEL.DECODER.get("LEARNABLE_KPS2D_UNCERTAINTY", None) in [
                    "v2"
                ]:
                    pose_output["pred_keypoints_2d_logsigma"] = (
                        self.keypoint2d_loss_logsigma(pose_token)
                    )
                elif self.cfg.MODEL.DECODER.get(
                    "LEARNABLE_KPS2D_UNCERTAINTY", None
                ) in ["v3"]:
                    num_keypoints = self.keypoint_embedding.weight.shape[0]
                    pose_output["pred_keypoints_2d_logsigma"] = (
                        self.keypoint2d_loss_logsigma.weight.repeat(
                            len(pose_token), 1
                        ).clone()
                    )
                    pose_output["pred_keypoints_2d_logsigma"][
                        :, self.keypoint_embedding_idxs
                    ] = self.keypoint2d_loss_logsigma_layer(
                        tokens[
                            :, kps_emb_start_idx : kps_emb_start_idx + num_keypoints, :
                        ]
                    ).squeeze(
                        2
                    )

                # Optionally, do uncertainty prediction for 3D
                if self.cfg.MODEL.DECODER.get("LEARNABLE_KPS3D_UNCERTAINTY", None) in [
                    "v2"
                ]:
                    pose_output["pred_keypoints_3d_logsigma"] = (
                        self.keypoint3d_loss_logsigma(pose_token)
                    )

                    if self.cfg.LOSS_WEIGHTS.get("SMPL_24_KEYPOINTS_3D", 0) != 0:
                        (
                            pose_output["pred_keypoints_3d_logsigma"],
                            pose_output["pred_keypoints_3d_smpl24_logsigma"],
                        ) = torch.split(
                            pose_output["pred_keypoints_3d_logsigma"], [70, 24], dim=-1
                        )
                elif self.cfg.MODEL.DECODER.get(
                    "LEARNABLE_KPS3D_UNCERTAINTY", None
                ) in ["v3"]:
                    num_keypoints3d = self.keypoint3d_embedding.weight.shape[0]
                    pose_output["pred_keypoints_3d_logsigma"] = (
                        self.keypoint3d_loss_logsigma.weight.repeat(
                            len(pose_token), 1
                        ).clone()
                    )
                    pose_output["pred_keypoints_3d_logsigma"][
                        :, self.keypoint3d_embedding_idxs
                    ] = self.keypoint3d_loss_logsigma_layer(
                        tokens[
                            :,
                            kps3d_emb_start_idx : kps3d_emb_start_idx + num_keypoints3d,
                            :,
                        ]
                    ).squeeze(
                        2
                    )
                    if self.cfg.LOSS_WEIGHTS.get("SMPL_24_KEYPOINTS_3D", 0) != 0:
                        pose_output["pred_keypoints_3d_smpl24_logsigma"] = (
                            self.keypoint3d_smpl24_loss_logsigma.weight.repeat(
                                len(pose_token), 1
                            ).clone()
                        )

                return pose_output

            keypoint_token_update_fn = None
            if self.cfg.MODEL.DECODER.get("KEYPOINT_TOKEN_UPDATE", None) in [
                "v1",
                "v2",
                "v3",
            ]:
                # For this one, we're going to take the projected 2D KPS, get posembs
                # (prompt encoder style, and set that as the keypoint posemb)
                def keypoint_token_update_fn(
                    token_embeddings, token_augment, pose_output, layer_idx
                ):
                    # It's already after the last layer, we're done.
                    if layer_idx == len(self.decoder.layers) - 1:
                        return token_embeddings, token_augment, pose_output, layer_idx

                    # Clone
                    token_embeddings = token_embeddings.clone()
                    token_augment = token_augment.clone()

                    num_keypoints = self.keypoint_embedding.weight.shape[0]

                    # Get current 2D KPS predictions # TODO (jinhyun1): Get 3D kps too, put into the model. 2D helps 2D, doesn't help 3D
                    pred_keypoints_2d_cropped = pose_output[
                        "pred_keypoints_2d_cropped"
                    ].clone()  # These are -0.5 ~ 0.5 (CHECK!!!!!!!)
                    pred_keypoints_2d_depth = pose_output[
                        "pred_keypoints_2d_depth"
                    ].clone()

                    # Optionally detach them (backpropping through random sincos.. hmm)
                    if self.cfg.MODEL.DECODER.get(
                        "KEYPOINT_TOKEN_UPDATE_DETACH_KPS", True
                    ):
                        pred_keypoints_2d_cropped = pred_keypoints_2d_cropped.detach()
                        pred_keypoints_2d_depth = pred_keypoints_2d_depth.detach()

                    # This is a hack, as during inference, smpl joints are used, so the 70 atlas kps gets tossed.
                    # As such, I've stacked smpl joints & 70 atlas kps joints in atlas head, so just for this part, we can keep the 70.
                    if self.head_pose.atlas.lod == "smpl":
                        assert (
                            not self.training
                        ), "Why are we doing training with SMPL topology?"
                        assert pred_keypoints_2d_cropped.shape[1] == (
                            61 + 70
                        ), pred_keypoints_2d_cropped.shape[1]
                        assert pred_keypoints_2d_depth.shape[1] == (
                            61 + 70
                        ), pred_keypoints_2d_depth.shape[1]
                        pred_keypoints_2d_cropped = pred_keypoints_2d_cropped[:, -70:]
                        pred_keypoints_2d_depth = pred_keypoints_2d_depth[:, -70:]

                    # Get the keypoints we're using here
                    # if not self.head_pose.atlas.lod in ["smplx"]:
                    assert (
                        self.head_pose.atlas.lod != "smplx"
                    ), "SMPLX is not supported for keypoint token update."

                    # have this for atlas or smpl
                    pred_keypoints_2d_cropped = pred_keypoints_2d_cropped[
                        :, self.keypoint_embedding_idxs
                    ]
                    pred_keypoints_2d_depth = pred_keypoints_2d_depth[
                        :, self.keypoint_embedding_idxs
                    ]

                    # Get 2D KPS to be 0 ~ 1
                    pred_keypoints_2d_cropped_01 = pred_keypoints_2d_cropped + 0.5

                    # Get a mask of those that are 1) beyond image boundaries or 2) behind the camera
                    invalid_mask = (
                        (pred_keypoints_2d_cropped_01[:, :, 0] < 0)
                        | (pred_keypoints_2d_cropped_01[:, :, 0] > 1)
                        | (pred_keypoints_2d_cropped_01[:, :, 1] < 0)
                        | (pred_keypoints_2d_cropped_01[:, :, 1] > 1)
                        | (pred_keypoints_2d_depth[:, :] < 1e-5)
                    )

                    # Run them through the prompt encoder's pos emb function
                    if not self.cfg.MODEL.DECODER.get(
                        "KEYPOINT_TOKEN_UPDATE_COORD_EMB_USE_MLP", False
                    ):
                        pred_keypoints_2d_cropped_emb = (
                            self.prompt_encoder.pe_layer._pe_encoding(
                                pred_keypoints_2d_cropped_01
                            )
                        )
                        # Zero invalid pos embs out
                        pred_keypoints_2d_cropped_emb = (
                            pred_keypoints_2d_cropped_emb * (~invalid_mask[:, :, None])
                        )
                        # Put them in
                        token_augment[
                            :, kps_emb_start_idx : kps_emb_start_idx + num_keypoints, :
                        ] = self.keypoint_posemb_linear(pred_keypoints_2d_cropped_emb)
                    else:
                        # TODO (jinhyun1): NOTE: Here, note that the OUTPUT is multiplied by 0. upstairs, the INPUT is.
                        token_augment[
                            :, kps_emb_start_idx : kps_emb_start_idx + num_keypoints, :
                        ] = self.keypoint_posemb_linear(pred_keypoints_2d_cropped) * (
                            ~invalid_mask[:, :, None]
                        )

                    # Also maybe update token_embeddings with the grid sampled 2D feature.
                    # Remember that pred_keypoints_2d_cropped are -0.5 ~ 0.5. We want -1 ~ 1
                    if self.cfg.MODEL.DECODER.KEYPOINT_TOKEN_UPDATE in ["v2", "v3"]:
                        # Sample points...
                        ## Get sampling points
                        pred_keypoints_2d_cropped_sample_points = (
                            pred_keypoints_2d_cropped * 2
                        )
                        if self.cfg.MODEL.BACKBONE.TYPE in [
                            "vit_hmr",
                            "hmr2",
                            "vit",
                            "vit_b",
                            "vit_l",
                            "vit_hmr_512_384",
                        ]:
                            # Need to go from 256 x 256 coords to 256 x 192 (HW) because image_embeddings is 16x12
                            # Aka, for x, what was normally -1 ~ 1 for 256 should be -16/12 ~ 16/12 (since to sample at original 256, need to overflow)
                            pred_keypoints_2d_cropped_sample_points[:, :, 0] = (
                                pred_keypoints_2d_cropped_sample_points[:, :, 0]
                                / 12
                                * 16
                            )

                        if self.cfg.MODEL.DECODER.KEYPOINT_TOKEN_UPDATE in ["v2"]:
                            # Version 2 is projecting & bilinear sampling
                            if self.head_pose.atlas.lod == "smplx":
                                # [Jinkun] Temporal skip for eval
                                assert False
                                pass
                            else:
                                # Version 2 is projecting & bilinear sampling
                                pred_keypoints_2d_cropped_feats = (
                                    F.grid_sample(
                                        image_embeddings,
                                        pred_keypoints_2d_cropped_sample_points[
                                            :, :, None, :
                                        ],  # -1 ~ 1, xy
                                        mode="bilinear",
                                        padding_mode="zeros",
                                        align_corners=False,
                                    )
                                    .squeeze(3)
                                    .permute(0, 2, 1)
                                )  # B x kps x C
                                # Zero out invalid locations...
                                pred_keypoints_2d_cropped_feats = (
                                    pred_keypoints_2d_cropped_feats
                                    * (~invalid_mask[:, :, None])
                                )
                                # This is ADDING
                                token_embeddings = token_embeddings.clone()
                                token_embeddings[
                                    :,
                                    kps_emb_start_idx : kps_emb_start_idx
                                    + num_keypoints,
                                    :,
                                ] += self.keypoint_feat_linear(
                                    pred_keypoints_2d_cropped_feats
                                )
                        elif self.cfg.MODEL.DECODER.KEYPOINT_TOKEN_UPDATE in ["v3"]:
                            # Version 3 is Deformable attention

                            # Reference: https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/modules/ms_deform_attn.py#L78
                            # self.keypoint_feat_sampling_layers
                            # TODO: Zero mult invalid before or after output proj?
                            # TODO: Add posembs before querying? Normalize after? It doesn't actually _exactly_ know where it is rn
                            # TODO: Maybe localize the sampling offsets more (normalize by n_points). Right now, it's a 4 "pixel" radius

                            # Note that this is post (16, 12) scaling.
                            pred_keypoints_2d_cropped_sample_points_01 = (
                                pred_keypoints_2d_cropped_sample_points + 1
                            ) / 2
                            pred_keypoints_2d_cropped_feats = self.keypoint_feat_sampling_layers[
                                layer_idx
                            ](
                                query=token_embeddings[
                                    :,
                                    kps_emb_start_idx : kps_emb_start_idx
                                    + num_keypoints,
                                    :,
                                ].contiguous(),
                                reference_points=pred_keypoints_2d_cropped_sample_points_01[
                                    :, :, None, :
                                ].contiguous(),
                                input_flatten=image_embeddings.flatten(2, 3)
                                .permute(0, 2, 1)
                                .contiguous(),
                                input_spatial_shapes=torch.LongTensor(
                                    [image_embeddings.shape[-2:]]
                                ).cuda(),
                                input_level_start_index=torch.LongTensor([0]).cuda(),
                            )
                            # Zero out invalid locations...
                            pred_keypoints_2d_cropped_feats = (
                                pred_keypoints_2d_cropped_feats
                                * (~invalid_mask[:, :, None])
                            )
                            # This is ADDING
                            token_embeddings = token_embeddings.clone()
                            token_embeddings[
                                :,
                                kps_emb_start_idx : kps_emb_start_idx + num_keypoints,
                                :,
                            ] += pred_keypoints_2d_cropped_feats

                    return token_embeddings, token_augment, pose_output, layer_idx

            else:
                assert self.cfg.MODEL.DECODER.get("KEYPOINT_TOKEN_UPDATE", None) is None

            # Now for 3D
            keypoint3d_token_update_fn = None
            if self.cfg.MODEL.DECODER.get("KEYPOINT3D_TOKEN_UPDATE", None) in ["v1"]:

                def keypoint3d_token_update_fn(
                    token_embeddings, token_augment, pose_output, layer_idx
                ):
                    # It's already after the last layer, we're done.
                    if layer_idx == len(self.decoder.layers) - 1:
                        return token_embeddings, token_augment, pose_output, layer_idx

                    num_keypoints3d = self.keypoint3d_embedding.weight.shape[0]

                    # Get current 3D kps predictions
                    pred_keypoints_3d = pose_output["pred_keypoints_3d"].clone()

                    # This is a hack, as during inference, smpl joints are used, so the 70 atlas kps gets tossed.
                    # As such, I've stacked smpl joints & 70 atlas kps joints in atlas head, so just for this part, we can keep the 70.
                    if self.head_pose.atlas.lod == "smpl":
                        assert (
                            not self.training
                        ), "Why are we doing training with SMPL topology?"
                        assert pred_keypoints_3d.shape[1] == (
                            61 + 70
                        ), pred_keypoints_3d.shape[1]
                        pred_keypoints_3d = pred_keypoints_3d[:, -70:]

                    assert (
                        self.head_pose.atlas.lod != "smplx"
                    ), "SMPLX is not supported for keypoint token update."
                    # Now, pelvis normalize TODO: (jinhyun1) detach pelvis?
                    pred_keypoints_3d = (
                        pred_keypoints_3d
                        - (
                            pred_keypoints_3d[:, [self.pelvis_idx[0]], :]
                            + pred_keypoints_3d[:, [self.pelvis_idx[1]], :]
                        )
                        / 2
                    )

                    # Get the kps we care about, _after_ pelvis norm (just in case idxs shift)
                    pred_keypoints_3d = pred_keypoints_3d[
                        :, self.keypoint3d_embedding_idxs
                    ]

                    # Run through embedding MLP & put in
                    token_augment = token_augment.clone()
                    token_augment[
                        :,
                        kps3d_emb_start_idx : kps3d_emb_start_idx + num_keypoints3d,
                        :,
                    ] = self.keypoint3d_posemb_linear(pred_keypoints_3d)

                    # TODO: (jinhyun1) these 3D KPS tokens should have auxiliary 3D kps pred, just like 2D? Like where they want to belong
                    return token_embeddings, token_augment, pose_output, layer_idx

            else:
                assert (
                    self.cfg.MODEL.DECODER.get("KEYPOINT3D_TOKEN_UPDATE", None) is None
                )

            # Combine the 2D and 3D functions
            def keypoint_token_update_fn_comb(*args):
                if keypoint_token_update_fn is not None:
                    args = keypoint_token_update_fn(*args)
                if keypoint3d_token_update_fn is not None:
                    args = keypoint3d_token_update_fn(*args)
                return args

            pose_token, pose_output = self.decoder(
                token_embeddings,
                image_embeddings,
                token_augment,
                image_augment,
                token_mask,
                token_to_pose_output_fn=token_to_pose_output_fn,
                keypoint_token_update_fn=keypoint_token_update_fn_comb,
            )

        return pose_token, pose_output


    def get_atlas_output(self, batch, return_keypoints):
        gt_atlas_output = self.head_pose.atlas(
            global_trans=torch.zeros_like(
                batch["atlas_params"]["global_orient"].squeeze(1)
            ),  # global_trans==0
            global_rot=batch["atlas_params"]["global_orient"].squeeze(1),
            body_pose_params=batch["atlas_params"]["body_pose"].squeeze(1),
            hand_pose_params=batch["atlas_params"]["hand"].squeeze(1),  # 24 x 64
            scale_params=batch["atlas_params"]["scale"].squeeze(1),
            shape_params=batch["atlas_params"]["shape"].squeeze(1),
            expr_params=batch["atlas_params"]["face"].squeeze(1),
            do_pcblend=True,
            return_keypoints=return_keypoints,
        )
        if return_keypoints:
            gt_verts, gt_j3d = gt_atlas_output
            gt_verts[..., [1, 2]] *= -1  # Camera system difference
            gt_j3d[..., [1, 2]] *= -1  # Camera system difference
            return gt_verts, gt_j3d
        else:
            gt_verts = gt_atlas_output
            gt_verts[..., [1, 2]] *= -1  # Camera system difference
            return gt_verts

    @torch.no_grad()
    def _get_keypoint_prompt(self, batch, output, force_dummy=False):
        pred_keypoints_2d = output["atlas"]["pred_keypoints_2d"].detach().clone()
        if self.camera_type == "perspective":
            pred_keypoints_2d = self._full_to_crop(batch, pred_keypoints_2d)

        gt_keypoints_2d = self._flatten_person(batch["keypoints_2d"]).clone()

        keypoint_prompt = self.keypoint_prompt_sampler.sample(
            gt_keypoints_2d,
            pred_keypoints_2d,
            is_train=self.training,
            force_dummy=force_dummy,
        )
        return keypoint_prompt

    def _get_mask_prompt(self, batch, image_embeddings):
        # image_embeddings = output["image_embeddings"]
        x_mask = self._flatten_person(batch["mask"])
        mask_embeddings, no_mask_embeddings = self.prompt_encoder.get_mask_embeddings(
            x_mask, image_embeddings.shape[0], image_embeddings.shape[2:]
        )
        if self.cfg.MODEL.BACKBONE.TYPE in ["vit_hmr", "hmr2", "vit"]:
            # HMR2.0 backbone (ViTPose) assumes a different aspect ratio as input size
            mask_embeddings = mask_embeddings[:, :, :, 2:-2]
        elif self.cfg.MODEL.BACKBONE.TYPE in ["vit_hmr_512_384"]:
            # for x2 resolution
            mask_embeddings = mask_embeddings[:, :, :, 4:-4]

        mask_score = self._flatten_person(batch["mask_score"]).view(-1, 1, 1, 1)
        mask_embeddings = torch.where(
            mask_score > 0,
            mask_score * mask_embeddings.to(image_embeddings),
            no_mask_embeddings.to(image_embeddings),
        )
        return mask_embeddings

    def _one_prompt_iter(self, batch, output, prev_prompt, full_output):
        image_embeddings = output["image_embeddings"]
        condition_info = output["condition_info"]
        pose_output = output["atlas"]

        # Use previous estimate as initialization
        prev_estimate = torch.cat(
            [
                pose_output["pred_pose_raw"].detach(),  # (B, 6)
                pose_output["shape"].detach(),
                pose_output["scale"].detach(),
                pose_output["hand"].detach(),
                pose_output["face"].detach(),
            ],
            dim=1,
        ).unsqueeze(dim=1)
        if hasattr(self, "init_camera") or self.cfg.MODEL.PERSON_HEAD.POSE_TYPE in [
            "atlas",
            "atlas46",
        ]:
            prev_estimate = torch.cat(
                [prev_estimate, pose_output["pred_cam"].detach().unsqueeze(1)], dim=-1
            )

        # Get keypoint prompts
        keypoint_prompt = self._get_keypoint_prompt(batch, output)
        if len(prev_prompt):
            cur_keypoint_prompt = torch.cat(prev_prompt + [keypoint_prompt], dim=1)
        else:
            cur_keypoint_prompt = keypoint_prompt  # [B, 1, 3]

        _, pose_output = self.forward_decoder(
            image_embeddings,
            init_estimate=None,  # not recurring previous estimate
            keypoints=cur_keypoint_prompt,
            prev_estimate=prev_estimate,
            condition_info=condition_info,
            batch=batch,
            full_output=full_output,
        )

        if self.cfg.MODEL.DECODER.get("DO_INTERM_PREDS", False):
            assert isinstance(
                pose_output, list
            ), "You're doing DO_INTERM_PREDS but pose_output is not a list?"  # Sanity
            pose_output_interm, pose_output = pose_output[:-1], pose_output[-1]
            output["atlas_interm"] = pose_output_interm

        # Update prediction output
        output["atlas"] = pose_output

        return output, keypoint_prompt

    def camera_project(self, pose_output: Dict, batch: Dict, full_output: Dict) -> Dict:
        """
        Project 3D keypoints to 2D using the camera parameters.
        Args:
            pose_output (Dict): Dictionary containing the pose output.
            batch (Dict): Dictionary containing the batch data.
        Returns:
            Dict: Dictionary containing the projected 2D keypoints.
        """
        if hasattr(self, "head_camera"):
            head_camera = self.head_camera
            pred_cam = pose_output["pred_cam"]
        else:
            head_camera = self.full_camera
            pred_cam = self._flatten_person(full_output["full_output"]["pred_cam"])

        cam_out = head_camera.perspective_projection(
            pose_output["pred_keypoints_3d"],
            pred_cam,
            self._flatten_person(batch["bbox_center"]),
            self._flatten_person(batch["bbox_scale"])[:, 0],
            self._flatten_person(batch["ori_img_size"]),
            self._flatten_person(
                batch["cam_int"]
                .unsqueeze(1)
                .expand(-1, batch["img"].shape[1], -1, -1)
                .contiguous()
            ),
            use_intrin_center=self.cfg.MODEL.DECODER.get("USE_INTRIN_CENTER", False),
        )

        if pose_output.get("pred_vertices", None) is not None:
            cam_out_vertices = head_camera.perspective_projection(
                pose_output["pred_vertices"],
                pred_cam,
                self._flatten_person(batch["bbox_center"]),
                self._flatten_person(batch["bbox_scale"])[:, 0],
                self._flatten_person(batch["ori_img_size"]),
                self._flatten_person(
                    batch["cam_int"]
                    .unsqueeze(1)
                    .expand(-1, batch["img"].shape[1], -1, -1)
                    .contiguous()
                ),
                use_intrin_center=self.cfg.MODEL.DECODER.get(
                    "USE_INTRIN_CENTER", False
                ),
            )
            pose_output["pred_keypoints_2d_verts"] = cam_out_vertices[
                "pred_keypoints_2d"
            ]

        pose_output.update(cam_out)

        return pose_output

    def get_ray_condition(self, batch):
        B, N, _, H, W = batch["img"].shape
        meshgrid_xy = (
            torch.stack(
                torch.meshgrid(torch.arange(H), torch.arange(W), indexing="xy"), dim=2
            )[None, None, :, :, :]
            .repeat(B, N, 1, 1, 1)
            .cuda()
        )  # B x N x H x W x 2
        meshgrid_xy = (
            meshgrid_xy / batch["affine_trans"][:, :, None, None, [0, 1], [0, 1]]
        )
        meshgrid_xy = (
            meshgrid_xy
            - batch["affine_trans"][:, :, None, None, [0, 1], [2, 2]]
            / batch["affine_trans"][:, :, None, None, [0, 1], [0, 1]]
        )

        # Subtract out center & normalize to be rays
        if self.cfg.MODEL.RAY_CONDITION_TYPE in ["decoder_v2"]:
            meshgrid_xy = (
                meshgrid_xy - batch["cam_int"][:, None, None, None, [0, 1], [2, 2]]
            )
        else:
            meshgrid_xy = (
                meshgrid_xy - batch["cam_int"][:, None, None, None, [0, 0], [2, 2]]
            )
        meshgrid_xy = (
            meshgrid_xy / batch["cam_int"][:, None, None, None, [0, 1], [0, 1]]
        )

        return meshgrid_xy.permute(0, 1, 4, 2, 3).to(
            batch["img"].dtype
        )  # This is B x num_person x 2 x H x W

    def forward_pose_branch(
        self,
        batch: Dict,
        full_output: Dict,
    ) -> Dict:
        """Run a forward pass for the crop-image (pose) branch."""
        batch_size, num_person = batch["img"].shape[:2]

        # Forward backbone encoder
        x = self.data_preprocess(
            self._flatten_person(batch["img"]),
            crop_width=(
                self.cfg.MODEL.BACKBONE.TYPE
                in ["vit_hmr", "hmr2", "vit", "vit_b", "vit_l", "vit_hmr_512_384"]
            ),
        )

        # Optionally get ray conditioining
        if self.cfg.MODEL.get("RAY_CONDITION_TYPE", None) is not None:
            ray_cond = self.get_ray_condition(
                batch
            )  # This is B x num_person x 2 x H x W
            ray_cond = self._flatten_person(ray_cond)
            if self.cfg.MODEL.BACKBONE.TYPE in [
                "vit_hmr",
                "hmr2",
                "vit",
                "vit_b",
                "vit_l",
            ]:
                ray_cond = ray_cond[:, :, :, 32:-32]
            elif self.cfg.MODEL.BACKBONE.TYPE in ["vit_hmr_512_384"]:
                ray_cond = ray_cond[:, :, :, 64:-64]

            if self.cfg.MODEL.RAY_CONDITION_TYPE == "backbone_v1":
                # Basically, have a zero conv patch embedding, then put into the backbone.
                ray_cond = self.ray_cond_emb(ray_cond)
            elif self.cfg.MODEL.RAY_CONDITION_TYPE in ["decoder_v1", "decoder_v2"]:
                batch["ray_cond"] = self.ray_cond_emb(ray_cond)
                ray_cond = None
            else:
                raise Exception
        else:
            ray_cond = None

        image_embeddings = self.backbone(
            x.type(self.backbone_dtype), extra_embed=ray_cond
        )  # (B, C, H, W)

        if self.cfg.MODEL.BACKBONE.TYPE in [
            "pe_core_b",
            "pe_core_l",
            "pe_spatial_l",
            "pe_spatial_b",
            "pe_spatial_s",
            "pe_spatial_t",
        ]:
            image_embeddings = image_embeddings[:, 1:, :]  # remove cls token

        if self.cfg.MODEL.BACKBONE.TYPE in [
            "pe_core_b",
            "pe_core_l",
            "pe_core_g",
            "pe_spatial_g",
            "pe_spatial_l",
            "pe_spatial_b",
            "pe_spatial_s",
            "pe_spatial_t",
        ]:
            B, N, C = image_embeddings.shape
            H = W = int(N**0.5)  # assumes square layout, here 18
            image_embeddings = image_embeddings.transpose(1, 2).reshape(B, C, H, W)

        if isinstance(image_embeddings, tuple):
            image_embeddings = image_embeddings[-1]
        image_embeddings = image_embeddings.type(x.dtype)

        # Mask condition if available
        if self.cfg.MODEL.PROMPT_ENCODER.get("MASK_EMBED_TYPE", None) is not None:
            # v1: non-iterative mask conditioning
            if self.cfg.MODEL.PROMPT_ENCODER.get("MASK_PROMPT", "v1") == "v1":
                mask_embeddings = self._get_mask_prompt(batch, image_embeddings)
                image_embeddings = image_embeddings + mask_embeddings
            else:
                raise NotImplementedError

        # Prepare input for promptable decoder
        condition_info = self._get_decoder_condition(batch)

        # Initial estimate with a dummy prompt
        keypoints_prompt = torch.zeros((batch_size * num_person, 1, 3)).to(batch["img"])
        keypoints_prompt[:, :, -1] = -2

        # Forward promptable decoder to get updated pose tokens and regression output
        _, pose_output = self.forward_decoder(
            image_embeddings,
            init_estimate=None,
            keypoints=keypoints_prompt,
            prev_estimate=None,
            condition_info=condition_info,
            batch=batch,
            full_output=full_output,
        )

        if self.cfg.MODEL.DECODER.get("DO_INTERM_PREDS", False):
            assert isinstance(
                pose_output, list
            ), "You're doing DO_INTERM_PREDS but pose_output is not a list?"  # Sanity
            pose_output_interm, pose_output = pose_output[:-1], pose_output[-1]
        else:
            pose_output_interm = None

        return {
            # "pose_token": pose_token,
            "atlas": pose_output,  # atlas prediction output
            "atlas_interm": pose_output_interm,  # List of intermediate decoder outputs.
            "condition_info": condition_info,
            "image_embeddings": image_embeddings,
        }

    def forward_step(self, batch: Dict) -> Tuple[Dict, Dict]:
        # Full-image encoder
        full_output = {}
        if hasattr(self, "full_encoder"):
            full_output = self.forward_full_branch(batch)

        # Crop-image (pose) branch
        pose_output = self.forward_pose_branch(batch, full_output)

        return pose_output, full_output

