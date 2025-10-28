from typing import Dict, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sam_3d_body.utils.logging import get_pylogger
import torchvision
import cv2
from sam_3d_body.data.transforms import get_warp_matrix
from ..backbones import create_backbone
from ..decoders import build_decoder, build_keypoint_sampler, PromptEncoder
from ..heads import build_head
from ..modules.transformer import FFN

from .base_model import BaseModel


logger = get_pylogger(__name__)


# fmt: off
PROMPT_KEYPOINTS = {  # keypoint_idx: prompt_idx
    "atlas70": {
        i: i for i in range(70)
    },  # all 70 keypoints are supported for prompting
}
KEY_BODY = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 41, 62]  # key body joints for prompting
# fmt: on

class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C
    

class SAM3DBodyTriplet(BaseModel):

    pelvis_idx = [9, 10]  # left_hip, right_hip

    def _initialze_model(self, estimate_cam_int=False):
        self.register_buffer(
            "image_mean", torch.tensor(self.cfg.MODEL.IMAGE_MEAN).view(-1, 1, 1), False
        )
        self.register_buffer(
            "image_std", torch.tensor(self.cfg.MODEL.IMAGE_STD).view(-1, 1, 1), False
        )

        # Create backbone feature extractor for human crops
        self.backbone = create_backbone(
            self.cfg.MODEL.BACKBONE.TYPE, self.cfg, pretrained=False
        )

        # Create header for pose estimation output
        self.head_pose = build_head(self.cfg, self.cfg.MODEL.PERSON_HEAD.POSE_TYPE)
        if self.cfg.MODEL.get('DISABLE_HAND_PCA', False):
            self.head_pose.atlas.hand_pose_comps_ori = nn.Parameter(self.head_pose.atlas.hand_pose_comps.clone(), requires_grad=False)
            self.head_pose.atlas.hand_pose_comps.data = torch.eye(54).to(self.head_pose.atlas.hand_pose_comps.data).float()
        if self.cfg.LOSS_WEIGHTS.get('HAND_LOSS_KINEMATIC_ANNEALING', False):
            # Cuda version doesn't support double backwards
            self.head_pose.atlas.lbs_fn.force_using_pytorch_functions = True
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

        if self.cfg.MODEL.get('TRIPLET_FUSION_METHOD', 'v0') == "v1":
            self.lhand_pe_layer = PositionEmbeddingRandom(self.backbone.embed_dims // 2)
            self.rhand_pe_layer = PositionEmbeddingRandom(self.backbone.embed_dims // 2)

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
                "vit_hmr_triplet",
                "vit_hmr_triplet_512_384",
                "vit_l_triplet_512_384",
            ]:
                self.ray_cond_emb = nn.Conv2d(
                    2,
                    self.backbone.embed_dim,
                    self.backbone.patch_size,
                    stride=self.backbone.patch_size,
                )
            else:
                assert False, "Not implemented yet"

            # Zero conv
            torch.nn.init.zeros_(self.ray_cond_emb.weight)
            torch.nn.init.zeros_(self.ray_cond_emb.bias)

        if self.cfg.MODEL.DECODER.get("DO_KEYPOINT_TOKENS", False):
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

        if self.cfg.MODEL.DECODER.get("KEYPOINT_TOKEN_UPDATE", None) in [
            "v1",
            "v2",
        ]:
            if not self.cfg.MODEL.DECODER.get(
                "KEYPOINT_TOKEN_UPDATE_COORD_EMB_USE_MLP", False
            ):
                # For some reason that escapes me, the backbone's pos embs are 1280, while the decoder is 1024.
                self.keypoint_posemb_linear = nn.Linear(
                    self.backbone.embed_dims, self.cfg.MODEL.DECODER.DIM
                )
            else:
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
        else:
            assert self.cfg.MODEL.DECODER.get("KEYPOINT_TOKEN_UPDATE", None) is None

        if self.cfg.MODEL.DECODER.get("DO_KEYPOINT3D_TOKENS", False):
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

        if self.cfg.MODEL.DECODER.get("KEYPOINT3D_TOKEN_UPDATE", None) in ["v1"]:
            # v1 is to use an MLP embedding, root normalized
            self.keypoint3d_posemb_linear = FFN(
                embed_dims=3,
                feedforward_channels=self.cfg.MODEL.DECODER.DIM,
                output_dims=self.cfg.MODEL.DECODER.DIM,
                num_fcs=2,
                add_identity=False,
            )

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
        hand_embeddings=None,
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
        assert num_pose_token == 1

        image_augment, token_augment, token_mask = None, None, None
        if hasattr(self, "prompt_encoder") and keypoints is not None:
            if prev_estimate is None:
                # Use initial embedding if no previous embedding
                prev_estimate = init_estimate
            # Previous estimate w/o the CLIFF condition.
            prev_embeddings = self.prev_to_token_atlas(prev_estimate).view(
                batch_size, 1, -1
            )  # 407 -> B x 1 x 1024; linear layer-ed
            hand_augment = None

            if self.cfg.MODEL.BACKBONE.TYPE in [
                "vit_hmr",
                "vit_hmr_triplet",
                "hmr2",
                "vit",
                "vit_b",
                "vit_l",
            ]:
                # HMR2.0 backbone (ViTPose) assumes a different aspect ratio as input size
                image_augment = self.prompt_encoder.get_dense_pe((16, 16))
                image_augment = image_augment[:, :, :, 2:-2]
                if self.cfg.MODEL.get('TRIPLET_FUSION_METHOD', 'v0') == "v1":
                    if self.cfg.MODEL.HAND_IMAGE_SIZE[0] == 256:
                        hand_augment = torch.cat([
                            self.lhand_pe_layer((16, 16)).unsqueeze(0)[:, :, :, 2:-2].flatten(2, 3),
                            self.rhand_pe_layer((16, 16)).unsqueeze(0)[:, :, :, 2:-2].flatten(2, 3),
                        ], dim=2)
                    elif self.cfg.MODEL.HAND_IMAGE_SIZE[0] == 128:
                        hand_augment = torch.cat([
                            self.lhand_pe_layer((8, 8)).unsqueeze(0)[:, :, :, 1:-1].flatten(2, 3),
                            self.rhand_pe_layer((8, 8)).unsqueeze(0)[:, :, :, 1:-1].flatten(2, 3),
                        ], dim=2)
            elif self.cfg.MODEL.BACKBONE.TYPE in [
                "vit_hmr_512_384",
                "vit_hmr_triplet_512_384",
                "vit_l_triplet_512_384",
            ]:
                # HMR2.0 backbone (ViTPose) assumes a different aspect ratio as input size
                image_augment = self.prompt_encoder.get_dense_pe((32, 32))
                image_augment = image_augment[:, :, :, 4:-4]
                if self.cfg.MODEL.get('TRIPLET_FUSION_METHOD', 'v0') == "v1":
                    hand_augment = torch.cat([
                        self.lhand_pe_layer((16, 16)).unsqueeze(0)[:, :, :, 2:-2].flatten(2, 3),
                        self.rhand_pe_layer((16, 16)).unsqueeze(0)[:, :, :, 2:-2].flatten(2, 3),
                    ], dim=2)
            else:
                image_augment = self.prompt_encoder.get_dense_pe(
                    image_embeddings.shape[-2:]
                )  # (1, C, H, W)
                if self.cfg.MODEL.get('TRIPLET_FUSION_METHOD', 'v0') == "v1":
                    # Shape here is hard-coded
                    hand_augment = torch.cat([
                        self.lhand_pe_layer((image_embeddings.shape[-2], image_embeddings.shape[-1])).unsqueeze(0).flatten(2, 3),
                        self.rhand_pe_layer((image_embeddings.shape[-2], image_embeddings.shape[-1])).unsqueeze(0).flatten(2, 3),
                    ], dim=2)

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

        if not self.cfg.MODEL.DECODER.get("DO_INTERM_PREDS", False):
            tokens = self.decoder(
                token_embeddings,
                image_embeddings,
                token_augment,
                image_augment,
                token_mask,
                hand_embeddings=hand_embeddings,
                hand_augment=hand_augment,
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
            def token_to_pose_output_fn(tokens, prev_pose_output, layer_idx):
                # Get the pose token
                pose_token = tokens[:, 0]

                # Pose & camera are done residually. Optionally, we may choose to make their predictions
                # residual through the decoder layers.
                if (
                    self.cfg.MODEL.DECODER.get("DO_INTERM_RESIDUAL_PRED", False)
                    and prev_pose_output is not None
                ):
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
                pose_output = self.head_pose(pose_token, prev_pose)
                # Get Camera Translation
                if hasattr(self, "head_camera"):
                    pred_cam = self.head_camera(pose_token, prev_camera)
                    pose_output["pred_cam"] = pred_cam
                # Run camera projection
                pose_output = self.camera_project(pose_output, batch, full_output)

                # Get 2D KPS in crop
                pose_output["pred_keypoints_2d_cropped"] = self._full_to_crop(
                    batch, pose_output["pred_keypoints_2d"]
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

                    # Get current 2D KPS predictions 
                    pred_keypoints_2d_cropped = pose_output[
                        "pred_keypoints_2d_cropped"
                    ].clone()  # These are -0.5 ~ 0.5
                    pred_keypoints_2d_depth = pose_output[
                        "pred_keypoints_2d_depth"
                    ].clone()

                    # Optionally detach them
                    if self.cfg.MODEL.DECODER.get(
                        "KEYPOINT_TOKEN_UPDATE_DETACH_KPS", True
                    ):
                        pred_keypoints_2d_cropped = pred_keypoints_2d_cropped.detach()
                        pred_keypoints_2d_depth = pred_keypoints_2d_depth.detach()

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
                            "vit_hmr_triplet",
                            "hmr2",
                            "vit",
                            "vit_b",
                            "vit_l",
                            "vit_hmr_512_384",
                            "vit_hmr_triplet_512_384",
                            "vit_l_triplet_512_384",
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

                    # Now, pelvis normalize
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
                hand_embeddings=hand_embeddings,
                hand_augment=hand_augment,
            )

        return pose_token, pose_output

    def get_atlas_output(self, batch, return_keypoints, return_joint_rotations=False, return_joint_params=False):
        gt_verts, gt_j3d, gt_rots, gt_joint_params = self.head_pose.atlas(
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
            return_keypoints=True,
            return_joint_rotations=True,
            return_joint_params=True,
        )
        gt_verts[..., [1, 2]] *= -1  # Camera system difference
        gt_j3d[..., [1, 2]] *= -1  # Camera system difference

        to_return = [gt_verts]

        if return_keypoints:
            to_return.append(gt_j3d)
        if return_joint_rotations:
            to_return.append(gt_rots)
        if return_joint_params:
            to_return.append(gt_joint_params)

        if len(to_return) == 1:
            return to_return[0]
        else:
            return tuple(to_return)

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
        if self.cfg.MODEL.BACKBONE.TYPE in ["vit_hmr", "hmr2", "vit", "vit_hmr_triplet"]:
            # HMR2.0 backbone (ViTPose) assumes a different aspect ratio as input size
            mask_embeddings = mask_embeddings[:, :, :, 2:-2]
        elif self.cfg.MODEL.BACKBONE.TYPE in ["vit_hmr_512_384", "vit_hmr_triplet_512_384", "vit_l_triplet_512_384"]:
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
        hand_embeddings = output.get('hand_embeddings', None)
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
            ] + ([pose_output["pred_global_wrist_raw"]] if pose_output.get('pred_global_wrist_raw', None) is not None else []),
            dim=1,
        ).unsqueeze(dim=1)
        if hasattr(self, "init_camera"):
            prev_estimate = torch.cat(
                [prev_estimate, pose_output["pred_cam"].detach().unsqueeze(1)], dim=-1
            )

        # Get keypoint prompts
        keypoint_prompt = self._get_keypoint_prompt(batch, output)
        if len(prev_prompt):
            cur_keypoint_prompt = torch.cat(prev_prompt + [keypoint_prompt], dim=1)
        else:
            cur_keypoint_prompt = keypoint_prompt

        _, pose_output = self.forward_decoder(
            image_embeddings,
            init_estimate=None,  # not recurring previous estimate
            keypoints=cur_keypoint_prompt,
            prev_estimate=prev_estimate,
            condition_info=condition_info,
            batch=batch,
            full_output=full_output,
            hand_embeddings=hand_embeddings,
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
        return_feature_only: bool = False
    ) -> Dict:
        """Run a forward pass for the crop-image (pose) branch."""
        batch_size, num_person = batch["img"].shape[:2]

        # Forward backbone encoder
        x = self.data_preprocess(
            self._flatten_person(batch["img"]),
            crop_width=(
                self.cfg.MODEL.BACKBONE.TYPE
                in [
                    "vit_hmr",
                    "hmr2",
                    "vit",
                    "vit_b",
                    "vit_l",
                    "vit_hmr_512_384",
                    "vit_hmr_triplet",
                    "vit_hmr_triplet_512_384",
                    "vit_l_triplet_512_384",
                ]
            ),
        )

        crop_hand = self.cfg.MODEL.HAND_IMAGE_SIZE[0] // 4 // 2  # make 4:3 ratio
        assert crop_hand * 8 == self.cfg.MODEL.HAND_IMAGE_SIZE[0]

        lhand_x = self.data_preprocess(
            self._flatten_person(batch["lhand_img"]),
            crop_width=(
                self.cfg.MODEL.BACKBONE.TYPE
                in [
                    "vit_hmr",
                    "hmr2",
                    "vit",
                    "vit_b",
                    "vit_l",
                    "vit_hmr_512_384",
                    "vit_hmr_triplet",
                    "vit_hmr_triplet_512_384",
                    "vit_l_triplet_512_384",
                ]
            ),
            crop_hand=crop_hand,
        )
        rhand_x = self.data_preprocess(
            self._flatten_person(batch["rhand_img"]),
            crop_width=(
                self.cfg.MODEL.BACKBONE.TYPE
                in [
                    "vit_hmr",
                    "hmr2",
                    "vit",
                    "vit_b",
                    "vit_l",
                    "vit_hmr_512_384",
                    "vit_hmr_triplet",
                    "vit_hmr_triplet_512_384",
                    "vit_l_triplet_512_384",
                ]
            ),
            crop_hand=crop_hand,
        )

        # Optionally get ray conditioining
        if self.cfg.MODEL.get("RAY_CONDITION_TYPE", None) is not None:
            ray_cond = self.get_ray_condition(
                batch
            )  # This is B x num_person x 2 x H x W
            ray_cond = self._flatten_person(ray_cond)
            if self.cfg.MODEL.BACKBONE.TYPE in [
                "vit_hmr",
                "vit_hmr_triplet",
                "hmr2",
                "vit",
                "vit_b",
                "vit_l",
            ]:
                ray_cond = ray_cond[:, :, :, 32:-32]
            elif self.cfg.MODEL.BACKBONE.TYPE in [
                "vit_hmr_512_384",
                "vit_hmr_triplet_512_384",
                "vit_l_triplet_512_384",
            ]:
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

        image_embeddings, lhand_image_embeddings, rhand_image_embeddings = (
            self.backbone(
                x.type(self.backbone_dtype),
                lhand_x.type(self.backbone_dtype),
                rhand_x.type(self.backbone_dtype),
                extra_embed=ray_cond,
            )
        )

        if self.cfg.MODEL.get('TRIPLET_FUSION_METHOD', 'v0') =="v0":
            fused_features = self.attention_fusion(
                image_embeddings, lhand_image_embeddings, rhand_image_embeddings
            )
            hand_embeddings = None
        elif self.cfg.MODEL.get('TRIPLET_FUSION_METHOD', 'v0') == "v1":
            fused_features = image_embeddings
            hand_embeddings = torch.cat([lhand_image_embeddings.flatten(2, 3), rhand_image_embeddings.flatten(2, 3)], dim=2)
        else:
            raise Exception

        if return_feature_only:
            return {
            "fused_features": fused_features,
            "hand_embeddings": hand_embeddings,
        }

        image_embeddings = fused_features

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
            hand_embeddings=hand_embeddings,
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
            "fused_features": fused_features,
            "hand_embeddings": hand_embeddings
        }


    def get_hand_box(self, pose_output, batch, scale_factor=None):
        # get the left and right hand boxes and images from the predictions
        left_hand_joint_idx = torch.arange(42, 63)  # 21 joints
        right_hand_joint_idx = torch.arange(21, 42)  # 21 joints
        left_wrist_idx = 62
        right_wrist_idx = 41

        # Get hand crops
        ## First, decode images
        batch_size, num_person = batch["img"].shape[:2]
        full_imgs = [
            img.data for img in batch["img_ori"] for _ in range(num_person)
        ]  # cv2.imdecode(np.frombuffer(img, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        full_imgs_hw = torch.LongTensor(
            [img.shape[:2] for img in full_imgs]
        ).numpy()

        if scale_factor is None:
            scale_factor = 2.0
            scale_factor = 1.6
        hand_body_ratio = 0.10
        square_box = self.cfg.DATASETS.AUGMENTATION.get("SQUARE_BOX", False)
        global save_img_count
        if getattr(self, "hand_crop_strategy", "kps_range") == "kps_range":
            ## Get hand 2d KPS
            left_hand_kps_2d = pose_output["atlas"]["pred_keypoints_2d"][
                :, left_hand_joint_idx
            ]
            right_hand_kps_2d = pose_output["atlas"]["pred_keypoints_2d"][
                :, right_hand_joint_idx
            ]

            ## Get minmaxes (xy)
            left_min = left_hand_kps_2d.amin(dim=1).cpu().long().numpy()
            left_max = left_hand_kps_2d.amax(dim=1).cpu().long().numpy()
            right_min = right_hand_kps_2d.amin(dim=1).cpu().long().numpy()
            right_max = right_hand_kps_2d.amax(dim=1).cpu().long().numpy()
            left_min[:, 0] = np.clip(left_min[:, 0], a_min=0, a_max=full_imgs_hw[:, 1])
            left_min[:, 1] = np.clip(left_min[:, 1], a_min=0, a_max=full_imgs_hw[:, 0])
            left_max[:, 0] = np.clip(left_max[:, 0], a_min=0, a_max=full_imgs_hw[:, 1])
            left_max[:, 1] = np.clip(left_max[:, 1], a_min=0, a_max=full_imgs_hw[:, 0])
            right_min[:, 0] = np.clip(
                right_min[:, 0], a_min=0, a_max=full_imgs_hw[:, 1]
            )
            right_min[:, 1] = np.clip(
                right_min[:, 1], a_min=0, a_max=full_imgs_hw[:, 0]
            )
            right_max[:, 0] = np.clip(
                right_max[:, 0], a_min=0, a_max=full_imgs_hw[:, 1]
            )
            right_max[:, 1] = np.clip(
                right_max[:, 1], a_min=0, a_max=full_imgs_hw[:, 0]
            )
            ## Get center & scale
            left_center = (left_max + left_min) / 2
            right_center = (right_max + right_min) / 2
            left_scale = (left_max - left_min) * scale_factor
            right_scale = (right_max - right_min) * scale_factor
            if square_box:
                left_scale[...] = left_scale.max(axis=1)[:, None]
                right_scale[...] = right_scale.max(axis=1)[:, None]
            ## Stack

            if 'left_scale' not in batch:
                batch['left_scale'] = left_scale
                batch['left_center'] = left_center
                batch['right_scale'] = right_scale
                batch['right_center'] = right_center
            else:
                print("Pulling left & right from batch")
                left_scale = batch['left_scale']
                left_center = batch['left_center']
                right_scale = batch['right_scale']
                right_center = batch['right_center']

        elif getattr(self, "hand_crop_strategy", False) == "wrist_center":
            right_center = pose_output["atlas"]["pred_keypoints_2d"][:, right_wrist_idx].cpu().long().numpy()
            left_center = pose_output["atlas"]["pred_keypoints_2d"][:, left_wrist_idx].cpu().long().numpy()

            wholebody_min = pose_output["atlas"]["pred_keypoints_2d"].amin(dim=1).cpu().long().numpy()
            wholebody_max = pose_output["atlas"]["pred_keypoints_2d"].amax(dim=1).cpu().long().numpy()
            wholebody_min[:, 0] = np.clip(wholebody_min[:, 0], a_min=0, a_max=full_imgs_hw[:, 1])
            wholebody_min[:, 1] = np.clip(wholebody_min[:, 1], a_min=0, a_max=full_imgs_hw[:, 0])
            body_scale = (wholebody_max - wholebody_min) * scale_factor
            hand_scale = body_scale * hand_body_ratio

            left_min = left_center - hand_scale / 2.0
            left_max = left_center + hand_scale / 2.0
            right_min = right_center - hand_scale / 2.0
            right_max = right_center + hand_scale / 2.0

            left_scale = (left_max - left_min) * scale_factor
            right_scale = (right_max - right_min) * scale_factor
            if square_box:
                left_scale[...] = left_scale.max(axis=1)[:, None]
                right_scale[...] = right_scale.max(axis=1)[:, None]

            batch['left_scale'] = left_scale
            batch['left_center'] = left_center
            batch['right_scale'] = right_scale
            batch['right_center'] = right_center
        else:
            assert False, "not supported hand crop strategy: {}".format(getattr(self, "hand_crop_strategy", "kps_range"))

        ## Get warp matrix
        warp_size = batch["lhand_img"].shape[-2:]
        left_warp_mats = [
            get_warp_matrix(left_center_, left_scale_, rot=0, output_size=warp_size)
            for left_center_, left_scale_ in zip(left_center, left_scale)
        ]
        right_warp_mats = [
            get_warp_matrix(
                right_center_, right_scale_, rot=0, output_size=warp_size
            )
            for right_center_, right_scale_ in zip(right_center, right_scale)
        ]
        ## Crop image
        left_crops = [
            cv2.warpAffine(full_img, warp_mat, warp_size, flags=cv2.INTER_LINEAR)
            for warp_mat, full_img in zip(left_warp_mats, full_imgs)
        ]
        right_crops = [
            cv2.warpAffine(full_img, warp_mat, warp_size, flags=cv2.INTER_LINEAR)
            for warp_mat, full_img in zip(right_warp_mats, full_imgs)
        ]

        batch["lhand_img"] = torch.stack(
            [
                torchvision.transforms.functional.to_tensor(tmp)
                for tmp in left_crops
            ],
            dim=0,
        )[:, None].cuda()
        batch["rhand_img"] = torch.stack(
            [
                torchvision.transforms.functional.to_tensor(tmp)
                for tmp in right_crops
            ],
            dim=0,
        )[:, None].cuda()

        return batch

    def forward_step(self, batch: Dict, return_feature_only: bool = False) -> Tuple[Dict, Dict]:
        # Full-image encoder
        full_output = {}

        # Crop-image (pose) branch
        pose_output = self.forward_pose_branch(batch, full_output, return_feature_only=return_feature_only)

        return pose_output, full_output
