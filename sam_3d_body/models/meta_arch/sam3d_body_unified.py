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
from ..modules.camera_embed import CameraEncoder

from .base_model import BaseModel


logger = get_pylogger(__name__)


# fmt: off
PROMPT_KEYPOINTS = {  # keypoint_idx: prompt_idx
    "atlas70": {
        i: i for i in range(70)
    },  # all 70 keypoints are supported for prompting
}
KEY_BODY = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 41, 62]  # key body joints for prompting
KEY_RIGHT_HAND = list(range(21, 42))
# fmt: on

class MLP(nn.Module):
    # borrowed from DET R
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    
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
    


class SAM3DBodyUnified(BaseModel):
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
        self.head_pose.hand_pose_comps_ori = nn.Parameter(self.head_pose.hand_pose_comps.clone(), requires_grad=False)
        self.head_pose.hand_pose_comps.data = torch.eye(54).to(self.head_pose.hand_pose_comps.data).float()
        
        # Initialize pose token with learnable params (not mean pose in SMPL)
        # Note: bias/initial value should be zero-pose in cont, not all-zeros
        self.init_pose = nn.Embedding(1, self.head_pose.npose)

        # Define header for hand pose estimation
        self.head_pose_hand = build_head(
            self.cfg, self.cfg.MODEL.PERSON_HEAD.POSE_TYPE, enable_hand_model=True
        )
        self.head_pose_hand.hand_pose_comps_ori = nn.Parameter(self.head_pose_hand.hand_pose_comps.clone(), requires_grad=False)
        self.head_pose_hand.hand_pose_comps.data = torch.eye(54).to(self.head_pose_hand.hand_pose_comps.data).float()
        self.init_pose_hand = nn.Embedding(1, self.head_pose_hand.npose)

        self.head_camera = build_head(
            self.cfg, self.cfg.MODEL.PERSON_HEAD.CAMERA_TYPE
        )
        self.init_camera = nn.Embedding(1, self.head_camera.ncam)
        nn.init.zeros_(self.init_camera.weight)

        self.head_camera_hand = build_head(
            self.cfg,
            self.cfg.MODEL.PERSON_HEAD.CAMERA_TYPE,
            default_scale_factor=self.cfg.MODEL.CAMERA_HEAD.get(
                "DEFAULT_SCALE_FACTOR_HAND", 1.0
            ),
        )
        self.init_camera_hand = nn.Embedding(1, self.head_camera_hand.ncam)
        nn.init.zeros_(self.init_camera_hand.weight)

        self.camera_type = "perspective"

        # Support conditioned information for decoder
        cond_dim = 3
        init_dim = self.head_pose.npose + self.head_camera.ncam + cond_dim
        self.init_to_token_atlas = nn.Linear(init_dim, self.cfg.MODEL.DECODER.DIM)
        self.prev_to_token_atlas = nn.Linear(
            init_dim - cond_dim, self.cfg.MODEL.DECODER.DIM
        )
        self.init_to_token_atlas_hand = nn.Linear(
            init_dim, self.cfg.MODEL.DECODER.DIM
        )
        self.prev_to_token_atlas_hand = nn.Linear(
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
                keybody_idx=(
                    KEY_BODY
                    if not self.cfg.MODEL.PROMPT_ENCODER.get("SAMPLE_HAND", False)
                    else KEY_RIGHT_HAND
                ),
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
        # shared config for the two decoders
        self.decoder_hand = build_decoder(
            self.cfg.MODEL.DECODER, context_dim=self.backbone.embed_dims
        )
        self.hand_pe_layer = PositionEmbeddingRandom(self.backbone.embed_dims // 2)

        # Manually convert the torso of the model to fp16.
        if self.cfg.TRAIN.USE_FP16:
            self.convert_to_fp16()
            if self.cfg.TRAIN.get("FP16_TYPE", "float16") == "float16":
                self.backbone_dtype = torch.float16
            else:
                self.backbone_dtype = torch.bfloat16
        else:
            self.backbone_dtype = torch.float32

        self.ray_cond_emb = CameraEncoder(
            self.backbone.embed_dim,
            self.backbone.patch_size,
        )
        self.ray_cond_emb_hand = CameraEncoder(
            self.backbone.embed_dim,
            self.backbone.patch_size,
        )

        self.keypoint_embedding_idxs = list(range(70))
        self.keypoint_embedding = nn.Embedding(
            len(self.keypoint_embedding_idxs), self.cfg.MODEL.DECODER.DIM
        )
        self.keypoint_embedding_idxs_hand = list(range(70))
        self.keypoint_embedding_hand = nn.Embedding(
            len(self.keypoint_embedding_idxs_hand), self.cfg.MODEL.DECODER.DIM
        )

        if self.cfg.MODEL.DECODER.get("DO_HAND_DETECT_TOKENS", False):
            self.hand_box_embedding = nn.Embedding(
                2, self.cfg.MODEL.DECODER.DIM
            )  # for two hands
            # decice if there is left or right hand inside the image
            self.hand_cls_embed = nn.Linear(self.cfg.MODEL.DECODER.DIM, 2)
            self.bbox_embed = MLP(
                self.cfg.MODEL.DECODER.DIM, self.cfg.MODEL.DECODER.DIM, 4, 3
            )

            if self.cfg.MODEL.DECODER.get("HAND_DETECT_DO_UNCERT", False):
                self.hand_box_logsigma_embed = FFN(
                    embed_dims=self.cfg.MODEL.DECODER.DIM,
                    feedforward_channels=self.cfg.MODEL.DECODER.DIM,
                    output_dims=1,
                    num_fcs=2,
                    add_identity=False,
                )

        self.keypoint_posemb_linear = FFN(
            embed_dims=2,
            feedforward_channels=self.cfg.MODEL.DECODER.DIM,
            output_dims=self.cfg.MODEL.DECODER.DIM,
            num_fcs=2,
            add_identity=False,
        )
        self.keypoint_posemb_linear_hand = FFN(
            embed_dims=2,
            feedforward_channels=self.cfg.MODEL.DECODER.DIM,
            output_dims=self.cfg.MODEL.DECODER.DIM,
            num_fcs=2,
            add_identity=False,
        )
        self.keypoint_feat_linear = nn.Linear(
            self.backbone.embed_dims, self.cfg.MODEL.DECODER.DIM
        )
        self.keypoint_feat_linear_hand = nn.Linear(
            self.backbone.embed_dims, self.cfg.MODEL.DECODER.DIM
        )

        # Do all KPS
        self.keypoint3d_embedding_idxs = list(range(70))
        self.keypoint3d_embedding = nn.Embedding(
            len(self.keypoint3d_embedding_idxs), self.cfg.MODEL.DECODER.DIM
        )

        # Assume always do full body for the hand decoder
        self.keypoint3d_embedding_idxs_hand = list(range(70))
        self.keypoint3d_embedding_hand = nn.Embedding(
            len(self.keypoint3d_embedding_idxs_hand), self.cfg.MODEL.DECODER.DIM
        )

        self.keypoint3d_posemb_linear = FFN(
            embed_dims=3,
            feedforward_channels=self.cfg.MODEL.DECODER.DIM,
            output_dims=self.cfg.MODEL.DECODER.DIM,
            num_fcs=2,
            add_identity=False,
        )
        self.keypoint3d_posemb_linear_hand = FFN(
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

            if self.cfg.MODEL.BACKBONE.TYPE in [
                "vit_hmr",
                "hmr2",
                "vit",
                "vit_b",
                "vit_l",
                "vit_hmr_decouple",
            ]:
                # HMR2.0 backbone (ViTPose) assumes a different aspect ratio as input size
                image_augment = self.prompt_encoder.get_dense_pe((16, 16))[
                    :, :, :, 2:-2
                ]
            elif self.cfg.MODEL.BACKBONE.TYPE in [
                "vit_hmr_512_384",
                "vit_hmr_decouple_512_384",
            ]:
                # HMR2.0 backbone (ViTPose) assumes a different aspect ratio as input size
                image_augment = self.prompt_encoder.get_dense_pe((32, 32))[
                    :, :, :, 4:-4
                ]
            else:
                image_augment = self.prompt_encoder.get_dense_pe(
                    image_embeddings.shape[-2:]
                )  # (1, C, H, W)

            if self.cfg.MODEL.get("RAY_CONDITION_TYPE", None) == "decoder_v3":
                image_embeddings = self.ray_cond_emb(
                    image_embeddings, batch["ray_cond"]
                )

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
            token_mask = None

            if self.cfg.MODEL.DECODER.get("DO_HAND_DETECT_TOKENS", False):
                # Put in a token for each hand
                hand_det_emb_start_idx = token_embeddings.shape[1]
                token_embeddings = torch.cat(
                    [
                        token_embeddings,
                        self.hand_box_embedding.weight[None, :, :].repeat(
                            batch_size, 1, 1
                        ),
                    ],
                    dim=1,
                )  # B x 5 + 70 x 1024
                # No positional embeddings
                token_augment = torch.cat(
                    [
                        token_augment,
                        torch.zeros_like(
                            token_embeddings[:, token_augment.shape[1] :, :]
                        ),
                    ],
                    dim=1,
                )  # B x 5 + 70 x 1024

            assert self.cfg.MODEL.DECODER.get("DO_KEYPOINT_TOKENS", False)
            # Put in a token for each keypoint
            kps_emb_start_idx = token_embeddings.shape[1]
            token_embeddings = torch.cat(
                [
                    token_embeddings,
                    self.keypoint_embedding.weight[None, :, :].repeat(batch_size, 1, 1),
                ],
                dim=1,
            )  # B x 3 + 70 x 1024
            # No positional embeddings
            token_augment = torch.cat(
                [
                    token_augment,
                    torch.zeros_like(token_embeddings[:, token_augment.shape[1] :, :]),
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

            assert not self.cfg.MODEL.DECODER.get("SEPARATE_AUX_TOKENS_3D", False)

        if not self.cfg.MODEL.DECODER.get("DO_INTERM_PREDS", False):
            assert False, "Not come here, we do interm preds"
            # tokens = self.decoder(
            #     token_embeddings,
            #     image_embeddings,
            #     token_augment,
            #     image_augment,
            #     token_mask,
            # )
            # pose_token = tokens[:, 0]

            # # Get pose outputs (atlas parameters)
            # pose_output = self.head_pose(pose_token, init_pose.view(batch_size, -1))
            # if hasattr(self, "head_camera"):
            #     pred_cam = self.head_camera(
            #         pose_token, init_camera.view(batch_size, -1)
            #     )
            #     pose_output["pred_cam"] = pred_cam

            # pose_output = self.camera_project(pose_output, batch, full_output)
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
                    batch, pose_output["pred_keypoints_2d"], self.body_batch_idx
                )
                
                return pose_output

            if self.cfg.MODEL.DECODER.get("KEYPOINT_TOKEN_UPDATE", None) in [
                "v1",
                "v2",
                "v3",
            ]:
                # For this one, we're going to take the projected 2D KPS, get posembs
                # (prompt encoder style, and set that as the keypoint posemb)
                kp_token_update_fn = self.keypoint_token_update_fn
            else:
                kp_token_update_fn = None

            # Now for 3D
            if self.cfg.MODEL.DECODER.get("KEYPOINT3D_TOKEN_UPDATE", None) in ["v1"]:
                kp3d_token_update_fn = self.keypoint3d_token_update_fn
            else:
                kp3d_token_update_fn = None

            # Combine the 2D and 3D functionse
            def keypoint_token_update_fn_comb(*args):
                if kp_token_update_fn is not None:
                    args = kp_token_update_fn(
                        kps_emb_start_idx, image_embeddings, *args
                    )
                if kp3d_token_update_fn is not None:
                    args = kp3d_token_update_fn(kps3d_emb_start_idx, *args)
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

        if self.cfg.MODEL.DECODER.get("DO_HAND_DETECT_TOKENS", False):
            return (
                pose_token[:, hand_det_emb_start_idx : hand_det_emb_start_idx + 2],
                pose_output,
            )
        else:
            return pose_token, pose_output

    def forward_decoder_hand(
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
            init_pose = self.init_pose_hand.weight.expand(batch_size, -1).unsqueeze(
                dim=1
            )
            if hasattr(self, "init_camera_hand"):
                init_camera = self.init_camera_hand.weight.expand(
                    batch_size, -1
                ).unsqueeze(dim=1)

            init_estimate = (
                init_pose
                if not hasattr(self, "init_camera_hand")
                else torch.cat([init_pose, init_camera], dim=-1)
            )  # This is basically pose & camera translation at the end. B x 1 x (404 + 3)

        if condition_info is not None:
            init_input = torch.cat(
                [condition_info.view(batch_size, 1, -1), init_estimate], dim=-1
            )  # B x 1 x 410 (this is with the CLIFF condition)
        else:
            init_input = init_estimate
        token_embeddings = self.init_to_token_atlas_hand(init_input).view(
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
            prev_embeddings = self.prev_to_token_atlas_hand(prev_estimate).view(
                batch_size, 1, -1
            )  # 407 -> B x 1 x 1024; linear layer-ed

            if self.cfg.MODEL.BACKBONE.TYPE in [
                "vit_hmr",
                "hmr2",
                "vit",
                "vit_b",
                "vit_l",
                "vit_hmr_decouple",
            ]:
                # HMR2.0 backbone (ViTPose) assumes a different aspect ratio as input size
                image_augment = self.hand_pe_layer((16, 16)).unsqueeze(0)[:, :, :, 2:-2]
            elif self.cfg.MODEL.BACKBONE.TYPE in [
                "vit_hmr_512_384",
                "vit_hmr_decouple_512_384",
            ]:
                # HMR2.0 backbone (ViTPose) assumes a different aspect ratio as input size
                image_augment = self.hand_pe_layer((32, 32)).unsqueeze(0)[:, :, :, 4:-4]
            else:
                image_augment = self.hand_pe_layer(
                    image_embeddings.shape[-2:]
                ).unsqueeze(
                    0
                )  # (1, C, H, W)

            if self.cfg.MODEL.get("RAY_CONDITION_TYPE", None) == "decoder_v3":
                image_embeddings = self.ray_cond_emb_hand(
                    image_embeddings, batch["ray_cond_hand"]
                )

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
            token_mask = None

            if self.cfg.MODEL.DECODER.get("DO_HAND_DETECT_TOKENS", False):
                # Put in a token for each hand
                hand_det_emb_start_idx = token_embeddings.shape[1]
                token_embeddings = torch.cat(
                    [
                        token_embeddings,
                        self.hand_box_embedding.weight[None, :, :].repeat(
                            batch_size, 1, 1
                        ),
                    ],
                    dim=1,
                )  # B x 5 + 70 x 1024
                # No positional embeddings
                token_augment = torch.cat(
                    [
                        token_augment,
                        torch.zeros_like(
                            token_embeddings[:, token_augment.shape[1] :, :]
                        ),
                    ],
                    dim=1,
                )  # B x 5 + 70 x 1024

            assert self.cfg.MODEL.DECODER.get("DO_KEYPOINT_TOKENS", False)
            # Put in a token for each keypoint
            kps_emb_start_idx = token_embeddings.shape[1]
            token_embeddings = torch.cat(
                [
                    token_embeddings,
                    self.keypoint_embedding_hand.weight[None, :, :].repeat(
                        batch_size, 1, 1
                    ),
                ],
                dim=1,
            )  # B x 3 + 70 x 1024
            # No positional embeddings
            token_augment = torch.cat(
                [
                    token_augment,
                    torch.zeros_like(token_embeddings[:, token_augment.shape[1] :, :]),
                ],
                dim=1,
            )  # B x 3 + 70 x 1024

            if self.cfg.MODEL.DECODER.get("DO_KEYPOINT3D_TOKENS", False):
                # Put in a token for each keypoint
                kps3d_emb_start_idx = token_embeddings.shape[1]
                token_embeddings = torch.cat(
                    [
                        token_embeddings,
                        self.keypoint3d_embedding_hand.weight[None, :, :].repeat(
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

            assert not self.cfg.MODEL.DECODER.get("SEPARATE_AUX_TOKENS_3D", False)

        if not self.cfg.MODEL.DECODER.get("DO_INTERM_PREDS", False):
            assert False, "Not come here, we do interm preds"
            # tokens = self.decoder(
            #     token_embeddings,
            #     image_embeddings,
            #     token_augment,
            #     image_augment,
            #     token_mask,
            # )
            # pose_token = tokens[:, 0]

            # # Get pose outputs (atlas parameters)
            # pose_output = self.head_pose(pose_token, init_pose.view(batch_size, -1))
            # if hasattr(self, "head_camera"):
            #     pred_cam = self.head_camera(
            #         pose_token, init_camera.view(batch_size, -1)
            #     )
            #     pose_output["pred_cam"] = pred_cam

            # pose_output = self.camera_project(pose_output, batch, full_output)
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
                pose_output = self.head_pose_hand(pose_token, prev_pose)
                
                # Get Camera Translation
                if hasattr(self, "head_camera_hand"):
                    pred_cam = self.head_camera_hand(pose_token, prev_camera)
                    pose_output["pred_cam"] = pred_cam
                # Run camera projection
                pose_output = self.camera_project_hand(pose_output, batch, full_output)

                # Get 2D KPS in crop (we don't need it usually, but it's cheap to compute anyway so why not)
                pose_output["pred_keypoints_2d_cropped"] = self._full_to_crop(
                    batch, pose_output["pred_keypoints_2d"], self.hand_batch_idx
                )

                return pose_output

            if self.cfg.MODEL.DECODER.get("KEYPOINT_TOKEN_UPDATE", None) in [
                "v1",
                "v2",
                "v3",
            ]:
                # For this one, we're going to take the projected 2D KPS, get posembs
                # (prompt encoder style, and set that as the keypoint posemb)
                kp_token_update_fn = self.keypoint_token_update_fn_hand
            else:
                kp_token_update_fn = None

            # Now for 3D
            if self.cfg.MODEL.DECODER.get("KEYPOINT3D_TOKEN_UPDATE", None) in ["v1"]:
                kp3d_token_update_fn = self.keypoint3d_token_update_fn_hand
            else:
                kp3d_token_update_fn = None

            # Combine the 2D and 3D functionse
            def keypoint_token_update_fn_comb(*args):
                if kp_token_update_fn is not None:
                    args = kp_token_update_fn(
                        kps_emb_start_idx, image_embeddings, *args
                    )
                if kp3d_token_update_fn is not None:
                    args = kp3d_token_update_fn(kps3d_emb_start_idx, *args)
                return args

            pose_token, pose_output = self.decoder_hand(
                token_embeddings,
                image_embeddings,
                token_augment,
                image_augment,
                token_mask,
                token_to_pose_output_fn=token_to_pose_output_fn,
                keypoint_token_update_fn=keypoint_token_update_fn_comb,
            )

        if self.cfg.MODEL.DECODER.get("DO_HAND_DETECT_TOKENS", False):
            return (
                pose_token[:, hand_det_emb_start_idx : hand_det_emb_start_idx + 2],
                pose_output,
            )
        else:
            return pose_token, pose_output

    def get_atlas_output(self, batch, return_keypoints, return_joint_rotations=False, return_joint_params=False):
        gt_verts, gt_j3d, gt_rots, gt_joint_params = self.head_pose.mhr_forward(
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
    def _get_keypoint_prompt(self, batch, pred_keypoints_2d, force_dummy=False):
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
        x_mask = self._flatten_person(batch["mask"])
        mask_embeddings, no_mask_embeddings = self.prompt_encoder.get_mask_embeddings(
            x_mask, image_embeddings.shape[0], image_embeddings.shape[2:]
        )
        if self.cfg.MODEL.BACKBONE.TYPE in [
            "vit_hmr",
            "hmr2",
            "vit",
            "vit_hmr_decouple",
        ]:
            # HMR2.0 backbone (ViTPose) assumes a different aspect ratio as input size
            mask_embeddings = mask_embeddings[:, :, :, 2:-2]
        elif self.cfg.MODEL.BACKBONE.TYPE in [
            "vit_hmr_512_384",
        ]:
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

        if "atlas" in output and output["atlas"] is not None:
            pose_output = output["atlas"]  # body-only output
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
            if hasattr(self, "init_camera"):
                prev_estimate = torch.cat(
                    [prev_estimate, pose_output["pred_cam"].detach().unsqueeze(1)],
                    dim=-1,
                )
            prev_shape = prev_estimate.shape[1:]

            pred_keypoints_2d = output["atlas"]["pred_keypoints_2d"].detach().clone()
            kpt_shape = pred_keypoints_2d.shape[1:]

        if "atlas_hand" in output and output["atlas_hand"] is not None:
            pose_output_hand = output["atlas_hand"]
            # Use previous estimate as initialization
            prev_estimate_hand = torch.cat(
                [
                    pose_output_hand["pred_pose_raw"].detach(),  # (B, 6)
                    pose_output_hand["shape"].detach(),
                    pose_output_hand["scale"].detach(),
                    pose_output_hand["hand"].detach(),
                    pose_output_hand["face"].detach(),
                ],
                dim=1,
            ).unsqueeze(dim=1)
            if hasattr(
                self, "init_camera_hand"
            ):
                prev_estimate_hand = torch.cat(
                    [
                        prev_estimate_hand,
                        pose_output_hand["pred_cam"].detach().unsqueeze(1),
                    ],
                    dim=-1,
                )
            prev_shape = prev_estimate_hand.shape[1:]

            pred_keypoints_2d_hand = (
                output["atlas_hand"]["pred_keypoints_2d"].detach().clone()
            )
            kpt_shape = pred_keypoints_2d_hand.shape[1:]

        all_prev_estimate = torch.zeros(
            (image_embeddings.shape[0], *prev_shape), device=image_embeddings.device
        )
        if "atlas" in output and output["atlas"] is not None:
            all_prev_estimate[self.body_batch_idx] = prev_estimate
        if "atlas_hand" in output and output["atlas_hand"] is not None:
            all_prev_estimate[self.hand_batch_idx] = prev_estimate_hand

        # Get keypoint prompts
        all_pred_keypoints_2d = torch.zeros(
            (image_embeddings.shape[0], *kpt_shape), device=image_embeddings.device
        )
        if "atlas" in output and output["atlas"] is not None:
            all_pred_keypoints_2d[self.body_batch_idx] = pred_keypoints_2d
        if "atlas_hand" in output and output["atlas_hand"] is not None:
            all_pred_keypoints_2d[self.hand_batch_idx] = pred_keypoints_2d_hand

        keypoint_prompt = self._get_keypoint_prompt(batch, all_pred_keypoints_2d)
        if len(prev_prompt):
            cur_keypoint_prompt = torch.cat(prev_prompt + [keypoint_prompt], dim=1)
        else:
            cur_keypoint_prompt = keypoint_prompt  # [B, 1, 3]

        pose_output, pose_output_hand = None, None
        if len(self.body_batch_idx):
            tokens_output, pose_output = self.forward_decoder(
                image_embeddings[self.body_batch_idx],
                init_estimate=None,  # not recurring previous estimate
                keypoints=cur_keypoint_prompt[self.body_batch_idx],
                prev_estimate=all_prev_estimate[self.body_batch_idx],
                condition_info=condition_info[self.body_batch_idx],
                batch=batch,
                full_output=None,
            )
            pose_output = pose_output[-1]

        if len(self.hand_batch_idx) and self.cfg.MODEL.DO_HAND_PROMPT:
            tokens_output_hand, pose_output_hand = self.forward_decoder_hand(
                image_embeddings[self.hand_batch_idx],
                init_estimate=None,  # not recurring previous estimate
                keypoints=cur_keypoint_prompt[self.hand_batch_idx],
                prev_estimate=all_prev_estimate[self.hand_batch_idx],
                condition_info=condition_info[self.hand_batch_idx],
                batch=batch,
                full_output=None,
            )
            pose_output_hand = pose_output_hand[-1]

        # Update prediction output
        output.update(
            {
                "atlas": pose_output,  # atlas prediction output
                "atlas_hand": pose_output_hand,  # atlas prediction output
            }
        )

        return output, keypoint_prompt

    def _full_to_crop(
        self,
        batch: Dict,
        pred_keypoints_2d: torch.Tensor,
        batch_idx: torch.Tensor = None,
    ) -> torch.Tensor:
        """Convert full-image keypoints coordinates to crop and normalize to [-0.5. 0.5]"""
        pred_keypoints_2d_cropped = torch.cat(
            [pred_keypoints_2d, torch.ones_like(pred_keypoints_2d[:, :, [-1]])], dim=-1
        )
        if batch_idx is not None:
            affine_trans = self._flatten_person(batch["affine_trans"])[batch_idx].to(
                pred_keypoints_2d_cropped
            )
            img_size = self._flatten_person(batch["img_size"])[batch_idx].unsqueeze(1)
        else:
            affine_trans = self._flatten_person(batch["affine_trans"]).to(
                pred_keypoints_2d_cropped
            )
            img_size = self._flatten_person(batch["img_size"]).unsqueeze(1)
        pred_keypoints_2d_cropped = pred_keypoints_2d_cropped @ affine_trans.mT
        pred_keypoints_2d_cropped = pred_keypoints_2d_cropped[..., :2] / img_size - 0.5

        return pred_keypoints_2d_cropped

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
            assert False

        cam_out = head_camera.perspective_projection(
            pose_output["pred_keypoints_3d"],
            pred_cam,
            self._flatten_person(batch["bbox_center"])[self.body_batch_idx],
            self._flatten_person(batch["bbox_scale"])[self.body_batch_idx, 0],
            self._flatten_person(batch["ori_img_size"])[self.body_batch_idx],
            self._flatten_person(
                batch["cam_int"]
                .unsqueeze(1)
                .expand(-1, batch["img"].shape[1], -1, -1)
                .contiguous()
            )[self.body_batch_idx],
            use_intrin_center=self.cfg.MODEL.DECODER.get("USE_INTRIN_CENTER", False),
        )

        if pose_output.get("pred_vertices", None) is not None:
            cam_out_vertices = head_camera.perspective_projection(
                pose_output["pred_vertices"],
                pred_cam,
                self._flatten_person(batch["bbox_center"])[self.body_batch_idx],
                self._flatten_person(batch["bbox_scale"])[self.body_batch_idx, 0],
                self._flatten_person(batch["ori_img_size"])[self.body_batch_idx],
                self._flatten_person(
                    batch["cam_int"]
                    .unsqueeze(1)
                    .expand(-1, batch["img"].shape[1], -1, -1)
                    .contiguous()
                )[self.body_batch_idx],
                use_intrin_center=self.cfg.MODEL.DECODER.get(
                    "USE_INTRIN_CENTER", False
                ),
            )
            pose_output["pred_keypoints_2d_verts"] = cam_out_vertices[
                "pred_keypoints_2d"
            ]

        pose_output.update(cam_out)

        return pose_output

    def camera_project_hand(
        self, pose_output: Dict, batch: Dict, full_output: Dict
    ) -> Dict:
        """
        Project 3D keypoints to 2D using the camera parameters.
        Args:
            pose_output (Dict): Dictionary containing the pose output.
            batch (Dict): Dictionary containing the batch data.
        Returns:
            Dict: Dictionary containing the projected 2D keypoints.
        """
        if hasattr(self, "head_camera_hand"):
            head_camera = self.head_camera_hand
            pred_cam = pose_output["pred_cam"]
        else:
            assert False

        cam_out = head_camera.perspective_projection(
            pose_output["pred_keypoints_3d"],
            pred_cam,
            self._flatten_person(batch["bbox_center"])[self.hand_batch_idx],
            self._flatten_person(batch["bbox_scale"])[self.hand_batch_idx, 0],
            self._flatten_person(batch["ori_img_size"])[self.hand_batch_idx],
            self._flatten_person(
                batch["cam_int"]
                .unsqueeze(1)
                .expand(-1, batch["img"].shape[1], -1, -1)
                .contiguous()
            )[self.hand_batch_idx],
            use_intrin_center=self.cfg.MODEL.DECODER.get("USE_INTRIN_CENTER", False),
        )

        if pose_output.get("pred_vertices", None) is not None:
            cam_out_vertices = head_camera.perspective_projection(
                pose_output["pred_vertices"],
                pred_cam,
                self._flatten_person(batch["bbox_center"])[self.hand_batch_idx],
                self._flatten_person(batch["bbox_scale"])[self.hand_batch_idx, 0],
                self._flatten_person(batch["ori_img_size"])[self.hand_batch_idx],
                self._flatten_person(
                    batch["cam_int"]
                    .unsqueeze(1)
                    .expand(-1, batch["img"].shape[1], -1, -1)
                    .contiguous()
                )[self.hand_batch_idx],
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
        if self.cfg.MODEL.RAY_CONDITION_TYPE in ["decoder_v2", "decoder_v3"]:
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
                in [
                    "vit_hmr",
                    "hmr2",
                    "vit",
                    "vit_b",
                    "vit_l",
                    "vit_hmr_512_384",
                    "vit_hmr_decouple",
                    "vit_hmr_decouple_512_384",
                ]
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
                "vit_hmr_decouple",
            ]:
                ray_cond = ray_cond[:, :, :, 32:-32]
            elif self.cfg.MODEL.BACKBONE.TYPE in [
                "vit_hmr_512_384",
                "vit_hmr_decouple_512_384",
            ]:
                ray_cond = ray_cond[:, :, :, 64:-64]

            if len(self.body_batch_idx):
                batch["ray_cond"] = ray_cond[self.body_batch_idx].clone()
            if len(self.hand_batch_idx):
                batch["ray_cond_hand"] = ray_cond[self.hand_batch_idx].clone()
            ray_cond = None
        else:
            ray_cond = None

        if "decouple" in self.cfg.MODEL.BACKBONE.TYPE:
            body_x = (
                x[self.body_batch_idx].type(self.backbone_dtype)
                if len(self.body_batch_idx)
                else None
            )
            hand_x = (
                x[self.hand_batch_idx].type(self.backbone_dtype)
                if len(self.hand_batch_idx)
                else None
            )
            body_embeddings, hand_embeddings = self.backbone(
                body_x,
                hand_x,
                extra_embed=None,
            )  # (B, C, H, W)
            if isinstance(body_embeddings, tuple):
                body_embeddings = body_embeddings[-1]
            if isinstance(hand_embeddings, tuple):
                hand_embeddings = hand_embeddings[-1]

            embedding_shape = (
                body_embeddings.shape[1:]
                if body_embeddings is not None
                else hand_embeddings.shape[1:]
            )
            dtype = (
                body_embeddings.dtype
                if body_embeddings is not None
                else hand_embeddings.dtype
            )
            device = (
                body_embeddings.device
                if body_embeddings is not None
                else hand_embeddings.device
            )
            image_embeddings = torch.zeros(
                (x.shape[0], *embedding_shape),
                dtype=dtype,
                device=device,
            )
            if len(self.body_batch_idx):
                image_embeddings[self.body_batch_idx] = body_embeddings
            if len(self.hand_batch_idx):
                image_embeddings[self.hand_batch_idx] = hand_embeddings
            image_embeddings = image_embeddings.type(x.dtype)
        else:
            image_embeddings = self.backbone(
                x.type(self.backbone_dtype), extra_embed=ray_cond
            )  # (B, C, H, W)

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
        pose_output, pose_output_hand = None, None
        if len(self.body_batch_idx):
            tokens_output, pose_output = self.forward_decoder(
                image_embeddings[self.body_batch_idx],
                init_estimate=None,
                keypoints=keypoints_prompt[self.body_batch_idx],
                prev_estimate=None,
                condition_info=condition_info[self.body_batch_idx],
                batch=batch,
                full_output=None,
            )
            pose_output = pose_output[-1]
        if len(self.hand_batch_idx):
            tokens_output_hand, pose_output_hand = self.forward_decoder_hand(
                image_embeddings[self.hand_batch_idx],
                init_estimate=None,
                keypoints=keypoints_prompt[self.hand_batch_idx],
                prev_estimate=None,
                condition_info=condition_info[self.hand_batch_idx],
                batch=batch,
                full_output=None,
            )
            pose_output_hand = pose_output_hand[-1]

        output = {
            # "pose_token": pose_token,
            "atlas": pose_output,  # atlas prediction output
            "atlas_hand": pose_output_hand,  # atlas prediction output
            "condition_info": condition_info,
            "image_embeddings": image_embeddings,
        }

        if self.cfg.MODEL.DECODER.get("DO_HAND_DETECT_TOKENS", False):
            if len(self.body_batch_idx):
                output_hand_box_tokens = tokens_output
                hand_coords = self.bbox_embed(
                    output_hand_box_tokens
                ).sigmoid()  # x1, y1, w, h for body samples, 0 ~ 1
                hand_logits = self.hand_cls_embed(output_hand_box_tokens)

                output["atlas"]["hand_box"] = hand_coords
                output["atlas"]["hand_logits"] = hand_logits
                if self.cfg.MODEL.DECODER.get("HAND_DETECT_DO_UNCERT", False):
                    pred_hand_box_logsigma = self.hand_box_logsigma_embed(output_hand_box_tokens)
                    output["atlas"]["pred_hand_box_logsigma"] = pred_hand_box_logsigma

            if len(self.hand_batch_idx):
                output_hand_box_tokens_hand_batch = tokens_output_hand

                hand_coords_hand_batch = self.bbox_embed(
                    output_hand_box_tokens_hand_batch
                ).sigmoid()  # x1, y1, w, h for hand samples
                hand_logits_hand_batch = self.hand_cls_embed(output_hand_box_tokens_hand_batch)

                output["atlas_hand"]["hand_box"] = hand_coords_hand_batch
                output["atlas_hand"]["hand_logits"] = hand_logits_hand_batch
                if self.cfg.MODEL.DECODER.get("HAND_DETECT_DO_UNCERT", False):
                    pred_hand_box_logsigma_hand_batch = self.hand_box_logsigma_embed(output_hand_box_tokens_hand_batch)
                    output["atlas_hand"]["pred_hand_box_logsigma"] = pred_hand_box_logsigma_hand_batch

        return output

    def forward_step(self, batch: Dict) -> Tuple[Dict, Dict]:
        # Full-image encoder
        full_output = {}

        # Crop-image (pose) branch
        pose_output = self.forward_pose_branch(batch, full_output)

        return pose_output, full_output

    def keypoint_token_update_fn(
        self,
        kps_emb_start_idx,
        image_embeddings,
        token_embeddings,
        token_augment,
        pose_output,
        layer_idx,
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
        pred_keypoints_2d_depth = pose_output["pred_keypoints_2d_depth"].clone()

        # Optionally detach them (backpropping through random sincos.. hmm)
        if self.cfg.MODEL.DECODER.get("KEYPOINT_TOKEN_UPDATE_DETACH_KPS", True):
            pred_keypoints_2d_cropped = pred_keypoints_2d_cropped.detach()
            pred_keypoints_2d_depth = pred_keypoints_2d_depth.detach()

        # have this for atlas
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
            pred_keypoints_2d_cropped_emb = self.prompt_encoder.pe_layer._pe_encoding(
                pred_keypoints_2d_cropped_01
            )
            # Zero invalid pos embs out
            pred_keypoints_2d_cropped_emb = pred_keypoints_2d_cropped_emb * (
                ~invalid_mask[:, :, None]
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
            pred_keypoints_2d_cropped_sample_points = pred_keypoints_2d_cropped * 2
            if self.cfg.MODEL.BACKBONE.TYPE in [
                "vit_hmr",
                "hmr2",
                "vit",
                "vit_b",
                "vit_l",
                "vit_hmr_512_384",
                "vit_hmr_decouple",
                "vit_hmr_decouple_512_384",
            ]:
                # Need to go from 256 x 256 coords to 256 x 192 (HW) because image_embeddings is 16x12
                # Aka, for x, what was normally -1 ~ 1 for 256 should be -16/12 ~ 16/12 (since to sample at original 256, need to overflow)
                pred_keypoints_2d_cropped_sample_points[:, :, 0] = (
                    pred_keypoints_2d_cropped_sample_points[:, :, 0] / 12 * 16
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
                    pred_keypoints_2d_cropped_feats * (~invalid_mask[:, :, None])
                )
                # This is ADDING
                token_embeddings = token_embeddings.clone()
                token_embeddings[
                    :,
                    kps_emb_start_idx : kps_emb_start_idx + num_keypoints,
                    :,
                ] += self.keypoint_feat_linear(pred_keypoints_2d_cropped_feats)

        return token_embeddings, token_augment, pose_output, layer_idx

    def keypoint3d_token_update_fn(
        self,
        kps3d_emb_start_idx,
        token_embeddings,
        token_augment,
        pose_output,
        layer_idx,
    ):
        # It's already after the last layer, we're done.
        if layer_idx == len(self.decoder.layers) - 1:
            return token_embeddings, token_augment, pose_output, layer_idx

        num_keypoints3d = self.keypoint3d_embedding.weight.shape[0]

        # Get current 3D kps predictions
        pred_keypoints_3d = pose_output["pred_keypoints_3d"].clone()

        # This is a hack, as during inference, smpl joints are used, so the 70 atlas kps gets tossed.
        # As such, I've stacked smpl joints & 70 atlas kps joints in atlas head, so just for this part, we can keep the 70.
        
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
        pred_keypoints_3d = pred_keypoints_3d[:, self.keypoint3d_embedding_idxs]

        # Run through embedding MLP & put in
        token_augment = token_augment.clone()
        token_augment[
            :,
            kps3d_emb_start_idx : kps3d_emb_start_idx + num_keypoints3d,
            :,
        ] = self.keypoint3d_posemb_linear(pred_keypoints_3d)

        # TODO: (jinhyun1) these 3D KPS tokens should have auxiliary 3D kps pred, just like 2D? Like where they want to belong
        return token_embeddings, token_augment, pose_output, layer_idx

    def keypoint_token_update_fn_hand(
        self,
        kps_emb_start_idx,
        image_embeddings,
        token_embeddings,
        token_augment,
        pose_output,
        layer_idx,
    ):
        # It's already after the last layer, we're done.
        if layer_idx == len(self.decoder_hand.layers) - 1:
            return token_embeddings, token_augment, pose_output, layer_idx

        # Clone
        token_embeddings = token_embeddings.clone()
        token_augment = token_augment.clone()

        num_keypoints = self.keypoint_embedding_hand.weight.shape[0]

        # Get current 2D KPS predictions # TODO (jinhyun1): Get 3D kps too, put into the model. 2D helps 2D, doesn't help 3D
        pred_keypoints_2d_cropped = pose_output[
            "pred_keypoints_2d_cropped"
        ].clone()  # These are -0.5 ~ 0.5 (CHECK!!!!!!!)
        pred_keypoints_2d_depth = pose_output["pred_keypoints_2d_depth"].clone()

        # Optionally detach them (backpropping through random sincos.. hmm)
        if self.cfg.MODEL.DECODER.get("KEYPOINT_TOKEN_UPDATE_DETACH_KPS", True):
            pred_keypoints_2d_cropped = pred_keypoints_2d_cropped.detach()
            pred_keypoints_2d_depth = pred_keypoints_2d_depth.detach()

        # have this for atlas or smpl
        pred_keypoints_2d_cropped = pred_keypoints_2d_cropped[
            :, self.keypoint_embedding_idxs_hand
        ]
        pred_keypoints_2d_depth = pred_keypoints_2d_depth[
            :, self.keypoint_embedding_idxs_hand
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
            pred_keypoints_2d_cropped_emb = self.prompt_encoder.pe_layer._pe_encoding(
                pred_keypoints_2d_cropped_01
            )
            # Zero invalid pos embs out
            pred_keypoints_2d_cropped_emb = pred_keypoints_2d_cropped_emb * (
                ~invalid_mask[:, :, None]
            )
            # Put them in
            token_augment[
                :, kps_emb_start_idx : kps_emb_start_idx + num_keypoints, :
            ] = self.keypoint_posemb_linear_hand(pred_keypoints_2d_cropped_emb)
        else:
            # TODO (jinhyun1): NOTE: Here, note that the OUTPUT is multiplied by 0. upstairs, the INPUT is.
            token_augment[
                :, kps_emb_start_idx : kps_emb_start_idx + num_keypoints, :
            ] = self.keypoint_posemb_linear_hand(pred_keypoints_2d_cropped) * (
                ~invalid_mask[:, :, None]
            )

        # Also maybe update token_embeddings with the grid sampled 2D feature.
        # Remember that pred_keypoints_2d_cropped are -0.5 ~ 0.5. We want -1 ~ 1
        if self.cfg.MODEL.DECODER.KEYPOINT_TOKEN_UPDATE in ["v2", "v3"]:
            # Sample points...
            ## Get sampling points
            pred_keypoints_2d_cropped_sample_points = pred_keypoints_2d_cropped * 2
            if self.cfg.MODEL.BACKBONE.TYPE in [
                "vit_hmr",
                "hmr2",
                "vit",
                "vit_b",
                "vit_l",
                "vit_hmr_512_384",
                "vit_hmr_decouple",
                "vit_hmr_decouple_512_384",
            ]:
                # Need to go from 256 x 256 coords to 256 x 192 (HW) because image_embeddings is 16x12
                # Aka, for x, what was normally -1 ~ 1 for 256 should be -16/12 ~ 16/12 (since to sample at original 256, need to overflow)
                pred_keypoints_2d_cropped_sample_points[:, :, 0] = (
                    pred_keypoints_2d_cropped_sample_points[:, :, 0] / 12 * 16
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
                    pred_keypoints_2d_cropped_feats * (~invalid_mask[:, :, None])
                )
                # This is ADDING
                token_embeddings = token_embeddings.clone()
                token_embeddings[
                    :,
                    kps_emb_start_idx : kps_emb_start_idx + num_keypoints,
                    :,
                ] += self.keypoint_feat_linear_hand(pred_keypoints_2d_cropped_feats)

        return token_embeddings, token_augment, pose_output, layer_idx

    def keypoint3d_token_update_fn_hand(
        self,
        kps3d_emb_start_idx,
        token_embeddings,
        token_augment,
        pose_output,
        layer_idx,
    ):
        # It's already after the last layer, we're done.
        if layer_idx == len(self.decoder_hand.layers) - 1:
            return token_embeddings, token_augment, pose_output, layer_idx

        num_keypoints3d = self.keypoint3d_embedding_hand.weight.shape[0]

        # Get current 3D kps predictions
        pred_keypoints_3d = pose_output["pred_keypoints_3d"].clone()

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
        pred_keypoints_3d = pred_keypoints_3d[:, self.keypoint3d_embedding_idxs_hand]

        # Run through embedding MLP & put in
        token_augment = token_augment.clone()
        token_augment[
            :,
            kps3d_emb_start_idx : kps3d_emb_start_idx + num_keypoints3d,
            :,
        ] = self.keypoint3d_posemb_linear_hand(pred_keypoints_3d)

        # TODO: (jinhyun1) these 3D KPS tokens should have auxiliary 3D kps pred, just like 2D? Like where they want to belong
        return token_embeddings, token_augment, pose_output, layer_idx