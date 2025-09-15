from typing import Optional, Tuple

import roma

import torch
import torch.nn as nn

from core.metadata import SMPL_TO_OPENPOSE
from core.models.modules.geometry_utils import perspective_projection

from ..modules import aa_to_rotmat, get_intrinsic_matrix, rot6d_to_rotmat, to_2tuple
from ..modules.atlas46 import ATLAS46
from ..modules.atlas_utils import (
    atlas46_param_hand_mask,
    batch6DFromXYZ,
    compact_cont_to_model_params_body,
    compact_model_params_to_cont_body,
)

from ..modules.transformer import FFN


class ATLAS46Head(nn.Module):
    """Predict ATLAS parameters and return LOD3/SMPL vertices"""

    def __init__(
        self,
        input_dim: int,
        mlp_depth: int = 1,
        drop_ratio: float = 0.0,
        joint_rep_type: str = "cont",
        num_body_joints: int = 77,
        num_hand_shape_comps: int = 0,
        num_shape_comps: int = 128,
        num_scale_comps: int = 38,
        num_hand_comps: int = 32,
        num_face_comps: int = 10,
        atlas_model_path: str = "",
        mesh_type: str = "lod3",
        extra_joint_regressor: str = "",
        smpl_model_path: str = "",
        fix_kps_eye_and_chin: bool = True,
        znorm_fullbody_scales: bool = True,
        ffn_zero_bias: bool = False,
        mlp_channel_div_factor: int = 8,
        enable_slim_keypoint_mapping: bool = False,
        zero_face: bool = False,
    ):
        super().__init__()

        # Always use 6D representation for the 3D rotations
        # "On the Continuity of Rotation Representations in Neural Networks"
        self.joint_rep_type = joint_rep_type

        # joint rotation + shape + camera
        # self.num_body_joints = num_body_joints
        self.num_hand_shape_comps = num_hand_shape_comps
        self.num_shape_comps = num_shape_comps
        self.num_scale_comps = num_scale_comps
        self.num_hand_comps = num_hand_comps
        self.num_face_comps = num_face_comps

        self.zero_face = zero_face

        self.body_cont_dim = 260
        self.npose = (
            6  # Global Rotation
            + self.body_cont_dim  # then body
            + num_shape_comps
            + num_scale_comps
            + num_hand_comps * 2
            + num_face_comps
        )

        self.proj = FFN(
            embed_dims=input_dim,
            feedforward_channels=input_dim // mlp_channel_div_factor,
            output_dims=self.npose,
            num_fcs=mlp_depth,
            ffn_drop=drop_ratio,
            add_identity=False,
        )

        if ffn_zero_bias:
            torch.nn.init.zeros_(self.proj.layers[-2].bias)

        self.mesh_type = mesh_type
        assert self.mesh_type in {"lod3", "smpl", "smplx"}
        self.atlas = ATLAS46(
            atlas_model_path,
            num_shape_comps,
            num_scale_comps,
            num_hand_comps,
            num_face_comps,
            lod=self.mesh_type,
            load_keypoint_mapping=(self.mesh_type in ["lod3", "smpl"]),
            verbose=True,
            fix_kps_eye_and_chin=fix_kps_eye_and_chin,
            znorm_fullbody_scales=znorm_fullbody_scales,
            num_hand_shape_comps=num_hand_shape_comps,
            enable_slim_keypoint_mapping=enable_slim_keypoint_mapping,
        )
        for param in self.atlas.parameters():
            param.requires_grad = False

        if self.mesh_type in ["smpl", "smplx"]:
            raise ValueError("ATLAS to SMPL/SMPLX conversion is not supported!")

    def get_zero_pose_init(self, factor=1.0):
        # Initialize pose token with zero-initialized learnable params
        # Note: bias/initial value should be zero-pose in cont, not all-zeros
        weights = torch.zeros(1, self.npose)
        weights[:, : 6 + self.body_cont_dim] = torch.cat(
            [
                torch.FloatTensor([1, 0, 0, 0, 1, 0]),
                compact_model_params_to_cont_body(torch.zeros(1, 133)).squeeze()
                * factor,
            ],
            dim=0,
        )
        return weights

    def forward(
        self,
        x: torch.Tensor,
        init_estimate: Optional[torch.Tensor] = None,
        do_pcblend=True,
        slim_keypoints=False,
    ):
        """
        Args:
            x: pose token with shape [B, C], usually C=DECODER.DIM
            init_estimate: [B, self.npose]
        """
        batch_size = x.shape[0]
        pred = self.proj(x)
        if init_estimate is not None:
            pred = pred + init_estimate

        # From pred, we want to pull out individual predictions.

        ## First, get globals
        ### Global rotation is first 6.
        count = 6
        global_rot_6d = pred[:, :count]
        global_rot_rotmat = rot6d_to_rotmat(global_rot_6d)  # B x 3 x 3
        global_rot_euler = roma.rotmat_to_euler("ZYX", global_rot_rotmat)  # B x 3
        ### Looking at the original code, global trans is zeros.
        global_trans = torch.zeros_like(global_rot_euler)

        ## Next, get body pose.
        ### Hold onto raw, continuous version for iterative correction.
        pred_pose_cont = pred[:, count : count + self.body_cont_dim]
        count += self.body_cont_dim
        ### Convert to eulers (and trans)
        pred_pose_euler = compact_cont_to_model_params_body(pred_pose_cont)
        ### Zero-out hands
        pred_pose_euler[:, atlas46_param_hand_mask] = 0
        ### Zero-out jaw
        pred_pose_euler[:, -3:] = 0

        ## Get remaining parameters
        pred_shape = pred[:, count : count + self.num_shape_comps]
        count += self.num_shape_comps
        pred_scale = pred[:, count : count + self.num_scale_comps]
        count += self.num_scale_comps
        pred_hand = pred[:, count : count + self.num_hand_comps * 2]
        count += self.num_hand_comps * 2
        pred_face = pred[:, count : count + self.num_face_comps]
        count += self.num_face_comps

        if self.zero_face:
            pred_face = pred_face * 0

        # Run everything through atlas
        output = self.atlas(
            global_trans=global_trans,  # global_trans==0
            global_rot=global_rot_euler,
            body_pose_params=pred_pose_euler,
            hand_pose_params=pred_hand,
            scale_params=pred_scale,
            shape_params=pred_shape,
            expr_params=pred_face,
            do_pcblend=do_pcblend,
            return_keypoints=(self.mesh_type in ["lod3", "smpl"]),
            slim_keypoints=slim_keypoints
            and self.mesh_type == "lod3",  # only during training
        )


        # Some existing code to get joints and fix camera system
        if self.mesh_type == "lod3":
            verts, j3d = output
            j3d = j3d[:, :70]  # 308 --> 70 keypoints
        else:
            raise ValueError

        if verts is not None:
            verts[..., [1, 2]] *= -1  # Camera system difference
        j3d[..., [1, 2]] *= -1  # Camera system difference

        # Prep outputs
        output = {
            "pred_pose_raw": torch.cat(
                [global_rot_6d, pred_pose_cont], dim=1
            ),  # Both global rot and continuous pose
            "pred_pose_rotmat": None,  # This normally used for atlas pose param rotmat supervision.
            "global_rot": global_rot_euler,
            "body_pose": pred_pose_euler,  # Unused during training
            "shape": pred_shape,
            "scale": pred_scale,
            "hand": pred_hand,
            "face": pred_face,
            "pred_keypoints_3d": j3d.reshape(batch_size, -1, 3),
            "pred_vertices": (
                verts.reshape(batch_size, -1, 3) if verts is not None else None
            ),
            "faces": self.atlas.faces.cpu().numpy(),
        }

        return output
