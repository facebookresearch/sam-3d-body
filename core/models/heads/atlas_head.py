from typing import Optional

import roma

import torch
import torch.nn as nn

from ..modules import rot6d_to_rotmat
from ..modules.atlas46 import ATLAS46
from ..modules.proto import Proto
from ..modules.atlas_utils import (
    atlas46_param_hand_mask,
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
        num_face_comps: int = 72,
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
        zero_face_for_nonparam_losses: bool = False,
        detach_face_for_nonparam_losses: bool = False,
        pred_global_wrist_rot=False,
        replace_local_with_pred_global_wrist_rot=False,
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
        self.pred_global_wrist_rot = pred_global_wrist_rot
        self.replace_local_with_pred_global_wrist_rot = (
            replace_local_with_pred_global_wrist_rot
        )

        self.zero_face = zero_face
        self.zero_face_for_nonparam_losses = zero_face_for_nonparam_losses
        self.detach_face_for_nonparam_losses = detach_face_for_nonparam_losses
        assert (
            sum(
                [
                    self.zero_face,
                    self.zero_face_for_nonparam_losses,
                    self.detach_face_for_nonparam_losses,
                ]
            )
            <= 1
        )

        self.body_cont_dim = 260
        self.npose = (
            6  # Global Rotation
            + self.body_cont_dim  # then body
            + num_shape_comps
            + num_scale_comps
            + num_hand_comps * 2
            + num_face_comps
            + (0 if not self.pred_global_wrist_rot else 12)
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
        assert self.mesh_type in {"lod3"}
        self.atlas = Proto(
            atlas_model_path,
            # num_shape_comps,
            num_hand_scale_comps=num_scale_comps - 18,
            num_hand_pose_comps=num_hand_comps,
            # num_face_comps,
            lod=self.mesh_type if self.mesh_type != "lod3" else "lod1",
            load_keypoint_mapping=(self.mesh_type in ["lod3", "smpl"]),
            verbose=True,
            # fix_kps_eye_and_chin=fix_kps_eye_and_chin,
            # znorm_fullbody_scales=znorm_fullbody_scales,
            # num_hand_shape_comps=num_hand_shape_comps,
            # enable_slim_keypoint_mapping=enable_slim_keypoint_mapping,
        )
        for param in self.atlas.parameters():
            param.requires_grad = False


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
        if self.pred_global_wrist_rot:
            weights[:, -12:] = torch.FloatTensor([1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0])
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
        if self.pred_global_wrist_rot:
            pred_global_wrist_raw = pred[:, count : count + 12]
            pred_global_wrist_rotmat = rot6d_to_rotmat(
                pred_global_wrist_raw.unflatten(1, (-1, 6)).flatten(0, 1)
            ).unflatten(0, (-1, 2))
            count += 12
        else:
            pred_global_wrist_raw = None
            pred_global_wrist_rotmat = None

        if self.zero_face:
            pred_face = pred_face * 0

        pred_face_for_forward = pred_face.clone()
        if self.zero_face_for_nonparam_losses:
            pred_face_for_forward = pred_face_for_forward * 0
        if self.detach_face_for_nonparam_losses:
            pred_face_for_forward = pred_face_for_forward.detach()

        ################################################################################################################################################
        if self.replace_local_with_pred_global_wrist_rot:
            # First, forward just FK
            self.atlas.lbs_fn.fk_only = True
            joint_rotations = self.atlas(
                global_trans=global_trans,  # global_trans==0
                global_rot=global_rot_euler,
                body_pose_params=pred_pose_euler,  # Drop jaw
                hand_pose_params=pred_hand,
                scale_params=pred_scale,
                shape_params=pred_shape,
                expr_params=pred_face_for_forward,
                do_pcblend=do_pcblend,
            )[1]
            self.atlas.lbs_fn.fk_only = False

            # Get lowarm
            lowarm_joint_idxs = torch.LongTensor([76, 40]).cuda()  # left, right
            lowarm_joint_rotations = joint_rotations[
                :, lowarm_joint_idxs
            ]  # B x 2 x 3 x 3

            # Get zero-wrist pose
            wrist_twist_joint_idxs = torch.LongTensor([77, 41]).cuda()  # left, right
            wrist_zero_rot_pose = (
                lowarm_joint_rotations
                @ self.atlas.lbs_fn.joint_rotation[wrist_twist_joint_idxs]
            )

            # Now we want to get the local poses that lead to the wrist being pred_global_wrist_rotmat
            fused_local_wrist_rotmat = torch.einsum(
                "kabc,kabd->kadc", pred_global_wrist_rotmat, wrist_zero_rot_pose
            )
            wrist_xzy = roma.rotmat_to_euler("XZY", fused_local_wrist_rotmat).flatten(
                1, 2
            )

            # Put it in.
            pred_pose_euler = pred_pose_euler.clone()
            pred_pose_euler[:, [41, 43, 42, 31, 33, 32]] = wrist_xzy

        ################################################################################################################################################

        # Run everything through atlas
        output = self.atlas(
            global_trans=global_trans,
            global_rot=global_rot_euler,
            body_pose_params=pred_pose_euler,
            hand_pose_params=pred_hand,
            scale_params=pred_scale,
            shape_params=pred_shape,
            expr_params=pred_face_for_forward,
            do_pcblend=do_pcblend,
            return_keypoints=(self.mesh_type in ["lod3", "smpl"]),
            return_joint_coords=self.mesh_type == "lod3",
            return_joint_rotations=self.mesh_type == "lod3",
            return_joint_params=self.mesh_type == "lod3",
        )

        # Some existing code to get joints and fix camera system
        verts, j3d, jcoords, joint_global_rots, joint_params = output
        j3d = j3d[:, :70]  # 308 --> 70 keypoints

        if verts is not None:
            verts[..., [1, 2]] *= -1  # Camera system difference
        j3d[..., [1, 2]] *= -1  # Camera system difference
        if jcoords is not None:
            jcoords[..., [1, 2]] *= -1

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
            "pred_joint_coords": (
                jcoords.reshape(batch_size, -1, 3) if jcoords is not None else None
            ),
            "faces": self.atlas.faces.cpu().numpy(),
            "joint_global_rots": joint_global_rots,
            "joint_params": joint_params,
            "pred_global_wrist_rotmat": pred_global_wrist_rotmat,
            "pred_global_wrist_raw": pred_global_wrist_raw,
        }

        return output