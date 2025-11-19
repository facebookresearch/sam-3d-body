# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import warnings
from typing import Optional

import roma
import torch
import torch.nn as nn

from ..modules import rot6d_to_rotmat
from ..modules.mhr_utils import (
    compact_cont_to_model_params_body,
    compact_cont_to_model_params_hand,
    compact_model_params_to_cont_body,
    mhr_param_hand_mask,
)

from ..modules.transformer import FFN

MOMENTUM_ENABLED = os.environ.get("MOMENTUM_ENABLED") is None
try:
    if MOMENTUM_ENABLED:
        from mhr.mhr import MHR

        MOMENTUM_ENABLED = True
        warnings.warn("Momentum is enabled")
    else:
        warnings.warn("Momentum is not enabled")
        raise ImportError
except:
    MOMENTUM_ENABLED = False
    warnings.warn("Momentum is not enabled")


class MHRHead(nn.Module):

    def __init__(
        self,
        input_dim: int,
        mlp_depth: int = 1,
        mhr_model_path: str = "",
        extra_joint_regressor: str = "",
        ffn_zero_bias: bool = True,
        mlp_channel_div_factor: int = 8,
        enable_hand_model=False,
    ):
        super().__init__()

        self.num_shape_comps = 45
        self.num_scale_comps = 28
        self.num_hand_comps = 54
        self.num_face_comps = 72
        self.enable_hand_model = enable_hand_model

        self.body_cont_dim = 260
        self.npose = (
            6  # Global Rotation
            + self.body_cont_dim  # then body
            + self.num_shape_comps
            + self.num_scale_comps
            + self.num_hand_comps * 2
            + self.num_face_comps
        )

        self.proj = FFN(
            embed_dims=input_dim,
            feedforward_channels=input_dim // mlp_channel_div_factor,
            output_dims=self.npose,
            num_fcs=mlp_depth,
            ffn_drop=0.0,
            add_identity=False,
        )

        if ffn_zero_bias:
            torch.nn.init.zeros_(self.proj.layers[-2].bias)

        # MHR Parameters
        self.model_data_dir = mhr_model_path
        self.num_hand_scale_comps = self.num_scale_comps - 18
        self.num_hand_pose_comps = self.num_hand_comps

        # Buffers to be filled in by model state dict
        self.joint_rotation = nn.Parameter(torch.zeros(127, 3, 3), requires_grad=False)
        self.scale_mean = nn.Parameter(torch.zeros(68), requires_grad=False)
        self.scale_comps = nn.Parameter(torch.zeros(28, 68), requires_grad=False)
        self.faces = nn.Parameter(torch.zeros(36874, 3).long(), requires_grad=False)
        self.hand_pose_mean = nn.Parameter(torch.zeros(54), requires_grad=False)
        self.hand_pose_comps = nn.Parameter(torch.eye(54), requires_grad=False)
        self.hand_joint_idxs_left = nn.Parameter(
            torch.zeros(27).long(), requires_grad=False
        )
        self.hand_joint_idxs_right = nn.Parameter(
            torch.zeros(27).long(), requires_grad=False
        )
        self.keypoint_mapping = nn.Parameter(
            torch.zeros(308, 18439 + 127), requires_grad=False
        )
        # Some special buffers for the hand-version
        self.right_wrist_coords = nn.Parameter(torch.zeros(3), requires_grad=False)
        self.root_coords = nn.Parameter(torch.zeros(3), requires_grad=False)
        self.local_to_world_wrist = nn.Parameter(torch.zeros(3, 3), requires_grad=False)
        self.nonhand_param_idxs = nn.Parameter(
            torch.zeros(145).long(), requires_grad=False
        )

        # Load MHR itself
        if MOMENTUM_ENABLED:
            self.mhr = MHR.from_files(
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                lod=1,
            )
        else:
            self.mhr = torch.jit.load(
                mhr_model_path,
                map_location=("cuda" if torch.cuda.is_available() else "cpu"),
            )

        for param in self.mhr.parameters():
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
        return weights

    def replace_hands_in_pose(self, full_pose_params, hand_pose_params):
        assert full_pose_params.shape[1] == 136

        # This drops in the hand poses from hand_pose_params (PCA 6D) into full_pose_params.
        # Split into left and right hands
        left_hand_params, right_hand_params = torch.split(
            hand_pose_params,
            [self.num_hand_pose_comps, self.num_hand_pose_comps],
            dim=1,
        )

        # Change from cont to model params
        left_hand_params_model_params = compact_cont_to_model_params_hand(
            self.hand_pose_mean
            + torch.einsum("da,ab->db", left_hand_params, self.hand_pose_comps)
        )
        right_hand_params_model_params = compact_cont_to_model_params_hand(
            self.hand_pose_mean
            + torch.einsum("da,ab->db", right_hand_params, self.hand_pose_comps)
        )

        # Drop it in
        full_pose_params[:, self.hand_joint_idxs_left] = left_hand_params_model_params
        full_pose_params[:, self.hand_joint_idxs_right] = right_hand_params_model_params

        return full_pose_params  # B x 207

    def mhr_forward(
        self,
        global_trans,
        global_rot,
        body_pose_params,
        hand_pose_params,
        scale_params,
        shape_params,
        expr_params=None,
        return_keypoints=False,
        do_pcblend=True,
        return_joint_coords=False,
        return_model_params=False,
        return_joint_rotations=False,
        scale_offsets=None,
        vertex_offsets=None,
    ):

        if self.enable_hand_model:
            # Transfer wrist-centric predictions to the body.
            global_rot_ori = global_rot.clone()
            global_trans_ori = global_trans.clone()
            global_rot = roma.rotmat_to_euler(
                "xyz",
                roma.euler_to_rotmat("xyz", global_rot_ori) @ self.local_to_world_wrist,
            )
            global_trans = (
                -(
                    roma.euler_to_rotmat("xyz", global_rot)
                    @ (self.right_wrist_coords - self.root_coords)
                    + self.root_coords
                )
                + global_trans_ori
            )

        body_pose_params = body_pose_params[..., :130]

        # Convert from scale and shape params to actual scales and vertices
        ## Add singleton batches in case...
        if len(scale_params.shape) == 1:
            scale_params = scale_params[None]
        if len(shape_params.shape) == 1:
            shape_params = shape_params[None]
        ## Convert scale...
        scales = self.scale_mean[None, :] + scale_params @ self.scale_comps
        if scale_offsets is not None:
            scales = scales + scale_offsets

        # Now, figure out the pose.
        ## 10 here is because it's more stable to optimize global translation in meters.
        full_pose_params = torch.cat(
            [global_trans * 10, global_rot, body_pose_params], dim=1
        )  # B x 127
        ## Put in hands
        if hand_pose_params is not None:
            full_pose_params = self.replace_hands_in_pose(
                full_pose_params, hand_pose_params
            )
        model_params = torch.cat([full_pose_params, scales], dim=1)

        if self.enable_hand_model:
            # Zero out non-hand parameters
            model_params[:, self.nonhand_param_idxs] = 0

        curr_skinned_verts, curr_skel_state = self.mhr(
            shape_params, model_params, expr_params
        )
        curr_joint_coords, curr_joint_quats, _ = torch.split(
            curr_skel_state, [3, 4, 1], dim=2
        )
        curr_skinned_verts = curr_skinned_verts / 100
        curr_joint_coords = curr_joint_coords / 100
        curr_joint_rots = roma.unitquat_to_rotmat(curr_joint_quats)

        # Prepare returns
        to_return = [curr_skinned_verts]
        if return_keypoints:
            # Get sapiens 308 keypoints
            model_vert_joints = torch.cat(
                [curr_skinned_verts, curr_joint_coords], dim=1
            )  # B x (num_verts + 127) x 3
            model_keypoints_pred = (
                (
                    self.keypoint_mapping
                    @ model_vert_joints.permute(1, 0, 2).flatten(1, 2)
                )
                .reshape(-1, model_vert_joints.shape[0], 3)
                .permute(1, 0, 2)
            )

            if self.enable_hand_model:
                # Zero out everything except for the right hand
                model_keypoints_pred[:, :21] = 0
                model_keypoints_pred[:, 42:] = 0

            to_return = to_return + [model_keypoints_pred]
        if return_joint_coords:
            to_return = to_return + [curr_joint_coords]
        if return_model_params:
            to_return = to_return + [model_params]
        if return_joint_rotations:
            to_return = to_return + [curr_joint_rots]

        if isinstance(to_return, list) and len(to_return) == 1:
            return to_return[0]
        else:
            return tuple(to_return)

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
        pred_pose_euler[:, mhr_param_hand_mask] = 0
        ### Zero-out jaw
        pred_pose_euler[:, -3:] = 0

        ## Get remaining parameters
        pred_shape = pred[:, count : count + self.num_shape_comps]
        count += self.num_shape_comps
        pred_scale = pred[:, count : count + self.num_scale_comps]
        count += self.num_scale_comps
        pred_hand = pred[:, count : count + self.num_hand_comps * 2]
        count += self.num_hand_comps * 2
        pred_face = pred[:, count : count + self.num_face_comps] * 0
        count += self.num_face_comps

        # Run everything through mhr
        output = self.mhr_forward(
            global_trans=global_trans,
            global_rot=global_rot_euler,
            body_pose_params=pred_pose_euler,
            hand_pose_params=pred_hand,
            scale_params=pred_scale,
            shape_params=pred_shape,
            expr_params=pred_face,
            do_pcblend=do_pcblend,
            return_keypoints=True,
            return_joint_coords=True,
            return_model_params=True,
            return_joint_rotations=True,
        )

        # Some existing code to get joints and fix camera system
        verts, j3d, jcoords, mhr_model_params, joint_global_rots = output
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
            "pred_pose_rotmat": None,  # This normally used for mhr pose param rotmat supervision.
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
            "faces": self.faces.cpu().numpy(),
            "joint_global_rots": joint_global_rots,
            "mhr_model_params": mhr_model_params,
        }

        return output
