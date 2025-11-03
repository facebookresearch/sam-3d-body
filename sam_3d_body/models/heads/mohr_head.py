from typing import Optional

import roma

import os
import torch
import torch.nn as nn

from ..modules import rot6d_to_rotmat
from ..modules.atlas_utils import (
    atlas46_param_hand_mask,
    compact_cont_to_model_params_body,
    compact_model_params_to_cont_body,
    compact_cont_to_model_params_hand,
    load_pickle
)

from ..modules.transformer import FFN

class MoHRHead(nn.Module):

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
        use_torchscript=True,
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

        # Load scale & hand stuff for MoHR
        self.model_data_dir = atlas_model_path
        self.num_hand_scale_comps = num_scale_comps - 18
        self.num_hand_pose_comps = num_hand_comps
        model_dict = load_pickle(os.path.join(atlas_model_path, "params.pkl"))
        self.scale_mean = nn.Parameter(model_dict['scale_mean'], requires_grad=False)
        self.scale_comps = nn.Parameter(model_dict['scale_comps'][:18 + self.num_hand_scale_comps], requires_grad=False)
        self.faces = nn.Parameter(model_dict['faces']["lod1"], requires_grad=False)
        self.hand_pose_mean = nn.Parameter(model_dict['hand_prior_dict']['hand_pose_mean'], requires_grad=False)
        self.hand_pose_comps = nn.Parameter(model_dict['hand_prior_dict']['hand_pose_comps'][:self.num_hand_pose_comps], requires_grad=False)
        self.hand_joint_idxs_left = model_dict['hand_prior_dict']['hand_joint_idxs_left']
        self.hand_joint_idxs_right = model_dict['hand_prior_dict']['hand_joint_idxs_right']
        # Load Keypoint Mapping
        self.keypoint_mapping = nn.Parameter(model_dict['keypoint_mapping_dict']['keypoint_mapping'], requires_grad=False)
        self.general_expression_skeleton_kps_dict = model_dict['keypoint_mapping_dict']['general_expression_skeleton_kps_dict']
        self.keypoint_names_308 = model_dict['keypoint_mapping_dict']['keypoint_names_308']

        # Load MoHR itself
        self.use_torchscript = use_torchscript
        if self.use_torchscript:
            self.mohr = torch.jit.load(os.path.join(atlas_model_path, "mhr_ts.pt"), map_location=('cuda' if torch.cuda.is_available() else 'cpu'))
        else:
            from MHR.mhr.mhr import MHR
            self.mohr = MHR.from_files(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), lod=1)

        for param in self.mohr.parameters():
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

    def replace_hands_in_pose(self, full_pose_params, hand_pose_params):
        assert full_pose_params.shape[1] == 136

        # This drops in the hand poses from hand_pose_params (PCA 6D) into full_pose_params.
        # Split into left and right hands
        left_hand_params, right_hand_params = torch.split(
            hand_pose_params, [self.num_hand_pose_comps, self.num_hand_pose_comps], dim=1)

        # Change from cont to model params
        left_hand_params_model_params = compact_cont_to_model_params_hand(
            self.hand_pose_mean + torch.einsum('da,ab->db', left_hand_params, self.hand_pose_comps))
        right_hand_params_model_params = compact_cont_to_model_params_hand(
            self.hand_pose_mean + torch.einsum('da,ab->db', right_hand_params, self.hand_pose_comps))

        # Drop it in
        full_pose_params[:, self.hand_joint_idxs_left] = left_hand_params_model_params
        full_pose_params[:, self.hand_joint_idxs_right] = right_hand_params_model_params

        return full_pose_params # B x 207

    def mohr_forward(
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
        return_joint_params=False,
        scale_offsets=None,
        vertex_offsets=None):
        
        if body_pose_params.shape[-1] == 133:
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
        ## LBS works in cm (global_scale is [1, 1, 1]).
        full_pose_params = torch.cat([global_trans * 10, global_rot, body_pose_params], dim=1) # B x 127
        ## Put in hands
        if hand_pose_params is not None:
            full_pose_params = self.replace_hands_in_pose(full_pose_params, hand_pose_params)
        model_params = torch.cat([full_pose_params, scales], dim=1)

        if self.use_torchscript:
            curr_skinned_verts, joint_params, curr_skel_state = self.mohr(shape_params, model_params, expr_params)
        else:
            curr_skinned_verts = self.mohr(shape_params, model_params, expr_params)
            joint_params = self.mohr.character_torch.model_parameters_to_joint_parameters(model_params)
            curr_skel_state = self.mohr.character_torch.joint_parameters_to_skeleton_state(joint_params)
        curr_joint_coords, curr_joint_quats, _ = torch.split(curr_skel_state, [3, 4, 1], dim=2)
        curr_skinned_verts = curr_skinned_verts / 100
        curr_joint_coords = curr_joint_coords / 100
        curr_joint_rots = roma.unitquat_to_rotmat(curr_joint_quats)

        # Prepare returns
        to_return = [curr_skinned_verts]
        if return_keypoints:
            # Get sapiens 308 keypoints
            model_vert_joints = torch.cat([curr_skinned_verts, curr_joint_coords], dim=1) # B x (num_verts + 127) x 3
            model_keypoints_pred = (self.keypoint_mapping @ model_vert_joints.permute(1, 0, 2).flatten(1, 2)).reshape(-1, model_vert_joints.shape[0], 3).permute(1, 0, 2)
            to_return = to_return + [model_keypoints_pred]
        if return_joint_coords:
            to_return = to_return + [curr_joint_coords]
        if return_model_params:
            to_return = to_return + [model_params]
        if return_joint_rotations:
            to_return = to_return + [curr_joint_rots]
        if return_joint_params:
            to_return = to_return + [joint_params.unflatten(1, (-1, 7))]

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
            assert False, "Not Updated Yet"
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
        output = self.mohr_forward(
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
            "faces": self.faces.cpu().numpy(),
            "joint_global_rots": joint_global_rots,
            "joint_params": joint_params,
            "pred_global_wrist_rotmat": pred_global_wrist_rotmat,
            "pred_global_wrist_raw": pred_global_wrist_raw,
        }

        return output