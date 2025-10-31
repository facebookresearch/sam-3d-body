import torch
import torch.nn as nn
import os
import os.path as osp
import math
import numpy as np
import roma

from .atlas_utils import load_pickle, batch6DFromXYZ, batchXYZfrom6D, get_pose_feats_6d_batched_from_joint_params, SparseLinear, compact_cont_to_model_params_hand

class Proto(nn.Module):
    def __init__(self, 
                 model_data_dir,
                 num_hand_scale_comps=25,
                 num_hand_pose_comps=32,
                 lod="lod1",
                 load_keypoint_mapping=False,
                 verbose=False
                 ):
        super().__init__()

        assert lod == "lod1"
        assert load_keypoint_mapping

        self.model_data_dir = model_data_dir
        self.num_hand_scale_comps = num_hand_scale_comps
        self.num_hand_pose_comps = num_hand_pose_comps

        model_dict = load_pickle(osp.join(model_data_dir, "params.pkl"))
        
        self.scale_mean = nn.Parameter(model_dict['scale_mean'], requires_grad=False)
        self.scale_comps = nn.Parameter(model_dict['scale_comps'][:18 + num_hand_scale_comps], requires_grad=False)
        self.faces = nn.Parameter(model_dict['faces']["lod1"], requires_grad=False)
        self.hand_pose_mean = nn.Parameter(model_dict['hand_prior_dict']['hand_pose_mean'], requires_grad=False)
        self.hand_pose_comps = nn.Parameter(model_dict['hand_prior_dict']['hand_pose_comps'][:self.num_hand_pose_comps], requires_grad=False)
        self.hand_joint_idxs_left = model_dict['hand_prior_dict']['hand_joint_idxs_left']
        self.hand_joint_idxs_right = model_dict['hand_prior_dict']['hand_joint_idxs_right']

        # Load Keypoint Mapping
        self.keypoint_mapping = nn.Parameter(model_dict['keypoint_mapping_dict']['keypoint_mapping'], requires_grad=False)
        self.general_expression_skeleton_kps_dict = model_dict['keypoint_mapping_dict']['general_expression_skeleton_kps_dict']
        self.keypoint_names_308 = model_dict['keypoint_mapping_dict']['keypoint_names_308']

        # Load mhr model
        self.mhr_model = torch.jit.load("/private/home/jinhyun1/sam-3d-body/sandbox/mhr_ts.pt", map_location=('cuda' if torch.cuda.is_available() else 'cpu'))

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

    def forward(self,
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

        curr_skinned_verts, joint_params, curr_skel_state = self.mhr_model(shape_params, model_params, expr_params)
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