import os
import os.path as osp
import math

import numpy as np
import torch
import torch.nn as nn

import roma
import pymomentum.geometry as pym_geo

# from linear_blend_skinning_cuda import LinearBlendSkinningCuda

from .atlas_utils import load_pickle, batch6DFromXYZ, batchXYZfrom6D, get_pose_feats_6d_batched_from_joint_params, SparseLinear, compact_cont_to_model_params_hand

# changed
def get_pose_feats_6d_batched_from_joint_params_127(joint_params):
    joint_euler_angles = joint_params.reshape(-1, 127, 7)[:, 2:, 3:6] # B x 214 x 3
    joint_6d_feat = batch6DFromXYZ(joint_euler_angles)
    joint_6d_feat[:, :, 0] -= 1 # so all 0 when no rotation.
    joint_6d_feat[:, :, 4] -= 1 # so all 0 when no rotation.
    joint_6d_feat = joint_6d_feat.flatten(1, 2)
    return joint_6d_feat

class Proto(nn.Module):
    # num_hand_shape_comps: A new parameter indicating # of hand shape comps *per hand*.
    # These are merged directly into the regular shape comps, but at the end.
    def __init__(self,
                 model_data_dir,
                 num_body_shape_comps=20,
                 num_face_shape_comps=20,
                 num_hand_shape_comps=5,
                 num_hand_scale_comps=25,
                 num_hand_pose_comps=32,
                 lod="lod1",
                 load_keypoint_mapping=False,
                 fix_kps_eye_and_chin=False,
                 verbose=False):
        super().__init__()

        # Set Class Fields
        self.model_data_dir = model_data_dir
        
        self.num_body_shape_comps = num_body_shape_comps
        self.num_face_shape_comps = num_face_shape_comps
        self.num_hand_shape_comps = num_hand_shape_comps
        self.num_shape_comps = self.num_body_shape_comps + self.num_face_shape_comps + self.num_hand_shape_comps

        self.num_body_scale_comps = 18 # Fixed; independent mean/std
        self.num_hand_scale_comps = num_hand_scale_comps
        self.num_scale_comps = self.num_body_scale_comps + self.num_hand_scale_comps
        
        self.num_hand_pose_comps = num_hand_pose_comps
        
        self.num_expr_comps = 72 # Fixed
        
        self.lod = lod.lower(); assert self.lod in ['lod1']
        self.load_keypoint_mapping = load_keypoint_mapping
        if self.load_keypoint_mapping: assert self.lod == 'lod1', "308 Keypoints only supported for lod1"
        self.verbose = verbose

        # Load Model
        if self.verbose: print(f"Loading ATLAS from {model_data_dir} ...")
        model_dict = load_pickle(osp.join(model_data_dir, "params.pkl"))
        if self.verbose: print(f"Done loading pkl!")

        # Load Shape & Scale Bases
        if self.verbose: print("ATLAS is using {} shape components ({} body, {} face, {} hand), {} scale components ({} body, {} hand), {} pose PCA components per hand.".format(
            self.num_shape_comps,
            self.num_body_shape_comps,
            self.num_face_shape_comps,
            self.num_hand_shape_comps,
            self.num_scale_comps,
            self.num_body_scale_comps,
            self.num_hand_scale_comps,
            self.num_hand_pose_comps,
        ))
        self.shape_mean = nn.Parameter(model_dict['shape_mean'], requires_grad=False) # 18439 x 3
        self.shape_comps = nn.Parameter(torch.cat([
            model_dict['shape_comps'][:20][:self.num_body_shape_comps],
            model_dict['shape_comps'][20:40][:self.num_face_shape_comps],
            model_dict['shape_comps'][40:45][:self.num_hand_shape_comps],
        ], dim=0), requires_grad=False) # num_shape_comps x 18439 x 3
        
        self.scale_mean = nn.Parameter(model_dict['scale_mean'], requires_grad=False)
        assert self.num_scale_comps >= 18, "Should have at least 18 scale comps, since the body has 18 indep. comps. Aftewards, it's hand."
        self.scale_comps = nn.Parameter(torch.cat([
            model_dict['scale_comps'][:18][:self.num_body_scale_comps],
            model_dict['scale_comps'][18:18+25][:self.num_hand_scale_comps],
        ], dim=0), requires_grad=False)

        # Load Faces
        self.faces = nn.Parameter(model_dict['faces'][self.lod], requires_grad=False) # F x 3

        # Load Pose Correctives
        if self.verbose: print(f"Loading Pose Correctives...")
        self.posedirs = nn.Sequential(
            SparseLinear(125 * 6, 125 * 24, sparse_mask=model_dict['posedirs_sparse_mask'], bias=False),
            nn.ReLU(),
            nn.Linear(125 * 24, 18439 * 3, bias=False))
        self.posedirs.load_state_dict(model_dict['posedirs_state_dict'])
        for p in self.posedirs.parameters():
            p.requires_grad = False
        if self.lod != "lod1":
            lod1_to_other_weight = torch.sparse.FloatTensor(*load_pickle(osp.join(self.model_data_dir, "lod_mapping.pkl"))[self.lod])
            self.posedirs[2].weight = nn.Parameter(
                (lod1_to_other_weight @ self.posedirs[2].weight.unflatten(0, (-1, 3)).flatten(1, 2))\
                    .unflatten(1, (3, -1)).flatten(0, 1), 
                requires_grad=False)

        self.lbs_fn = pym_geo.Character.load_fbx(
            osp.join(self.model_data_dir, "trinity_basesewn_lod1_tris_v4.0.2.fbx"), 
            osp.join(self.model_data_dir, 'compact_v6_0.model'),
        )
        self.lbs_fn_infos = nn.ParameterDict({k: nn.Parameter(v.float(), requires_grad=False) for k, v in load_pickle(osp.join(self.model_data_dir, "lbs_fn_infos.pkl")).items()})

        # Load Hand PCA
        self._load_hand_prior()

        # Load Expressions
        self.exprdirs = nn.Parameter(model_dict['exprdirs'], requires_grad=False)

        # Load Keypoint Mapping
        if self.load_keypoint_mapping:
            assert self.lod in ["lod1", "smpl", "smplx"]
            if not fix_kps_eye_and_chin:
                assert self.lod in ["lod1"]
                self.keypoint_mapping = nn.Parameter(model_dict['keypoint_mapping_dict']['keypoint_mapping'], requires_grad=False)
            else:
                if self.lod in ["lod1", "smpl"]:
                    self.keypoint_mapping = nn.Parameter(
                        load_pickle(
                            osp.join(
                                self.model_data_dir,
                                f"{self.lod}_joint_to_kps_v4_fixEyeAndChin.pkl",
                            )
                        ),
                        requires_grad=False,
                    )
                elif self.lod == 'smplx':
                    # TODO: mapping lod3 vertices to smplx vertices, not a suboptimal way!
                    # self.lod3_to_smplx_verts_mapping = torch.sparse.FloatTensor(*load_pickle("/private/home/jinhyun1/codes/3po/atlas_250624/assets/lod_mapping_new.pkl")['lod3_to_smplx']).to_dense()
                    self.keypoint_mapping = nn.Parameter(
                        load_pickle(
                            osp.join(self.model_data_dir, "lod1_joint_to_kps_v4_fixEyeAndChin.pkl")
                        ),
                        requires_grad=False,
                    )
            self.general_expression_skeleton_kps_dict = model_dict['keypoint_mapping_dict']['general_expression_skeleton_kps_dict']
            self.keypoint_names_308 = model_dict['keypoint_mapping_dict']['keypoint_names_308']
                
            
        # Map from LOD5 to other LODs.
        if self.lod != "lod1":
            self._process_lod()

        # Load parameter limits
        self.model_param_limits = nn.Parameter(torch.FloatTensor(self.lbs_fn_infos['model_param_limits']), requires_grad=False)

        # Make life easier by having a mask of flexibles
        self.flexible_model_params_mask = nn.Parameter(
            (self.model_param_limits[:, 0] == 0) 
            & (self.model_param_limits[:, 1] == 0) 
            & (self.model_param_limits[:, 2] != 0), requires_grad=False)

    def _process_lod(self):
        if self.verbose: print(f"Converting to {self.lod}...")
        lod_mapping_dict = load_pickle(osp.join(self.model_data_dir, "lod_mapping.pkl"))
        mapping_matrix = torch.sparse.FloatTensor(*lod_mapping_dict[self.lod])

        # Straightforward/naive barycentric mapping from LOD5 to lower for model parameters
        self.shape_mean = nn.Parameter(mapping_matrix @ self.shape_mean, requires_grad=self.shape_mean.requires_grad)
        self.shape_comps = nn.Parameter((mapping_matrix @ self.shape_comps.permute(1, 0, 2).flatten(1, 2))\
                                        .reshape(-1, self.num_shape_comps, 3).permute(1, 0, 2), 
                                        requires_grad=self.shape_comps.requires_grad)
        self.exprdirs = nn.Parameter((mapping_matrix @ self.exprdirs.permute(1, 0, 2).flatten(1, 2))\
                                        .reshape(-1, self.num_expr_comps, 3).permute(1, 0, 2), 
                                        requires_grad=self.exprdirs.requires_grad)

    def _load_hand_prior(self):
        if self.verbose: print(f"Loading Hand Prior...")
        hand_prior_dict = load_pickle(osp.join(self.model_data_dir, "hand_pose_prior_compact.pkl"))

        self.hand_pose_mean = nn.Parameter(hand_prior_dict['hand_pose_mean'], requires_grad=False)
        self.hand_pose_comps = nn.Parameter(hand_prior_dict['hand_pose_comps'][:self.num_hand_pose_comps], requires_grad=False)
        self.hand_joint_idxs_left = hand_prior_dict['hand_joint_idxs_left']
        self.hand_joint_idxs_right = hand_prior_dict['hand_joint_idxs_right']

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
                mask_flexibles=False,
                mask_flexibles_pose_only=False,
                scale_offsets=None,
                vertex_offsets=None,
                slim_keypoints=False,
                return_joint_rotations=False,
                return_joint_params=False,
            ):

        assert not slim_keypoints
        if body_pose_params.shape[-1] == 133:
            body_pose_params = body_pose_params[..., :130]
            
        if os.environ.get('ZERO_HAND', "0") == "1":
            # This is an unbelievable hack
            print("ZEROING OUT HAND")
            # body_pose_params = body_pose_params.clone()
            # body_pose_params[..., [32, 33, 42, 43]] = torch.FloatTensor([[ 0.2134464 , -0.3877203 ,  0.17775373, -0.43506554]]).cuda()
            hand_pose_params = hand_pose_params.clone()
            if hand_pose_params.shape[-1] == 64:
                hand_pose_params[...] = torch.FloatTensor([ 0.60720223,  0.07500209, -0.03285376, -0.2053549 , -0.17148465,
                    0.26726273,  0.1794208 , -0.18497463, -0.15640062,  0.09794571,
                    0.24042967, -0.41130725, -0.12343965,  0.06465638, -0.5233267 ,
                    -0.9388142 ,  0.4494357 , -0.27890766,  0.17729089, -0.07480174,
                    -0.25141224, -0.01817356, -0.00400371, -0.01773551,  0.15311617,
                    -0.04983772,  0.19858299,  0.21404092,  0.15088876, -0.14906985,
                    0.12159939, -0.12288672,  0.58869326,  0.08022392, -0.18389095,
                    -0.29275241, -0.18641567,  0.24420524,  0.37931398, -0.30575007,
                    -0.25855404,  0.0782503 ,  0.4285042 , -0.46332794,  0.1848662 ,
                    0.09675991, -0.59010047, -0.90106446,  0.3820503 , -0.10754132,
                    0.24496357, -0.06562126, -0.38107422,  0.05865024, -0.03641961,
                    0.08158261,  0.01217111, -0.04469005,  0.35922152,  0.05598378,
                    0.14038268, -0.18285632,  0.1296904 , -0.24280865]).cuda()
            else:
                hand_pose_params[...] = (torch.FloatTensor([ 0.60720223,  0.07500209, -0.03285376, -0.2053549 , -0.17148465,
                    0.26726273,  0.1794208 , -0.18497463, -0.15640062,  0.09794571,
                    0.24042967, -0.41130725, -0.12343965,  0.06465638, -0.5233267 ,
                    -0.9388142 ,  0.4494357 , -0.27890766,  0.17729089, -0.07480174,
                    -0.25141224, -0.01817356, -0.00400371, -0.01773551,  0.15311617,
                    -0.04983772,  0.19858299,  0.21404092,  0.15088876, -0.14906985,
                    0.12159939, -0.12288672,  0.58869326,  0.08022392, -0.18389095,
                    -0.29275241, -0.18641567,  0.24420524,  0.37931398, -0.30575007,
                    -0.25855404,  0.0782503 ,  0.4285042 , -0.46332794,  0.1848662 ,
                    0.09675991, -0.59010047, -0.90106446,  0.3820503 , -0.10754132,
                    0.24496357, -0.06562126, -0.38107422,  0.05865024, -0.03641961,
                    0.08158261,  0.01217111, -0.04469005,  0.35922152,  0.05598378,
                    0.14038268, -0.18285632,  0.1296904 , -0.24280865]).cuda().reshape(2, 32) @ self.hand_pose_comps_ori).flatten()

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
        ## Convert shape...
        template_verts = self.shape_mean[None, :, :] + torch.einsum('da,abc->dbc', shape_params, self.shape_comps)
        if vertex_offsets is not None:
            template_verts = template_verts + vertex_offsets
        ## Put in expressions
        if expr_params is not None:
            expression_offsets = torch.einsum('cab,dc->dab', self.exprdirs, expr_params)
            template_verts = template_verts + expression_offsets

        # Now, figure out the pose.
        ## 10 here is because it's more stable to optimize global translation in meters.
        ## LBS works in cm (global_scale is [1, 1, 1]).
        full_pose_params = torch.cat([global_trans * 10, global_rot, body_pose_params], dim=1) # B x 127
        ## Put in hands
        if hand_pose_params is not None:
            full_pose_params = self.replace_hands_in_pose(full_pose_params, hand_pose_params)
        ## Get the 1512 joint params
        if scales.shape[0] == 1:
            scales = scales.squeeze(0)
        ## Optinally mask out flexibles
        if mask_flexibles and mask_flexibles_pose_only:
            full_pose_params = full_pose_params * ~self.flexible_model_params_mask[None, :136]
        model_params = self.lbs_fn.assemblePoseAndScale(full_pose_params, scales)
        ## Optinally mask out flexibles
        if mask_flexibles and not mask_flexibles_pose_only:
            model_params = model_params * ~self.flexible_model_params_mask[None, :]
        joint_params = self.lbs_fn.jointParamsFromModelParams(model_params)

        # Get pose correctives
        if do_pcblend:
            pose_6d_feats = get_pose_feats_6d_batched_from_joint_params_127(joint_params)
            pose_corrective_offsets = self.posedirs(pose_6d_feats).reshape(len(pose_6d_feats), -1, 3)
            template_verts = template_verts + pose_corrective_offsets

        # Finally, LBS
        curr_skinned_verts, curr_joint_coords, curr_joint_rotations = self.lbs_fn.forwardFromJointParams(joint_params, template_verts)[:3]
        curr_skinned_verts = curr_skinned_verts / 100
        curr_joint_coords = curr_joint_coords / 100

        # Prepare returns
        to_return = [curr_skinned_verts]
        if return_keypoints:
            # Get sapiens 308 keypoints
            assert self.load_keypoint_mapping
            model_vert_joints = torch.cat([curr_skinned_verts, curr_joint_coords], dim=1) # B x (num_verts + 127) x 3
            model_keypoints_pred = (self.keypoint_mapping @ model_vert_joints.permute(1, 0, 2).flatten(1, 2)).reshape(-1, model_vert_joints.shape[0], 3).permute(1, 0, 2)
            to_return = to_return + [model_keypoints_pred]
        if return_joint_coords:
            to_return = to_return + [curr_joint_coords]
        if return_model_params:
            to_return = to_return + [model_params]
        if return_joint_rotations:
            to_return = to_return + [curr_joint_rotations]
        if return_joint_params:
            to_return = to_return + [joint_params]

        if slim_keypoints:
            # vertices are nonsense
            to_return[0] = None

        if len(to_return) == 1: return to_return[0]
        else: return tuple(to_return)