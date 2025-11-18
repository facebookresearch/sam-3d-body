import os.path as osp
import pickle
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import cv2

def load_json(f):
    with open(f, "r") as ff:
        return json.load(ff)


def save_json(obj, f):
    with open(f, "w+") as ff:
        json.dump(obj, ff)

def load_pickle(f):
    with open(f, "rb") as ff:
        return pickle.load(ff)

def save_pickle(obj, f):
    with open(f, "wb+") as ff:
        pickle.dump(obj, ff)

def rotation_angle_difference(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute the angle difference (magnitude) between two batches of SO(3) rotation matrices.
    Args:
        A: Tensor of shape (*, 3, 3), batch of rotation matrices.
        B: Tensor of shape (*, 3, 3), batch of rotation matrices.
    Returns:
        Tensor of shape (*,), angle differences in radians.
    """
    # Compute relative rotation matrix
    R_rel = torch.matmul(A, B.transpose(-2, -1))  # (B, 3, 3)
    # Compute trace of relative rotation
    trace = R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2]  # (B,)
    # Compute angle using the trace formula
    cos_theta = (trace - 1) / 2
    # Clamp for numerical stability
    cos_theta_clamped = torch.clamp(cos_theta, -1.0, 1.0)
    # Compute angle difference
    angle = torch.acos(cos_theta_clamped)
    return angle
        
def batch6DFromXYZ(r, return_9D=False):
    """
    Generate a matrix representing a rotation defined by a XYZ-Euler
    rotation.

    Args:
        r: ... x 3 rotation vectors

    Returns:
        ... x 6
    """
    rc = torch.cos(r)
    rs = torch.sin(r)
    cx = rc[..., 0]
    cy = rc[..., 1]
    cz = rc[..., 2]
    sx = rs[..., 0]
    sy = rs[..., 1]
    sz = rs[..., 2]

    result = torch.empty(list(r.shape[:-1])+[3, 3], dtype=r.dtype).to(r.device)

    result[..., 0, 0] = cy * cz
    result[..., 0, 1] = -cx * sz + sx * sy * cz
    result[..., 0, 2] = sx * sz + cx * sy * cz
    result[..., 1, 0] = cy * sz
    result[..., 1, 1] = cx * cz + sx * sy * sz
    result[..., 1, 2] = -sx * cz + cx * sy * sz
    result[..., 2, 0] = -sy
    result[..., 2, 1] = sx * cy
    result[..., 2, 2] = cx * cy
    
    if not return_9D:
        return torch.cat([result[..., :, 0], result[..., :, 1]], dim=-1)
    else:
        return result
        
# https://github.com/papagina/RotationContinuity/blob/758b0ce551c06372cab7022d4c0bdf331c89c696/shapenet/code/tools.py#L82
def batchXYZfrom6D(poses):
    # Args: poses: ... x 6, where "6" is the combined first and second columns
    # First, get the rotaiton matrix
    x_raw = poses[..., :3]
    y_raw = poses[..., 3:]

    x = F.normalize(x_raw, dim=-1)
    z = torch.cross(x, y_raw, dim=-1)
    z = F.normalize(z, dim=-1)
    y = torch.cross(z, x, dim=-1)

    matrix = torch.stack([x, y, z], dim=-1) # ... x 3 x 3

    # Now get it into euler
    # https://github.com/papagina/RotationContinuity/blob/758b0ce551c06372cab7022d4c0bdf331c89c696/shapenet/code/tools.py#L412
    sy = torch.sqrt(matrix[..., 0, 0] * matrix[..., 0, 0] + matrix[..., 1, 0] * matrix[..., 1, 0])
    singular = sy < 1e-6
    singular = singular.float()
        
    x = torch.atan2(matrix[..., 2, 1], matrix[..., 2, 2])
    y = torch.atan2(-matrix[..., 2, 0], sy)
    z = torch.atan2(matrix[..., 1, 0],matrix[..., 0, 0])
    
    xs = torch.atan2(-matrix[..., 1, 2], matrix[..., 1, 1])
    ys = torch.atan2(-matrix[..., 2, 0], sy)
    zs = matrix[..., 1, 0] * 0
        
    out_euler = torch.zeros_like(matrix[..., 0])
    out_euler[..., 0] = x * (1 - singular) + xs * singular
    out_euler[..., 1] = y * (1 - singular) + ys * singular
    out_euler[..., 2] = z * (1 - singular) + zs * singular
    
    return out_euler
    
def get_pose_feats_6d_batched_from_joint_params(joint_params):
    joint_euler_angles = joint_params.reshape(-1, 161, 7)[:, 2:, 3:6] # B x 159 x 3
    joint_6d_feat = batch6DFromXYZ(joint_euler_angles)
    joint_6d_feat[:, :, 0] -= 1 # so all 0 when no rotation.
    joint_6d_feat[:, :, 4] -= 1 # so all 0 when no rotation.
    joint_6d_feat = joint_6d_feat.flatten(1, 2)
    return joint_6d_feat

class SparseLinear(nn.Module):
    def __init__(self, in_channels, out_channels, sparse_mask, bias=True):
        super().__init__()
        self.in_channels = in_channels; self.out_channels = out_channels
        
        self.sparse_indices = nn.Parameter(sparse_mask.nonzero().T, requires_grad=False) # 2 x K
        self.sparse_shape = sparse_mask.shape
        
        weight = torch.zeros(out_channels, in_channels)
        if bias:
            self.bias = torch.zeros(out_channels)
        else:
            self.bias = None

        # Initialize!
        for out_idx in range(out_channels):
            # By default, self.weight is initialized with kaiming,
            # fan_in, linear default.
            # Here, the entire thing (even stuff that should be 0) are initialized,
            # only relevant stuff will be kept
            fan_in = sparse_mask[out_idx].sum()
            gain = torch.nn.init.calculate_gain('leaky_relu', math.sqrt(5))
            std = gain / math.sqrt(fan_in)
            bound = math.sqrt(3.0) * std
            weight[out_idx].uniform_(-bound, bound)
            if self.bias is not None:
                bound = 1 / math.sqrt(fan_in)
                self.bias[out_idx:out_idx+1].uniform_(-bound, bound)
        self.sparse_weight = nn.Parameter(weight[self.sparse_indices[0], self.sparse_indices[1]])
        if self.bias is not None:
            self.bias = nn.Parameter(self.bias)
    
    def forward(self, x):
        curr_weight = torch.sparse_coo_tensor(self.sparse_indices, self.sparse_weight, self.sparse_shape)
        # Convert to CSR matrix to support half-precision matrix multiplication:
        # https://github.com/pytorch/pytorch/issues/41069#issuecomment-1006606202
        curr_weight = curr_weight.to_sparse_csr()
        if self.bias is None:
            return (curr_weight @ x.T).T
            # return torch.sparse.mm(curr_weight, x.T).T
        else:
            return (curr_weight @ x.T).T + self.bias
            # return torch.sparse.mm(curr_weight, x.T).T + self.bias

    def __repr__(self):
        return f"SparseLinear(in_channels={self.in_channels}, out_channels={self.out_channels}, bias={self.bias is not None})"

class PosePriorVAE(nn.Module):
    def __init__(self, input_dim=198, feature_dim=512, latent_dim=32, dropout_prob=0.2, eps=1e-6):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, feature_dim, bias=False),
            nn.BatchNorm1d(feature_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(feature_dim, feature_dim, bias=False),
            nn.BatchNorm1d(feature_dim),
            nn.LeakyReLU())
        
        self.mu = nn.Linear(feature_dim, latent_dim)
        self.logvar = nn.Linear(feature_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, feature_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(feature_dim, feature_dim),
            nn.LeakyReLU(),
            nn.Linear(feature_dim, input_dim))

        self.eps = eps

    def latent_to_462(self, x, jaw_params=None):
        nonhand_468 = torch.LongTensor([9,10,11,15,16,17,21,22,23,27,28,29,33,34,35,39,40,41,45,46,47,51,52,53,57,58,59,63,64,65,69,70,71,75,76,77,81,82,83,87,88,89,93,94,95,99,100,101,105,106,107,111,112,113,117,118,119,123,124,125,129,130,131,135,136,137,141,142,143,147,148,149,153,154,155,159,160,161,165,166,167,171,172,173,177,178,179,183,184,185,189,190,191,195,196,197,465,466,467])
        nonhand_462 = nonhand_468 - 6
        sampled_poses = self.decoder(x)
        sampled_poses_euler = batchXYZfrom6D(sampled_poses.reshape(-1, 33, 6)).reshape(-1, 99)
        res = torch.zeros(len(x), 462).to(x.device)
        res[:, nonhand_462] = sampled_poses_euler
        if jaw_params is not None:
            res[:, [459, 460, 461]] = jaw_params
        return res

class BodyPoseGMMPrior(nn.Module):
    def __init__(self, model_data_dir, gmm_comps=32):
        super().__init__()

        assert gmm_comps in [8, 16, 32]
        (self.model_mu, self.model_var, self.model_pi, self.model_precision, self.model_logdet, self.model_logweights) = [
            nn.Parameter(tmp, requires_grad=False) 
            for tmp in load_pickle(osp.join(model_data_dir, "full_body_gmm_prior.pkl"))[gmm_comps]]
        
        self.log_2pi = self.model_mu.shape[2] * np.log(2. * math.pi)

    def forward(self, pose_params):
        sub_x_mu = pose_params[:, None, :] - self.model_mu #[N, sub_K, D]
        sub_x_mu_T_precision = (sub_x_mu.transpose(0, 1) @ self.model_precision).transpose(0, 2)
        sub_x_mu_T_precision_x_mu = (sub_x_mu_T_precision.squeeze(2) * sub_x_mu).sum(dim=2, keepdim=True) #[N, sub_K, 1]
        log_prob = sub_x_mu_T_precision_x_mu
        log_prob = log_prob + self.log_2pi
        log_prob = log_prob - self.model_logdet
        log_prob = log_prob * -0.5
        log_prob = log_prob + self.model_logweights
        log_prob = log_prob.squeeze(2)
        log_prob = log_prob.amax(dim=1)

        return -log_prob
        
class BodyPosePerJointGMMPrior(nn.Module):
    def __init__(self, model_data_dir, num_comps=4):
        super().__init__()
        
        assert num_comps in [1, 2, 4, 8]
        (self.model_mu, self.model_var, self.model_pi, self.model_precision, self.model_logdet, self.model_logweights) = \
        [
            nn.Parameter(torch.stack(tmp, dim=1), requires_grad=False) 
            for tmp in list(zip(*load_pickle(osp.join(model_data_dir, "per_joint_gmm_prior.pkl"))[num_comps]))
        ]

        self.log_2pi = self.model_mu.shape[3] * np.log(2. * math.pi)

    def forward(self, pose_params):
        pose_params = pose_params.reshape(-1, 33, 6)
        sub_x_mu = pose_params[:, :, None, :] - self.model_mu #[N, 33, sub_K, D]
        sub_x_mu_T_precision = torch.einsum('afbc,dfbce->afbde', sub_x_mu[:, :, :, :], self.model_precision[:, :, :, :, :])
        sub_x_mu_T_precision_x_mu = (sub_x_mu_T_precision.squeeze(3) * sub_x_mu).sum(dim=3, keepdim=True) # N x 33 x sub_K x 1
        log_prob = sub_x_mu_T_precision_x_mu
        log_prob = log_prob + self.log_2pi
        log_prob = log_prob - self.model_logdet.permute(1, 0, 2) # 33 x sub_K x 1
        log_prob = log_prob * -0.5
        log_prob = log_prob + self.model_logweights
        log_prob = log_prob.squeeze(3)
        # print(log_prob.argmax(dim=2))
        log_prob = log_prob.amax(dim=2)

        return -log_prob

def resize_image(image_array, scale_factor, interpolation=cv2.INTER_LINEAR):
    new_height = int(image_array.shape[0] // scale_factor)
    new_width = int(image_array.shape[1] // scale_factor)
    resized_image = cv2.resize(image_array, (new_width, new_height), interpolation=interpolation)
    
    return resized_image



def compact_cont_to_model_params_hand(hand_cont):
    # These are ordered by joint, not model params ^^
    assert hand_cont.shape[-1] == 54
    hand_dofs_in_order = torch.tensor([3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 2, 3, 1, 1])
    assert sum(hand_dofs_in_order) == 27
    # Mask of 3DoFs into hand_cont
    mask_cont_threedofs = torch.cat([torch.ones(2 * k).bool() * (k in [3]) for k in hand_dofs_in_order])
    # Mask of 1DoFs (including 2DoF) into hand_cont
    mask_cont_onedofs = torch.cat([torch.ones(2 * k).bool() * (k in [1, 2]) for k in hand_dofs_in_order])
    # Mask of 3DoFs into hand_model_params
    mask_model_params_threedofs = torch.cat([torch.ones(k).bool() * (k in [3]) for k in hand_dofs_in_order])
    # Mask of 1DoFs (including 2DoF) into hand_model_params
    mask_model_params_onedofs = torch.cat([torch.ones(k).bool() * (k in [1, 2]) for k in hand_dofs_in_order])

    # Convert hand_cont to eulers
    ## First for 3DoFs
    hand_cont_threedofs = hand_cont[..., mask_cont_threedofs].unflatten(-1, (-1, 6))
    hand_model_params_threedofs = batchXYZfrom6D(hand_cont_threedofs).flatten(-2, -1)
    ## Next for 1DoFs
    hand_cont_onedofs = hand_cont[..., mask_cont_onedofs].unflatten(-1, (-1, 2)) #(sincos)
    hand_model_params_onedofs = torch.atan2(hand_cont_onedofs[..., -2], hand_cont_onedofs[..., -1])

    # Finally, assemble into a 27-dim vector, ordered by joint, then XYZ.
    hand_model_params = torch.zeros(*hand_cont.shape[:-1], 27).to(hand_cont)
    hand_model_params[..., mask_model_params_threedofs] = hand_model_params_threedofs
    hand_model_params[..., mask_model_params_onedofs] = hand_model_params_onedofs

    return hand_model_params

def compact_model_params_to_cont_hand(hand_model_params):
    # These are ordered by joint, not model params ^^
    assert hand_model_params.shape[-1] == 27
    hand_dofs_in_order = torch.tensor([3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 2, 3, 1, 1])
    assert sum(hand_dofs_in_order) == 27
    # Mask of 3DoFs into hand_cont
    mask_cont_threedofs = torch.cat([torch.ones(2 * k).bool() * (k in [3]) for k in hand_dofs_in_order])
    # Mask of 1DoFs (including 2DoF) into hand_cont
    mask_cont_onedofs = torch.cat([torch.ones(2 * k).bool() * (k in [1, 2]) for k in hand_dofs_in_order])
    # Mask of 3DoFs into hand_model_params
    mask_model_params_threedofs = torch.cat([torch.ones(k).bool() * (k in [3]) for k in hand_dofs_in_order])
    # Mask of 1DoFs (including 2DoF) into hand_model_params
    mask_model_params_onedofs = torch.cat([torch.ones(k).bool() * (k in [1, 2]) for k in hand_dofs_in_order])

    # Convert eulers to hand_cont hand_cont
    ## First for 3DoFs
    hand_model_params_threedofs = hand_model_params[..., mask_model_params_threedofs].unflatten(-1, (-1, 3))
    hand_cont_threedofs = batch6DFromXYZ(hand_model_params_threedofs).flatten(-2, -1)
    ## Next for 1DoFs
    hand_model_params_onedofs = hand_model_params[..., mask_model_params_onedofs]
    hand_cont_onedofs = torch.stack([hand_model_params_onedofs.sin(), hand_model_params_onedofs.cos()], dim=-1).flatten(-2, -1)

    # Finally, assemble into a 27-dim vector, ordered by joint, then XYZ.
    hand_cont = torch.zeros(*hand_model_params.shape[:-1], 54).to(hand_model_params)
    hand_cont[..., mask_cont_threedofs] = hand_cont_threedofs
    hand_cont[..., mask_cont_onedofs] = hand_cont_onedofs

    return hand_cont

def batch9Dfrom6D(poses):
    # Args: poses: ... x 6, where "6" is the combined first and second columns
    # First, get the rotaiton matrix
    x_raw = poses[..., :3]
    y_raw = poses[..., 3:]

    x = F.normalize(x_raw, dim=-1)
    z = torch.cross(x, y_raw, dim=-1)
    z = F.normalize(z, dim=-1)
    y = torch.cross(z, x, dim=-1)

    matrix = torch.stack([x, y, z], dim=-1).flatten(-2, -1) # ... x 3 x 3 -> x9

    return matrix

def batch4Dfrom2D(poses):
    # Args: poses: ... x 2, where "2" is sincos
    poses_norm = F.normalize(poses, dim=-1)

    poses_4d = torch.stack([
        poses_norm[..., 1],
        poses_norm[..., 0],
        -poses_norm[..., 0],
        poses_norm[..., 1],
    ], dim=-1) # Flattened SO2. Why am I doing this? I truly could not tell you.

    return poses_4d # .... x 4

def compact_cont_to_rotmat_body(body_pose_cont, inflate_trans=False):
    all_param_3dof_rot_idxs = torch.LongTensor([(0, 2, 4), (6, 8, 10), (12, 13, 14), (15, 16, 17), (18, 19, 20), (21, 22, 23), (24, 25, 26), (27, 28, 29), (34, 35, 36), (37, 38, 39), (44, 45, 46), (53, 54, 55), (64, 65, 66), (85, 69, 73), (86, 70, 79), (87, 71, 82), (88, 72, 76), (91, 92, 93), (112, 96, 100), (113, 97, 106), (114, 98, 109), (115, 99, 103), (130, 131, 132)])
    all_param_1dof_rot_idxs = torch.LongTensor([1, 3, 5, 7, 9, 11, 30, 31, 32, 33, 40, 41, 42, 43, 47, 48, 49, 50, 51, 52, 56, 57, 58, 59, 60, 61, 62, 63, 67, 68, 74, 75, 77, 78, 80, 81, 83, 84, 89, 90, 94, 95, 101, 102, 104, 105, 107, 108, 110, 111, 116, 117, 118, 119, 120, 121, 122, 123])
    all_param_1dof_trans_idxs = torch.LongTensor([124, 125, 126, 127, 128, 129])
    num_3dof_angles = len(all_param_3dof_rot_idxs) * 3
    num_1dof_angles = len(all_param_1dof_rot_idxs)
    num_1dof_trans = len(all_param_1dof_trans_idxs)
    assert body_pose_cont.shape[-1] == (2 * num_3dof_angles + 2 * num_1dof_angles + num_1dof_trans)
    # Get subsets
    body_cont_3dofs = body_pose_cont[..., :2*num_3dof_angles]
    body_cont_1dofs = body_pose_cont[..., 2*num_3dof_angles:2*num_3dof_angles+2*num_1dof_angles]
    body_cont_trans = body_pose_cont[..., 2*num_3dof_angles+2*num_1dof_angles:]
    # Convert conts to model params
    ## First for 3dofs
    body_cont_3dofs = body_cont_3dofs.unflatten(-1, (-1, 6))
    body_rotmat_3dofs = batch9Dfrom6D(body_cont_3dofs).flatten(-2, -1)
    ## Next for 1dofs
    body_cont_1dofs = body_cont_1dofs.unflatten(-1, (-1, 2)) #(sincos)
    body_rotmat_1dofs = batch4Dfrom2D(body_cont_1dofs).flatten(-2, -1)
    if inflate_trans:
        assert False, "This is left as a possibility to increase the space/contribution/supervision trans params gets compared to rots"
    else:
        ## Nothing to do for trans
        body_rotmat_trans = body_cont_trans
    # Put them together
    body_rotmat_params = torch.cat([
        body_rotmat_3dofs, body_rotmat_1dofs, body_rotmat_trans
    ], dim=-1)
    return body_rotmat_params

def compact_cont_to_model_params_body(body_pose_cont):
    all_param_3dof_rot_idxs = torch.LongTensor([(0, 2, 4), (6, 8, 10), (12, 13, 14), (15, 16, 17), (18, 19, 20), (21, 22, 23), (24, 25, 26), (27, 28, 29), (34, 35, 36), (37, 38, 39), (44, 45, 46), (53, 54, 55), (64, 65, 66), (85, 69, 73), (86, 70, 79), (87, 71, 82), (88, 72, 76), (91, 92, 93), (112, 96, 100), (113, 97, 106), (114, 98, 109), (115, 99, 103), (130, 131, 132)])
    all_param_1dof_rot_idxs = torch.LongTensor([1, 3, 5, 7, 9, 11, 30, 31, 32, 33, 40, 41, 42, 43, 47, 48, 49, 50, 51, 52, 56, 57, 58, 59, 60, 61, 62, 63, 67, 68, 74, 75, 77, 78, 80, 81, 83, 84, 89, 90, 94, 95, 101, 102, 104, 105, 107, 108, 110, 111, 116, 117, 118, 119, 120, 121, 122, 123])
    all_param_1dof_trans_idxs = torch.LongTensor([124, 125, 126, 127, 128, 129])
    num_3dof_angles = len(all_param_3dof_rot_idxs) * 3
    num_1dof_angles = len(all_param_1dof_rot_idxs)
    num_1dof_trans = len(all_param_1dof_trans_idxs)
    assert body_pose_cont.shape[-1] == (2 * num_3dof_angles + 2 * num_1dof_angles + num_1dof_trans)
    # Get subsets
    body_cont_3dofs = body_pose_cont[..., :2*num_3dof_angles]
    body_cont_1dofs = body_pose_cont[..., 2*num_3dof_angles:2*num_3dof_angles+2*num_1dof_angles]
    body_cont_trans = body_pose_cont[..., 2*num_3dof_angles+2*num_1dof_angles:]
    # Convert conts to model params
    ## First for 3dofs
    body_cont_3dofs = body_cont_3dofs.unflatten(-1, (-1, 6))
    body_params_3dofs = batchXYZfrom6D(body_cont_3dofs).flatten(-2, -1)
    ## Next for 1dofs
    body_cont_1dofs = body_cont_1dofs.unflatten(-1, (-1, 2)) #(sincos)
    body_params_1dofs = torch.atan2(body_cont_1dofs[..., -2], body_cont_1dofs[..., -1])
    ## Nothing to do for trans
    body_params_trans = body_cont_trans
    # Put them together
    body_pose_params = torch.zeros(*body_pose_cont.shape[:-1], 133).to(body_pose_cont)
    body_pose_params[..., all_param_3dof_rot_idxs.flatten()] = body_params_3dofs
    body_pose_params[..., all_param_1dof_rot_idxs] = body_params_1dofs
    body_pose_params[..., all_param_1dof_trans_idxs] = body_params_trans
    return body_pose_params

def compact_model_params_to_cont_body(body_pose_params):
    all_param_3dof_rot_idxs = torch.LongTensor([(0, 2, 4), (6, 8, 10), (12, 13, 14), (15, 16, 17), (18, 19, 20), (21, 22, 23), (24, 25, 26), (27, 28, 29), (34, 35, 36), (37, 38, 39), (44, 45, 46), (53, 54, 55), (64, 65, 66), (85, 69, 73), (86, 70, 79), (87, 71, 82), (88, 72, 76), (91, 92, 93), (112, 96, 100), (113, 97, 106), (114, 98, 109), (115, 99, 103), (130, 131, 132)])
    all_param_1dof_rot_idxs = torch.LongTensor([1, 3, 5, 7, 9, 11, 30, 31, 32, 33, 40, 41, 42, 43, 47, 48, 49, 50, 51, 52, 56, 57, 58, 59, 60, 61, 62, 63, 67, 68, 74, 75, 77, 78, 80, 81, 83, 84, 89, 90, 94, 95, 101, 102, 104, 105, 107, 108, 110, 111, 116, 117, 118, 119, 120, 121, 122, 123])
    all_param_1dof_trans_idxs = torch.LongTensor([124, 125, 126, 127, 128, 129])
    num_3dof_angles = len(all_param_3dof_rot_idxs) * 3
    num_1dof_angles = len(all_param_1dof_rot_idxs)
    num_1dof_trans = len(all_param_1dof_trans_idxs)
    assert body_pose_params.shape[-1] == (num_3dof_angles + num_1dof_angles + num_1dof_trans)
    # Take out params
    body_params_3dofs = body_pose_params[..., all_param_3dof_rot_idxs.flatten()]
    body_params_1dofs = body_pose_params[..., all_param_1dof_rot_idxs]
    body_params_trans = body_pose_params[..., all_param_1dof_trans_idxs]
    # params to cont
    body_cont_3dofs = batch6DFromXYZ(body_params_3dofs.unflatten(-1, (-1, 3))).flatten(-2, -1)
    body_cont_1dofs = torch.stack([body_params_1dofs.sin(), body_params_1dofs.cos()], dim=-1).flatten(-2, -1)
    body_cont_trans = body_params_trans
    # Put them together
    body_pose_cont = torch.cat([body_cont_3dofs, body_cont_1dofs, body_cont_trans], dim=-1)
    return body_pose_cont

atlas46_param_hand_idxs = [62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115]
atlas46_cont_hand_idxs = [72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237]
atlas46_param_hand_mask = torch.zeros(133).bool(); atlas46_param_hand_mask[atlas46_param_hand_idxs] = True
atlas46_cont_hand_mask = torch.zeros(260).bool(); atlas46_cont_hand_mask[atlas46_cont_hand_idxs] = True