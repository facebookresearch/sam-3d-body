# TODO: Sasha wrapper for the PoseTarget decoder
# TODO: Hao connect the function

import torch
import numpy as np
import open3d as o3d
import trimesh
from pytorch3d.structures import Meshes
from pytorch3d.transforms import quaternion_to_matrix, Transform3d, matrix_to_quaternion
from sam3d_image.data.dataset.tdfy.transforms_3d import compose_transform, decompose_transform
from sam3d_image.data.dataset.tdfy.pose_target import PoseTargetConverter
from sam3d_image.pipeline.layout_post_optimization_utils import (
    run_ICP,
    compute_iou,
    set_seed,
    apply_transform,
    get_mesh,
    get_mask_renderer,
    run_alignment,
    run_render_compare,
    check_occlusion,
)


SLAT_STD = torch.tensor(
    [
        2.377650737762451,
        2.386378288269043,
        2.124418020248413,
        2.1748552322387695,
        2.663944721221924,
        2.371192216873169,
        2.6217446327209473,
        2.684523105621338,
    ]
)
SLAT_MEAN = torch.tensor(
    [
        -2.1687545776367188,
        -0.004347046371549368,
        -0.13352349400520325,
        -0.08418072760105133,
        -0.5271206498146057,
        0.7238689064979553,
        -1.1414450407028198,
        1.2039363384246826,
    ]
)

ROTATION_6D_MEAN = torch.tensor(
    [
        -0.06366084883674913,
        0.008438224692279752,
        0.00017084786438302483,
        0.0007126610473540038,
        -0.0030916726538816417,
        0.5166093753457688,
    ]
)
ROTATION_6D_STD = torch.tensor(
    [
        0.6656971967514863,
        0.6787012271867754,
        0.30345010594844524,
        0.4394504420678794,
        0.39817973931717104,
        0.6176286868761914,
    ]
)


def layout_post_optimization(
    Mesh,
    Quaternion,
    Translation,
    Scale,
    Mask,
    Point_Map,
    Intrinsics,
    Enable_shape_ICP=True,
    Enable_rendering_optimization=True,
    min_size=512,
):

    set_seed(100)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # init transform and process mesh
    Rotation = quaternion_to_matrix(Quaternion.squeeze(1))
    center = Translation[0].clone()
    tfm_ori = compose_transform(scale=Scale, rotation=Rotation, translation=Translation)
    mesh, faces_idx, textures = get_mesh(Mesh, tfm_ori, device)

    # get mask and renderer
    mask, renderer = get_mask_renderer(Mask, min_size, Intrinsics, device)

    # check occlusion
    if check_occlusion(mask[0, 0].cpu().numpy(), Point_Map.cpu().numpy()):
        return (
            Quaternion,
            Translation,
            Scale,
            -1.0,
            False,
            False,
        )

    # Step 1: Manual Alignment
    source_points, target_points, center, tfm1, mesh, ori_iou, final_iou, flag_notgt = (
        run_alignment(
            Point_Map, mask, mesh, center, faces_idx, textures, renderer, device
        )
    )

    # return original layout if no target points. 
    if flag_notgt:
        return (
            Quaternion,
            Translation,
            Scale,
            -1.0,
            False,
            False,
        )

    # Step 2: Shape ICP
    if Enable_shape_ICP:
        Flag_ICP = True
        points_aligned_icp, transformation = run_ICP(
            mesh, source_points, target_points, threshold=0.05
        )
        mesh_ICP = Meshes(
            verts=[points_aligned_icp], faces=[faces_idx], textures=textures
        )
        rendered = renderer(mesh_ICP)
        ori_iou_shapeICP = compute_iou(
            rendered[..., 3][0][None, None], mask, threshold=0.5
        )
        # determine whether accept ICP
        if ori_iou_shapeICP > ori_iou:
            mesh = mesh_ICP
            final_iou = ori_iou_shapeICP.cpu().item()
            T_o3d = torch.tensor(transformation, dtype=torch.float32, device=device)
            T_o3d = T_o3d.T
            A = T_o3d[:3, :3]
            t = T_o3d[3, :3]
            scale = A.norm(dim=1)
            R = A / scale[:, None]
            center = ((center[None] * scale) @ R + t)[0]  # transform center
            tfm2 = (
                Transform3d(device=device)
                .scale(scale[None])
                .rotate(R[None])
                .translate(t[None])
            )
        else:
            Flag_ICP = False
            scale_2, translation_2 = torch.tensor(1).to(device), torch.zeros([3]).to(
                device
            )
            tfm2 = (
                Transform3d(device=device)
                .scale(scale_2.expand(3)[None])
                .translate(translation_2[None])
            )
    else:
        Flag_ICP = False
        scale_2, translation_2 = torch.tensor(1).to(device), torch.zeros([3]).to(device)
        tfm2 = (
            Transform3d(device=device)
            .scale(scale_2.expand(3)[None])
            .translate(translation_2[None])
        )

    # Step 3: Render-and-Compare
    if not Enable_rendering_optimization:
        Flag_optim = False
        tfm = tfm_ori.compose(tfm1).compose(tfm2)
    else:
        quat, translation, scale, R = run_render_compare(
            mesh, center, renderer, mask, device
        )
        with torch.no_grad():
            transformed = apply_transform(mesh, center, quat, translation, scale)
            rendered = renderer(transformed)
        optimized_iou = compute_iou(
            rendered[..., 3][0][None, None], mask, threshold=0.5
        )
        # Criterior to use layout optimization
        if optimized_iou < 0.5 or optimized_iou <= ori_iou:
            Flag_optim = False
            tfm = tfm_ori  # reject manual alignment and ICP as well.
            # tfm = tfm_ori.compose(tfm1).compose(tfm2)  # only reject render-compare but keep manual alignment and ICP.
        else:
            Flag_optim = True
            final_iou = optimized_iou.detach().cpu().item()
            tfm3 = (
                Transform3d(device=device)
                .translate(-center[None])  # move to center
                .scale(scale.expand(3)[None])
                .rotate(R.T[None])
                .translate(center[None])  # move back
                .translate(translation[None])
            )
            tfm = tfm_ori.compose(tfm1).compose(tfm2).compose(tfm3)

    M = tfm.get_matrix()[0]
    T_final = M[3, :3][None]
    A = M[:3, :3]
    scale_final = A.norm(dim=1)[None]
    R_final = A / scale_final[:, None]
    quat_final = matrix_to_quaternion(R_final)[None]

    return (
        quat_final,
        T_final,
        scale_final,
        round(float(final_iou), 4),
        Flag_ICP,
        Flag_optim,
    )


def pose_decoder(
    pose_target_convention,
):
    def decode(model_output_dict, scene_scale=None, scene_shift=None):
        x = model_output_dict

        # BEGIN: copied from generative.py
        key_mapping = {
            "shape": "x_shape_latent",
            "quaternion": "x_instance_rotation",
            "6drotation": "x_instance_rotation_6d",
            "6drotation_normalized": "x_instance_rotation_6d_normalized",
            "translation": "x_instance_translation",
            "scale": "x_instance_scale",
            "translation_scale": "x_translation_scale",
        }

        # Decodes for metrics
        pose_target_dict = {}
        for k, v in x.items():
            pose_target_dict[key_mapping.get(k, k)] = v

        # TODO: Hao & Bowen please do clean this up!
        # Convert 6D rotation to quaternion if needed
        if (
            "x_instance_rotation_6d" in pose_target_dict
            or "x_instance_rotation_6d_normalized" in pose_target_dict
        ):
            # Extract the two 3D vectors
            if "x_instance_rotation_6d_normalized" in pose_target_dict:
                rot_6d = pose_target_dict[
                    "x_instance_rotation_6d_normalized"
                ] * ROTATION_6D_STD.to(
                    pose_target_dict["x_instance_rotation_6d_normalized"].device
                ) + ROTATION_6D_MEAN.to(
                    pose_target_dict["x_instance_rotation_6d_normalized"].device
                )
            else:
                rot_6d = pose_target_dict["x_instance_rotation_6d"]
            a1 = rot_6d[..., 0:3]
            a2 = rot_6d[..., 3:6]

            # Normalize first vector
            b1 = torch.nn.functional.normalize(a1, dim=-1)

            # Make second vector orthogonal to first
            b2 = a2 - torch.sum(b1 * a2, dim=-1, keepdim=True) * b1
            b2 = torch.nn.functional.normalize(b2, dim=-1)

            # Compute third vector as cross product
            b3 = torch.cross(b1, b2, dim=-1)

            # Stack to create rotation matrix
            rotation_matrix = torch.stack([b1, b2, b3], dim=-1)

            # Convert to quaternion
            quaternion = matrix_to_quaternion(rotation_matrix)
            pose_target_dict["x_instance_rotation"] = quaternion

        if "x_instance_scale" in pose_target_dict:
            pose_target_dict["x_instance_scale"] = torch.exp(
                pose_target_dict["x_instance_scale"]
            )

        if "x_translation_scale" in pose_target_dict:
            pose_target_dict["x_translation_scale"] = torch.exp(
                pose_target_dict["x_translation_scale"]
            )

        pose_target_dict["pose_target_convention"] = [pose_target_convention] * x[
            "shape"
        ].shape[0]
        # END: copied from generative.py

        # Fake pointmap moments
        device = x["shape"].device
        _scene_scale = (
            scene_scale if scene_scale is not None else torch.tensor(1.0, device=device)
        )
        _scene_shift = (
            scene_shift
            if scene_shift is not None
            else torch.tensor([[0, 0, 1.0]], device=device)
        )
        pose_target_dict["x_scene_scale"] = _scene_scale
        pose_target_dict["x_scene_center"] = _scene_shift

        # Convert to instance pose
        pose_instance_dict = PoseTargetConverter.dicts_pose_target_to_instance_pose(
            pose_target_convention=pose_target_convention,
            x_instance_scale=pose_target_dict["x_instance_scale"],
            x_instance_translation=pose_target_dict["x_instance_translation"],
            x_instance_rotation=pose_target_dict["x_instance_rotation"],
            x_translation_scale=pose_target_dict["x_translation_scale"],
            x_scene_scale=pose_target_dict["x_scene_scale"],
            x_scene_center=pose_target_dict["x_scene_center"],
        )
        return {
            "translation": pose_instance_dict["instance_position_l2c"].squeeze(0),
            "rotation": pose_instance_dict["instance_quaternion_l2c"].squeeze(0),
            "scale": pose_instance_dict["instance_scale_l2c"].squeeze(0),
        }

    return decode


def get_default_pose_decoder():
    def decode(model_output_dict, **kwargs):
        return {}

    return decode


POSE_DECODERS = {
    "default": get_default_pose_decoder(),
    "ApparentSize": pose_decoder("ApparentSize"),
    "ScaleShiftInvariant": pose_decoder("ScaleShiftInvariant"),
}


def get_pose_decoder(name):
    if name not in POSE_DECODERS:
        raise NotImplementedError

    return POSE_DECODERS[name]


def prune_sparse_structure(
    coord_batch,
    max_neighbor_axes_dist=1,
):
    coords, batch = coord_batch[:, 1:], coord_batch[:, 0].unsqueeze(-1)
    device = coords.device
    # 1) shift coords so minimum is zero
    min_xyz = coords.min(0)[0]
    coords0 = coords - min_xyz
    # 2) build occupancy grid
    max_xyz = coords0.max(0)[0] + 1  # size in each dim
    D, H, W = max_xyz.tolist()
    # shape (1,1,D,H,W)
    occ = torch.zeros((1, 1, D, H, W), dtype=torch.uint8, device=device)
    x, y, z = coords0.unbind(1)
    occ[0, 0, x, y, z] = 1
    # 3) 3×3×3 convolution to count each voxel + neighbors
    kernel = torch.ones(
        (
            1,
            1,
            2 * max_neighbor_axes_dist + 1,
            2 * max_neighbor_axes_dist + 1,
            2 * max_neighbor_axes_dist + 1,
        ),
        dtype=torch.uint8,
        device=device,
    )
    # pad so output is same size
    pad = max_neighbor_axes_dist
    counts = torch.nn.functional.conv3d(occ.float(), kernel.float(), padding=pad)
    # interior voxels have count == (2*max_neighbor_axes_dist+1)**3
    full_count = (2 * max_neighbor_axes_dist + 1) ** 3
    # 4) lookup counts at each original coord
    counts_at_pts = counts[0, 0, x, y, z]  # (N,)
    is_surface = counts_at_pts < full_count
    # 5) return filtered batch+coords (shift back if you want original coords)
    kept = is_surface.nonzero(as_tuple=False).squeeze(1)
    out_batch = batch[kept]
    out_coords = coords[kept]
    coords = torch.cat([out_batch, out_coords], dim=1)

    return torch.cat([out_batch, out_coords], dim=1)


def downsample_sparse_structure(
    coord_batch,
    max_coords=42000,
    downsample_factor=2,
):
    """
    Downsample sparse structure coordinates when there are more than max_coords.

    Downsamples by rescaling coordinates, effectively shrinking the grid while preserving
    the structure. The downsampled grid is centered in the original space.

    Args:
        coord_batch: tensor of shape (N, 4) where [:, 0] is batch index and [:, 1:] are coords
        max_coords: maximum number of coordinates to keep
            42000 should be safe number. Calculation: max(int32) / (64*768) ~= 43691
            Only needed for mesh decoding.
        downsample_factor: factor by which to downsample (e.g., 2 means half resolution)

    Returns:
        Downsampled coord_batch with coordinates rescaled if downsampling is needed
    """
    if coord_batch.shape[0] <= max_coords:
        return coord_batch

    # Extract coordinates and batch indices
    coords = coord_batch[:, 1:].float()  # Shape: (N, 3), convert to float for scaling
    batch_indices = coord_batch[:, 0:1]  # Shape: (N, 1)

    # Find the actual coordinate bounds
    coords_min = coords.min(dim=0)[0]  # Shape: (3,)
    coords_max = coords.max(dim=0)[0]  # Shape: (3,)
    original_size = coords_max - coords_min + 1  # Add 1 since coordinates are discrete

    # Calculate target size after downsampling
    target_size = original_size / downsample_factor

    # Calculate the offset to center the downsampled grid
    offset = (original_size - target_size) / 2
    target_min = coords_min + offset
    target_max = coords_min + offset + target_size - 1

    # Normalize coordinates to [0, 1] within their actual range
    coords_normalized = (coords - coords_min) / (coords_max - coords_min)

    # Scale to the target range
    coords_rescaled = coords_normalized * (target_size - 1) + target_min

    # Round to integers to get discrete grid coordinates
    coords_rescaled = torch.round(coords_rescaled).int()

    # Clamp to ensure we stay within bounds
    coords_rescaled = torch.clamp(coords_rescaled, target_min.int(), target_max.int())

    # Remove duplicates that may have been created by the downsampling
    # Concatenate batch and coords for duplicate removal
    combined = torch.cat([batch_indices, coords_rescaled], dim=1)
    unique_combined = torch.unique(combined, dim=0)

    # If still too many after deduplication, randomly subsample
    if unique_combined.shape[0] > max_coords:
        indices = torch.randperm(unique_combined.shape[0], device=coord_batch.device)[
            :max_coords
        ]
        unique_combined = unique_combined[indices]

    return unique_combined.int()


def normalize_mesh_verts(verts):
    vmin = verts.min(axis=0)
    vmax = verts.max(axis=0)
    center = (vmax + vmin) / 2.0
    extent = vmax - vmin  # largest side length
    max_extent = np.max(extent)
    if max_extent == 0:
        vertices = verts - center
        scale = 1
    else:
        scale = 1.0 / max_extent
        vertices = (verts - center) * scale
    return vertices, scale, center


def voxelize_mesh(mesh, resolution=64):
    verts = np.asarray(mesh.vertices)
    # rotate mesh (from z-up to y-up)
    verts = verts @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).T
    # normalize vertices
    # skip vertices to avoid losing points, likely already normalized
    if np.abs(verts.min() + 0.5) < 1e-3 and np.abs(verts.max() - 0.5) < 1e-3:
        vertices, scale, center = verts, None, None
    else:
        vertices, scale, center = normalize_mesh_verts(verts)

    vertices = np.clip(vertices, -0.5 + 1e-6, 0.5 - 1e-6)
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        mesh,
        voxel_size=1 / 64,
        min_bound=(-0.5, -0.5, -0.5),
        max_bound=(0.5, 0.5, 0.5),
    )
    vertices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
    vertices = (vertices + 0.5) / 64 - 0.5
    coords = ((torch.tensor(vertices) + 0.5) * resolution).int().contiguous()
    ss = torch.zeros(1, resolution, resolution, resolution, dtype=torch.long)
    ss[:, coords[:, 0], coords[:, 1], coords[:, 2]] = 1
    return ss, scale, center


def preprocess_mesh(mesh: trimesh.Trimesh):
    verts = mesh.vertices
    if np.abs(verts.min() + 0.5) < 1e-3 and np.abs(verts.max() - 0.5) < 1e-3:
        return mesh
    vertices, _, _ = normalize_mesh_verts(verts)
    mesh.vertices = vertices
    return mesh


def trimesh2o3d_mesh(trimesh_mesh):
    verts = np.asarray(trimesh_mesh.vertices)
    faces = np.asarray(trimesh_mesh.faces)
    return o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(faces)
    )


def update_layout(pred_t, pred_s, pred_quat, center, scale, to_halo=True):
    if center is None and not to_halo:
        return pred_t, pred_s, pred_quat
    pred_transform = compose_transform(
        pred_s, quaternion_to_matrix(pred_quat[0]), pred_t
    )
    if center is None:
        comb_transform = pred_transform
    else:
        norm_transform = compose_transform(
            scale * torch.ones_like(pred_t),
            torch.eye(3, dtype=pred_t.dtype).to(pred_t.device)[None],
            scale * -torch.tensor(center, dtype=pred_t.dtype).to(pred_t.device)[None],
        )
        comb_transform = norm_transform.compose(pred_transform)
    comb_transform = convert_to_halo(comb_transform, pred_t.device, pred_t.dtype)
    decomposed = decompose_transform(comb_transform)
    quat = matrix_to_quaternion(decomposed.rotation)
    return decomposed.translation, decomposed.scale, quat


def convert_to_halo(pred_transform, device, dtype):
    on_mesh_transform = Transform3d(dtype=dtype, device=device).rotate(
        torch.tensor(
            [
                [1, 0, 0],
                [0, 0, 1],
                [0, -1, 0],
            ],
            dtype=dtype,
        )
    )
    on_pm_transform = Transform3d(dtype=dtype, device=device).rotate(
        torch.tensor(
            [
                [-1, 0, 0],
                [0, -1, 0],
                [0, 0, 1],
            ],
            dtype=dtype,
        )
    )
    return on_mesh_transform.compose(pred_transform).compose(on_pm_transform)


def quat_wxyz_to_euler_XYZ(q: torch.Tensor) -> torch.Tensor:
    """
    Convert PyTorch3D quaternions (w,x,y,z) to SciPy-style Euler angles
    with sequence 'XYZ' (extrinsic, radians). Works with batch dims.

    Args:
        q: (..., 4) tensor in w,x,y,z order. Doesn't need to be normalized.
    Returns:
        angles: (..., 3) tensor [alpha_X, beta_Y, gamma_Z] in radians.
    """
    q = q / q.norm(dim=-1, keepdim=True)  # normalize
    R = quaternion_to_matrix(q)  # (..., 3, 3)
    R = R.transpose(-1, -2)

    r00 = R[..., 0, 0]
    r10 = R[..., 1, 0]
    r20 = R[..., 2, 0]
    r21 = R[..., 2, 1]
    r22 = R[..., 2, 2]

    # For extrinsic XYZ (R = Rz(gamma) @ Ry(beta) @ Rx(alpha)):
    # beta = atan2(-r20, sqrt(r00^2 + r10^2))
    # alpha = atan2(r21, r22)
    # gamma = atan2(r10, r00)
    eps = torch.finfo(R.dtype).eps
    beta = torch.atan2(-r20, torch.clamp((r00 * r00 + r10 * r10).sqrt(), min=eps))
    alpha = torch.atan2(r21, r22)
    gamma = torch.atan2(r10, r00)

    return -torch.stack((alpha, beta, gamma), dim=-1)


def format_to_halo(layout_output):
    json_out = {}
    quaternion = layout_output["quaternion"][0, 0]
    translation = layout_output["translation"][0]
    scale = list(layout_output["scale"][0])

    euler = quat_wxyz_to_euler_XYZ(quaternion)
    json_out["roll"] = float(euler[0])
    json_out["pitch"] = float(euler[1])
    json_out["yaw"] = float(euler[2])
    json_out["pred_scale"] = [float(s) for s in scale]
    rot_matrix = quaternion_to_matrix(quaternion)
    pred_transform = torch.eye(4, dtype=quaternion.dtype).to(quaternion.device)
    pred_transform[:3, :3] = rot_matrix
    pred_transform[:3, 3] = translation
    pred_transform_list = [
        [float(t) for t in trans_row] for trans_row in pred_transform
    ]
    json_out["pred_transform"] = pred_transform_list
    return json_out


def json_to_halo_payloads(target_data):
    pred_transform = target_data["pred_transform"]
    pred_scale = target_data["pred_scale"]
    roll = target_data.get("roll", 0)
    pitch = target_data.get("pitch", 0)
    yaw = target_data.get("yaw", 0)
    # Update positions, rotation, and scale in the payload
    item_attachments = {}
    item_attachments["positions"] = {
        "x": pred_transform[0][3],
        "y": pred_transform[1][3],
        "z": pred_transform[2][3] - 1,  # Adjust for Halo design
    }
    item_attachments["rotation"] = {"x": roll, "y": pitch, "z": yaw}
    item_attachments["scale"] = {
        "x": pred_scale[0],
        "y": pred_scale[1],
        "z": pred_scale[2],
    }
    return item_attachments
