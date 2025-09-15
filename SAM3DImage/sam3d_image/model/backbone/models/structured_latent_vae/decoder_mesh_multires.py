from typing import *
import torch
import torch.nn as nn
from ...modules.utils import zero_module, convert_module_to_f16
from ...modules import sparse as sp
from ...representations import MeshExtractResult
from .decoder_mesh import SLatMeshDecoder


def get_occupied_voxels(
    occupancy: sp.SparseTensor,
    sparse_tensor: sp.SparseTensor,
):
    """
    Prune the sparse_tensor by occupancy > 0

    Args:
        occupancy: SparseTensor with .coords [N, 4] and .feats [N, 1]
        sparse_tensor: SparseTensor with .coords [N, 4] and .feats [N, 1]

    Returns:
        filtered sparse_tensor
    """
    # Create mask for occupied voxels
    mask = occupancy.feats.squeeze(-1) > 0

    # TODO: @weiyaowang, handle empty selection cases

    # Filter coordinates
    occupied_coords = sparse_tensor.coords[mask]  # [M, 4]

    # Optionally get the occupancy values too
    occupied_feats = sparse_tensor.feats[mask]  # [M, dim]

    return sp.SparseTensor(
        coords=occupied_coords,
        feats=occupied_feats,
    )


def pack_keys(t, base=2048):
    """
    t: (N,4) int tensor with columns [B, X, Y, Z]
    returns: (N,) int64 unique key per row
    """
    t = t.long()  # ensure 64-bit ops
    # choose base >= max(value)+1; 2048 (2^11) works for 0..1024
    B, X, Y, Z = t.unbind(-1)
    # radix packing (no overflow with int64 for typical sizes)
    return ((B * base + X) * base + Y) * base + Z


def compute_selection(pred_coords, target_coords, hash_base=2048):
    """
    pred_coords: (N,4) int tensor [B,X,Y,Z]
    target_coords: (M,4) int tensor [B,X,Y,Z]
    Returns:
      mask: (N,) bool, True where pred_coords row exists in target_coords
    """
    pred_hashes = pack_keys(pred_coords, hash_base)
    target_hashes = pack_keys(target_coords, hash_base)
    matches = torch.isin(pred_hashes, target_hashes)
    return matches


def select_voxels(
    voxel_coords: List[torch.Tensor],
    sparse_tensor: sp.SparseTensor,
):
    """
    Prune the sparse_tensor by voxel_coords

    Args:
        voxel_coords: Voxels we want to select from in List[Tensor[N, 3]]
        sparse_tensor: SparseTensor with .coords [N, 4] and .feats [N, 1]

    Returns:
        selected sparse_tensor
    """
    # add batch dimension
    coords = []
    for i, coord in enumerate(voxel_coords):
        batch_ind = (
            torch.ones([coord.shape[0], 1], dtype=coord.dtype, device=coord.device) * i
        )
        coord = torch.concat([batch_ind, coord], dim=1)
        coords.append(coord.type(torch.int32))

    target_coords = torch.concat(coords, dim=0).to(sparse_tensor.coords.device)

    # coords_match = (sparse_tensor.coords.unsqueeze(1) == target_coords.unsqueeze(0)).all(dim=2).any(dim=1)
    coords_match = compute_selection(sparse_tensor.coords, target_coords)

    selected_coords = sparse_tensor.coords[coords_match]
    selected_feats = sparse_tensor.feats[coords_match]
    return sp.SparseTensor(
        coords=selected_coords,
        feats=selected_feats,
    )


class SLatMeshDecoderMultiRes(SLatMeshDecoder):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        occupancy_out_layers = []
        for i in range(self.num_upsample_layers):
            if i == 0:
                up_out_channels = self.model_channels // 4
            else:
                up_out_channels = self.model_channels // 8
            occupancy_out_layers.append(
                # occupancy head
                nn.Sequential(
                    sp.SparseGroupNorm32(32, up_out_channels),
                    sp.SparseSiLU(),
                    zero_module(sp.SparseLinear(up_out_channels, 1)),
                )
            )
        self.occupancy_out_layers = nn.ModuleList(occupancy_out_layers)
        if self.use_fp16:
            self.occupancy_out_layers.apply(convert_module_to_f16)

    def forward(
        self,
        x: sp.SparseTensor,
        *args,
        **kwargs,
    ) -> List[MeshExtractResult]:
        # call the transformer directly
        h = super(SLatMeshDecoder, self).forward(x)

        out_occupancy = {}
        gt_voxels = kwargs.get("mesh_voxels", {})
        for i, block in enumerate(self.upsample):
            h = block(h)
            occupancy = self.occupancy_out_layers[i](h)
            curr_resolution = block.out_resolution
            if int(curr_resolution) in gt_voxels:
                new_voxel = gt_voxels[int(curr_resolution)]
                h = select_voxels(new_voxel, h)
            else:
                h = get_occupied_voxels(occupancy, h)
            out_occupancy[int(curr_resolution)] = occupancy
            # print(f"h.coords.shape: {curr_resolution}, {h.coords.shape}")
        h = h.type(x.dtype)
        h = self.out_layer(h)
        return self.to_representation(h), out_occupancy
