from timm.models.vision_transformer import Block
import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from loguru import logger

from .point_remapper import PointRemapper


class PointPatchEmbed(nn.Module):
    """
    Projects (x,y,z) → D
    Splits into patches (patch_size x patch_size)
    Runs a tiny self-attention block inside each window
    Returns one token per window.
    """

    def __init__(
        self,
        input_size: int = 256,
        patch_size: int = 8,
        embed_dim: int = 768,
        remap_output: str = "exp",  # Add remap_output parameter
        dropout_prob: float = 0.0,  # Dropout probability for pointmap
        force_dropout_always: bool = False,  # Force dropout during validation/inference
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.dropout_prob = dropout_prob
        self.force_dropout_always = force_dropout_always

        # Point remapper
        self.point_remapper = PointRemapper(remap_output)

        # (1) point embedding
        self.point_proj = nn.Linear(3, embed_dim)
        self.invalid_xyz_token = nn.Parameter(torch.zeros(embed_dim))
        
        # Special embedding for dropped patches (used during dropout)
        # Alternative dropout strategies to consider:
        # 1. Drop all tokens entirely or use a single token only
        # 2. Different dropout patterns per window
        # 3. Use dropped_xyz_token/invalid_xyz_token per pixel
        if dropout_prob > 0:
            self.dropped_xyz_token = nn.Parameter(torch.zeros(embed_dim))

        # (2) positional embedding
        num_patches = input_size // patch_size
        # For patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, embed_dim, num_patches, num_patches)
        )
        # For points in a patch
        self.pos_embed_window = nn.Parameter(
            torch.zeros(1, 1 + patch_size * patch_size, embed_dim)
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # (3) within-patch transformer block(s)
        # From MCC: https://github.com/facebookresearch/MCC/blob/b04c97518360e4fdedfb6f090db7e90d0c2f8ae6/mcc_model.py#L97
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads=16,
                    mlp_ratio=2.0,
                    qkv_bias=True,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                )
            ]
        )
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize positional embeddings with small std
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.pos_embed_window, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.invalid_xyz_token, std=0.02)
        
        # Initialize dropped pointmap token if dropout is enabled
        if self.dropout_prob > 0:
            nn.init.normal_(self.dropped_xyz_token, std=0.02)

        # Initialize point projection with xavier uniform for better gradient flow
        # This is crucial since pointmaps can have large value ranges
        nn.init.xavier_uniform_(self.point_proj.weight, gain=0.02)
        if self.point_proj.bias is not None:
            nn.init.constant_(self.point_proj.bias, 0)

    def _get_pos_embed(self, hw):
        h, w = hw
        pos_embed = F.interpolate(
            self.pos_embed, size=(h, w), mode="bilinear", align_corners=False
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1)  # (B, H, W, C)
        return pos_embed

    def resize_input(self, xyz: torch.Tensor) -> torch.Tensor:
        resized_xyz = F.interpolate(xyz, size=self.input_size, mode="nearest")
        resized_xyz = resized_xyz.permute(0, 2, 3, 1)  # (B, H, W, C)
        return resized_xyz

    def apply_pointmap_dropout(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Apply dropout to pointmap embeddings.
        Drops entire pointmap for selected samples during training or when forced.
        
        When force_dropout_always is True, always drops pointmap regardless of training mode.
        """
        # Check if we should apply dropout
        should_apply_dropout = (self.training or self.force_dropout_always) and self.dropout_prob > 0
        
        if not should_apply_dropout:
            return embeddings
        
        # Check if dropout infrastructure exists
        if not hasattr(self, 'dropped_xyz_token'):
            if self.force_dropout_always:
                raise RuntimeError(
                    "Cannot force dropout: model was initialized with dropout_prob=0. "
                    "Re-initialize with dropout_prob > 0 to enable forced dropout."
                )
            return embeddings
            
        batch_size, n_windows, embed_dim = embeddings.shape
        
        # Decide dropout behavior
        if self.force_dropout_always and not self.training:
            # When forced during inference, always drop (100% dropout)
            drop_mask = torch.ones(batch_size, device=embeddings.device, dtype=torch.bool)
        else:
            # Normal training dropout - use configured probability
            drop_mask = torch.rand(batch_size, device=embeddings.device) < self.dropout_prob
        
        # Create dropped embedding for all windows - use same token for all patches
        # Shape: (batch_size, n_windows, embed_dim)
        dropped_embedding = self.dropped_xyz_token.view(1, 1, embed_dim).expand(batch_size, n_windows, embed_dim)
        
        # Add positional embeddings to dropped tokens (same as regular embeddings get)
        n_windows_h = n_windows_w = int(n_windows ** 0.5)
        pos_embed_patch = self._get_pos_embed((n_windows_h, n_windows_w)).reshape(
            1, n_windows, embed_dim
        )
        dropped_embedding = dropped_embedding + pos_embed_patch
        drop_mask_expanded = drop_mask.view(batch_size, 1, 1).expand_as(embeddings)
        embeddings = torch.where(drop_mask_expanded, dropped_embedding, embeddings)
        
        return embeddings

    def embed_pointmap_windows(
        self, xyz: torch.Tensor, valid_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Process pointmap into window embeddings without positional encoding"""
        with torch.no_grad():
            xyz = self.resize_input(xyz)
            if valid_mask is None:
                valid_mask = xyz.isfinite().all(dim=-1)

            B, H, W, _ = xyz.shape
            assert (
                H % self.patch_size == 0 and W % self.patch_size == 0
            ), "image must be divisible by patch_size"

            # (1) Handle NaN values before remapping to prevent propagation
            xyz_safe = xyz.clone()
            xyz_safe[~valid_mask] = 0.0  # Set invalid points to 0 before remapping

            # (1b) remap points to normalize their range
            xyz_remapped = self.point_remapper(xyz_safe)

        # (2) project + invalid token
        x = self.point_proj(xyz_remapped)  # (B,H,W,D)

        x[~valid_mask] = 0.0  # Stop gradient for invalid points
        x[~valid_mask] += self.invalid_xyz_token

        # (3) reshape into windows
        x = x.view(
            B,
            H // self.patch_size,
            self.patch_size,
            W // self.patch_size,
            self.patch_size,
            self.embed_dim,
        )  # (B, hW, wW, ws, ws, D)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # (B, hW, wW, ws, ws, D)
        x = x.view(-1, self.patch_size * self.patch_size, self.embed_dim)

        # (4) CLS token that contains the patch information
        cls_tok = self.cls_token.expand(x.shape[0], -1, -1)
        toks = torch.cat([cls_tok, x], dim=1)

        # (5) add positional embedding for window
        toks = toks + self.pos_embed_window

        # (6) intra-window attention
        for blk in self.blocks:
            toks = blk(toks)

        # (7) Extract CLS tokens and reshape to (B, n_windows, embed_dim)
        n_windows_h = H // self.patch_size
        n_windows_w = W // self.patch_size
        toks = toks[:, 0].view(B, n_windows_h * n_windows_w, self.embed_dim)
        
        return toks, (n_windows_h, n_windows_w)

    def forward(
        self, xyz: torch.Tensor, valid_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        xyz        : (B, 3, H, W) map of (x,y,z) coordinates
        valid_mask : (B, H, W) boolean - True for valid points (optional)

        returns: (B, num_windows, D)
        """
        # Get window embeddings
        window_embeddings, (n_windows_h, n_windows_w) = self.embed_pointmap_windows(xyz, valid_mask)
        
        # Add positional embeddings
        pos_embed_patch = self._get_pos_embed((n_windows_h, n_windows_w)).reshape(
            1, n_windows_h * n_windows_w, self.embed_dim
        )
        out = window_embeddings + pos_embed_patch
        
        # Apply dropout if enabled (during training OR when forced)
        if (self.training or self.force_dropout_always) and self.dropout_prob > 0:
            out = self.apply_pointmap_dropout(out)
        
        return out
