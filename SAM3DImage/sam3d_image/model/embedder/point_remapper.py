import torch
import torch.nn as nn


class PointRemapper(nn.Module):
    """Handles remapping of 3D point coordinates and their inverse transformations."""

    VALID_TYPES = ["linear", "sinh", "exp", "sinh_exp"]

    def __init__(self, remap_type: str = "exp"):
        super().__init__()
        self.remap_type = remap_type

        if remap_type not in self.VALID_TYPES:
            raise ValueError(
                f"Invalid remap type: {remap_type}. Must be one of {self.VALID_TYPES}"
            )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """Apply remapping to point coordinates."""
        if self.remap_type == "linear":
            return points

        elif self.remap_type == "sinh":
            return torch.asinh(points)

        elif self.remap_type == "exp":
            xy_scaled, z_exp = points.split([2, 1], dim=-1)
            # Use log1p for better numerical stability near zero
            z = torch.log1p(z_exp)
            xy = xy_scaled / (1 + z_exp)
            return torch.cat([xy, z], dim=-1)

        elif self.remap_type == "sinh_exp":
            xy_sinh, z_exp = points.split([2, 1], dim=-1)
            xy = torch.asinh(xy_sinh)
            z = torch.log(z_exp.clamp(min=1e-8))
            return torch.cat([xy, z], dim=-1)

        else:
            raise ValueError(f"Unknown remap type: {self.remap_type}")

    def inverse(self, points: torch.Tensor) -> torch.Tensor:
        """Apply inverse remapping to recover original point coordinates."""
        if self.remap_type == "linear":
            return points

        elif self.remap_type == "sinh":
            return torch.sinh(points)

        elif self.remap_type == "exp":
            xy, z = points.split([2, 1], dim=-1)
            # Inverse of log1p is expm1(z) = exp(z) - 1
            z_exp = torch.expm1(z)
            # Inverse of xy/(1+z_exp) is xy*(1+z_exp)
            return torch.cat([xy * (1 + z_exp), z_exp], dim=-1)

        elif self.remap_type == "sinh_exp":
            xy, z = points.split([2, 1], dim=-1)
            return torch.cat([torch.sinh(xy), torch.exp(z)], dim=-1)

        else:
            raise ValueError(f"Unknown remap type: {self.remap_type}")

    def extra_repr(self) -> str:
        return f"remap_type='{self.remap_type}'"
