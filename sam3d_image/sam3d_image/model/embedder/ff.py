from torch import nn
import torch.nn.functional as F
from typing import Optional


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None,
        output_dim: Optional[int] = None,
        skip_w2: bool = False,
    ):
        """
        Llama3 FeedForward layer
            https://github.com/meta-llama/llama3/blob/a0940f9cf7065d45bb6675660f80d305c041a754/llama/model.py#L193
        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        if output_dim is None:
            output_dim = dim

        self.skip_w2 = skip_w2
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        if not self.skip_w2:
            self.w2 = nn.Linear(hidden_dim, output_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        x = F.silu(self.w1(x)) * self.w3(x)
        if self.skip_w2:
            return x
        return self.w2(x)
