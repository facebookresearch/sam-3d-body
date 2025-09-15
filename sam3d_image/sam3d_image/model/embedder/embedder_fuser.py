import math
import torch
from loguru import logger
from torch import nn
from typing import Optional, Tuple, List, Literal, Dict
from .ff import FeedForward


class EmbedderFuser(torch.nn.Module):
    """
    Fusing individual condition embedder. Require kwargs for the forward!
    Args:
        embedder_list: List of Tuples. Each Tuple consists of a condition_embedder
            and a list of tuple. In the list, each tuple consists of a string, indicating
            a kward, and astring, indicating the group of positional encoding to be used.
        use_pos_embedding: whether to add positional embedding. If add, follow the index in
            embedder_list. Choices of None (no pos emb), random, and learned.
        projection_pre_norm: pre-normalize features before feeding into projector layers.
        projection_net_hidden_dim_multiplier: hidden dimension for projection layer. If 0, don't use.
    """

    def __init__(
        self,
        embedder_list: List[Tuple[nn.Module, List[Tuple[str, Optional[str]]]]],
        use_pos_embedding: Optional[Literal["random", "learned"]] = "learned",
        projection_pre_norm: bool = True,
        projection_net_hidden_dim_multiplier: float = 4.0,
        compression_projection_multiplier: float = 0,
        freeze: bool = False,
        drop_modalities_weight: Dict[List[str], float] = None,
        dropout_prob: float = 0.0,
        force_drop_modalities: List[str] = None,
    ):
        super().__init__()
        self.embedder_list = embedder_list
        self.embed_dims = 0
        self.compression_projection_multiplier = compression_projection_multiplier
        self.concate_embed_dims = 0
        # keep moduleList to be compatible with nn module
        self.module_list = []
        max_positional_embed_idx = 0
        self.positional_embed_map = {}
        for condition_embedder, kwargs_info in self.embedder_list:
            self.embed_dims = max(self.embed_dims, condition_embedder.embed_dim)
            self.module_list.append(condition_embedder)
            for _, pos_group in kwargs_info:
                self.concate_embed_dims += condition_embedder.embed_dim
                if pos_group is not None:
                    if pos_group not in self.positional_embed_map:
                        self.positional_embed_map[pos_group] = max_positional_embed_idx
                        max_positional_embed_idx += 1
        self.module_list = nn.ModuleList(self.module_list)
        self.use_pos_embedding = use_pos_embedding
        if self.use_pos_embedding == "random":
            idx_emb = torch.randn(max_positional_embed_idx + 1, 1, self.embed_dims)
            self.register_buffer("idx_emb", idx_emb)
        elif self.use_pos_embedding == "learned":
            self.idx_emb = nn.Parameter(
                torch.empty(max_positional_embed_idx + 1, self.embed_dims)
            )
            nn.init.normal_(
                self.idx_emb, mean=0.0, std=1.0 / math.sqrt(self.embed_dims)
            )
        else:
            raise NotImplementedError(f"Unknown pos embedding {self.use_pos_embedding}")

        self.projection_pre_norm = projection_pre_norm
        self.projection_net_hidden_dim_multiplier = projection_net_hidden_dim_multiplier
        if projection_net_hidden_dim_multiplier > 0:
            self.projection_nets = []
            for condition_embedder, _ in self.embedder_list:
                self.projection_nets.append(
                    self._make_projection_net(
                        condition_embedder.embed_dim,
                        self.embed_dims,
                        self.projection_net_hidden_dim_multiplier,
                    )
                )
            self.projection_nets = nn.ModuleList(self.projection_nets)

        if compression_projection_multiplier > 0:
            self.compression_projector = self._make_projection_net(
                self.concate_embed_dims,
                self.embed_dims,
                self.compression_projection_multiplier,
            )

        self.drop_modalities_weight = drop_modalities_weight if drop_modalities_weight is not None else []
        self.dropout_prob = dropout_prob
        self.force_drop_modalities = force_drop_modalities

        if freeze:
            self.requires_grad_(False)
            self.eval()

    def _make_projection_net(
        self,
        input_embed_dim,
        output_embed_dim: int,
        multiplier: int,
    ):
        if self.projection_pre_norm:
            pre_norm = nn.LayerNorm(input_embed_dim)
        else:
            pre_norm = nn.Identity()

        # Per-token projection + gated activation
        ff_net = FeedForward(
            dim=input_embed_dim,
            hidden_dim=int(multiplier * output_embed_dim),
            output_dim=output_embed_dim,
        )
        return nn.Sequential(pre_norm, ff_net)

    def _build_dropout_distribution(self, device):
        """
        Build the probability distribution for dropout configurations.
        
        Returns:
            dropout_configs: List of sets containing modalities to drop
            cumsum_weights: Cumulative sum of weights for sampling
        """
        dropout_configs = []
        weights = []
        
        # Add no-dropout configuration with remaining probability
        dropout_configs.append(set())
        weights.append(1.0 - self.dropout_prob)
        
        # Add configured dropout patterns
        total_dropout_weight = sum(w for _, w in self.drop_modalities_weight)
        assert total_dropout_weight > 0, "Total dropout weight must be positive when drop_modalities_weight is provided"
        for modality_list, weight in self.drop_modalities_weight:
            dropout_configs.append(set(modality_list))
            # Scale weight by dropout_prob to ensure total probability sums to 1
            weights.append(self.dropout_prob * weight / total_dropout_weight)
        
        # Convert weights to cumulative distribution
        weights_tensor = torch.tensor(weights, device=device)
        
        was_deterministic = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(False)
        cumsum_weights = torch.cumsum(weights_tensor, dim=0)
        torch.use_deterministic_algorithms(was_deterministic)
        
        return dropout_configs, cumsum_weights
    
    def _apply_force_drop(self, kwarg_names: List[str], tokens: List[torch.Tensor]):
        if not self.force_drop_modalities:
            return tokens
        
        force_drop_set = set(self.force_drop_modalities)
        result_tokens = []
        
        for kwarg_name, token_tensor in zip(kwarg_names, tokens):
            # Create mask: 0 for forced drop, 1 otherwise
            mask = 0.0 if kwarg_name in force_drop_set else 1.0
            result_tokens.append(token_tensor * mask)
        
        return result_tokens
    
    def _dropout_modalities(self, kwarg_names: List[str], tokens: List[torch.Tensor]):
        # First apply forced drops (deterministic, always applied)
        tokens = self._apply_force_drop(kwarg_names, tokens)
        
        # Then apply probabilistic dropout (only in training)
        if not self.training or self.dropout_prob <= 0 or not self.drop_modalities_weight:
            return tokens
        
        batch_size = tokens[0].shape[0]
        device = tokens[0].device
        
        # Build dropout configurations and sample which to use per batch element
        dropout_configs, cumsum_weights = self._build_dropout_distribution(device)
        rand_vals = torch.rand(batch_size, device=device)
        # Clamp indices to valid range (handle edge case where rand_val == 1.0)
        config_indices = torch.searchsorted(cumsum_weights, rand_vals).clamp(max=len(dropout_configs) - 1)
        
        # Apply dropout masks with vectorized operations
        result_tokens = []
        for kwarg_name, token_tensor in zip(kwarg_names, tokens):
            # Start with all ones (no dropout)
            mask = torch.ones(batch_size, dtype=token_tensor.dtype, device=device)
            
            # Vectorized mask creation: check all configurations at once
            for config_idx, modalities_to_drop in enumerate(dropout_configs):
                if kwarg_name in modalities_to_drop:
                    # Set mask to 0 for all batch elements using this configuration
                    mask[config_indices == config_idx] = 0.0
            
            # Reshape mask to match token dimensions
            mask = mask.view([batch_size] + [1] * (token_tensor.ndim - 1))
            result_tokens.append(token_tensor * mask)
        
        return result_tokens

    def forward(self, *args, **kwargs):
        tokens = []
        kwarg_names = []
        
        for i, (condition_embedder, kwargs_info) in enumerate(self.embedder_list):
            # Ideally, we would batch the inputs; but that assumes same-sized inputs
            for kwarg_name, pos_group in kwargs_info:
                if kwarg_name not in kwargs:
                    logger.warning(f"{kwarg_name} not in kwargs to condition embedder!")
                input_cond = kwargs[kwarg_name]
                cond_token = condition_embedder(input_cond)
                if self.projection_net_hidden_dim_multiplier > 0:
                    cond_token = self.projection_nets[i](cond_token)
                if pos_group is not None:
                    pos_idx = self.positional_embed_map[pos_group]
                    if self.use_pos_embedding == "random":
                        cond_token += self.idx_emb[pos_idx : pos_idx + 1]
                    elif self.use_pos_embedding == "learned":
                        cond_token += self.idx_emb[pos_idx : pos_idx + 1, None]
                    else:
                        raise NotImplementedError(
                            f"Unknown pos embedding {self.use_pos_embedding}"
                        )
                tokens.append(cond_token)
                kwarg_names.append(kwarg_name)

        # Apply dropout modalities with preserved order
        tokens = self._dropout_modalities(kwarg_names, tokens)

        if self.compression_projection_multiplier > 0:
            tokens = torch.cat(tokens, dim=-1)
            tokens = self.compression_projector(tokens)
        else:
            tokens = torch.cat(tokens, dim=1)

        return tokens
