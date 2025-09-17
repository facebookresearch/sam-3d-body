import torch
import random
import optree


def _zeros_like(struct):
    def make_zeros(x):
        if isinstance(x, torch.Tensor):
            return torch.zeros_like(x)
        return x

    return optree.tree_map(make_zeros, struct, none_is_leaf=True)


def zero_out(args, kwargs):
    args = _zeros_like(args)
    kwargs = _zeros_like(kwargs)
    return args, kwargs


def discard(args, kwargs):
    return (), {}


def _drop_tensors(struct):
    """
    Drop any conditioning that are tensors
    Not using optree since we actually want to throw them instead of keeping them.
    """
    if isinstance(struct, dict):
        return {
            k: _drop_tensors(v)
            for k, v in struct.items()
            if not isinstance(v, torch.Tensor)
        }
    elif isinstance(struct, (list, tuple)):
        filtered = [_drop_tensors(x) for x in struct if not isinstance(x, torch.Tensor)]
        return tuple(filtered) if isinstance(struct, tuple) else filtered
    else:
        return struct


def drop_tensors(args, kwargs):
    args = _drop_tensors(args)
    kwargs = _drop_tensors(kwargs)
    return args, kwargs


def add_flag(args, kwargs):
    kwargs["cfg"] = True
    return args, kwargs


class ClassifierFreeGuidance(torch.nn.Module):
    UNCONDITIONAL_HANDLING_TYPES = {
        "zeros": zero_out,
        "discard": discard,
        "drop_tensors": drop_tensors,
        "add_flag": add_flag,
    }

    def __init__(
        self,
        backbone,  # backbone should be a backbone/generator (e.g. DDPM/DDIM/FlowMatching)
        p_unconditional=0.1,
        strength=3.0,
        # "zeros" = set cond tensors to 0,
        # "discard" = remove cond arguments and let underlying model handle it
        # "drop_tensors" = drop all tensors but leave non-tensors
        # "add_flag" = add an argument in kwargs as "cfg" and defer the handling to generator backbone
        unconditional_handling="zeros",
        interval=None,  # only perform cfg if t within interval
    ):
        super().__init__()

        if not (
            unconditional_handling
            in ClassifierFreeGuidance.UNCONDITIONAL_HANDLING_TYPES
        ):
            raise RuntimeError(
                f"'{unconditional_handling}' is not valid for `unconditional_handling`, should be in {ClassifierFreeGuidance.UNCONDITIONAL_HANDLING_TYPES}"
            )

        self.backbone = backbone
        self.p_unconditional = p_unconditional
        self.strength = strength
        self.unconditional_handling = unconditional_handling
        self.interval = interval
        self._make_unconditional_args = (
            ClassifierFreeGuidance.UNCONDITIONAL_HANDLING_TYPES[
                self.unconditional_handling
            ]
        )

    def _cfg_step_tensor(self, y_cond, y_uncond):
        return (1 + self.strength) * y_cond - self.strength * y_uncond

    def _cfg_step(self, y_cond, y_uncond):
        return optree.tree_map(self._cfg_step_tensor, y_cond, y_uncond)

    def forward(self, x, t, *args_cond, **kwargs_cond):
        # handle case when no conditional arguments are provided
        if len(args_cond) + len(kwargs_cond) == 0:  # unconditional
            if self.unconditional_handling != "discard":
                raise RuntimeError(
                    f"cannot call `ClassifierFreeGuidance` module without condition"
                )
            return self.backbone(x, t)
        else:  # conditional arguments are provided
            # training mode
            if self.training:
                coin_flip = random.random() < self.p_unconditional
                if coin_flip:  # unconditional
                    args_cond, kwargs_cond = self._make_unconditional_args(
                        args_cond,
                        kwargs_cond,
                    )
                return self.backbone(x, t, *args_cond, **kwargs_cond)
            else:  # inference mode
                y_cond = self.backbone(x, t, *args_cond, **kwargs_cond)
                in_interval = (self.interval is None) or (
                    self.interval[0] <= t <= self.interval[1]
                )
                if self.strength > 0.0 and in_interval:
                    args_cond, kwargs_cond = self._make_unconditional_args(
                        args_cond,
                        kwargs_cond,
                    )
                    y_uncond = self.backbone(x, t, *args_cond, **kwargs_cond)
                    return self._cfg_step(y_cond, y_uncond)
                return y_cond
