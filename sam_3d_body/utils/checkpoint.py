# Copyright (c) Meta Platforms, Inc. and affiliates.

from collections import namedtuple

import pytorch_lightning as pl
import torch

from .logging import get_pylogger

log = get_pylogger(__name__)


class CheckpointCallback(pl.callbacks.ModelCheckpoint):
    """Disable model checkpoint after validation to avoid DDP job hanging after resume"""

    def on_validation_end(self, trainer, pl_module):
        # Override to do nothing
        pass


class _IncompatibleKeys(
    namedtuple("IncompatibleKeys", ["missing_keys", "unexpected_keys"])
):

    def __repr__(self):
        if not self.missing_keys and not self.unexpected_keys:
            return "<All keys matched successfully>"
        return super().__repr__()

    __str__ = __repr__


def load_state_dict(module, state_dict, strict=False, logger=None):
    """Load state_dict to a module.

    This method is modified from :meth:`torch.nn.Module.load_state_dict`.
    Default value for ``strict`` is set to ``False`` and the message for
    param mismatch will be shown even if strict is False.

    Args:
        module (Module): Module that receives the state_dict.
        state_dict (OrderedDict): Weights.
        strict (bool): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Defaults to False.
        logger (:obj:`logging.Logger`, optional): Logger to log the error
            message. If not specified, print function will be used.
    """
    unexpected_keys = []
    missing_keys = []
    err_msg = []

    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    # use _load_from_state_dict to enable checkpoint version control
    def load(module, local_state_dict, prefix=""):
        # recursively check parallel module in case that the model has a
        # complicated structure, e.g., nn.Module(nn.Module(DDP))
        if isinstance(module, torch.nn.parallel.DistributedDataParallel):
            module = module.module
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            local_state_dict,
            prefix,
            local_metadata,
            True,
            missing_keys,
            unexpected_keys,
            err_msg,
        )
        for name, child in module._modules.items():
            if child is not None:
                child_prefix = prefix + name + "."
                child_state_dict = {
                    k: v
                    for k, v in local_state_dict.items()
                    if k.startswith(child_prefix)
                }
                load(child, child_state_dict, child_prefix)

        # Note that the hook can modify missing_keys and unexpected_keys.
        incompatible_keys = _IncompatibleKeys(missing_keys, unexpected_keys)
        if hasattr(module, "_load_state_dict_post_hooks"):
            for hook in module._load_state_dict_post_hooks.values():
                out = hook(module, incompatible_keys)
                assert out is None, (
                    "Hooks registered with "
                    "``register_load_state_dict_post_hook`` are not expected "
                    "to return new values, if incompatible_keys need to be "
                    "modified, it should be done inplace."
                )

    load(module, state_dict)
    load = None  # break load->load reference cycle

    # ignore "num_batches_tracked" of BN layers
    missing_keys = [key for key in missing_keys if "num_batches_tracked" not in key]

    if unexpected_keys:
        err_msg.append(
            "unexpected key in source " f'state_dict: {", ".join(unexpected_keys)}\n'
        )
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n'
        )

    if len(err_msg) > 0:
        err_msg.insert(0, "The model and loaded state dict do not match exactly\n")
        err_msg = "\n".join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        else:
            log.warning(err_msg)
