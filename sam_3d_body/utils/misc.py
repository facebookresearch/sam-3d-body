import time
import warnings
from importlib.util import find_spec
from collections import abc
from typing import Any, Callable, Type, Union, Sequence
import numpy as np
from torch import Tensor
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only

from . import rich_utils
from .logging import get_pylogger

log = get_pylogger(__name__)


def is_seq_of(seq: Any,
              expected_type: Union[Type, tuple],
              seq_type: Type = None) -> bool:
    """Check whether it is a sequence of some type.

    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type or tuple): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type. Defaults to None.

    Returns:
        bool: Return True if ``seq`` is valid else False.

    Examples:
        >>> from mmengine.utils import is_seq_of
        >>> seq = ['a', 'b', 'c']
        >>> is_seq_of(seq, str)
        True
        >>> is_seq_of(seq, int)
        False
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


def to_numpy(x: Union[Tensor, Sequence[Tensor]],
             unzip: bool = False) -> Union[np.ndarray, tuple]:
    """Convert torch tensor to numpy.ndarray.
    Args:
        x (Tensor | Sequence[Tensor]): A single tensor or a sequence of
            tensors
        unzip (bool): Whether unzip the input sequence. Defaults to ``False``
    Returns:
        np.ndarray | tuple: If ``return_device`` is ``True``, return a tuple
        of converted numpy array(s) and the device indicator; otherwise only
        return the numpy array(s)
    """

    if isinstance(x, Tensor):
        arrays = x.detach().cpu().float().numpy()
        device = x.device
    elif is_seq_of(x, Tensor):
        if unzip:
            # convert (A, B) -> [(A[0], B[0]), (A[1], B[1]), ...]
            arrays = [
                tuple(to_numpy(_x[None, :]) for _x in _each)
                for _each in zip(*x)
            ]
        else:
            arrays = [to_numpy(_x) for _x in x]

        device = x[0].device

    else:
        raise ValueError(f'Invalid input type {type(x)}')

    return arrays


@rank_zero_only
def log_hyperparameters(object_dict: dict) -> None:
    """Controls which config parts are saved by lightning loggers.

    Additionally saves:
    - Number of model parameters
    """

    hparams = {}

    cfg = object_dict["cfg"]
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    for k in cfg.keys():
        hparams[k] = cfg.get(k)

    # Resolve all interpolations
    def _resolve(_cfg):
        if isinstance(_cfg, DictConfig):
            _cfg = OmegaConf.to_container(_cfg, resolve=True)
        return _cfg

    hparams = {k: _resolve(v) for k, v in hparams.items()}

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that wraps the task function in extra utilities.

    Makes multirun more resistant to failure.

    Utilities:
    - Calling the `utils.extras()` before the task is started
    - Calling the `utils.close_loggers()` after the task is finished
    - Logging the exception if occurs
    - Logging the task total execution time
    - Logging the output dir
    """

    def wrap(cfg: DictConfig):

        # apply extra utilities
        extras(cfg)

        # execute the task
        try:
            start_time = time.time()
            ret = task_func(cfg=cfg)
        except Exception as ex:
            log.exception("")  # save exception to `.log` file
            raise ex
        finally:
            path = Path(cfg.paths.output_dir, "exec_time.log")
            content = f"'{cfg.task_name}' execution time: {time.time() - start_time} (s)"
            save_file(path, content)  # save task execution time (even if exception occurs)
            close_loggers()  # close loggers (even if exception occurs so multirun won't fail)

        log.info(f"Output dir: {cfg.paths.output_dir}")

        return ret

    return wrap


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
    - Ignoring python warnings
    - Setting tags from command line
    - Rich config printing
    """

    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


@rank_zero_only
def save_file(path: str, content: str) -> None:
    """Save file in rank zero mode (only on one process in multi-GPU setup)."""
    with open(path, "w+") as file:
        file.write(content)


def close_loggers() -> None:
    """Makes sure all loggers closed properly (prevents logging failure during multirun)."""

    log.info("Closing loggers...")

    if find_spec("wandb"):  # if wandb is installed
        import wandb

        if wandb.run:
            log.info("Closing wandb!")
            wandb.finish()
