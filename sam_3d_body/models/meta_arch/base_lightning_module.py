# Copyright (c) Meta Platforms, Inc. and affiliates.

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger


class BaseLightningModule(pl.LightningModule):
    def _log_metric(self, name, value, step=None):
        for logger in self.trainer.loggers:
            if isinstance(logger, WandbLogger):
                if step is not None:
                    logger.experiment.log({name: value, "step": step})
                else:
                    logger.experiment.log({name: value})
            elif isinstance(logger, TensorBoardLogger):
                logger.experiment.add_scalar(name, value, step)
            else:
                raise ValueError(f"Unsupported logger: {logger}")

    def _log_image(self, name, img_tensor, dataformats="CHW", step_count=None):
        """Log image tensor to both W&B and TensorBoard."""
        step = step_count if step_count is not None else self.global_step
        for logger in self.trainer.loggers:
            if isinstance(logger, WandbLogger):
                import wandb

                img = img_tensor
                if dataformats.upper() == "CHW":
                    # If in PyTorch format (C,H,W), convert to (H,W,C) for wandb
                    img = img_tensor.permute(1, 2, 0).cpu().numpy()
                logger.experiment.log({name: wandb.Image(img), "step": step})
            elif isinstance(logger, TensorBoardLogger):
                logger.experiment.add_image(
                    name, img_tensor, step, dataformats=dataformats
                )
            else:
                raise ValueError(f"Unsupported logger: {logger}")

    def _log_hist(self, name, array, step_count=None):
        for logger in self.trainer.loggers:
            if isinstance(logger, WandbLogger):
                import wandb

                value = wandb.Histogram(
                    np_histogram=(array, np.arange(array.shape[0] + 1)),
                )
                logger.experiment.log({name: value, "step": step_count})
