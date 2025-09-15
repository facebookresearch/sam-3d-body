import copy
from typing import Dict, List, Optional

import numpy as np
import pytorch_lightning as pl
import webdataset as wds
from omegaconf import OmegaConf

from core.metadata.dataset_train import dataset_cfg, train_cfg, val_cfg

from core.utils.config import to_lower
from core.utils.dist import get_world_size

from core.utils.logging import get_pylogger
from torch.utils.data import DataLoader, default_collate
from yacs.config import CfgNode

from .dataset_registry import DATASET_REGISTRY
from .filters_registry import FILTERS_REGISTRY
from .utils.mocap_dataset import MoCapDataset
from .utils.webdataset import MixedWebDataset

logging = get_pylogger(__name__)


class ThreePODataModule(pl.LightningDataModule):

    def __init__(self, cfg: CfgNode) -> None:
        """
        Initialize LightningDataModule for 3PO model training
        Args:
            cfg (CfgNode): Config file as a yacs CfgNode containing necessary dataset info.
        """
        super().__init__()
        self.cfg = cfg
        self.train_dataset = None
        self.val_dataset = None
        self.mocap_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        """Prepare datasets for training"""
        if self.train_dataset is None:
            train_datasets, sampling_weights = [], []
            self.train_length = 0  # track total size to estimate steps per epoch
            for dataset_name, v in self.cfg.DATASETS.TRAIN.items():
                # Load the dataset config
                dataset_key = dataset_name.lower()
                if dataset_key in dataset_cfg:
                    cur_cfg = copy.deepcopy(dataset_cfg[dataset_name.lower()])
                else:
                    logging.info(
                        f"{dataset_key} not found in train_cfg, using yaml as input"
                    )
                    cur_cfg = OmegaConf.to_container(v, resolve=True)

                # Get the data filtering strategy for the dataset
                if v.get("FILTERS", None) is not None:
                    cur_cfg["filters"] = FILTERS_REGISTRY.get(v.FILTERS)()
                cur_cfg["conf"] = v.get("CONF", None)
                cur_cfg["annot_file"] = v.get("ANNOT_FILE", None)
                cur_cfg["DENSE_FILTERS"] = v.get("DENSE_FILTERS", None)
                cur_cfg["flip_rgb"] = v.get("FLIP_RGB", False)
                cur_cfg["HAS_MP"] = v.get("HAS_MP", False)

                if v.get("USE_FOV_LOADER", False):
                    cur_cfg["type"] = "fov_train"

                # Define the dataset
                dataset = DATASET_REGISTRY.get(cur_cfg["type"])(
                    self.cfg,
                    cur_cfg,
                    mode="train",
                )
                train_datasets.append(dataset)
                self.train_length += cur_cfg.get("size", 1)

                # Get the sampling ratio of the datset
                sampling_weights.append(float(v.WEIGHT))

            # Concatenate multiple training datasets
            self.train_dataset = MixedWebDataset(
                train_datasets, np.array([sampling_weights])
            )

            if "MOCAP" in self.cfg.DATASETS:
                self.mocap_dataset = MoCapDataset(
                    **to_lower(dataset_cfg[self.cfg.DATASETS.MOCAP.lower()])
                )

        if self.val_dataset is None:
            val_datasets = []
            for dataset_name, v in self.cfg.DATASETS.VAL.items():
                # Load the dataset config
                dataset_key = dataset_name.lower()
                if dataset_key in dataset_cfg:
                    cur_cfg = copy.deepcopy(dataset_cfg[dataset_name.lower()])
                else:
                    logging.info(
                        f"{dataset_key} not found in dataset_cfg, using yaml as input"
                    )
                    cur_cfg = OmegaConf.to_container(v, resolve=True)

                # Get the data filtering strategy for the dataset
                cur_cfg["filters"] = FILTERS_REGISTRY.get(v.FILTERS)()
                cur_cfg["conf"] = v.get("CONF", None)
                cur_cfg["flip_rgb"] = v.get("FLIP_RGB", False)

                if v.get("USE_FOV_LOADER", False):
                    cur_cfg["type"] = "fov_train"

                # Define the dataset
                dataset = DATASET_REGISTRY.get(cur_cfg["type"])(
                    self.cfg,
                    cur_cfg,
                    mode="val",
                )
                val_datasets.append(dataset)

            # Concatenate multiple validation datasets
            sampling_weights = np.ones(len(val_datasets)) / len(val_datasets)
            self.val_dataset = MixedWebDataset(val_datasets, sampling_weights)

    def train_dataloader(self) -> Dict:
        """
        Setup training data loader.
        Returns:
            Dict: Dictionary containing image and mocap data dataloaders
        """
        # Reference: https://github.com/webdataset/webdataset/issues/250#issuecomment-2010938776
        train_dataset = self.train_dataset.batched(16)
        train_dataloader = (
            wds.WebLoader(
                train_dataset,
                batch_size=None,
                num_workers=self.cfg.GENERAL.NUM_WORKERS,
                pin_memory=True,
                shuffle=False,
            )
            .unbatched()
            .shuffle(self.cfg.GENERAL.get("SHUFFLE_BUFFER", 4000))
        )
        # To handle nested dictionary collation
        train_dataloader = train_dataloader.batched(
            self.cfg.TRAIN.BATCH_SIZE,
            collation_fn=default_collate,
        )
        if self.cfg.GENERAL.get("NO_EPOCH", False):
            train_dataloader = train_dataloader.with_epoch(int(1e9))
        else:
            train_dataloader = train_dataloader.with_epoch(
                self.train_length // self.cfg.TRAIN.BATCH_SIZE // get_world_size()
            )

        if self.mocap_dataset is not None:
            mocap_dataloader = DataLoader(
                self.mocap_dataset,
                self.cfg.TRAIN.NUM_TRAIN_SAMPLES * self.cfg.TRAIN.BATCH_SIZE,
                shuffle=True,
                drop_last=True,
                num_workers=1,
            )
            return {"img": train_dataloader, "mocap": mocap_dataloader}

        return {"img": train_dataloader}

    def val_dataloader(self) -> DataLoader:
        """
        Setup val data loader.
        Returns:
            torch.utils.data.DataLoader: Validation dataloader
        """
        val_dataloader = DataLoader(
            self.val_dataset,
            self.cfg.TRAIN.BATCH_SIZE,
            drop_last=True,
            shuffle=False,
            num_workers=self.cfg.GENERAL.NUM_WORKERS,
        )
        return val_dataloader


class EvalDataModule(pl.LightningDataModule):

    def __init__(
        self,
        dataset_cfg: Dict,
        model_cfg: Dict,
        eval_cfg: Dict,
        annot: Optional[List] = None,
    ) -> None:
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.model_cfg = model_cfg
        self.eval_cfg = eval_cfg
        self.annot = annot

    def setup(self, stage: str) -> None:
        assert stage == "test" or "predict"

        self.test_dataset = DATASET_REGISTRY.get(self.dataset_cfg["type"])(
            dataset_cfg=self.dataset_cfg,
            model_cfg=self.model_cfg,
            eval_cfg=self.eval_cfg,
            annot=self.annot,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            self.eval_cfg.get("batch_size", 16),
            drop_last=False,
            num_workers=self.eval_cfg.get("num_workers", 4),
        )
