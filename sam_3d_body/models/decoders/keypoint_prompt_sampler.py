# Copyright (c) Meta Platforms, Inc. and affiliates.

import random
from abc import ABC, abstractmethod
from typing import Dict, List

import torch

from omegaconf import DictConfig


def build_keypoint_sampler(sampler_cfg, prompt_keypoints, keybody_idx):
    sampler_type = sampler_cfg.get("TYPE", "v1")
    if sampler_type == "v1":
        sampler_cls = KeypointSamplerV1
    else:
        raise ValueError("Invalid sampler type: ", sampler_type)

    return sampler_cls(sampler_cfg, prompt_keypoints, keybody_idx)


class BaseKeypointSampler(ABC):
    @abstractmethod
    def sample(
        self, gt_keypoints: torch.Tensor, pred_keypoints: torch.Tensor, is_train: bool
    ) -> torch.Tensor:
        pass

    def _get_worst_keypoint(self, distances, keypoint_list):
        # Set distance to -1 for non-promptable keypoints
        cur_dist = torch.ones_like(distances) * -1
        cur_dist[keypoint_list] = distances[keypoint_list]
        keypoint_idx = int(cur_dist.argmax())
        if cur_dist[keypoint_idx] > self.distance_thresh:
            valid_keypoint = True
        else:
            valid_keypoint = False
        return keypoint_idx, valid_keypoint

    def _get_random_keypoint(self, distances, keypoint_list):
        candidates = [idx for idx in keypoint_list if distances[idx] > 0]
        if len(candidates):
            keypoint_idx = random.choice(candidates)
            valid_keypoint = True
        else:
            keypoint_idx = None
            valid_keypoint = False
        return keypoint_idx, valid_keypoint

    def _masked_distance(self, x, y, mask=None):
        """
        Args:
            x, y: [B, K, D]
            mask: [B, K]
        Return:
            distances: [K, B]
        """
        distances = (x - y).pow(2).sum(dim=-1)
        if mask is not None:
            distances[mask] = -1
        return distances.T


class KeypointSamplerV1(BaseKeypointSampler):
    def __init__(
        self,
        sampler_cfg: DictConfig,
        prompt_keypoints: Dict,
        keybody_idx: List,
    ):
        self.prompt_keypoints = prompt_keypoints
        self._keybody_idx = keybody_idx
        self._non_keybody_idx = [
            idx for idx in self.prompt_keypoints if idx not in self._keybody_idx
        ]

        self.keybody_ratio = sampler_cfg.get("KEYBODY_RATIO", 0.8)
        self.worst_ratio = sampler_cfg.get("WORST_RATIO", 0.8)
        self.negative_ratio = sampler_cfg.get("NEGATIVE_RATIO", 0.0)
        self.dummy_ratio = sampler_cfg.get("DUMMY_RATIO", 0.1)
        self.distance_thresh = sampler_cfg.get("DISTANCE_THRESH", 0.0)

    def sample(
        self,
        gt_keypoints_2d: torch.Tensor,
        pred_keypoints_2d: torch.Tensor,
        is_train: bool = True,
        force_dummy: bool = False,
    ) -> torch.Tensor:
        # Get the distance between each predicted and gt keypoint
        # Elements will be ignored if (1) the gt has low confidence or
        # (2) both the gt and pred are outside of the image
        mask_1 = gt_keypoints_2d[:, :, -1] < 0.5
        mask_2 = (
            (gt_keypoints_2d[:, :, :2] > 0.5) | (gt_keypoints_2d[:, :, :2] < -0.5)
        ).any(dim=-1)

        # Elements to be ignored
        if not is_train or torch.rand(1).item() > self.negative_ratio:
            mask = mask_1 | mask_2
            # print_base = "positive"
        else:
            mask_3 = (
                (pred_keypoints_2d[:, :, :2] > 0.5)
                | (pred_keypoints_2d[:, :, :2] < -0.5)
            ).any(dim=-1)
            # To include negative prompts
            mask = mask_1 | (mask_2 & mask_3)
            # print_base = "negative"

        # Get pairwise distances with shape [K, B]
        distances = self._masked_distance(
            pred_keypoints_2d, gt_keypoints_2d[..., :2], mask
        )

        batch_size = distances.shape[1]
        keypoints_prompt = []
        for b in range(batch_size):
            # print_str = print_base

            # Decide to get the worst keypoint or a random keypoint
            if not is_train or torch.rand(1).item() < self.worst_ratio:
                sampler = self._get_worst_keypoint
                # print_str += "_worst"
            else:
                sampler = self._get_random_keypoint
                # print_str += "_random"

            # Decide to prompt keybody kepoints or non-keybody ones
            if not is_train or torch.rand(1).item() < self.keybody_ratio:
                cur_idx = self._keybody_idx
                alt_idx = self._non_keybody_idx
                # print_str += "_keybody"
            else:
                cur_idx = self._non_keybody_idx
                alt_idx = self._keybody_idx
                # print_str += "_nonkey"

            # Get a valid or dummy prompt
            if not is_train or torch.rand(1).item() > self.dummy_ratio:
                keypoint_idx, valid_keypoint = sampler(distances[:, b], cur_idx)

                if not valid_keypoint:
                    # Try the alternative keypoints
                    keypoint_idx, valid_keypoint = self._get_worst_keypoint(
                        distances[:, b], alt_idx
                    )
            else:
                valid_keypoint = False

            if valid_keypoint:
                cur_point = gt_keypoints_2d[b, keypoint_idx].clone()
                if torch.any(cur_point[:2] > 0.5) or torch.any(cur_point[:2] < -0.5):
                    # Negative prompt --> indicating the predicted keypoint is incorrect
                    cur_point[:2] = pred_keypoints_2d[b, keypoint_idx][:2]
                    cur_point = torch.clamp(
                        cur_point + 0.5, min=0.0, max=1.0
                    )  # shift from [-0.5, 0.5] to [0, 1]
                    cur_point[-1] = -1
                    # print_str += "_negative"
                else:
                    cur_point = torch.clamp(
                        cur_point + 0.5, min=0.0, max=1.0
                    )  # shift from [-0.5, 0.5] to [0, 1]
                    cur_point[-1] = self.prompt_keypoints[
                        keypoint_idx
                    ]  # map to prompt_idx
                    # print_str += "_positive"
            else:
                cur_point = torch.zeros(3).to(gt_keypoints_2d)
                cur_point[-1] = -2
                # print_str += "_dummy"

            if force_dummy:
                cur_point = torch.zeros(3).to(gt_keypoints_2d)
                cur_point[-1] = -2

            keypoints_prompt.append(cur_point)
            # print(print_str)

        keypoints_prompt = torch.stack(keypoints_prompt, dim=0).view(batch_size, 1, 3)
        return keypoints_prompt
