from typing import List
import webdataset as wds
import numpy as np


class MixedWebDataset(wds.WebDataset):
    def __init__(
        self, 
        datasets: List[wds.WebDataset],
        weights: np.ndarray
    ) -> None:
        super(wds.WebDataset, self).__init__()
        weights = weights / weights.sum()  # normalize
        self.append(wds.RandomMix(datasets, weights))
