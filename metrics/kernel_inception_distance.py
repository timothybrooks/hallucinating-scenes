__all__ = ["compute_kid"]

import pathlib
from typing import Dict, Optional

import data
import torch
import models
import numpy as np

from . import metric_utils


@torch.no_grad()
def compute_kid(
    generator: models.Generator,
    dataset: data.HumansDataset,
    batch_size: int = 128,
    device: torch.device = torch.device("cpu"),
    dataset_cache_dir: Optional[str] = None,
    num_gen_samples: int = 50000,
    num_data_samples: int = 1000000,
    num_subsets: int = 100,
    max_subset_size: int = 1000,
) -> float:

    detector_dir = pathlib.Path(__file__).parent.joinpath("pretrained")
    detector_path = str(detector_dir.joinpath("inception-2015-12-05.pt"))

    # Returns raw features before the softmax layer.
    detector_kwargs = dict(return_features=True)

    data_stats = metric_utils.compute_dataset_stats(
        dataset,
        detector_path,
        detector_kwargs,
        batch_size,
        num_data_samples,
        device,
        dataset_cache_dir,
        capture_all=True,
    )
    data_features = data_stats.get_all()

    gen_stats = metric_utils.compute_generator_stats(
        generator,
        dataset,
        detector_path,
        detector_kwargs,
        batch_size,
        num_gen_samples,
        device,
        capture_all=True,
    )
    gen_features = gen_stats.get_all()

    n = data_features.shape[1]
    m = min(min(data_features.shape[0], gen_features.shape[0]), max_subset_size)
    t = 0
    for _subset_idx in range(num_subsets):
        x = gen_features[
            np.random.choice(gen_features.shape[0], m, replace=False)
        ]
        y = data_features[
            np.random.choice(data_features.shape[0], m, replace=False)
        ]
        a = (x @ x.T / n + 1) ** 3 + (y @ y.T / n + 1) ** 3
        b = (x @ y.T / n + 1) ** 3
        t += (a.sum() - np.diag(a).sum()) / (m - 1) - b.sum() * 2 / m
    kid = t / num_subsets / m
    return float(kid)
