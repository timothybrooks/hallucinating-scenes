__all__ = ["compute_is"]

import pathlib
from typing import Optional, Tuple

import data
import models
import numpy as np
import torch

from . import metric_utils


@torch.no_grad()
def compute_is(
    generator: models.Generator,
    dataset: data.HumansDataset,
    batch_size: int = 128,
    device: torch.device = torch.device("cpu"),
    num_gen_samples: int = 50000,
    num_splits: int = 10,
) -> Tuple[float, float]:

    detector_dir = pathlib.Path(__file__).parent.joinpath("pretrained")
    detector_path = str(detector_dir.joinpath("inception-2015-12-05.pt"))

    # Doesn't apply bias in the softmax layer.
    detector_kwargs = dict(no_output_bias=True)

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
    gen_probs = gen_stats.get_all()

    scores = []
    for i in range(num_splits):
        part = gen_probs[
            i
            * gen_stats.num_items
            // num_splits : (i + 1)
            * gen_stats.num_items
            // num_splits
        ]
        kl = part * (
            np.log(part) - np.log(np.mean(part, axis=0, keepdims=True))  # type: ignore
        )
        kl = np.mean(np.sum(kl, axis=1))
        scores.append(np.exp(kl))
    return float(np.mean(scores)), float(np.std(scores))
