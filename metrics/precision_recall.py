__all__ = ["compute_pr"]

import pathlib
from typing import Optional, Tuple

import data
import models
import torch
import torch.nn.functional as F

from . import metric_utils


@torch.no_grad()
def compute_pr(
    generator: models.Generator,
    dataset: data.HumansDataset,
    batch_size: int = 128,
    device: torch.device = torch.device("cpu"),
    dataset_cache_dir: Optional[str] = None,
    num_gen_samples: int = 12800,
    num_data_samples: int = 12800,
    nhood_size: int = 3,
    row_batch_size: int = 5000,
    col_batch_size: int = 5000,
) -> Tuple[float, float]:

    detector_dir = pathlib.Path(__file__).parent.joinpath("pretrained")
    detector_path = str(detector_dir.joinpath("vgg16.pt"))

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

    data_features = data_stats.get_all_torch().to(torch.float16)
    gen_features = gen_stats.get_all_torch().to(torch.float16)

    data_features = data_features[torch.randperm(len(data_features))]
    gen_features = gen_features[torch.randperm(len(gen_features))]

    results = dict()

    for name, manifold, probes in [
        ("precision", data_features, gen_features),
        ("recall", gen_features, data_features),
    ]:

        manifold = manifold.to(device)
        probes = probes.to(device)

        kth = []
        for manifold_batch in manifold.split(row_batch_size):
            dist = _compute_distances(
                row_features=manifold_batch,
                col_features=manifold,
                col_batch_size=col_batch_size,
            )
            kth.append(
                dist.to(torch.float32)
                .kthvalue(nhood_size + 1)
                .values.to(torch.float16)
            )
        kth = torch.cat(kth)

        pred = []
        for probes_batch in probes.split(row_batch_size):
            dist = _compute_distances(
                row_features=probes_batch,
                col_features=manifold,
                col_batch_size=col_batch_size,
            )
            pred.append((dist <= kth).any(dim=1))

        results[name] = float(torch.cat(pred).to(torch.float32).mean())

    return results["precision"], results["recall"]


@torch.no_grad()
def _compute_distances(
    row_features: torch.Tensor, col_features: torch.Tensor, col_batch_size: int
) -> torch.Tensor:

    num_cols = col_features.size(0)
    num_batches = (num_cols - 1) // col_batch_size + 1

    col_batches = F.pad(col_features, [0, 0, 0, -num_cols % num_batches])
    col_batches = col_batches.chunk(num_batches)

    dist_batches = []
    for col_batch in col_batches:
        dist_batch = torch.cdist(row_features[None], col_batch[None])[0]
        dist_batches.append(dist_batch)

    distances = torch.cat(dist_batches, dim=1)[:, :num_cols]
    return distances
