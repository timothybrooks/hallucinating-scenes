__all__ = ["compute_fid", "compute_fid_from_files"]

import pathlib
from typing import Dict, Optional

import data
import models
import numpy as np
import scipy.linalg
import torch

from . import metric_utils


@torch.no_grad()
def compute_fid(
    generator: models.Generator,
    dataset: data.HumansDataset,
    batch_size: int = 8,
    device: torch.device = torch.device("cpu"),
    dataset_cache_dir: Optional[str] = None,
    num_gen_samples: int = 12800,
    num_data_samples: Optional[int] = None,
    truncation_strength: float = 0.0,
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
        capture_mean_cov=True,
    )
    data_mu, data_sigma = data_stats.get_mean_cov()

    gen_stats = metric_utils.compute_generator_stats(
        generator,
        dataset,
        detector_path,
        detector_kwargs,
        batch_size,
        num_gen_samples,
        device,
        capture_mean_cov=True,
        truncation_strength=truncation_strength,
    )
    gen_mu, gen_sigma = gen_stats.get_mean_cov()

    m = np.square(gen_mu - data_mu).sum()  # type: ignore
    s, _ = scipy.linalg.sqrtm(  # type: ignore
        np.dot(gen_sigma, data_sigma), disp=False
    )
    fid = np.real(m + np.trace(gen_sigma + data_sigma - s * 2))  # type: ignore
    return float(fid)


@torch.no_grad()
def compute_fid_from_files(
    path: str,
    dataset: data.HumansDataset,
    batch_size: int = 8,
    device: torch.device = torch.device("cpu"),
    dataset_cache_dir: Optional[str] = None,
    num_data_samples: Optional[int] = None,
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
        capture_mean_cov=True,
    )
    data_mu, data_sigma = data_stats.get_mean_cov()

    gen_stats = metric_utils.compute_files_stats(
        path,
        detector_path,
        detector_kwargs,
        batch_size,
        device,
        capture_mean_cov=True,
    )
    gen_mu, gen_sigma = gen_stats.get_mean_cov()

    m = np.square(gen_mu - data_mu).sum()  # type: ignore
    s, _ = scipy.linalg.sqrtm(  # type: ignore
        np.dot(gen_sigma, data_sigma), disp=False
    )
    fid = np.real(m + np.trace(gen_sigma + data_sigma - s * 2))  # type: ignore
    return float(fid)
