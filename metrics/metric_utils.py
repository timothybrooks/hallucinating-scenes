__all__ = [
    "compute_dataset_stats",
    "compute_generator_stats",
    "compute_files_stats",
]

import hashlib
import os
import pathlib
import pickle
import uuid
from typing import Any, Dict, Optional

import data
import einops
import models
import numpy as np
import torch
import torch.jit as jit
import torch.utils.data as torch_data
import tqdm


@torch.no_grad()
def compute_dataset_stats(
    dataset: data.HumansDataset,
    detector_path: str,
    detector_kwargs: Dict[str, Any] = {},
    batch_size: int = 8,
    num_samples: Optional[int] = None,
    device: torch.device = torch.device("cpu"),
    dataset_cache_dir: Optional[str] = None,
    **stats_kwargs,
):
    cache_path = None
    if dataset_cache_dir is not None:

        cache_path = _make_cache_path(
            dataset_cache_dir,
            dataset_split=dataset.split,
            dataset_subsets=dataset.subsets,
            dataset_num_frames=dataset.num_frames,
            dataset_spacing=dataset.spacing,
            detector_path=detector_path,
            detector_kwargs=detector_kwargs,
            num_samples=num_samples,
            stats_kwargs=stats_kwargs,
        )

        if os.path.isfile(cache_path):
            print("Loading dataset features from cache.")
            return _FeatureStats.load(cache_path)

    detector = _load_detector(detector_path).eval().to(device)

    max_items = len(dataset) * dataset.num_frames  # type: ignore
    if num_samples is not None:
        max_items = min(max_items, num_samples * dataset.num_frames)

    stats = _FeatureStats(max_items=max_items, **stats_kwargs)
    progress_bar = tqdm.tqdm(total=max_items, desc="Computing dataset features")

    data_loader = torch_data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=3,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=2,
    )

    for image, _ in data_loader:
        image = image.to(device)
        image = einops.rearrange(image, "n t c h w -> (n t) c h w")
        image = data.to_uint8(image)

        features = detector(image, **detector_kwargs)
        stats.append(features)

        progress_bar.n = stats.num_items
        progress_bar.refresh()

        if stats.num_items >= max_items:
            break

    progress_bar.close()

    if cache_path is not None:

        assert dataset_cache_dir is not None
        os.makedirs(dataset_cache_dir, exist_ok=True)

        temp_path = f"{cache_path}.{uuid.uuid4().hex}"
        stats.save(temp_path)
        os.replace(temp_path, cache_path)

    return stats


@torch.no_grad()
def compute_generator_stats(
    generator: models.Generator,
    dataset: data.HumansDataset,
    detector_path: str,
    detector_kwargs: Dict[str, Any] = {},
    batch_size: int = 8,
    num_samples: Optional[int] = None,
    device: torch.device = torch.device("cpu"),
    truncation_strength: float = 0.0,
    **stats_kwargs,
):
    generator = generator.eval().to(device)
    detector = _load_detector(detector_path).eval().to(device)

    max_items = len(dataset) * dataset.num_frames  # type: ignore
    if num_samples is not None:
        max_items = min(max_items, num_samples * dataset.num_frames)

    stats = _FeatureStats(max_items=max_items, **stats_kwargs)
    progress_bar = tqdm.tqdm(
        total=max_items, desc="Computing generator features"
    )

    data_loader = torch_data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=3,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=2,
    )

    for _, keypoints in data_loader:
        assert keypoints.size(1) == 1

        keypoints = keypoints.to(device)

        # if generator.num_poses == 1:  #generator.keypoint_embed_dim == 0:
        #     # assert generator.num_poses == 1

        #     if generator.keypoint_heatmaps == False:
        #         # StyleGAN2

        #     else:
        #         # Spatial heatmaps.
        #         style_keypoints = keypoints[:, :1]
        #         pose_keypoints = keypoints

        # else:
        #     assert generator.num_poses == keypoints.size(1)
        #     style_keypoints = keypoints
        #     pose_keypoints = keypoints

        if truncation_strength > 0:
            keypoints_repeat = einops.repeat(
                keypoints, "n 1 k c -> (n r) 1 k c", r=10001
            )
            styles = generator.mapping_network.sample_styles(keypoints_repeat)
            styles = einops.rearrange(styles, "(n r) k c -> n r k c", r=10001)
            cluster = styles[:, 1:].mean(dim=1)
            styles = torch.lerp(styles[:, 0], cluster, truncation_strength)
            keypoints = einops.rearrange(keypoints, "n 1 k c -> n k c")
            image = generator.synthesis_network(styles, keypoints)

        # elif truncation_strength < 0:
        #     keypoints_repeat = einops.repeat(
        #         keypoints, "n 1 k c -> (n r) 1 k c", r=10000
        #     )
        #     styles = generator.mapping_network.sample_styles(keypoints_repeat)
        #     styles = einops.rearrange(styles, "(n r) k c -> n r k c", r=10000)
        #     cluster_std, cluster = torch.std_mean(styles, dim=1)

        #     while True:
        #         styles = generator.mapping_network.sample_styles(keypoints)
        #         styles_stds = torch.abs(cluster - styles) / cluster_std
        #         if torch.all(styles_stds <= -truncation_strength):
        #             break

        #     keypoints = einops.rearrange(keypoints, "n 1 k c -> n k c")
        #     image = generator.synthesis_network(styles, keypoints)

        # elif truncation_strength < 0:

        #     styles_list = []
        #     for keypoints_single in keypoints:
        #         keypoints_repeat = einops.repeat(
        #             keypoints_single, "1 k c -> r 1 k c", r=10000
        #         )
        #         styles = generator.mapping_network.sample_styles(
        #             keypoints_repeat
        #         )
        #         cluster_std, cluster_mean = torch.std_mean(
        #             styles, dim=0, keepdim=True
        #         )

        #         while True:
        #             styles = generator.mapping_network.sample_styles(
        #                 keypoints_repeat
        #             )
        #             styles_stds = torch.abs(cluster_mean - styles) / cluster_std
        #             mask = styles_stds <= -truncation_strength
        #             mask = einops.reduce(mask, "r k c -> r", "prod")
        #             if mask.any():
        #                 index = mask.nonzero()
        #                 styles_list.append(styles[index[0, 0]])
        #                 break

        #     styles = torch.stack(styles_list)
        #     keypoints = einops.rearrange(keypoints, "n 1 k c -> n k c")
        #     image = generator.synthesis_network(styles, keypoints)

        # elif truncation_strength < 0:
        #     # Mean over channels

        #     styles_list = []
        #     for keypoints_single in keypoints:
        #         keypoints_repeat = einops.repeat(
        #             keypoints_single, "1 k c -> r 1 k c", r=10000
        #         )
        #         styles = generator.mapping_network.sample_styles(
        #             keypoints_repeat
        #         )
        #         cluster_std, cluster_mean = torch.std_mean(
        #             styles, dim=(0, 2), keepdim=True
        #         )

        #         while True:
        #             styles = generator.mapping_network.sample_styles(
        #                 keypoints_repeat
        #             )
        #             styles_stds = (
        #                 torch.abs(cluster_mean - styles).mean(
        #                     dim=2, keepdim=True
        #                 )
        #                 / cluster_std
        #             )
        #             mask = styles_stds <= -truncation_strength
        #             mask = einops.reduce(mask, "r k c -> r", "prod")
        #             if mask.any():
        #                 index = mask.nonzero()
        #                 styles_list.append(styles[index[0, 0]])
        #                 break

        #     styles = torch.stack(styles_list)
        #     keypoints = einops.rearrange(keypoints, "n 1 k c -> n k c")
        #     image = generator.synthesis_network(styles, keypoints)

        # elif truncation_strength < 0:
        #     # By percentile

        #     styles_list = []
        #     for keypoints_single in keypoints:
        #         N = 10000
        #         P = -truncation_strength
        #         keypoints_repeat = einops.repeat(
        #             keypoints_single, "1 k c -> r 1 k c", r=N
        #         )
        #         styles = generator.mapping_network.sample_styles(
        #             keypoints_repeat
        #         )
        #         cluster_std, cluster_mean = torch.std_mean(
        #             styles, dim=0, keepdim=True
        #         )
        #         styles = generator.mapping_network.sample_styles(
        #             keypoints_repeat
        #         )
        #         sq_error = (cluster_mean - styles).square()
        #         mse = einops.reduce(sq_error, "r k c -> r", "max")
        #         indices = torch.argsort(mse, dim=0)
        #         index = torch.randint(int(P * N), ()).item()
        #         index = indices[index]  # type: ignore
        #         styles_list.append(styles[index])

        #     styles = torch.stack(styles_list)
        #     keypoints = einops.rearrange(keypoints, "n 1 k c -> n k c")
        #     image = generator.synthesis_network(styles, keypoints)

        else:
            image = generator(keypoints, keypoints)
            image = einops.rearrange(image, "n t c h w -> (n t) c h w")

        image = data.to_uint8(image)

        features = detector(image, **detector_kwargs)
        stats.append(features)

        progress_bar.n = stats.num_items
        progress_bar.refresh()

        if stats.num_items >= max_items:
            break

    progress_bar.close()

    return stats


@torch.no_grad()
def compute_files_stats(
    path: str,
    detector_path: str,
    detector_kwargs: Dict[str, Any] = {},
    batch_size: int = 8,
    device: torch.device = torch.device("cpu"),
    **stats_kwargs,
):
    detector = _load_detector(detector_path).eval().to(device)

    image_paths = [p for p in list(pathlib.Path(path).iterdir()) if str(p).endswith(".png")]
    image_paths.sort()

    assert len(image_paths) % batch_size == 0

    stats = _FeatureStats(max_items=len(image_paths), **stats_kwargs)
    progress_bar = tqdm.tqdm(image_paths, desc="Computing file features")

    images = []
    for i, image_path in enumerate(progress_bar):

        image = data.io.read_image(image_path)
        images.append(image)

        if (i + 1) % batch_size == 0:
            image = torch.stack(images).to(device)
            image = data.to_uint8(image)
            features = detector(image, **detector_kwargs)
            stats.append(features)
            images = []

    progress_bar.close()

    return stats


# ==============================================================================
# Private functions and classes.
# ==============================================================================


_detector_cache = dict()


def _load_detector(path: str):
    if path not in _detector_cache:
        detector = jit.load(path).eval()
        _detector_cache[path] = detector
    return _detector_cache[path]


def _make_cache_path(cache_dir: str, **kwargs):
    hash_str = repr(sorted(kwargs.items())).encode()
    hash = hashlib.blake2b(hash_str, digest_size=4).hexdigest()
    cache_path = os.path.join(cache_dir, f"features-{hash}.pkl")
    return cache_path


class _FeatureStats:
    def __init__(
        self, capture_all=False, capture_mean_cov=False, max_items=None
    ):
        self.capture_all = capture_all
        self.capture_mean_cov = capture_mean_cov
        self.max_items = max_items
        self.num_items = 0
        self.num_features = None
        self.all_features = None
        self.raw_mean = None
        self.raw_cov = None

    def set_num_features(self, num_features):
        if self.num_features is not None:
            assert num_features == self.num_features
        else:
            self.num_features = num_features
            self.all_features = []
            self.raw_mean = np.zeros([num_features], dtype=np.float64)
            self.raw_cov = np.zeros(
                [num_features, num_features], dtype=np.float64
            )

    def is_full(self):
        return (self.max_items is not None) and (
            self.num_items >= self.max_items
        )

    def append(self, x):
        assert isinstance(x, torch.Tensor) and x.ndim == 2
        x = x.cpu().numpy()
        x = np.asarray(x, dtype=np.float32)  # type: ignore
        assert x.ndim == 2  # type: ignore
        if (self.max_items is not None) and (
            self.num_items + x.shape[0] > self.max_items  # type: ignore
        ):
            if self.num_items >= self.max_items:
                return
            x = x[: self.max_items - self.num_items]  # type: ignore

        self.set_num_features(x.shape[1])  # type: ignore
        self.num_items += x.shape[0]  # type: ignore
        if self.capture_all:
            self.all_features.append(x)
        if self.capture_mean_cov:
            # FIXME: Keep this part on cuda.
            x64 = x.astype(np.float64)  # type: ignore
            self.raw_mean += x64.sum(axis=0)
            self.raw_cov += x64.T @ x64

    def get_all(self) -> np.ndarray:
        assert self.capture_all
        return np.concatenate(self.all_features, axis=0)  # type: ignore

    def get_all_torch(self) -> torch.Tensor:
        return torch.from_numpy(self.get_all())

    def get_mean_cov(self):
        assert self.capture_mean_cov
        mean = self.raw_mean / self.num_items
        cov = self.raw_cov / self.num_items
        cov = cov - np.outer(mean, mean)
        return mean, cov

    def save(self, pkl_file):
        with open(pkl_file, "wb") as f:
            pickle.dump(self.__dict__, f)

    @staticmethod
    def load(pkl_file):
        with open(pkl_file, "rb") as f:
            s = pickle.load(f)
        obj = _FeatureStats(
            capture_all=s["capture_all"], max_items=s["max_items"]
        )
        obj.__dict__.update(s)
        return obj
