from __future__ import annotations

__all__ = ["compute_pck"]

import torch.linalg as linalg

import open_pose
import data
import einops
import models
import torch
import tqdm
from torch.utils.data import DataLoader
from data.humans_dataset import HumansDatasetSubset
import pathlib


@torch.no_grad()
def compute_pck(
    generator: models.Generator,
    dataset: data.HumansDataset,
    batch_size: int = 32,
    device: torch.device = torch.device("cpu"),
    num_samples: int = 12800,
    threshold: float = 0.5,
) -> tuple[float, list[float]]:

    generator = generator.to(device)
    pose_model = open_pose.OpenPoseModel().to(device)

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=3,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=2,
    )

    max_items = len(dataset) * dataset.num_frames  # type: ignore
    if num_samples is not None:
        max_items = min(max_items, num_samples * dataset.num_frames)

    progress_bar = tqdm.tqdm(total=max_items, desc="Predicting poses")

    correct_sum = torch.zeros(dataset.num_keypoints, device=device)
    total_sum = torch.zeros(dataset.num_keypoints, device=device)
    count = 0

    for _, keypoints in data_loader:
        keypoints = keypoints.to(device)
        style_keypoints = einops.rearrange(keypoints, "n t k c -> (n t) 1 k c")
        pose_keypoints = style_keypoints

        # if generator.keypoint_embed_dim == 0:
        #     assert generator.num_poses == 1

        #     if generator.keypoint_heatmaps == False:
        #         # StyleGAN2
        #         style_keypoints = einops.rearrange(
        #             keypoints, "n t k c -> (n t) 1 k c"
        #         )
        #         pose_keypoints = style_keypoints

        #     else:
        #         # Spatial heatmaps.
        #         style_keypoints = keypoints[:, :1]
        #         pose_keypoints = keypoints

        # else:
        #     assert generator.num_poses == keypoints.size(1)
        #     style_keypoints = keypoints
        #     pose_keypoints = keypoints

        images = generator(style_keypoints, pose_keypoints)
        images = einops.rearrange(images, "n t c h w -> (n t) c h w")

        keypoints_real = einops.rearrange(keypoints, "n t k c -> (n t) k c")
        keypoints_fake = pose_model.keypoints(images)

        neck_position = keypoints_real[:, 0:1, :2]
        nose_position = keypoints_real[:, 1:2, :2]
        head_size = linalg.norm(neck_position - nose_position, dim=2)

        neck_visibility = keypoints_real[:, 0:1, 2]
        nose_visibility = keypoints_real[:, 1:2, 2]
        head_visibility = neck_visibility * nose_visibility

        threshold_scaled = threshold * head_size

        positions_real = keypoints_real[:, :, :2]
        positions_fake = keypoints_fake[:, :, :2]
        distance = linalg.norm(positions_real - positions_fake, dim=2)

        visibility_real = keypoints_real[:, :, 2]
        visibility_fake = keypoints_fake[:, :, 2]
        visibility = head_visibility * visibility_real * visibility_fake

        correct = (distance < threshold_scaled) * visibility
        correct = correct.sum(dim=0)
        correct_sum += correct

        total = head_visibility * visibility_real
        total = total.sum(dim=0)
        total_sum += total

        count += batch_size * dataset.num_frames
        progress_bar.update(batch_size * dataset.num_frames)

        if count == max_items:
            break

        if count > max_items:
            raise RuntimeError("TODO: Dataset not divisible by batch size.")

    progress_bar.close()

    pck = 100 * correct_sum / total_sum.clamp(min=1)
    pck_avg = pck.mean()

    pck = pck.tolist()
    pck_avg = float(pck_avg.item())

    return pck_avg, pck


class FileDataset:
    def __init__(self, path):
        self.dataset_subset_cache = {}

        self.image_paths = [p for p in list(pathlib.Path(path).iterdir()) if str(p).endswith(".png")]
        self.image_paths.sort()

        self.key_paths = [p for p in list(pathlib.Path(path).iterdir()) if str(p).endswith(".txt")]
        self.key_paths.sort()

    def load_keypoints(self, key_path):
        with open(key_path) as fp:
            lines = fp.readlines()

        dataset_path = lines[0].strip()
        key = lines[1].strip()

        if dataset_path in self.dataset_subset_cache:
            print(dataset_path)
            dataset = self.dataset_subset_cache[dataset_path]
        else:
            dataset = HumansDatasetSubset(dataset_path, num_frames=1, spacing=1, deterministic=True)

        pose = dataset.poses_db[key][0].float()
        pose[:, :2] /= 2
        return pose

    def __getitem__(self, i):
        image_path = self.image_paths[i]
        image = data.io.read_image(image_path)
        key_path = self.key_paths[i]
        keypoints = self.load_keypoints(key_path)
        return image, keypoints

    def __len__(self):
        return len(self.image_paths)


@torch.no_grad()
def compute_pck_from_files(
    path: str, batch_size: int = 32, device: torch.device = torch.device("cpu"), threshold: float = 0.5,
):
    pose_model = open_pose.OpenPoseModel().to(device)
    dataset = FileDataset(path)

    progress_bar = tqdm.tqdm(total=len(dataset), desc="Predicting poses")

    correct_sum = torch.zeros(18, device=device)
    total_sum = torch.zeros(18, device=device)

    image_batch = []
    keypoints_batch = []

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=3,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=2,
    )

    for images, keypoints_real in data_loader:
        images = images.to(device)
        keypoints_real = keypoints_real.to(device)

        keypoints_fake = pose_model.keypoints(images)

        neck_position = keypoints_real[:, 0:1, :2]
        nose_position = keypoints_real[:, 1:2, :2]
        head_size = linalg.norm(neck_position - nose_position, dim=2)

        neck_visibility = keypoints_real[:, 0:1, 2]
        nose_visibility = keypoints_real[:, 1:2, 2]
        head_visibility = neck_visibility * nose_visibility

        threshold_scaled = threshold * head_size

        positions_real = keypoints_real[:, :, :2]
        positions_fake = keypoints_fake[:, :, :2]
        distance = linalg.norm(positions_real - positions_fake, dim=2)

        visibility_real = keypoints_real[:, :, 2]
        visibility_fake = keypoints_fake[:, :, 2]
        visibility = head_visibility * visibility_real * visibility_fake

        correct = (distance < threshold_scaled) * visibility
        correct = correct.sum(dim=0)
        correct_sum += correct

        total = head_visibility * visibility_real
        total = total.sum(dim=0)
        total_sum += total

        progress_bar.update(batch_size)

    progress_bar.close()

    pck = 100 * correct_sum / total_sum.clamp(min=1)
    pck_avg = pck.mean()

    pck = pck.tolist()
    pck_avg = float(pck_avg.item())

    return pck_avg


#     detector = _load_detector(detector_path).eval().to(device)

#     image_paths = [p for p in list(pathlib.Path(path).iterdir()) if p.endswith(".png")]
#     image_paths.sort()

#     assert len(image_paths) % batch_size == 0

#     stats = _FeatureStats(max_items=len(image_paths), **stats_kwargs)
#     progress_bar = tqdm.tqdm(image_paths, desc="Computing file features")

#     images = []
#     for i, image_path in enumerate(progress_bar):

#         image = data.io.read_image(image_path)
#         images.append(image)

#         if (i + 1) % batch_size == 0:
#             image = torch.stack(images).to(device)
#             image = data.to_uint8(image)
#             features = detector(image, **detector_kwargs)
#             stats.append(features)
#             images = []

#     progress_bar.close()

#     return stats
