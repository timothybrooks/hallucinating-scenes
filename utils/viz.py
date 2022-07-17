from __future__ import annotations

__all__ = [
    "display_grid",
    "pack_frames",
    "unpack_frames",
    "pack_keypoints",
    "unpack_keypoints",
    "visualize_heatmaps",
    "visualize_skeletons",
    "sample_styles",
    "write_video",
]

from typing import Optional
import cv2

import numpy as np
import einops
import seaborn
import torch
import torch.nn.functional as F

import utils
import data
import models
import numpy as np


@torch.no_grad()
def display_grid(frames: torch.Tensor, save_path: Optional[str] = None, **kwargs):
    assert frames.dim() == 4 or frames.dim() == 5

    nrow = kwargs.pop("nrow", None)
    pad_value = kwargs.pop("pad_value", 1)

    if frames.dim() == 5:
        if nrow is None:
            nrow = frames.size(1)
        frames = pack_frames(frames)

    frames_grid = data.make_grid(frames, nrow=nrow, pad_value=pad_value)
    data.io.display_image(frames_grid)

    if save_path is not None:
        data.io.write_image(frames_grid.cpu(), save_path)


def pack_frames(frames: torch.Tensor) -> torch.Tensor:
    frames = einops.rearrange(frames, "n t c h w -> (n t) c h w")
    return frames


def unpack_frames(frames: torch.Tensor, length: int) -> torch.Tensor:
    frames = einops.rearrange(frames, "(n t) c h w -> n t c h w", t=length)
    return frames


def pack_keypoints(keypoints: torch.Tensor) -> torch.Tensor:
    keypoints = einops.rearrange(keypoints, "n t k c -> (n t) k c")
    return keypoints


def unpack_keypoints(keypoints: torch.Tensor, length: int) -> torch.Tensor:
    keypoints = einops.rearrange(keypoints, "(n t) k c -> n t k c", t=length)
    return keypoints


@torch.no_grad()
def interleave(*frames_args: torch.Tensor, dim: int = 1) -> torch.Tensor:
    frames = torch.stack(frames_args, dim=dim)
    frames = einops.rearrange(frames, "n i t c h w -> (n i) t c h w")
    return frames


@torch.no_grad()
def visualize_heatmaps(
    keypoints: torch.Tensor, dataset: data.HumansDataset
) -> torch.Tensor:

    assert keypoints.dim() == 4
    assert keypoints.size(2) == dataset.num_keypoints
    assert keypoints.size(3) == 3

    num_frames = keypoints.size(1)
    keypoints = pack_keypoints(keypoints)

    heatmaps = utils.keypoint_heatmaps(
        keypoints, dataset.resolution, dataset.resolution
    )
    heatmaps = einops.repeat(heatmaps, "n k h w -> n k c h w", c=3)

    colors = seaborn.color_palette("hls", n_colors=dataset.num_keypoints)
    colors = torch.tensor(colors, device=keypoints.device)
    colors = einops.rearrange(colors, "k c -> 1 k c 1 1")

    heatmaps = heatmaps * colors
    heatmaps = einops.reduce(heatmaps, "n k c h w -> n c h w", "max")
    heatmaps = 2 * heatmaps - 0.75
    heatmaps = unpack_frames(heatmaps, num_frames)
    return heatmaps


@torch.no_grad()
def visualize_skeletons(
    keypoints: torch.Tensor, dataset: data.HumansDataset
) -> torch.Tensor:

    assert keypoints.dim() == 4
    assert keypoints.size(2) == dataset.num_keypoints
    assert keypoints.size(3) == 3

    num_frames = keypoints.size(1)
    keypoints = pack_keypoints(keypoints)

    background = keypoints.new_full(
        (keypoints.size(0), 3, dataset.resolution, dataset.resolution), -1
    )

    skeletons = utils.draw_openpose(background, keypoints)
    skeletons += 0.25
    skeletons = unpack_frames(skeletons, num_frames)
    return skeletons


@torch.no_grad()
def sample_styles(
    keypoints: torch.Tensor,
    generator: models.Generator,
    samples: int = 1,
    strength: float = 0.0,
    exclude: float = 0.0,
    repeat: bool = True,
    seed: Optional[int] = None,
):
    keypoints_repeat = einops.repeat(keypoints, "n t k c -> (n r) t k c", r=samples)

    if seed is None:
        random_generator = None
    else:
        random_generator = torch.Generator(keypoints.device).manual_seed(seed)

    if exclude > 0:
        styles_list = []
        for keypoints_single in keypoints_repeat:
            N = 10000
            keypoints_repeat2 = einops.repeat(keypoints_single, "1 k c -> r 1 k c", r=N)
            styles = generator.mapping_network.sample_styles(
                keypoints_repeat2, generator=random_generator
            )
            cluster_std, cluster_mean = torch.std_mean(styles, dim=0, keepdim=True)
            styles = generator.mapping_network.sample_styles(
                keypoints_repeat2, generator=random_generator
            )
            sq_error = (cluster_mean - styles).square()
            mse = einops.reduce(sq_error, "r k c -> r", "mean")
            indices = torch.argsort(mse, dim=0)
            index = torch.randint(
                int((1 - exclude) * N),
                (),
                device=keypoints.device,
                generator=random_generator,
            ).item()
            index = indices[index]  # type: ignore
            styles_list.append(styles[index])
        styles = torch.stack(styles_list)

    else:
        styles = generator.mapping_network.sample_styles(
            keypoints_repeat, generator=random_generator
        )

    if strength > 0:
        keypoints = einops.repeat(keypoints, "n t k c -> (n r) t k c", r=10000)
        cluster = generator.mapping_network.sample_styles(
            keypoints, generator=random_generator
        )
        cluster = einops.reduce(cluster, "(n r) d c -> n d c", "mean", r=10000)
        cluster = einops.repeat(cluster, "n d c -> (n r) d c", r=samples)
        styles = torch.lerp(styles, cluster, strength)

    if repeat:
        styles = einops.repeat(styles, "n d c -> (n t) d c", t=keypoints.size(1))
        keypoints = pack_keypoints(keypoints_repeat)

    return styles, keypoints


@torch.no_grad()
def write_video(path: str, video: torch.Tensor, fps: int = 30):
    assert path.endswith(".mp4")
    assert video.dim() == 4
    assert video.size(1) == 3

    fourcc = cv2.VideoWriter_fourcc(*"MP4V")
    writer = cv2.VideoWriter(
        path, fourcc, fps=fps, frameSize=(video.size(3), video.size(2))
    )

    tensor = einops.rearrange(video, "t c h w -> t h w c")
    tensor = data.to_uint8(tensor.cpu())
    tensor = tensor[..., [2, 1, 0]]

    for frame in tensor:
        writer.write(np.array(frame).astype(np.uint8))

    writer.release()
