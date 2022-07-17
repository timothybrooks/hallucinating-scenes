__all__ = ["identity_grid", "keypoint_heatmaps", "limb_heatmaps"]

from typing import Optional

import einops
import torch


def identity_grid(
    height: int,
    width: int,
    normalize: bool = True,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:

    line_y = torch.arange(height, dtype=dtype, device=device)
    line_x = torch.arange(width, dtype=dtype, device=device)

    if normalize:
        line_y = 2 * line_y / (height - 1) - 1
        line_x = 2 * line_x / (width - 1) - 1

    grid_y, grid_x = torch.meshgrid(line_y, line_x)
    grid = torch.stack([grid_x, grid_y])
    return grid


def keypoint_heatmaps(
    keypoints: torch.Tensor, height: int, width: int
) -> torch.Tensor:

    grid = identity_grid(
        height,
        width,
        normalize=False,
        dtype=keypoints.dtype,
        device=keypoints.device,
    )
    grid = einops.rearrange(grid, "c h w -> 1 1 c h w")

    keypoints = einops.rearrange(keypoints, "n k c -> n k c 1 1")
    visibility = keypoints[:, :, 2, :, :]
    keypoints = keypoints[:, :, :2, :, :]

    distance = grid - keypoints
    distance_sq = (distance ** 2).sum(dim=2)

    resolution = min(height, width)
    sigma = max(1.0, 0.1 * resolution)

    heatmaps = torch.exp(-distance_sq / (sigma ** 2))
    heatmaps *= visibility
    return heatmaps


def limb_heatmaps(
    keypoints: torch.Tensor, height: int, width: int
) -> torch.Tensor:

    limb_indices = torch.tensor(
        [
            [0, 1],
            [0, 14],
            [0, 15],
            [1, 2],
            [1, 5],
            [1, 8],
            [1, 11],
            [2, 3],
            [3, 4],
            [5, 6],
            [6, 7],
            [8, 9],
            [9, 10],
            [11, 12],
            [12, 13],
            [14, 16],
            [15, 17],
        ]
    )

    limbs = keypoints[:, limb_indices]
    visibility = limbs[:, :, :, 2].prod(dim=2)
    visibility = einops.rearrange(visibility, "n k -> n k 1 1")

    limbs = limbs[:, :, :, :2]
    limbs = einops.rearrange(limbs, "n k p c -> n k p c 1 1")

    sq_dist = (limbs[:, :, 0:1] - limbs[:, :, 1:2]) ** 2
    sq_dist = sq_dist.sum(dim=3, keepdim=True)
    sq_dist = sq_dist.clamp(min=1e-5)

    grid = identity_grid(
        height,
        width,
        normalize=False,
        dtype=keypoints.dtype,
        device=keypoints.device,
    )
    grid = einops.rearrange(grid, "c h w -> 1 1 1 c h w")

    displacement = limbs[:, :, 1:2] - limbs[:, :, 0:1]
    displacement_grid = grid - limbs[:, :, 0:1]

    position = (displacement * displacement_grid).sum(dim=3, keepdim=True)
    position /= sq_dist
    position = position.clamp(min=0.0, max=1.0)

    projection = limbs[:, :, 0:1] + position * displacement

    distance = grid - projection
    distance_sq = (distance ** 2).sum(dim=(2, 3))

    resolution = min(height, width)
    sigma = max(1.0, 0.1 * resolution)

    heatmaps = torch.exp(-distance_sq / (sigma ** 2))
    heatmaps *= visibility
    return heatmaps
