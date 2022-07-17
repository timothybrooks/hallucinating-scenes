from __future__ import annotations

__all__ = ["AugmentImageAndKeypoints"]

from typing import Optional

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers import nvidia_ops


class AugmentImageAndKeypoints(nn.Module):
    def __init__(
        self,
        enabled: bool = True,
        brightness: tuple[float, float] = (-0.5, 0.5),
        saturation: tuple[float, float] = (0.0, 2.0),
        contrast: tuple[float, float] = (0.5, 1.5),
        flip_prob: float = 0.5,
        scale: float = 1.25,
        translate: float = 0.125,
        erasing_ratio: float = 0.5,
        pose_dropout: float = 0.0,
    ):
        super().__init__()
        self.enabled = enabled

        self.color = RandomColor(brightness, saturation, contrast)
        self.flip = RandomFlipHorizontal(flip_prob)
        self.transform = RandomScaleTranslate(scale, translate)
        self.erasing = RandomErasing(erasing_ratio)
        self.pose_dropout = PoseDropout(pose_dropout)

    def forward(
        self, frames: torch.Tensor, keypoints: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:

        assert frames.dim() == 5
        assert keypoints.dim() == 4

        if self.enabled:
            frames = self.color(frames)
            frames, keypoints = self.flip(frames, keypoints)
            frames, keypoints = self.transform(frames, keypoints)
            frames = self.erasing(frames)
            keypoints = self.pose_dropout(keypoints)

        return frames, keypoints


class RandomColor(nn.Module):
    def __init__(
        self,
        brightness: tuple[float, float] = (-0.5, 0.5),
        saturation: tuple[float, float] = (0.0, 2.0),
        contrast: tuple[float, float] = (0.5, 1.5),
    ):
        super().__init__()
        self.brightness = brightness
        self.saturation = saturation
        self.contrast = contrast

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        assert frames.dim() == 5
        assert frames.size(2) == 3

        size = (frames.size(0), 1, 1, 1, 1)

        brightness = frames.new_empty(size).uniform_(*self.brightness)
        frames = frames + brightness

        frames_gray = frames.mean(dim=2, keepdim=True)
        saturation = frames.new_empty(size).uniform_(*self.saturation)
        frames = torch.lerp(frames_gray, frames, saturation)

        frames_avg = frames.mean(dim=(1, 2, 3, 4), keepdim=True)
        contrast = frames.new_empty(size).uniform_(*self.contrast)
        frames = torch.lerp(frames_avg, frames, contrast)

        return frames


class RandomFlipHorizontal(nn.Module):
    def __init__(self, flip_prob: float = 0.5):
        super().__init__()
        self.flip_prob = flip_prob

    def forward(
        self, frames: torch.Tensor, keypoints: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:

        frames_flip = frames.flip(dims=(4,))

        keypoints_x = keypoints[:, :, :, 0]
        keypoints_x_flip = frames.size(4) - keypoints_x - 1

        size = (frames.size(0), 1, 1)
        mask = frames.new_empty(size).uniform_() < self.flip_prob

        keypoints_x = torch.where(mask, keypoints_x_flip, keypoints_x)
        keypoints = torch.cat(
            (keypoints_x[:, :, :, None], keypoints[:, :, :, 1:3]), dim=3
        )

        frames = torch.where(mask[:, :, :, None, None], frames_flip, frames)

        return frames, keypoints


class RandomScaleTranslate(nn.Module):
    def __init__(self, scale: float = 1.25, translate: float = 0.125):
        assert scale >= 1
        assert translate >= 0
        super().__init__()

        self.scale = scale
        self.translate = translate

    def forward(
        self, frames: torch.Tensor, keypoints: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:

        assert frames.dim() == 5
        assert frames.size(2) == 3
        assert keypoints.dim() == 4
        assert keypoints.size(0) == frames.size(0)
        assert keypoints.size(3) == 3

        scale, translate_x, translate_y = self._sample_params(
            frames.size(0), frames.device
        )

        frames = self._transform_frames(frames, scale, translate_x, translate_y)

        keypoints = self._transform_keypoints(
            keypoints,
            scale,
            translate_x,
            translate_y,
            frames.size(3),
            frames.size(4),
        )

        return frames, keypoints

    @torch.no_grad()
    def _sample_params(
        self, batch_size: int, device: Optional[torch.device]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        scale = torch.empty(batch_size, device=device)
        scale = scale.uniform_(1 / self.scale, self.scale)

        translate_x = torch.empty(batch_size, device=device)
        translate_x = translate_x.uniform_(-self.translate, self.translate)

        translate_y = torch.empty(batch_size, device=device)
        translate_y = translate_y.uniform_(-self.translate, self.translate)

        return scale, translate_x, translate_y

    @staticmethod
    def _transform_frames(
        frames: torch.Tensor,
        scale: torch.Tensor,
        translate_x: torch.Tensor,
        translate_y: torch.Tensor,
    ) -> torch.Tensor:

        num_frames = frames.size(1)
        frames = einops.rearrange(frames, "n t c h w -> (n t) c h w")

        scale = einops.repeat(scale, "n -> (n t)", t=num_frames)
        translate_x = einops.repeat(translate_x, "n -> (n t)", t=num_frames)
        translate_y = einops.repeat(translate_y, "n -> (n t)", t=num_frames)

        with torch.no_grad():
            matrix = frames.new_zeros((scale.size(0), 2, 3))

            matrix[:, 0, 0] = 1 / scale
            matrix[:, 1, 1] = 1 / scale
            matrix[:, 0, 2] = -translate_x * 2
            matrix[:, 1, 2] = -translate_y * 2

            grid = F.affine_grid(
                matrix, list(frames.size()), align_corners=False
            )

        grid = grid.to(frames.dtype)
        frames = nvidia_ops.grid_sample(frames, grid)
        frames = einops.rearrange(
            frames, "(n t) c h w -> n t c h w", t=num_frames
        )
        return frames

    @staticmethod
    @torch.no_grad()
    def _transform_keypoints(
        keypoints: torch.Tensor,
        scale: torch.Tensor,
        translate_x: torch.Tensor,
        translate_y: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:

        scale = scale[:, None, None]
        translate_x = translate_x[:, None, None] * width
        translate_y = translate_y[:, None, None] * height

        keypoints_x = keypoints[:, :, :, 0]
        keypoints_x -= width / 2
        keypoints_x *= scale
        keypoints_x += width / 2 + translate_x

        keypoints_y = keypoints[:, :, :, 1]
        keypoints_y -= height / 2
        keypoints_y *= scale
        keypoints_y += height / 2 + translate_y

        keypoints = torch.stack(
            (keypoints_x, keypoints_y, keypoints[:, :, :, 2]), dim=3
        )

        mask_x = (keypoints_x >= 0) & (keypoints_x < width)
        mask_y = (keypoints_y >= 0) & (keypoints_y < height)
        mask = mask_x & mask_y

        keypoints *= mask[:, :, :, None]
        return keypoints


class RandomErasing(nn.Module):
    def __init__(self, erasing_ratio: float = 0.5):
        super().__init__()
        self.erasing_ratio = erasing_ratio

    def forward(self, frames: torch.Tensor) -> torch.Tensor:

        num_frames = frames.size(1)
        frames = einops.rearrange(frames, "n t c h w -> (n t) c h w")

        cutout_size = (
            int(frames.size(2) * self.erasing_ratio + 0.5),
            int(frames.size(3) * self.erasing_ratio + 0.5),
        )
        offset_x = torch.randint(
            0,
            frames.size(2) + (1 - cutout_size[0] % 2),
            size=[frames.size(0), 1, 1],
            device=frames.device,
        )
        offset_y = torch.randint(
            0,
            frames.size(3) + (1 - cutout_size[1] % 2),
            size=[frames.size(0), 1, 1],
            device=frames.device,
        )
        grid_batch, grid_x, grid_y = torch.meshgrid(
            torch.arange(frames.size(0), device=frames.device),
            torch.arange(cutout_size[0], device=frames.device),
            torch.arange(cutout_size[1], device=frames.device),
        )
        grid_x = torch.clamp(
            grid_x + offset_x - cutout_size[0] // 2,
            min=0,
            max=frames.size(2) - 1,
        )
        grid_y = torch.clamp(
            grid_y + offset_y - cutout_size[1] // 2,
            min=0,
            max=frames.size(3) - 1,
        )
        mask = frames.new_ones((frames.size(0), frames.size(2), frames.size(3)))
        mask[grid_batch, grid_x, grid_y] = 0

        frames = frames * mask.unsqueeze(1)
        frames = einops.rearrange(
            frames, "(n t) c h w -> n t c h w", t=num_frames
        )
        return frames


class PoseDropout(nn.Module):
    def __init__(self, pose_dropout: float = 0.0):
        super().__init__()
        self.pose_dropout = pose_dropout

    def forward(self, keypoints: torch.Tensor) -> torch.Tensor:
        mask = keypoints.new_empty(*keypoints.shape[:3], 1)
        mask = mask.uniform_() >= self.pose_dropout
        keypoints = mask * keypoints
        return keypoints
