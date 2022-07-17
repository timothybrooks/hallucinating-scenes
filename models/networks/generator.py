from __future__ import annotations

__all__ = ["Generator", "MappingNetwork", "SynthesisNetwork"]

import math
from dataclasses import dataclass
from typing import Optional

import einops
import torch
import torch.nn as nn
import utils

from .. import layers


@dataclass(eq=False)
class MappingNetwork(nn.Module):
    num_styles: int
    num_keypoints: int
    num_poses: int
    style_dim: int = 512
    noise_dim: int = 512
    keypoint_embed_dim: int = 512
    hidden_dim: int = 512
    num_layers: int = 8
    lr_multiplier: float = 0.01

    def __post_init__(self):
        super().__init__()

        if self.keypoint_embed_dim > 0:
            self.embed_pose = layers.FullyConnected(
                self.num_poses * self.num_keypoints * 3,
                self.keypoint_embed_dim,
                lr_multiplier=self.lr_multiplier,
            )
        else:
            self.embed_pose = None

        self.layers = nn.ModuleList()

        for i in range(self.num_layers):

            input_dim = self.hidden_dim
            output_dim = self.hidden_dim

            if i == 0:
                input_dim = self.noise_dim + self.keypoint_embed_dim

            if i == self.num_layers - 1:
                output_dim = self.style_dim

            self.layers.append(
                layers.FullyConnected(
                    input_dim,
                    output_dim,
                    activation="lrelu",
                    lr_multiplier=self.lr_multiplier,
                )
            )

    def sample_noise(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:

        noise = torch.randn(
            batch_size, self.noise_dim, device=device, generator=generator
        )
        return noise

    def sample_styles(
        self,
        keypoints: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ):
        if noise is None:
            batch_size = keypoints.size(0)
            noise = self.sample_noise(batch_size, keypoints.device, generator)

        features = _normalize_2nd_moment(noise)

        if self.embed_pose is not None:
            keypoints = einops.rearrange(keypoints, "n t k c -> n (t k c)")
            pose_embedding = self.embed_pose(keypoints)
            pose_embedding /= math.sqrt(self.keypoint_embed_dim)
            features = torch.cat((features, pose_embedding), dim=1)

        for layer in self.layers:
            features = layer(features)

        styles = einops.repeat(features, "n c -> n d c", d=self.num_styles)
        return styles

    def forward(
        self, keypoints: torch.Tensor, mixing: float = 0.0,
    ) -> torch.Tensor:

        styles = self.sample_styles(keypoints)

        if mixing > 0:

            mixing_index = torch.randint(
                1, self.num_styles - 1, (), device=styles.device
            )
            mixing_styles = self.sample_styles(keypoints)
            mixing_styles[:, :mixing_index] = styles[:, :mixing_index]

            condition = torch.rand((), device=styles.device) < mixing
            styles = torch.where(condition, mixing_styles, styles)

        return styles


@dataclass(eq=False)
class _SynthesisBlock(nn.Module):
    in_channel: int
    out_channel: int
    resolution: int
    style_dim: int = 512
    spatial_noise: bool = False
    activation_clamp: Optional[float] = None

    def __post_init__(self):
        super().__init__()

        self.style_conv_0 = layers.StyleConv2d(
            self.in_channel,
            self.out_channel,
            self.style_dim,
            self.resolution,
            upsample=2,
            spatial_noise=self.spatial_noise,
            activation_clamp=self.activation_clamp,
        )

        self.style_conv_1 = layers.StyleConv2d(
            self.out_channel,
            self.out_channel,
            self.style_dim,
            self.resolution,
            spatial_noise=self.spatial_noise,
            activation_clamp=self.activation_clamp,
        )

        self.to_image = layers.ToRGB(
            self.out_channel,
            self.style_dim,
            activation_clamp=self.activation_clamp,
        )

    def forward(
        self, frames: torch.Tensor, hidden: torch.Tensor, style: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:

        hidden = self.style_conv_0(hidden, style)
        hidden = self.style_conv_1(hidden, style)
        frames = self.to_image(hidden, style, frames)
        return frames, hidden


@dataclass(eq=False)
class SynthesisNetwork(nn.Module):
    resolution: int
    num_keypoints: int
    style_dim: int = 512
    base_channels: int = 32768
    max_channels: int = 1024
    spatial_noise: bool = False
    keypoint_heatmaps: bool = True
    limb_heatmaps: bool = False
    mixed_precision: bool = False
    activation_clamp: Optional[float] = None

    def __post_init__(self):
        super().__init__()
        self.log_size = int(math.log2(self.resolution))
        self.num_layers = (self.log_size - 2) * 2 + 1

        channels_dict = {}

        num_blocks = self.log_size - 1
        for block_index in range(num_blocks):

            block_resolution = 2 ** (2 + block_index)
            num_channels = self.base_channels // block_resolution
            num_channels = min(num_channels, self.max_channels)
            channels_dict[block_resolution] = num_channels

        self.input = nn.Parameter(torch.randn(1, channels_dict[4], 4, 4))

        self.conv1 = layers.StyleConv2d(
            channels_dict[4],
            channels_dict[4],
            self.style_dim,
            resolution=4,
            spatial_noise=self.spatial_noise,
            activation_clamp=self.activation_clamp,
        )
        self.to_rgb1 = layers.ToRGB(
            channels_dict[4],
            self.style_dim,
            activation_clamp=self.activation_clamp,
        )

        self.blocks = nn.ModuleList()

        in_channels = channels_dict[4]

        for i in range(3, self.log_size + 1):
            block_resolution = 2 ** i
            out_channels = channels_dict[block_resolution]

            if self.keypoint_heatmaps:
                in_channels += self.num_keypoints

            if self.limb_heatmaps:
                num_limbs = self.num_keypoints - 1
                in_channels += num_limbs

            self.blocks.append(
                _SynthesisBlock(
                    in_channels,
                    out_channels,
                    block_resolution,
                    self.style_dim,
                    self.spatial_noise,
                    self.activation_clamp,
                )
            )

            in_channels = out_channels

    def forward(
        self,
        styles: torch.Tensor,
        keypoints: torch.Tensor,
        zero_constant: bool = False,
    ) -> torch.Tensor:

        hidden = self.input.repeat(styles.size(0), 1, 1, 1)
        hidden = self.conv1(hidden, styles[:, 0])
        frames = self.to_rgb1(hidden, styles[:, 0])

        if zero_constant:
            hidden = hidden * 0
            frames = frames * 0

        dtype = torch.half if self.mixed_precision else torch.float

        styles = styles.to(dtype)
        keypoints = keypoints.to(dtype)
        hidden = hidden.to(dtype)
        frames = frames.to(dtype)

        for i, block in enumerate(self.blocks, start=1):

            if self.keypoint_heatmaps or self.limb_heatmaps:
                factor = hidden.size(2) / self.resolution
                block_keypoints = keypoints * factor

            if self.keypoint_heatmaps:
                pose_heatmap = utils.keypoint_heatmaps(
                    block_keypoints, hidden.size(2), hidden.size(3)  # type: ignore
                )
                hidden = torch.cat((hidden, pose_heatmap), dim=1)

            if self.limb_heatmaps:
                pose_heatmap = utils.limb_heatmaps(
                    block_keypoints, hidden.size(2), hidden.size(3)  # type: ignore
                )
                hidden = torch.cat((hidden, pose_heatmap), dim=1)

            frames, hidden = block(frames, hidden, styles[:, i])

        return frames


@dataclass(eq=False)
class Generator(nn.Module):
    resolution: int
    num_keypoints: int
    num_poses: int
    style_dim: int = 512
    noise_dim: int = 512
    keypoint_embed_dim: int = 512
    mapping_hidden_dim: int = 512
    mapping_num_layers: int = 8
    mapping_lr_multiplier: float = 0.01
    base_channels: int = 32768
    max_channels: int = 1024
    spatial_noise: bool = False
    keypoint_heatmaps: bool = True
    limb_heatmaps: bool = False
    mixed_precision: bool = False
    activation_clamp: Optional[float] = None

    def __post_init__(self):
        super().__init__()

        self.synthesis_network = SynthesisNetwork(
            self.resolution,
            self.num_keypoints,
            self.style_dim,
            self.base_channels,
            self.max_channels,
            self.spatial_noise,
            self.keypoint_heatmaps,
            self.limb_heatmaps,
            self.mixed_precision,
            self.activation_clamp,
        )

        num_styles = len(self.synthesis_network.blocks) + 1

        self.mapping_network = MappingNetwork(
            num_styles,
            self.num_keypoints,
            self.num_poses,
            self.style_dim,
            self.noise_dim,
            self.keypoint_embed_dim,
            self.mapping_hidden_dim,
            self.mapping_num_layers,
            self.mapping_lr_multiplier,
        )

    def forward(
        self,
        keypoints: torch.Tensor,
        frame_keypoints: torch.Tensor,
        mixing: float = 0.0,
    ) -> torch.Tensor:

        assert keypoints.dim() == 4
        assert frame_keypoints.dim() == 4

        styles = self.mapping_network(keypoints, mixing)

        num_frames = frame_keypoints.size(1)
        styles = einops.repeat(styles, "n s c -> (n t) s c", t=num_frames)

        frame_keypoints = einops.rearrange(
            frame_keypoints, "n t k c -> (n t) k c"
        )

        frames = self.synthesis_network(styles, frame_keypoints)
        frames = einops.rearrange(
            frames, "(n t) c h w -> n t c h w", t=num_frames
        )
        return frames


# FIXME: Replace with F.normalize()?
def _normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()
