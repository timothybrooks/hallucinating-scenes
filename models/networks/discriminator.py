__all__ = ["KeypointDiscriminator"]

import math
from dataclasses import dataclass
from typing import Optional

import einops
import einops.layers.torch as einop_layers
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

from .. import layers


@dataclass(eq=False)
class _MappingNetwork(nn.Module):
    num_keypoints: int
    num_poses: int
    hidden_dim: int = 512
    out_dim: int = 512
    num_layers: int = 8
    lr_multiplier: float = 0.01

    def __post_init__(self):
        super().__init__()

        self.embed_pose = layers.FullyConnected(
            self.num_poses * self.num_keypoints * 3,
            self.hidden_dim,
            lr_multiplier=self.lr_multiplier,
        )

        self.layers = nn.ModuleList()

        for i in range(self.num_layers):

            if i == self.num_layers - 1:
                output_dim = self.out_dim
            else:
                output_dim = self.hidden_dim

            self.layers.append(
                layers.FullyConnected(
                    self.hidden_dim,
                    output_dim,
                    activation="lrelu",
                    lr_multiplier=self.lr_multiplier,
                )
            )

    def forward(self, keypoints: torch.Tensor) -> torch.Tensor:
        keypoints = einops.rearrange(keypoints, "n t k c -> n (t k c)")
        features = self.embed_pose(keypoints)
        features /= math.sqrt(self.hidden_dim)

        for layer in self.layers:
            features = layer(features)

        return features


class _DiscriminatorBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation_clamp: Optional[float] = None,
    ):
        super().__init__()

        self.conv_0 = layers.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            activation="lrelu",
            activation_clamp=activation_clamp,
        )
        self.conv_1 = layers.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            activation="lrelu",
            downsample=2,
            activation_clamp=activation_clamp,
        )
        self.skip = layers.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False, downsample=2,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.conv_0(input)
        output = self.conv_1(output, gain=math.sqrt(0.5))

        skip = self.skip(input, gain=math.sqrt(0.5))
        output = output + skip
        return output


class _Discriminator(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int = 3,
        out_channels: int = 1,
        stddev_groups: int = 4,
        stddev_channels: int = 1,
        base_channels: int = 32768,
        max_channels: int = 512,
        mixed_precision: bool = True,
        activation_clamp: Optional[float] = None,
    ):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.stddev_groups = stddev_groups
        self.stddev_channels = stddev_channels
        self.mixed_precision = mixed_precision

        log2_res = int(math.log2(resolution))
        block_resolutions = [2 ** i for i in range(2, log2_res + 1)]
        channels_dict = {
            block_res: min(base_channels // block_res, max_channels)
            for block_res in block_resolutions
        }

        self.from_rgb = layers.Conv2d(
            in_channels,
            channels_dict[resolution],
            kernel_size=1,
            activation="lrelu",
            activation_clamp=activation_clamp,
        )

        self.blocks = nn.ModuleList()

        log_size = int(math.log(resolution, 2))

        in_channel = channels_dict[resolution]

        for i in range(log_size, 2, -1):
            out_channel = channels_dict[2 ** (i - 1)]

            self.blocks.append(
                _DiscriminatorBlock(in_channel, out_channel, activation_clamp)
            )

            in_channel = out_channel

        assert stddev_channels == 0 or in_channel % stddev_channels == 0

        self.predict = nn.Sequential(
            layers.Conv2d(
                in_channel + stddev_channels,
                channels_dict[4],
                kernel_size=3,
                activation="lrelu",
                activation_clamp=activation_clamp,
            ),
            einop_layers.Rearrange("n c h w -> n (c h w)"),
            layers.FullyConnected(
                channels_dict[4] * 4 * 4, channels_dict[4], activation="lrelu",
            ),
            layers.FullyConnected(channels_dict[4], out_channels),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # assert input.size(0) % self.stddev_groups == 0
        assert input.size(1) == self.in_channels
        assert input.size(2) == self.resolution
        assert input.size(3) == self.resolution

        dtype = torch.half if self.mixed_precision else torch.float
        input = input.to(dtype)

        hidden = self.from_rgb(input)
        for block in self.blocks:
            hidden = block(hidden)

        hidden = hidden.float()

        if self.stddev_channels > 0:
            stddev = self._stddev_feature(hidden)
            hidden = torch.cat((hidden, stddev), dim=1)

        output = self.predict(hidden)
        return output

    def loss_real(self, *args, skip_mean=False, **kwargs) -> torch.Tensor:
        prediction = self.forward(*args, **kwargs)
        if skip_mean:
            loss = F.softplus(-prediction)
        else:
            loss = F.softplus(-prediction).mean()
        return loss

    def loss_fake(self, *args, skip_mean=False, **kwargs) -> torch.Tensor:
        prediction = self.forward(*args, **kwargs)
        if skip_mean:
            loss = F.softplus(prediction)
        else:
            loss = F.softplus(prediction).mean()
        return loss

    def _stddev_feature(self, features: torch.Tensor) -> torch.Tensor:

        if features.size(0) % self.stddev_groups == 0:

            stddev = einops.reduce(
                features,
                "(g n) (f c) h w -> n f c h w",
                torch.std,
                g=self.stddev_groups,
                f=self.stddev_channels,
            )
            stddev = einops.reduce(stddev, "n f c h w -> n f", "mean")
            stddev = einops.repeat(
                stddev,
                "n f -> (g n) f h w",
                g=self.stddev_groups,
                h=features.size(2),
                w=features.size(3),
            )

        else:
            assert not self.training
            stddev = torch.full_like(features[:, :1], 4.4690)

        return stddev


class KeypointDiscriminator(_Discriminator):
    def __init__(
        self,
        resolution: int,
        num_keypoints: int,
        num_poses: int,
        stddev_groups: int = 4,
        stddev_channels: int = 1,
        base_channels: int = 32768,
        max_channels: int = 512,
        keypoint_mapping: bool = True,
        keypoint_heatmaps: bool = True,
        limb_heatmaps: bool = False,
        mixed_precision: bool = True,
        activation_clamp: Optional[float] = None,
    ):
        in_channels = 3
        if keypoint_heatmaps:
            in_channels += num_keypoints

        if limb_heatmaps:
            num_limbs = num_keypoints - 1
            in_channels += num_limbs

        out_channels = 512 if keypoint_mapping else 1

        super().__init__(
            resolution,
            in_channels,
            out_channels,
            stddev_groups,
            stddev_channels,
            base_channels,
            max_channels,
            mixed_precision,
            activation_clamp,
        )

        if keypoint_mapping:
            self.mapping = _MappingNetwork(
                num_keypoints, num_poses, out_channels, out_channels
            )
        else:
            self.mapping = None

        self.num_keypoints = num_keypoints
        self.num_poses = num_poses
        self.keypoint_heatmaps = keypoint_heatmaps
        self.limb_heatmaps = limb_heatmaps

    def forward(
        self,
        frames: torch.Tensor,
        keypoints: torch.Tensor,
        frame_keypoints: torch.Tensor,
    ) -> torch.Tensor:

        assert frames.dim() == 5
        assert frames.size(2) == 3

        assert keypoints.size(0) == frames.size(0)
        assert keypoints.size(1) == self.num_poses
        assert keypoints.size(2) == self.num_keypoints
        assert keypoints.size(3) == 3

        assert frame_keypoints.size(1) == frames.size(1)

        frame_keypoints = einops.repeat(frame_keypoints, "n t k c -> (n t) k c")
        input = einops.rearrange(frames, "n t c h w -> (n t) c h w")

        if self.keypoint_heatmaps:
            heatmaps = utils.keypoint_heatmaps(
                frame_keypoints, frames.size(3), frames.size(4)
            )
            input = torch.cat((input, heatmaps), dim=1)

        if self.limb_heatmaps:
            heatmaps = utils.limb_heatmaps(
                frame_keypoints, frames.size(3), frames.size(4)
            )
            input = torch.cat((input, heatmaps), dim=1)

        output = super().forward(input)

        if self.mapping is not None:
            conditioning = self.mapping(keypoints)
            output = (output * conditioning).sum(dim=1, keepdim=True)
            output = output / math.sqrt(self.mapping.out_dim)

        return output
