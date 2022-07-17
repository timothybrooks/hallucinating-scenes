from __future__ import annotations

__all__ = ["ToRGB", "StyleConv2d"]

import math
from typing import Optional

import torch
import torch.nn as nn

from . import layers, nvidia_ops


class StyleConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        style_dim: int,
        resolution: int,
        kernel_size: int = 3,
        upsample: int = 1,
        spatial_noise: bool = False,
        activation: str = "lrelu",
        activation_clamp: Optional[float] = None,
    ):
        super().__init__()
        self.resolution = resolution
        self.upsample = upsample
        self.spatial_noise = spatial_noise
        self.activation = activation
        self.activation_clamp = activation_clamp

        self.register_buffer(
            "resample_filter", nvidia_ops.setup_filter((1, 3, 3, 1))
        )
        self.padding = kernel_size // 2
        self.activation_gain = nvidia_ops.activation_funcs[activation].def_gain

        self.affine = layers.FullyConnected(
            style_dim, in_channels, bias_init=1.0
        )

        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))

        if spatial_noise:
            self.register_buffer(
                "constant_noise", torch.randn((resolution, resolution))
            )
            self.noise_strength = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        input: torch.Tensor,
        styles: torch.Tensor,
        noise_mode: str = "random",
        gain: float = 1.0,
    ):
        assert noise_mode in ("random", "constant", "none")

        styles = self.affine(styles)

        if self.spatial_noise and noise_mode != "none":

            if noise_mode == "random":
                batch, _, height, width = input.size()
                height *= self.upsample
                width *= self.upsample

                noise = input.new_empty(batch, 1, height, width).normal_()
                noise = noise * self.noise_strength
            else:
                noise = self.constant_noise * self.noise_strength
        else:
            noise = None

        flip_weight = self.upsample == 1

        output = _modulated_conv2d(
            input,
            self.weight,
            styles,
            noise,
            self.upsample,
            padding=self.padding,
            resample_filter=self.resample_filter,
            flip_weight=flip_weight,
        )

        bias = self.bias.to(output.dtype)

        activation_gain = self.activation_gain * gain

        if self.activation_clamp is not None:
            activation_clamp = self.activation_clamp * gain
        else:
            activation_clamp = None

        output = nvidia_ops.bias_act(
            output,
            bias,
            act=self.activation,
            gain=activation_gain,
            clamp=activation_clamp,
        )
        return output


class ToRGB(nn.Module):
    def __init__(
        self,
        in_channels: int,
        style_dim: int = 512,
        activation_clamp: Optional[float] = None,
    ):
        super().__init__()
        self.activation_clamp = activation_clamp

        self.affine = layers.FullyConnected(
            style_dim, in_channels, bias_init=1.0
        )

        self.weight = nn.Parameter(torch.randn(3, in_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(3))

        self.register_buffer(
            "resample_filter", nvidia_ops.setup_filter((1, 3, 3, 1))
        )

        self.weight_gain = 1 / math.sqrt(in_channels)

    def forward(
        self,
        input: torch.Tensor,
        styles: torch.Tensor,
        prev_image: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        styles = self.affine(styles) * self.weight_gain

        output = _modulated_conv2d(
            input, self.weight, styles, demodulate=False,
        )

        bias = self.bias.to(output.dtype)
        image = nvidia_ops.bias_act(output, bias, clamp=self.activation_clamp)

        if prev_image is not None:
            prev_image = nvidia_ops.upsample2d(prev_image, self.resample_filter)
            image = image + prev_image

        return image


def _modulated_conv2d(
    x,  # Input tensor of shape [batch_size, in_channels, in_height, in_width].
    weight,  # Weight tensor of shape [out_channels, in_channels, kernel_height, kernel_width].
    styles,  # Modulation coefficients of shape [batch_size, in_channels].
    noise=None,  # Optional noise tensor to add to the output activations.
    up=1,  # Integer upsampling factor.
    down=1,  # Integer downsampling factor.
    padding=0,  # Padding with respect to the upsampled image.
    resample_filter=None,  # Low-pass filter to apply when resampling activations. Must be prepared beforehand by calling upfirdn2d.setup_filter().
    demodulate=True,  # Apply weight demodulation?
    flip_weight=True,  # False = convolution, True = correlation (matches torch.nn.functional.conv2d).
):
    batch_size = x.shape[0]
    out_channels, in_channels, kh, kw = weight.size()

    # Pre-normalize inputs to avoid FP16 overflow.
    if x.dtype == torch.float16 and demodulate:
        weight = weight * (
            1
            / math.sqrt(in_channels * kh * kw)
            / weight.norm(float("inf"), dim=[1, 2, 3], keepdim=True)
        )  # max_Ikk
        styles = styles / styles.norm(
            float("inf"), dim=1, keepdim=True
        )  # max_I

    # Calculate per-sample weights and demodulation coefficients.
    w = None
    dcoefs = None
    if demodulate:
        w = weight.unsqueeze(0)  # [NOIkk]
        w = w * styles.reshape(batch_size, 1, -1, 1, 1)  # [NOIkk]
        dcoefs = (w.square().sum(dim=[2, 3, 4]) + 1e-8).rsqrt()  # [NO]

    x = x * styles.to(x.dtype).reshape(batch_size, -1, 1, 1)
    x = nvidia_ops.conv2d_resample(
        x=x,
        w=weight.to(x.dtype),
        f=resample_filter,
        up=up,
        down=down,
        padding=padding,
        flip_weight=flip_weight,
    )
    if demodulate and noise is not None:
        x = nvidia_ops.fma(
            x,
            dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1),
            noise.to(x.dtype),
        )
    elif demodulate:
        x = x * dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1)
    elif noise is not None:
        x = x.add_(noise.to(x.dtype))
    return x
