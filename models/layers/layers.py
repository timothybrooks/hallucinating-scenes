from __future__ import annotations

__all__ = ["FullyConnected", "Conv2d"]

import math
from typing import Optional

import torch
import torch.nn as nn

from . import nvidia_ops


class FullyConnected(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        activation: str = "linear",
        lr_multiplier: float = 1.0,
        bias_init: float = 0.0,
    ):
        super().__init__()
        self.activation = activation

        self.weight = nn.Parameter(
            torch.randn((out_dim, in_dim)) / lr_multiplier
        )

        if bias:
            self.bias = nn.Parameter(torch.full((out_dim,), bias_init))
        else:
            self.bias = None

        self.weight_gain = lr_multiplier / math.sqrt(in_dim)
        self.bias_gain = lr_multiplier

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = self.weight * self.weight_gain
        weight = weight.to(input.dtype)

        bias = self.bias
        if bias is not None:
            bias = bias.to(input.dtype)

            if self.bias_gain != 1.0:
                bias = bias * self.bias_gain

        if self.activation == "linear" and bias is not None:
            bias = None if bias is None else self.bias.to(input.dtype)
            output = torch.addmm(bias.unsqueeze(0), input, weight.t())

        else:
            output = input.matmul(weight.t())
            bias = None if bias is None else self.bias.to(output.dtype)
            output = nvidia_ops.bias_act(output, bias, act=self.activation)

        return output


class Conv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        bias: bool = True,
        activation: str = "linear",
        upsample: int = 1,
        downsample: int = 1,
        activation_clamp: Optional[float] = None,
    ):
        super().__init__()
        self.activation = activation
        self.upsample = upsample
        self.downsample = downsample
        self.activation_clamp = activation_clamp

        if upsample != 1 or downsample != 1:
            self.register_buffer(
                "resample_filter", nvidia_ops.setup_filter((1, 3, 3, 1))
            )

        else:
            self.resample_filter = None

        self.padding = kernel_size // 2
        self.weight_gain = 1 / math.sqrt(in_channels * (kernel_size ** 2))
        self.activation_gain = nvidia_ops.activation_funcs[activation].def_gain

        self.weight = nn.Parameter(
            torch.randn((out_channels, in_channels, kernel_size, kernel_size))
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

    def forward(self, input: torch.Tensor, gain: float = 1.0) -> torch.Tensor:

        weight = self.weight * self.weight_gain
        weight = weight.to(input.dtype)
        flip_weight = self.upsample == 1

        output = nvidia_ops.conv2d_resample(
            input,
            weight,
            self.resample_filter,
            self.upsample,
            self.downsample,
            self.padding,
            flip_weight=flip_weight,
        )

        if self.bias is not None:
            bias = self.bias.to(output.dtype)
        else:
            bias = None

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
