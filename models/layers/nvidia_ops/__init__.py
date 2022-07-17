# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# empty

from .ops.bias_act import activation_funcs, bias_act
from .ops.conv2d_gradfix import conv2d, conv_transpose2d, no_weight_gradients
from .ops.conv2d_resample import conv2d_resample
from .ops.fma import fma
from .ops.grid_sample_gradfix import grid_sample
from .ops.upfirdn2d import setup_filter, upsample2d
