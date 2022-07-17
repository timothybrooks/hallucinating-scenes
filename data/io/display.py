__all__ = ["clear_display", "display_image"]

import sys

import torch
import torchvision.io as io

from .. import utils

# Automatically detects IPython vs. normal python.
# https://github.com/tqdm/tqdm/blob/master/tqdm/autonotebook.py
try:
    get_ipython = sys.modules["IPython"].get_ipython  # type: ignore
    if "IPKernelApp" not in get_ipython().config:
        raise ImportError("Not inside an IPython notebook.")
except:
    _IN_IPYTHON = False
else:
    _IN_IPYTHON = True
    from IPython import display


def clear_display(wait: bool = True):
    if not _IN_IPYTHON:
        _raise_not_ipython_error()

    display.clear_output(wait=wait)


def display_image(image: torch.Tensor):
    if not _IN_IPYTHON:
        _raise_not_ipython_error()

    image = utils.to_uint8(image)
    data = io.encode_jpeg(image.cpu())
    display_image = display.Image(bytes(data))
    display.display(display_image)


# ==============================================================================
# Private functions.
# ==============================================================================


def _raise_not_ipython_error():
    raise RuntimeError(
        "Cannot access display from outside an IPython notebook."
    )
