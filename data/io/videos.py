__all__ = ["write_video"]

import torch
import torchvision.io as io

from .. import utils


def write_video(video: torch.Tensor, path: str, fps: int = 30):
    video = utils.to_uint8(video)
    video = video.permute(0, 2, 3, 1)
    io.write_video(path, video.cpu(), fps=fps)
