from __future__ import annotations

import os
import pickle
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.io as io
import tqdm

from .. import utils

__all__ = ["ImageFolder", "read_image", "write_image"]


class ImageFolder(data.Dataset):
    def __init__(self, folder_path: str, resolution: Optional[int] = None):
        self.folder_path = folder_path
        self.resolution = resolution

        list_path = os.path.join(folder_path, "image_paths.pkl")

        if os.path.exists(list_path):
            with open(list_path, "rb") as open_file:
                self.paths = pickle.load(open_file)
        else:
            self.paths = _list_image_paths(folder_path)

            with open(list_path, "wb") as open_file:
                pickle.dump(self.paths, open_file)

            print("Saved list of image file paths.")

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, float]:
        path = os.path.join(self.folder_path, self.paths[index])

        if self.resolution is not None:
            image, _, _ = utils.read_and_resize_image(
                path, self.resolution, letterbox=True
            )

            height, width = image.shape[1:3]

            pad_x0 = (self.resolution - width) // 2
            pad_x1 = self.resolution - width - pad_x0
            pad_y0 = (self.resolution - height) // 2
            pad_y1 = self.resolution - height - pad_y0

            padding = (pad_x0, pad_x1, pad_y0, pad_y1)
            image = F.pad(image, padding)

            scale = max(width / height, height / width)
        else:
            image = read_image(path)
            scale = 1.0

        return image, scale

    def __len__(self) -> int:
        return len(self.paths)


def read_image(path: os.AnyPath) -> torch.Tensor:
    if not _has_image_file_extension(path):
        raise ValueError(f"Not a valid image path: {path}")

    image = io.read_image(str(path))
    image = utils.to_float(image)
    return image


def write_image(image: torch.Tensor, path: os.AnyPath):
    if not _has_image_file_extension(path):
        raise ValueError(f"Not a valid image path: {path}")

    image = utils.to_uint8(image)

    path = str(path)

    if path.endswith("png"):
        io.write_png(image, path)
    else:
        io.write_jpeg(image, path)


# ==============================================================================
# Private functions.
# ==============================================================================


def _list_image_paths(root_dir: str) -> List[str]:
    if not os.path.isdir(root_dir):
        raise ValueError(f"Directory not found: {root_dir}")

    walk_progress_bar = tqdm.tqdm(desc="Listing all image file paths")

    image_paths = []
    directories = [root_dir]

    while directories:

        directory = directories.pop()
        for entry in os.scandir(directory):

            if entry.is_dir(follow_symlinks=False):
                directories.append(entry.path)

            elif _has_image_file_extension(entry.name):
                image_path = os.path.relpath(entry.path, root_dir)
                image_paths.append(image_path)
                walk_progress_bar.update(1)

    walk_progress_bar.close()

    image_paths.sort()
    return image_paths


def _has_image_file_extension(file_path: os.AnyPath) -> bool:
    image_extensions = (".jpg", ".jpeg", ".png")
    return str(file_path).lower().endswith(image_extensions)
