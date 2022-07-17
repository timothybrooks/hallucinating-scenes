from __future__ import annotations

__all__ = ["HumansDataset", "HumansDatasetSubset"]

import os
import pathlib
from dataclasses import dataclass
from typing import ClassVar, Union, Optional

import torch
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset
import bisect

from . import utils
from .database import Database, ImageDatabase


@dataclass
class HumansDataset(ConcatDataset):
    path: Union[str, os.PathLike[str]]
    resolution: int = 128
    num_frames: int = 8
    spacing: int = 4
    deterministic: bool = False
    split: Optional[str] = "train"
    num_keypoints: ClassVar[int] = 18
    test_length: ClassVar[int] = 12800
    seed: ClassVar[int] = 10000
    return_path: bool = False

    def __post_init__(self):
        assert self.split in (None, "train", "test")

        path = pathlib.Path(self.path)
        datasets = []
        self.subsets = []

        for subset_path in path.iterdir():
            self.subsets.append(subset_path.name)

            dataset = HumansDatasetSubset(
                subset_path, self.resolution, self.num_frames, self.spacing, self.deterministic, self.return_path,
            )
            datasets.append(dataset)

        self.subsets.sort()
        super().__init__(datasets)

        length = self.cumulative_sizes[-1]
        generator = torch.Generator().manual_seed(self.seed)
        self.indices = torch.randperm(length, generator=generator).tolist()

        if self.split == "test":
            self.indices = self.indices[: self.test_length]

        elif self.split == "train":
            self.indices = self.indices[self.test_length :]

    def __getitem__(self, index: int):
        return super().__getitem__(self.indices[index])

    def __len__(self) -> int:
        return len(self.indices)

    def sample(self, seed: int):
        generator = torch.Generator().manual_seed(seed)
        index = torch.randint(len(self), (), generator=generator).item()
        index = self.indices[index]  # type: ignore

        dataset_idx = bisect.bisect_right(self.cumulative_sizes, index)
        if dataset_idx == 0:
            sample_idx = index
        else:
            sample_idx = index - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx].sample(sample_idx, generator)  # type: ignore


@dataclass
class HumansDatasetSubset(Dataset):
    path: Union[str, os.PathLike[str]]
    resolution: int = 128
    num_frames: int = 8
    spacing: int = 4
    deterministic: bool = False
    num_keypoints: ClassVar[int] = 18
    return_path: bool = False

    def __post_init__(self):
        path = pathlib.Path(self.path)

        frames_db_path = str(path.joinpath("frames_db"))
        self.frames_db = ImageDatabase(frames_db_path, readahead=False, lock=False)

        poses_db_path = str(path.joinpath("poses_db"))
        self.poses_db = Database(poses_db_path, readahead=False, lock=False)

        clips_db_path = str(path.joinpath("clips_db"))
        clips_db = Database(clips_db_path, lock=False)

        self.keys = []
        self.min_length = self.spacing * (self.num_frames - 1) + 1

        clip_keys = clips_db.keys()
        for clip_key in clip_keys:

            frame_keys = clips_db[clip_key]
            if len(frame_keys) >= self.min_length:
                self.keys.append(frame_keys)

        if self.deterministic:
            self.seed = int.from_bytes(path.name.encode(), byteorder="big")
            self.seed &= 0xFFFFFFFF
        else:
            self.seed = None

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        frame_keys = self.keys[index]

        if self.return_path:
            return self.path, frame_keys

        if self.seed is not None:
            self.generator = torch.Generator().manual_seed(self.seed)
        else:
            self.generator = None

        num_extra_frames = len(frame_keys) - self.min_length
        index_start = torch.randint(0, num_extra_frames + 1, (), generator=self.generator).item()
        index_end = index_start + self.min_length

        frame_keys = frame_keys[index_start : index_end : self.spacing]

        frames = []
        poses = []

        for frame_key in frame_keys:

            frame = self.frames_db[frame_key]
            assert frame.height == frame.width

            pose = self.poses_db[frame_key][0].float()

            if frame.height != self.resolution:
                pose[:, :2] *= self.resolution / frame.height

                size = (self.resolution, self.resolution)
                frame = frame.resize(size, resample=Image.LANCZOS)

            frame = utils.to_tensor(frame)
            frames.append(frame)
            poses.append(pose)

        # print(self.path, frame_keys)

        frames = torch.stack(frames)
        poses = torch.stack(poses)
        return frames, poses

    def __len__(self):
        return len(self.keys)

    def sample(self, index, generator):
        frame_keys = self.keys[index]
        frame_index = torch.randint(len(frame_keys), (), generator=generator).item()
        frame = self.frames_db[frame_keys[frame_index]]
        pose = self.poses_db[frame_keys[frame_index]][0].float()

        if frame.height != self.resolution:
            pose[:, :2] *= self.resolution / frame.height

            size = (self.resolution, self.resolution)
            frame = frame.resize(size, resample=Image.LANCZOS)

        frame = utils.to_tensor(frame)
        return frame[None], pose[None]

