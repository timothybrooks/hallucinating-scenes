__all__ = ["draw_skeleton", "draw_openpose"]

import cv2
import einops
import numpy as np
import torch

import data

_INDICES_HUMAN_DYNAMICS = [-1, 0, 1, 2, 3, 1, 5, 6, 1, 8, 24, 1, 11, 23, 0, 0, 14, 15, -1, 23, 24, 19, 20, 12, 9]  # type: ignore
_INDICES_MPII = [1, 2, 6, 6, 3, 4, 7, 8, -1, 8, 11, 12, 8, 8, 13, 14]  # type: ignore
_INDICES_PENN_ACTION = [-1, -1, -1, 1, 2, 3, 4, -1, -1, 7, 8, 9, 10]  # type: ignore
_INDICES_COCO = [-1, 0, 0, 1, 2, -1, 5, 5, 6, 7, 8, 5, 6, 11, 12, 13, 14]  # type: ignore
_INDICES_OPENPOSE = [-1, 0, 1, 2, 3, 1, 5, 6, 1, 8, 9, 1, 11, 12, 0, 0, 14, 15]  # type: ignore
_COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]  # type: ignore


def draw_skeleton(image: torch.Tensor, keypoints: torch.Tensor) -> torch.Tensor:

    assert image.dim() == 4
    assert keypoints.dim() == 3
    assert keypoints.size(0) == image.size(0)
    assert keypoints.size(2) == 3

    if keypoints.size(1) == 25:
        parent_indices = _INDICES_HUMAN_DYNAMICS
    elif keypoints.size(1) == 16:
        parent_indices = _INDICES_MPII
    elif keypoints.size(1) == 13:
        parent_indices = _INDICES_PENN_ACTION
    elif keypoints.size(1) == 17:
        parent_indices = _INDICES_COCO
    elif keypoints.size(1) == 18:
        parent_indices = _INDICES_OPENPOSE
    else:
        raise ValueError(f"Invalid skeleton format.")

    image = data.to_uint8(image)
    image = einops.rearrange(image, "n c h w -> n h w c")

    numpy_keypoints = keypoints.round().long().cpu().numpy()

    radius = int(max(2, min(image.shape[2:4]) / 100))

    for batch_index in range(image.size(0)):

        numpy_image = image[batch_index].cpu().numpy().copy()

        for child_index in range(len(parent_indices)):

            child_visibility = numpy_keypoints[batch_index, child_index, 2]

            if child_visibility == 0:
                continue

            child_x, child_y = numpy_keypoints[batch_index, child_index, :2]

            cv2.circle(
                numpy_image,
                (child_x, child_y),
                radius,
                color=(255, 255, 255),
                thickness=-1,
                lineType=cv2.LINE_AA,
            )

            parent_index = parent_indices[child_index]

            if parent_index == -1:
                continue

            parent_visibility = numpy_keypoints[batch_index, parent_index, 2]

            if parent_visibility == 0:
                continue

            parent_x, parent_y = numpy_keypoints[batch_index, parent_index, :2]

            cv2.line(
                numpy_image,
                (child_x, child_y),
                (parent_x, parent_y),
                color=(255, 255, 255),
                thickness=radius - 1,
                lineType=cv2.LINE_AA,
            )

        image[batch_index] = torch.tensor(numpy_image).to(image.device)

    image = einops.rearrange(image, "n h w c -> n c h w")
    image = data.to_float(image)

    return image


def draw_openpose(image: torch.Tensor, keypoints: torch.Tensor) -> torch.Tensor:

    assert image.dim() == 4
    assert keypoints.dim() == 3
    assert keypoints.size(0) == image.size(0)
    assert keypoints.size(1) == 18
    assert keypoints.size(2) == 3

    image = data.to_uint8(image)
    image = einops.rearrange(image, "n c h w -> n h w c")

    numpy_keypoints = keypoints.round().long().cpu().numpy()

    radius = int(max(3, min(image.shape[2:4]) // 40))

    for batch_index in range(image.size(0)):

        numpy_image = image[batch_index].cpu().numpy()

        for child_index in range(len(_INDICES_OPENPOSE)):

            numpy_image_copy = numpy_image.copy()

            child_visibility = numpy_keypoints[batch_index, child_index, 2]

            if child_visibility == 0:
                continue

            child_x, child_y = numpy_keypoints[batch_index, child_index, :2]

            cv2.circle(
                numpy_image_copy,
                (child_x, child_y),
                radius,
                color=_COLORS[child_index],
                thickness=-1,
                lineType=cv2.LINE_AA,
            )

            parent_index = _INDICES_OPENPOSE[child_index]

            if parent_index == -1:
                continue

            parent_visibility = numpy_keypoints[batch_index, parent_index, 2]

            if parent_visibility == 0:
                continue

            parent_x, parent_y = numpy_keypoints[batch_index, parent_index, :2]

            cv2.line(
                numpy_image_copy,
                (child_x, child_y),
                (parent_x, parent_y),
                color=_COLORS[child_index],
                thickness=radius - 1,
                lineType=cv2.LINE_AA,
            )

            numpy_image = cv2.addWeighted(
                numpy_image, 0.4, numpy_image_copy, 0.6, 0
            )

        image[batch_index] = torch.tensor(numpy_image).to(image.device)

    image = einops.rearrange(image, "n h w c -> n c h w")
    image = data.to_float(image)
    return image
