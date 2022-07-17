__all__ = ["draw_boxes", "draw_text"]

from typing import List, Optional, Tuple

import data
import torch
import torchvision.utils as utils
import cv2

import einops


@torch.no_grad()
def draw_boxes(
    images: torch.Tensor, boxes_list: List[torch.Tensor]
) -> torch.Tensor:

    assert images.size(0) == len(boxes_list)

    images = data.to_uint8(images)
    outputs = []

    for image, boxes in zip(images, boxes_list):

        output = utils.draw_bounding_boxes(image.cpu(), boxes.cpu())
        outputs.append(output)

    outputs = torch.stack(outputs)
    outputs = data.to_float(outputs)
    return outputs


def draw_text(
    images: torch.Tensor,
    texts: List[str],
    scale: float = 1.0,
    thickness: int = 1,
    position: Tuple[int, int] = (10, 10),
    color: Tuple[int, int, int] = (255, 255, 255),
) -> torch.Tensor:

    assert images.dim() == 4
    assert images.size(0) == len(texts)

    images = data.to_uint8(images)
    images = einops.rearrange(images, "n c h w -> n h w c")

    for i, (image, text) in enumerate(zip(images, texts)):
        image = image.cpu().numpy().copy()

        (_, text_height), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_COMPLEX, scale, thickness
        )

        cv2.putText(
            image,
            text,
            (position[0], position[1] + text_height),
            cv2.FONT_HERSHEY_COMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

        images[i] = torch.tensor(image).to(images.device)

    images = einops.rearrange(images, "n h w c -> n c h w")
    images = data.to_float(images)
    return images
