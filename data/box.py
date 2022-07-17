__all__ = ["keypoints_to_box", "box_to_mask"]

import einops
import torch


def keypoints_to_box(keypoints: torch.Tensor) -> torch.Tensor:
    assert keypoints.dim() == 3
    assert keypoints.size(2) == 3

    box = []
    for instance in keypoints.unbind():

        indices = instance[:, 2].nonzero(as_tuple=False)
        x = instance[indices, 0]
        y = instance[indices, 1]

        x0 = x.min()
        y0 = y.min()
        x1 = x.max()
        y1 = y.max()

        instance_box = torch.tensor([[x0, y0], [x1, y1]])
        box.append(instance_box)

    box = torch.stack(box)
    return box


def box_to_mask(
    box: torch.Tensor, height: int, width: int, padding: float = 10.0
) -> torch.Tensor:

    assert box.dim() == 3
    assert box.size(1) == 2
    assert box.size(2) == 2

    corner_0 = einops.rearrange(box[:, 0], "n c -> n c 1 1")
    corner_1 = einops.rearrange(box[:, 1], "n c -> n c 1 1")

    corner_0 = corner_0 - padding
    corner_1 = corner_1 + padding

    grid = _identity_grid(height, width, box.device)
    grid = einops.rearrange(grid, "c h w -> 1 c h w")

    mask = (grid > corner_0) & (grid < corner_1)
    mask = torch.prod(mask, dim=1, keepdim=True)
    return mask


# FIXME: Reconcile with utils.identity_grid
def _identity_grid(
    height: int, width: int, device: torch.device
) -> torch.Tensor:

    line_y = torch.linspace(0.5, height - 0.5, steps=height, device=device)
    line_x = torch.linspace(0.5, width - 0.5, steps=width, device=device)

    grid_y, grid_x = torch.meshgrid(line_y, line_x)
    grid = torch.stack([grid_x, grid_y])
    return grid
