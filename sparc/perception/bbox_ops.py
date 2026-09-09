"""Vectorized bounding box operations and tensor utilities."""

import numpy as np


def shift_boxes(boxes: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Shift bounding boxes by (dx, dy). Returns a new array.

    Args:
        boxes: (N, 4) array of [x1, y1, x2, y2] boxes.
        dx: Horizontal shift (positive = right).
        dy: Vertical shift (positive = down).

    Returns:
        (N, 4) shifted boxes.
    """
    shifted = boxes.copy()
    shifted += np.array([dx, dy, dx, dy])
    return shifted


def shift_box_dicts(box_dicts: list, dx: int, dy: int) -> None:
    """Shift candidate/target box dicts in-place.

    Each dict must have a "box" key with [x1, y1, x2, y2].
    """
    for d in box_dicts:
        d["box"][0] += dx
        d["box"][1] += dy
        d["box"][2] += dx
        d["box"][3] += dy


def frames_to_tensor(frames_np: np.ndarray):
    """View (T, H, W, C) numpy frames as a (1, T, C, H, W) torch tensor.

    The input dtype is preserved so uint8 videos stay compact during CPU-to-GPU
    IPC. Tracking workers convert them to float32 after transfer to the GPU.
    """
    import torch

    arr = np.ascontiguousarray(frames_np)
    tensor = torch.from_numpy(arr)          # zero-copy
    tensor = tensor.permute(0, 3, 1, 2)     # (T, C, H, W) — view, no copy
    tensor = tensor.unsqueeze(0)             # (1, T, C, H, W) — view, no copy
    return tensor




def box_from_mask(mask: np.ndarray):
    """Extract [x1, y1, x2, y2] bounding box from a binary mask.

    Returns None if the mask is empty.
    """
    if not mask.any():
        return None
    y_idx, x_idx = np.where(mask)
    return [float(x_idx.min()), float(y_idx.min()),
            float(x_idx.max()), float(y_idx.max())]
