"""AllTracker loading, point sampling, and inference adapter."""

import importlib
import logging
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


def _fps_sample(coords_f, k, seed_indices=None):
    """Farthest point sampling over coords_f (N, 2) float32, returns k indices.

    seed_indices are treated as already selected — distances are initialised from
    all seeds before the greedy loop starts, so the result always includes the seeds
    and fills remaining slots with maximally spread points.
    """
    n = len(coords_f)
    if n <= k:
        return np.arange(n)

    distances = np.full(n, np.inf, dtype=np.float32)
    selected = []

    if seed_indices is not None and len(seed_indices) > 0:
        for idx in seed_indices:
            d = np.sum((coords_f - coords_f[idx]) ** 2, axis=1)
            distances = np.minimum(distances, d)
        selected = list(seed_indices)
    else:
        selected = [0]
        d = np.sum((coords_f - coords_f[0]) ** 2, axis=1)
        distances = np.minimum(distances, d)

    while len(selected) < k:
        next_idx = int(np.argmax(distances))
        selected.append(next_idx)
        d = np.sum((coords_f - coords_f[next_idx]) ** 2, axis=1)
        distances = np.minimum(distances, d)

    return np.array(selected[:k], dtype=np.int64)


def _sample_points_regular(coords, target_points, mask=None):
    """Sample tracking query points: boundary pixels first, then FPS fill.

    Up to half the budget is spent on boundary pixels (object outline), which are
    most informative for contact detection and 3D shape. The remaining budget is
    filled with FPS over the full coordinate set, seeded by the boundary selection
    so interior points are maximally spread relative to the boundary.

    Falls back to pure FPS when no mask is provided (bbox-grid case).
    """
    if len(coords) == 0 or target_points <= 0:
        return np.empty((0, 2), dtype=np.float32)

    coords = np.asarray(coords, dtype=np.int32)
    if len(coords) <= target_points:
        return coords.astype(np.float32)

    coords_f = coords.astype(np.float32)

    # --- boundary extraction ---
    seed_indices = None
    if mask is not None:
        mask_np = mask.cpu().numpy() if hasattr(mask, 'numpy') else np.asarray(mask)
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(mask_np.astype(np.uint8), kernel, iterations=1)
        boundary_mask = (mask_np.astype(np.uint8) > 0) & (eroded == 0)
        by, bx = np.where(boundary_mask)
        if len(by) > 0:
            coord_map = {(int(c[0]), int(c[1])): i for i, c in enumerate(coords)}
            boundary_indices = np.array(
                [coord_map[(int(bx[i]), int(by[i]))] for i in range(len(bx))
                 if (int(bx[i]), int(by[i])) in coord_map],
                dtype=np.int64,
            )
            if len(boundary_indices) > 0:
                n_boundary = min(len(boundary_indices), max(1, target_points // 2))
                if len(boundary_indices) <= n_boundary:
                    seed_indices = boundary_indices
                else:
                    # FPS among boundary pixels to pick the most spread-out subset
                    local_idx = _fps_sample(coords_f[boundary_indices], n_boundary)
                    seed_indices = boundary_indices[local_idx]

    return coords_f[_fps_sample(coords_f, target_points, seed_indices=seed_indices)]


def sample_tracking_points(pred_boxes, pred_masks, grid_size_bbox, device, num_points=None):
    """Sample grid/mask points for each box — model-agnostic.

    Returns:
        point_samples: list of [K, 2] tensors (x, y pixel coords) per box,
                       or empty list if no boxes.
    """
    point_samples = []
    target_points = num_points if num_points is not None else grid_size_bbox**2
    for box_idx, box in enumerate(pred_boxes):
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

        if pred_masks is None and num_points is None:
            x1 = int(x1 + (x2 - x1) * 0.2)
            y1 = int(y1 + (y2 - y1) * 0.2)
            x2 = int(x2 - (x2 - x1) * 0.2)
            y2 = int(y2 - (y2 - y1) * 0.2)
            grid_points = np.mgrid[x1:x2:grid_size_bbox*1j, y1:y2:grid_size_bbox*1j].reshape(2, -1).T
            grid_points = torch.tensor(grid_points).float().to(device)
        else:
            if pred_masks is None:
                x1 = int(x1 + (x2 - x1) * 0.2)
                y1 = int(y1 + (y2 - y1) * 0.2)
                x2 = int(x2 - (x2 - x1) * 0.2)
                y2 = int(y2 - (y2 - y1) * 0.2)
                xs = np.arange(min(x1, x2), max(x1, x2))
                ys = np.arange(min(y1, y2), max(y1, y2))
                if len(xs) == 0:
                    xs = np.array([int(box[0])])
                if len(ys) == 0:
                    ys = np.array([int(box[1])])
                coords = np.stack(np.meshgrid(xs, ys, indexing="xy"), axis=-1).reshape(-1, 2)
            else:
                mask = pred_masks[box_idx]
                if hasattr(mask, 'cpu'):
                    mask = mask.cpu().numpy()
                ys, xs = np.where(mask > 0)
                coords = np.stack([xs, ys], axis=1)
                if len(coords) == 0:
                    coords = np.mgrid[x1:x2:grid_size_bbox*1j, y1:y2:grid_size_bbox*1j].reshape(2, -1).T

            sampled_coords = _sample_points_regular(
                coords, target_points,
                mask=pred_masks[box_idx] if pred_masks is not None else None,
            )
            grid_points = torch.tensor(sampled_coords).float().to(device)

        point_samples.append(grid_points)

    return point_samples


def ensure_alltracker_repo(cache_dir=None):
    """Return the vendored AllTracker repo, cloning into cache as a fallback."""
    vendored_repo_dir = Path(__file__).resolve().parents[1] / "detectors" / "alltracker"
    if vendored_repo_dir.exists():
        return vendored_repo_dir

    repo_dir = Path(cache_dir or Path.home() / ".cache" / "robog-dataset-pipeline" / "alltracker")
    if repo_dir.exists():
        return repo_dir

    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    logging.info("Cloning AllTracker into %s", repo_dir)
    subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/aharley/alltracker.git", str(repo_dir)],
        check=True,
    )
    return repo_dir


def load_alltracker_raw_model(device, tiny=False, window_len=16, cache_dir=None):
    """Load the upstream AllTracker model and weights."""
    repo_dir = ensure_alltracker_repo(cache_dir=cache_dir)
    repo_dir_str = str(repo_dir)
    if repo_dir_str not in sys.path:
        sys.path.insert(0, repo_dir_str)

    # AllTracker uses absolute imports like `utils.*` and `nets.*`.
    # The GPU server runs in a dedicated process, so rebinding them here
    # avoids collisions with this repo's top-level `utils` package.
    for module_name in list(sys.modules):
        if module_name == "utils" or module_name.startswith("utils.") or module_name == "nets" or module_name.startswith("nets."):
            sys.modules.pop(module_name, None)

    alltracker_module = importlib.import_module("nets.alltracker")
    net_cls = alltracker_module.Net

    if tiny:
        model = net_cls(window_len, use_basicencoder=True, no_split=True)
        weights_url = "https://huggingface.co/aharley/alltracker/resolve/main/alltracker_tiny.pth"
    else:
        model = net_cls(window_len)
        weights_url = "https://huggingface.co/aharley/alltracker/resolve/main/alltracker.pth"

    state_dict = torch.hub.load_state_dict_from_url(weights_url, map_location="cpu")
    model.load_state_dict(state_dict["model"], strict=True)
    model.to(device).eval()

    for parameter in model.parameters():
        parameter.requires_grad = False

    return model


class AllTrackerModel:
    """Wraps AllTracker dense trajectory maps behind the sparse TrackingModel API."""

    def __init__(self, model, inference_iters=4, window_len=16, use_sliding=True, sample_point_count=512):
        self.model = model
        self.inference_iters = inference_iters
        self.window_len = window_len
        self.use_sliding = use_sliding
        self.sample_point_count = sample_point_count
        self.pad_to_sample_point_count = True

    def _grid_xy(self, height, width, device, dtype):
        ys, xs = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype),
            indexing="ij",
        )
        return torch.stack([xs, ys], dim=0)[None, None]

    def _normalize_dense_output(self, flow_e, visconf_e):
        if flow_e.ndim == 4:
            flow_e = flow_e[:, None]
        if visconf_e.ndim == 4:
            visconf_e = visconf_e[:, None]
        return flow_e, visconf_e

    def track_points(self, frames_torch, query_points, device, point_counts=None):
        _, T, _, H, W = frames_torch.shape

        images = frames_torch.to(device).float()
        with torch.no_grad():
            if self.use_sliding and T > self.window_len:
                flow_e, visconf_e, _, _ = self.model.forward_sliding(
                    images,
                    iters=self.inference_iters,
                    sw=None,
                    is_training=False,
                    window_len=self.window_len,
                )
            else:
                flow_e, visconf_e, _, _ = self.model(
                    images,
                    iters=self.inference_iters,
                    sw=None,
                    is_training=False,
                )

        flow_e, visconf_e = self._normalize_dense_output(flow_e.to(device), visconf_e.to(device))
        grid_xy = self._grid_xy(H, W, device=flow_e.device, dtype=flow_e.dtype)
        traj_maps = flow_e + grid_xy

        xy = query_points.round().long()
        xy[:, 0] = xy[:, 0].clamp(0, W - 1)
        xy[:, 1] = xy[:, 1].clamp(0, H - 1)

        pred_tracks = traj_maps[:, :, :, xy[:, 1], xy[:, 0]].permute(0, 1, 3, 2)

        pred_visibility = visconf_e[:, :, 0, xy[:, 1], xy[:, 0]]
        if visconf_e.shape[2] > 1:
            pred_visibility = pred_visibility * visconf_e[:, :, 1, xy[:, 1], xy[:, 0]]

        if point_counts and self.pad_to_sample_point_count:
            padded_tracks = []
            padded_visibility = []
            start_idx = 0
            for count in point_counts:
                end_idx = start_idx + count
                cur_tracks = pred_tracks[:, :, start_idx:end_idx]
                cur_visibility = pred_visibility[:, :, start_idx:end_idx]
                start_idx = end_idx

                if count < self.sample_point_count:
                    pad_count = self.sample_point_count - count
                    pad_tracks = torch.full(
                        (pred_tracks.shape[0], pred_tracks.shape[1], pad_count, pred_tracks.shape[3]),
                        -1,
                        device=pred_tracks.device,
                        dtype=pred_tracks.dtype,
                    )
                    pad_visibility = torch.zeros(
                        (pred_visibility.shape[0], pred_visibility.shape[1], pad_count),
                        device=pred_visibility.device,
                        dtype=pred_visibility.dtype,
                    )
                    cur_tracks = torch.cat([cur_tracks, pad_tracks], dim=2)
                    cur_visibility = torch.cat([cur_visibility, pad_visibility], dim=2)
                elif count > self.sample_point_count:
                    cur_tracks = cur_tracks[:, :, :self.sample_point_count]
                    cur_visibility = cur_visibility[:, :, :self.sample_point_count]

                padded_tracks.append(cur_tracks)
                padded_visibility.append(cur_visibility)

            pred_tracks = torch.cat(padded_tracks, dim=2) if padded_tracks else pred_tracks
            pred_visibility = torch.cat(padded_visibility, dim=2) if padded_visibility else pred_visibility

        return pred_tracks, pred_visibility
