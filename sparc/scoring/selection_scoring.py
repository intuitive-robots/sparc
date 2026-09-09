"""Array-based candidate scoring and serialization helpers.

Computes RobotSeg overlap, phase motion statistics, 3D gripper-sphere point
fractions, and target-point scores. The live annotator uses these signals for
selection. Publication masks and geometry are serialized separately by the
pipeline; this module also supplies array encoding and decoding utilities.

Inputs are NumPy arrays. Per-arm gripper selection is the caller's responsibility.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

# ---------------------------------------------------------------------------
# Default parameters for the paper's selection method.
# ---------------------------------------------------------------------------

ROBOT_MASK_GRID_SIZE = 96
ROBOT_MASK_DILATION_RADIUS = 2
ROBOT_MASK_PADDING_PX = 4.0

ROBOT_MASK_OVERLAP_AGG = "top_mean"   # one of: "top_mean", "quantile", "max", "mean"
ROBOT_MASK_TOP_FRACTION = 0.10
ROBOT_MASK_MIN_TOP_FRAMES = 1
ROBOT_MASK_QUANTILE = 0.90
ROBOT_MASK_FIRST_MATCHING_TRACK_FRAME_ONLY = True
ROBOT_MASK_SELECT_ACTIVE_ARM_COMPONENT = True
ROBOTSEG_MASK_DILATION_RADIUS = 4
ROBOT_MASK_MAX_KEYFRAMES = 64

ADAPTIVE_DET_SOFT_SNR_SP8_SIGNAL_NAME = "adaptive_det_soft_snr_sp8"
ADAPTIVE_DET_SOFT_SNR_SP8_OVERLAP_THRESH = 0.3
ADAPTIVE_DET_SOFT_SNR_SP8_ALPHA = 0.6
ADAPTIVE_DET_SOFT_SNR_SP8_BETA = 0.2
ADAPTIVE_DET_SOFT_SNR_SP8_EPS = 1.0
ADAPTIVE_DET_SOFT_SNR_SP8_SNR_WEIGHT = 0.5
ADAPTIVE_DET_SOFT_SNR_SP8_DET_WEIGHT_BASE = 0.75
ADAPTIVE_DET_SOFT_SNR_SP8_DET_WEIGHT_SPHERE_REDUCTION = 0.15
ADAPTIVE_DET_SOFT_SNR_SP8_SPHERE_WEIGHT = 0.3
# Continuous quadratic robot-overlap penalty with fingertip-depth relief.
SOFT_SNR_QUAD_RELIEF_PENALTY_WEIGHT = 1.45
SOFT_SNR_QUAD_RELIEF_RELIEF_STRENGTH = 0.85
SOFT_SNR_QUAD_RELIEF_HARD_OVERLAP = 0.98
SOFT_SNR_QUAD_RELIEF_HARD_BONUS = 0.20
SOFT_SNR_QUAD_RELIEF_DEPTH_LOW = 0.05
SOFT_SNR_QUAD_RELIEF_DEPTH_HIGH = 0.25
SOFT_SNR_DEPTH_RESCUE_OVERLAP_THRESH = 0.3
SOFT_SNR_DEPTH_RESCUE_DEPTH_THRESH = 0.08
SOFT_SNR_DEPTH_RESCUE_TIP_K = 10

# Scene-movement baseline (px/sec). Temporal consistency gates points whose
# mean frame-to-frame
# motion exceeds 2 × this value, regardless of the per-sample baseline.
SCENE_BASELINE_PXPS = 5.863
TEMPORAL_CONSISTENCY_GATE_PXPS = 2.0 * SCENE_BASELINE_PXPS  # 11.726


# ---------------------------------------------------------------------------
# Per-arm gripper bundle
# ---------------------------------------------------------------------------

@dataclass
class GripperTrackBundle:
    """Per-arm gripper artifacts needed by the overlap helpers.

    All arrays live in the same coordinate frame as ``candidate_tracks``
    (i.e. tracking-stage pixel coords; the caller is responsible for any
    crop/resize unscaling before passing data in).
    """
    arm: str | None                         # "left" | "right" | None
    tracks_2d: np.ndarray | None            # (T_track, N, 2)
    visibility: np.ndarray | None           # (T_track, N)  bool/float
    bbox: np.ndarray | None                 # (4,) xyxy
    tracking_frame_indices: np.ndarray | None  # (T_track,) int (full-traj frame idx)
    robotseg_masks: np.ndarray | None       # (T_keys, H, W) bool/uint8
    robotseg_frame_indices: np.ndarray | None  # (T_keys,) int (full-traj frame idx)


def resolve_active_arm(*hints: str | None) -> str | None:
    """Return the first ``"left"`` / ``"right"`` from ``hints``, else ``None``.

    Typical call from the live annotator::

        resolve_active_arm(
            task_obj_info.get("arm"),
            det_result.arm,
        )
    """
    for arm in hints:
        if arm in {"left", "right"}:
            return arm
    return None


# ---------------------------------------------------------------------------
# Task-group routing
# ---------------------------------------------------------------------------

_TASK_GROUP_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("movement_relocate", (
        "pick", "place", "move", "lift", "remove", "transfer", "hand over",
        "stack", "unstack", "carry", "pass", "receive", "put away",
        "retrieve", "throw", "hang", "grasp", "uncover", "separate", "grab",
    )),
    ("non_movement_open_close", ("open", "close")),
    ("movement_articulate", (
        "push", "pull", "turn", "rotate", "flip", "flatten", "smooth",
        "spread", "clamp", "manipulate",
    )),
    ("non_movement_articulate", ("press", "switch")),
    ("non_movement_fluid", ("pour", "scoop", "wash", "wipe", "sweep")),
    ("non_movement_tool_or_shape", (
        "cut", "peeler", "fold", "unfold", "plug", "attach",
    )),
)


def get_task_group(action: str | None) -> str:
    """Bucket an action string into the scoring task groups.
    Pass the first non-empty of ``task_obj_info.action`` /
    ``movement_type`` / ``task_type`` from the live annotator.
    """
    if not action:
        return "other"
    text = str(action).lower()
    for group, keywords in _TASK_GROUP_KEYWORDS:
        if any(k in text for k in keywords):
            return group
    return "other"


def select_active_arm_gripper(
    bundles: Iterable[GripperTrackBundle],
    active_arm: str | None,
) -> GripperTrackBundle | None:
    """Pick the bundle whose arm matches ``active_arm``, preferring one with
    RobotSeg masks. Falls back to the first bundle if the arm is unknown.

    """
    bundles = [b for b in bundles if b is not None]
    if not bundles:
        return None
    if active_arm in {"left", "right"}:
        matching = [b for b in bundles if b.arm == active_arm]
        for b in matching:
            if b.robotseg_masks is not None:
                return b
        if matching:
            return matching[0]
    return bundles[0]


# ---------------------------------------------------------------------------
# Geometry primitives for array inputs
# ---------------------------------------------------------------------------

def visible_points_on_frame(
    tracks_2d: np.ndarray,
    visibility: np.ndarray,
    frame_idx: int,
) -> np.ndarray:
    """Return visible finite (x, y) points at one tracking timestep, shape (K, 2)."""
    tracks = np.asarray(tracks_2d, dtype=float)
    vis = np.asarray(visibility, dtype=float)
    if (
        tracks.ndim != 3
        or tracks.shape[-1] < 2
        or vis.ndim != 2
        or vis.shape != tracks.shape[:2]
        or frame_idx < 0
        or frame_idx >= len(tracks)
    ):
        return np.empty((0, 2), dtype=float)
    valid = (vis[frame_idx] > 0) & np.all(np.isfinite(tracks[frame_idx, :, :2]), axis=1)
    if not valid.any():
        return np.empty((0, 2), dtype=float)
    return np.asarray(tracks[frame_idx, valid, :2], dtype=float)


def _rasterize_points_to_mask(
    points_xy: np.ndarray,
    bounds_xyxy: tuple[float, float, float, float],
    grid_size: int = ROBOT_MASK_GRID_SIZE,
) -> np.ndarray:
    mask = np.zeros((grid_size, grid_size), dtype=bool)
    points_xy = np.asarray(points_xy, dtype=float)
    if len(points_xy) == 0:
        return mask
    x0, y0, x1, y1 = bounds_xyxy
    sx = (grid_size - 1) / max(x1 - x0, 1e-6)
    sy = (grid_size - 1) / max(y1 - y0, 1e-6)
    xi = np.clip(np.round((points_xy[:, 0] - x0) * sx).astype(int), 0, grid_size - 1)
    yi = np.clip(np.round((points_xy[:, 1] - y0) * sy).astype(int), 0, grid_size - 1)
    mask[yi, xi] = True
    return mask


def binary_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Square+circle binary dilation with no SciPy dependency."""
    if radius <= 0:
        return np.asarray(mask, dtype=bool).copy()
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    out = np.zeros_like(mask)
    r2 = radius * radius
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > r2:
                continue
            sy0, sy1 = max(0, -dy), min(h, h - dy)
            sx0, sx1 = max(0, -dx), min(w, w - dx)
            dy0, dy1 = max(0, dy), min(h, h + dy)
            dx0, dx1 = max(0, dx), min(w, w + dx)
            out[dy0:dy1, dx0:dx1] |= mask[sy0:sy1, sx0:sx1]
    return out


def point_fraction_inside_mask(points_xy: np.ndarray, mask: np.ndarray) -> float:
    mask = np.asarray(mask, dtype=bool)
    points_xy = np.asarray(points_xy, dtype=float)
    if mask.ndim != 2 or len(points_xy) == 0:
        return 0.0
    h, w = mask.shape
    finite = np.all(np.isfinite(points_xy[:, :2]), axis=1)
    in_bounds = (
        finite
        & (points_xy[:, 0] >= 0.0) & (points_xy[:, 0] < float(w))
        & (points_xy[:, 1] >= 0.0) & (points_xy[:, 1] < float(h))
    )
    if not in_bounds.any():
        return 0.0
    pts = points_xy[in_bounds]
    xi = np.clip(np.rint(pts[:, 0]).astype(int), 0, w - 1)
    yi = np.clip(np.rint(pts[:, 1]).astype(int), 0, h - 1)
    return float(np.mean(mask[yi, xi]))


def _mask_crop_bounds(
    gripper_points: np.ndarray,
    candidate_points: np.ndarray,
    gripper_bbox: np.ndarray | None,
    candidate_bbox: np.ndarray | None,
) -> tuple[float, float, float, float] | None:
    xs: list[float] = []
    ys: list[float] = []
    if len(gripper_points) > 0:
        xs.extend([float(np.min(gripper_points[:, 0])), float(np.max(gripper_points[:, 0]))])
        ys.extend([float(np.min(gripper_points[:, 1])), float(np.max(gripper_points[:, 1]))])
    if len(candidate_points) > 0:
        xs.extend([float(np.min(candidate_points[:, 0])), float(np.max(candidate_points[:, 0]))])
        ys.extend([float(np.min(candidate_points[:, 1])), float(np.max(candidate_points[:, 1]))])
    for bbox in (gripper_bbox, candidate_bbox):
        if bbox is None:
            continue
        bbox = np.asarray(bbox, dtype=float)
        if bbox.shape != (4,):
            continue
        xs.extend([float(bbox[0]), float(bbox[2])])
        ys.extend([float(bbox[1]), float(bbox[3])])
    if not xs or not ys:
        return None
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    if x1 <= x0:
        x0 -= 0.5; x1 += 0.5
    if y1 <= y0:
        y0 -= 0.5; y1 += 0.5
    return (
        x0 - ROBOT_MASK_PADDING_PX,
        y0 - ROBOT_MASK_PADDING_PX,
        x1 + ROBOT_MASK_PADDING_PX,
        y1 + ROBOT_MASK_PADDING_PX,
    )


# ---------------------------------------------------------------------------
# Active-arm RobotSeg component selection
# ---------------------------------------------------------------------------

def _connected_components_2d(mask: np.ndarray) -> tuple[np.ndarray, int]:
    try:
        from scipy import ndimage
        return ndimage.label(np.asarray(mask, dtype=bool))
    except Exception:
        # Iterative flood-fill fallback so we don't hard-depend on SciPy.
        mask = np.asarray(mask, dtype=bool)
        labels = np.zeros(mask.shape, dtype=np.int32)
        component_id = 0
        h, w = mask.shape
        for y in range(h):
            for x in range(w):
                if not mask[y, x] or labels[y, x] != 0:
                    continue
                component_id += 1
                stack = [(y, x)]
                labels[y, x] = component_id
                while stack:
                    cy, cx = stack.pop()
                    for ny in range(max(0, cy - 1), min(h, cy + 2)):
                        for nx in range(max(0, cx - 1), min(w, cx + 2)):
                            if mask[ny, nx] and labels[ny, nx] == 0:
                                labels[ny, nx] = component_id
                                stack.append((ny, nx))
        return labels, component_id


def active_arm_robotseg_component(
    mask: np.ndarray,
    active_arm: str | None,
    *,
    enabled: bool = ROBOT_MASK_SELECT_ACTIVE_ARM_COMPONENT,
) -> np.ndarray:
    """Keep RobotSeg components on the declared arm side (image x-midpoint).

    If the declared side has no component, return the full mask — falling back
    to all-zero would silently drop valid evidence whenever the arm label is
    wrong or the expected arm wasn't detected.
    """
    mask = np.asarray(mask, dtype=bool)
    if (
        not enabled
        or active_arm not in {"left", "right"}
        or mask.ndim != 2
        or not mask.any()
    ):
        return mask

    labels, n_components = _connected_components_2d(mask)
    if n_components <= 0:
        return np.zeros_like(mask, dtype=bool)

    image_mid_x = mask.shape[1] / 2.0
    selected_ids: list[int] = []
    for component_id in range(1, n_components + 1):
        ys, xs = np.nonzero(labels == component_id)
        if len(xs) == 0:
            continue
        cx = float(np.mean(xs))
        component_arm = "left" if cx < image_mid_x else "right"
        if component_arm == active_arm:
            selected_ids.append(component_id)
    if not selected_ids:
        return mask
    return np.isin(labels, selected_ids)


def prepare_robotseg_mask(
    mask: np.ndarray,
    active_arm: str | None,
    *,
    dilation_radius: int = ROBOTSEG_MASK_DILATION_RADIUS,
    select_active_arm_component: bool = ROBOT_MASK_SELECT_ACTIVE_ARM_COMPONENT,
) -> np.ndarray:
    mask = active_arm_robotseg_component(mask, active_arm, enabled=select_active_arm_component)
    if dilation_radius > 0:
        mask = binary_dilate(mask, dilation_radius)
    return mask


# ---------------------------------------------------------------------------
# Frame-aggregation policies
# ---------------------------------------------------------------------------

def _aggregate_overlap(
    frame_overlaps: np.ndarray,
    frame_counts: np.ndarray,
    *,
    agg: str,
    top_fraction: float,
    min_top_frames: int,
    quantile: float,
) -> float:
    frame_overlaps = np.asarray(frame_overlaps, dtype=float)
    frame_counts = np.asarray(frame_counts, dtype=float)
    valid = np.isfinite(frame_overlaps) & (frame_counts > 0.0)
    if not valid.any():
        return 0.0

    overlaps = frame_overlaps[valid]
    counts = frame_counts[valid]

    if agg == "mean":
        return float(np.average(overlaps, weights=counts))
    if agg == "max":
        return float(np.max(overlaps))
    if agg == "quantile":
        return float(np.quantile(overlaps, quantile))

    # default: top_mean
    n_top = max(min_top_frames, int(np.ceil(len(overlaps) * top_fraction)))
    n_top = min(n_top, len(overlaps))
    top_idx = np.argsort(overlaps)[-n_top:]
    return float(np.average(overlaps[top_idx], weights=counts[top_idx]))


def aggregate_per_frame_overlap(
    per_frame_overlap: np.ndarray,   # (n_candidates, n_keyframes)
    per_frame_counts: np.ndarray,    # (n_candidates, n_keyframes)
    *,
    agg: str = ROBOT_MASK_OVERLAP_AGG,
    top_fraction: float = ROBOT_MASK_TOP_FRACTION,
    min_top_frames: int = ROBOT_MASK_MIN_TOP_FRAMES,
    quantile: float = ROBOT_MASK_QUANTILE,
    first_matching_only: bool = ROBOT_MASK_FIRST_MATCHING_TRACK_FRAME_ONLY,
    keyframe_track_t: np.ndarray | None = None,
) -> np.ndarray:
    """Re-aggregate the per-frame overlap matrix into per-candidate scalars.

    Letting the caller invoke this with different knobs is the whole reason we
    persist `per_frame_overlap` instead of only the aggregate scalar.
    """
    per_frame_overlap = np.asarray(per_frame_overlap, dtype=float)
    per_frame_counts = np.asarray(per_frame_counts, dtype=float)
    n_candidates = per_frame_overlap.shape[0]
    if n_candidates == 0 or per_frame_overlap.size == 0:
        return np.zeros(n_candidates, dtype=float)

    if first_matching_only:
        if keyframe_track_t is None:
            raise ValueError(
                "first_matching_only requires keyframe_track_t to pick the earliest frame"
            )
        first_i = int(np.argmin(np.asarray(keyframe_track_t, dtype=int)))
        return per_frame_overlap[:, first_i].astype(float).copy()

    out = np.zeros(n_candidates, dtype=float)
    for i in range(n_candidates):
        out[i] = _aggregate_overlap(
            per_frame_overlap[i],
            per_frame_counts[i],
            agg=agg,
            top_fraction=top_fraction,
            min_top_frames=min_top_frames,
            quantile=quantile,
        )
    return out


def _uniform_keyframe_selection(n_frames: int, max_frames: int) -> np.ndarray:
    if n_frames <= max_frames:
        return np.arange(n_frames, dtype=int)
    return np.unique(np.linspace(0, n_frames - 1, max_frames, dtype=int))


# ---------------------------------------------------------------------------
# Start-frame point-cloud fallback (when RobotSeg masks aren't available)
# ---------------------------------------------------------------------------

def candidate_robot_start_frame_point_overlap_approx(
    candidate_tracks: np.ndarray,
    candidate_visibility: np.ndarray,
    candidate_bboxes: np.ndarray | None,
    gripper: GripperTrackBundle,
    *,
    grid_size: int = ROBOT_MASK_GRID_SIZE,
    dilation_radius: int = ROBOT_MASK_DILATION_RADIUS,
) -> np.ndarray:
    """Fallback when no RobotSeg masks are available.

    For each candidate, rasterizes start-frame points of the gripper and the
    candidate onto a shared grid, dilates the gripper mask, and returns the
    fraction of candidate-mask cells covered.
    """
    n_candidates = len(candidate_tracks)
    if n_candidates == 0 or gripper.tracks_2d is None or gripper.visibility is None:
        return np.zeros(n_candidates, dtype=float)

    gripper_points = visible_points_on_frame(gripper.tracks_2d, gripper.visibility, 0)
    gripper_bbox = gripper.bbox
    if len(gripper_points) == 0 or gripper_bbox is None:
        return np.zeros(n_candidates, dtype=float)

    overlaps = np.zeros(n_candidates, dtype=float)
    for i in range(n_candidates):
        cand_pts = visible_points_on_frame(
            candidate_tracks[i], candidate_visibility[i], 0
        )
        cand_bbox = None if candidate_bboxes is None else candidate_bboxes[i]
        if len(cand_pts) == 0 or cand_bbox is None:
            continue
        bounds = _mask_crop_bounds(
            gripper_points, cand_pts,
            np.asarray(gripper_bbox, dtype=float),
            np.asarray(cand_bbox, dtype=float),
        )
        if bounds is None:
            continue
        gripper_mask = _rasterize_points_to_mask(gripper_points, bounds, grid_size)
        cand_mask = _rasterize_points_to_mask(cand_pts, bounds, grid_size)
        if not cand_mask.any():
            continue
        gripper_cover = binary_dilate(gripper_mask, dilation_radius)
        overlaps[i] = float(np.mean(gripper_cover[cand_mask]))
    return overlaps


# ---------------------------------------------------------------------------
# Main entry point: per-candidate RobotSeg keyframe overlap
# ---------------------------------------------------------------------------

@dataclass
class RobotSegOverlapResult:
    overlap: np.ndarray                       # (n_candidates,) aggregated scalar
    per_frame_overlap: np.ndarray             # (n_candidates, n_keyframes)
    per_frame_counts: np.ndarray              # (n_candidates, n_keyframes)
    keyframe_track_t: np.ndarray              # (n_keyframes,) tracking timesteps used
    keyframe_full_frame_idx: np.ndarray       # (n_keyframes,) full-trajectory frame idx
    fallback_used: bool                       # True if we fell back to start-frame approx
    meta: dict                                # knobs baked into ``overlap``


def compute_robotseg_keyframe_overlap(
    candidate_tracks: np.ndarray,             # (n_candidates, T_track, N, 2)
    candidate_visibility: np.ndarray,         # (n_candidates, T_track, N)
    candidate_bboxes: np.ndarray | None,      # (n_candidates, 4) or None
    gripper: GripperTrackBundle,
    tracking_frame_indices: np.ndarray | None,
    active_arm: str | None,
    *,
    dilation_radius: int = ROBOTSEG_MASK_DILATION_RADIUS,
    select_active_arm_component: bool = ROBOT_MASK_SELECT_ACTIVE_ARM_COMPONENT,
    agg: str = ROBOT_MASK_OVERLAP_AGG,
    top_fraction: float = ROBOT_MASK_TOP_FRACTION,
    min_top_frames: int = ROBOT_MASK_MIN_TOP_FRAMES,
    quantile: float = ROBOT_MASK_QUANTILE,
    first_matching_only: bool = ROBOT_MASK_FIRST_MATCHING_TRACK_FRAME_ONLY,
    max_keyframes: int = ROBOT_MASK_MAX_KEYFRAMES,
) -> RobotSegOverlapResult:
    """Per-candidate fraction of visible keyframe points landing in RobotSeg masks.

    The per-frame matrix is always returned alongside the aggregated scalar
    so callers can re-tune ``agg`` / ``first_matching_only`` without
    recomputing the point/mask intersections.

    Args:
        candidate_tracks: ``(n_candidates, T_track, N, 2)`` xy in track-stage pixels.
        candidate_visibility: ``(n_candidates, T_track, N)`` boolean/float.
        candidate_bboxes: ``(n_candidates, 4)`` xyxy, only used by the
            start-frame fallback.
        gripper: per-arm bundle from ``select_active_arm_gripper``.
        tracking_frame_indices: ``(T_track,)`` mapping each tracking timestep
            to a full-trajectory frame index. Must agree with
            ``gripper.tracking_frame_indices`` (we prefer the gripper's copy
            but fall back to this one).
        active_arm: ``"left" | "right" | None`` — used for component selection.
    """
    candidate_tracks = np.asarray(candidate_tracks)
    candidate_visibility = np.asarray(candidate_visibility)
    n_candidates = len(candidate_tracks)

    meta = dict(
        dilation_radius=int(dilation_radius),
        select_active_arm_component=bool(select_active_arm_component),
        agg=str(agg),
        top_fraction=float(top_fraction),
        min_top_frames=int(min_top_frames),
        quantile=float(quantile),
        first_matching_only=bool(first_matching_only),
        max_keyframes=int(max_keyframes),
        active_arm=active_arm if active_arm in {"left", "right"} else None,
    )

    if n_candidates == 0:
        return RobotSegOverlapResult(
            overlap=np.zeros(0, dtype=float),
            per_frame_overlap=np.zeros((0, 0), dtype=float),
            per_frame_counts=np.zeros((0, 0), dtype=float),
            keyframe_track_t=np.zeros(0, dtype=int),
            keyframe_full_frame_idx=np.zeros(0, dtype=int),
            fallback_used=False,
            meta=meta,
        )

    if (
        gripper is None
        or gripper.robotseg_masks is None
        or gripper.robotseg_frame_indices is None
    ):
        if gripper is None:
            overlap = np.zeros(n_candidates, dtype=float)
        else:
            overlap = candidate_robot_start_frame_point_overlap_approx(
                candidate_tracks, candidate_visibility, candidate_bboxes, gripper,
            )
        return RobotSegOverlapResult(
            overlap=overlap,
            per_frame_overlap=np.zeros((n_candidates, 0), dtype=float),
            per_frame_counts=np.zeros((n_candidates, 0), dtype=float),
            keyframe_track_t=np.zeros(0, dtype=int),
            keyframe_full_frame_idx=np.zeros(0, dtype=int),
            fallback_used=True,
            meta=meta,
        )

    masks = np.asarray(gripper.robotseg_masks)
    if masks.dtype != bool:
        masks = masks > 0
    rs_frame_idx = np.asarray(gripper.robotseg_frame_indices, dtype=int).reshape(-1)
    if (
        masks.ndim != 3
        or rs_frame_idx.ndim != 1
        or len(masks) != len(rs_frame_idx)
        or len(masks) == 0
    ):
        overlap = candidate_robot_start_frame_point_overlap_approx(
            candidate_tracks, candidate_visibility, candidate_bboxes, gripper,
        )
        return RobotSegOverlapResult(
            overlap=overlap,
            per_frame_overlap=np.zeros((n_candidates, 0), dtype=float),
            per_frame_counts=np.zeros((n_candidates, 0), dtype=float),
            keyframe_track_t=np.zeros(0, dtype=int),
            keyframe_full_frame_idx=np.zeros(0, dtype=int),
            fallback_used=True,
            meta=meta,
        )

    # Build the full-traj-frame -> tracking-timestep map. Prefer the gripper's
    # own tracking_frame_indices (the "gripper mismatch" fix); fall back to the
    # generic one if that isn't set.
    track_indices = (
        gripper.tracking_frame_indices
        if gripper.tracking_frame_indices is not None
        else tracking_frame_indices
    )
    if track_indices is not None:
        track_indices = np.asarray(track_indices, dtype=int).reshape(-1)
        frame_to_track_t = {int(fi): int(t) for t, fi in enumerate(track_indices)}
    else:
        frame_to_track_t = None

    matched_masks: list[np.ndarray] = []
    matched_track_ts: list[int] = []
    matched_full_frame_idx: list[int] = []
    for frame_idx, mask in zip(rs_frame_idx, masks):
        if frame_idx < 0:
            continue
        if frame_to_track_t is not None:
            if int(frame_idx) not in frame_to_track_t:
                continue
            track_t = frame_to_track_t[int(frame_idx)]
        else:
            track_t = int(frame_idx)
        prepared = prepare_robotseg_mask(
            mask, active_arm,
            dilation_radius=dilation_radius,
            select_active_arm_component=select_active_arm_component,
        )
        if not prepared.any():
            continue
        matched_masks.append(prepared)
        matched_track_ts.append(track_t)
        matched_full_frame_idx.append(int(frame_idx))

    if not matched_masks:
        if select_active_arm_component and active_arm in {"left", "right"}:
            overlap = np.zeros(n_candidates, dtype=float)
            fallback_used = False
        else:
            overlap = candidate_robot_start_frame_point_overlap_approx(
                candidate_tracks, candidate_visibility, candidate_bboxes, gripper,
            )
            fallback_used = True
        return RobotSegOverlapResult(
            overlap=overlap,
            per_frame_overlap=np.zeros((n_candidates, 0), dtype=float),
            per_frame_counts=np.zeros((n_candidates, 0), dtype=float),
            keyframe_track_t=np.zeros(0, dtype=int),
            keyframe_full_frame_idx=np.zeros(0, dtype=int),
            fallback_used=fallback_used,
            meta=meta,
        )

    # Cap the number of keyframes used (uniform pick across the matched set).
    sel = _uniform_keyframe_selection(len(matched_masks), max_keyframes)
    if len(sel) < len(matched_masks):
        matched_masks = [matched_masks[i] for i in sel]
        matched_track_ts = [matched_track_ts[i] for i in sel]
        matched_full_frame_idx = [matched_full_frame_idx[i] for i in sel]

    n_keyframes = len(matched_masks)
    per_frame_overlap = np.zeros((n_candidates, n_keyframes), dtype=float)
    per_frame_counts = np.zeros((n_candidates, n_keyframes), dtype=float)
    valid_any = np.zeros(n_candidates, dtype=bool)
    for ci in range(n_candidates):
        for fi, (track_t, mask) in enumerate(zip(matched_track_ts, matched_masks)):
            cand_pts = visible_points_on_frame(
                candidate_tracks[ci], candidate_visibility[ci], track_t
            )
            if len(cand_pts) == 0:
                continue
            per_frame_counts[ci, fi] = float(len(cand_pts))
            per_frame_overlap[ci, fi] = point_fraction_inside_mask(cand_pts, mask)
        valid_any[ci] = bool(np.any(per_frame_counts[ci] > 0.0))

    overlap = aggregate_per_frame_overlap(
        per_frame_overlap, per_frame_counts,
        agg=agg, top_fraction=top_fraction, min_top_frames=min_top_frames,
        quantile=quantile, first_matching_only=first_matching_only,
        keyframe_track_t=np.asarray(matched_track_ts, dtype=int),
    )

    fallback_used = False
    if not valid_any.all():
        approx = candidate_robot_start_frame_point_overlap_approx(
            candidate_tracks, candidate_visibility, candidate_bboxes, gripper,
        )
        overlap[~valid_any] = approx[~valid_any]
        fallback_used = True

    return RobotSegOverlapResult(
        overlap=overlap,
        per_frame_overlap=per_frame_overlap,
        per_frame_counts=per_frame_counts,
        keyframe_track_t=np.asarray(matched_track_ts, dtype=int),
        keyframe_full_frame_idx=np.asarray(matched_full_frame_idx, dtype=int),
        fallback_used=fallback_used,
        meta=meta,
    )


# ---------------------------------------------------------------------------
# Phase-SNR motion stats
# ---------------------------------------------------------------------------
#
# We persist the raw per-phase mean+std numbers so any (alpha, beta, eps)
# can be re-derived downstream.


def _estimate_n_orig(grasp_phases: list, T_fallback: int) -> int:
    """Original-trajectory length estimated from phase end-frame boundaries."""
    ends: list[int] = []
    for ph in (grasp_phases or []):
        if isinstance(ph, dict):
            ends.append(int(ph.get("end_frame", 0)))
        elif len(ph) >= 2:
            ends.append(int(ph[1]))
    return int(max(ends)) + 1 if ends else int(T_fallback)


def _build_phase_masks_scaled(
    grasp_phases: list, T: int, n_orig: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-tracking-frame interact / non_interact boolean masks.

    Phases live in original-trajectory frames (0..n_orig); we rescale to the
    tracking timeline of length T and skip phases that span ≥95% of the
    trajectory (those drown out the signal).
    """
    scale = T / max(n_orig, 1)
    interact = np.zeros(T, dtype=bool)
    for ph in (grasp_phases or []):
        if isinstance(ph, dict):
            pt = ph.get("phase_type", ph.get("phase", ""))
            sf0 = ph.get("start_frame", 0)
            ef0 = ph.get("end_frame", n_orig)
        else:
            sf0, ef0 = ph[0], ph[1]
            pt = ph[2] if len(ph) > 2 else ""
        if (ef0 - sf0) >= n_orig * 0.95:
            continue
        sf = max(0, int(round(sf0 * scale)))
        ef = min(T, int(round(ef0 * scale)) + 1)
        if pt == "interact":
            interact[sf:ef] = True
    margin = min(3, T // 10)
    non_interact = ~interact
    if margin > 0:
        non_interact[:margin] = False
        non_interact[-margin:] = False
    return interact, non_interact


@dataclass
class PhaseMotionStats:
    """Per-candidate, per-phase motion statistics (px/sec).

    Fields are 1-D arrays of length ``n_candidates`` unless noted. ``fallback``
    is True when phase info was absent or insufficient (≤1 phase, or fewer
    than 3 non-interact frames, or zero interact frames) — in that case the
    "interact_*" fields hold whole-track stats and "non_interact_*" are 0.
    """
    interact_frames: np.ndarray              # int
    non_interact_frames: np.ndarray          # int
    n_total: np.ndarray                      # int (T - 1 effectively)
    interact_mean: np.ndarray
    non_interact_mean: np.ndarray
    interact_std: np.ndarray
    non_interact_std: np.ndarray
    interact_sum: np.ndarray
    fallback: np.ndarray                     # bool


def compute_phase_motion_stats(
    candidate_tracks_2d: np.ndarray,         # (n_candidates, T, N, 2)
    candidate_visibility: np.ndarray | None, # (n_candidates, T, N) or None
    grasp_phases: list,
    fps: float,
    *,
    invalid_sentinel: float = -1.0,
) -> PhaseMotionStats:
    """Per-candidate, per-phase frame-to-frame motion stats in px/sec.

    Input arrays may use a ``-1`` sentinel for invalid tracks. If the
    annotator passes real visibility flags instead, they are honored: a point
    is "valid for this candidate" iff it has at least one visible & finite
    timestep.
    """
    candidate_tracks_2d = np.asarray(candidate_tracks_2d, dtype=np.float32)
    n_candidates = len(candidate_tracks_2d)

    interact_frames = np.zeros(n_candidates, dtype=int)
    non_interact_frames = np.zeros(n_candidates, dtype=int)
    n_total = np.zeros(n_candidates, dtype=int)
    interact_mean = np.zeros(n_candidates, dtype=float)
    non_interact_mean = np.zeros(n_candidates, dtype=float)
    interact_std = np.zeros(n_candidates, dtype=float)
    non_interact_std = np.zeros(n_candidates, dtype=float)
    interact_sum = np.zeros(n_candidates, dtype=float)
    fallback = np.ones(n_candidates, dtype=bool)

    for ci in range(n_candidates):
        tracks = candidate_tracks_2d[ci]
        if tracks.ndim != 3 or tracks.shape[0] < 2 or tracks.shape[-1] < 2:
            continue
        T = tracks.shape[0]

        # Decide which points to keep. Prefer real visibility if provided.
        if candidate_visibility is not None:
            vis = np.asarray(candidate_visibility[ci]) > 0
            point_mask = vis.any(axis=0) & np.all(np.isfinite(tracks).all(axis=-1), axis=0)
        else:
            point_mask = ~np.all(tracks == invalid_sentinel, axis=(0, 2))
        if not point_mask.any():
            continue

        sub = tracks[:, point_mask, :2]
        # Frame-to-frame mean displacement (px/sec) averaged over points.
        frame_disp = np.linalg.norm(np.diff(sub, axis=0), axis=2).mean(axis=1) * fps
        n_disp = len(frame_disp)
        n_total[ci] = n_disp

        if not grasp_phases or len(grasp_phases) <= 1:
            interact_frames[ci] = n_disp
            interact_mean[ci] = float(frame_disp.mean()) if n_disp else 0.0
            interact_std[ci] = float(frame_disp.std()) if n_disp else 0.0
            interact_sum[ci] = float(frame_disp.sum())
            continue

        n_orig = _estimate_n_orig(grasp_phases, T)
        interact_mask, non_interact_mask = _build_phase_masks_scaled(grasp_phases, T, n_orig)
        # Phase masks are per-frame; displacement is between consecutive
        # frames, so we drop the first sample (this matches the offline code).
        interact_disp_mask = interact_mask[1:][:n_disp]
        non_interact_disp_mask = non_interact_mask[1:][:n_disp]

        n_int = int(interact_disp_mask.sum())
        n_non = int(non_interact_disp_mask.sum())
        interact_frames[ci] = n_int
        non_interact_frames[ci] = n_non

        is_fallback = (n_non < 3) or (n_int == 0)
        if is_fallback:
            interact_mean[ci] = (
                float(frame_disp[interact_disp_mask].mean()) if n_int else float(frame_disp.mean())
            )
            interact_std[ci] = (
                float(frame_disp[interact_disp_mask].std()) if n_int else float(frame_disp.std())
            )
            interact_sum[ci] = (
                float(frame_disp[interact_disp_mask].sum()) if n_int else float(frame_disp.sum())
            )
        else:
            fallback[ci] = False
            interact_mean[ci] = float(frame_disp[interact_disp_mask].mean())
            non_interact_mean[ci] = float(frame_disp[non_interact_disp_mask].mean())
            interact_std[ci] = float(frame_disp[interact_disp_mask].std())
            non_interact_std[ci] = float(frame_disp[non_interact_disp_mask].std())
            interact_sum[ci] = float(frame_disp[interact_disp_mask].sum())

    return PhaseMotionStats(
        interact_frames=interact_frames,
        non_interact_frames=non_interact_frames,
        n_total=n_total,
        interact_mean=interact_mean,
        non_interact_mean=non_interact_mean,
        interact_std=interact_std,
        non_interact_std=non_interact_std,
        interact_sum=interact_sum,
        fallback=fallback,
    )


def phase_snr(
    stats: PhaseMotionStats,
    *,
    alpha: float = 2.0,
    beta: float = 1.6,
    eps: float = 1.0,
) -> np.ndarray:
    """``interact_mean^alpha / (non_interact_mean + eps)^beta``.

    Falls back to plain ``interact_mean`` when phase info was insufficient
    (mirrors `phase_snr_movement_signal`). Returns ``-inf`` for candidates
    with zero usable frames so they sort last in any selection.
    """
    out = np.full(len(stats.interact_mean), -np.inf, dtype=float)
    for i in range(len(out)):
        if stats.interact_frames[i] == 0 and stats.non_interact_frames[i] == 0:
            continue
        if stats.fallback[i]:
            out[i] = float(stats.interact_mean[i])
        else:
            out[i] = float(stats.interact_mean[i] ** alpha
                           / (stats.non_interact_mean[i] + eps) ** beta)
    return out


# ---------------------------------------------------------------------------
# 3D gripper-sphere point fraction
# ---------------------------------------------------------------------------
#
# Open3D is *optional* — when absent, point-cloud outlier filtering reduces
# to a pass-through plus the depth-MAD trim.

SPHERE3D_PCD_FILTER_ENABLED = True
SPHERE3D_PCD_NB_NEIGHBORS = 20
SPHERE3D_PCD_STD_RATIO = 1.0
SPHERE3D_PCD_DEPTH_MAD_K: float | None = 2.0
SPHERE3D_MIN_VALID_POINTS = 3
SPHERE3D_GRIP_RADIUS_K_NEIGHBORS = 5
SPHERE3D_GRIP_RADIUS_SCALE = 2.0
SPHERE3D_GRIP_RADIUS_MIN = 0.05
SPHERE3D_GRIP_RADIUS_MAX_OBJ_SCALE = 1.0
SPHERE3D_OBJ_OBB_PERCENTILE = 80.0
SPHERE3D_POINT_FRACTION_PEAK_RADIUS_MULTIPLIER = 1.75
SPHERE3D_POINT_FRACTION_PEAK_TOP_FRACTION = 0.10
SPHERE3D_POINT_FRACTION_PEAK_MIN_FRAMES = 3

_OPEN3D_MODULE = None
_OPEN3D_IMPORT_ATTEMPTED = False


def _get_open3d():
    global _OPEN3D_MODULE, _OPEN3D_IMPORT_ATTEMPTED
    if not _OPEN3D_IMPORT_ATTEMPTED:
        _OPEN3D_IMPORT_ATTEMPTED = True
        try:
            import open3d as o3d  # type: ignore
            _OPEN3D_MODULE = o3d
        except Exception:
            _OPEN3D_MODULE = None
    return _OPEN3D_MODULE


def _pcd_inlier_indices(
    points_xyz: np.ndarray,
    *,
    nb_neighbors: int = SPHERE3D_PCD_NB_NEIGHBORS,
    std_ratio: float = SPHERE3D_PCD_STD_RATIO,
    depth_mad_k: float | None = SPHERE3D_PCD_DEPTH_MAD_K,
) -> np.ndarray:
    """Statistical outlier removal + depth-MAD trim. Returns surviving indices."""
    n = len(points_xyz)
    all_idx = np.arange(n, dtype=int)
    idx = all_idx
    if n >= max(3, nb_neighbors):
        o3d = _get_open3d()
        if o3d is not None:
            try:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points_xyz[:, :3].astype(np.float64))
                _, ind = pcd.remove_statistical_outlier(
                    nb_neighbors=int(nb_neighbors), std_ratio=float(std_ratio)
                )
                if len(ind) >= 1:
                    idx = np.asarray(ind, dtype=int)
            except Exception:
                pass

    if depth_mad_k is not None and len(idx) >= 3:
        z = points_xyz[idx, 2]
        med = float(np.median(z))
        mad = float(np.median(np.abs(z - med)))
        if mad > 1e-9:
            keep = np.abs(z - med) <= depth_mad_k * mad
            if keep.any():
                idx = idx[keep]

    return idx if len(idx) >= 1 else all_idx


def _filtered_3d_per_frame(
    tracks_3d: np.ndarray,                # (T, N, 3)
    visibility: np.ndarray,               # (T, N)
    *,
    pcd_filter: bool = SPHERE3D_PCD_FILTER_ENABLED,
    min_valid_points: int = SPHERE3D_MIN_VALID_POINTS,
) -> list[np.ndarray | None]:
    """Per-timestep array of inlier 3D points (or ``None`` when too few)."""
    tracks_3d = np.asarray(tracks_3d, dtype=float)
    visibility = np.asarray(visibility, dtype=float)
    if tracks_3d.ndim != 3 or tracks_3d.shape[-1] < 3:
        return []
    if visibility.shape != tracks_3d.shape[:2]:
        return []
    out: list[np.ndarray | None] = [None] * tracks_3d.shape[0]
    for t in range(tracks_3d.shape[0]):
        valid = (visibility[t] > 0) & np.all(np.isfinite(tracks_3d[t, :, :3]), axis=1)
        if not valid.any():
            continue
        pts = tracks_3d[t, valid, :3]
        if pcd_filter:
            pts = pts[_pcd_inlier_indices(pts)]
        if len(pts) >= min_valid_points:
            out[t] = pts
    return out


def _filtered_gripper_per_frame(
    gripper_tracks_2d: np.ndarray,        # (T, N, 2)
    gripper_tracks_3d: np.ndarray,        # (T, N, 3)
    gripper_visibility: np.ndarray,       # (T, N)
    *,
    pcd_filter: bool = SPHERE3D_PCD_FILTER_ENABLED,
    min_valid_points: int = SPHERE3D_MIN_VALID_POINTS,
) -> tuple[list[np.ndarray | None], list[np.ndarray | None]]:
    g2 = np.asarray(gripper_tracks_2d, dtype=float)
    g3 = np.asarray(gripper_tracks_3d, dtype=float)
    gv = np.asarray(gripper_visibility, dtype=float)
    T = g3.shape[0]
    pts3: list[np.ndarray | None] = [None] * T
    pts2: list[np.ndarray | None] = [None] * T
    if g3.ndim != 3 or g2.ndim != 3 or g2.shape[:2] != g3.shape[:2] or gv.shape != g3.shape[:2]:
        return pts3, pts2
    for t in range(T):
        valid = (
            (gv[t] > 0)
            & np.all(np.isfinite(g3[t, :, :3]), axis=1)
            & np.all(np.isfinite(g2[t, :, :2]), axis=1)
        )
        if not valid.any():
            continue
        p3 = g3[t, valid, :3]
        p2 = g2[t, valid, :2]
        if pcd_filter:
            inliers = _pcd_inlier_indices(p3)
            p3 = p3[inliers]
            p2 = p2[inliers]
        if len(p3) >= min_valid_points:
            pts3[t] = p3
            pts2[t] = p2
    return pts3, pts2


def _fit_obb_3d(pts: np.ndarray, percentile: float = SPHERE3D_OBJ_OBB_PERCENTILE) -> dict:
    center = pts[:, :3].mean(axis=0)
    centered = pts[:, :3] - center
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    proj = centered @ Vt.T
    half_extents = np.percentile(np.abs(proj), percentile, axis=0)
    return {"center": center, "axes": Vt, "half_extents": half_extents}


def _object_obbs(
    filtered_3d: list[np.ndarray | None], n_frames: int,
) -> tuple[list[dict | None], np.ndarray]:
    obbs: list[dict | None] = [None] * n_frames
    valid = np.zeros(n_frames, dtype=bool)
    common = min(n_frames, len(filtered_3d))
    for t in range(common):
        pts = filtered_3d[t]
        if pts is None:
            continue
        obbs[t] = _fit_obb_3d(pts)
        valid[t] = True
    return obbs, valid


def _gripper_spheres(
    filtered_grip_3d: list[np.ndarray | None],
    filtered_grip_2d: list[np.ndarray | None],
    img_center_xy: tuple[float, float],
    obj_scale: float,
    *,
    k_neighbors: int = SPHERE3D_GRIP_RADIUS_K_NEIGHBORS,
    radius_scale: float = SPHERE3D_GRIP_RADIUS_SCALE,
    radius_min: float = SPHERE3D_GRIP_RADIUS_MIN,
    radius_max_obj_scale: float = SPHERE3D_GRIP_RADIUS_MAX_OBJ_SCALE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame gripper sphere (center_3d, center_2d, radius, valid_mask)."""
    n = len(filtered_grip_3d)
    centers = np.full((n, 3), np.nan, dtype=float)
    centers_2d = np.full((n, 2), np.nan, dtype=float)
    radii = np.zeros(n, dtype=float)
    cx, cy = img_center_xy
    for t in range(n):
        p3 = filtered_grip_3d[t]
        p2 = filtered_grip_2d[t]
        if p3 is None or p2 is None:
            continue
        # Pick the point whose 2D projection is closest to the image center,
        # then take its 3D location as the sphere center. This mirrors the
        # offline heuristic and avoids depending on a calibration we don't have.
        dists2d = np.hypot(p2[:, 0] - cx, p2[:, 1] - cy)
        closest = int(np.argmin(dists2d))
        center = p3[closest]
        centers[t] = center
        centers_2d[t] = p2[closest]
        k = min(k_neighbors, len(p3) - 1)
        if k > 0:
            nn = np.sort(np.linalg.norm(p3 - center, axis=1))[1:k + 1]
            radii[t] = max(float(np.median(nn)) * radius_scale, radius_min)
        else:
            radii[t] = radius_min
    radius_cap = radius_max_obj_scale * obj_scale if np.isfinite(obj_scale) else np.inf
    radii = np.minimum(radii, radius_cap)
    return centers, centers_2d, radii, np.all(np.isfinite(centers), axis=1)


def _estimate_image_center(
    candidate_bboxes_2d: np.ndarray | None,
    fallback_xy: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """Image-center heuristic: half of the largest bbox extent over candidates.

    Used when no calibrated principal point is available. ``fallback_xy``
    applies only when no boxes are passed.
    """
    if candidate_bboxes_2d is None or len(candidate_bboxes_2d) == 0:
        return fallback_xy or (0.0, 0.0)
    arr = np.asarray(candidate_bboxes_2d, dtype=float)
    max_x = float(np.max(arr[:, 2])) if arr.size else 0.0
    max_y = float(np.max(arr[:, 3])) if arr.size else 0.0
    return max_x / 2.0, max_y / 2.0


def _object_scale_from_obbs(
    candidate_filtered_3d: list[list[np.ndarray | None]],
    n_frames: int,
) -> float:
    extents: list[float] = []
    for filtered in candidate_filtered_3d:
        obbs, valid = _object_obbs(filtered, n_frames)
        for t in range(n_frames):
            if valid[t] and obbs[t] is not None:
                extents.append(float(np.max(obbs[t]["half_extents"])))
    return float(np.median(extents)) if extents else float(np.inf)


@dataclass
class SpherePointFractionResult:
    mean_fraction: np.ndarray                  # (n_candidates,)
    peak_fraction: np.ndarray                  # (n_candidates,)
    mean_fraction_r8: np.ndarray               # (n_candidates,) — radius×8 variant
    peak_fraction_r8: np.ndarray               # (n_candidates,) — radius×8 peak variant
    per_frame_fraction_mean: np.ndarray        # (n_candidates, T) — base radius
    per_frame_fraction_peak: np.ndarray        # (n_candidates, T) — contact radius
    per_frame_valid: np.ndarray                # (n_candidates, T) bool
    sphere_centers: np.ndarray                 # (T, 3) — NaN where invalid
    sphere_centers_2d: np.ndarray              # (T, 2) — 2D pixel of the sphere-center point
    sphere_radii: np.ndarray                   # (T,)
    sphere_radii_contact: np.ndarray           # (T,) — peak variant
    obj_scale: float


def compute_sphere_point_fraction(
    candidate_tracks_3d: np.ndarray,           # (n_cand, T, N, 3)
    candidate_visibility: np.ndarray,          # (n_cand, T, N)
    gripper_tracks_2d: np.ndarray,             # (T, N_g, 2)
    gripper_tracks_3d: np.ndarray,             # (T, N_g, 3)
    gripper_visibility: np.ndarray,            # (T, N_g)
    *,
    candidate_bboxes_2d: np.ndarray | None = None,
    image_center_xy: tuple[float, float] | None = None,
    peak_radius_multiplier: float = SPHERE3D_POINT_FRACTION_PEAK_RADIUS_MULTIPLIER,
    peak_top_fraction: float = SPHERE3D_POINT_FRACTION_PEAK_TOP_FRACTION,
    peak_min_frames: int = SPHERE3D_POINT_FRACTION_PEAK_MIN_FRAMES,
) -> SpherePointFractionResult:
    """Per-candidate fraction of valid 3D points inside the gripper sphere.

    Both the *mean-over-frames* score and the *peak* (top-K-frame) score are
    returned, matching `_compute_3d_sphere_point_fraction[_peak]`.
    The full per-frame matrices are also returned so downstream can re-tune
    the peak window or pick a different aggregation.
    """
    candidate_tracks_3d = np.asarray(candidate_tracks_3d, dtype=float)
    candidate_visibility = np.asarray(candidate_visibility, dtype=float)
    n_cand = len(candidate_tracks_3d)
    T = candidate_tracks_3d.shape[1] if n_cand > 0 else 0

    empty_per_frame = np.full((n_cand, T), np.nan, dtype=float)
    empty_valid = np.zeros((n_cand, T), dtype=bool)
    empty_centers = np.full((T, 3), np.nan, dtype=float)
    empty_centers_2d = np.full((T, 2), np.nan, dtype=float)
    empty_radii = np.zeros(T, dtype=float)
    if n_cand == 0 or T == 0:
        return SpherePointFractionResult(
            mean_fraction=np.zeros(n_cand, dtype=float),
            peak_fraction=np.zeros(n_cand, dtype=float),
            mean_fraction_r8=np.zeros(n_cand, dtype=float),
            peak_fraction_r8=np.zeros(n_cand, dtype=float),
            per_frame_fraction_mean=empty_per_frame,
            per_frame_fraction_peak=empty_per_frame.copy(),
            per_frame_valid=empty_valid,
            sphere_centers=empty_centers,
            sphere_centers_2d=empty_centers_2d,
            sphere_radii=empty_radii,
            sphere_radii_contact=empty_radii.copy(),
            obj_scale=float("inf"),
        )

    candidate_filtered_3d = [
        _filtered_3d_per_frame(candidate_tracks_3d[ci], candidate_visibility[ci])
        for ci in range(n_cand)
    ]
    grip_3d, grip_2d = _filtered_gripper_per_frame(
        gripper_tracks_2d, gripper_tracks_3d, gripper_visibility,
    )
    if len(grip_3d) == 0:
        return SpherePointFractionResult(
            mean_fraction=np.zeros(n_cand, dtype=float),
            peak_fraction=np.zeros(n_cand, dtype=float),
            mean_fraction_r8=np.zeros(n_cand, dtype=float),
            peak_fraction_r8=np.zeros(n_cand, dtype=float),
            per_frame_fraction_mean=empty_per_frame,
            per_frame_fraction_peak=empty_per_frame.copy(),
            per_frame_valid=empty_valid,
            sphere_centers=empty_centers,
            sphere_centers_2d=empty_centers_2d,
            sphere_radii=empty_radii,
            sphere_radii_contact=empty_radii.copy(),
            obj_scale=float("inf"),
        )

    img_center = image_center_xy or _estimate_image_center(candidate_bboxes_2d)
    obj_scale = _object_scale_from_obbs(candidate_filtered_3d, T)
    centers, centers_2d, radii, valid_sphere = _gripper_spheres(grip_3d, grip_2d, img_center, obj_scale)
    radii_contact = radii * peak_radius_multiplier
    # The base radii already include SPHERE3D_GRIP_RADIUS_SCALE, so this is an
    # effective K-NN radius of 16x when the default base scale is 2x.
    radii_r8 = radii * 8.0
    radii_r8_contact = radii_r8 * peak_radius_multiplier

    per_frame_mean = np.full((n_cand, T), np.nan, dtype=float)
    per_frame_peak = np.full((n_cand, T), np.nan, dtype=float)
    per_frame_valid = np.zeros((n_cand, T), dtype=bool)
    mean_frac = np.zeros(n_cand, dtype=float)
    peak_frac = np.zeros(n_cand, dtype=float)
    mean_frac_r8 = np.zeros(n_cand, dtype=float)
    peak_frac_r8 = np.zeros(n_cand, dtype=float)

    for ci in range(n_cand):
        filtered = candidate_filtered_3d[ci]
        for t in range(min(T, len(filtered))):
            pts = filtered[t]
            if pts is None or not valid_sphere[t]:
                continue
            d = np.linalg.norm(pts - centers[t], axis=1)
            per_frame_mean[ci, t] = float(np.mean(d <= radii[t]))
            per_frame_peak[ci, t] = float(np.mean(d <= radii_contact[t]))
            per_frame_valid[ci, t] = True

        valid_t = per_frame_valid[ci]
        if valid_t.any():
            mean_vals = per_frame_mean[ci, valid_t]
            mean_frac[ci] = float(np.mean(mean_vals))
            peak_vals = per_frame_peak[ci, valid_t & np.isfinite(per_frame_peak[ci])]
            if len(peak_vals) > 0:
                n_top = max(
                    peak_min_frames,
                    int(np.ceil(len(peak_vals) * peak_top_fraction)),
                )
                n_top = min(len(peak_vals), n_top)
                top = np.partition(peak_vals, len(peak_vals) - n_top)[-n_top:]
                peak_frac[ci] = float(np.mean(top))

    # ── radius×8 pass (same valid_sphere mask, no new per-frame matrix stored) ─
    for ci in range(n_cand):
        filtered = candidate_filtered_3d[ci]
        r8_mean_vals: list[float] = []
        r8_peak_vals: list[float] = []
        for t in range(min(T, len(filtered))):
            pts = filtered[t]
            if pts is None or not valid_sphere[t]:
                continue
            d = np.linalg.norm(pts - centers[t], axis=1)
            r8_mean_vals.append(float(np.mean(d <= radii_r8[t])))
            r8_peak_vals.append(float(np.mean(d <= radii_r8_contact[t])))
        if r8_mean_vals:
            mean_frac_r8[ci] = float(np.mean(r8_mean_vals))
            if r8_peak_vals:
                n_top = max(
                    peak_min_frames,
                    int(np.ceil(len(r8_peak_vals) * peak_top_fraction)),
                )
                n_top = min(len(r8_peak_vals), n_top)
                top = np.partition(r8_peak_vals, len(r8_peak_vals) - n_top)[-n_top:]
                peak_frac_r8[ci] = float(np.mean(top))

    return SpherePointFractionResult(
        mean_fraction=mean_frac,
        peak_fraction=peak_frac,
        mean_fraction_r8=mean_frac_r8,
        peak_fraction_r8=peak_frac_r8,
        per_frame_fraction_mean=per_frame_mean,
        per_frame_fraction_peak=per_frame_peak,
        per_frame_valid=per_frame_valid,
        sphere_centers=centers,
        sphere_centers_2d=centers_2d,
        sphere_radii=radii,
        sphere_radii_contact=radii_contact,
        obj_scale=float(obj_scale),
    )


# ---------------------------------------------------------------------------
# Target-point score (retunable temporal-consistency term)
# ---------------------------------------------------------------------------
#
# Retain the per-point signals so callers can change the movement gate or
# aggregation without re-running the annotator.
#
# Persisted artifacts (see TargetPointScoreResult):
#   - score_ungated, score_motion_gated_2x  -- ready-to-use (n_start, n_target)
#     matrices; the gated matrix uses `2 * scene_baseline` (the offline default).
#   - point_motion                          -- (n_start, N) px/sec per tracked point
#   - point_visible_final                   -- (n_start, N) bool, visible at last frame
#   - point_in_target                       -- (n_start, n_target, N) bool
# Downstream can rebuild the gated score at any threshold:
#   sel       = point_visible_final & (point_motion > gate)
#   score[s,t]= point_in_target[s, t, sel[s]].mean()


@dataclass
class TargetPointScoreResult:
    score_ungated: np.ndarray                 # (n_start, n_target) — gate="none"
    score_motion_gated: np.ndarray            # (n_start, n_target) — at motion_gate_pxps
    point_motion: np.ndarray                  # (n_start, N) px/sec, NaN for never-visible
    point_visible_final: np.ndarray           # (n_start, N) bool
    point_in_target: np.ndarray               # (n_start, n_target, N) bool
    motion_gate_pxps: float                   # threshold used for score_motion_gated


def _per_point_mean_motion(
    tracks: np.ndarray,                # (T, N, 2)
    visibility: np.ndarray,            # (T, N) bool/float
    fps: float,
) -> np.ndarray:
    """Per-point mean frame-to-frame motion in px/sec; NaN where never visible."""
    T = tracks.shape[0]
    N = tracks.shape[1]
    if T < 2:
        return np.full(N, np.nan, dtype=float)
    disp = np.linalg.norm(tracks[1:] - tracks[:-1], axis=2)        # (T-1, N)
    vis_pair = (visibility[:-1] > 0) & (visibility[1:] > 0)        # (T-1, N)
    out = np.full(N, np.nan, dtype=float)
    counts = vis_pair.sum(axis=0)                                  # (N,)
    valid_pts = counts > 0
    if valid_pts.any():
        # Sum over visible-pair frames, divided by count, scaled to px/sec.
        sums = np.where(vis_pair, disp, 0.0).sum(axis=0)
        out[valid_pts] = (sums[valid_pts] / counts[valid_pts]) * fps
    return out


def compute_target_point_score(
    candidate_tracks: np.ndarray,             # (n_start, T, N, 2)
    candidate_visibility: np.ndarray,         # (n_start, T, N)
    target_boxes: np.ndarray,                 # (n_target, 4) xyxy
    fps: float,
    *,
    motion_gate_pxps: float = TEMPORAL_CONSISTENCY_GATE_PXPS,
) -> TargetPointScoreResult:
    """Per-(start, target) target-point score, plus the raw inputs the gate uses.

    The gated matrix uses ``motion_gate_pxps`` (default: twice the scene
    movement baseline, 11.726 px/s). Downstream can re-derive
    any other gate from ``point_motion`` + ``point_visible_final`` +
    ``point_in_target``.
    """
    candidate_tracks = np.asarray(candidate_tracks, dtype=float)
    candidate_visibility = np.asarray(candidate_visibility, dtype=float)
    target_boxes = np.asarray(target_boxes, dtype=float)
    n_start = len(candidate_tracks)
    n_target = len(target_boxes)
    T = candidate_tracks.shape[1] if n_start > 0 else 0
    N = candidate_tracks.shape[2] if n_start > 0 and candidate_tracks.ndim >= 3 else 0

    score_ungated = np.zeros((n_start, n_target), dtype=float)
    score_gated = np.zeros((n_start, n_target), dtype=float)
    point_motion = np.full((n_start, N), np.nan, dtype=float)
    point_visible_final = np.zeros((n_start, N), dtype=bool)
    point_in_target = np.zeros((n_start, n_target, N), dtype=bool)
    gate = float(motion_gate_pxps)

    if n_start == 0 or n_target == 0 or T < 1 or N == 0:
        return TargetPointScoreResult(
            score_ungated=score_ungated,
            score_motion_gated=score_gated,
            point_motion=point_motion,
            point_visible_final=point_visible_final,
            point_in_target=point_in_target,
            motion_gate_pxps=gate,
        )

    for si in range(n_start):
        tracks = candidate_tracks[si]
        vis = candidate_visibility[si]
        final_visible = vis[-1] > 0
        point_visible_final[si] = final_visible

        motion = _per_point_mean_motion(tracks, vis, fps)
        point_motion[si] = motion

        if not final_visible.any():
            continue

        final_pts = tracks[-1]                                      # (N, 2)
        for ti, box in enumerate(target_boxes):
            inside = (
                (final_pts[:, 0] >= box[0])
                & (final_pts[:, 0] <= box[2])
                & (final_pts[:, 1] >= box[1])
                & (final_pts[:, 1] <= box[3])
            )
            point_in_target[si, ti] = inside

            sel_ungated = final_visible & inside
            denom_ungated = int(final_visible.sum())
            score_ungated[si, ti] = (
                float(sel_ungated.sum() / denom_ungated) if denom_ungated > 0 else 0.0
            )

            mvt_ok = np.where(np.isfinite(motion), motion > gate, False)
            sel_gated_pool = final_visible & mvt_ok
            denom_gated = int(sel_gated_pool.sum())
            if denom_gated > 0:
                score_gated[si, ti] = float((sel_gated_pool & inside).sum() / denom_gated)

    return TargetPointScoreResult(
        score_ungated=score_ungated,
        score_motion_gated=score_gated,
        point_motion=point_motion,
        point_visible_final=point_visible_final,
        point_in_target=point_in_target,
        motion_gate_pxps=gate,
    )


def compute_centroid_traces(
    candidate_tracks: np.ndarray,             # (n_cand, T, N, 2)
    candidate_visibility: np.ndarray,         # (n_cand, T, N)
    *,
    min_visible_points: int = 1,
) -> np.ndarray:
    """Per-candidate centroid trace: mean of visible points per frame.

    Returns ``(n_cand, T, 2)`` float; entries are NaN where the candidate has
    fewer than ``min_visible_points`` visible-and-finite points that frame.
    Replaces shipping the full ``(n_cand, T, N, 2)`` cotracker tensor — the
    user explicitly wants one trace per candidate box.
    """
    tracks = np.asarray(candidate_tracks, dtype=float)
    vis = np.asarray(candidate_visibility, dtype=float)
    if tracks.ndim != 4 or tracks.shape[-1] < 2 or vis.shape != tracks.shape[:3]:
        n_cand = tracks.shape[0] if tracks.ndim >= 1 else 0
        T = tracks.shape[1] if tracks.ndim >= 2 else 0
        return np.full((n_cand, T, 2), np.nan, dtype=float)

    n_cand, T, _, _ = tracks.shape
    out = np.full((n_cand, T, 2), np.nan, dtype=float)
    valid = (vis > 0) & np.all(np.isfinite(tracks[..., :2]), axis=-1)  # (n,T,N)
    counts = valid.sum(axis=-1)                                        # (n,T)
    enough = counts >= int(min_visible_points)
    if not enough.any():
        return out

    masked_xy = np.where(valid[..., None], tracks[..., :2], 0.0)       # (n,T,N,2)
    sums = masked_xy.sum(axis=2)                                       # (n,T,2)
    safe_counts = np.where(enough, counts, 1).astype(float)
    means = sums / safe_counts[..., None]
    out[enough] = means[enough]
    return out


def _tip_centroid_per_frame(
    tracks_3d: np.ndarray | None,
    visibility: np.ndarray | None,
    *,
    tip_k: int = SOFT_SNR_DEPTH_RESCUE_TIP_K,
) -> np.ndarray | None:
    if tracks_3d is None or visibility is None:
        return None
    xyz = np.asarray(tracks_3d, dtype=float)
    vis = np.asarray(visibility, dtype=float)
    if xyz.ndim != 3 or xyz.shape[-1] < 3 or vis.shape != xyz.shape[:2]:
        return None
    T = xyz.shape[0]
    centroid = np.full((T, 3), np.nan, dtype=float)
    for frame_idx in range(T):
        valid = (vis[frame_idx] > 0) & np.all(np.isfinite(xyz[frame_idx, :, :3]), axis=1)
        if not valid.any():
            continue
        pts = xyz[frame_idx, valid, :3]
        k = min(int(tip_k), len(pts))
        centroid[frame_idx] = pts[np.argsort(pts[:, 2])[:k]].mean(axis=0)
    return centroid


def compute_depth_rescue_keep_mask(
    candidate_tracks_3d: np.ndarray | None,
    candidate_visibility: np.ndarray | None,
    gripper_tracks_3d: np.ndarray | None,
    gripper_visibility: np.ndarray | None,
    robotseg_overlap: np.ndarray | None,
    *,
    overlap_thresh: float = SOFT_SNR_DEPTH_RESCUE_OVERLAP_THRESH,
    depth_thresh: float = SOFT_SNR_DEPTH_RESCUE_DEPTH_THRESH,
    tip_k: int = SOFT_SNR_DEPTH_RESCUE_TIP_K,
) -> np.ndarray:
    overlap = np.asarray(robotseg_overlap if robotseg_overlap is not None else [], dtype=float).reshape(-1)
    if len(overlap) == 0:
        return np.ones((0,), dtype=bool)

    keep = overlap < float(overlap_thresh)
    if keep.all():
        return keep

    cand_xyz = np.asarray(candidate_tracks_3d, dtype=float) if candidate_tracks_3d is not None else None
    cand_vis = np.asarray(candidate_visibility, dtype=float) if candidate_visibility is not None else None
    if (
        cand_xyz is None
        or cand_vis is None
        or cand_xyz.ndim != 4
        or cand_xyz.shape[-1] < 3
        or cand_vis.shape != cand_xyz.shape[:3]
        or len(cand_xyz) != len(keep)
    ):
        return keep

    grip_tip = _tip_centroid_per_frame(gripper_tracks_3d, gripper_visibility, tip_k=tip_k)
    if grip_tip is None:
        return keep

    for cand_idx in range(len(keep)):
        if keep[cand_idx]:
            continue
        obj_tip = _tip_centroid_per_frame(cand_xyz[cand_idx], cand_vis[cand_idx], tip_k=tip_k)
        if obj_tip is None:
            continue
        n_frames = min(len(grip_tip), len(obj_tip))
        if n_frames == 0:
            continue
        dists = [
            np.linalg.norm(grip_tip[frame_idx] - obj_tip[frame_idx])
            for frame_idx in range(n_frames)
            if not (np.isnan(grip_tip[frame_idx]).any() or np.isnan(obj_tip[frame_idx]).any())
        ]
        if dists and float(np.median(dists)) > float(depth_thresh):
            keep[cand_idx] = True
    return keep


def compute_candidate_median_abs_dz(
    candidate_tracks_3d: np.ndarray | None,    # (n_cand, T, N, 3)
    candidate_visibility: np.ndarray | None,   # (n_cand, T, N)
    gripper_tracks_3d: np.ndarray | None,      # (T, N_g, 3)
    gripper_visibility: np.ndarray | None,     # (T, N_g)
    *,
    tip_k: int = SOFT_SNR_DEPTH_RESCUE_TIP_K,
) -> np.ndarray:
    """Per-candidate median fingertip |Δz| against the gripper fingertip.

    Per frame, the fingertip is the centroid of the ``tip_k`` lowest-z (3D) tracked points;
    the score is the median over frames of ``|grip_tip_z - obj_tip_z|``.
    Returns ``NaN`` for candidates with no jointly valid frame (this
    propagates to a non-finite robot-overlap penalty, exactly as offline).
    """
    cand_xyz = np.asarray(candidate_tracks_3d, dtype=float) if candidate_tracks_3d is not None else None
    cand_vis = np.asarray(candidate_visibility, dtype=float) if candidate_visibility is not None else None
    n_cand = cand_xyz.shape[0] if cand_xyz is not None and cand_xyz.ndim == 4 else 0
    values = np.full(n_cand, np.nan, dtype=float)
    if n_cand == 0 or cand_vis is None or cand_vis.shape != cand_xyz.shape[:3]:
        return values

    grip_tip = _tip_centroid_per_frame(gripper_tracks_3d, gripper_visibility, tip_k=tip_k)
    if grip_tip is None:
        return values

    for cand_idx in range(n_cand):
        obj_tip = _tip_centroid_per_frame(cand_xyz[cand_idx], cand_vis[cand_idx], tip_k=tip_k)
        if obj_tip is None:
            continue
        n_frames = min(len(grip_tip), len(obj_tip))
        dzs = [
            abs(grip_tip[t][2] - obj_tip[t][2])
            for t in range(n_frames)
            if not (np.isnan(grip_tip[t]).any() or np.isnan(obj_tip[t]).any())
        ]
        if dzs:
            values[cand_idx] = float(np.median(dzs))
    return values


def soft_snr_quad_relief_penalty(
    robotseg_overlap: np.ndarray,
    candidate_median_abs_dz: np.ndarray,
    *,
    overlap_threshold: float = ADAPTIVE_DET_SOFT_SNR_SP8_OVERLAP_THRESH,
) -> np.ndarray:
    """Continuous quadratic robot-overlap penalty with fingertip-depth relief.

    A non-finite Δz (e.g.
    when depth data is unavailable) is treated as dz=0: no relief is granted
    and the full quadratic penalty applies, but the candidate is not dropped.
    """
    overlap = np.asarray(robotseg_overlap, dtype=float)
    dz = np.asarray(candidate_median_abs_dz, dtype=float)
    dz = np.where(np.isfinite(dz), dz, 0.0)
    relief = np.clip(
        (dz - SOFT_SNR_QUAD_RELIEF_DEPTH_LOW)
        / max(SOFT_SNR_QUAD_RELIEF_DEPTH_HIGH - SOFT_SNR_QUAD_RELIEF_DEPTH_LOW, 1e-6),
        0.0,
        1.0,
    )
    overlap_excess = np.clip(
        (overlap - float(overlap_threshold)) / max(1.0 - float(overlap_threshold), 1e-6),
        0.0,
        1.0,
    )
    penalty = (
        SOFT_SNR_QUAD_RELIEF_PENALTY_WEIGHT
        * (overlap_excess ** 2)
        * (1.0 - SOFT_SNR_QUAD_RELIEF_RELIEF_STRENGTH * relief)
    )
    penalty = penalty + (
        SOFT_SNR_QUAD_RELIEF_HARD_BONUS
        * (overlap > SOFT_SNR_QUAD_RELIEF_HARD_OVERLAP)
        * (1.0 - relief)
    )
    return penalty


# ---------------------------------------------------------------------------
# JSON-friendly array bundling
# ---------------------------------------------------------------------------
#
# The heavy lightweight-scoring arrays (centroid traces + retunable
# target-point arrays + optional per-frame matrices) are inlined into the
# JSONL as a single base64-encoded ``np.savez_compressed`` blob. This keeps
# the on-disk size ~10x smaller than nested rounded-float JSON without
# adding any new files (the user has tight inode constraints).
#
# Round-trip is lossless except for the deliberate float16 downcast on the
# floating-point arrays, which is the same precision the heavy NPZ uses.


_FLOAT_KEYS_FLOAT16 = (
    "centroid_traces",
    "gripper_rep_track_2d",
    "gripper_rep_track_3d",
    "target_point_motion_pxps",
    "sphere3d_per_frame_mean",
    "sphere3d_per_frame_peak",
    "robotseg_per_frame_overlap",
    "robotseg_per_frame_counts",
)


def encode_arrays_blob(arrays: Mapping[str, np.ndarray | None]) -> str | None:
    """Serialize a dict of numpy arrays as a base64 ``np.savez_compressed`` blob.

    ``None`` entries and empty arrays are dropped. Floating-point arrays are
    cast to float16 before saving (matches the precision used by the heavy
    NPZ). Returns ``None`` if no arrays remain.
    """
    payload: dict[str, np.ndarray] = {}
    for key, value in arrays.items():
        if value is None:
            continue
        arr = np.asarray(value)
        if arr.size == 0:
            continue
        if np.issubdtype(arr.dtype, np.floating) and key in _FLOAT_KEYS_FLOAT16:
            arr = arr.astype(np.float16)
        payload[key] = arr
    if not payload:
        return None
    buf = io.BytesIO()
    np.savez_compressed(buf, **payload)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def decode_arrays_blob(blob: str | None) -> dict[str, np.ndarray]:
    """Inverse of :func:`encode_arrays_blob`. Returns ``{}`` for falsy input."""
    if not blob:
        return {}
    raw = base64.b64decode(blob.encode("ascii"))
    with np.load(io.BytesIO(raw)) as npz:
        return {key: np.asarray(npz[key]) for key in npz.files}


# ---------------------------------------------------------------------------
# Adaptive detection and soft-SNR candidate selection
# ---------------------------------------------------------------------------

def minmax_norm(arr: np.ndarray) -> np.ndarray:
    """Min-max into [0, 1] within one sample. Matches signals.minmax_norm:
    non-finite → 0, all-equal-finite → 1.
    """
    arr = np.asarray(arr, dtype=float)
    out = np.zeros_like(arr)
    finite = np.isfinite(arr)
    if not finite.any():
        return out
    lo = float(arr[finite].min())
    hi = float(arr[finite].max())
    if hi > lo:
        out[finite] = (arr[finite] - lo) / (hi - lo)
    else:
        out[finite] = 1.0
    return out


@dataclass
class AdaptiveDetSoftSnrSp8Result:
    best_start: int | None
    final_score: np.ndarray
    components: dict[str, np.ndarray]
    keep_mask: np.ndarray
    phase_snr_norm: np.ndarray
    sphere_norm: np.ndarray
    adaptive_det_weight: np.ndarray
    constants: dict[str, float | str]
    rejection_reason: str | None


def adaptive_det_soft_snr_sp8_constants() -> dict[str, float | str]:
    return {
        "signal_name": ADAPTIVE_DET_SOFT_SNR_SP8_SIGNAL_NAME,
        "soft_snr_alpha": ADAPTIVE_DET_SOFT_SNR_SP8_ALPHA,
        "soft_snr_beta": ADAPTIVE_DET_SOFT_SNR_SP8_BETA,
        "soft_snr_eps": ADAPTIVE_DET_SOFT_SNR_SP8_EPS,
        "robotseg_overlap_threshold": ADAPTIVE_DET_SOFT_SNR_SP8_OVERLAP_THRESH,
        "snr_weight": ADAPTIVE_DET_SOFT_SNR_SP8_SNR_WEIGHT,
        "det_weight_base": ADAPTIVE_DET_SOFT_SNR_SP8_DET_WEIGHT_BASE,
        "det_weight_sphere_reduction": ADAPTIVE_DET_SOFT_SNR_SP8_DET_WEIGHT_SPHERE_REDUCTION,
        "sphere_weight": ADAPTIVE_DET_SOFT_SNR_SP8_SPHERE_WEIGHT,
        "sphere_field": "sphere3d_point_fraction_mean_r8",
        "quad_penalty_weight": SOFT_SNR_QUAD_RELIEF_PENALTY_WEIGHT,
        "quad_relief_strength": SOFT_SNR_QUAD_RELIEF_RELIEF_STRENGTH,
        "quad_hard_overlap": SOFT_SNR_QUAD_RELIEF_HARD_OVERLAP,
        "quad_hard_bonus": SOFT_SNR_QUAD_RELIEF_HARD_BONUS,
        "depth_relief_low": SOFT_SNR_QUAD_RELIEF_DEPTH_LOW,
        "depth_relief_high": SOFT_SNR_QUAD_RELIEF_DEPTH_HIGH,
    }


def pick_adaptive_det_soft_snr_sp8(
    *,
    detection_confidence: np.ndarray,
    phase_snr: np.ndarray,
    sphere_point_fraction_r8: np.ndarray,
    robotseg_overlap: np.ndarray,
    candidate_median_abs_dz: np.ndarray,
    overlap_threshold: float = ADAPTIVE_DET_SOFT_SNR_SP8_OVERLAP_THRESH,
) -> AdaptiveDetSoftSnrSp8Result:
    """Select candidates using motion, detection, and depth-relieved robot overlap.

    Uses the paper's continuous quadratic overlap penalty
    (paper_full / paper_prog_6) exactly:

        base    = 0.5*minmax(phase_snr) + (0.75-0.15*sphere_norm)*det
                  + 0.3*sphere_norm
        penalty = quadratic robot-overlap penalty with fingertip-depth relief
                  (see `soft_snr_quad_relief_penalty`)
        final   = base - penalty

    The soft-SNR term is normalized from the *raw* phase-SNR (no robot
    pre-filter), and robot overlap is handled purely by the continuous
    penalty — there is no hard overlap cutoff. ``keep_mask`` is the set of
    candidates with a finite final score (a non-finite fingertip Δz drops a
    candidate, exactly as offline). Candidate order is preserved.
    """
    det = np.asarray(detection_confidence, dtype=float)
    snr = np.asarray(phase_snr, dtype=float)
    sphere = np.asarray(sphere_point_fraction_r8, dtype=float)
    overlap = np.asarray(robotseg_overlap, dtype=float)
    dz = np.asarray(candidate_median_abs_dz, dtype=float)
    n_cand = len(det)
    if (
        len(snr) != n_cand
        or len(sphere) != n_cand
        or len(overlap) != n_cand
        or len(dz) != n_cand
    ):
        raise ValueError("adaptive_det_soft_snr_sp8 arrays must have the same candidate length")

    snr_norm = minmax_norm(snr)
    sphere_norm = minmax_norm(sphere)
    adaptive_det_weight = (
        ADAPTIVE_DET_SOFT_SNR_SP8_DET_WEIGHT_BASE
        - ADAPTIVE_DET_SOFT_SNR_SP8_DET_WEIGHT_SPHERE_REDUCTION * sphere_norm
    )

    snr_part = ADAPTIVE_DET_SOFT_SNR_SP8_SNR_WEIGHT * snr_norm
    det_part = adaptive_det_weight * det
    sphere_part = ADAPTIVE_DET_SOFT_SNR_SP8_SPHERE_WEIGHT * sphere_norm
    penalty = soft_snr_quad_relief_penalty(
        overlap, dz, overlap_threshold=overlap_threshold,
    )

    final = snr_part + det_part + sphere_part - penalty
    final[~np.isfinite(final)] = -np.inf
    keep = np.isfinite(final)

    rejection_reason = None
    best_start = None
    if keep.any():
        best_start = int(np.argmax(final))
    else:
        rejection_reason = "adaptive_det_soft_snr_sp8_no_finite_score"

    return AdaptiveDetSoftSnrSp8Result(
        best_start=best_start,
        final_score=final,
        components={
            "phase_snr_norm_weighted": snr_part,
            "adaptive_detection_conf_weighted": det_part,
            "sphere3d_point_fraction_mean_r8_norm_weighted": sphere_part,
            "robot_overlap_penalty": -penalty,
        },
        keep_mask=keep,
        phase_snr_norm=snr_norm,
        sphere_norm=sphere_norm,
        adaptive_det_weight=adaptive_det_weight,
        constants=adaptive_det_soft_snr_sp8_constants(),
        rejection_reason=rejection_reason,
    )


def select_target_box_density(target_point_scores_row, target_boxes, start_box=None):
    """Density-based target-box selection (the paper method).

    Scores each target *candidate* box by ``point_score / sqrt(area / max_area)``
    so that boxes which catch many tracked points without being needlessly large
    win. Candidates smaller than ``0.5 * start_box_area`` are gated out. Returns
    the best target **index**, or ``None`` when the inputs are missing/degenerate
    (the caller should then fall back to the argmax target pick).

    Ported from the offline viewer's ``_select_target_box_density`` so the live
    pipeline and the paper plots share one implementation.

    Args:
        target_point_scores_row: per-target point scores for the winning start
            candidate, i.e. row ``winner_idx`` of ``target_point_score_ungated``;
            length must equal ``len(target_boxes)``.
        target_boxes: candidate target boxes as xyxy rows.
        start_box: the winning start box (xyxy); used only for the area gate.
    """
    if target_boxes is None or len(target_boxes) == 0:
        return None
    scores = target_point_scores_row
    if scores is None or len(scores) != len(target_boxes):
        return None

    areas = [
        max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
        for b in target_boxes
    ]
    max_area = max(areas) if areas else 0.0
    if max_area <= 0.0:
        return None

    min_area = 0.0
    if start_box is not None and len(start_box) >= 4:
        min_area = 0.5 * (
            max(0.0, float(start_box[2]) - float(start_box[0]))
            * max(0.0, float(start_box[3]) - float(start_box[1]))
        )

    density_scores = [
        float(pt) / (area / max_area) ** 0.5
        if area >= min_area and area > 0.0 and float(pt) > 0.0
        else 0.0
        for pt, area in zip(scores, areas)
    ]
    if not density_scores or max(density_scores) <= 0.0:
        return None
    return int(max(range(len(density_scores)), key=lambda i: density_scores[i]))
