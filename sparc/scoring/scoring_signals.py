"""Fine-grained scoring signals for candidate selection.

Implements the four sub-scores from the best-sweep scoring variant:
  - Phase SNR movement score
  - Detection confidence gate
  - Single dominant detection gate
  - 3D gripper sphere point fraction

These are computed from in-memory pipeline data (raw_tracks_data) and stored
in confidence_breakdown so downstream reweighting is possible without re-running.
"""
from __future__ import annotations

import numpy as np


# ── Scoring constants ────────────────────────────────────────────────────

_PCD_FILTER_ENABLED = True
_PCD_NB_NEIGHBORS = 20
_PCD_STD_RATIO = 1.0
_PCD_DEPTH_MAD_K = 2.0
_MIN_VALID_POINTS = 3
_GRIP_RADIUS_K_NEIGHBORS = 5
_GRIP_RADIUS_SCALE = 2.0
_GRIP_RADIUS_MIN = 0.05
_GRIP_RADIUS_MAX_OBJ_SCALE = 1
_POINT_FRACTION_ACTIVE_THRESHOLD = 0.10
_PHASE_SNR_ALPHA = 2.0
_PHASE_SNR_BETA = 1.6
_PHASE_SNR_EPS = 1.0
_GATE_HIGH_THRESH = 0.5
_GATE_LOW_THRESH = 0.25
_REPLAY_ROBOT_POINT_OVERLAP_THRESH = 0.3
_PRIMARY_TRACKS_KEY = "tracks_raw"
_ALTERNATE_TRACKS_KEY = "tracks_smooth"
_TRACK_SMOOTHING_WINDOW_SECONDS = 0.3


# ── Task group classification ─────────────────────────────────────────────────

def get_task_group(action: str) -> str:
    a = (action or "").lower()
    if any(w in a for w in ["pick", "place", "move", "lift", "remove", "transfer",
                             "hand over", "stack", "unstack", "carry", "pass", "receive",
                             "put away", "retrieve", "throw", "hang", "grasp", "uncover",
                             "separate", "grab"]):
        return "movement_relocate"
    if any(w in a for w in ["open", "close"]):
        return "non_movement_open_close"
    if any(w in a for w in ["push", "pull", "turn", "rotate", "flip", "flatten",
                             "smooth", "spread", "clamp", "manipulate"]):
        return "movement_articulate"
    if any(w in a for w in ["press", "switch"]):
        return "non_movement_articulate"
    if any(w in a for w in ["pour", "scoop", "wash", "wipe", "sweep"]):
        return "non_movement_fluid"
    if any(w in a for w in ["cut", "peeler", "fold", "unfold", "plug", "attach"]):
        return "non_movement_tool_or_shape"
    if any(w in a for w in ["stamp"]):
        return "other_interaction"
    return "other"


def get_task_adaptive_weights(task_group: str) -> dict:
    if task_group == "non_movement_articulate":
        return {"snr_w": 0.0, "det_w": 1.0, "gate_w": 0.2, "point_fraction_w": 0.35}
    return {"snr_w": 0.5, "det_w": 0.5, "gate_w": 0.2, "point_fraction_w": 0.20}


# ── Phase SNR signal ──────────────────────────────────────────────────────────

def _build_phase_masks_scaled(
    grasp_phases: list, T: int, n_orig: int
) -> tuple[np.ndarray, np.ndarray]:
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
    non_interact[:margin] = False
    non_interact[-margin:] = False
    return interact, non_interact


def _estimate_n_orig(grasp_phases: list, T_fallback: int) -> int:
    ends = []
    for ph in (grasp_phases or []):
        if isinstance(ph, dict):
            ends.append(ph.get("end_frame", 0))
        elif len(ph) >= 2:
            ends.append(ph[1])
    return int(max(ends)) + 1 if ends else T_fallback


def _box_diag(box: np.ndarray) -> float:
    box = np.asarray(box, dtype=float).reshape(-1)
    if box.size != 4:
        return 0.0
    width = max(0.0, float(box[2] - box[0]))
    height = max(0.0, float(box[3] - box[1]))
    return float(np.hypot(width, height))


def _smooth_tracks_moving_average(
    tracks: np.ndarray,
    trackvis: np.ndarray | None,
    window_frames: int,
) -> np.ndarray:
    """Smooth each contiguous visible track segment with an edge-padded mean."""
    tracks = np.asarray(tracks)
    if window_frames <= 1 or tracks.ndim != 3:
        return np.array(tracks, copy=True)

    smoothed = np.array(tracks, copy=True)
    if trackvis is None:
        trackvis = np.ones(tracks.shape[:2], dtype=bool)
    trackvis = np.asarray(trackvis)
    if trackvis.shape[:2] != tracks.shape[:2]:
        trackvis = np.ones(tracks.shape[:2], dtype=bool)

    for pt_idx in range(tracks.shape[1]):
        valid = trackvis[:, pt_idx] > 0
        if not valid.any():
            continue

        valid_idx = np.where(valid)[0]
        segment_start = 0
        while segment_start < len(valid_idx):
            segment_end = segment_start + 1
            while (
                segment_end < len(valid_idx)
                and valid_idx[segment_end] == valid_idx[segment_end - 1] + 1
            ):
                segment_end += 1

            segment = valid_idx[segment_start:segment_end]
            if len(segment) > 1:
                segment_window = min(window_frames, len(segment))
                kernel = np.ones(segment_window, dtype=float) / segment_window
                pad_left = segment_window // 2
                pad_right = segment_window - 1 - pad_left
                for coord in range(tracks.shape[2]):
                    values = tracks[segment, pt_idx, coord]
                    padded = np.pad(values, (pad_left, pad_right), mode="edge")
                    smoothed[segment, pt_idx, coord] = np.convolve(
                        padded,
                        kernel,
                        mode="valid",
                    )
            segment_start = segment_end

    return smoothed


def _smooth_candidate_tracks(raw_tracks_data: dict, fps: float) -> np.ndarray | None:
    tracks = raw_tracks_data.get("cotracker_tracks")
    if tracks is None:
        return None
    tracks = np.asarray(tracks)
    if tracks.ndim != 4:
        return tracks
    if len(tracks) == 0:
        return tracks
    visibility = raw_tracks_data.get("cotracker_visibility")
    visibility = np.asarray(visibility) if visibility is not None else None
    window_frames = max(1, int(round(_TRACK_SMOOTHING_WINDOW_SECONDS * float(fps))))
    return np.stack(
        [
            _smooth_tracks_moving_average(
                tracks[i],
                visibility[i] if visibility is not None and len(visibility) > i else None,
                window_frames,
            )
            for i in range(len(tracks))
        ],
        axis=0,
    )


def compute_motion_phase_stats(
    tracks_list: list[np.ndarray],
    grasp_phases: list,
    fps: float,
    candidate_boxes: np.ndarray | None = None,
    alpha: float = _PHASE_SNR_ALPHA,
    beta: float = _PHASE_SNR_BETA,
    eps: float = _PHASE_SNR_EPS,
) -> list[dict]:
    """Per-candidate movement and phase-SNR ingredients for JSONL replay."""
    rows: list[dict] = []
    boxes = np.asarray(candidate_boxes, dtype=float) if candidate_boxes is not None else None
    for cand_idx, tracks_raw in enumerate(tracks_list):
        tracks = np.asarray(tracks_raw, dtype=np.float32)
        bbox_diag = _box_diag(boxes[cand_idx]) if boxes is not None and len(boxes) > cand_idx else 0.0
        invalid = {
            "n_total": 0,
            "n_valid_points": 0,
            "bbox_diag": bbox_diag,
            "mean_movement": -np.inf,
            "median_movement": -np.inf,
            "total_movement": -np.inf,
            "norm_movement": -np.inf,
            "interact_frames": 0,
            "non_interact_frames": 0,
            "interact_mean": 0.0,
            "non_interact_mean": 0.0,
            "interact_sum": 0.0,
            "non_interact_sum": 0.0,
            "phase_fallback": True,
            "phase_snr_raw": -np.inf,
        }
        if tracks.ndim != 3 or tracks.shape[0] < 2:
            rows.append(invalid)
            continue

        point_mask = ~np.all(tracks == -1, axis=(0, 2))
        if not point_mask.any():
            rows.append(invalid)
            continue

        tracks = tracks[:, point_mask]
        disp = np.linalg.norm(np.diff(tracks, axis=0), axis=2)
        frame_disp = disp.mean(axis=1) * fps
        n_total = int(len(frame_disp))
        mean_movement = float(frame_disp.mean()) if n_total else 0.0
        total_movement = float(disp.mean(axis=1).sum()) if n_total else 0.0
        per_frame_median = [
            float(np.median(disp[t])) * fps if disp.shape[1] > 0 else 0.0
            for t in range(len(disp))
        ]
        median_movement = float(np.mean(per_frame_median)) if per_frame_median else 0.0
        norm_movement = median_movement / bbox_diag if bbox_diag > 0 else -np.inf

        if not grasp_phases or len(grasp_phases) <= 1:
            rows.append({
                "n_total": n_total,
                "n_valid_points": int(point_mask.sum()),
                "bbox_diag": bbox_diag,
                "mean_movement": mean_movement,
                "median_movement": median_movement,
                "total_movement": total_movement,
                "norm_movement": norm_movement,
                "interact_frames": n_total,
                "non_interact_frames": 0,
                "interact_mean": mean_movement,
                "non_interact_mean": 0.0,
                "interact_sum": float(frame_disp.sum()) if n_total else 0.0,
                "non_interact_sum": 0.0,
                "phase_fallback": True,
                "phase_snr_raw": mean_movement,
            })
            continue

        T = len(tracks)
        n_orig = _estimate_n_orig(grasp_phases, T)
        interact_mask, non_interact_mask = _build_phase_masks_scaled(grasp_phases, T, n_orig)
        d_int = interact_mask[1:][:n_total]
        d_non = non_interact_mask[1:][:n_total]

        interact_frames = int(d_int.sum())
        non_interact_frames = int(d_non.sum())
        fallback = non_interact_frames < 3 or interact_frames == 0
        interact_mean = float(frame_disp[d_int].mean()) if d_int.any() else mean_movement
        non_interact_mean = float(frame_disp[d_non].mean()) if d_non.any() else 0.0
        interact_sum = float(frame_disp[d_int].sum()) if d_int.any() else float(frame_disp.sum())
        non_interact_sum = float(frame_disp[d_non].sum()) if d_non.any() else 0.0
        phase_snr_raw = (
            mean_movement
            if fallback
            else interact_mean ** alpha / (non_interact_mean + eps) ** beta
        )
        rows.append({
            "n_total": n_total,
            "n_valid_points": int(point_mask.sum()),
            "bbox_diag": bbox_diag,
            "mean_movement": mean_movement,
            "median_movement": median_movement,
            "total_movement": total_movement,
            "norm_movement": norm_movement,
            "interact_frames": interact_frames,
            "non_interact_frames": non_interact_frames,
            "interact_mean": interact_mean,
            "non_interact_mean": non_interact_mean,
            "interact_sum": interact_sum,
            "non_interact_sum": non_interact_sum,
            "phase_fallback": bool(fallback),
            "phase_snr_raw": float(phase_snr_raw),
        })
    return rows


def _round_optional(value, ndigits: int = 6):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return round(value, ndigits) if np.isfinite(value) else None


def _jsonable_phases(grasp_phases: list) -> list:
    out = []
    for ph in grasp_phases or []:
        if isinstance(ph, dict):
            out.append({
                key: (int(val) if key in {"start_frame", "end_frame"} and val is not None else val)
                for key, val in ph.items()
            })
        elif len(ph) >= 2:
            item = [int(ph[0]), int(ph[1])]
            if len(ph) > 2:
                item.append(ph[2])
            out.append(item)
    return out


def _prefixed_motion_fields(prefix: str, motion: dict, snr_value: float | None) -> dict:
    return {
        f"{prefix}_bbox_diag": _round_optional(motion.get("bbox_diag")),
        f"{prefix}_mean_movement": _round_optional(motion.get("mean_movement")),
        f"{prefix}_median_movement": _round_optional(motion.get("median_movement")),
        f"{prefix}_total_movement": _round_optional(motion.get("total_movement")),
        f"{prefix}_norm_movement": _round_optional(motion.get("norm_movement")),
        f"{prefix}_n_valid_track_points": int(motion.get("n_valid_points", 0)),
        f"{prefix}_phase_interact_frames": int(motion.get("interact_frames", 0)),
        f"{prefix}_phase_non_interact_frames": int(motion.get("non_interact_frames", 0)),
        f"{prefix}_phase_total_frames": int(motion.get("n_total", 0)),
        f"{prefix}_phase_interact_mean": _round_optional(motion.get("interact_mean")),
        f"{prefix}_phase_non_interact_mean": _round_optional(motion.get("non_interact_mean")),
        f"{prefix}_phase_interact_sum": _round_optional(motion.get("interact_sum")),
        f"{prefix}_phase_non_interact_sum": _round_optional(motion.get("non_interact_sum")),
        f"{prefix}_phase_fallback": bool(motion.get("phase_fallback", True)),
        f"{prefix}_snr_score_raw": (
            round(float(snr_value), 6)
            if snr_value is not None and np.isfinite(snr_value)
            else None
        ),
    }


# ── Gate signal ───────────────────────────────────────────────────────────────

def compute_gate_scores(
    candidate_confidences: np.ndarray,
    high_thresh: float = _GATE_HIGH_THRESH,
    low_thresh: float = _GATE_LOW_THRESH,
) -> np.ndarray:
    """1.0 for the sole dominant candidate, 0.0 everywhere else."""
    confs = np.asarray(candidate_confidences, dtype=float)
    high_mask = confs > high_thresh
    low_mask = confs > low_thresh
    out = np.zeros(len(confs), dtype=float)
    if high_mask.sum() == 1 and low_mask.sum() == 1:
        out[high_mask] = 1.0
    return out


# ── 3D point fraction signal ──────────────────────────────────────────────────

def _get_open3d():
    try:
        import open3d as o3d
        return o3d
    except ImportError:
        return None


def _pcd_inlier_indices(points_xyz: np.ndarray, nb_neighbors: int, std_ratio: float) -> np.ndarray:
    n = len(points_xyz)
    all_idx = np.arange(n, dtype=int)
    idx = all_idx

    if n >= max(3, nb_neighbors):
        try:
            o3d = _get_open3d()
            if o3d is None:
                raise ImportError
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_xyz[:, :3].astype(np.float64))
            _, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
            if len(ind) >= 1:
                idx = np.asarray(ind, dtype=int)
        except Exception:
            pass

    if _PCD_DEPTH_MAD_K is not None and len(idx) >= 3:
        z = points_xyz[idx, 2]
        med = float(np.median(z))
        mad = float(np.median(np.abs(z - med)))
        if mad > 1e-9:
            keep = np.abs(z - med) <= _PCD_DEPTH_MAD_K * mad
            if keep.any():
                idx = idx[keep]

    return idx if len(idx) >= 1 else all_idx


def _filtered_3d_points_per_frame(
    xyz: np.ndarray,  # (T, N, 3)
    vis: np.ndarray,  # (T, N)
) -> list[np.ndarray | None]:
    result: list[np.ndarray | None] = [None] * xyz.shape[0]
    for t in range(xyz.shape[0]):
        valid = (vis[t] > 0) & np.all(np.isfinite(xyz[t, :, :3]), axis=1)
        if not valid.any():
            continue
        pts = xyz[t, valid, :3]
        if _PCD_FILTER_ENABLED and len(pts) >= max(3, _PCD_NB_NEIGHBORS):
            pts = pts[_pcd_inlier_indices(pts, _PCD_NB_NEIGHBORS, _PCD_STD_RATIO)]
        if len(pts) >= _MIN_VALID_POINTS:
            result[t] = pts
    return result


def _compute_gripper_spheres(
    grip_xyz: np.ndarray,   # (T, N, 3)
    grip_vis: np.ndarray,   # (T, N)
    grip_2d: np.ndarray,    # (T, N, 2)
    candidate_boxes: np.ndarray,  # (n_candidates, 4) for radius cap
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (centers [T,3], radii [T], frame_valid [T])."""
    T = grip_xyz.shape[0]
    grip_centers = np.full((T, 3), np.nan, dtype=float)
    grip_radii = np.zeros(T, dtype=float)

    filtered_3d = _filtered_3d_points_per_frame(grip_xyz, grip_vis)

    # Estimate image center from candidate boxes
    img_cx = float(candidate_boxes[:, 2].max()) / 2.0 if len(candidate_boxes) > 0 else 0.0
    img_cy = float(candidate_boxes[:, 3].max()) / 2.0 if len(candidate_boxes) > 0 else 0.0

    for t in range(T):
        pts_3d = filtered_3d[t]
        if pts_3d is None:
            continue
        pts_2d = grip_2d[t]
        vis_mask = grip_vis[t] > 0
        if not vis_mask.any():
            continue
        pts_2d_valid = pts_2d[vis_mask]
        dists = np.hypot(pts_2d_valid[:, 0] - img_cx, pts_2d_valid[:, 1] - img_cy)
        closest_idx = int(np.argmin(dists))
        grip_centers[t] = pts_3d[closest_idx] if len(pts_3d) > closest_idx else pts_3d[0]

        k = min(_GRIP_RADIUS_K_NEIGHBORS, len(pts_3d) - 1)
        if k > 0:
            nn_dists = np.sort(np.linalg.norm(pts_3d - grip_centers[t], axis=1))[1:k + 1]
            grip_radii[t] = max(float(np.median(nn_dists)) * _GRIP_RADIUS_SCALE, _GRIP_RADIUS_MIN)
        else:
            grip_radii[t] = _GRIP_RADIUS_MIN

    # Radius cap based on typical object scale
    if len(candidate_boxes) > 0:
        diags = np.sqrt((candidate_boxes[:, 2] - candidate_boxes[:, 0]) ** 2
                        + (candidate_boxes[:, 3] - candidate_boxes[:, 1]) ** 2)
        obj_scale = float(np.median(diags)) * 0.5 if len(diags) > 0 else np.inf
        if np.isfinite(obj_scale) and obj_scale > 0:
            grip_radii = np.minimum(grip_radii, _GRIP_RADIUS_MAX_OBJ_SCALE * obj_scale)

    frame_valid = np.all(np.isfinite(grip_centers), axis=1)
    return grip_centers, grip_radii, frame_valid


def compute_3d_point_fraction_scores(
    obj_tracks_3d: np.ndarray | None,   # (n_candidates, T, N, 3) float16/32
    obj_visibility: np.ndarray | None,   # (n_candidates, T, N)
    gripper_tracks: np.ndarray | None,   # (T, N, 2)
    gripper_tracks_3d: np.ndarray | None,# (T, N, 3) float16/32
    gripper_visibility: np.ndarray | None,# (T, N)
    candidate_boxes: np.ndarray,         # (n_candidates, 4)
) -> np.ndarray:
    """Mean fraction of each candidate's 3D pts inside the gripper sphere."""
    n_candidates = len(candidate_boxes)
    if (obj_tracks_3d is None or gripper_tracks_3d is None
            or gripper_tracks is None or gripper_visibility is None
            or obj_visibility is None):
        return np.zeros(n_candidates, dtype=float)

    obj_xyz = np.asarray(obj_tracks_3d, dtype=float)
    obj_vis = np.asarray(obj_visibility, dtype=float)
    grip_xyz = np.asarray(gripper_tracks_3d, dtype=float)
    grip_vis = np.asarray(gripper_visibility, dtype=float)
    grip_2d = np.asarray(gripper_tracks, dtype=float)

    grip_centers, grip_radii, grip_valid = _compute_gripper_spheres(
        grip_xyz, grip_vis, grip_2d, candidate_boxes
    )
    if not grip_valid.any():
        return np.zeros(n_candidates, dtype=float)

    scores = np.zeros(n_candidates, dtype=float)
    n_frames = len(grip_centers)

    for ci in range(n_candidates):
        filtered = _filtered_3d_points_per_frame(obj_xyz[ci], obj_vis[ci])
        fractions = np.full(n_frames, np.nan, dtype=float)
        valid_mask = np.zeros(n_frames, dtype=bool)
        for t in range(min(n_frames, len(filtered))):
            pts = filtered[t]
            if pts is None or not grip_valid[t]:
                continue
            dists = np.linalg.norm(pts - grip_centers[t], axis=1)
            fractions[t] = float(np.mean(dists <= grip_radii[t]))
            valid_mask[t] = True
        valid_vals = fractions[valid_mask]
        scores[ci] = float(np.mean(valid_vals)) if len(valid_vals) > 0 else 0.0

    return scores


# ── Main entry point ──────────────────────────────────────────────────────────

def compute_scoring_signals(
    raw_tracks_data: dict,
    candidate_confidences: np.ndarray,
    candidate_boxes: np.ndarray,
    grasp_phases: list,
    task_obj_info: dict | None,
    movement_type: str | None,
    fps: float,
    winner_idx: int,
) -> dict:
    """Compute all 4 sub-scores and return a breakdown dict.

    Stores per-winner scores (snr_score_raw, snr_score_normalized, gate_score,
    point_fraction_score) plus all_candidate_breakdowns for reweighting.
    """
    n_candidates = len(candidate_boxes)
    confs = np.asarray(candidate_confidences, dtype=float)

    # Task group and weights
    action = (
        (task_obj_info or {}).get("action")
        or movement_type
        or ""
    )
    task_group = get_task_group(action)
    weights = get_task_adaptive_weights(task_group)

    # Compute motion and phase SNR from raw tracks; retain smoothed signals for replay.
    cotracker_tracks = raw_tracks_data.get("cotracker_tracks")
    if cotracker_tracks is not None and len(cotracker_tracks) > 0:
        tracks_list = [np.asarray(cotracker_tracks[i], dtype=np.float32)
                       for i in range(len(cotracker_tracks))]
        motion_phase_stats = compute_motion_phase_stats(
            tracks_list,
            grasp_phases,
            fps,
            candidate_boxes=candidate_boxes,
            alpha=_PHASE_SNR_ALPHA,
            beta=_PHASE_SNR_BETA,
            eps=_PHASE_SNR_EPS,
        )
        snr_raw = np.asarray(
            [row["phase_snr_raw"] for row in motion_phase_stats],
            dtype=float,
        )
    else:
        motion_phase_stats = [
            {
                "n_total": 0,
                "n_valid_points": 0,
                "bbox_diag": _box_diag(candidate_boxes[i]) if i < len(candidate_boxes) else 0.0,
                "mean_movement": 0.0,
                "median_movement": 0.0,
                "total_movement": 0.0,
                "norm_movement": 0.0,
                "interact_frames": 0,
                "non_interact_frames": 0,
                "interact_mean": 0.0,
                "non_interact_mean": 0.0,
                "interact_sum": 0.0,
                "non_interact_sum": 0.0,
                "phase_fallback": True,
                "phase_snr_raw": 0.0,
            }
            for i in range(n_candidates)
        ]
        snr_raw = np.zeros(n_candidates, dtype=float)

    smooth_motion_phase_stats = []
    smooth_snr_raw = np.full(n_candidates, np.nan, dtype=float)
    smooth_tracks = _smooth_candidate_tracks(raw_tracks_data, fps)
    if smooth_tracks is not None and len(smooth_tracks) > 0:
        smooth_tracks_list = [
            np.asarray(smooth_tracks[i], dtype=np.float32)
            for i in range(len(smooth_tracks))
        ]
        smooth_motion_phase_stats = compute_motion_phase_stats(
            smooth_tracks_list,
            grasp_phases,
            fps,
            candidate_boxes=candidate_boxes,
            alpha=_PHASE_SNR_ALPHA,
            beta=_PHASE_SNR_BETA,
            eps=_PHASE_SNR_EPS,
        )
        smooth_snr_raw = np.asarray(
            [row["phase_snr_raw"] for row in smooth_motion_phase_stats],
            dtype=float,
        )

    finite_snr = snr_raw[np.isfinite(snr_raw)]
    if len(finite_snr) > 1 and (finite_snr.max() - finite_snr.min()) > 1e-9:
        lo, hi = finite_snr.min(), finite_snr.max()
        snr_norm = np.where(np.isfinite(snr_raw), (snr_raw - lo) / (hi - lo), 0.0)
    else:
        lo, hi = np.nan, np.nan
        snr_norm = np.zeros(n_candidates, dtype=float)

    # 2. Gate scores
    gate_scores = compute_gate_scores(confs)

    # 3. 3D point fraction scores
    point_fraction_scores = compute_3d_point_fraction_scores(
        obj_tracks_3d=raw_tracks_data.get("cotracker_tracks_3d"),
        obj_visibility=raw_tracks_data.get("cotracker_visibility"),
        gripper_tracks=raw_tracks_data.get("gripper_tracks"),
        gripper_tracks_3d=raw_tracks_data.get("gripper_tracks_3d"),
        gripper_visibility=raw_tracks_data.get("gripper_visibility"),
        candidate_boxes=candidate_boxes,
    )

    # Per-candidate breakdowns
    all_candidate_breakdowns = []
    for i in range(n_candidates):
        motion = motion_phase_stats[i] if i < len(motion_phase_stats) else {}
        all_candidate_breakdowns.append({
            "source_candidate_idx": int(i),
            "box": [round(float(v), 2) for v in candidate_boxes[i]],
            "det_confidence": round(float(confs[i]), 6),
            "bbox_diag": _round_optional(motion.get("bbox_diag")),
            "mean_movement": _round_optional(motion.get("mean_movement")),
            "median_movement": _round_optional(motion.get("median_movement")),
            "total_movement": _round_optional(motion.get("total_movement")),
            "norm_movement": _round_optional(motion.get("norm_movement")),
            "n_valid_track_points": int(motion.get("n_valid_points", 0)),
            "phase_interact_frames": int(motion.get("interact_frames", 0)),
            "phase_non_interact_frames": int(motion.get("non_interact_frames", 0)),
            "phase_total_frames": int(motion.get("n_total", 0)),
            "phase_interact_mean": _round_optional(motion.get("interact_mean")),
            "phase_non_interact_mean": _round_optional(motion.get("non_interact_mean")),
            "phase_interact_sum": _round_optional(motion.get("interact_sum")),
            "phase_non_interact_sum": _round_optional(motion.get("non_interact_sum")),
            "phase_fallback": bool(motion.get("phase_fallback", True)),
            "snr_score_raw": round(float(snr_raw[i]), 6) if np.isfinite(snr_raw[i]) else None,
            "snr_score_normalized": round(float(snr_norm[i]), 6),
            "gate_score": round(float(gate_scores[i]), 6),
            "point_fraction_score": round(float(point_fraction_scores[i]), 6),
            "is_winner": i == winner_idx,
            **_prefixed_motion_fields(
                "tracks_smooth",
                smooth_motion_phase_stats[i] if i < len(smooth_motion_phase_stats) else {},
                smooth_snr_raw[i] if i < len(smooth_snr_raw) else None,
            ),
        })

    w = winner_idx
    finite_conf_order = np.argsort(np.nan_to_num(confs, nan=-np.inf))[::-1]
    return {
        "score_reconstruction_version": 3,
        "score_reconstruction": {
            "candidate_order": "raw_tracks_data.candidate_boxes",
            "primary_tracks_key": _PRIMARY_TRACKS_KEY,
            "track_source": "raw_tracks_data.cotracker_tracks",
            "alternate_tracks_key": _ALTERNATE_TRACKS_KEY,
            "alternate_track_fields_prefix": "tracks_smooth",
            "track_smoothing_window_seconds": _TRACK_SMOOTHING_WINDOW_SECONDS,
            "track_smoothing_window_frames": max(
                1,
                int(round(_TRACK_SMOOTHING_WINDOW_SECONDS * float(fps))),
            ),
            "ranked_candidate_order": "candidate_confidence_desc",
            "ranked_to_source_candidate_idx": [int(i) for i in finite_conf_order],
            "fps": round(float(fps), 6),
            "grasp_phases_frame_space": "same_as_input_to_compute_scoring_signals",
            "grasp_phases_used": _jsonable_phases(grasp_phases),
            "phase_snr": {
                "alpha": _PHASE_SNR_ALPHA,
                "beta": _PHASE_SNR_BETA,
                "eps": _PHASE_SNR_EPS,
                "normalization": "per_sample_minmax_over_finite_raw_snr",
                "raw_min": _round_optional(lo),
                "raw_max": _round_optional(hi),
            },
            "gate": {
                "high_thresh": _GATE_HIGH_THRESH,
                "low_thresh": _GATE_LOW_THRESH,
            },
            "replay_signals": {
                "movement_plus_detection_conf_new_keyframe_robotseg_filter": {
                    "formula": "0.5 * minmax(mean_movement after robot filter) + 0.75 * det_confidence",
                    "robotseg_overlap_field": "confidence_breakdown.lightweight.robotseg_keyframe_overlap",
                    "robotseg_keep": f"overlap < {_REPLAY_ROBOT_POINT_OVERLAP_THRESH}",
                    "movement_normalization": "per_sample_minmax_after_setting_filtered_candidates_to_-inf",
                },
                "gated_snr_det_robot_point_filtered": {
                    "formula": "0.5 * minmax(phase_snr_raw after robot filter) + 0.5 * det_confidence + 0.2 * dominant_gate",
                    "robotseg_overlap_field": "confidence_breakdown.lightweight.robotseg_keyframe_overlap",
                    "robotseg_keep": f"overlap < {_REPLAY_ROBOT_POINT_OVERLAP_THRESH}",
                    "snr_normalization": "per_sample_minmax_after_setting_filtered_candidates_to_-inf",
                    "dominant_gate": "1.0 only for the sole candidate with det_confidence > high_thresh when it is also the sole candidate > low_thresh",
                },
            },
            "available_per_candidate_fields": [
                "box",
                "det_confidence",
                "bbox_diag",
                "mean_movement",
                "median_movement",
                "total_movement",
                "norm_movement",
                "phase_interact_frames",
                "phase_non_interact_frames",
                "phase_total_frames",
                "phase_interact_mean",
                "phase_non_interact_mean",
                "phase_interact_sum",
                "phase_non_interact_sum",
                "phase_fallback",
                "snr_score_raw",
                "snr_score_normalized",
                "gate_score",
                "point_fraction_score",
                "tracks_smooth_*",
            ],
        },
        "snr_score_raw": round(float(snr_raw[w]), 6) if np.isfinite(snr_raw[w]) else None,
        "snr_score_normalized": round(float(snr_norm[w]), 6),
        "mean_movement": _round_optional(motion_phase_stats[w].get("mean_movement") if w < len(motion_phase_stats) else None),
        "median_movement": _round_optional(motion_phase_stats[w].get("median_movement") if w < len(motion_phase_stats) else None),
        "total_movement": _round_optional(motion_phase_stats[w].get("total_movement") if w < len(motion_phase_stats) else None),
        "norm_movement": _round_optional(motion_phase_stats[w].get("norm_movement") if w < len(motion_phase_stats) else None),
        "phase_interact_mean": _round_optional(motion_phase_stats[w].get("interact_mean") if w < len(motion_phase_stats) else None),
        "phase_non_interact_mean": _round_optional(motion_phase_stats[w].get("non_interact_mean") if w < len(motion_phase_stats) else None),
        "phase_fallback": bool(motion_phase_stats[w].get("phase_fallback", True)) if w < len(motion_phase_stats) else True,
        "gate_score": round(float(gate_scores[w]), 6),
        "point_fraction_score": round(float(point_fraction_scores[w]), 6),
        "task_group": task_group,
        "snr_weight": weights["snr_w"],
        "det_weight": weights["det_w"],
        "gate_weight": weights["gate_w"],
        "point_fraction_weight": weights["point_fraction_w"],
        "all_candidate_breakdowns": all_candidate_breakdowns,
    }


# ── Lightweight scoring features for live candidate selection

from sparc.scoring import selection_scoring as _lws


def _round_list_1d(arr, ndigits: int = 6) -> list | None:
    if arr is None:
        return None
    flat = np.asarray(arr).ravel()
    return [round(float(v), ndigits) if np.isfinite(v) else None for v in flat]


def _round_list_2d(arr, ndigits: int = 6) -> list[list] | None:
    if arr is None:
        return None
    a = np.asarray(arr)
    if a.ndim != 2:
        return None
    return [
        [round(float(v), ndigits) if np.isfinite(v) else None for v in row]
        for row in a
    ]


def _round_box_list(arr, ndigits: int = 2) -> list[list] | None:
    boxes = _round_list_2d(arr, ndigits=ndigits)
    if boxes is None:
        return None
    return boxes


def compute_lightweight_features(
    raw_tracks_data: dict,
    candidate_boxes: np.ndarray,
    candidate_confidences: np.ndarray,
    target_boxes: np.ndarray | None,
    grasp_phases: list,
    fps: float,
    *,
    task_obj_info: dict | None,
    movement_type: str | None,
    detected_arm: str | None,
    scene_movement_baseline_pxps: float,
    verbose: bool = False,
) -> dict:
    """Compute per-candidate scalars for the live lightweight selector.

    The output is JSON-friendly (no numpy arrays) and intended to land under
    ``confidence_breakdown['lightweight']`` so it ships in the JSONL.

    Heavy per-frame matrices (RobotSeg per-keyframe overlap, sphere per-frame
    fraction) are only emitted when ``verbose=True`` — otherwise we keep just
    the aggregated scalars to bound JSONL size.

    Required ``raw_tracks_data`` keys (each may be missing/None — code handles
    it gracefully):
      - ``cotracker_tracks``       (n_cand, T, N, 2)
      - ``cotracker_visibility``   (n_cand, T, N)
      - ``cotracker_tracks_3d``    (n_cand, T, N, 3) — needed for sphere score
      - ``gripper_tracks``         (T, N_g, 2)
      - ``gripper_tracks_3d``      (T, N_g, 3)
      - ``gripper_visibility``     (T, N_g)
      - ``gripper_box``            (1, 4) or (4,)
      - ``gripper_arm``            np.array(b'left' | b'right' | str)
      - ``tracking_frame_indices`` (T,) full-traj frame idx per tracking step
      - ``gripper_video_masks_robotseg``                (T_keys, H, W) uint8
      - ``gripper_video_masks_robotseg_frame_indices``  (T_keys,) int
    """
    out: dict = {}

    cand_tracks = raw_tracks_data.get("cotracker_tracks")
    cand_vis = raw_tracks_data.get("cotracker_visibility")
    if cand_tracks is None or cand_vis is None:
        return out

    cand_tracks = np.asarray(cand_tracks)
    cand_vis = np.asarray(cand_vis)
    n_cand = len(cand_tracks)
    if n_cand == 0:
        return out

    # ── sample-level labels ────────────────────────────────────────────────
    action = (task_obj_info or {}).get("action") or movement_type or ""
    task_group = _lws.get_task_group(action)

    arm_hint_from_tracks = raw_tracks_data.get("gripper_arm")
    if isinstance(arm_hint_from_tracks, np.ndarray):
        try:
            arm_hint_from_tracks = arm_hint_from_tracks.item()
            if isinstance(arm_hint_from_tracks, bytes):
                arm_hint_from_tracks = arm_hint_from_tracks.decode("utf-8", "ignore")
        except Exception:
            arm_hint_from_tracks = None
    active_arm = _lws.resolve_active_arm(
        (task_obj_info or {}).get("arm"),
        detected_arm,
        arm_hint_from_tracks,
    )
    out["task_group"] = task_group
    out["active_arm"] = active_arm
    out["scene_movement_baseline_pxps"] = float(scene_movement_baseline_pxps)
    out["fps"] = float(fps)
    out["signal_name"] = _lws.ADAPTIVE_DET_SOFT_SNR_SP8_SIGNAL_NAME
    out["candidate_order"] = "npz/raw_tracks_data.candidate_boxes"
    out["score_array_index_contract"] = (
        "index i matches NPZ candidate_boxes[i], cotracker_tracks[i], cotracker_visibility[i]"
    )
    out["candidate_boxes"] = _round_box_list(candidate_boxes)
    out["candidate_confidences"] = _round_list_1d(candidate_confidences)
    source_candidate_idx = raw_tracks_data.get("source_candidate_indices")
    if source_candidate_idx is None:
        source_candidate_idx = np.arange(n_cand, dtype=int)
    source_candidate_idx = np.asarray(source_candidate_idx, dtype=int).reshape(-1)
    if len(source_candidate_idx) != n_cand:
        source_candidate_idx = np.arange(n_cand, dtype=int)
    out["source_candidate_idx"] = [int(i) for i in source_candidate_idx]
    if target_boxes is not None and len(target_boxes) > 0:
        out["target_candidate_boxes"] = _round_box_list(target_boxes)
        target_confs = raw_tracks_data.get("target_candidate_confidences")
        if target_confs is not None and len(target_confs) > 0:
            out["target_candidate_confidences"] = _round_list_1d(target_confs)

    # ── centroid traces (one trace per candidate box) ──────────────────────
    # Persisted as part of the base64 ``arrays_blob`` below — a nested rounded
    # JSON list bloats the JSONL by ~10x for the same numerical content.
    centroid = _lws.compute_centroid_traces(cand_tracks, cand_vis)

    # ── phase-SNR motion stats ─────────────────────────────────────────────
    # Keep point columns unless they are entirely the -1 sentinel.
    # Visibility is still used below for target and 3D scoring.
    pms = _lws.compute_phase_motion_stats(cand_tracks, None, grasp_phases, fps)
    snr = _lws.phase_snr(
        pms,
        alpha=_lws.ADAPTIVE_DET_SOFT_SNR_SP8_ALPHA,
        beta=_lws.ADAPTIVE_DET_SOFT_SNR_SP8_BETA,
        eps=_lws.ADAPTIVE_DET_SOFT_SNR_SP8_EPS,
    )
    out["phase_motion_stats"] = {
        "interact_frames": [int(v) for v in pms.interact_frames],
        "non_interact_frames": [int(v) for v in pms.non_interact_frames],
        "n_total": [int(v) for v in pms.n_total],
        "interact_mean": _round_list_1d(pms.interact_mean),
        "non_interact_mean": _round_list_1d(pms.non_interact_mean),
        "interact_std": _round_list_1d(pms.interact_std),
        "non_interact_std": _round_list_1d(pms.non_interact_std),
        "interact_sum": _round_list_1d(pms.interact_sum),
        "fallback": [bool(v) for v in pms.fallback],
    }
    out["phase_snr_a0p6_b0p2"] = _round_list_1d(snr)

    # ── 3D sphere point fraction (mean + peak) ─────────────────────────────
    cand_3d = raw_tracks_data.get("cotracker_tracks_3d")
    g2 = raw_tracks_data.get("gripper_tracks")
    g3 = raw_tracks_data.get("gripper_tracks_3d")
    gv = raw_tracks_data.get("gripper_visibility")
    g_box = raw_tracks_data.get("gripper_box")
    if g_box is not None:
        g_box = np.asarray(g_box, dtype=float)
        if g_box.ndim == 2 and g_box.shape[0] >= 1:
            g_box = g_box[0]
        if g_box.shape != (4,):
            g_box = None
    out["gripper_arm"] = active_arm
    if g_box is not None:
        out["gripper_box"] = [round(float(v), 2) for v in g_box]

    if cand_3d is not None and g2 is not None and g3 is not None and gv is not None:
        try:
            sphere = _lws.compute_sphere_point_fraction(
                np.asarray(cand_3d, dtype=float),
                np.asarray(cand_vis, dtype=float),
                np.asarray(g2, dtype=float),
                np.asarray(g3, dtype=float),
                np.asarray(gv, dtype=float),
                candidate_bboxes_2d=np.asarray(candidate_boxes, dtype=float),
            )
            out["sphere3d_point_fraction_mean"] = _round_list_1d(sphere.mean_fraction)
            out["sphere3d_point_fraction_peak"] = _round_list_1d(sphere.peak_fraction)
            out["sphere3d_point_fraction_mean_r8"] = _round_list_1d(sphere.mean_fraction_r8)
            out["sphere3d_point_fraction_peak_r8"] = _round_list_1d(sphere.peak_fraction_r8)
            out["sphere3d_obj_scale"] = (
                round(float(sphere.obj_scale), 4) if np.isfinite(sphere.obj_scale) else None
            )
        except Exception as exc:
            sphere = None
            out["sphere3d_error"] = repr(exc)
            out["sphere3d_point_fraction_mean"] = _round_list_1d(np.zeros(n_cand, dtype=float))
            out["sphere3d_point_fraction_peak"] = _round_list_1d(np.zeros(n_cand, dtype=float))
            out["sphere3d_point_fraction_mean_r8"] = _round_list_1d(np.zeros(n_cand, dtype=float))
            out["sphere3d_point_fraction_peak_r8"] = _round_list_1d(np.zeros(n_cand, dtype=float))
    else:
        sphere = None
        out["sphere3d_point_fraction_mean"] = _round_list_1d(np.zeros(n_cand, dtype=float))
        out["sphere3d_point_fraction_peak"] = _round_list_1d(np.zeros(n_cand, dtype=float))
        out["sphere3d_point_fraction_mean_r8"] = _round_list_1d(np.zeros(n_cand, dtype=float))
        out["sphere3d_point_fraction_peak_r8"] = _round_list_1d(np.zeros(n_cand, dtype=float))

    # ── RobotSeg keyframe overlap (active-arm component, dilated, aligned) ─
    rs_masks = raw_tracks_data.get("gripper_video_masks_robotseg")
    rs_frames = raw_tracks_data.get("gripper_video_masks_robotseg_frame_indices")
    track_full_idx = raw_tracks_data.get("tracking_frame_indices")
    bundle = _lws.GripperTrackBundle(
        arm=active_arm,
        tracks_2d=np.asarray(g2, dtype=float) if g2 is not None else None,
        visibility=np.asarray(gv, dtype=float) if gv is not None else None,
        bbox=g_box,
        tracking_frame_indices=(np.asarray(track_full_idx, dtype=int)
                                if track_full_idx is not None else None),
        robotseg_masks=(np.asarray(rs_masks) if rs_masks is not None else None),
        robotseg_frame_indices=(np.asarray(rs_frames, dtype=int)
                                if rs_frames is not None else None),
    )
    try:
        rs = _lws.compute_robotseg_keyframe_overlap(
            cand_tracks, cand_vis, np.asarray(candidate_boxes, dtype=float),
            gripper=bundle,
            tracking_frame_indices=(np.asarray(track_full_idx, dtype=int)
                                    if track_full_idx is not None else None),
            active_arm=active_arm,
        )
        out["robotseg_keyframe_overlap"] = _round_list_1d(rs.overlap)
        out["robotseg_keyframe_meta"] = {
            **rs.meta,
            "fallback_used": bool(rs.fallback_used),
            "n_keyframes_used": int(len(rs.keyframe_track_t)),
        }
    except Exception as exc:
        rs = None
        out["robotseg_error"] = repr(exc)
        out["robotseg_keyframe_overlap"] = _round_list_1d(np.zeros(n_cand, dtype=float))

    depth_rescue_keep = _lws.compute_depth_rescue_keep_mask(
        np.asarray(cand_3d, dtype=float) if cand_3d is not None else None,
        np.asarray(cand_vis, dtype=float),
        np.asarray(g3, dtype=float) if g3 is not None else None,
        np.asarray(gv, dtype=float) if gv is not None else None,
        np.asarray(out["robotseg_keyframe_overlap"], dtype=float),
        overlap_thresh=_lws.SOFT_SNR_DEPTH_RESCUE_OVERLAP_THRESH,
        depth_thresh=_lws.SOFT_SNR_DEPTH_RESCUE_DEPTH_THRESH,
        tip_k=_lws.SOFT_SNR_DEPTH_RESCUE_TIP_K,
    )
    out["depth_rescue_keep_mask"] = [bool(v) for v in depth_rescue_keep]
    out["depth_rescue_meta"] = {
        "overlap_thresh": float(_lws.SOFT_SNR_DEPTH_RESCUE_OVERLAP_THRESH),
        "depth_thresh": float(_lws.SOFT_SNR_DEPTH_RESCUE_DEPTH_THRESH),
        "tip_k": int(_lws.SOFT_SNR_DEPTH_RESCUE_TIP_K),
    }

    # ── target-point score (gated + ungated + retunable arrays) ────────────
    tps = None
    if target_boxes is not None and len(target_boxes) > 0:
        try:
            tps = _lws.compute_target_point_score(
                cand_tracks, cand_vis,
                np.asarray(target_boxes, dtype=float),
                fps=fps,
                motion_gate_pxps=_lws.TEMPORAL_CONSISTENCY_GATE_PXPS,
            )
            out["target_point_score_ungated"] = _round_list_2d(tps.score_ungated)
            out["target_point_score_motion_gated"] = _round_list_2d(tps.score_motion_gated)
            out["target_point_motion_gate_pxps"] = round(float(tps.motion_gate_pxps), 4)
        except Exception as exc:
            tps = None
            out["target_point_error"] = repr(exc)

    # ── pack the heavy arrays into a single base64 NPZ blob ────────────────
    # Cuts JSONL size by ~10x compared to nested rounded-float lists.
    arrays: dict[str, np.ndarray | None] = {
        "centroid_traces": np.asarray(centroid, dtype=np.float32),
    }
    # Candidate SAM2 masks on initial frame (n_cand, H, W) uint8
    cand_masks = raw_tracks_data.get("candidate_masks")
    if cand_masks is not None and len(cand_masks) > 0:
        arrays["candidate_masks"] = np.asarray(cand_masks, dtype=np.uint8)
    # Target candidate SAM2 masks on final frame (n_target, H, W) uint8
    tgt_masks = raw_tracks_data.get("target_candidate_masks")
    if tgt_masks is not None and len(tgt_masks) > 0:
        arrays["target_candidate_masks"] = np.asarray(tgt_masks, dtype=np.uint8)
    # The representative gripper track is the per-frame sphere-center point
    # actually used by the 3D sphere score: for each frame, the filtered
    # gripper point whose 2D projection is closest to the image center, and
    # its 3D location. Storing exactly that keeps offline JSONL replay's
    # single-point gripper track identical to the scored sphere centers.
    if sphere is not None and np.any(np.isfinite(sphere.sphere_centers)):
        arrays["gripper_rep_track_2d"] = np.asarray(sphere.sphere_centers_2d, dtype=np.float32)
        arrays["gripper_rep_track_3d"] = np.asarray(sphere.sphere_centers, dtype=np.float32)
        arrays["gripper_rep_visibility"] = np.all(np.isfinite(sphere.sphere_centers), axis=1)
        if track_full_idx is not None:
            arrays["gripper_rep_tracking_frame_indices"] = np.asarray(track_full_idx, dtype=np.int32)
        out["gripper_rep_track_selection_rule"] = "per_frame_sphere_center_closest_to_image_center"
    if tps is not None:
        arrays["target_point_motion_pxps"] = tps.point_motion.astype(np.float32)
        arrays["target_point_visible_final"] = tps.point_visible_final
        arrays["target_point_in_target"] = tps.point_in_target
    if verbose and sphere is not None:
        arrays["sphere3d_per_frame_mean"] = sphere.per_frame_fraction_mean.astype(np.float32)
        arrays["sphere3d_per_frame_peak"] = sphere.per_frame_fraction_peak.astype(np.float32)
        arrays["sphere3d_per_frame_valid"] = sphere.per_frame_valid
    if verbose and rs is not None:
        arrays["robotseg_per_frame_overlap"] = rs.per_frame_overlap.astype(np.float32)
        arrays["robotseg_per_frame_counts"] = rs.per_frame_counts.astype(np.float32)
        arrays["robotseg_keyframe_track_t"] = rs.keyframe_track_t.astype(np.int32)
    blob = _lws.encode_arrays_blob(arrays)
    if blob is not None:
        out["arrays_blob"] = blob
        out["arrays_blob_keys"] = sorted(k for k, v in arrays.items() if v is not None)
        out["arrays_blob_encoding"] = "base64_encoded_np_savez_compressed"
        out["arrays_blob_schema_version"] = 1

    # ── live winner pick: adaptive_det_soft_snr_sp8 ───────────────────────
    # Persist the tracking-space phases so replay uses the same frame indices.
    snr_arr = np.asarray(snr, dtype=float)  # populated above next to phase_motion_stats
    det_arr = np.asarray(candidate_confidences, dtype=float)
    sphere_arr = (np.asarray(sphere.mean_fraction_r8, dtype=float) if sphere is not None
                  else np.zeros(n_cand, dtype=float))
    overlap_arr = (np.asarray(rs.overlap, dtype=float) if rs is not None
                   else np.zeros(n_cand, dtype=float))
    # Per-candidate fingertip |Δz| — drives the depth-relief term of the
    # continuous robot-overlap penalty (NaN where it cannot be computed).
    dz_arr = np.full(n_cand, np.nan, dtype=float)
    if cand_3d is not None and g3 is not None and gv is not None:
        try:
            dz_arr = _lws.compute_candidate_median_abs_dz(
                np.asarray(cand_3d, dtype=float),
                np.asarray(cand_vis, dtype=float),
                np.asarray(g3, dtype=float),
                np.asarray(gv, dtype=float),
                tip_k=_lws.SOFT_SNR_DEPTH_RESCUE_TIP_K,
            )
        except Exception as exc:
            dz_arr = np.full(n_cand, np.nan, dtype=float)
            out["candidate_median_abs_dz_error"] = repr(exc)
    out["candidate_median_abs_dz"] = _round_list_1d(dz_arr)
    try:
        pick = _lws.pick_adaptive_det_soft_snr_sp8(
            detection_confidence=det_arr,
            phase_snr=snr_arr,
            sphere_point_fraction_r8=sphere_arr,
            robotseg_overlap=overlap_arr,
            candidate_median_abs_dz=dz_arr,
        )
        components = {k: _round_list_1d(v) for k, v in pick.components.items()}
        out["phase_snr_a0p6_b0p2_norm"] = _round_list_1d(pick.phase_snr_norm)
        out["sphere3d_point_fraction_mean_r8_norm"] = _round_list_1d(pick.sphere_norm)
        out["keep_mask"] = [bool(v) for v in pick.keep_mask]
        out["adaptive_det_weight"] = _round_list_1d(pick.adaptive_det_weight)
        out["score_components"] = components
        out["final_score"] = _round_list_1d(pick.final_score)
        out["winner_idx"] = None if pick.best_start is None else int(pick.best_start)
        out["per_candidate_score_breakdown"] = [
            {
                "candidate_idx": int(i),
                "source_candidate_idx": int(source_candidate_idx[i]),
                "box": out["candidate_boxes"][i] if out.get("candidate_boxes") else None,
                "det_confidence": out["candidate_confidences"][i] if out.get("candidate_confidences") else None,
                "phase_snr_a0p6_b0p2": out["phase_snr_a0p6_b0p2"][i],
                "phase_snr_a0p6_b0p2_norm": out["phase_snr_a0p6_b0p2_norm"][i],
                "sphere3d_point_fraction_mean_r8": (
                    out["sphere3d_point_fraction_mean_r8"][i]
                    if out.get("sphere3d_point_fraction_mean_r8") else None
                ),
                "sphere3d_point_fraction_mean_r8_norm": out["sphere3d_point_fraction_mean_r8_norm"][i],
                "robotseg_keyframe_overlap": (
                    out["robotseg_keyframe_overlap"][i]
                    if out.get("robotseg_keyframe_overlap") else None
                ),
                "candidate_median_abs_dz": (
                    out["candidate_median_abs_dz"][i]
                    if out.get("candidate_median_abs_dz") else None
                ),
                "depth_rescue_keep": bool(depth_rescue_keep[i]) if i < len(depth_rescue_keep) else None,
                "keep": bool(pick.keep_mask[i]),
                "adaptive_det_weight": out["adaptive_det_weight"][i],
                "final_score": out["final_score"][i],
                "components": {k: v[i] for k, v in components.items()},
            }
            for i in range(n_cand)
        ]
        out["winner_pick"] = {
            "signal_name": _lws.ADAPTIVE_DET_SOFT_SNR_SP8_SIGNAL_NAME,
            "best_start_idx": pick.best_start,
            "best_target_idx": None,
            "rejection_reason": pick.rejection_reason,
            "final_score": _round_list_1d(pick.final_score),
            "keep_mask": [bool(v) for v in pick.keep_mask],
            "adaptive_det_weight": _round_list_1d(pick.adaptive_det_weight),
            "phase_snr_a0p6_b0p2_norm": _round_list_1d(pick.phase_snr_norm),
            "sphere3d_point_fraction_mean_r8_norm": _round_list_1d(pick.sphere_norm),
            "constants": pick.constants,
            "components": components,
        }
    except Exception as exc:
        out["winner_pick_error"] = repr(exc)

    return out
