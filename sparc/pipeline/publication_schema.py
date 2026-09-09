"""Publication schema v2 construction and array loading.

The annotator's in-memory dictionary is shared by scoring and debug rendering.
This module builds the public v2 annotation from that current pipeline state.
Persisted annotation readers accept only sparc.annotation version 2.0.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from sparc.scoring.selection_scoring import decode_arrays_blob, encode_arrays_blob


SCHEMA_NAME = "sparc.annotation"
SCHEMA_VERSION = "2.0"
ARRAYS_ENCODING = "base64_encoded_np_savez_compressed"


def is_publication_annotation(annotation: Mapping[str, Any]) -> bool:
    schema = annotation.get("schema") if isinstance(annotation, Mapping) else None
    return (
        isinstance(schema, Mapping)
        and schema.get("name") == SCHEMA_NAME
        and str(schema.get("version")) == SCHEMA_VERSION
    )


def require_publication_annotation(annotation: Mapping[str, Any]) -> None:
    """Reject unsupported persisted annotation formats."""
    if not is_publication_annotation(annotation):
        raise ValueError(
            f"Expected {SCHEMA_NAME} schema version {SCHEMA_VERSION}; "
            "old annotation formats are not supported by this release"
        )


def load_publication_arrays(
    annotation: Mapping[str, Any],
    *,
    sidecar_root: Optional[Path | str] = None,
) -> dict[str, np.ndarray]:
    """Load v2 arrays from embedded storage, an HDF5 group, or an NPZ shard."""
    require_publication_annotation(annotation)
    arrays = annotation.get("arrays")
    if not isinstance(arrays, Mapping):
        return {}
    blob = arrays.get("blob")
    if blob:
        return decode_arrays_blob(str(blob))

    storage = arrays.get("storage")
    if not isinstance(storage, Mapping) or not storage.get("path"):
        return {}
    path = Path(str(storage["path"]))
    if not path.is_absolute():
        if sidecar_root is None:
            raise ValueError(
                "sidecar_root is required to resolve a relative publication array path"
            )
        path = Path(sidecar_root) / path
    requested = storage.get("array_keys")
    group_name = storage.get("group")
    if group_name is not None or path.suffix.lower() in {".h5", ".hdf5"}:
        import h5py

        if group_name is None:
            raise ValueError("HDF5 publication array storage is missing its group")
        with h5py.File(path, "r") as shard:
            group = shard[str(group_name)]
            keys = requested if requested is not None else list(group.keys())
            result = {}
            for key in keys:
                if str(key) not in group:
                    continue
                value = np.asarray(group[str(key)])
                if value.dtype.kind == "S":
                    value = value.astype("U")
                elif value.dtype.kind == "O" and value.size and all(
                    isinstance(item, (bytes, np.bytes_)) for item in value.flat
                ):
                    value = np.asarray([
                        item.decode("utf-8", "replace") for item in value.flat
                    ]).reshape(value.shape)
                result[str(key)] = value
            return result

    prefix = str(storage.get("key_prefix", "")).rstrip("/")
    with np.load(path, allow_pickle=False) as shard:
        if requested is None:
            marker = f"{prefix}/" if prefix else ""
            requested = [
                key[len(marker):]
                for key in shard.files
                if not marker or key.startswith(marker)
            ]
        result = {}
        for key in requested:
            shard_key = f"{prefix}/{key}" if prefix else str(key)
            if shard_key in shard.files:
                result[str(key)] = np.asarray(shard[shard_key])
        return result


def trajectory_id_from_annotation(annotation: Mapping[str, Any]) -> Optional[str]:
    """Return the loader-compatible trajectory ID from a v2 annotation."""
    require_publication_annotation(annotation)
    source = annotation.get("source") or {}
    value = source.get("trajectory_id")
    return str(value) if value is not None else None


def subtask_index_from_annotation(annotation: Mapping[str, Any]) -> Any:
    require_publication_annotation(annotation)
    return (annotation.get("source") or {}).get("subtask_index")


def _single_box(value: Any) -> Optional[list[float]]:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.size != 4:
        return None
    arr = arr.reshape(4)
    if not np.all(np.isfinite(arr)):
        return None
    return [float(v) for v in arr]


def _clean_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): item for key, item in value.items() if item is not None}


def _lightweight(annotation: Mapping[str, Any]) -> dict[str, Any]:
    breakdown = annotation.get("confidence_breakdown") or {}
    if not isinstance(breakdown, Mapping):
        return {}
    value = breakdown.get("lightweight") or {}
    return dict(value) if isinstance(value, Mapping) else {}


def _phase_rows(annotation: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for phase in annotation.get("phase_annotations") or []:
        if not isinstance(phase, Mapping):
            continue
        phase_type = str(phase.get("phase", "unknown"))
        start = phase.get("start_frame")
        end = phase.get("end_frame")
        row = {
            "type": phase_type,
            "start_frame": int(start) if start is not None else None,
            "end_frame_inclusive": int(end) if end is not None else None,
            "description": phase.get("description"),
        }
        rows.append(_clean_mapping(row))
    return rows


def _mask_series(
    arrays: Mapping[str, np.ndarray],
    prefix: str,
) -> list[tuple[str, int, np.ndarray]]:
    masks = arrays.get(f"{prefix}_phase_keyframe_masks")
    frames = arrays.get(f"{prefix}_phase_keyframe_absolute_frame_indices")
    names = arrays.get(f"{prefix}_phase_keyframe_phase_names")
    if masks is None or frames is None or names is None:
        return []
    masks = np.asarray(masks)
    frames = np.asarray(frames).reshape(-1)
    names = np.asarray(names).reshape(-1)
    count = min(len(masks), len(frames), len(names))
    result = []
    for idx in range(count):
        name = names[idx]
        if isinstance(name, bytes):
            name = name.decode("utf-8", "replace")
        result.append((str(name), int(frames[idx]), np.asarray(masks[idx], dtype=np.uint8)))
    return result


def _mask_transform(
    lightweight: Mapping[str, Any],
) -> tuple[Optional[list[int]], tuple[float, float], str]:
    metadata = lightweight.get("phase_keyframe_mask_metadata") or {}
    transform = metadata.get("transform_mask_pixels_to_original_frame") or {}
    if "global_scale_xy" not in transform:
        raise ValueError("missing transform in phase_keyframe_mask_metadata")
    crop_box = transform.get("crop_xyxy_in_globally_resized_frame")
    scale_xy = tuple(float(v) for v in transform["global_scale_xy"])
    return (
        [int(v) for v in crop_box] if crop_box is not None else None,
        (scale_xy[0], scale_xy[1]),
        str(metadata.get("spatial_coordinate_frame", "unknown")),
    )


def _restore_mask(
    mask: np.ndarray,
    original_size_hw: tuple[int, int],
    crop_box: Optional[list[int]],
    global_scale_xy: tuple[float, float],
    coordinate_frame: str,
) -> np.ndarray:
    mask = (np.asarray(mask) > 0).astype(np.uint8)
    original_h, original_w = map(int, original_size_hw)
    if mask.shape == (original_h, original_w):
        return mask
    if "original" in coordinate_frame and "processing" not in coordinate_frame:
        raise ValueError(
            f"mask claims original coordinates but has shape {mask.shape}, expected {(original_h, original_w)}"
        )

    import cv2

    sx, sy = global_scale_xy
    if sx <= 0 or sy <= 0:
        raise ValueError(f"invalid global scale {global_scale_xy}")
    if coordinate_frame == "unknown" and crop_box is None and (sx, sy) == (1.0, 1.0):
        raise ValueError(
            f"missing transform for mask shape {mask.shape} and original image "
            f"shape {(original_h, original_w)}"
        )
    resized_h = max(1, int(round(original_h * sy)))
    resized_w = max(1, int(round(original_w * sx)))
    canvas = np.zeros((resized_h, resized_w), dtype=np.uint8)
    if crop_box is None:
        if mask.shape != canvas.shape:
            mask = cv2.resize(mask, (resized_w, resized_h), interpolation=cv2.INTER_NEAREST)
        canvas = mask
    else:
        x1, y1, x2, y2 = crop_box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(resized_w, x2), min(resized_h, y2)
        target_h, target_w = max(0, y2 - y1), max(0, x2 - x1)
        if target_h == 0 or target_w == 0:
            raise ValueError(f"invalid crop box {crop_box} for resized image {(resized_h, resized_w)}")
        if mask.shape != (target_h, target_w):
            mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        canvas[y1:y2, x1:x2] = mask
    if canvas.shape != (original_h, original_w):
        canvas = cv2.resize(canvas, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
    return (canvas > 0).astype(np.uint8)


def _align_masks_to_phases(
    phases: list[dict[str, Any]],
    object_series: list[tuple[str, int, np.ndarray]],
    robot_series: list[tuple[str, int, np.ndarray]],
    original_size_hw: tuple[int, int],
    restore,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    if not phases:
        seen = []
        for name, frame, _ in object_series + robot_series:
            key = (name, frame)
            if key not in seen:
                seen.append(key)
                phases.append({
                    "type": name,
                    "start_frame": frame,
                    "end_frame_inclusive": frame,
                })

    height, width = original_size_hw
    object_masks = np.zeros((len(phases), height, width), dtype=np.uint8)
    robot_masks = np.zeros((len(phases), height, width), dtype=np.uint8)
    object_valid = np.zeros(len(phases), dtype=bool)
    robot_valid = np.zeros(len(phases), dtype=bool)

    def phase_index(name: str, frame: int) -> Optional[int]:
        exact = [
            idx for idx, phase in enumerate(phases)
            if phase.get("type") == name and phase.get("end_frame_inclusive") == frame
        ]
        if exact:
            return exact[0]
        by_frame = [
            idx for idx, phase in enumerate(phases)
            if phase.get("end_frame_inclusive") == frame
        ]
        return by_frame[0] if by_frame else None

    for name, frame, mask in object_series:
        idx = phase_index(name, frame)
        if idx is not None:
            restored = restore(mask)
            if restored is not None:
                object_masks[idx] = restored
                object_valid[idx] = True
    for name, frame, mask in robot_series:
        idx = phase_index(name, frame)
        if idx is not None:
            restored = restore(mask)
            if restored is not None:
                robot_masks[idx] = restored
                robot_valid[idx] = True

    arrays = {
        "object_masks": object_masks,
        "object_mask_valid": object_valid,
        "robot_masks": robot_masks,
        "robot_mask_valid": robot_valid,
    }
    return phases, arrays


def _append_missing_mask_phases(
    phases: list[dict[str, Any]],
    series_groups: list[list[tuple[str, int, np.ndarray]]],
) -> None:
    """Ensure every exported keyframe has a corresponding task phase row."""
    existing = {
        (phase.get("type"), phase.get("end_frame_inclusive"))
        for phase in phases
    }
    for series in series_groups:
        for name, frame, _ in series:
            key = (name, frame)
            if key in existing:
                continue
            phases.append({
                "type": name,
                "start_frame": frame,
                "end_frame_inclusive": frame,
            })
            existing.add(key)
    phases.sort(key=lambda phase: (
        int(phase.get("end_frame_inclusive", -1)),
        str(phase.get("type", "")),
    ))


def _selection(annotation: Mapping[str, Any]) -> dict[str, Any]:
    breakdown = annotation.get("confidence_breakdown") or {}
    if not isinstance(breakdown, Mapping):
        breakdown = {}
    lightweight = _lightweight(annotation)
    override = breakdown.get("selection_winner_override") or {}
    winner_pick = lightweight.get("winner_pick") or {}
    winner_idx = override.get("new_winner_idx", winner_pick.get("best_start_idx", lightweight.get("winner_idx")))
    target_idx = override.get("new_target_idx", winner_pick.get("best_target_idx"))
    score = override.get("new_final_score")
    if score is None:
        score = annotation.get("initial_object_box_selection_score")
    if score is None and winner_idx is not None:
        scores = lightweight.get("final_score")
        if not isinstance(scores, (list, tuple)):
            scores = []
        if 0 <= int(winner_idx) < len(scores):
            score = scores[int(winner_idx)]

    candidate_breakdown = None
    candidates = lightweight.get("per_candidate_score_breakdown") or []
    if winner_idx is not None and 0 <= int(winner_idx) < len(candidates):
        selected_candidate = dict(candidates[int(winner_idx)])
        for redundant_key in (
            "candidate_idx",
            "source_candidate_idx",
            "box",
            "det_confidence",
            "final_score",
        ):
            selected_candidate.pop(redundant_key, None)
        score_components = selected_candidate.pop("components", None)
        candidate_breakdown = _clean_mapping({
            "score_components": score_components,
            "signals": _clean_mapping(selected_candidate),
        })

    result = {
        "method": (
            override.get("signal_name")
            or winner_pick.get("signal_name")
            or lightweight.get("signal_name")
            or annotation.get("initial_object_box_selection_score_name")
        ),
        "score": float(score) if score is not None else None,
        "score_type": "ranking_score_not_calibrated_probability" if score is not None else None,
        "selected_candidate_index": int(winner_idx) if winner_idx is not None else None,
        "selected_target_index": int(target_idx) if target_idx is not None else None,
        "target_selection_method": override.get("target_source"),
        "rejection_reason": override.get("reason", winner_pick.get("rejection_reason")),
        "breakdown": candidate_breakdown,
        "constants": winner_pick.get("constants"),
    }
    verification = {
        "method": breakdown.get("verification_mode"),
        "answer": breakdown.get("verification_answer"),
        "score": breakdown.get("verification_score"),
    }
    if any(value is not None for value in verification.values()):
        result["verification"] = _clean_mapping(verification)
    return _clean_mapping(result)


def _highest_detector_score_candidate(
    boxes_value: Any,
    scores_value: Any,
) -> tuple[Optional[int], Optional[list[float]], Optional[float]]:
    try:
        boxes = np.asarray(boxes_value, dtype=float).reshape(-1, 4)
    except (TypeError, ValueError):
        return None, None, None
    if len(boxes) == 0:
        return None, None, None

    try:
        scores = np.asarray(scores_value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        scores = np.empty(0, dtype=float)
    usable = min(len(boxes), len(scores))
    finite = np.isfinite(scores[:usable]) if usable else np.empty(0, dtype=bool)
    index = int(np.nanargmax(np.where(finite, scores[:usable], -np.inf))) if finite.any() else 0
    score = float(scores[index]) if index < usable and np.isfinite(scores[index]) else None
    return index, [float(value) for value in boxes[index]], score


def _track_points_in_original_coordinates(
    annotation: Mapping[str, Any],
    lightweight: Mapping[str, Any],
    points: np.ndarray,
) -> np.ndarray:
    annotation_format = annotation.get("annotation_format") or {}
    transform = (
        annotation_format.get("processing_frame_transform_to_original") or {}
        if isinstance(annotation_format, Mapping)
        else {}
    )
    crop_origin = transform.get("crop_origin_xy")
    scale = transform.get("global_scale_xy")
    if crop_origin is None or scale is None:
        metadata = lightweight.get("phase_keyframe_mask_metadata") or {}
        mask_transform = (
            metadata.get("transform_mask_pixels_to_original_frame") or {}
            if isinstance(metadata, Mapping)
            else {}
        )
        crop_box = mask_transform.get("crop_xyxy_in_globally_resized_frame")
        if crop_origin is None and crop_box is not None:
            crop_origin = crop_box[:2]
        if scale is None:
            scale = mask_transform.get("global_scale_xy")

    crop_xy = np.asarray(crop_origin or (0.0, 0.0), dtype=float).reshape(2)
    scale_xy = np.asarray(scale or (1.0, 1.0), dtype=float).reshape(2)
    if np.any(scale_xy <= 0):
        raise ValueError(f"invalid detector baseline coordinate scale {scale_xy.tolist()}")
    return (np.asarray(points, dtype=float) + crop_xy) / scale_xy


def _centroid_track_for_candidate(
    annotation: Mapping[str, Any],
    lightweight: Mapping[str, Any],
    candidate_index: Optional[int],
    *,
    start_frame: int,
    end_frame: int,
    detection_rel: int,
) -> Optional[dict[str, Any]]:
    """Visible-point centroid track for one candidate, in original image coordinates.

    ``centroid_traces`` is produced at scoring time by
    selection_scoring.compute_centroid_traces(cand_tracks, cand_vis), i.e. with
    CoTracker visibility applied. Both the published "ours" track and the detector
    baseline are built from this one array, differing only in which candidate index
    is selected -- deliberately shared so the two can never drift apart in how the
    trace is extracted.
    """
    if candidate_index is None:
        return None

    decoded = decode_arrays_blob(lightweight.get("arrays_blob"))
    centroid_traces = decoded.get("centroid_traces")
    if centroid_traces is None:
        return None

    centroid_traces = np.asarray(centroid_traces, dtype=float)
    if not (
        centroid_traces.ndim == 3
        and centroid_traces.shape[-1] >= 2
        and 0 <= int(candidate_index) < len(centroid_traces)
    ):
        return None

    points = _track_points_in_original_coordinates(
        annotation,
        lightweight,
        centroid_traces[int(candidate_index), :, :2],
    )
    stored_frame_indices = decoded.get("gripper_rep_tracking_frame_indices")
    frame_indices = None
    if stored_frame_indices is not None:
        frame_indices = np.asarray(stored_frame_indices, dtype=int).reshape(-1)
    has_public_timeline = frame_indices is not None and len(frame_indices) == len(points)
    if not has_public_timeline:
        track_end = max(detection_rel, end_frame - start_frame - 1)
        frame_indices = np.linspace(detection_rel, track_end, len(points), dtype=int)

    finite = np.all(np.isfinite(points), axis=1)
    points = points[finite]
    frame_indices = frame_indices[finite]
    if not len(points):
        return None

    if has_public_timeline:
        return {
            "array_reference": {
                "xy_pixels_array": "centroid_traces",
                "candidate_index": int(candidate_index),
                "frame_indices_array": "centroid_trace_frame_indices",
                "validity": "finite_xy",
            },
            "point_selection": {"method": "visible_point_centroid"},
        }

    return {
        "xy_pixels": points.tolist(),
        "frame_indices": [start_frame + int(value) for value in frame_indices],
        "point_selection": {"method": "visible_point_centroid"},
    }


def _public_centroid_arrays(
    annotation: Mapping[str, Any],
    is_bimanual: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Per-candidate centroid traces for the published arrays blob.

    Scoring-time centroids are converted to original image coordinates and
    relative tracking indices become absolute trajectory frame indices.

    Returns ``(arrays, manifest_entries)``; both empty when unavailable.
    """
    if is_bimanual:
        # Per-arm candidate sets do not share one candidate axis; skip rather
        # than publish an array whose indexing does not match selection indices.
        return {}, {}

    lightweight = _lightweight(annotation)
    decoded = decode_arrays_blob(lightweight.get("arrays_blob"))
    centroid_traces = decoded.get("centroid_traces")
    if centroid_traces is None:
        return {}, {}

    centroid_traces = np.asarray(centroid_traces, dtype=float)
    if centroid_traces.ndim != 3 or centroid_traces.shape[-1] < 2:
        return {}, {}

    try:
        converted = np.stack([
            _track_points_in_original_coordinates(
                annotation, lightweight, centroid_traces[idx, :, :2],
            )
            for idx in range(len(centroid_traces))
        ]).astype(np.float32)
    except ValueError:
        return {}, {}

    # The shared blob encoder stores centroids at float16 precision.
    arrays: dict[str, Any] = {"centroid_traces": converted.astype(np.float16)}
    manifest: dict[str, Any] = {
        "centroid_traces": {
            "dtype": str(arrays["centroid_traces"].dtype),
            "shape": list(converted.shape),
            "axes": ["candidates", "tracking_timeline", "xy"],
            "source": "selection_scoring.compute_centroid_traces",
            "description": (
                "per-candidate mean of visible CoTracker points; candidate axis "
                "matches selection.selected_candidate_index and "
                "baselines.detector.initial.tracked_candidate_index"
            ),
        },
    }

    frame_indices = decoded.get("gripper_rep_tracking_frame_indices")
    if frame_indices is not None:
        frame_indices = np.asarray(frame_indices, dtype=np.int32).reshape(-1)
        if len(frame_indices) == converted.shape[1]:
            frame_indices = frame_indices + int(annotation.get("start_frame", 0))
            arrays["centroid_trace_frame_indices"] = frame_indices
            manifest["centroid_trace_frame_indices"] = {
                "dtype": "int32",
                "shape": list(frame_indices.shape),
                "axes": ["tracking_timeline"],
                "coordinates": "zero_based_trajectory_indices",
                "description": "absolute trajectory frame indices for centroid_traces",
            }

    return arrays, manifest


def _public_source_frame_arrays(
    annotation: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Exact trajectory-index mapping back to the source episode timeline."""
    context = annotation.get("_publication_context") or {}
    if not isinstance(context, Mapping):
        return {}, {}

    source_indices = context.get("source_episode_frame_indices")
    timestamps = context.get("frame_timestamps_seconds")
    if source_indices is None or timestamps is None:
        return {}, {}
    source_indices = np.asarray(source_indices, dtype=np.int64).reshape(-1)
    timestamps = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if not len(source_indices) or len(source_indices) != len(timestamps):
        return {}, {}

    arrays = {
        "trajectory_source_episode_frame_indices": source_indices,
        "trajectory_timestamps_seconds": timestamps,
    }
    manifest = {
        "trajectory_source_episode_frame_indices": {
            "dtype": "int64",
            "shape": list(source_indices.shape),
            "axes": ["trajectory_timeline"],
            "description": (
                "source episode frame_index for every zero-based trajectory frame"
            ),
        },
        "trajectory_timestamps_seconds": {
            "dtype": "float64",
            "shape": list(timestamps.shape),
            "axes": ["trajectory_timeline"],
            "origin": "source_episode_start",
            "description": (
                "source episode-relative timestamp for every trajectory frame"
            ),
        },
    }
    return arrays, manifest


def _public_keyframe_pointcloud_arrays(
    payloads: list[Mapping[str, Any]],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Normalize ragged masked MoGe clouds from one or more arm payloads."""
    public_arrays: dict[str, np.ndarray] = {}
    manifest: dict[str, Any] = {}

    for public_prefix, internal_prefix in (
        ("object", "winner_object"),
        ("gripper", "robotseg"),
    ):
        xyz_parts = []
        uv_parts = []
        offsets = [0]
        frame_indices = []
        phase_names = []
        arm_indices = []
        sources = []

        for arm_index, payload in enumerate(payloads):
            decoded = payload.get("arrays") or {}
            xyz = decoded.get(f"{internal_prefix}_phase_keyframe_points_xyz")
            uv = decoded.get(f"{internal_prefix}_phase_keyframe_points_uv")
            source_offsets = decoded.get(
                f"{internal_prefix}_phase_keyframe_point_offsets"
            )
            frames = decoded.get(
                f"{internal_prefix}_phase_keyframe_point_absolute_frame_indices"
            )
            names = decoded.get(
                f"{internal_prefix}_phase_keyframe_point_phase_names"
            )
            if any(value is None for value in (xyz, uv, source_offsets, frames, names)):
                continue

            xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
            uv = np.asarray(uv, dtype=np.int32).reshape(-1, 2)
            source_offsets = np.asarray(source_offsets, dtype=np.int64).reshape(-1)
            frames = np.asarray(frames, dtype=np.int32).reshape(-1)
            names = np.asarray(names).astype("U16").reshape(-1)
            record_count = min(len(frames), len(names), max(0, len(source_offsets) - 1))
            if len(xyz) != len(uv) or record_count == 0:
                continue

            for record_index in range(record_count):
                begin = int(source_offsets[record_index])
                end = int(source_offsets[record_index + 1])
                if begin < 0 or end < begin or end > len(xyz):
                    continue
                xyz_part = xyz[begin:end]
                uv_part = uv[begin:end]
                xyz_parts.append(xyz_part)
                uv_parts.append(uv_part)
                offsets.append(offsets[-1] + len(xyz_part))
                frame_indices.append(int(frames[record_index]))
                phase_names.append(str(names[record_index]))
                arm_indices.append(arm_index)
                sources.append(
                    (payload.get("lightweight") or {}).get(
                        "phase_keyframe_pointcloud_metadata", {}
                    )
                )

        if not frame_indices:
            continue

        xyz_key = f"{public_prefix}_keyframe_points_xyz"
        uv_key = f"{public_prefix}_keyframe_points_uv"
        offsets_key = f"{public_prefix}_keyframe_point_offsets"
        frames_key = f"{public_prefix}_keyframe_point_frame_indices"
        phases_key = f"{public_prefix}_keyframe_point_phase_names"
        public_arrays.update({
            xyz_key: np.concatenate(xyz_parts, axis=0).astype(np.float32, copy=False),
            uv_key: np.concatenate(uv_parts, axis=0).astype(np.int32, copy=False),
            offsets_key: np.asarray(offsets, dtype=np.int64),
            frames_key: np.asarray(frame_indices, dtype=np.int32),
            phases_key: np.asarray(phase_names, dtype="U16"),
        })
        if len(payloads) > 1:
            public_arrays[f"{public_prefix}_keyframe_point_arm_indices"] = np.asarray(
                arm_indices, dtype=np.int8,
            )

        metadata = next((source for source in sources if source), {})
        manifest.update({
            xyz_key: {
                "dtype": "float32",
                "shape": list(public_arrays[xyz_key].shape),
                "axes": ["points", "xyz"],
                "coordinate_frame": metadata.get(
                    "xyz_coordinate_frame", "MoGe_camera_coordinates"
                ),
                "offsets_array": offsets_key,
                "source": "MoGe_points_inside_phase_keyframe_mask",
            },
            uv_key: {
                "dtype": "int32",
                "shape": list(public_arrays[uv_key].shape),
                "axes": ["points", "uv"],
                "coordinate_frame": metadata.get(
                    "uv_coordinate_frame",
                    "original_uncropped_unresized_image",
                ),
                "offsets_array": offsets_key,
            },
            offsets_key: {
                "dtype": "int64",
                "shape": list(public_arrays[offsets_key].shape),
                "axes": ["keyframes_plus_one"],
                "description": "ragged point ranges for the corresponding XYZ and UV arrays",
            },
            frames_key: {
                "dtype": "int32",
                "shape": list(public_arrays[frames_key].shape),
                "axes": ["keyframes"],
                "coordinates": "zero_based_trajectory_indices",
            },
            phases_key: {
                "dtype": "unicode",
                "shape": list(public_arrays[phases_key].shape),
                "axes": ["keyframes"],
            },
        })
        arm_key = f"{public_prefix}_keyframe_point_arm_indices"
        if arm_key in public_arrays:
            manifest[arm_key] = {
                "dtype": "int8",
                "shape": list(public_arrays[arm_key].shape),
                "axes": ["keyframes"],
                "description": "index into the top-level arms array",
            }

    return public_arrays, manifest


def _selected_candidate_index(annotation: Mapping[str, Any]) -> Optional[int]:
    """Winner candidate index, resolved exactly as _selection() reports it."""
    breakdown = annotation.get("confidence_breakdown") or {}
    if not isinstance(breakdown, Mapping):
        breakdown = {}
    lightweight = _lightweight(annotation)
    override = breakdown.get("selection_winner_override") or {}
    winner_pick = lightweight.get("winner_pick") or {}
    winner_idx = override.get(
        "new_winner_idx",
        winner_pick.get("best_start_idx", lightweight.get("winner_idx")),
    )
    if winner_idx is None:
        return None
    try:
        return int(winner_idx)
    except (TypeError, ValueError):
        return None


def _detector_baseline(annotation: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    """Return the compact max-detector-score baseline from pipeline candidates."""
    lightweight = _lightweight(annotation)
    initial_idx, initial_box, initial_score = _highest_detector_score_candidate(
        lightweight.get("candidate_boxes"),
        lightweight.get("candidate_confidences"),
    )
    if initial_idx is None or initial_box is None:
        return None

    start_frame = int(annotation.get("start_frame", 0))
    end_frame = int(annotation.get("end_frame", start_frame))
    detection_rel = int(annotation.get("detection_frame_index", 0))
    source_indices = lightweight.get("source_candidate_idx") or []
    source_idx = (
        int(source_indices[initial_idx])
        if initial_idx < len(source_indices)
        else None
    )
    initial = _clean_mapping({
        "frame_index": start_frame + detection_rel,
        "box_xyxy_pixels": initial_box,
        "detector_score": initial_score,
        "tracked_candidate_index": initial_idx,
        "source_detection_index": source_idx,
    })

    target_idx, target_box, target_score = _highest_detector_score_candidate(
        lightweight.get("target_candidate_boxes"),
        lightweight.get("target_candidate_confidences"),
    )
    target = None
    if target_idx is not None and target_box is not None:
        target = _clean_mapping({
            "frame_index": max(start_frame, end_frame - 1),
            "box_xyxy_pixels": target_box,
            "detector_score": target_score,
            "candidate_index": target_idx,
        })

    track = _centroid_track_for_candidate(
        annotation,
        lightweight,
        initial_idx,
        start_frame=start_frame,
        end_frame=end_frame,
        detection_rel=detection_rel,
    )

    return _clean_mapping({
        "method": "maximum_detector_score",
        "initial": initial,
        "target": target,
        "track": track,
    })


def _source_identifiers(
    annotation: Mapping[str, Any],
) -> dict[str, Any]:
    trajectory_value = annotation.get("trajectory_name")
    dataset_value = annotation.get("dataset_name")
    trajectory_id = "" if trajectory_value is None else str(trajectory_value)
    dataset_id = "" if dataset_value is None else str(dataset_value)
    numeric_parts = [
        int(match.group())
        for part in trajectory_id.split("/")
        if (match := re.search(r"\d+", part)) is not None
    ]
    is_agibot = "agibot" in dataset_id.lower() or annotation.get("longtask_id") is not None
    episode_index = annotation.get("source_episode_index")
    subtrajectory_index = annotation.get("source_subtrajectory_index")
    if episode_index is not None:
        episode_index = int(episode_index)
    if subtrajectory_index is not None:
        subtrajectory_index = int(subtrajectory_index)
    if episode_index is None and numeric_parts:
        if is_agibot and len(numeric_parts) >= 2:
            episode_index = numeric_parts[-2]
        else:
            episode_index = numeric_parts[-1]
    if subtrajectory_index is None and is_agibot and len(numeric_parts) >= 2:
        subtrajectory_index = numeric_parts[-1]

    context = annotation.get("_publication_context") or {}
    camera_view = context.get("camera_view") if isinstance(context, Mapping) else None
    source = {
        "dataset_id": dataset_id,
        "dataset_format": (
            context.get("source_dataset_format")
            if isinstance(context, Mapping) else None
        ),
        "trajectory_id": trajectory_id,
        "source_uuid": annotation.get("source_uuid"),
        "source_episode_id": annotation.get("source_episode_id"),
        "episode_index": episode_index,
        "subtrajectory_index": subtrajectory_index,
        "episode_frame_start": annotation.get("source_episode_frame_start"),
        "episode_frame_end_exclusive": annotation.get(
            "source_episode_frame_end_exclusive"
        ),
        "subtask_index": annotation.get("subtask_index"),
        "split": context.get("split") if isinstance(context, Mapping) else None,
        "fps": context.get("fps") if isinstance(context, Mapping) else None,
        "camera_view": camera_view,
    }
    if (
        isinstance(context, Mapping)
        and context.get("source_episode_frame_indices") is not None
        and context.get("frame_timestamps_seconds") is not None
    ):
        source["frame_mapping"] = {
            "trajectory_frame_index": (
                "array index on trajectory_source_episode_frame_indices and "
                "trajectory_timestamps_seconds"
            ),
            "source_episode_frame_indices_array": (
                "trajectory_source_episode_frame_indices"
            ),
            "timestamps_seconds_array": "trajectory_timestamps_seconds",
            "timestamp_origin": "source_episode_start",
        }
    long_task = _clean_mapping({
        "id": annotation.get("longtask_id"),
        "step": annotation.get("longtask_step"),
        "length": annotation.get("longtask_len"),
        "goal": annotation.get("longtask_goal"),
    })
    if long_task:
        source["long_task"] = long_task
    return _clean_mapping(source)


def _annotation_id(source: Mapping[str, Any], window: Mapping[str, Any]) -> str:
    identity = {
        "dataset_id": source.get("dataset_id"),
        "trajectory_id": source.get("trajectory_id"),
        "subtask_index": source.get("subtask_index"),
        "start_frame": window.get("start_frame"),
        "end_frame_exclusive": window.get("end_frame_exclusive"),
        "camera_view": source.get("camera_view"),
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _robot_record(annotation: Mapping[str, Any]) -> dict[str, Any]:
    """Publication provenance for robot masks and detector fallback."""
    context = annotation.get("_publication_context") or {}
    if not isinstance(context, Mapping):
        context = {}
    decoded = decode_arrays_blob(_lightweight(annotation).get("arrays_blob"))
    has_robotseg_masks = decoded.get("robotseg_phase_keyframe_masks") is not None
    robotseg_enabled = context.get("robotseg_enabled")
    if robotseg_enabled is None and has_robotseg_masks:
        robotseg_enabled = True

    segmentation = None
    if has_robotseg_masks:
        segmentation = {
            "source": "RobotSeg",
            "category_prompt": context.get("robotseg_category", "robot"),
            "threshold": "logit_gt_0",
            "postprocessing": "none",
            "detector_fallback_used": False,
        }
    elif robotseg_enabled:
        segmentation = {
            "source": "detector_plus_SAM2_fallback",
            "detector_fallback_used": True,
        }

    return _clean_mapping({
        "arm": annotation.get("arm", annotation.get("gripper_arm")),
        "segmentation": segmentation,
    })


def _object_record(annotation: Mapping[str, Any]) -> dict[str, Any]:
    start_frame = int(annotation.get("start_frame", 0))
    end_frame = int(annotation.get("end_frame", start_frame))
    detection_rel = int(annotation.get("detection_frame_index", 0))
    initial = {
        "frame_index": start_frame + detection_rel,
        "box_xyxy_pixels": _single_box(annotation.get("initial_object_box")),
        "detector_score": _detector_score(annotation),
    }
    target = {
        "frame_index": max(start_frame, end_frame - 1),
        "box_xyxy_pixels": _single_box(annotation.get("target_object_box")),
        "detector_score": annotation.get("target_object_box_confidence"),
        "verified": annotation.get("verified_target"),
    }
    intermediate = []
    for frame_rel, box in sorted(
        (annotation.get("intermediate_object_boxes") or {}).items(),
        key=lambda item: int(item[0]),
    ):
        normalized = _single_box(box)
        if normalized is not None:
            intermediate.append({
                "frame_index": start_frame + int(frame_rel),
                "box_xyxy_pixels": normalized,
            })

    # Preferred: the winner candidate's visible-point centroid, taken from the same
    # scoring-time ``centroid_traces`` array the detector baseline uses. The annotator's
    # obj_traces_cotracker_point export is computed from ``final.obj_traces`` at a point
    # where CoTracker visibility does not resolve (observed: visibility_used=false with
    # every point averaged), so it is only a fallback for rows without centroid_traces.
    track = _centroid_track_for_candidate(
        annotation,
        _lightweight(annotation),
        _selected_candidate_index(annotation),
        start_frame=start_frame,
        end_frame=end_frame,
        detection_rel=detection_rel,
    )

    point_track = annotation.get("obj_traces_cotracker_point")
    point_meta = annotation.get("obj_traces_cotracker_point_meta")
    if track is None and point_track is not None:
        arr = np.asarray(point_track, dtype=float)
        if arr.ndim == 3 and arr.shape[1] == 1 and arr.shape[2] >= 2:
            arr = arr[:, 0, :2]
        if arr.ndim == 2 and arr.shape[1] >= 2:
            point_meta = dict(point_meta) if isinstance(point_meta, Mapping) else {}
            relative_indices = point_meta.pop("frame_indices_relative_to_window", None)
            absolute_indices = None
            if relative_indices is not None and len(relative_indices) == len(arr):
                absolute_indices = [start_frame + int(idx) for idx in relative_indices]
            track = {
                "xy_pixels": arr[:, :2].tolist(),
                "frame_indices": absolute_indices,
                "sampling": "tracking_timeline" if absolute_indices is None else None,
                "point_selection": point_meta or None,
            }
            track = _clean_mapping(track)

    task_info = annotation.get("task_obj_info") or {}
    if not isinstance(task_info, Mapping):
        task_info = {}
    result = {
        "label": task_info.get("object"),
        "instance_id": task_info.get("object_instance"),
        "relation_to_previous": task_info.get("instance_relation_to_previous"),
        "initial": _clean_mapping(initial),
        "target": _clean_mapping(target),
        "intermediate": intermediate or None,
        "track": track,
    }
    return _clean_mapping(result)


def _detector_score(annotation: Mapping[str, Any]) -> Any:
    score = annotation.get("initial_object_box_detection_confidence")
    if score is not None:
        return score
    breakdown = annotation.get("confidence_breakdown") or {}
    if isinstance(breakdown, Mapping) and breakdown.get("detection_confidence") is not None:
        return breakdown.get("detection_confidence")
    return None


def build_publication_annotation(
    annotation: Mapping[str, Any],
    *,
    original_image_size_hw: Optional[tuple[int, int]] = None,
    embed_arrays: bool = True,
    save_pointclouds: bool = False,
) -> dict[str, Any]:
    """Build v2 output; defer encoding for transport to AnnotationSaver when requested."""
    context = annotation.get("_publication_context")
    if "schema" in annotation or not isinstance(context, Mapping):
        raise ValueError("Expected current pipeline state with _publication_context")
    image_size = original_image_size_hw or context.get("image_size_hw")
    if image_size is None or len(image_size) != 2:
        raise ValueError("Current pipeline state is missing original image size")
    image_size = (int(image_size[0]), int(image_size[1]))

    is_bimanual = bool(annotation.get("is_bimanual"))
    mask_sources = [
        arm for arm in (annotation.get("arm_annotations") or [])
        if isinstance(arm, Mapping)
    ] if is_bimanual else [annotation]
    mask_payloads = []
    for source_annotation in mask_sources:
        lightweight = _lightweight(source_annotation)
        decoded = decode_arrays_blob(lightweight.get("arrays_blob"))
        mask_payloads.append({
            "annotation": source_annotation,
            "lightweight": lightweight,
            "arrays": decoded,
            "object_series": _mask_series(decoded, "winner_object"),
            "robot_series": _mask_series(decoded, "robotseg"),
        })

    phases = _phase_rows(annotation)
    arrays_section = None
    geometry_section = None
    has_masks = any(
        payload["object_series"] or payload["robot_series"]
        for payload in mask_payloads
    )
    if has_masks:
        _append_missing_mask_phases(
            phases,
            [
                [item for item in payload[key] if item[0] != "detection"]
                for payload in mask_payloads
                for key in ("object_series", "robot_series")
            ],
        )
        per_source_arrays = []
        for payload in mask_payloads:
            crop_box, global_scale, coordinate_frame = _mask_transform(
                payload["lightweight"],
            )

            def restore(mask, _crop=crop_box, _scale=global_scale, _frame=coordinate_frame):
                return _restore_mask(mask, image_size, _crop, _scale, _frame)

            _, source_arrays = _align_masks_to_phases(
                phases,
                [
                    item for item in payload["object_series"]
                    if item[0] != "detection"
                ],
                [
                    item for item in payload["robot_series"]
                    if item[0] != "detection"
                ],
                image_size,
                restore,
            )

            height, width = image_size
            detection_frame_index = -1
            for public_name, series_name in (
                ("detection_object", "object_series"),
                ("detection_robot", "robot_series"),
            ):
                restored_detection = None
                for name, frame, mask in payload[series_name]:
                    if name == "detection":
                        detection_frame_index = int(frame)
                        restored_detection = restore(mask)
                        break
                source_arrays[f"{public_name}_mask"] = (
                    restored_detection
                    if restored_detection is not None
                    else np.zeros((height, width), dtype=np.uint8)
                )
                source_arrays[f"{public_name}_mask_valid"] = np.asarray(
                    restored_detection is not None,
                    dtype=bool,
                )
            source_arrays["detection_frame_index"] = np.asarray(
                detection_frame_index,
                dtype=np.int32,
            )
            per_source_arrays.append(source_arrays)

        if is_bimanual:
            object_masks = np.stack([
                item["object_masks"] for item in per_source_arrays
            ])
            object_valid = np.stack([
                item["object_mask_valid"] for item in per_source_arrays
            ])
            robot_masks = np.maximum.reduce([
                item["robot_masks"] for item in per_source_arrays
            ])
            robot_valid = np.logical_or.reduce([
                item["robot_mask_valid"] for item in per_source_arrays
            ])
            public_arrays = {
                "object_masks": object_masks,
                "object_mask_valid": object_valid,
                "robot_masks": robot_masks,
                "robot_mask_valid": robot_valid,
                "detection_object_mask": np.stack([
                    item["detection_object_mask"]
                    for item in per_source_arrays
                ]),
                "detection_object_mask_valid": np.stack([
                    item["detection_object_mask_valid"]
                    for item in per_source_arrays
                ]),
                "detection_robot_mask": np.stack([
                    item["detection_robot_mask"]
                    for item in per_source_arrays
                ]),
                "detection_robot_mask_valid": np.stack([
                    item["detection_robot_mask_valid"]
                    for item in per_source_arrays
                ]),
                "detection_frame_indices": np.stack([
                    item["detection_frame_index"]
                    for item in per_source_arrays
                ]),
            }
            object_axes = ["arms", "task.phases", "height", "width"]
            object_valid_axes = ["arms", "task.phases"]
        else:
            public_arrays = per_source_arrays[0]
            object_axes = ["task.phases", "height", "width"]
            object_valid_axes = ["task.phases"]

        public_arrays["phase_mask_frame_indices"] = np.asarray([
            int(phase.get("end_frame_inclusive", 0)) for phase in phases
        ], dtype=np.int32)

        centroid_arrays, centroid_manifest = _public_centroid_arrays(
            annotation, is_bimanual,
        )
        public_arrays.update(centroid_arrays)
        source_frame_arrays, source_frame_manifest = _public_source_frame_arrays(
            annotation,
        )
        public_arrays.update(source_frame_arrays)
        pointcloud_arrays, pointcloud_manifest = (
            _public_keyframe_pointcloud_arrays(mask_payloads)
            if save_pointclouds else ({}, {})
        )
        public_arrays.update(pointcloud_arrays)
        if pointcloud_arrays:
            source_metadata = []
            for payload in mask_payloads:
                metadata = (payload.get("lightweight") or {}).get(
                    "phase_keyframe_pointcloud_metadata"
                )
                if metadata:
                    source_metadata.append(dict(metadata))
            geometry_section = {
                "keyframe_pointclouds": {
                    "array_prefixes": ["object_keyframe", "gripper_keyframe"],
                    "keyframes": (
                        "detection frame and ends of grasp/interact/release phases "
                        "when valid"
                    ),
                    "frame_indices": "zero_based_trajectory_indices",
                    "source_episode_frame_mapping": (
                        "arrays.trajectory_source_episode_frame_indices when "
                        "present; otherwise source.episode_frame_start + "
                        "trajectory_frame_index"
                    ),
                    "point_selection": (
                        "valid MoGe points whose pixels lie inside the corresponding mask"
                    ),
                    "pixel_aligned_reconstruction": (
                        "for each keyframe slice from offsets, scatter points_xyz "
                        "at points_uv into an image-sized XYZ array"
                    ),
                    "coordinate_scope": (
                        "camera coordinates per keyframe; not a shared metric world frame"
                    ),
                    "inference": source_metadata[0] if source_metadata else None,
                }
            }

        blob = encode_arrays_blob(public_arrays) if embed_arrays else None
        arrays_section = {
            "encoding": ARRAYS_ENCODING,
            "blob": blob,
            "manifest": {
                "spatial_coordinates": "original_uncropped_unresized_image",
                "object_masks": {
                    "dtype": "uint8",
                    "shape": list(public_arrays["object_masks"].shape),
                    "axes": object_axes,
                    "validity_array": "object_mask_valid",
                },
                "object_mask_valid": {
                    "dtype": "bool",
                    "shape": list(public_arrays["object_mask_valid"].shape),
                    "axes": object_valid_axes,
                },
                "robot_masks": {
                    "dtype": "uint8",
                    "shape": list(public_arrays["robot_masks"].shape),
                    "axes": ["task.phases", "height", "width"],
                    "validity_array": "robot_mask_valid",
                    "source": "RobotSeg_category_robot",
                },
                "robot_mask_valid": {
                    "dtype": "bool",
                    "shape": list(public_arrays["robot_mask_valid"].shape),
                    "axes": ["task.phases"],
                },
                "phase_mask_frame_indices": {
                    "dtype": "int32",
                    "shape": list(public_arrays["phase_mask_frame_indices"].shape),
                    "axes": ["task.phases"],
                    "coordinates": "zero_based_trajectory_indices",
                },
                "detection_object_mask": {
                    "dtype": "uint8",
                    "shape": list(public_arrays["detection_object_mask"].shape),
                    "axes": (
                        ["arms", "height", "width"]
                        if is_bimanual else ["height", "width"]
                    ),
                    "frame_index_array": (
                        "detection_frame_indices" if is_bimanual
                        else "detection_frame_index"
                    ),
                    "validity_array": "detection_object_mask_valid",
                },
                "detection_object_mask_valid": {
                    "dtype": "bool",
                    "shape": list(
                        public_arrays["detection_object_mask_valid"].shape
                    ),
                },
                "detection_robot_mask": {
                    "dtype": "uint8",
                    "shape": list(public_arrays["detection_robot_mask"].shape),
                    "axes": (
                        ["arms", "height", "width"]
                        if is_bimanual else ["height", "width"]
                    ),
                    "frame_index_array": (
                        "detection_frame_indices" if is_bimanual
                        else "detection_frame_index"
                    ),
                    "validity_array": "detection_robot_mask_valid",
                    "source": "RobotSeg_category_robot",
                },
                "detection_robot_mask_valid": {
                    "dtype": "bool",
                    "shape": list(
                        public_arrays["detection_robot_mask_valid"].shape
                    ),
                },
                (
                    "detection_frame_indices"
                    if is_bimanual else "detection_frame_index"
                ): {
                    "dtype": "int32",
                    "shape": list(public_arrays[
                        "detection_frame_indices"
                        if is_bimanual else "detection_frame_index"
                    ].shape),
                    "axes": ["arms"] if is_bimanual else [],
                    "coordinates": "zero_based_trajectory_indices",
                },
                **centroid_manifest,
                **source_frame_manifest,
                **pointcloud_manifest,
            },
        }

        if not embed_arrays:
            arrays_section.pop("blob")
            arrays_section["_values"] = {
                key: value for key, value in public_arrays.items()
                if value is not None and np.asarray(value).size
            }

    start_frame = int(annotation.get("start_frame", 0))
    end_frame = int(annotation.get("end_frame", start_frame))
    window = {
        "start_frame": start_frame,
        "end_frame_exclusive": end_frame,
        "trajectory_frame_count": annotation.get("num_frames"),
        "image_size_hw": list(image_size) if image_size is not None else None,
    }
    source = _source_identifiers(annotation)
    if not source.get("trajectory_id"):
        raise ValueError("pipeline state is missing trajectory_name")
    if not source.get("dataset_id"):
        raise ValueError(
            "pipeline state is missing dataset_name"
        )
    task_info = annotation.get("task_obj_info") or {}
    if not isinstance(task_info, Mapping):
        task_info = {}
    task = {
        "instruction": annotation.get("language_instruction"),
        "parsing_mode": annotation.get("task_object_parsing_mode", "text"),
        "action": annotation.get("movement_type", task_info.get("action")),
        "start_location": task_info.get("start_location"),
        "target_location": task_info.get("target_location"),
        "phases": phases,
    }

    result = {
        "schema": {"name": SCHEMA_NAME, "version": SCHEMA_VERSION},
        "annotation_id": _annotation_id(source, window),
        "coordinates": {
            "image_reference": "original_uncropped_unresized_image",
            "boxes": "pixel_xyxy",
            "masks": "binary_image_aligned",
            "frame_indices": "zero_based_trajectory_indices",
        },
        "source": source,
        "window": _clean_mapping(window),
        "task": _clean_mapping(task),
        "arrays": arrays_section,
        "geometry": geometry_section,
    }

    if not is_bimanual:
        result["object"] = _object_record(annotation)
        result["selection"] = _selection(annotation)
        detector_baseline = _detector_baseline(annotation)
        if detector_baseline is not None:
            result["baselines"] = {"detector": detector_baseline}
        robot = _robot_record(annotation)
        if robot:
            result["robot"] = robot

    if annotation.get("is_tool_task"):
        result["tool"] = _clean_mapping({
            "name": annotation.get("tool_name"),
            "usage_description": annotation.get("tool_usage_description"),
            "box_xyxy_pixels": _single_box(annotation.get("tool_box")),
            "detector_score": annotation.get("tool_box_confidence"),
        })
    if is_bimanual:
        result["arms"] = []
        for array_index, arm in enumerate(mask_sources):
            arm_annotation = dict(annotation)
            arm_annotation.update(arm)
            arm_annotation["task_obj_info"] = {"object": arm.get("object")}
            arm_annotation.pop("verified_target", None)
            detector_baseline = _detector_baseline(arm_annotation)
            result["arms"].append(_clean_mapping({
                "arm": arm.get("arm"),
                "subtask_index": arm.get("subtask_index"),
                "object_mask_array_index": array_index if arrays_section is not None else None,
                "object": _object_record(arm_annotation),
                "selection": _selection(arm_annotation),
                "baselines": (
                    {"detector": detector_baseline}
                    if detector_baseline is not None
                    else None
                ),
            }))
    return {key: value for key, value in result.items() if value is not None}
