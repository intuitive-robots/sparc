"""Dataclasses for intermediate results passed between pipeline stages."""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class DetectionResult:
    """Output of _detect_objects()."""
    pred_boxes: object  # sv.Detections (list wrapper)
    candidate_boxes: list  # list of dicts with box, det_confidence, frame_index, confidence
    robot_det_dict: dict  # frame_idx -> {"box": [x1,y1,x2,y2], "confidence": float}
    robot_candidates_all: list  # per-keyframe list of sv.Detections (all robot boxes, for debug)
    is_tool_task: bool
    tool_name: Optional[str]
    tool_boxes_fullres: Optional[np.ndarray]  # (N, 4) or None
    object_box_for_tool_task: Optional[np.ndarray]  # (1, 4) or None
    object_box_for_tool_task_confidence: Optional[float]
    pnp_task: bool
    all_boxes_np: np.ndarray  # (N, 4) union of all detected boxes
    detected_boxes_fullres: np.ndarray  # (N, 4) copy of initial detections
    arm: Optional[str] = None  # "left" | "right" | None for single-arm
    intermediate_detections: Optional[dict] = None
    # {rel_frame_idx: [{"box": [x1,y1,x2,y2], "confidence": float}, ...]}
    keyframe_indices: Optional[list] = None
    # task_grasp_phase_frame_indices — positional index into robot_candidates_all
    gripper_box: Optional[np.ndarray] = None  # (1, 4) or None, from det_frame_idx
    gripper_box_confidence: Optional[float] = None


@dataclass
class TargetResult:
    """Output of _detect_targets()."""
    target_boxes: list  # list of dicts with box, det_confidence, frame_index, label
    target_boxes_fullres: Optional[np.ndarray]  # (N, 4) or None
    pred_boxes_target_merged: object  # sv.Detections (list wrapper) or None
    target_masks: Optional[np.ndarray] = None  # (M, H, W) uint8 SAM2 masks on target frame


@dataclass
class CropInfo:
    """Output of _maybe_crop(). Captures cropping state for coordinate restoration."""
    did_crop: bool
    crop_box: Optional[tuple]  # (min_x, min_y, max_x, max_y) or None
    offsets: Optional[tuple]  # (min_x, min_y, min_x_fwd, min_y_fwd) or None
    cropped_start_frame: Optional[np.ndarray]
    cropped_detected_boxes: Optional[np.ndarray]
    cropped_target_boxes: Optional[np.ndarray]
    original_shape: Optional[tuple] = None  # (H, W) before crop, after any global resize


@dataclass
class TrackingResult:
    """Output of _track_and_select()."""
    best_obj_box: Optional[np.ndarray]  # (1, 4) or None
    best_obj_box_confidence: Optional[float]
    best_obj_box_target: Optional[np.ndarray]  # (1, 4) or None
    obj_traces: Optional[np.ndarray]  # (T, N, 2) or None
    verified_target: bool
    intermediate_boxes: Optional[dict] = None
    # {rel_frame_idx: [x1,y1,x2,y2]} — tracked box at each intermediate keyframe
    best_obj_mask: Optional[np.ndarray] = None
    # (H, W) uint8 — SAM2 image-predictor mask for the selected best box at det frame
    confidence_breakdown: Optional[dict] = None
    # Per-component confidence scores for annotation quality filtering
    raw_tracks_data: Optional[dict] = None
    # Per-candidate raw track arrays for offline movement analysis.
    # Keys: 'cotracker_tracks' (n,T,N,2), 'cotracker_visibility' (n,T,N),
    #       'winner_idx' (int),
    #       'candidate_boxes' (n,4)


@dataclass
class FinalBoxes:
    """Output of _restore_coordinates(). All boxes in original (uncropped) coordinates."""
    initial_object_box: np.ndarray  # (1, 4)
    initial_object_box_confidence: Optional[float]
    target_object_box: np.ndarray  # (1, 4) or (0, 4)
    target_object_box_confidence: Optional[float]
    obj_traces: Optional[np.ndarray]
    obj_traces_target: Optional[np.ndarray]
    verified_target: bool
    # Tool-specific
    tool_box: Optional[np.ndarray]
    tool_box_confidence: Optional[float]
    tool_traces: Optional[np.ndarray]
    arm: Optional[str] = None  # "left" | "right" | None for single-arm
    intermediate_boxes: Optional[dict] = None
    # {rel_frame_idx: [x1,y1,x2,y2]} — tracked box at each intermediate keyframe
    target_candidate_box_confidences: Optional[list] = None
    # List of det_confidence floats for each target candidate box (before selection)
    confidence_breakdown: Optional[dict] = None
    # Per-component confidence scores for annotation quality filtering
