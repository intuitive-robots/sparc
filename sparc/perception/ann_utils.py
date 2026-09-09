import math
import cv2
import numpy as np
from sparc.perception.detection_utils import get_top_k_boxes
from supervision import Detections

import torch
from torchvision.ops import nms


DEFAULT_MAX_BOX_AREA_RATIO = 0.8
DEFAULT_DETECTION_TEXT_THRESHOLD = 0.1
STANDARD_DETECTION_TEXT_THRESHOLD = 0.005
TOOL_OBJECT_DETECTION_THRESHOLD = 0.25
DEFAULT_TRACK_VISIBILITY_THRESHOLD = 0.5


def _candidate_point_slices(pred_tracks_per_box):
    """Return flattened point slices without assuming square sampling grids."""
    counts = [int(tracks.shape[1]) for tracks in pred_tracks_per_box]
    offsets = np.cumsum([0, *counts], dtype=np.int64)
    slices = [
        slice(int(offsets[idx]), int(offsets[idx + 1]))
        for idx in range(len(counts))
    ]
    return slices, counts


def robust_tracked_box(
    initial_box,
    tracks,
    visibility,
    frame_index,
    image_shape,
    *,
    visibility_threshold=DEFAULT_TRACK_VISIBILITY_THRESHOLD,
    min_points=4,
):
    """Move an initial box with robustly matched visible point tracks.

    A partial-affine RANSAC fit handles translation, rotation, and modest scale
    change. Median translation is used when the affine fit is underconstrained
    or implausible. Points must be finite, confidently visible in both frames,
    and inside the image; large coordinate errors therefore cannot expand the
    exported box.
    """
    tracks_np = np.asarray(
        tracks.detach().cpu().numpy() if hasattr(tracks, "detach") else tracks,
        dtype=np.float32,
    )
    visibility_np = np.asarray(
        visibility.detach().cpu().numpy()
        if hasattr(visibility, "detach")
        else visibility,
        dtype=np.float32,
    )
    box = np.asarray(initial_box, dtype=np.float32).reshape(4)
    height, width = (int(image_shape[0]), int(image_shape[1]))
    frame_index = int(frame_index)
    diagnostics = {
        "method": "invalid",
        "tracking_tensor_index": frame_index,
        "n_eligible": 0,
        "n_inliers": 0,
    }

    if (
        tracks_np.ndim != 3
        or tracks_np.shape[-1] != 2
        or visibility_np.shape != tracks_np.shape[:2]
        or not (0 <= frame_index < tracks_np.shape[0])
        or width <= 1
        or height <= 1
    ):
        diagnostics["reason"] = "invalid_track_shape"
        return None, diagnostics

    source = tracks_np[0]
    destination = tracks_np[frame_index]
    valid = (
        (visibility_np[0] >= float(visibility_threshold))
        & (visibility_np[frame_index] >= float(visibility_threshold))
        & np.all(np.isfinite(source), axis=1)
        & np.all(np.isfinite(destination), axis=1)
        & (source[:, 0] >= 0)
        & (source[:, 0] < width)
        & (source[:, 1] >= 0)
        & (source[:, 1] < height)
        & (destination[:, 0] >= 0)
        & (destination[:, 0] < width)
        & (destination[:, 1] >= 0)
        & (destination[:, 1] < height)
    )
    source = source[valid]
    destination = destination[valid]
    diagnostics["n_eligible"] = int(len(source))
    if len(source) < int(min_points):
        diagnostics["reason"] = "insufficient_visible_in_bounds_points"
        return None, diagnostics

    box_width = max(float(box[2] - box[0]), 1.0)
    box_height = max(float(box[3] - box[1]), 1.0)
    reprojection_threshold = max(2.0, 0.05 * math.hypot(box_width, box_height))
    transform = None
    inlier_mask = None
    try:
        transform, inlier_mask = cv2.estimateAffinePartial2D(
            source,
            destination,
            method=cv2.RANSAC,
            ransacReprojThreshold=reprojection_threshold,
            maxIters=1000,
            confidence=0.99,
            refineIters=10,
        )
    except cv2.error:
        transform = None

    tracked_box = None
    if transform is not None and np.all(np.isfinite(transform)):
        scale = float(math.hypot(float(transform[0, 0]), float(transform[0, 1])))
        n_inliers = int(np.asarray(inlier_mask).sum()) if inlier_mask is not None else len(source)
        # A sudden >2x scale change is almost always tracking corruption in the
        # short DROID interaction windows. Preserve translation as a fallback.
        if 0.5 <= scale <= 2.0 and n_inliers >= int(min_points):
            corners = np.asarray(
                [[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]],
                dtype=np.float32,
            )
            transformed = cv2.transform(corners[None], transform)[0]
            tracked_box = np.asarray(
                [
                    transformed[:, 0].min(),
                    transformed[:, 1].min(),
                    transformed[:, 0].max(),
                    transformed[:, 1].max(),
                ],
                dtype=np.float32,
            )
            diagnostics.update(
                method="partial_affine_ransac",
                n_inliers=n_inliers,
                scale=scale,
            )

    if tracked_box is None:
        displacement = destination - source
        median_displacement = np.median(displacement, axis=0)
        residual = np.linalg.norm(displacement - median_displacement, axis=1)
        median_residual = float(np.median(residual))
        mad = float(np.median(np.abs(residual - median_residual)))
        residual_limit = max(2.0, median_residual + 3.0 * 1.4826 * mad)
        inliers = residual <= residual_limit
        if int(inliers.sum()) < int(min_points):
            diagnostics["reason"] = "insufficient_motion_inliers"
            return None, diagnostics
        translation = np.median(displacement[inliers], axis=0)
        tracked_box = box + np.asarray(
            [translation[0], translation[1], translation[0], translation[1]],
            dtype=np.float32,
        )
        diagnostics.update(
            method="median_translation",
            n_inliers=int(inliers.sum()),
            scale=1.0,
        )

    unclipped = tracked_box.copy()
    tracked_box[[0, 2]] = np.clip(tracked_box[[0, 2]], 0, width - 1)
    tracked_box[[1, 3]] = np.clip(tracked_box[[1, 3]], 0, height - 1)
    if tracked_box[2] <= tracked_box[0] or tracked_box[3] <= tracked_box[1]:
        diagnostics["reason"] = "box_outside_image"
        return None, diagnostics

    diagnostics["clipped"] = bool(not np.allclose(unclipped, tracked_box))
    return tracked_box.astype(float).tolist(), diagnostics


def robust_intermediate_boxes(
    initial_box,
    tracks,
    visibility,
    tracking_frame_indices,
    requested_frame_indices,
    image_shape,
    *,
    visibility_threshold=DEFAULT_TRACK_VISIBILITY_THRESHOLD,
):
    """Build robust boxes at requested trajectory-window frame indices."""
    timeline = np.asarray(tracking_frame_indices, dtype=np.int32).reshape(-1)
    boxes = {}
    diagnostics = {}
    if len(timeline) == 0:
        return boxes, diagnostics

    for requested in sorted({int(frame) for frame in requested_frame_indices}):
        tracking_idx = int(np.argmin(np.abs(timeline - requested)))
        box, frame_diagnostics = robust_tracked_box(
            initial_box,
            tracks,
            visibility,
            tracking_idx,
            image_shape,
            visibility_threshold=visibility_threshold,
        )
        frame_diagnostics["tracking_frame_index"] = int(timeline[tracking_idx])
        diagnostics[requested] = frame_diagnostics
        if box is not None:
            boxes[requested] = box
    return boxes, diagnostics


def _compute_iou(box1, box2):
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (a1 + a2 - inter) if (a1 + a2 - inter) > 0 else 0.0


def _log_baseline_subtracted_movement(movement_matrix, scene_movement_baseline):
    """Compress movement after subtracting scene-level noise floor."""
    movement_signal = np.maximum(np.asarray(movement_matrix) - scene_movement_baseline, 0.0)
    return np.minimum(np.log1p(movement_signal * 100.0) / 5.0, 1.0)


def _compute_baseline_aware_movement_gate(movement, scene_movement_baseline):
    """Soft gate in [0, 1] for target_match and temporal contributions.

    Returns ~0 when a box moves at or below the scene baseline (static objects),
    saturates to 1 when movement reaches 2× the baseline (clearly moving object).
    When no baseline is available, falls back to a small absolute threshold.
    """
    if movement is None or movement <= 0:
        return 0.0
    if scene_movement_baseline <= 0:
        return min(1.0, movement / 0.01)
    return min(1.0, max(0.0, movement / scene_movement_baseline - 1.0))


def _filter_detection_by_area_ratio(detection, image_shape, max_area_ratio):
    if detection is None or len(detection) == 0 or max_area_ratio is None:
        return detection

    image_h, image_w = image_shape[:2]
    if image_h <= 0 or image_w <= 0:
        return detection

    area_threshold = float(image_h * image_w) * float(max_area_ratio)
    valid_mask = filter_boxes_by_size(detection.xyxy, area_threshold)
    return detection[valid_mask]


def get_interacted_obj_boxes(
    detection_model,
    frames,
    task_obj_info,
    threshold=0.1,
    use_target_location=False,
    return_robot_det=False,
    text_threshold=DEFAULT_DETECTION_TEXT_THRESHOLD,
    arm=None,
    return_all_robot_dets=False,
    nms_threshold=0.9,
    max_box_area_ratio=DEFAULT_MAX_BOX_AREA_RATIO,
    skip_robot_detection=False,
):
    """Detect task objects and, unless skipped, robot/gripper boxes."""
    if isinstance(task_obj_info, list):
        task_obj_info = task_obj_info[0]
    obj_name = task_obj_info["object"]
    start_location = task_obj_info["start_location"]
    target_location = task_obj_info["target_location"]

    if start_location is not None:
        prompt = obj_name + " in " + start_location
    else:
        prompt = obj_name

    if use_target_location and target_location is not None:
        prompt = target_location
    else:
        prompt = obj_name

    # Use arm-specific robot prompts for bimanual to help the detector focus on the correct arm.
    # IMPORTANT: keep exactly 2 prompts so class_id 0 and 1 are robot, class_id 2 is the object.
    if arm in ("left", "right"):
        robot_prompts = [f"The {arm} robot arm", f"The {arm} robotic gripper of the robot"]
    else:
        robot_prompts = ["robot arm", "robotic gripper"]

    obj_detections = detection_model.detect_objects(frames, [prompt], threshold=threshold, use_slicing=False, text_threshold=text_threshold)

    if isinstance(obj_detections, dict):
        obj_detections = [Detections(xyxy = obj_detections["xyxy"], confidence = obj_detections["confidence"], class_id = obj_detections["class_id"]) for obj_detections in obj_detections]
    if isinstance(obj_detections[0], dict):
        obj_detections = [Detections(xyxy = detection["xyxy"], confidence = detection["confidence"], class_id = detection["class_id"]) for detection in obj_detections]

    if skip_robot_detection:
        detections = obj_detections
        for detection in detections:
            detection.class_id = np.full(len(detection), 2, dtype=int)
    else:
        detections = detection_model.detect_objects(frames, robot_prompts + [prompt], threshold=threshold, use_slicing=False, text_threshold=text_threshold)

        if isinstance(detections, dict):
            detections = [Detections(xyxy = detections["xyxy"], confidence = detections["confidence"], class_id = detections["class_id"])]
        if isinstance(detections[0], dict):
            detections = [Detections(xyxy = detection["xyxy"], confidence = detection["confidence"], class_id = detection["class_id"]) for detection in detections]

        if len(detections) != len(obj_detections):
            raise ValueError(
                f"get_interacted_obj_boxes: detections length mismatch "
                f"({len(detections)} vs {len(obj_detections)}) — likely a stale RPC reply"
            )

        merged_detections = []
        # Merge the object-only detections with robot/gripper detections.
        for det_idx in range(len(obj_detections)):
            obj_detections[det_idx].class_id = np.full(
                len(obj_detections[det_idx]), 2, dtype=int,
            )
            merged_detections.append(Detections.merge([obj_detections[det_idx], detections[det_idx]]))

        detections = merged_detections

    #only keep detections with class prompt (0)
    cleaned_detections = []
    robot_detections_cleaned = []
    all_robot_candidates = []  # all robot boxes per frame (not just best), for debug

    #check if prompt or target contains "robot, gripper, arm" if so, we dont filter robot detections
    prompt_lower = prompt.lower()
    robot_in_prompt = ("robot" in prompt_lower) or ("gripper" in prompt_lower) or ("arm" in prompt_lower)
    if target_location is not None:
        target_lower = target_location.lower()
        robot_in_prompt = robot_in_prompt or ("robot" in target_lower) or ("gripper" in target_lower) or ("arm" in target_lower)

    robot_in_prompt = False
    for frame_idx, detection in enumerate(detections):

        try:
            if detection.confidence is not None and not robot_in_prompt:
                nms_indices = nms(torch.tensor(detection.xyxy).float(), torch.tensor(detection.confidence), nms_threshold)
                detection = detection[nms_indices]
        except Exception:
            pass

        if isinstance(detection, dict):
            detection = Detections(xyxy = detection["xyxy"], confidence = detection["confidence"], class_id = detection["class_id"])

        valid_detections = detection.class_id == 2

        #robot dets can be id 0 or 1
        robot_dets_mask = (detection.class_id == 0) | (detection.class_id == 1)
        robot_detections = detection[robot_dets_mask]
        all_robot_candidates.append(robot_detections if len(robot_detections) > 0 else None)
        #get best robot detection (highest conf)
        if len(robot_detections) > 0:
            best_robot_det_idx = torch.argmax(torch.tensor(robot_detections.confidence))
            best_robot_det = robot_detections[[best_robot_det_idx]]
            robot_detections_cleaned.append(best_robot_det)
        else:
            robot_detections_cleaned.append(None)

        object_detections = detection[valid_detections]
        object_detections = _filter_detection_by_area_ratio(
            object_detections,
            frames[frame_idx].shape,
            max_box_area_ratio,
        )
        cleaned_detections.append(object_detections)


    #get top k boxes
    detections = get_top_k_boxes(cleaned_detections, k=22)

    if return_robot_det:
        if return_all_robot_dets:
            return detections, robot_detections_cleaned, all_robot_candidates
        return detections, robot_detections_cleaned


    return detections


def filter_boxes_by_size(boxes, area_threshold):
    boxes = np.asarray(boxes)
    if boxes.size == 0:
        return np.zeros((0,), dtype=bool)

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    areas = widths * heights
    return areas <= area_threshold


def compute_box_containment_matrix(boxes):
    """
    Compute a matrix indicating which boxes are contained inside other boxes.

    Args:
        boxes: np.ndarray of shape (N, 4) with boxes in xyxy format

    Returns:
        containment_matrix: np.ndarray of shape (N, N) where containment_matrix[i, j]
                           indicates the ratio of box i that is inside box j (0-1)
        is_inside_matrix: np.ndarray of shape (N, N) boolean, True if box i is mostly inside box j
    """
    n_boxes = len(boxes)
    containment_matrix = np.zeros((n_boxes, n_boxes))
    is_inside_matrix = np.zeros((n_boxes, n_boxes), dtype=bool)

    for i in range(n_boxes):
        x1_i, y1_i, x2_i, y2_i = boxes[i]
        area_i = (x2_i - x1_i) * (y2_i - y1_i)

        if area_i <= 0:
            continue

        for j in range(n_boxes):
            if i == j:
                continue

            x1_j, y1_j, x2_j, y2_j = boxes[j]
            area_j = (x2_j - x1_j) * (y2_j - y1_j)

            if area_j <= 0:
                continue

            # Compute intersection
            inter_x1 = max(x1_i, x1_j)
            inter_y1 = max(y1_i, y1_j)
            inter_x2 = min(x2_i, x2_j)
            inter_y2 = min(y2_i, y2_j)

            if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                # Ratio of box i that is inside box j
                containment_ratio = inter_area / area_i
                containment_matrix[i, j] = containment_ratio

                # Box i is considered "inside" box j if >30% of its area overlaps
                # and box j is larger than box i
                if containment_ratio > 0.3 and area_j > area_i:
                    is_inside_matrix[i, j] = True

    return containment_matrix, is_inside_matrix


def compute_nested_box_penalty(
    start_boxes,
    point_count_matrix,
    movement_matrix,
    similarity_threshold=0.5,
    penalty_factor=0.5
):
    """
    Compute penalty scores for boxes that are nested inside other boxes
    and have similar movement characteristics.

    The intuition: if a small box is inside a larger box and both have similar
    numbers of moving points, the small box is likely just a part of the object
    and should be penalized in favor of the whole object box.

    Args:
        start_boxes: np.ndarray of shape (N, 4) with starting boxes in xyxy format
        point_count_matrix: np.ndarray of shape (N, M) with point counts per (start, target) pair
        movement_matrix: np.ndarray of shape (N, M) with movement scores per (start, target) pair
        similarity_threshold: How similar the point counts need to be (relative difference)
        penalty_factor: Multiplier for the penalty (0-1, applied to score)

    Returns:
        penalty_matrix: np.ndarray of shape (N, M) with penalties to subtract from scores
        nested_info: Dict with information about detected nested boxes
    """
    n_start_boxes = len(start_boxes)
    n_target_boxes = point_count_matrix.shape[1]

    penalty_matrix = np.zeros((n_start_boxes, n_target_boxes))
    nested_info = {
        'nested_pairs': [],  # List of (inner_idx, outer_idx) pairs
        'penalties_applied': []
    }

    # Compute containment relationships between start boxes
    _, is_inside_matrix = compute_box_containment_matrix(start_boxes)

    # For each pair of start boxes where one is inside the other
    for inner_idx in range(n_start_boxes):
        for outer_idx in range(n_start_boxes):
            if not is_inside_matrix[inner_idx, outer_idx]:
                continue

            # inner_idx is inside outer_idx
            nested_info['nested_pairs'].append((inner_idx, outer_idx))

            # Check similarity of point counts for each target
            for target_idx in range(n_target_boxes):
                inner_points = point_count_matrix[inner_idx, target_idx]
                outer_points = point_count_matrix[outer_idx, target_idx]

                inner_movement = movement_matrix[inner_idx, target_idx]
                outer_movement = movement_matrix[outer_idx, target_idx]

                # Skip if either has no points
                if inner_points < 1 or outer_points < 1:
                    continue

                # Compute relative similarity of point counts
                max_points = max(inner_points, outer_points)
                point_similarity = min(inner_points, outer_points) / max_points

                # Compute movement similarity
                max_movement = max(inner_movement, outer_movement)
                if max_movement > 0:
                    movement_similarity = min(inner_movement, outer_movement) / max_movement
                else:
                    movement_similarity = 1.0

                # If both point counts and movements are similar, penalize the inner (smaller) box
                # This suggests the inner box is just a part of the object detected by outer box
                if point_similarity > similarity_threshold and movement_similarity > similarity_threshold:
                    # The penalty is proportional to how similar they are
                    combined_similarity = (point_similarity + movement_similarity) / 2
                    penalty = penalty_factor * combined_similarity

                    # Apply penalty to the inner (smaller) box
                    penalty_matrix[inner_idx, target_idx] = max(
                        penalty_matrix[inner_idx, target_idx],
                        penalty
                    )

                    nested_info['penalties_applied'].append({
                        'inner_box': inner_idx,
                        'outer_box': outer_idx,
                        'target': target_idx,
                        'point_similarity': point_similarity,
                        'movement_similarity': movement_similarity,
                        'penalty': penalty
                    })

    return penalty_matrix, nested_info


def compute_continuous_movement_penalty(
    n_start_boxes,
    n_target_boxes,
    per_frame_movement_per_box,
    grasp_phases,
    n_frames,
    min_movement_threshold=0.001,
    continuous_movement_ratio_threshold=2.0,
    penalty_factor=1.0
):
    """
    Compute a phase-aware penalty from normalized per-frame movement.

    The intended object motion pattern is:
    `grasp -> interact (main motion) -> release`.

    Instead of penalizing any box that moves during non-interact frames, this
    computes how concentrated the motion is in the interact phase and penalizes
    boxes whose movement is not meaningfully larger during interact than during
    grasp/release. This is more robust for true object tracks, which can still
    move a bit during grasp or release.

    Args:
        n_start_boxes: Number of starting boxes
        n_target_boxes: Number of target boxes
        per_frame_movement_per_box: List of tensors, one per start box, each of shape
                                    [T-1] or [T-1, N_valid_points] with normalized
                                    per-frame displacement
        grasp_phases: List of phase dicts with keys: start_frame, end_frame, phase_type
        n_frames: Total number of frames
        min_movement_threshold: Minimum displacement to consider as "movement"
        continuous_movement_ratio_threshold: Interact/non-interact motion ratio
                                             that gets full credit
        penalty_factor: Maximum penalty to apply (0-1)

    Returns:
        penalty_per_box: np.ndarray of shape (n_start_boxes,) with penalties for each box
        continuous_movement_info: Dict with analysis details
    """
    penalty_per_box = np.zeros(n_start_boxes)
    continuous_movement_info = {
        'has_multiple_phases': False,
        'non_interact_frames': 0,
        'per_box_analysis': []
    }

    # Only apply this penalty if we have proper phase information
    # (more than just a single "interact" phase)
    if grasp_phases is None or len(grasp_phases) <= 1:
        return penalty_per_box, continuous_movement_info

    # Check if we have phases other than just "interact"
    phase_types = set()
    for phase in grasp_phases:
        phase_type = phase.get("phase_type", phase.get("phase", ""))
        phase_types.add(phase_type)

    # We need at least one non-interact phase to apply this penalty
    non_interact_phases = phase_types - {"interact"}
    if len(non_interact_phases) == 0:
        return penalty_per_box, continuous_movement_info

    continuous_movement_info['has_multiple_phases'] = True
    continuous_movement_info['phase_types'] = list(phase_types)

    # Build per-phase masks. The penalty operates on displacement, so we align
    # each movement step with its destination frame (frame t corresponds to
    # displacement from t-1 -> t).
    interact_frame_mask = np.zeros(n_frames, dtype=bool)
    phase_frame_masks = {
        "grasp": np.zeros(n_frames, dtype=bool),
        "interact": np.zeros(n_frames, dtype=bool),
        "release": np.zeros(n_frames, dtype=bool),
        "other": np.zeros(n_frames, dtype=bool),
    }
    for phase in grasp_phases:
        phase_type = phase.get("phase_type", phase.get("phase", ""))
        start_f = max(0, phase.get("start_frame", 0))
        end_f = min(n_frames, phase.get("end_frame", n_frames - 1) + 1)
        bucket = phase_type if phase_type in phase_frame_masks else "other"
        phase_frame_masks[bucket][start_f:end_f] = True
        if phase_type == "interact":
            interact_frame_mask[start_f:end_f] = True

    non_interact_frame_mask = ~interact_frame_mask
    margin = min(3, n_frames // 10)
    non_interact_frame_mask[:margin] = False
    non_interact_frame_mask[-margin:] = False
    for mask in phase_frame_masks.values():
        mask[:margin] = False
        mask[-margin:] = False

    n_non_interact_frames = non_interact_frame_mask.sum()
    continuous_movement_info['non_interact_frames'] = int(n_non_interact_frames)

    if n_non_interact_frames < 3:
        return penalty_per_box, continuous_movement_info

    non_interact_displacement_mask = non_interact_frame_mask[1:]
    interact_displacement_mask = interact_frame_mask[1:]
    phase_displacement_masks = {
        name: mask[1:] for name, mask in phase_frame_masks.items()
    }

    for start_idx in range(n_start_boxes):
        if start_idx >= len(per_frame_movement_per_box) or per_frame_movement_per_box[start_idx] is None:
            continue

        displacement = per_frame_movement_per_box[start_idx]

        if displacement.ndim == 1:
            frame_movements = displacement
        else:
            frame_movements = displacement.mean(dim=1) if hasattr(displacement, 'mean') else np.mean(displacement, axis=1)

        if hasattr(frame_movements, 'numpy'):
            frame_movements = frame_movements.numpy()

        if len(frame_movements) != len(non_interact_displacement_mask):
            continue

        interact_sum = float(frame_movements[interact_displacement_mask].sum()) if interact_displacement_mask.any() else 0.0
        non_interact_sum = float(frame_movements[non_interact_displacement_mask].sum()) if non_interact_displacement_mask.any() else 0.0
        total_sum = interact_sum + non_interact_sum

        grasp_sum = float(frame_movements[phase_displacement_masks["grasp"]].sum()) if phase_displacement_masks["grasp"].any() else 0.0
        release_sum = float(frame_movements[phase_displacement_masks["release"]].sum()) if phase_displacement_masks["release"].any() else 0.0
        other_sum = float(frame_movements[phase_displacement_masks["other"]].sum()) if phase_displacement_masks["other"].any() else 0.0

        if total_sum <= min_movement_threshold:
            phase_ratio = 0.0
            gate = 0.0
        elif non_interact_sum <= min_movement_threshold:
            phase_ratio = float("inf")
            gate = 1.0
        else:
            phase_ratio = interact_sum / max(non_interact_sum, 1e-8)
            gate = float(np.clip(
                (phase_ratio - 1.0) / max(continuous_movement_ratio_threshold - 1.0, 1e-6),
                0.0,
                1.0,
            ))

        penalty = penalty_factor * (1.0 - gate)

        box_analysis = {
            'box_idx': start_idx,
            'interact_movement_sum': interact_sum,
            'grasp_movement_sum': grasp_sum,
            'release_movement_sum': release_sum,
            'other_movement_sum': other_sum,
            'non_interact_movement_sum': non_interact_sum,
            'total_movement_sum': total_sum,
            'interact_vs_non_interact_ratio': float(phase_ratio) if np.isfinite(phase_ratio) else None,
            'phase_movement_gate': gate,
            'penalty_applied': penalty,
        }
        penalty_per_box[start_idx] = penalty
        continuous_movement_info['per_box_analysis'].append(box_analysis)

    return penalty_per_box, continuous_movement_info


def compute_per_frame_motion_scores(pred_tracks_per_box, pred_visibility_per_box, fps, window=3):
    """Compute per-frame velocity (pixels/second) for each tracked object.

    For each object and each frame transition, takes the median L2 displacement
    over visible points, normalizes by fps, then applies a rolling mean for
    temporal smoothing.

    Args:
        pred_tracks_per_box: list of [T, N, 2] tensors or arrays, one per object
        pred_visibility_per_box: list of [T, N] tensors or arrays, one per object
        fps: frames per second of the source video
        window: temporal smoothing window size (frames)

    Returns:
        velocities: np.ndarray of shape (n_objects, T-1), smoothed pixels/second
    """
    n_objects = len(pred_tracks_per_box)
    if n_objects == 0:
        return np.zeros((0, 0), dtype=np.float32)

    T = pred_tracks_per_box[0].shape[0]
    if T < 2:
        return np.zeros((n_objects, 0), dtype=np.float32)

    velocities = np.zeros((n_objects, T - 1), dtype=np.float32)

    for k, (tracks, vis) in enumerate(zip(pred_tracks_per_box, pred_visibility_per_box)):
        if hasattr(tracks, 'numpy'):
            tracks = tracks.numpy()
        if hasattr(vis, 'numpy'):
            vis = vis.numpy()

        disp = np.linalg.norm(tracks[1:] - tracks[:-1], axis=2)  # [T-1, N]
        vis_mask = (
            (vis[:-1] >= DEFAULT_TRACK_VISIBILITY_THRESHOLD)
            & (vis[1:] >= DEFAULT_TRACK_VISIBILITY_THRESHOLD)
        )                                                        # [T-1, N]

        for t in range(T - 1):
            valid = vis_mask[t]
            velocities[k, t] = float(np.median(disp[t, valid])) * fps if valid.any() else 0.0

        if T - 1 >= window:
            kernel = np.ones(window) / window
            velocities[k] = np.convolve(velocities[k], kernel, mode='same')

    return velocities




def select_best_obj_box_tracking(
    candidate_boxes,
    candidate_target_boxes,
    frames_torch,
    pred_tracks_per_box,
    pred_visibility_per_box,
    grasp_phases=None,
    robot_tracks=None,
    target_box_confidences=None,
    movement_weight=0.35,
    target_match_weight=0.35,
    temporal_alignment_weight=0.15,
    intermediate_match_weight=0.15,
    min_movement_threshold=0.001,
    min_visibility_ratio=0.1,
    robot_iou_threshold=0.85,
    intermediate_detections=None,
    det_frame_idx=0,
    scoring_mode="full",
    filter_robot=False,
    fps=1.0,
    track_coordinate_image_shape=None,
    tracking_frame_indices=None,
):
    """
    Selects the best object box by matching initial boxes to target boxes using tracking.

    The algorithm scores each (start_box, target_box) pair based on:
    1. How many tracked points from start_box end up in target_box
    2. How much movement was observed in those tracks
    3. Temporal alignment: movement concentrated during interact phases (if grasp_phases provided)
    4. Robot arm filtering: excludes points that follow robot arm trajectory

    Args:
        candidate_boxes: List of initial bounding boxes (Detections or xyxy arrays)
        candidate_target_boxes: List of target bounding boxes (xyxy format)
        frames_torch: Tensor of video frames [1, T, C, H, W]
        pred_tracks_per_box: List of track tensors per box [T, N_points, 2]
        pred_visibility_per_box: List of visibility tensors per box [T, N_points]
        grasp_phases: List of phase dicts with keys: start_frame, end_frame, phase_type
                      Phase types: "grasp", "interact", "release", "grasp_failure"
                      If no interact phase exists, returns None (no valid interaction detected)
        robot_tracks: Optional tensor of robot arm tracks [T, N_robot_points, 2] for filtering
        target_box_confidences: Optional array of confidence scores for each target box (0-1)
                                Used for tie-breaking when multiple targets have similar scores
        movement_weight: Weight for movement score component (0-1)
        target_match_weight: Weight for target matching score component (0-1)
        temporal_alignment_weight: Weight for temporal alignment score (0-1)
        min_movement_threshold: Minimum movement to consider a track as "moving"
        min_visibility_ratio: Minimum fraction of frames a point must be visible
        robot_iou_threshold: IoU threshold to filter points that overlap with robot trajectory

    Returns:
        best_start_box_idx: Index of the best starting box (None if no interact phase)
        best_target_box_idx: Index of the best matching target box (None if no interact phase)
        score_matrix: np.ndarray of shape (n_start_boxes, n_target_boxes) with scores
        metadata: Dict with additional analysis info including temporal scores
    """
    # --- Apply scoring_mode overrides ---
    if scoring_mode == "simple":
        movement_weight = 0.5
        target_match_weight = 0.5
        temporal_alignment_weight = 0.0
        intermediate_match_weight = 0.0

    metadata = {
        'has_interact_phase': False,
        'interact_phases': [],
        'robot_filtered_points': 0,
        'per_box_temporal_scores': [],
    }

    if len(candidate_boxes) == 0 or len(candidate_target_boxes) == 0:
        return None, None, None, metadata

    # --- Check for interact phase ---
    interact_phases = []
    if grasp_phases is not None:
        for phase in grasp_phases:
            phase_type = phase.get("phase_type", phase.get("phase", ""))
            if phase_type == "interact":
                interact_phases.append({
                    'start': phase.get("start_frame", 0),
                    'end': phase.get("end_frame", frames_torch.shape[1] - 1)
                })

        # If no interact phase detected, return None - no valid interaction
        if len(interact_phases) == 0 and scoring_mode != "simple":
            metadata['has_interact_phase'] = False
            return None, None, None, metadata

    metadata['has_interact_phase'] = len(interact_phases) > 0
    metadata['interact_phases'] = interact_phases

    # Get candidate boxes as numpy array
    if hasattr(candidate_boxes, 'xyxy'):
        start_boxes = candidate_boxes.xyxy
    elif isinstance(candidate_boxes, list) and hasattr(candidate_boxes[0], 'xyxy'):
        start_boxes = candidate_boxes[0].xyxy
    else:
        start_boxes = np.array(candidate_boxes)

    target_boxes = np.array(candidate_target_boxes) if not isinstance(candidate_target_boxes, np.ndarray) else candidate_target_boxes

    n_start_boxes = len(start_boxes)
    n_target_boxes = len(target_boxes)
    n_frames = frames_torch.shape[1]
    if tracking_frame_indices is None:
        tracking_timeline = np.arange(n_frames, dtype=np.int32) + int(det_frame_idx)
    else:
        tracking_timeline = np.asarray(tracking_frame_indices, dtype=np.int32).reshape(-1)
        if len(tracking_timeline) != n_frames:
            raise ValueError(
                "tracking_frame_indices must match the tracking tensor: "
                f"indices={len(tracking_timeline)}, frames={n_frames}"
            )

    if len(pred_tracks_per_box) != n_start_boxes or len(pred_visibility_per_box) != n_start_boxes:
        raise ValueError(
            "Tracking output count must match candidate boxes: "
            f"boxes={n_start_boxes}, tracks={len(pred_tracks_per_box)}, "
            f"visibility={len(pred_visibility_per_box)}"
        )

    candidate_slices, point_counts_per_box = _candidate_point_slices(pred_tracks_per_box)
    for box_idx, (tracks, visibility) in enumerate(
        zip(pred_tracks_per_box, pred_visibility_per_box)
    ):
        if tracks.shape[:2] != visibility.shape:
            raise ValueError(
                f"Track/visibility shape mismatch for candidate {box_idx}: "
                f"tracks={tuple(tracks.shape)}, visibility={tuple(visibility.shape)}"
            )

    pred_tracks = torch.cat(pred_tracks_per_box, dim=1)[None]
    pred_visibility = torch.cat(pred_visibility_per_box, dim=1)[None]

    if track_coordinate_image_shape is None:
        h, w = frames_torch.shape[3:]
    else:
        h, w = (int(track_coordinate_image_shape[0]), int(track_coordinate_image_shape[1]))
    pred_tracks_normalized = pred_tracks / torch.tensor([w, h]).float()[None, None, None, :].to(pred_tracks.device)

    # Visibility is a probability for AllTracker. Using `.bool()` would treat
    # every non-zero value as visible, so threshold it explicitly first.
    visibility_mask = pred_visibility[0] >= DEFAULT_TRACK_VISIBILITY_THRESHOLD
    valid_idx = visibility_mask.sum(0) > min_visibility_ratio * pred_tracks.shape[1]

    # --- Robot arm filtering ---
    # Create mask for points that follow robot trajectory
    robot_point_mask = torch.zeros(pred_tracks.shape[2], dtype=torch.bool)
    if filter_robot and robot_tracks is not None and robot_tracks.shape[1] > 0:
        # robot_tracks: [T, N_robot_points, 2]
        robot_centroid = robot_tracks.mean(dim=1)  # [T, 2]
        robot_radius = torch.norm(robot_tracks - robot_centroid[:, None, :], dim=2).max(dim=1)[0]  # [T]

        # For each tracked point, check if it follows robot trajectory
        for point_idx in range(pred_tracks.shape[2]):
            point_traj = pred_tracks[0, :, point_idx, :]  # [T, 2]
            # Calculate distance to robot centroid over time
            dist_to_robot = torch.norm(point_traj - robot_centroid, dim=1)  # [T]
            # If point is consistently close to robot (>50% of frames), mark as robot
            frames_near_robot = (dist_to_robot < robot_radius * 1.5).float().mean()
            if frames_near_robot > robot_iou_threshold:
                robot_point_mask[point_idx] = True

        metadata['robot_filtered_points'] = robot_point_mask.sum().item()

    # Combine validity: must be visible AND not robot
    valid_idx = valid_idx & ~robot_point_mask

    # --- Create interact phase frame mask ---
    interact_frame_mask = torch.zeros(n_frames, dtype=torch.bool)
    if interact_phases:
        for phase in interact_phases:
            start_f = max(0, phase['start'])
            end_f = min(n_frames, phase['end'] + 1)
            interact_frame_mask[start_f:end_f] = True
    else:
        # If no phases provided, consider all frames as potential interact
        interact_frame_mask[:] = True

    n_interact_frames = interact_frame_mask.sum().item()

    # Initialize score matrices
    score_matrix = np.zeros((n_start_boxes, n_target_boxes))
    movement_matrix = np.zeros((n_start_boxes, n_target_boxes))
    point_count_matrix = np.zeros((n_start_boxes, n_target_boxes))
    temporal_score_matrix = np.zeros((n_start_boxes, n_target_boxes))

    all_point_counts = []
    all_temporal_scores = []
    raw_movement_per_box = []

    # Pre-compute per-frame velocities (median, FPS-normalised, smoothed) for all boxes
    velocities = compute_per_frame_motion_scores(pred_tracks_per_box, pred_visibility_per_box, fps)

    # Store per-frame movement for each box (for continuous movement penalty)
    per_frame_movement_per_box = []

    # First pass: collect all scores for normalization
    for start_idx in range(n_start_boxes):
        point_slice = candidate_slices[start_idx]

        cur_tracks = pred_tracks_normalized[0][:, point_slice]
        cur_tracks_raw = pred_tracks[0][:, point_slice]
        cur_valid = valid_idx[point_slice]

        # Filter to valid points (visible and not robot)
        cur_tracks = cur_tracks[:, cur_valid]
        cur_tracks_raw = cur_tracks_raw[:, cur_valid]

        if cur_tracks.shape[1] == 0:
            raw_movement_per_box.append(None)
            per_frame_movement_per_box.append(None)
            continue

        # Calculate per-frame displacement (kept for temporal alignment scoring below)
        consecutive_displacement = torch.norm(cur_tracks[1:] - cur_tracks[:-1], dim=2)  # [T-1, N_points]
        mean_movement_per_point = consecutive_displacement.mean(dim=0)  # [N_points]

        # Use pre-computed median+FPS+smoothed velocities for movement bookkeeping
        box_velocities = velocities[start_idx] if start_idx < len(velocities) else np.zeros(max(0, n_frames - 1))
        raw_movement_per_box.append(float(box_velocities.mean()) if len(box_velocities) > 0 else 0.0)
        per_frame_movement_per_box.append(box_velocities)

        # --- Temporal alignment: movement during interact phases ---
        if n_interact_frames > 1 and interact_phases:
            # Movement during interact phases
            interact_displacement = consecutive_displacement[interact_frame_mask[1:], :]  # [N_interact-1, N_points]
            non_interact_displacement = consecutive_displacement[~interact_frame_mask[1:], :]  # [N_non_interact-1, N_points]

            movement_during_interact = interact_displacement.mean().item() if interact_displacement.numel() > 0 else 0
            movement_outside_interact = non_interact_displacement.mean().item() if non_interact_displacement.numel() > 0 else 0

            # Temporal alignment score: ratio of movement during interact vs total
            total_movement = movement_during_interact + movement_outside_interact
            if total_movement > min_movement_threshold:
                temporal_alignment = movement_during_interact / total_movement
            else:
                temporal_alignment = 0.0
        else:
            temporal_alignment = 1.0  # No phase info, assume all movement is valid

        # Get last frame positions
        last_frame_points = cur_tracks_raw[-1]

        for target_idx in range(n_target_boxes):
            tx1, ty1, tx2, ty2 = target_boxes[target_idx]

            # Check which points end up in this target box
            in_target = (
                (last_frame_points[:, 0] >= tx1) &
                (last_frame_points[:, 0] <= tx2) &
                (last_frame_points[:, 1] >= ty1) &
                (last_frame_points[:, 1] <= ty2)
            )

            n_points_in_target = in_target.float().sum().item()

            # Calculate movement only for points that end up in target
            if n_points_in_target > 0:
                movement_of_matched = mean_movement_per_point[in_target].mean().item()

                # Temporal alignment for matched points specifically
                if n_interact_frames > 1 and interact_phases:
                    matched_interact_disp = consecutive_displacement[interact_frame_mask[1:], :][:, in_target]
                    matched_non_interact_disp = consecutive_displacement[~interact_frame_mask[1:], :][:, in_target]

                    matched_interact_mov = matched_interact_disp.mean().item() if matched_interact_disp.numel() > 0 else 0
                    matched_non_interact_mov = matched_non_interact_disp.mean().item() if matched_non_interact_disp.numel() > 0 else 0

                    matched_total = matched_interact_mov + matched_non_interact_mov
                    if matched_total > min_movement_threshold:
                        matched_temporal_alignment = matched_interact_mov / matched_total
                    else:
                        matched_temporal_alignment = 0.0
                else:
                    matched_temporal_alignment = 1.0
            else:
                movement_of_matched = 0.0
                matched_temporal_alignment = 0.0

            point_count_matrix[start_idx, target_idx] = n_points_in_target
            movement_matrix[start_idx, target_idx] = movement_of_matched
            temporal_score_matrix[start_idx, target_idx] = matched_temporal_alignment

            all_point_counts.append(n_points_in_target)
            all_temporal_scores.append(matched_temporal_alignment)

        metadata['per_box_temporal_scores'].append({
            'box_idx': start_idx,
            'overall_temporal_alignment': temporal_alignment,
            'n_valid_points': cur_tracks.shape[1]
        })

    # --- Intermediate detection alignment scoring ---
    intermediate_score_per_box = np.zeros(n_start_boxes)
    intermediate_tracked_boxes_per_box = {}  # {start_idx: {rel_frame_idx: [x1,y1,x2,y2]}}
    intermediate_box_diagnostics = {}

    if intermediate_detections and len(intermediate_detections) > 0:
        for start_idx in range(n_start_boxes):
            frame_scores = []
            tracked_boxes_this_box = {}
            diagnostics_this_box = {}

            for kf_rel, dets in intermediate_detections.items():
                # Detections use source-window indices while the tracking tensor
                # may be FPS-subsampled. Use the nearest explicit timeline entry.
                track_frame = int(np.argmin(np.abs(tracking_timeline - int(kf_rel))))
                tracked_frame_index = int(tracking_timeline[track_frame])

                tracked_box, box_diagnostics = robust_tracked_box(
                    start_boxes[start_idx],
                    pred_tracks_per_box[start_idx],
                    pred_visibility_per_box[start_idx],
                    track_frame,
                    (h, w),
                )
                box_diagnostics["requested_frame_index"] = int(kf_rel)
                diagnostics_this_box[tracked_frame_index] = box_diagnostics
                if tracked_box is None:
                    continue
                tracked_boxes_this_box[tracked_frame_index] = tracked_box

                # Compute best IoU * confidence against detections at this frame
                best_score = 0.0
                for det in dets:
                    iou = _compute_iou(tracked_box, det["box"])
                    score = iou * det["confidence"]
                    if score > best_score:
                        best_score = score
                frame_scores.append(best_score)

            if frame_scores:
                intermediate_score_per_box[start_idx] = np.mean(frame_scores)
                intermediate_tracked_boxes_per_box[start_idx] = tracked_boxes_this_box
            intermediate_box_diagnostics[start_idx] = diagnostics_this_box
    else:
        # No intermediate detections — set weight to 0 and redistribute
        intermediate_match_weight = 0.0
        # Redistribute to other weights proportionally
        total_other = movement_weight + target_match_weight + temporal_alignment_weight
        if total_other > 0:
            scale = 1.0 / total_other
            movement_weight *= scale
            target_match_weight *= scale
            temporal_alignment_weight *= scale

    # Normalize scores
    valid_raw_movement = [m for m in raw_movement_per_box if m is not None]
    scene_movement_baseline = float(np.median(valid_raw_movement)) if len(valid_raw_movement) >= 3 else 0.0
    max_points = max(all_point_counts) if all_point_counts and max(all_point_counts) > 0 else 1
    selection_movement_scores = _log_baseline_subtracted_movement(
        movement_matrix, scene_movement_baseline
    )

    if max_points == 0:
        max_points = 1

    # --- Filter out cases where no significant movement was detected ---
    # If max movement across all boxes is below threshold, the object likely wasn't detected

    # --- Compute target box sizes for tie-breaking (smaller is better) ---
    target_box_areas = np.array([(tb[2] - tb[0]) * (tb[3] - tb[1]) for tb in target_boxes])
    max_target_area = target_box_areas.max() if target_box_areas.max() > 0 else 1
    # Normalized inverse area: smaller boxes get higher scores (0-1 range)
    target_size_scores = 1.0 - (target_box_areas / max_target_area)

    # Calculate combined scores
    for start_idx in range(n_start_boxes):
        box_mvt = raw_movement_per_box[start_idx] if start_idx < len(raw_movement_per_box) and raw_movement_per_box[start_idx] is not None else 0.0
        target_match_gate = _compute_baseline_aware_movement_gate(box_mvt, scene_movement_baseline)

        for target_idx in range(n_target_boxes):
            norm_point_count = point_count_matrix[start_idx, target_idx] / max_points
            norm_movement = selection_movement_scores[start_idx, target_idx]
            temporal_score = temporal_score_matrix[start_idx, target_idx]

            # If this specific pair has no movement, give it zero score
            if movement_matrix[start_idx, target_idx] < min_movement_threshold * 0.5:
                score_matrix[start_idx, target_idx] = 0.0
                continue

            # If no points ended up in target, give zero score
            if point_count_matrix[start_idx, target_idx] < 1:
                score_matrix[start_idx, target_idx] = 0.0
                continue

            # Combined score with weights; target_match gated by movement to prevent
            # static objects from trivially matching the target
            score_matrix[start_idx, target_idx] = (
                target_match_weight * norm_point_count * target_match_gate +
                movement_weight * norm_movement +
                temporal_alignment_weight * temporal_score +
                intermediate_match_weight * intermediate_score_per_box[start_idx]
            )

            # Bonus: if many points moved during interact phase AND ended in target
            if (point_count_matrix[start_idx, target_idx] > 2 and
                movement_matrix[start_idx, target_idx] > min_movement_threshold and
                temporal_score_matrix[start_idx, target_idx] > 0.5):
                score_matrix[start_idx, target_idx] *= 1.3

    # --- Nested box penalty: penalize boxes that are inside other boxes with similar movement ---
    # This helps avoid selecting a "part" of an object when the full object box is available
    nested_penalty_matrix = None
    if scoring_mode != "simple":
        nested_penalty_matrix, nested_info = compute_nested_box_penalty(
            start_boxes,
            point_count_matrix,
            movement_matrix,
            similarity_threshold=0.5,  # Point counts must be >50% similar
            penalty_factor=0.15
        )
        score_matrix = score_matrix * (1.0 - nested_penalty_matrix)
        metadata['nested_box_analysis'] = nested_info

    # --- Continuous movement penalty: penalize boxes that move in every frame (robot-like) ---
    # This helps filter out robot arm detections that weren't caught by other methods
    continuous_penalty_per_box = None
    if scoring_mode != "simple" and filter_robot:
        continuous_penalty_per_box, continuous_movement_info = compute_continuous_movement_penalty(
            n_start_boxes,
            n_target_boxes,
            per_frame_movement_per_box,
            grasp_phases,
            n_frames,
            min_movement_threshold=min_movement_threshold,
            continuous_movement_ratio_threshold=2.0,
            penalty_factor=1.0,
        )
        for start_idx in range(n_start_boxes):
            if continuous_penalty_per_box[start_idx] > 0:
                score_matrix[start_idx, :] *= (1.0 - continuous_penalty_per_box[start_idx])
        metadata['continuous_movement_analysis'] = continuous_movement_info

    # Find best match
    if score_matrix.max() == 0:
        metadata['rejection_reason'] = 'all_scores_zero'
        return None, None, score_matrix, metadata

    # --- Tie-breaking: when multiple targets have similar scores, prefer smaller boxes ---
    best_score = score_matrix.max()
    tie_threshold = 0.3  # Consider scores within 30% of the maximum as ties

    # Find all (start, target) pairs that are within tie threshold
    tie_candidates = np.argwhere(score_matrix >= best_score * (1 - tie_threshold))

    # Prepare target box confidences for tie-breaking
    if target_box_confidences is not None:
        target_confidences = np.array(target_box_confidences)
        if len(target_confidences) != n_target_boxes:
            # Fallback if lengths don't match
            target_confidences = np.ones(n_target_boxes)
    else:
        target_confidences = np.ones(n_target_boxes)  # Default: all equal confidence

    if len(tie_candidates) > 1:
        # Multiple candidates - use size and confidence as tie-breaker
        best_combined_score = -1
        best_start_box_idx = None
        best_target_box_idx = None

        for start_idx, target_idx in tie_candidates:
            base_score = score_matrix[start_idx, target_idx]
            size_bonus = target_size_scores[target_idx] * 0.1  # Small bonus for smaller boxes

            # Confidence bonus: higher confidence target boxes are preferred
            confidence_bonus = target_confidences[target_idx] * 0.08  # Small bonus for higher confidence

            # Point density: points per unit area (favor boxes where points are concentrated)
            target_area = target_box_areas[target_idx]
            if target_area > 0:
                point_density = point_count_matrix[start_idx, target_idx] / (target_area / max_target_area)
            else:
                point_density = 0
            density_bonus = min(point_density / max_points, 1.0) * 0.05  # Small density bonus

            combined_score = base_score + size_bonus + confidence_bonus + density_bonus

            if combined_score > best_combined_score:
                best_combined_score = combined_score
                best_start_box_idx = start_idx
                best_target_box_idx = target_idx

        metadata['tie_breaking_applied'] = True
        metadata['n_tie_candidates'] = len(tie_candidates)
    else:
        best_idx = np.unravel_index(np.argmax(score_matrix), score_matrix.shape)
        best_start_box_idx = best_idx[0]
        best_target_box_idx = best_idx[1]
        metadata['tie_breaking_applied'] = False

    # Final validation: ensure best match has meaningful movement and points
    best_movement = movement_matrix[best_start_box_idx, best_target_box_idx]
    best_point_count = point_count_matrix[best_start_box_idx, best_target_box_idx]

    if best_movement < min_movement_threshold:
        metadata['rejection_reason'] = 'best_match_insufficient_movement'
        metadata['best_movement'] = best_movement
    elif best_point_count < 2:
        metadata['rejection_reason'] = 'best_match_insufficient_points'
        metadata['best_point_count'] = best_point_count

    metadata['score_breakdown'] = {
        'point_counts': point_count_matrix,
        'raw_movements': movement_matrix,
        'movements': movement_matrix,
        'confidence_movement_scores': selection_movement_scores,
        'selection_movement_scores': selection_movement_scores,
        'temporal_scores': temporal_score_matrix,
        'intermediate_scores': intermediate_score_per_box,
        'nested_penalties': nested_penalty_matrix,
        'continuous_movement_penalties': continuous_penalty_per_box,
        'final_scores': score_matrix,
        'scene_movement_baseline': scene_movement_baseline,
        'n_points_per_box': (
            point_counts_per_box[0]
            if len(set(point_counts_per_box)) == 1
            else None
        ),
        'point_counts_per_box': point_counts_per_box,
    }
    metadata['intermediate_box_diagnostics'] = intermediate_box_diagnostics
    metadata['raw_movement_per_box'] = raw_movement_per_box
    metadata['scene_movement_baseline'] = scene_movement_baseline
    metadata['best_movement'] = best_movement
    metadata['best_point_count'] = best_point_count
    # Start with detection-aligned tracked boxes (grasp phase end frames)
    merged_intermediate = dict(intermediate_tracked_boxes_per_box.get(best_start_box_idx, {}))

    # Uniformly sample N intermediate boxes from the AllTracker point tracks
    # so that even single-phase trajectories have multiple intermediate keyframes.
    _N_UNIFORM = 6
    best_tracks_raw = pred_tracks_per_box[best_start_box_idx]
    best_vis = pred_visibility_per_box[best_start_box_idx]
    T = best_tracks_raw.shape[0]
    if T > 2 and best_tracks_raw.shape[1] > 0:
        # Sample frames excluding first (0) and last (T-1) — those are start/end keyframes
        uni_t = np.linspace(0, T - 1, _N_UNIFORM + 2, dtype=int)[1:-1]
        for t_idx in uni_t:
            kf_rel = int(tracking_timeline[t_idx])
            if kf_rel not in merged_intermediate:
                tracked_box, box_diagnostics = robust_tracked_box(
                    start_boxes[best_start_box_idx],
                    best_tracks_raw,
                    best_vis,
                    t_idx,
                    (h, w),
                )
                metadata['intermediate_box_diagnostics'].setdefault(
                    best_start_box_idx, {}
                )[kf_rel] = box_diagnostics
                if tracked_box is not None:
                    merged_intermediate[kf_rel] = tracked_box

    metadata['intermediate_tracked_boxes'] = merged_intermediate

    return best_start_box_idx, best_target_box_idx, score_matrix, metadata
