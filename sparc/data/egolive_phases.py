"""Single-object interaction phases from EgoLive's per-frame hand-object boxes.

EgoLive annotates, per frame and per hand, the box of the object that hand is
interacting with. Three properties of that annotation drive everything here:

* it flickers — raw contact runs have a median length of 3 frames at 60 fps, so
  the mask needs morphological closing before it describes anything physical;
* ``track_id`` does not persist (median run also ~3 frames), so object identity
  has to be re-established from box overlap rather than trusted;
* two hands never share a ``track_id``. One object held by both hands is
  signalled *only* by the ``both_obj`` slot, which is what makes bimanual
  phases detectable at all.

The result is a list of phases in the shape the pipeline already consumes for
AgiBot (``phase_type`` / ``start_frame`` / ``end_frame``, trajectory-relative),
with the hand and the object track attached.
"""

from typing import Optional

import numpy as np

# Column offset of each slot inside the (T, 30) hand_object_info feature.
SLOT_OFFSETS = {"left_hand": 0, "right_hand": 6, "left_obj": 12, "right_obj": 18, "both_obj": 24}
HAND_SLOTS = {"left": "left_obj", "right": "right_obj", "both": "both_obj"}


def slot_boxes(info: np.ndarray, slot: str) -> np.ndarray:
    """(T, 4) boxes for one slot, NaN where the slot is not annotated."""
    offset = SLOT_OFFSETS[slot]
    boxes = np.asarray(info, dtype=np.float32)[:, offset: offset + 4].copy()
    boxes[(boxes < 0).all(axis=1)] = np.nan
    return boxes


def close_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    """Bridge dropouts shorter than ``max_gap``, then drop blips of the same length."""
    if max_gap <= 1:
        return mask.copy()
    closed = mask.copy()
    for fill_value in (False, True):
        start = 0
        for i in range(1, len(closed) + 1):
            if i == len(closed) or closed[i] != closed[i - 1]:
                if closed[start] == fill_value and (i - start) < max_gap:
                    closed[start:i] = not fill_value
                start = i
    return closed


def contact_mask(info: np.ndarray, hand: str, gap_frames: int = 0) -> np.ndarray:
    """Frames where ``hand`` is holding something, alone or together with the other hand."""
    if hand == "both":
        mask = np.isfinite(slot_boxes(info, "both_obj")[:, 0])
    else:
        mask = (np.isfinite(slot_boxes(info, HAND_SLOTS[hand])[:, 0])
                | np.isfinite(slot_boxes(info, "both_obj")[:, 0]))
    return close_gaps(mask, gap_frames)


def _mask_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open [start, end) spans of True."""
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[0::2].tolist(), edges[1::2].tolist()))


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return 0.0
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    areas = ((a[2] - a[0]) * (a[3] - a[1])) + ((b[2] - b[0]) * (b[3] - b[1]))
    union = areas - intersection
    return float(intersection / union) if union > 0 else 0.0


def split_on_object_change(boxes: np.ndarray, span: tuple[int, int], min_iou: float) -> list[tuple[int, int]]:
    """Cut a contact span wherever the box jumps to a different object.

    ``track_id`` cannot do this job, so identity is carried by overlap with the
    most recent annotated box instead.
    """
    start, end = span
    cuts, previous = [start], None
    for i in range(start, end):
        box = boxes[i]
        if not np.isfinite(box).all():
            continue
        if previous is not None and box_iou(boxes[previous], box) < min_iou:
            cuts.append(i)
        previous = i
    cuts.append(end)
    return [(a, b) for a, b in zip(cuts[:-1], cuts[1:]) if b > a]


def interaction_phases(
    info: np.ndarray,
    fps: float,
    gap_seconds: float = 0.25,
    min_seconds: float = 0.3,
    min_iou: float = 0.1,
    hands: tuple[str, ...] = ("left", "right", "both"),
) -> list[dict]:
    """Single-object interaction phases, one per (hand, object) episode of contact.

    ``left``/``right`` phases include frames where that hand shares the object
    with the other one; the ``both`` phases are exactly those shared stretches,
    so a bimanual grasp appears in all three lists and can be filtered either way.
    """
    info = np.asarray(info, dtype=np.float32)
    if info.ndim != 2 or len(info) == 0:
        return []

    gap_frames = max(1, int(round(gap_seconds * fps)))
    min_frames = max(1, int(round(min_seconds * fps)))

    phases = []
    for hand in hands:
        boxes = slot_boxes(info, HAND_SLOTS[hand])
        shared = slot_boxes(info, "both_obj")
        # A one-handed phase may be annotated in the hand's own slot, the shared
        # slot, or alternate between them; fall back to the shared box.
        if hand != "both":
            fallback = ~np.isfinite(boxes[:, 0]) & np.isfinite(shared[:, 0])
            boxes = boxes.copy()
            boxes[fallback] = shared[fallback]

        for span in _mask_runs(contact_mask(info, hand, gap_frames)):
            for start, end in split_on_object_change(boxes, span, min_iou):
                if end - start < min_frames:
                    continue
                present = np.flatnonzero(np.isfinite(boxes[start:end, 0])) + start
                if len(present) == 0:
                    continue
                phases.append({
                    "phase_type": "interact",
                    "hand": hand,
                    "start_frame": int(start),
                    "end_frame": int(end),
                    "duration_seconds": float((end - start) / fps),
                    "object_box_first": boxes[present[0]].tolist(),
                    "object_box_last": boxes[present[-1]].tolist(),
                    "box_coverage": float(len(present) / (end - start)),
                })
    phases.sort(key=lambda phase: (phase["start_frame"], phase["hand"]))
    return phases


def annotate_phase_quality(
    phases: list[dict],
    info: np.ndarray,
    fps: float,
    min_open_seconds: float = 0.3,
    min_box_coverage: float = 0.6,
    max_hand_iou: float = 0.7,
) -> list[dict]:
    """Mark which phases are a single, clearly delimited interaction.

    A phase is only usable as an interaction example if you can see the hand
    take the object: contact that is already underway when the clip starts has
    no observable onset, and a hand that simply holds something throughout is
    better treated as no interaction at all. Objects that are essentially the
    hand box, or that the annotation loses track of, are dropped for the same
    reason — there is nothing distinct to ground.
    """
    info = np.asarray(info, dtype=np.float32)
    min_open = max(1, int(round(min_open_seconds * fps)))
    n_frames = len(info)

    by_hand: dict[str, list[dict]] = {}
    for phase in sorted(phases, key=lambda p: p["start_frame"]):
        by_hand.setdefault(phase["hand"], []).append(phase)

    for hand, hand_phases in by_hand.items():
        hand_slot = "right_hand" if hand == "both" else f"{hand}_hand"
        hand_boxes = slot_boxes(info, hand_slot)
        object_boxes = slot_boxes(info, HAND_SLOTS[hand])

        for index, phase in enumerate(hand_phases):
            start, end = phase["start_frame"], phase["end_frame"]
            previous_end = hand_phases[index - 1]["end_frame"] if index else 0
            next_start = hand_phases[index + 1]["start_frame"] if index + 1 < len(hand_phases) else n_frames

            overlaps = [
                box_iou(object_boxes[t], hand_boxes[t])
                for t in range(start, min(end, n_frames))
                if np.isfinite(object_boxes[t]).all() and np.isfinite(hand_boxes[t]).all()
            ]
            hand_iou = float(np.median(overlaps)) if overlaps else 1.0

            phase["open_before"] = int(start - previous_end)
            phase["open_after"] = int(next_start - end)
            phase["hand_object_iou"] = hand_iou
            phase["is_clean"] = bool(
                phase["open_before"] >= min_open
                and phase["box_coverage"] >= min_box_coverage
                and hand_iou <= max_hand_iou
            )
    return phases


def mark_bimanual_pairs(phases: list[dict], min_overlap_frames: int = 1,
                        max_object_iou: float = 0.3) -> list[dict]:
    """Label how each clean phase pairs up across hands.

    ``shared`` is one object held by both hands (EgoLive's ``both_obj`` slot);
    ``two_object`` is the genuinely bimanual case of each hand on its own,
    distinct object; ``single`` is one hand acting alone.
    """
    for phase in phases:
        phase.setdefault("bimanual_role", "single")

    shared = [p for p in phases if p["hand"] == "both"]
    left = [p for p in phases if p["hand"] == "left"]
    right = [p for p in phases if p["hand"] == "right"]

    for phase in shared:
        phase["bimanual_role"] = "shared"

    for left_phase in left:
        for right_phase in right:
            overlap = (min(left_phase["end_frame"], right_phase["end_frame"])
                       - max(left_phase["start_frame"], right_phase["start_frame"]))
            if overlap < min_overlap_frames:
                continue
            # Both hands busy at once on boxes that do not coincide: two objects.
            if box_iou(np.asarray(left_phase["object_box_first"], dtype=np.float32),
                       np.asarray(right_phase["object_box_first"], dtype=np.float32)) <= max_object_iou:
                for phase in (left_phase, right_phase):
                    if phase["bimanual_role"] == "single":
                        phase["bimanual_role"] = "two_object"
    return phases


def rescale_phases(
    phases: list[dict],
    stride: int,
    scale: float,
    n_output_frames: Optional[int] = None,
) -> list[dict]:
    """Move phase frame indices onto a decimated clip and boxes onto its resolution."""
    rescaled = []
    for phase in phases:
        start = phase["start_frame"] // stride
        end = -(-phase["end_frame"] // stride)  # ceil, so short phases keep a frame
        if n_output_frames is not None:
            start, end = min(start, n_output_frames), min(end, n_output_frames)
        if end <= start:
            continue
        updated = dict(phase, start_frame=int(start), end_frame=int(end))
        for key in ("object_box_first", "object_box_last"):
            updated[key] = (np.asarray(phase[key], dtype=np.float32) * scale).tolist()
        rescaled.append(updated)
    return rescaled




def gripper_state_from_phases(phases: list[dict], hand: str, n_frames: int) -> np.ndarray:
    """Pipeline-convention gripper signal (1 = open, 0 = closed) for one hand.

    Built from the phase list rather than the raw contact mask so the scalar and
    the phases can never disagree — the phases additionally drop blips too short
    to be a real grasp.

    This is a far stronger stand-in for a gripper than any fingertip-aperture
    heuristic, because it reads EgoLive's own hand-object annotation instead of
    trying to infer contact from hand shape.

    Only phases marked clean count as contact. Sustained holds with no visible
    onset are deliberately reported as open: they carry no grasp event, and
    calling them contact is what previously made a busy hand look like an
    unusable full-length interaction.
    """
    state = np.ones(n_frames, dtype=np.float32)
    for phase in phases:
        if phase.get("is_clean", True) and phase["hand"] in (hand, "both"):
            state[phase["start_frame"]: phase["end_frame"]] = 0.0
    return state
