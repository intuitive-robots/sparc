"""Utilities for visually grounded task-object parsing."""

from __future__ import annotations

import io
import re
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw


SEMANTIC_CACHE_VERSION = "semantic_v9"
TASK_OBJECT_PARSING_MODES = {"auto", "text", "semantic"}
_EFFECTIVE_TASK_OBJECT_PARSING_MODES = {"text", "semantic"}


def resolve_task_object_parsing_mode(mode, grasp_phases: Iterable[tuple]) -> str:
    """Resolve ``auto`` from the number of completed grasp/release cycles."""
    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in TASK_OBJECT_PARSING_MODES:
        raise ValueError(
            f"Unknown task-object parsing mode {mode!r}; "
            f"expected one of {sorted(TASK_OBJECT_PARSING_MODES)}"
        )
    if normalized_mode != "auto":
        return normalized_mode

    completed_cycles = sum(
        bool(episode) and episode[-1][2] == "release"
        for episode in split_manipulation_episodes(grasp_phases)
    )
    return "semantic" if completed_cycles > 1 else "text"


def task_object_cache_key(mode, trajectory_name, language_instruction, phase_sequence_key):
    """Return an instruction/phase text key or a trajectory-specific semantic key."""
    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in _EFFECTIVE_TASK_OBJECT_PARSING_MODES:
        raise ValueError(
            f"Task-object parsing mode {mode!r} is not an effective mode; "
            "resolve auto mode before building a cache key"
        )
    text_key = (language_instruction.strip(), phase_sequence_key)
    if normalized_mode == "text":
        return text_key
    return (SEMANTIC_CACHE_VERSION, str(trajectory_name), *text_key)


def split_manipulation_episodes(grasp_phases: Iterable[tuple]) -> list[list[tuple]]:
    """Split a phase stream after each detected release, retaining unfinished final episodes."""
    episodes = []
    current = []
    for phase in grasp_phases:
        normalized = (int(phase[0]), int(phase[1]), str(phase[2]))
        current.append(normalized)
        if normalized[2] == "release":
            episodes.append(current)
            current = []
    if current:
        episodes.append(current)
    return episodes


def semantic_storyboard_samples(grasp_phases: Iterable[tuple]) -> list[dict]:
    """Choose one interaction frame for every release-delimited episode."""
    samples = []
    for episode_index, episode in enumerate(split_manipulation_episodes(grasp_phases)):
        interact = next((phase for phase in episode if phase[2] == "interact"), None)
        interaction_frame = (
            (interact[0] + interact[1]) // 2
            if interact is not None
            else (episode[0][0] + episode[-1][1]) // 2
        )
        samples.append(
            {
                "episode_index": episode_index,
                "phases": episode,
                "frames": [interaction_frame],
                "labels": ["interaction"],
            }
        )
    return samples


def flatten_storyboard_frame_indices(samples: list[dict]) -> list[int]:
    return [frame for sample in samples for frame in sample["frames"]]


def semantic_episode_window(grasp_phases: list[dict], trajectory_length: int) -> tuple[int, int]:
    """Return the exact [start, end) interval covered by a semantic episode."""
    if not grasp_phases:
        raise ValueError("Semantic episode has no grasp phases")
    start = max(0, int(grasp_phases[0].get("start_frame", 0)))
    end = min(
        int(trajectory_length),
        int(grasp_phases[-1].get("end_frame", trajectory_length - 1)) + 1,
    )
    return start, max(start + 1, end)


def _known_object_instance(entry: dict) -> str | None:
    value = str(entry.get("object_instance") or "").strip().lower()
    return None if value in {"", "unknown", "none", "null"} else value


def _object_name_tokens(value) -> set[str]:
    words = re.findall(r"[a-z0-9]+", str(value or "").lower())
    normalized = []
    for word in words:
        if word in {"a", "an", "the"}:
            continue
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        normalized.append(word)
    return set(normalized)


def _same_object_identity(previous: dict, current: dict) -> bool:
    previous_instance = _known_object_instance(previous)
    current_instance = _known_object_instance(current)
    if (
        previous_instance is not None
        and current_instance is not None
        and previous_instance != current_instance
    ):
        return False

    previous_tokens = _object_name_tokens(previous.get("object"))
    current_tokens = _object_name_tokens(current.get("object"))
    if not previous_tokens or not current_tokens:
        return False
    return previous_tokens.issubset(current_tokens) or current_tokens.issubset(previous_tokens)


def validate_and_merge_semantic_entries(entries: list[dict], samples: list[dict]) -> list[dict]:
    """Validate one entry per episode and conservatively merge matching identities."""
    if len(entries) != len(samples):
        raise ValueError(
            f"Semantic parser returned {len(entries)} entries for {len(samples)} candidate episodes"
        )

    validated = []
    for entry, sample in zip(entries, samples):
        expected = [tuple(phase) for phase in sample["phases"]]
        actual = [
            (
                int(phase.get("start_frame")),
                int(phase.get("end_frame")),
                str(phase.get("phase_type")),
            )
            for phase in entry.get("grasp_phases", [])
        ]
        if actual != expected:
            raise ValueError(
                f"Semantic parser changed phases for episode {sample['episode_index']}: "
                f"expected={expected}, actual={actual}"
            )
        validated.append(entry)

    merged = []
    merge_relations = {"same_instance_retry", "same_instance_continuation"}
    for entry in validated:
        relation = entry.get("instance_relation_to_previous")
        should_merge = (
            bool(merged)
            and relation in merge_relations
            and _same_object_identity(merged[-1], entry)
        )
        if should_merge:
            merged[-1]["grasp_phases"].extend(entry["grasp_phases"])
            merged[-1].setdefault("merged_episode_relations", []).append(relation)
            continue
        if merged and relation in merge_relations:
            entry = dict(entry)
            entry["instance_relation_to_previous"] = "uncertain"
        merged.append(entry)
    return merged


def build_multiview_storyboard_jpeg(
    frames_by_view: list[tuple[str, np.ndarray]],
    samples: list[dict],
    cell_width: int = 320,
    cell_height: int = 200,
    jpeg_quality: int = 85,
) -> bytes:
    """Build one image with episode columns and synchronized camera-view rows."""
    if not samples:
        raise ValueError("Cannot build an empty semantic storyboard")
    if not frames_by_view:
        raise ValueError("Cannot build a semantic storyboard without camera views")

    expected_frames = len(samples)
    for view_label, frames in frames_by_view:
        if len(frames) != expected_frames:
            raise ValueError(
                f"Expected {expected_frames} frames for view {view_label!r}, got {len(frames)}"
            )

    header_height = 28
    label_height = 22
    sheet = Image.new(
        "RGB",
        (cell_width * len(samples), header_height + cell_height * len(frames_by_view)),
        color=(20, 20, 20),
    )
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 7), "Columns: episodes | Rows: camera views", fill=(255, 255, 255))

    for row, (view_label, frames) in enumerate(frames_by_view):
        short_view_label = str(view_label).rsplit(".", 1)[-1]
        if "wrist" in short_view_label.lower() or "gripper" in short_view_label.lower():
            display_view_label = "Wrist"
        elif "external" in short_view_label.lower():
            display_view_label = "External"
        else:
            display_view_label = short_view_label
        for column, sample in enumerate(samples):
            frame = np.asarray(frames[column]).astype(np.uint8, copy=False)
            image = Image.fromarray(frame).convert("RGB")
            available_height = cell_height - label_height
            image.thumbnail((cell_width, available_height), Image.Resampling.LANCZOS)
            x0 = column * cell_width
            y0 = header_height + row * cell_height
            paste_x = x0 + (cell_width - image.width) // 2
            paste_y = y0 + label_height + (available_height - image.height) // 2
            sheet.paste(image, (paste_x, paste_y))
            draw.text(
                (x0 + 6, y0 + 4),
                f"E{sample['episode_index']} | {display_view_label}",
                fill=(255, 255, 0),
            )

    output = io.BytesIO()
    sheet.save(output, format="JPEG", quality=int(jpeg_quality), optimize=True)
    return output.getvalue()
