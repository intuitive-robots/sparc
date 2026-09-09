import io

import numpy as np
from PIL import Image
from sparc.llm.prompts.prompts import PROMPT_USER_TASK_OBJ_SEMANTIC

from sparc.llm.semantic_task_parsing import (
    build_multiview_storyboard_jpeg,
    flatten_storyboard_frame_indices,
    resolve_task_object_parsing_mode,
    semantic_storyboard_samples,
    semantic_episode_window,
    split_manipulation_episodes,
    task_object_cache_key,
    validate_and_merge_semantic_entries,
)


PHASES = [
    (0, 10, "grasp"),
    (11, 20, "interact"),
    (21, 25, "release"),
    (26, 35, "grasp"),
    (36, 45, "interact"),
    (46, 50, "release"),
]


def _entry(sample, relation, instance, object_name="cube"):
    return {
        "object": object_name,
        "object_instance": instance,
        "instance_relation_to_previous": relation,
        "grasp_phases": [
            {
                "start_frame": start,
                "end_frame": end,
                "phase_type": phase_type,
                "description": phase_type,
            }
            for start, end, phase_type in sample["phases"]
        ],
    }


def test_semantic_cache_is_trajectory_specific_and_text_key_is_unchanged():
    assert task_object_cache_key("text", "0/1", "move cubes", "grasp_release") == (
        "move cubes",
        "grasp_release",
    )
    first_semantic_key = task_object_cache_key(
        "semantic", "0/1", "move cubes", "grasp_release"
    )
    assert first_semantic_key[0] == "semantic_v9"
    assert first_semantic_key != task_object_cache_key(
        "semantic", "0/2", "move cubes", "grasp_release"
    )


def test_auto_mode_uses_legacy_for_one_completed_grasp_cycle():
    assert resolve_task_object_parsing_mode("auto", PHASES[:3]) == "text"


def test_auto_mode_uses_semantic_for_multiple_completed_grasp_cycles():
    assert resolve_task_object_parsing_mode("auto", PHASES) == "semantic"


def test_auto_mode_ignores_an_unfinished_second_grasp_cycle():
    assert resolve_task_object_parsing_mode("auto", PHASES[:5]) == "text"


def test_semantic_prompt_sends_only_grouped_episodes():
    prompt = PROMPT_USER_TASK_OBJ_SEMANTIC.format(
        task="move the cubes",
        episodes_json="[]",
    )

    assert "Detected episodes:" in prompt
    assert "Detected phases:" not in prompt


def test_semantic_system_prompt_prioritizes_task_grounding_and_visible_attributes():
    from sparc.llm.prompts.prompts import PROMPT_SYSTEM_TASK_OBJ_SEMANTIC

    assert "Object identity priority: task text and semantic role first" in (
        PROMPT_SYSTEM_TASK_OBJ_SEMANTIC
    )
    assert '"yellow block", "green block"' in PROMPT_SYSTEM_TASK_OBJ_SEMANTIC
    assert "Never select a background object" in PROMPT_SYSTEM_TASK_OBJ_SEMANTIC
    assert "states only a high-level goal" in PROMPT_SYSTEM_TASK_OBJ_SEMANTIC


def test_semantic_system_prompt_distinguishes_task_object_from_tool():
    from sparc.llm.prompts.prompts import PROMPT_SYSTEM_TASK_OBJ_SEMANTIC

    assert "task item being moved or affected" in PROMPT_SYSTEM_TASK_OBJ_SEMANTIC
    assert 'object: "pan", tool_name_required: "towel"' in (
        PROMPT_SYSTEM_TASK_OBJ_SEMANTIC
    )
    assert 'object: "towel", tool_name_required: null' in (
        PROMPT_SYSTEM_TASK_OBJ_SEMANTIC
    )


def test_semantic_system_prompt_requires_imperative_instructions():
    from sparc.llm.prompts.prompts import PROMPT_SYSTEM_TASK_OBJ_SEMANTIC

    assert "short canonical robot action category" in PROMPT_SYSTEM_TASK_OBJ_SEMANTIC
    assert "Do not include object names, attributes, locations" in (
        PROMPT_SYSTEM_TASK_OBJ_SEMANTIC
    )
    assert "imperative phase instruction" in PROMPT_SYSTEM_TASK_OBJ_SEMANTIC
    assert "Use diverse, context-appropriate verbs and phrasing" in (
        PROMPT_SYSTEM_TASK_OBJ_SEMANTIC
    )
    assert 'Do not force descriptions into fixed "Grasp ...", "Move ...", or "Release ..."' in (
        PROMPT_SYSTEM_TASK_OBJ_SEMANTIC
    )
    assert 'Never narrate in third person ("the robot picks up ...")' in (
        PROMPT_SYSTEM_TASK_OBJ_SEMANTIC
    )


def test_storyboard_sampling_preserves_two_release_delimited_episodes():
    episodes = split_manipulation_episodes(PHASES)
    samples = semantic_storyboard_samples(PHASES)

    assert len(episodes) == 2
    assert [sample["frames"] for sample in samples] == [[15], [40]]
    assert flatten_storyboard_frame_indices(samples) == [15, 40]


def test_semantic_entries_only_merge_when_vlm_explicitly_marks_same_instance():
    samples = semantic_storyboard_samples(PHASES)
    separate = validate_and_merge_semantic_entries(
        [
            _entry(samples[0], "first_instance", "instance_0"),
            _entry(samples[1], "different_instance", "instance_1"),
        ],
        samples,
    )
    merged = validate_and_merge_semantic_entries(
        [
            _entry(samples[0], "first_instance", "instance_0"),
            _entry(samples[1], "same_instance_retry", "instance_0"),
        ],
        samples,
    )

    assert len(separate) == 2
    assert len(merged) == 1
    assert len(merged[0]["grasp_phases"]) == 6


def test_semantic_entries_do_not_merge_different_object_classes():
    samples = semantic_storyboard_samples(PHASES)
    entries = validate_and_merge_semantic_entries(
        [
            _entry(samples[0], "first_instance", "instance_1", "drawer"),
            _entry(
                samples[1],
                "same_instance_continuation",
                "instance_0",
                "green object",
            ),
        ],
        samples,
    )

    assert len(entries) == 2
    assert entries[1]["instance_relation_to_previous"] == "uncertain"


def test_semantic_entries_do_not_merge_conflicting_visual_attributes():
    samples = semantic_storyboard_samples(PHASES)
    entries = validate_and_merge_semantic_entries(
        [
            _entry(samples[0], "first_instance", "instance_0", "patterned pillow"),
            _entry(
                samples[1],
                "same_instance_continuation",
                "instance_0",
                "blue pillow",
            ),
        ],
        samples,
    )

    assert len(entries) == 2
    assert entries[1]["instance_relation_to_previous"] == "uncertain"


def test_semantic_entries_merge_generic_and_refined_names_for_same_instance():
    samples = semantic_storyboard_samples(PHASES)
    entries = validate_and_merge_semantic_entries(
        [
            _entry(samples[0], "first_instance", "instance_0", "towel"),
            _entry(
                samples[1],
                "same_instance_continuation",
                "instance_0",
                "yellow towel",
            ),
        ],
        samples,
    )

    assert len(entries) == 1
    assert len(entries[0]["grasp_phases"]) == 6


def test_storyboard_jpeg_has_episode_columns_and_camera_rows():
    samples = semantic_storyboard_samples(PHASES)
    external_frames = np.zeros((2, 90, 160, 3), dtype=np.uint8)
    wrist_frames = np.zeros((2, 90, 160, 3), dtype=np.uint8)
    external_frames[:, :, :, 1] = 120
    wrist_frames[:, :, :, 2] = 180

    encoded = build_multiview_storyboard_jpeg(
        [
            ("observation.images.right_external", external_frames),
            ("observation.images.wrist", wrist_frames),
        ],
        samples,
        cell_width=160,
        cell_height=100,
    )
    image = Image.open(io.BytesIO(encoded))

    assert image.size == (320, 228)
    external_center = image.getpixel((80, 89))
    wrist_center = image.getpixel((80, 189))
    assert external_center[1] > external_center[2]
    assert wrist_center[2] > wrist_center[1]


def test_semantic_subtask_boundaries_use_exact_episode_interval():
    task = {"grasp_phases": _entry(
        semantic_storyboard_samples(PHASES)[1],
        "different_instance",
        "instance_1",
    )["grasp_phases"]}

    start, end = semantic_episode_window(task["grasp_phases"], 51)

    assert (start, end) == (26, 51)


def test_retired_parsing_mode_name_is_rejected():
    import pytest

    with pytest.raises(ValueError, match='expected one of'):
        resolve_task_object_parsing_mode('legacy', PHASES)
