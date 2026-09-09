from sparc.pipeline.trajectory_annotator import OBJ_DET_IDX, TrajectoryAnnotator


def _detection_frame(policy):
    annotator = TrajectoryAnnotator.__new__(TrajectoryAnnotator)
    annotator.detection_frame_policy = policy
    task = {
        "grasp_phases": [
            {"phase_type": "grasp", "start_frame": 30, "end_frame": 50},
            {"phase_type": "release", "start_frame": 80, "end_frame": 90},
        ]
    }
    _, _, keyframes = annotator._build_subtask_phases(
        task,
        grasp_phases=[],
        grasp_phases_with_offset=[],
        start_frame=10,
        end_frame=99,
        num_frames=90,
    )
    return keyframes[OBJ_DET_IDX]


def test_pregrasp_midpoint_policy_preserves_existing_behavior():
    assert _detection_frame("pregrasp_midpoint") == 20


def test_subtask_start_policy_uses_first_frame():
    assert _detection_frame("subtask_start") == 0
