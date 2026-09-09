from types import SimpleNamespace

import numpy as np
import pytest
import supervision as sv

import sparc.pipeline.trajectory_annotator as trajectory_annotator_module
from sparc.perception.ann_utils import get_interacted_obj_boxes
from sparc.pipeline.trajectory_annotator import TrajectoryAnnotator


def _detection(box, confidence, class_id):
    return sv.Detections(
        xyxy=np.asarray([box], dtype=np.float32),
        confidence=np.asarray([confidence], dtype=np.float32),
        class_id=np.asarray([class_id], dtype=int),
    )


class _FakeDetectionModel:
    def __init__(self):
        self.prompts = []

    def detect_objects(self, frames, prompts, **kwargs):
        self.prompts.append(list(prompts))
        if len(prompts) == 1:
            detection = _detection([2, 2, 10, 10], 0.8, 0)
        else:
            detection = _detection([20, 20, 28, 28], 0.9, 1)
        return [detection for _ in frames]




@pytest.fixture
def task_info():
    return {
        "object": "cup",
        "start_location": None,
        "target_location": "table",
        "action": "pick and place",
    }


def test_robotseg_path_skips_robot_detection_pass(task_info):
    model = _FakeDetectionModel()
    frames = np.zeros((2, 32, 32, 3), dtype=np.uint8)

    objects, robots, robot_candidates = get_interacted_obj_boxes(
        model,
        frames,
        task_info,
        return_robot_det=True,
        return_all_robot_dets=True,
        skip_robot_detection=True,
    )

    assert model.prompts == [["cup"]]
    assert len(objects) == len(frames)
    assert all(len(detection) == 1 for detection in objects)
    assert robots == [None, None]
    assert robot_candidates == [None, None]


def test_robotseg_disabled_preserves_robot_detection_pass(task_info):
    model = _FakeDetectionModel()
    frames = np.zeros((2, 32, 32, 3), dtype=np.uint8)

    objects, robots, robot_candidates = get_interacted_obj_boxes(
        model,
        frames,
        task_info,
        return_robot_det=True,
        return_all_robot_dets=True,
    )

    assert model.prompts == [
        ["cup"],
        ["robot arm", "robotic gripper", "cup"],
    ]
    assert all(len(detection) == 1 for detection in objects)
    assert all(detection is not None for detection in robots)
    assert all(detection is not None for detection in robot_candidates)


@pytest.mark.parametrize(("robotseg", "expected_skip"), ((object(), True), (None, False)))
def test_detect_objects_enables_bypass_only_with_robotseg(
    monkeypatch,
    task_info,
    robotseg,
    expected_skip,
):
    skip_values = []

    def fake_get_interacted_obj_boxes(
        detection_model,
        frames,
        current_task_info,
        **kwargs,
    ):
        skip_values.append(kwargs.get("skip_robot_detection", False))
        detections = [
            _detection([2, 2, 10, 10], 0.8, 2) for _ in range(len(frames))
        ]
        if kwargs.get("return_robot_det"):
            empty_robot_detections = [None] * len(frames)
            return detections, empty_robot_detections, empty_robot_detections.copy()
        return detections

    monkeypatch.setattr(
        trajectory_annotator_module,
        "get_interacted_obj_boxes",
        fake_get_interacted_obj_boxes,
    )
    annotator = TrajectoryAnnotator.__new__(TrajectoryAnnotator)
    annotator.robotseg = robotseg
    annotator.detection_model = object()
    annotator.det_threshold = 0.1
    annotator.nms_threshold = 0.6
    annotator.movement_task_types = ["pick"]

    frames = np.zeros((3, 32, 32, 3), dtype=np.uint8)
    result = annotator._detect_objects(frames, task_info, [0, 1, 2])

    assert skip_values == [expected_skip]
    assert result.gripper_box is None


def test_robotseg_boxes_replace_detector_gripper_metadata():
    det_result = SimpleNamespace(
        gripper_box=None,
        gripper_box_confidence=None,
        robot_candidates_all=[object()],
        robot_det_dict={0: {"box": [0, 0, 1, 1], "confidence": 0.2}},
    )
    masks = np.zeros((2, 16, 16), dtype=np.uint8)
    masks[0, 4:10, 3:9] = 1
    masks[1, 5:12, 6:13] = 1

    TrajectoryAnnotator._set_robotseg_gripper_detection(
        det_result,
        np.asarray([[3, 4, 9, 10]], dtype=np.float32),
        {"masks": masks, "frame_indices": np.asarray([1, 4])},
        det_frame_idx=1,
    )

    np.testing.assert_array_equal(
        det_result.gripper_box,
        np.asarray([[3, 4, 9, 10]], dtype=np.float32),
    )
    assert det_result.gripper_box_confidence == 1.0
    assert det_result.robot_candidates_all == []
    assert set(det_result.robot_det_dict) == {1, 4}
    assert all(
        detection["confidence"] == 1.0
        for detection in det_result.robot_det_dict.values()
    )


def test_robotseg_failure_restores_detector_gripper(monkeypatch, task_info):
    calls = []
    gripper_detection = _detection([7, 8, 12, 14], 0.75, 1)

    def fake_get_interacted_obj_boxes(*args, **kwargs):
        calls.append(kwargs)
        return [None, None], [None, gripper_detection], [None, gripper_detection]

    monkeypatch.setattr(
        trajectory_annotator_module,
        "get_interacted_obj_boxes",
        fake_get_interacted_obj_boxes,
    )
    annotator = TrajectoryAnnotator.__new__(TrajectoryAnnotator)
    annotator.detection_model = object()
    annotator.det_threshold = 0.1
    annotator.nms_threshold = 0.6
    det_result = SimpleNamespace(
        keyframe_indices=[0, 1],
        is_tool_task=False,
        tool_name=None,
        robot_candidates_all=[],
        robot_det_dict={},
        gripper_box=None,
        gripper_box_confidence=None,
    )

    annotator._restore_detector_gripper_fallback(
        np.zeros((2, 32, 32, 3), dtype=np.uint8),
        task_info,
        det_result,
    )

    assert len(calls) == 1
    assert calls[0].get("skip_robot_detection", False) is False
    np.testing.assert_array_equal(
        det_result.gripper_box,
        np.asarray([[7, 8, 12, 14]], dtype=np.float32),
    )
    assert det_result.gripper_box_confidence == pytest.approx(0.75)
