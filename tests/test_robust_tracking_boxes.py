import numpy as np
import torch

from sparc.perception.ann_utils import (
    _candidate_point_slices,
    robust_intermediate_boxes,
    robust_tracked_box,
    select_best_obj_box_tracking,
)
from sparc.pipeline.pipeline_types import TrackingResult
from sparc.pipeline.trajectory_annotator import TrajectoryAnnotator


def _grid_points(box, nx=8, ny=8):
    x1, y1, x2, y2 = box
    xs = np.linspace(x1, x2, nx, dtype=np.float32)
    ys = np.linspace(y1, y2, ny, dtype=np.float32)
    return np.stack(np.meshgrid(xs, ys, indexing="xy"), axis=-1).reshape(-1, 2)


def test_candidate_point_slices_preserve_alltracker_512_point_groups():
    tracks = [
        torch.zeros((3, 512, 2), dtype=torch.float32),
        torch.zeros((3, 512, 2), dtype=torch.float32),
        torch.zeros((3, 137, 2), dtype=torch.float32),
    ]

    slices, counts = _candidate_point_slices(tracks)

    assert counts == [512, 512, 137]
    assert slices == [slice(0, 512), slice(512, 1024), slice(1024, 1161)]


def test_live_selector_uses_all_512_points_without_cross_candidate_slicing():
    object_a = np.tile(_grid_points([10, 10, 20, 20]), (8, 1))
    object_b = np.tile(_grid_points([50, 50, 60, 60]), (8, 1))
    tracks_a = np.stack([object_a] * 4).astype(np.float32)
    tracks_b = np.stack([
        object_b,
        object_b + [6, 0],
        object_b + [12, 0],
        object_b + [20, 0],
    ]).astype(np.float32)
    visibility = torch.ones((4, 512), dtype=torch.float32)

    best_start, _, _, metadata = select_best_obj_box_tracking(
        candidate_boxes=np.array([[10, 10, 20, 20], [50, 50, 60, 60]], dtype=np.float32),
        candidate_target_boxes=np.array([[68, 48, 82, 62]], dtype=np.float32),
        frames_torch=torch.zeros((1, 4, 3, 100, 100), dtype=torch.float32),
        pred_tracks_per_box=[torch.from_numpy(tracks_a), torch.from_numpy(tracks_b)],
        pred_visibility_per_box=[visibility, visibility],
        grasp_phases=[{"phase_type": "interact", "start_frame": 0, "end_frame": 3}],
        intermediate_detections={
            17: [{"box": [62, 50, 72, 60], "confidence": 0.9}],
        },
        tracking_frame_indices=np.array([4, 10, 18, 25], dtype=np.int32),
        scoring_mode="simple",
        fps=4.0,
    )

    assert best_start == 1
    assert metadata["score_breakdown"]["n_points_per_box"] == 512
    assert metadata["score_breakdown"]["point_counts_per_box"] == [512, 512]
    assert 18 in metadata["intermediate_tracked_boxes"]


def test_robust_tracked_box_ignores_low_visibility_and_high_confidence_outliers():
    initial_box = np.array([20.0, 30.0, 60.0, 70.0], dtype=np.float32)
    points0 = _grid_points(initial_box)
    points1 = points0 + np.array([12.0, -7.0], dtype=np.float32)

    # A low-confidence prediction must not count as visible merely because its
    # confidence is non-zero. A high-confidence geometric outlier must be
    # rejected by the robust motion fit.
    points1[0] = [500.0, -200.0]
    points1[1] = [450.0, 250.0]
    visibility = np.ones((2, len(points0)), dtype=np.float32)
    visibility[1, 0] = 1e-4

    box, diagnostics = robust_tracked_box(
        initial_box,
        np.stack([points0, points1]),
        visibility,
        frame_index=1,
        image_shape=(240, 320),
    )

    np.testing.assert_allclose(box, [32.0, 23.0, 72.0, 63.0], atol=1.0)
    assert diagnostics["n_inliers"] >= 50
    assert diagnostics["method"] in {"partial_affine_ransac", "median_translation"}


def test_robust_intermediate_boxes_use_explicit_frame_mapping_and_stay_in_bounds():
    initial_box = np.array([10.0, 10.0, 30.0, 30.0], dtype=np.float32)
    points0 = _grid_points(initial_box)
    tracks = np.stack([
        points0,
        points0 + [5.0, 2.0],
        points0 + [10.0, 4.0],
    ]).astype(np.float32)
    visibility = np.ones(tracks.shape[:2], dtype=np.float32)

    boxes, diagnostics = robust_intermediate_boxes(
        initial_box,
        tracks,
        visibility,
        tracking_frame_indices=np.array([4, 10, 18], dtype=np.int32),
        requested_frame_indices=[10, 18],
        image_shape=(64, 80),
    )

    assert sorted(boxes) == [10, 18]
    np.testing.assert_allclose(boxes[10], [15.0, 12.0, 35.0, 32.0], atol=1.0)
    np.testing.assert_allclose(boxes[18], [20.0, 14.0, 40.0, 34.0], atol=1.0)
    assert diagnostics[10]["tracking_frame_index"] == 10


def test_lightweight_override_recomputes_boxes_for_new_winner():
    initial_a = _grid_points([10.0, 10.0, 30.0, 30.0])
    initial_b = _grid_points([70.0, 20.0, 90.0, 40.0])
    tracks = np.stack([
        np.stack([initial_a, initial_a + [2.0, 0.0], initial_a + [4.0, 0.0]]),
        np.stack([initial_b, initial_b + [0.0, 8.0], initial_b + [0.0, 16.0]]),
    ]).astype(np.float32)
    visibility = np.ones(tracks.shape[:-1], dtype=np.float32)
    raw_tracks = {
        "candidate_boxes": np.array([[10, 10, 30, 30], [70, 20, 90, 40]], dtype=np.float32),
        "candidate_confidences": np.array([0.8, 0.7], dtype=np.float32),
        "candidate_masks": np.ones((2, 4, 4), dtype=np.uint8),
        "cotracker_tracks": tracks,
        "cotracker_visibility": visibility,
        "tracking_frame_indices": np.array([4, 10, 18], dtype=np.int32),
        "tracking_coordinate_image_hw": np.array([100, 120], dtype=np.int32),
        "winner_idx": np.array(0, dtype=np.int32),
    }
    result = TrackingResult(
        best_obj_box=np.array([[10, 10, 30, 30]], dtype=np.float32),
        best_obj_box_confidence=0.8,
        best_obj_box_target=None,
        obj_traces=tracks[0],
        verified_target=False,
        intermediate_boxes={10: [12, 10, 32, 30], 18: [14, 10, 34, 30]},
        confidence_breakdown={},
        raw_tracks_data=raw_tracks,
    )
    lightweight = {
        "winner_pick": {
            "best_start_idx": 1,
            "best_target_idx": None,
            "final_score": [0.2, 0.9],
            "signal_name": "test",
        }
    }

    annotator = TrajectoryAnnotator.__new__(TrajectoryAnnotator)
    annotator._apply_selection_winner(result, lightweight, preliminary_winner_idx=0)

    np.testing.assert_allclose(result.best_obj_box, [[70, 20, 90, 40]])
    np.testing.assert_allclose(result.intermediate_boxes[10], [70, 28, 90, 48], atol=1.0)
    np.testing.assert_allclose(result.intermediate_boxes[18], [70, 36, 90, 56], atol=1.0)
    assert int(result.raw_tracks_data["winner_idx"]) == 1
