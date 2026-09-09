import numpy as np
import pytest

from sparc.pipeline.trajectory_annotator import TrajectoryAnnotator


class _FakePointCloudProxy:
    def __init__(self):
        self.calls = []

    def get_pointcloud_batch(self, frames, fov_x=None, resolution_level=9):
        self.calls.append(
            {
                "frame_ids": frames[:, 0, 0, 0].tolist(),
                "fov_x": fov_x,
                "resolution_level": resolution_level,
            }
        )
        batch_size, height, width = frames.shape[:3]
        points = np.empty((batch_size, height, width, 3), dtype=np.float32)
        masks = np.ones((batch_size, height, width), dtype=bool)
        xs, ys = np.meshgrid(
            np.arange(width, dtype=np.float32),
            np.arange(height, dtype=np.float32),
        )
        for batch_idx, frame in enumerate(frames):
            frame_id = float(frame[0, 0, 0])
            points[batch_idx] = np.stack(
                [xs, ys, np.full_like(xs, frame_id)],
                axis=-1,
            )
            if frame_id == 1:
                masks[batch_idx, 1, 2] = False
        return {"points": points, "mask": masks}


def _annotator_with_pointcloud():
    annotator = TrajectoryAnnotator.__new__(TrajectoryAnnotator)
    annotator.pointcloud = _FakePointCloudProxy()
    annotator.pointcloud_max_resolution = None
    annotator.pointcloud_batch_size = 2
    annotator.pointcloud_resolution_level = 2
    annotator.fov_x = 60.0
    return annotator


def test_multiple_track_groups_share_each_moge_batch():
    annotator = _annotator_with_pointcloud()
    frames = np.stack(
        [np.full((4, 5, 3), frame_idx, dtype=np.uint8) for frame_idx in range(3)]
    )
    object_tracks = np.asarray(
        [
            [
                [[1, 2], [4, 3]],
                [[2, 1], [0, 0]],
                [[3, 2], [1, 1]],
            ],
            [
                [[0, 1], [2, 3]],
                [[4, 2], [1, 3]],
                [[2, 0], [4, 1]],
            ],
        ],
        dtype=np.float32,
    )
    gripper_tracks = np.asarray(
        [
            [
                [[0, 0], [2, 2], [4, 3]],
                [[1, 1], [3, 2], [4, 0]],
                [[2, 3], [3, 1], [0, 2]],
            ]
        ],
        dtype=np.float32,
    )

    object_3d, gripper_3d = annotator._lift_track_groups_to_3d(
        [object_tracks, gripper_tracks],
        frames,
    )

    assert len(annotator.pointcloud.calls) == 2
    assert annotator.pointcloud.calls == [
        {"frame_ids": [0, 1], "fov_x": 60.0, "resolution_level": 2},
        {"frame_ids": [2], "fov_x": 60.0, "resolution_level": 2},
    ]
    assert object_3d.shape == (2, 3, 2, 3)
    assert gripper_3d.shape == (1, 3, 3, 3)
    np.testing.assert_array_equal(object_3d[0, 0, 0], [1, 2, 0])
    np.testing.assert_array_equal(object_3d[1, 2, 1], [4, 1, 2])
    np.testing.assert_array_equal(gripper_3d[0, 2, 0], [2, 3, 2])
    assert np.isnan(object_3d[0, 1, 0]).all()


def test_track_group_frame_count_is_validated_before_moge_call():
    annotator = _annotator_with_pointcloud()
    frames = np.zeros((3, 4, 5, 3), dtype=np.uint8)
    tracks = np.zeros((1, 2, 4, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="same frame count"):
        annotator._lift_track_groups_to_3d([tracks], frames)

    assert annotator.pointcloud.calls == []
