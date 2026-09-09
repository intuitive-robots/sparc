import numpy as np
import pytest

from sparc.pipeline.config import load_annotator_config
from sparc.pipeline.trajectory_annotator import TrajectoryAnnotator


def test_pointcloud_saving_defaults_off_without_disabling_scoring():
    cfg = load_annotator_config(dataset="droid_lerobot", overrides=["dataset.root=/datasets/test"])
    assert cfg.annotator.save_pointclouds is False
    assert cfg.annotator.enable_pointcloud is True
    cfg = load_annotator_config(dataset="droid_lerobot", overrides=[
        "dataset.root=/datasets/test", "annotator.save_pointclouds=true",
    ])
    assert cfg.annotator.save_pointclouds is True
    assert cfg.annotator.enable_pointcloud is True


@pytest.mark.parametrize("save_pointclouds", [False, True])
def test_diagnostic_npz_only_saves_3d_when_requested(tmp_path, save_pointclouds):
    annotator = TrajectoryAnnotator.__new__(TrajectoryAnnotator)
    annotator.tracks_dir = str(tmp_path)
    annotator.save_pointclouds = save_pointclouds
    raw = {
        "cotracker_tracks": np.ones((1, 2, 3, 2), dtype=np.float32),
        "cotracker_tracks_3d": np.ones((1, 2, 3, 3), dtype=np.float32),
        "gripper_tracks_3d": np.ones((2, 3, 3), dtype=np.float32),
    }
    path = annotator._save_tracks_npz("episode/1", 0, raw)
    assert path is not None
    with np.load(path, allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["cotracker_tracks"], raw["cotracker_tracks"])
        for key in ("cotracker_tracks_3d", "gripper_tracks_3d"):
            assert (key in arrays) == save_pointclouds
            if save_pointclouds:
                np.testing.assert_array_equal(arrays[key], raw[key])
    # Export filtering must not discard the in-memory inputs used by scoring.
    assert "cotracker_tracks_3d" in raw and "gripper_tracks_3d" in raw
