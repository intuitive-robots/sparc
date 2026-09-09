from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from sparc.data.dataset_loaders import LeRobotDatasetLoader, MultiLeRobotDatasetLoader
from sparc.pipeline.config import load_annotator_config


@pytest.mark.parametrize(
    ("config_name", "dataset_name", "fps", "subdataset", "view_key"),
    (
        (
            "libero",
            "libero_mixed",
            10.0,
            "libero",
            "observation.images.image",
        ),
        (
            "libero_plus",
            "libero_plus",
            20.0,
            "libero_plus",
            "observation.images.front",
        ),
    ),
)
def test_libero_configs_use_inline_starvla_subdatasets(
    config_name,
    dataset_name,
    fps,
    subdataset,
    view_key,
):
    cfg = load_annotator_config(dataset=config_name, overrides=["dataset.root=/datasets/test"])

    assert cfg.dataset.name == dataset_name
    assert cfg.dataset.type == "multi_lerobot"
    assert cfg.dataset.fps == fps
    assert cfg.dataset.loader_kwargs.starvla_trajectory_names is True
    assert cfg.dataset.loader_kwargs.subdatasets[subdataset].view_key == view_key
    assert (
        cfg.dataset.loader_kwargs.subdatasets[subdataset].gripper.transform
        == "neg1open_pos1close_to_01"
    )
    assert cfg.extract_task_objects.mode == "auto"
    assert len(cfg.extract_task_objects.semantic.view_keys) == 2


def test_inline_subdataset_specs_override_file_specs():
    loader = MultiLeRobotDatasetLoader.__new__(MultiLeRobotDatasetLoader)
    loader._config = {
        "subdatasets": {
            "libero": {
                "view_key": "old.camera",
                "gripper": {"transform": "identity"},
            }
        }
    }

    specs = loader._resolve_subdataset_specs(
        Path("/unused"),
        {
            "libero": {
                "view_key": "observation.images.image",
                "gripper": {"transform": "neg1open_pos1close_to_01"},
            }
        },
    )

    assert specs == {
        "libero": {
            "view_key": "observation.images.image",
            "gripper": {"transform": "neg1open_pos1close_to_01"},
        }
    }


def test_inline_subdataset_can_be_disabled():
    loader = MultiLeRobotDatasetLoader.__new__(MultiLeRobotDatasetLoader)
    loader._config = {}

    specs = loader._resolve_subdataset_specs(
        Path("/unused"),
        {"libero": {"enabled": False}},
    )

    assert specs == {}


def test_multi_lerobot_delegates_indexed_frame_loading():
    expected = np.zeros((2, 4, 4, 3), dtype=np.uint8)

    class ChildLoader:
        def load_frames_at_indices(self, path, frame_indices, view_key=None):
            assert path == Path("0/7")
            assert frame_indices == [3, 9]
            assert view_key == "observation.images.image2"
            return expected

    loader = MultiLeRobotDatasetLoader.__new__(MultiLeRobotDatasetLoader)
    loader._loaders = {"libero": ChildLoader()}

    actual = loader.load_frames_at_indices(
        Path("libero/0/7"),
        [3, 9],
        view_key="observation.images.image2",
    )

    assert actual is expected


def test_starvla_episode_keys_use_collision_free_logical_chunks():
    loader = LeRobotDatasetLoader.__new__(LeRobotDatasetLoader)
    loader._starvla_trajectory_names = True
    loader._chunk_size = 1000

    assert loader._episode_key(0, 0) == "0/0"
    assert loader._episode_key(0, 999) == "0/999"
    assert loader._episode_key(0, 1000) == "1/0"
    assert loader._episode_key(0, 1692) == "1/692"


def test_lerobot_trajectory_carries_stable_source_identifiers():
    loader = LeRobotDatasetLoader.__new__(LeRobotDatasetLoader)
    loader._episode_meta = {
        1102: {
            "chunk_idx": 0,
            "task_idx": None,
            "source_uuid": "raw/bridge/source/traj0",
        }
    }
    loader._task_map = {}
    loader._fps = 10.0
    loader.view_key = "observation.images.image"
    loader.root_path = Path("/dataset")
    loader.key = "bridge_lerobot"
    loader._resolve_episode = lambda path: (0, 1102, None)
    loader._video_path = lambda chunk_idx, episode_idx: Path("/dataset/video.mp4")
    loader.get_split_type = lambda path: "train"

    trajectory = loader.load_trajectory(
        Path("1/102"),
        skip_frames=True,
        skip_observations=True,
    )

    assert trajectory.source_episode_index == 1102
    assert trajectory.source_uuid == "raw/bridge/source/traj0"


def test_frame_level_task_index_populates_missing_instruction():
    loader = LeRobotDatasetLoader.__new__(LeRobotDatasetLoader)
    loader._task_map = {28: "Put the bowl on the plate"}
    loader._episode_meta = {1000: {"task_idx": None}}
    trajectory = SimpleNamespace(lang_ann="")
    episode = pd.DataFrame({"task_index": [28, 28, 28]})

    loader._populate_task_annotation(trajectory, episode, 1000)

    assert trajectory.lang_ann == "put the bowl on the plate"
    assert loader._episode_meta[1000]["task_idx"] == 28
