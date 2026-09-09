from pathlib import Path

import pandas as pd

from sparc.data.dataset_loaders import AgiBotWorldDatasetLoader


def test_extracts_source_identity_from_both_agibot_releases():
    beta = AgiBotWorldDatasetLoader._source_identity_from_episode_row(
        pd.Series({"episode_index": 0, "agibot_episode_ident": 688896})
    )
    world_2026 = AgiBotWorldDatasetLoader._source_identity_from_episode_row(
        pd.Series({
            "episode_index": 0,
            "uuid": "task_3400/source/chunk-000/episode_000000",
        })
    )

    assert beta == {"source_uuid": None, "source_episode_id": "688896"}
    assert world_2026 == {
        "source_uuid": "task_3400/source/chunk-000/episode_000000",
        "source_episode_id": None,
    }


def test_loaded_subtrajectory_retains_original_episode_coordinates(tmp_path):
    loader = AgiBotWorldDatasetLoader.__new__(AgiBotWorldDatasetLoader)
    loader.root_path = tmp_path
    loader.key = "agibot_world_task_454"
    loader.view_key = "observation.images.head"
    loader._subtask_meta = {
        "0/0/3": {
            "chunk_idx": 0,
            "ep_idx": 0,
            "pair_idx": 3,
            "start_frame": 140,
            "end_frame": 570,
            "lang_ann": "pick up the cup",
            "interaction_phases": [],
            "longtask_id": "agibot#agibot_world_task_454#0",
            "longtask_step": 3,
            "longtask_len": 5,
            "longtask_goal": "put away the cup",
        }
    }
    loader._episode_video_meta = {
        0: {
            "video_chunk_idx": 0,
            "video_file_idx": 4,
            "source_uuid": None,
            "source_episode_id": "688896",
        }
    }
    loader.get_split_type = lambda _: "train"

    trajectory = loader.load_trajectory(
        Path("0/0/3"), skip_frames=True, skip_observations=True
    )

    assert trajectory.source_uuid is None
    assert trajectory.camera_view_key == "observation.images.head"
    assert trajectory.source_episode_id == "688896"
    assert trajectory.source_episode_index == 0
    assert trajectory.source_subtrajectory_index == 3
    assert trajectory.source_episode_frame_start == 140
    assert trajectory.source_episode_frame_end_exclusive == 570


def test_old_agibot_cache_is_hydrated_from_episode_metadata(tmp_path):
    episode_dir = tmp_path / "meta" / "episodes" / "chunk-000"
    episode_dir.mkdir(parents=True)
    pd.DataFrame({
        "episode_index": [0],
        "agibot_episode_ident": [688896],
    }).to_parquet(episode_dir / "file-000.parquet")

    loader = AgiBotWorldDatasetLoader.__new__(AgiBotWorldDatasetLoader)
    loader.root_path = tmp_path
    loader._episode_video_meta = {0: {"video_chunk_idx": 0, "video_file_idx": 4}}

    assert loader._hydrate_source_episode_metadata()
    assert loader._episode_video_meta[0]["source_episode_id"] == "688896"
    assert loader._episode_video_meta[0]["source_uuid"] is None
