import pytest

from sparc.pipeline.config import load_annotator_config


@pytest.mark.parametrize(
    ("config_name", "dataset_name", "view_key"),
    (
        (
            "droid_lerobot",
            "droid_lerobot",
            "observation.images.left_external",
        ),
        (
            "droid_lerobot_right",
            "droid_lerobot_right",
            "observation.images.right_external",
        ),
    ),
)
def test_droid_camera_configs_resolve_score_ref_defaults(
    config_name,
    dataset_name,
    view_key,
):
    cfg = load_annotator_config(dataset=config_name, overrides=["dataset.root=/datasets/test"])

    assert cfg.dataset.name == dataset_name
    assert cfg.dataset.type == "droid_lerobot"
    assert cfg.dataset.loader_kwargs.view_key == view_key
    assert cfg.annotator.output_file.endswith(f"/{dataset_name}_boxes.jsonl")
    assert cfg.annotator.enable_crop is True
    assert cfg.annotator.tracking_model == "alltracker"
    assert cfg.annotator.detection_model == "llmdet"
    assert cfg.annotator.enable_robotseg is True
    assert cfg.annotator.scoring_mode == "full"
    assert cfg.annotator.nms_threshold == 0.6
    assert cfg.annotator.tracking_target_fps == 6.0
    assert cfg.annotator.tracking_max_frames == 800
    assert cfg.annotator.pointcloud_resolution_level == 2


def test_task_object_parsing_defaults_to_legacy_with_semantic_settings_available():
    cfg = load_annotator_config(dataset="droid_lerobot_right", overrides=["dataset.root=/datasets/test"])

    assert cfg.extract_task_objects.mode == "text"
    assert cfg.extract_task_objects.shuffle_seed is None
    assert cfg.extract_task_objects.semantic.max_views == 2
    assert cfg.extract_task_objects.semantic.max_pending_requests == 1024
