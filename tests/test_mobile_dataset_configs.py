import pytest

from sparc.data.mobile_dataset_loaders import (
    GalaxeaOpenWorldDatasetLoader,
    RoboCOINDatasetLoader,
)
from sparc.pipeline.config import get_loader_class, load_annotator_config


@pytest.mark.parametrize(
    ("config_name", "dataset_type", "fps", "loader_class"),
    (
        ("galaxea", "galaxea", 15.0, GalaxeaOpenWorldDatasetLoader),
        ("robocoin", "robocoin", 30.0, RoboCOINDatasetLoader),
    ),
)
def test_mobile_dataset_configs_resolve_loader(
    config_name,
    dataset_type,
    fps,
    loader_class,
):
    cfg = load_annotator_config(dataset=config_name, overrides=["dataset.root=/datasets/test"])

    assert cfg.dataset.name == config_name
    assert cfg.dataset.type == dataset_type
    assert cfg.dataset.fps == fps
    assert get_loader_class(dataset_type) is loader_class
    assert cfg.annotator.output_file.endswith(f"/{config_name}_boxes.jsonl")


def test_galaxea_detection_matches_released_gt_frame():
    cfg = load_annotator_config(dataset="galaxea", overrides=["dataset.root=/datasets/test"])

    assert cfg.annotator.detection_frame_policy == "subtask_start"
