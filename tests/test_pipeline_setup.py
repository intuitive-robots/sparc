from pathlib import Path

import pytest
from omegaconf import OmegaConf

from sparc.pipeline.config import get_loader_class, load_annotator_config


DATASET_CONFIGS = sorted(
    path.stem for path in (Path(__file__).resolve().parents[1] / "configs/dataset").glob("*.yaml")
)


@pytest.mark.parametrize("dataset", DATASET_CONFIGS)
def test_dataset_presets_resolve_the_supported_pipeline(dataset):
    cfg = load_annotator_config(dataset=dataset, overrides=["dataset.root=/datasets/test"])
    assert cfg.annotator.detection_model == "llmdet"
    assert cfg.annotator.tracking_model == "alltracker"
    assert get_loader_class(cfg.dataset.type) is not None


@pytest.mark.parametrize("overrides", [
    ["annotator.detection_model=owlv2"],
    ["annotator.tracking_model=cotracker"],
    ["annotator.tracking_model=vggt"],
    ["+annotator.tracking_mode=hybrid"],
    ["+annotator.gripper_detection_model=qwen3"],
    ["+annotator.detectors.qwen3.max_workers=8"],
    ["+annotator.n_cotracker_instances=3"],
])
def test_retired_backends_fail_during_config_loading(overrides):
    with pytest.raises(ValueError):
        load_annotator_config(dataset="agibot_world", overrides=["dataset.root=/datasets/test", *overrides])


def test_resolved_config_cannot_silently_select_a_retired_mode(tmp_path):
    cfg = load_annotator_config(dataset="agibot_world", overrides=["dataset.root=/datasets/test"])
    OmegaConf.update(cfg, "annotator.tracking_mode", "detect_only", force_add=True)
    path = tmp_path / "resolved.yaml"
    OmegaConf.save(cfg, path)

    with pytest.raises(ValueError, match="tracking_mode was removed"):
        load_annotator_config(config_path=str(path))


def test_tracker_instance_override_is_preserved():
    cfg = load_annotator_config(overrides=["dataset.root=/datasets/test", "annotator.n_tracker_instances=3"])
    assert cfg.annotator.n_tracker_instances == 3


@pytest.mark.parametrize("dataset", DATASET_CONFIGS)
def test_dataset_root_is_required(dataset):
    with pytest.raises(ValueError, match="Set dataset.root"):
        load_annotator_config(dataset=dataset)


@pytest.mark.parametrize("dataset", DATASET_CONFIGS)
def test_dataset_paths_follow_the_supplied_root(dataset, tmp_path):
    cfg = load_annotator_config(dataset=dataset, overrides=[f"dataset.root={tmp_path}"])
    for path in (cfg.dataset.task_obj_dict_path, cfg.annotator.output_file, cfg.debug.debug_image_dir):
        assert Path(path).is_relative_to(tmp_path)
