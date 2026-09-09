from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf

from sparc.pipeline.config import load_annotator_config
from sparc.pipeline.debug_images import DebugImageSaver


def test_shipped_debug_limit_caps_each_trajectory(tmp_path):
    cfg = load_annotator_config(dataset="agibot_world", overrides=[
        f"dataset.root={tmp_path}", "debug=verbose", "debug.debug_image_freq=1",
        "debug.debug_max_per_traj=2",
    ])
    saver = DebugImageSaver(OmegaConf.to_container(cfg.debug), gpu_id=0)
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    box = np.array([[1, 2, 8, 9]])
    for name in ("first", "second"):
        trajectory = SimpleNamespace(name=name, dataset_name="test")
        for subtask in range(3):
            saver.maybe_save(trajectory, subtask, frame, frame, box, box,
                             {"language_instruction": "move object"})
    assert dict(saver._debug_saves_per_traj) == {"first": 2, "second": 2}
    assert len(list(saver.debug_image_dir.rglob("*.png"))) == 4


def test_debug_metadata_serializes_nested_numpy_values(tmp_path):
    import json

    saver = DebugImageSaver({'save_debug_images': True, 'debug_image_dir': tmp_path}, 0)
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    box = np.array([[1, 2, 8, 9]])
    annotation = {
        'language_instruction': 'move object',
        'initial_object_box': box,
        'nested': {'score': np.float32(0.5), 'valid': np.bool_(True), 'index': np.int64(2)},
    }
    saver.maybe_save(SimpleNamespace(name='episode', dataset_name='test'),
                     0, frame, frame, box, box, annotation)
    path, = saver.debug_image_dir.rglob('*.json')
    result = json.loads(path.read_text())
    assert result['initial_object_box'] == [[1, 2, 8, 9]]
    assert result['nested'] == {'score': 0.5, 'valid': True, 'index': 2}
