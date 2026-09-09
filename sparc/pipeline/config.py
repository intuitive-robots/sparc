"""Configuration loading for the annotation pipeline.

Uses Hydra's compose API for config composition — same defaults resolution
as @hydra.main but without taking over the process (compatible with
multiprocessing/spawn).

Dataset configs use standard Hydra defaults syntax:
    defaults:
      - /debug: verbose

    dataset:
      name: agibot_world
      ...
"""

import os
import importlib
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf, DictConfig

CONFIGS_DIR = Path(__file__).parent.parent.parent / "configs"

# Dataset type names resolve to loader classes in sparc/data/.
LOADER_MAP = {
    "bridge_lerobot": "sparc.data.dataset_loaders.BridgeLeRobotDatasetLoader",
    "droid_lerobot": "sparc.data.dataset_loaders.DroidLeRobotDatasetLoader",
    "robomind2": "sparc.data.dataset_loaders.RoboMind2DatasetLoader",
    "oxe_lerobot": "sparc.data.dataset_loaders.OXELeRobotDatasetLoader",
    "multi_lerobot": "sparc.data.dataset_loaders.MultiLeRobotDatasetLoader",
    "agibot_world": "sparc.data.dataset_loaders.AgiBotWorldDatasetLoader",
    "agibot_world_multi": "sparc.data.dataset_loaders.AgiBotWorldMultiTaskLoader",
    "vla_arena": "sparc.data.dataset_loaders.VLAArenaDatasetLoader",
    "egolive": "sparc.data.dataset_loaders.EgoLiveDatasetLoader",
    "egolive_multi": "sparc.data.dataset_loaders.EgoLiveMultiDatasetLoader",
    "galaxea": "sparc.data.mobile_dataset_loaders.GalaxeaOpenWorldDatasetLoader",
    "galaxea_task": "sparc.data.mobile_dataset_loaders.GalaxeaTaskDatasetLoader",
    "robocoin": "sparc.data.mobile_dataset_loaders.RoboCOINDatasetLoader",
}


def load_annotator_config(
    config_path: str = None,
    dataset: str = None,
    overrides: list = None,
) -> DictConfig:
    """Load config using Hydra's compose API.

    Uses the same defaults resolution as @hydra.main (including
    defaults lists in dataset configs like ``- /debug: verbose``)
    but without taking over the process.

    Args:
        config_path: Path to a fully resolved YAML (skips Hydra compose).
        dataset: Dataset name — passed as ``dataset={name}`` override to Hydra.
        overrides: Additional Hydra-style overrides (e.g. ``["annotator.gpu_ids=[0]"]``).
    """
    if config_path:
        # Load an explicitly resolved config directly.
        cfg = OmegaConf.load(config_path)
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
        _resolve_config(cfg)
        return cfg

    # Use Hydra compose API for full defaults resolution
    hydra_overrides = list(overrides or [])
    if dataset:
        hydra_overrides.insert(0, f"dataset={dataset}")

    GlobalHydra.instance().clear()
    with initialize_config_dir(
        version_base=None,
        config_dir=str(CONFIGS_DIR.absolute()),
    ):
        cfg = compose(
            config_name="annotation_pipeline",
            overrides=hydra_overrides,
        )

    _resolve_config(cfg)
    return cfg


def load_config_from_starvla(
    train_yaml: str,
    starvla_root: str = None,
    overrides: list = None,
) -> DictConfig:
    """Build an annotator config from a starVLA training YAML.

    Reads ``datasets.vla_data.{data_root_dir, data_mix}`` from the starVLA config,
    resolves the named mixture to its LeRobot subdataset folders, and composes the
    base ``annotation_pipeline.yaml`` with a ``multi_lerobot`` dataset pointing at
    them. Trajectory keys follow starVLA's training convention so the resulting
    annotations line up with starVLA's CoT lookup. The starVLA YAML stays the
    single source of truth for *which* data is annotated.
    """
    from sparc.data.starvla_bridge import load_vla_data_cfg, resolve_subdatasets

    vla = load_vla_data_cfg(train_yaml)
    data_root = vla.get("data_root_dir")
    data_mix = vla.get("data_mix")
    if not data_root or not data_mix:
        raise ValueError(f"{train_yaml}: datasets.vla_data needs data_root_dir and data_mix")
    subdatasets = resolve_subdatasets(data_mix, starvla_root)

    # Compose the base config without any dataset group (~dataset), then attach a
    # multi_lerobot dataset built from the starVLA mixture.
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        version_base=None,
        config_dir=str(CONFIGS_DIR.absolute()),
    ):
        cfg = compose(
            config_name="annotation_pipeline",
            overrides=["~dataset"] + list(overrides or []),
        )

    OmegaConf.update(
        cfg,
        "dataset",
        {
            "name": data_mix,
            "type": "multi_lerobot",
            "root": str(data_root),
            "fps": float(vla.get("fps")) if vla.get("fps") else 20.0,
            "task_obj_dict_path": f"{data_root}/{data_mix}_task_obj.pkl",
            "loader_kwargs": {
                "subdatasets": subdatasets,
                "starvla_trajectory_names": True,
                "config_path": None,
            },
        },
        force_add=True,
    )

    _resolve_config(cfg)
    return cfg


def _resolve_config(cfg: DictConfig):
    if OmegaConf.is_missing(cfg.dataset, "root") or not cfg.dataset.get("root"):
        raise ValueError("Set dataset.root=/path/to/dataset for the selected dataset")
    OmegaConf.resolve(cfg)
    _apply_computed_defaults(cfg)


def _apply_computed_defaults(cfg: DictConfig):
    """Fill in defaults that depend on other config values."""
    ds = cfg.dataset
    ann = cfg.annotator

    for key, supported in (("detection_model", "llmdet"), ("tracking_model", "alltracker")):
        if ann.get(key, supported) != supported:
            raise ValueError(f"annotator.{key} supports only {supported!r}")
    for key in ("tracking_mode", "gripper_detection_model"):
        if key in ann:
            raise ValueError(f"annotator.{key} was removed; delete it from the config")
    if "n_cotracker_instances" in ann:
        raise ValueError("Rename annotator.n_cotracker_instances to annotator.n_tracker_instances")
    unsupported_detectors = set(ann.get("detectors", {})) - {"llmdet"}
    if unsupported_detectors:
        raise ValueError(f"Remove unsupported detector settings: {sorted(unsupported_detectors)}")

    # Output file
    if not ann.get("output_file"):
        if "oxe" in ds.name.lower():
            oxe_ds_name = "_".join(ds.name.split("_")[1:])
            output_file = os.path.join(ds.root, oxe_ds_name, f"{ds.name}_boxes.jsonl")
        else:
            output_file = os.path.join(ds.root, f"{ds.name}_boxes.jsonl")
        OmegaConf.update(cfg, "annotator.output_file", output_file)

    # Debug image dir
    dbg = cfg.get("debug", {})
    if dbg and not dbg.get("debug_image_dir"):
        OmegaConf.update(
            cfg, "debug.debug_image_dir",
            os.path.join(ds.root, "debug_annotations"),
        )


def get_loader_class(dataset_type: str):
    """Map dataset.type string to the corresponding loader class."""
    qualified_name = LOADER_MAP.get(dataset_type)
    if qualified_name is None:
        raise ValueError(
            f"Unknown dataset type '{dataset_type}'. "
            f"Available: {list(LOADER_MAP.keys())}"
        )
    module_path, class_name = qualified_name.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
