"""Bridge between starVLA training configs and the SPARC annotation pipeline.

starVLA defines the dataset to train on in its training YAML under
``datasets.vla_data`` as ``data_root_dir`` + ``data_mix`` (a named mixture, e.g.
``libero_mixed``). The mixture registry lives in the starVLA repo at
``starVLA/dataloader/gr00t_lerobot/mixtures.py`` as ``DATASET_NAMED_MIXTURES``,
mapping a mix name to ``[(data_name, sampling_weight, robot_type), ...]`` where
each LeRobot dataset is at ``data_root_dir/data_name``.

This module reads that config so the annotation pipeline can annotate exactly the
datasets that will be trained on — the starVLA YAML stays the single source of
truth. ``sampling_weight`` / ``robot_type`` drive training transforms only and are
irrelevant to annotation, so they are ignored here.
"""

import importlib.util
import logging
import os
from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf


def resolve_starvla_root(explicit: Optional[str] = None) -> Path:
    """Locate the starVLA repo: explicit arg > $STARVLA_ROOT > sibling ../starVLA."""
    candidates = [
        explicit,
        os.environ.get("STARVLA_ROOT"),
        Path(__file__).resolve().parents[2].parent / "starVLA",
    ]
    for cand in candidates:
        if not cand:
            continue
        root = Path(cand)
        if (root / "starVLA" / "dataloader" / "gr00t_lerobot" / "mixtures.py").exists():
            return root
    raise FileNotFoundError(
        "Could not locate the starVLA repo. Pass --starvla-root or set STARVLA_ROOT "
        "to the repo root (the directory containing starVLA/dataloader/...)."
    )


def _load_named_mixtures(starvla_root: Path) -> dict:
    """Import DATASET_NAMED_MIXTURES directly from mixtures.py (no package import)."""
    mixtures_py = starvla_root / "starVLA" / "dataloader" / "gr00t_lerobot" / "mixtures.py"
    spec = importlib.util.spec_from_file_location("_starvla_mixtures", mixtures_py)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DATASET_NAMED_MIXTURES


def load_vla_data_cfg(train_yaml: str):
    """Return the ``datasets.vla_data`` block of a starVLA training YAML."""
    cfg = OmegaConf.load(train_yaml)
    vla = cfg.get("datasets", {}).get("vla_data") if hasattr(cfg, "get") else None
    if vla is None:
        raise KeyError(f"{train_yaml} has no datasets.vla_data block")
    return vla


def resolve_mixture(data_mix: str, starvla_root: Optional[str] = None) -> list[tuple]:
    """Return the mixture spec ``[(data_name, weight, robot_type), ...]``."""
    root = resolve_starvla_root(starvla_root)
    mixtures = _load_named_mixtures(root)
    if data_mix not in mixtures:
        raise KeyError(
            f"data_mix '{data_mix}' not in starVLA DATASET_NAMED_MIXTURES "
            f"({len(mixtures)} entries). Checked {root}."
        )
    return list(mixtures[data_mix])


def subdataset_names(mixture: list[tuple]) -> list[str]:
    """data_name folders for a mixture, de-duplicated, order preserved."""
    seen, names = set(), []
    for entry in mixture:
        name = entry[0]
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


def resolve_subdatasets(data_mix: str, starvla_root: Optional[str] = None) -> list[str]:
    """Convenience: data_mix name -> list of LeRobot subdataset folder names."""
    names = subdataset_names(resolve_mixture(data_mix, starvla_root))
    logging.info("starVLA mixture '%s' -> subdatasets %s", data_mix, names)
    return names
