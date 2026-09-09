from pathlib import Path

from sparc.perception.tracking_models import ensure_alltracker_repo
from sparc.pipeline.gpu_inference import SAM2_PACKAGE_ROOT, _resolve_sam2_checkpoint


def test_vendored_model_packages_resolve_after_sparc_reorganization():
    assert (SAM2_PACKAGE_ROOT / "sam2" / "sam2_image_predictor.py").is_file()
    assert ensure_alltracker_repo() == (
        Path(__file__).resolve().parents[1] / "sparc" / "detectors" / "alltracker"
    )


def test_sam2_checkpoint_resolves_canonical_and_legacy_locations(tmp_path):
    canonical = (
        tmp_path / "sparc/detectors/sam2/checkpoints/sam2.1_hiera_small.pt"
    )
    canonical.parent.mkdir(parents=True)
    canonical.touch()
    assert _resolve_sam2_checkpoint(tmp_path) == canonical

    canonical.unlink()
    legacy = (
        tmp_path / "annotator/detectors/sam2/checkpoints/sam2.1_hiera_small.pt"
    )
    legacy.parent.mkdir(parents=True)
    legacy.touch()
    assert _resolve_sam2_checkpoint(tmp_path) == legacy
