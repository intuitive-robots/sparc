"""Prepare the pinned RobotSeg inference source and its local checkpoint."""

import argparse
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "https://github.com/showlab/RobotSeg.git"
REVISION = "dafb8c0d507276e2f96d2b07ac3661a7b3a41a5f"
CHECKPOINT_PAGE = "https://github.com/showlab/RobotSeg#72-download"



def remove_upstream_binary(destination: Path):
    """Discard upstream's x86/CUDA binary while preserving a local rebuild."""
    binary = destination / "robotseg" / "_C.so"
    if not binary.is_file():
        return
    upstream_hash = subprocess.check_output(
        ["git", "-C", str(destination), "rev-parse", "HEAD:robotseg/_C.so"], text=True,
    ).strip()
    local_hash = subprocess.check_output(
        ["git", "-C", str(destination), "hash-object", str(binary.resolve())], text=True,
    ).strip()
    if local_hash == upstream_hash:
        binary.unlink()


def prepare_source(destination: Path):
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--no-checkout", REPOSITORY, str(destination)], check=True)
        subprocess.run(["git", "-C", str(destination), "checkout", "--detach", REVISION], check=True)
    revision = subprocess.check_output(
        ["git", "-C", str(destination), "rev-parse", "HEAD"], text=True,
    ).strip()
    if revision != REVISION:
        raise ValueError(f"Expected RobotSeg revision {REVISION} at {destination}, found {revision}")
    remove_upstream_binary(destination)
    patch = ROOT / "setup" / "robotseg_numpy.patch"
    command = ["git", "-C", str(destination), "apply"]
    if subprocess.run(command + ["--reverse", "--check", str(patch)], capture_output=True).returncode == 0:
        return
    subprocess.run(command + ["--check", str(patch)], check=True)
    subprocess.run(command + [str(patch)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="Downloaded robotseg.pt from the upstream checkpoint page")
    args = parser.parse_args()
    checkpoint = ROOT / "checkpoints" / "robotseg" / "robotseg.pt"
    source_checkpoint = args.checkpoint or checkpoint
    if not source_checkpoint.is_file():
        parser.error(f"Download robotseg.pt from {CHECKPOINT_PAGE} and pass --checkpoint /path/to/robotseg.pt")
    prepare_source(ROOT / "vendors" / "robotseg" / "RobotSeg")
    if source_checkpoint.resolve() != checkpoint.resolve():
        if checkpoint.exists():
            parser.error(f"Checkpoint already exists at {checkpoint}; it was not overwritten")
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_checkpoint, checkpoint)
    print(f"RobotSeg source prepared; checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
