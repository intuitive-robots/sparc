import os
from pathlib import Path
import shutil
import subprocess


SCRIPT = Path(__file__).resolve().parents[1] / "sparc/detectors/sam2/checkpoints/download_ckpts.sh"


def _download(tmp_path, fail=False):
    directory = tmp_path / "checkpoints"
    directory.mkdir()
    script = directory / SCRIPT.name
    shutil.copyfile(SCRIPT, script)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    curl = bin_dir / "curl"
    curl.write_text(
        '#!/bin/bash\nprintf "%s\\n" "$@" >> "$DOWNLOAD_LOG"\n'
        'while [ "$1" != --output ]; do shift; done\n'
        'printf checkpoint > "$2"\n'
        + ('exit 1\n' if fail else '')
    )
    curl.chmod(0o755)
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", DOWNLOAD_LOG=str(tmp_path / "download.log"))
    return script, env


def test_downloads_only_small_into_script_directory_and_reuses_it(tmp_path):
    script, env = _download(tmp_path)
    subprocess.run(["bash", str(script)], cwd=tmp_path, env=env, check=True)
    log = Path(env["DOWNLOAD_LOG"]).read_text()
    assert log.count("https://") == 1
    assert log.endswith("/sam2.1_hiera_small.pt\n")
    assert (script.parent / "sam2.1_hiera_small.pt").read_text() == "checkpoint"
    subprocess.run(["bash", str(script)], cwd=tmp_path, env=env, check=True)
    assert Path(env["DOWNLOAD_LOG"]).read_text() == log


def test_failed_download_leaves_no_checkpoint_or_partial_file(tmp_path):
    script, env = _download(tmp_path, fail=True)
    result = subprocess.run(["bash", str(script)], cwd=tmp_path, env=env)
    assert result.returncode != 0
    assert list(script.parent.glob("*.pt*")) == []
