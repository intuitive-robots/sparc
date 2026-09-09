import importlib.util
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prepare_robotseg', ROOT / 'setup/prepare_robotseg.py')
prepare_robotseg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare_robotseg)


def test_remove_upstream_binary_preserves_local_rebuild(tmp_path):
    def git(*args):
        subprocess.run(['git', '-C', str(tmp_path), *args], check=True, capture_output=True)

    git('init')
    binary = tmp_path / 'robotseg/_C.so'
    binary.parent.mkdir()
    binary.write_bytes(b'upstream architecture-specific binary')
    git('add', 'robotseg/_C.so')
    git('-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
        'commit', '-m', 'fixture')
    prepare_robotseg.remove_upstream_binary(tmp_path)
    assert not binary.exists()
    prepare_robotseg.remove_upstream_binary(tmp_path)
    binary.write_bytes(b'locally compiled extension')
    prepare_robotseg.remove_upstream_binary(tmp_path)
    assert binary.read_bytes() == b'locally compiled extension'
