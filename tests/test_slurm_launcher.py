import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def _environment(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    capture = tmp_path / "capture.py"
    capture.write_text(
        "import json, os, sys\n"
        "with open(os.environ['CAPTURE'], 'a') as f:\n"
        " f.write(json.dumps({'args': sys.argv[1:], 'cwd': os.getcwd(), "
        "'devices': os.environ.get('CUDA_VISIBLE_DEVICES')}) + '\\n')\n"
    )
    for command in ("python", "srun"):
        path = bin_dir / command
        path.write_text(f"#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(str(capture))} \"$@\"\n")
        path.chmod(0o755)
    curl = bin_dir / "curl"
    curl.write_text("#!/bin/sh\nexit 0\n")
    curl.chmod(0o755)
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}",
               CAPTURE=str(tmp_path / "calls.jsonl"), SLURM_JOB_ID="test",
               SLURM_SUBMIT_DIR=str(ROOT), CUDA_VISIBLE_DEVICES="2,5",
               VLLM_BASE_URL="http://server:8000/v1", PIPELINE_PYTHON=str(bin_dir / "python"),
               VLLM_HOST_FILE=str(tmp_path / "vlm_host"))
    env.pop("PIPELINE_ROOT", None)
    env.pop("_WORKER", None)
    return env


def test_worker_keeps_allocated_devices_and_quoted_overrides(tmp_path):
    env = _environment(tmp_path)
    env.update(_WORKER="1", SLURM_NODEID="0")
    subprocess.run(["bash", str(ROOT / "slurm/launch_annotation.sh"), "libero",
                    "dataset.root=/data/with spaces"], env=env, check=True)
    call = json.loads(Path(env["CAPTURE"]).read_text())
    assert call["cwd"] == str(ROOT)
    assert call["devices"] == "2,5"
    assert "annotator.gpu_ids=[0,1]" in call["args"]
    assert "dataset.root=/data/with spaces" in call["args"]


def test_spooled_script_dispatches_submission_checkout_and_merges(tmp_path):
    env = _environment(tmp_path)
    env["SLURM_JOB_NUM_NODES"] = "2"
    spool = tmp_path / "slurm_script"
    spool.write_text((ROOT / "slurm/launch_annotation.sh").read_text())
    subprocess.run(["bash", str(spool), "libero", "dataset.root=/data/with spaces"],
                   cwd=tmp_path, env=env, check=True)
    dispatch, merge = [json.loads(line) for line in Path(env["CAPTURE"]).read_text().splitlines()]
    assert str(ROOT / "slurm/launch_annotation.sh") in dispatch["args"]
    assert "dataset.root=/data/with spaces" in dispatch["args"]
    assert "--merge-only" in merge["args"]
    assert "dataset.root=/data/with spaces" in merge["args"]
    assert merge["cwd"] == str(ROOT)


def test_launcher_requires_explicit_server_configuration(tmp_path):
    env = _environment(tmp_path)
    env.pop("VLLM_BASE_URL")
    env.pop("VLLM_LAUNCH_SCRIPT", None)
    result = subprocess.run(["bash", str(ROOT / "slurm/launch_annotation.sh"), "libero"],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 2
    assert "Set VLLM_BASE_URL or VLLM_LAUNCH_SCRIPT" in result.stderr


def test_dispatch_resolves_config_before_changing_checkout(tmp_path):
    env = _environment(tmp_path)
    config = tmp_path / 'resolved.yaml'
    config.write_text('dataset: {}\n')
    subprocess.run(['bash', str(ROOT / 'slurm/launch_annotation.sh'),
                    '--config', 'resolved.yaml'], cwd=tmp_path, env=env, check=True)
    dispatch = json.loads(Path(env['CAPTURE']).read_text())
    assert str(config) in dispatch['args']
    assert 'resolved.yaml' not in dispatch['args']


def test_local_vlm_stays_alive_until_other_nodes_finish(tmp_path):
    import time

    env = _environment(tmp_path)
    env.pop('VLLM_BASE_URL')
    env.update(_WORKER='1', SLURM_NODEID='0', SLURM_JOB_NUM_NODES='2')
    server = tmp_path / 'server.py'
    server.write_text(
        'import os, signal, time\n'
        'from pathlib import Path\n'
        "root = Path(os.environ['VLLM_HOST_FILE']).parent\n"
        "signal.signal(signal.SIGTERM, lambda *_: exit(0))\n"
        "(root / 'server_started').touch()\n"
        'try:\n'
        ' while True: time.sleep(0.01)\n'
        'finally:\n'
        " (root / 'server_stopped').touch()\n"
    )
    launch = tmp_path / 'serve.sh'
    launch.write_text(f'exec {shlex.quote(sys.executable)} {shlex.quote(str(server))}\n')
    env['VLLM_LAUNCH_SCRIPT'] = str(launch)
    worker = subprocess.Popen(['bash', str(ROOT / 'slurm/launch_annotation.sh'), 'libero'], env=env)
    try:
        deadline = time.monotonic() + 10
        done0 = Path(env['VLLM_HOST_FILE'] + '.done.0')
        while not (done0.exists() and (tmp_path / 'server_started').exists()):
            assert time.monotonic() < deadline
            assert worker.poll() is None
            time.sleep(0.02)
        assert worker.poll() is None
        assert not (tmp_path / 'server_stopped').exists()
        Path(env['VLLM_HOST_FILE'] + '.done.1').write_text('0\n')
        assert worker.wait(timeout=10) == 0
        assert (tmp_path / 'server_stopped').exists()
        assert not Path(env['VLLM_HOST_FILE']).exists()
    finally:
        if worker.poll() is None:
            worker.terminate()
            worker.wait(timeout=10)
