<div align="center">

# SPARC

### Reliable Spatial Annotations from Robot Demonstrations at Scale

**CoRL 2026**

[Paper](https://arxiv.org/abs/2606.13497) · [Project website](https://intuitive-robots.github.io/sparc-labeling/) · [BibTeX](#citation)

[Quickstart](#quickstart) | [Workflow](#workflow) | [Outputs](#outputs) | [Documentation](#documentation) | [Citation](#citation)

</div>

<a id="about"></a>

## About

**SPARC turns robot demonstrations into structured spatial annotations:**
object boxes, motion traces, manipulation phases, and interaction-based selection
scores. It combines task parsing, object detection, segmentation, and tracking
to identify the objects a robot actually interacts with.

See the
[paper](https://arxiv.org/abs/2606.13497) for the method and experiments, or explore
[demonstrations and results](https://intuitive-robots.github.io/sparc-labeling/)
on the project page.

This repository contains the annotation pipeline. The project page covers the
broader paper, including annotation reliability and downstream learning results.

<a id="quickstart"></a>
<a id="installation"></a>

## Quickstart

**You’ll need:** Linux, Python 3.12, NVIDIA GPUs, Conda, Git, FFmpeg, and a C/C++
compiler. Run the commands from this repository’s root on a GPU machine; on a
managed cluster, use a compute allocation.

### 1. Install SPARC and prepare model weights

```bash
conda create -n sparc -c conda-forge python=3.12 pip ffmpeg
conda activate sparc
python -m pip install --no-deps -r requirements-lock.txt

# SAM2 checkpoint
bash sparc/detectors/sam2/checkpoints/download_ckpts.sh

# Download robotseg.pt first, then prepare RobotSeg
python setup/prepare_robotseg.py --checkpoint /path/to/robotseg.pt
```

Get `robotseg.pt` from the [RobotSeg checkpoint links](https://github.com/showlab/RobotSeg#72-download).
LLMDet, AllTracker, and MoGe download weights on first use.
The lock’s `--no-deps` flag is intentional; see
[installation notes and CUDA extension builds](docs/usage.md#installation)
for dependency constraints, offline preparation, and validation scope.

### 2. Connect a VLM server

Task parsing uses an OpenAI-compatible endpoint, with `qwen3-vl-30b` as the
served model name. Use an existing endpoint, or start the supplied Qwen server
in a **separate environment** on a GPU allocated for serving:

```bash
conda create -n sparc-vlm python=3.12 pip
conda activate sparc-vlm
python -m pip install -r requirements-vllm.txt
bash slurm/serve_vlm.sh
```

Keep the server running. In another shell, activate `sparc` for annotation.
See [VLM setup](docs/usage.md#vlm-server) for model overrides and in-job serving.

Next, follow the workflow below to build a task cache, annotate trajectories,
and select annotations by their reliability score.

<a id="workflow"></a>

## Pipeline workflow

**Demonstrations → task-object cache → spatial annotations → score filtering**

SPARC first decomposes instructions into subtasks and identifies task objects.
For each subtask, it detects candidate objects with LLMDet, segments them with
SAM2, and tracks their motion with AllTracker. RobotSeg identifies the robot,
and MoGe supplies 3D geometry for scoring. Phase-aware motion, gripper proximity,
and robot overlap help select the interacted object and its target box.
The resulting interaction-based reliability score is stored as `selection.score`.

### 1. Extract task objects → pickle cache

With the VLM server running, precompute the task descriptions:

```bash
conda activate sparc
mkdir -p runs/droid

python extract_task_objects.py --dataset droid_lerobot \
  dataset.root=/path/to/droid_success \
  dataset.task_obj_dict_path=./runs/droid/task_objects.pkl \
  vllm.base_url=http://your-vlm-host:8000/v1
```

This writes a reusable `task_objects.pkl` cache. Use the same dataset, parsing
mode, and cache path in the annotation step. The pre-pass processes the dataset;
`extract_task_objects.max_trajectories` can limit it. It is optional: annotation
can also resolve and cache missing task descriptions through the VLM server.

Tune extraction with `extract_task_objects.n_workers=8` and
`extract_task_objects.vlm_concurrency=64`; vLLM batches the concurrent requests.
See [batching controls](docs/tuning.md#extraction-concurrency-and-vllm-batching).

### 2. Annotate trajectories → JSONL + arrays

Point `dataset.root` at a LeRobot DROID directory containing `meta/info.json`,
`data/`, and `videos/`. Replace the endpoint with your VLM server’s address.

```bash
conda activate sparc
mkdir -p runs/droid

python annotate.py --dataset droid_lerobot \
  dataset.root=/path/to/droid_success \
  dataset.task_obj_dict_path=./runs/droid/task_objects.pkl \
  annotator.output_file=./runs/droid/annotations.jsonl \
  annotator.gpu_ids=[0] annotator.n_processes_per_gpu=1 \
  debug.n_trajectories=2 debug.debug_image_freq=1 \
  debug.debug_image_dir=./runs/droid/debug \
  vllm.base_url=http://your-vlm-host:8000/v1
```

This uses the left external camera and writes annotations, array shards, and
debug images under `runs/droid/`. To process the full dataset without debug
images, set `debug.n_trajectories=null debug.save_debug_images=false`.
Rerunning with the same output path skips existing annotations.

For lower VRAM use, start with `annotator.pointcloud_batch_size=8`,
`annotator.tracking_max_frames=200`, and `annotator.robotseg_max_frames=32`.
See [memory tuning](docs/tuning.md#reducing-annotation-vram-use) for all controls.

### 3. Filter annotations → retained JSONL

Use `selection.score`, rather than detector confidence, to filter annotations.
**Start with a threshold of `0.95`, then tune it for your dataset.** Higher
thresholds generally trade coverage for quality; `0.95` does **not** mean 95%
correctness. Check representative examples before choosing an operating point.
Filtering happens per subtask, so a trajectory may retain only some subtasks.

**Detector confidence is the detector-only baseline score**, not SPARC's
reliability score:

| Field | Meaning |
|---|---|
| `selection.score` | SPARC's interaction-based reliability ranking; use this for the filter below |
| `baselines.detector.initial.detector_score` | Detector confidence for the baseline's highest-confidence initial box |
| `baselines.detector.target.detector_score` | Detector confidence for the baseline's highest-confidence target box, when available |
| `object.initial.detector_score` | Detector confidence of the box selected by SPARC; that box can differ from the detector baseline |

Detector confidence and SPARC's score use different criteria; the `0.95`
reliability threshold should not be interpreted as a detector-confidence cutoff.

```bash
python filter_annotations.py runs/droid/annotations.jsonl --threshold 0.95
```

For bimanual annotations, every arm's `selection.score` must meet the threshold;
the command keeps or drops the complete annotation.

This creates `runs/droid/annotations_filtered.jsonl` and reports how many records
were retained. Missing, nonnumeric, and nonfinite scores are skipped. Use
`--output /path/to/filtered.jsonl` to choose another location; relative array
references are adjusted automatically. Existing files are never overwritten.

Filtering leaves the source JSONL and shared array shards intact. When sharing
the filtered output, include its referenced `annotations_artifacts/` directory.

**More examples:** [right-camera DROID](docs/usage.md#droid-example) ·
[LIBERO](docs/usage.md#libero-example) ·
[multi-node SLURM](docs/usage.md#multi-node-slurm) ·
[all configuration options](docs/usage.md#configuration)

<a id="outputs"></a>

## What you get

Each subtask produces one JSONL annotation record. Coordinates refer
to the original image; frame windows are start-inclusive and end-exclusive.

| Output | Contents |
|---|---|
| `annotations.jsonl` | Task, phases, selected boxes, scores, source identity, and array references |
| `annotations_artifacts/` | HDF5 shards containing masks and tracks; point clouds are opt-in |
| `debug/` | Visual annotation checks and JSON metadata |
| `tracks/` *(optional)* | Raw diagnostic exports with `annotator.save_tracks_npz=true` |

**Point-cloud saving is off by default.** Set `annotator.save_pointclouds=true`
to save object/gripper keyframe clouds and include 3D tracks in optional NPZ
exports. This requires `annotator.enable_pointcloud=true` (the default).
Saving is separate from computation: MoGe still supplies 3D scoring signals
when `save_pointclouds=false`. To disable MoGe itself, use
`annotator.enable_pointcloud=false`, which also changes the available scoring signals.

**Keep the array-shard directory alongside the JSONL** when moving outputs.
Load arrays with the same helper for HDF5, NPZ sidecars, or embedded storage:

```python
import json
from pathlib import Path
from sparc.pipeline.publication_schema import load_publication_arrays

path = Path("runs/droid/annotations.jsonl")
with path.open() as stream:
    annotation = json.loads(next(stream))
arrays = load_publication_arrays(annotation, sidecar_root=path.parent)
```

Selection scores rank annotations; they are not calibrated probabilities.
See the [output reference](docs/usage.md#output) for fields, coordinate
conventions, bimanual annotations, and storage options.

<a id="documentation"></a>

## Documentation

| I want to… | Start here |
|---|---|
| Set up checkpoints or CUDA extensions | [Installation](docs/usage.md#installation) |
| Annotate another dataset | [Supported datasets](docs/usage.md#supported-datasets) |
| Tune extraction batching or GPU memory | [Performance tuning](docs/tuning.md) |
| Change GPUs, workers, parsing, or debug settings | [Configuration](docs/usage.md#configuration) |
| Use a starVLA dataset mixture | [starVLA integration](docs/usage.md#starvla-integration) |
| Launch, resume, or merge a multi-node run | [SLURM guide](docs/usage.md#multi-node-slurm) |
| Understand the implementation | [How it works](docs/usage.md#how-it-works) · [Repository layout](docs/usage.md#repository-layout) |
| Run the tests | [Testing](docs/usage.md#scope-and-tests) |

<a id="citation"></a>

## Citation

If you use SPARC, please cite our paper.

[Paper](https://arxiv.org/abs/2606.13497) · [BibTeX file](CITATION.bib)

```bibtex
@misc{blank2026sparc,
  title         = {{SPARC}: Reliable Spatial Annotations from Robot Demonstrations at Scale},
  author        = {Nils Blank and Paul Mattes and Maximilian Xiling Li and Jakub Suliga and Thomas Roth and Moritz Reuss and Pankhuri Vanjani and Rudolf Lioutikov},
  year          = {2026},
  eprint        = {2606.13497},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2606.13497}
}
```

## Acknowledgments

SPARC builds on LLMDet, AllTracker, SAM2, RobotSeg, and MoGe. We thank their
authors for making these models available. See [vendor scope](docs/vendor-scope.md)
for the inference components included here and their upstream licenses.
