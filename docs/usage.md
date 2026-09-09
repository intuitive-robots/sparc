# SPARC usage guide

[← Back to SPARC](../README.md)

Commands below run from the repository root. For the end-to-end task-cache,
annotation, and reliability-threshold workflow, start with the
[README workflow](../README.md#workflow).

## How it works

```
Main process
  ├── GPUInferenceServer (1 per GPU) — loads the tracker, SAM2, and the detector
  │     └── serves RPC requests over per-model queues
  ├── CPU workers (N per GPU) — run TrajectoryAnnotator, call the GPU server
  │     └── via ModelProxy / SAM2Proxy / TrackerProxy / ...
  └── Writer process — AnnotationSaver consuming a queue → JSONL
```

Per trajectory the annotator: (1) detects gripper open/close phases, (2) parses
the instruction into subtasks and task objects with a VLM, then for each subtask
(3) detects candidate object/target boxes, (4) segments and tracks them, and
(5) selects the best initial box and — using **density-based matching**
(`point_score / sqrt(area)`) — the best target box.

## Installation

Use Linux with Python 3.12, NVIDIA CUDA-capable GPUs, Git, FFmpeg, and a C/C++
compiler (some ARM dependencies build from source).
Create a new annotation environment from the reference runtime's complete
package lock:

```bash
conda create -n robog-release -c conda-forge python=3.12 pip ffmpeg
conda activate robog-release
python -m pip install --no-deps -r requirements-lock.txt
python annotate.py --help
```

`requirements-lock.txt` pins the annotation/model dependency closure, including
transitive dependencies and Git revisions. `--no-deps` is intentional: it
reproduces the tested runtime despite OpenCV 4.12's declared NumPy `<2.3`
constraint against NumPy 2.4.4 and langchain-core's packaging `<26` constraint
against packaging 26.2. A blanket `pip check` reports these known metadata
conflicts; on ARM, uv also flags NVIDIA's upstream SBSA wheel tag. The direct version snapshots remain in
`requirements.txt` and `requirements-models.txt`. Do not install vLLM in this
environment; its Torch requirements differ.

The existing `robog-dataset-pipeline-tf4571` environment remains the development
and test baseline (Python 3.12.13, Torch 2.10.0+cu130, torchvision 0.25.0+cu130,
Transformers 4.57.6). The installation lock does not modify that environment. A fresh Linux aarch64
installation passed the test suite and RobotSeg CPU construction/frame-loading
checks; full GPU annotation in that fresh environment remains to be validated.
The reference environment separately passed a 10-trajectory DROID GPU run
with cached task parsing, producing 15 validated annotations.

Prepare the SAM2 checkpoint:

```bash
bash sparc/detectors/sam2/checkpoints/download_ckpts.sh
```

RobotSeg is enabled by default. Download **robotseg.pt** from the
[upstream checkpoint links](https://github.com/showlab/RobotSeg#72-download), then:

```bash
python setup/prepare_robotseg.py --checkpoint /path/to/robotseg.pt
```

This checks out a pinned RobotSeg revision under `vendors/robotseg/RobotSeg`,
removes the upstream prebuilt x86 binary, applies the NumPy frame-input adapter
used by this pipeline, and copies the
checkpoint to `checkpoints/robotseg/robotseg.pt`. Source and weights are installed
locally, outside the release archive. LLMDet, AllTracker, and MoGe fetch their
weights on first use; allow network access or populate their caches beforehand.

The vendored models have optional CUDA extensions. With a matching CUDA toolkit
and compiler available, build them using:

```bash
python -m pip install --no-deps --no-build-isolation -e sparc/detectors/sam2
python -m pip install --no-deps --no-build-isolation -e vendors/robotseg/RobotSeg
```

## VLM server

Use a separate environment or an existing OpenAI-compatible server:

```bash
conda create -n robog-vlm python=3.12 pip
conda activate robog-vlm
python -m pip install -r requirements-vllm.txt
bash slurm/serve_vlm.sh
```

The script serves `Qwen/Qwen3-VL-30B-A3B-Thinking` as `qwen3-vl-30b`, matching
the default preset. Set `VLLM_MODEL` to another compatible checkpoint or pass
vLLM flags to the script for your GPU capacity. In a second shell, activate the
annotation environment and set `vllm.base_url=http://your-vlm-host:8000/v1`.

For an in-job SLURM server, set `VLLM_LAUNCH_SCRIPT=$PWD/slurm/serve_vlm.sh` and
`VLLM_PYTHON` to the Python executable in the VLM environment. The annotation
launcher uses the separately activated annotation interpreter.

## DROID example

Run commands from this repository's root in the reference environment, with the
VLM server running. Replace `/path/to/...` with your dataset paths. DROID expects
the LeRobot dataset directory itself, containing `meta/info.json`, `data/`, and
`videos/`.

This example uses the left external camera, one GPU, and up to two trajectories:

```bash
mkdir -p runs/droid
python annotate.py --dataset droid_lerobot \
  dataset.root=/path/to/droid_success \
  dataset.task_obj_dict_path=./runs/droid/task_objects.pkl \
  annotator.output_file=./runs/droid/left_boxes.jsonl \
  annotator.gpu_ids=[0] \
  debug.debug_image_dir=./runs/droid/debug_left \
  debug.n_trajectories=2 debug.debug_image_freq=1
```

For the right external camera, use its separate preset and output file. Both
runs can share the task-object cache:

```bash
python annotate.py --dataset droid_lerobot_right \
  dataset.root=/path/to/droid_success \
  dataset.task_obj_dict_path=./runs/droid/task_objects.pkl \
  annotator.output_file=./runs/droid/right_boxes.jsonl \
  annotator.gpu_ids=[0] \
  debug.debug_image_dir=./runs/droid/debug_right \
  debug.n_trajectories=2 debug.debug_image_freq=1
```

The presets select `observation.images.left_external` and
`observation.images.right_external`, respectively, at 15 FPS, with cropping
enabled. They have distinct dataset IDs so their outputs can be resumed
independently. Their configured limit is 500 trajectories; use
`debug.n_trajectories=null` to process the full dataset.

## LIBERO example

`--dataset libero` selects the `libero_mixed` dataset ID. Its `dataset.root` is
**the parent of the `libero/` LeRobot directory**, for example:

```text
/path/to/libero_lerobot/
└── libero/
    ├── meta/info.json
    ├── data/
    └── videos/
```

```bash
mkdir -p runs/libero
python annotate.py --dataset libero \
  dataset.root=/path/to/libero_lerobot \
  dataset.task_obj_dict_path=./runs/libero/task_objects.pkl \
  annotator.output_file=./runs/libero/libero_mixed_boxes.jsonl \
  annotator.gpu_ids=[0] \
  debug.debug_image_dir=./runs/libero/debug \
  debug.n_trajectories=2 debug.debug_image_freq=1
```

This preset uses `observation.images.image` at 10 FPS and can include
`observation.images.image2` for semantic task parsing. Gripper state comes from
the last action dimension: `-1` is open, `+1` is closed. `extract_task_objects.mode=auto`
uses semantic parsing when there is more than one completed grasp cycle and
text-based parsing otherwise. Trajectory names follow the starVLA convention.

For LIBERO-plus, use `--dataset libero_plus` and a root containing a
`libero_plus/` LeRobot directory. That preset uses `observation.images.front`
(and `observation.images.wrist` for semantic parsing) at 20 FPS. Use separate
cache/output/debug paths for it. Both LIBERO presets default to 100 trajectories;
set `debug.n_trajectories=null` for a full run.

The examples override the dataset, task-cache, annotation-output, and debug
portable defaults derived from `dataset.root`. For multiple GPUs, set `annotator.gpu_ids=[0,1]`; adjust
`annotator.n_processes_per_gpu` for the CPU worker count. To disable debug images,
set `debug.save_debug_images=false`.

## Other entry points

```bash
# Another dataset preset
python annotate.py --dataset agibot_world dataset.root=/path/to/agibot

# An explicitly resolved config
python annotate.py --config /path/to/config.yaml
```

## Configuration

Every dataset preset requires `dataset.root=/path/to/dataset`. Task-object caches,
output files, and debug directories default to locations under that root;
overrides remain available for each.

All config is composed by Hydra from `configs/`:

- `configs/annotation_pipeline.yaml` — base defaults for every run (GPU/worker
  counts, detector, tracker, scoring, output). Defaults match the paper's
  efficiency setup: AllTracker + LLM-Det + RobotSeg, density target selection.
- `configs/dataset/*.yaml` — per-dataset `name`/`type`/`root`/`fps` and any
  `annotator.*` overrides.
- `configs/debug/{off,on,verbose}.yaml` — debug-image presets.
- `configs/vllm/*.yaml` — VLM model and sampling presets; the current default
  is `qwen3_vl_30b.yaml` (served model name `qwen3-vl-30b`).

The base defaults compose `debug`, `dataset`, and `vllm` groups; CLI overrides take final precedence. `output_file` and `debug_image_dir` default to paths under
`dataset.root` when left null (`sparc/pipeline/config.py`).

The base preset enables debug images and samples up to 50 trajectories.
Dataset presets can override this limit (for example, DROID uses 500).
For a full run without debug images, set `debug.n_trajectories=null` and
`debug.save_debug_images=false` explicitly.

Key fields:

| Key | Default | Meaning |
|-----|---------|---------|
| `dataset.root` | — | Path to the dataset (set this) |
| `annotator.gpu_ids` | `[0,1,2,3]` | Local GPU indices |
| `annotator.n_processes_per_gpu` | `6` | CPU workers per GPU |
| `annotator.detection_model` | `llmdet` | Supported object detector |
| `annotator.tracking_model` | `alltracker` | Supported point tracker |
| `annotator.n_tracker_instances` | `1` | Tracker instances per GPU |
| `annotator.enable_robotseg` | `true` | RobotSeg gripper segmentation |
| `annotator.enable_pointcloud` | `true` | MoGe depth lifting (3D scoring) |
| `annotator.save_pointclouds` | `false` | Save keyframe clouds and diagnostic 3D tracks |
| `extract_task_objects.mode` | `text` | Task parsing: `text`, multiview `semantic`, or `auto` |
| `vllm.base_url` | `localhost:8000/v1` | VLM endpoint for task parsing |

## Supported datasets

Dataset presets are in `configs/dataset/`: `agibot_world`, `bridge_lerobot`,
`droid_lerobot`, `droid_lerobot_right`, `libero`, `libero_plus`, `multi_lerobot`,
`oxe_lerobot`, `robomind2`, `vla_arena`, `egolive`, `galaxea`, `robocoin`,
`ours_real_robot`, and `ours_real_robot_ambiguity`. Loaders live in `sparc/data/`.

## Output

One JSON object per subtask annotation, written to
`{dataset.root}/{name}_boxes.jsonl`. The release pipeline writes
the `sparc.annotation` schema. All boxes, point tracks, and
masks refer to the original uncropped, unresized image. Frame indices are
zero-based trajectory indices; annotation windows are start-inclusive and
end-exclusive. For AgiBot, `source.source_episode_id` retains the Beta
`agibot_episode_ident` (or `source.source_uuid` retains the AgiBotWorld2026
UUID), while `source.episode_frame_start` and
`source.episode_frame_end_exclusive` locate the loader's subtrajectory inside
the original episode.

| Field group | Fields | Meaning |
|-------------|--------|---------|
| Schema and identity | `schema`, `annotation_id`, `source` | Schema version, stable annotation ID, dataset/trajectory/subtask/camera identifiers, split, and FPS |
| Coordinates and time | `coordinates`, `window` | Explicit image/box/mask conventions, original image size, and annotation interval |
| Task | `task.instruction`, `task.action`, `task.phases` | Instruction and absolute grasp/interact/release phase boundaries |
| Selected object | `object.label`, `object.initial`, `object.target`, `object.intermediate`, `object.track` | Selected boxes, detector scores, and optional selected point track |
| Selection | `selection.method`, `selection.score`, `selection.breakdown` | Final paper-method ranking score, its components/signals, constants, and density/argmax target choice |
| Detector baseline | `baselines.detector.initial`, `baselines.detector.target`, `baselines.detector.track` | Highest-detector-score initial/target boxes and the corresponding initial-candidate centroid track |
| Arrays | `arrays.storage`, `arrays.manifest` | Array storage reference and declared shapes, dtypes, and axes |
| Optional | `tool`, `robot.arm`, `arms` | Tool metadata, single-arm identity, or non-redundant per-arm object/selection records for bimanual samples |

`object.initial.detector_score` is the detector confidence. It is distinct from
`selection.score`, which is the final ranking score used to choose the emitted
box and is explicitly marked as not being a calibrated probability. The score
breakdown is kept only under `selection`; internal candidate arrays and legacy
confidence fields are not published. `baselines.detector` keeps only the
max-score detector choices needed to reproduce the detector-only baseline; it
does not duplicate the complete rejected-candidate set.

For bimanual records, object and selection fields live in each `arms[]` entry.
`filter_annotations.py` retains the complete record only when every arm has a
finite `selection.score` at or above `--threshold`.

Resume, merge, writing, and array loading require this schema.

### Arrays and masks

By default, `annotator.externalize_arrays=true` stores arrays in HDF5 shards
under `<output_stem>_artifacts/`. Each annotation's `arrays.storage` identifies
the relative shard path and group. Keep that directory alongside the JSONL when
moving or sharing output. Setting `annotator.externalize_arrays=false` keeps
compressed arrays inside the JSONL instead. `load_publication_arrays` also
supports NPZ sidecars for annotations through `arrays.storage.path`, with
optional `key_prefix` and `array_keys`.

Optional per-annotation raw track exports remain available with
`annotator.save_tracks_npz=true`; these diagnostic NPZ files store frame indices
and coordinates independently of the public JSONL arrays. NPZ keys such as
`cotracker_tracks` contain the AllTracker results.

For HDF5, NPZ, or embedded arrays:

```python
import json
from pathlib import Path
from sparc.pipeline.publication_schema import load_publication_arrays

jsonl_path = Path("/path/to/agibot_world_boxes.jsonl")
with jsonl_path.open() as stream:
    annotation = json.loads(next(stream))
arrays = load_publication_arrays(annotation, sidecar_root=jsonl_path.parent)
```

`arrays.manifest` describes the arrays actually present. These can include
object, robot, and detector-baseline masks with validity flags, centroid tracks,
source-frame mappings, and optionally MoGe keyframe point clouds. Masks use `0` for
background and `1` for foreground in the original image resolution. MoGe points
are in each keyframe's camera coordinates.

Point-cloud exports default to off. Set `annotator.save_pointclouds=true` to
save masked object/gripper keyframe clouds and include 3D tracks in diagnostic
NPZ files. `annotator.enable_pointcloud=true` is also required. Keeping
`save_pointclouds=false` preserves MoGe-based scoring while avoiding the extra
keyframe cloud export work and storage.

## starVLA integration

The pipeline can be driven directly from a starVLA training config so the dataset
you train on is exactly the one you annotate:

```bash
python annotate.py --starvla-config /path/to/starvla_train.yaml \
  --starvla-root /path/to/starVLA
```

This reads `datasets.vla_data.{data_root_dir, data_mix}`, resolves the mixture via
starVLA's `DATASET_NAMED_MIXTURES`, annotates those LeRobot subdatasets, and emits
`source.trajectory_id` values matching starVLA's CoT lookup key. Downstream model training
and presentation tools are outside this repository's main-pipeline scope.

## Multi-node (SLURM)

Activate the annotation environment and submit from this checkout's root:

```bash
export VLLM_BASE_URL=http://your-vlm-host:8000/v1
sbatch --partition=YOUR_PARTITION --nodes=2 --cpus-per-task=32 \
  --time=24:00:00 slurm/launch_annotation.sh droid_lerobot \
  dataset.root=/path/to/droid_success
```

The launcher uses the submission checkout and inherited Python/CUDA/cache
settings. Set `PIPELINE_ROOT` when submitting elsewhere, or `PIPELINE_PYTHON`
for an explicit annotation interpreter. GPU indices follow the SLURM allocation.
Set partition, CPU, memory, and time requests for your cluster at submission.

To start a VLM within the job, unset `VLLM_BASE_URL` and set
`VLLM_LAUNCH_SCRIPT` to a shared launch script. It receives `PORT` (default 8000)
and the first allocated GPU on node 0 through `CUDA_VISIBLE_DEVICES`; remaining
GPUs annotate. Node 0 therefore needs at least two GPUs. Shared job state lives
under `runs/slurm/<job_id>/`; override `VLLM_HOST_FILE` if needed.

Each node writes `{name}_boxes_node{rank}.jsonl`; the launcher merges outputs
when workers finish. The launcher accepts `--config /path/to/resolved.yaml`
or `--resume-dir /path/to/run` (`--continue-dir` is an alias). A run directory
must contain exactly one `*_config.yaml`. The Python CLI accepts `--config`;
resume-directory flags belong to the launcher. Existing annotations are skipped
automatically when rerunning with the same output path. For a manual merge, select the same dataset and paths used for annotation:

```bash
python annotate.py --dataset droid_lerobot \
  dataset.root=/path/to/droid_success --merge-only
```

If the run overrode `annotator.output_file`, pass that same override when merging.

## Repository layout

| Path | Role |
|------|------|
| `annotate.py` | Entry point: config, process spawning, dataset loading |
| `sparc/` | GPU server, proxies, trajectory annotator, scoring, box ops |
| `sparc/data/` | LeRobot dataset loaders |
| `sparc/robot/`, `sparc/perception/`, `sparc/llm/` | Gripper phases, detection helpers, LLM task parsing |
| `sparc/detectors/` | Model adapters and vendored inference code |
| `configs/` | Hydra config tree |
| `extract_task_objects.py` | Optional VLM pre-pass to cache task objects |
| `filter_annotations.py` | Retain annotations above a selection-score threshold |

## Scope and tests

This checkout contains the annotation pipeline, task-object cache pre-pass,
configuration, debug images, tests, and the SLURM launcher. Download model
weights separately using the installation instructions.

SAM2 and AllTracker retain their inference source and licenses. Their upstream
READMEs also describe training and demos that are not shipped here; see
[vendor scope](vendor-scope.md) for the retained files.
Install the test dependency and run the first-party tests in your annotation
environment:

```bash
python -m pip install --no-deps -r requirements-dev.txt
python -m pytest tests -q
```
