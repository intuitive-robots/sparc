# Repository guidance

## Runtime and commands

Use the existing Conda environment `robog-dataset-pipeline-tf4571` for running and
testing. Requirements record its installed versions; do not replace them with a
new environment solve during cleanup.

```bash
conda activate robog-dataset-pipeline-tf4571
python annotate.py --dataset agibot_world dataset.root=/path/to/agibot_world
python annotate.py --dataset droid_lerobot dataset.root=/path/to/droid annotator.gpu_ids=[0,1]
python annotate.py --config /path/to/resolved.yaml
python -m pytest tests -q
```

Hydra composes `configs/annotation_pipeline.yaml` with dataset, debug, and VLM
presets. CLI dot-list overrides take final precedence. The optional
`extract_task_objects.py` pre-pass caches task objects; annotation can also
resolve them live. `slurm/launch_annotation.sh` launches multi-node annotation.

## Architecture

The supported setup is LLMDet + AllTracker + SAM2 image segmentation, with
RobotSeg and MoGe enabled by default and Qwen/VLM task parsing. Alternative
detector/tracker and SAM2-video/hybrid modes have been retired.

One GPU server runs per GPU, with model queues for detection, segmentation,
tracking, point clouds, and RobotSeg. CPU workers call model proxies; the writer
consumes annotations through a separate queue. Large arrays use shared memory
for IPC. `annotator.n_tracker_instances` controls tracker instances per GPU.

| File | Role |
|------|------|
| `annotate.py` | CLI, configuration, process spawning, dataset loading |
| `sparc/pipeline/config.py` | Config composition, supported setup validation, loader dispatch |
| `sparc/pipeline/gpu_inference.py` | GPU server, model proxies, shared memory |
| `sparc/pipeline/trajectory_annotator.py` | Single-arm and bimanual orchestration, detection, and segmentation |
| `sparc/pipeline/annotation_coordinates.py` | Cropping and coordinate restoration |
| `sparc/pipeline/annotation_scoring.py` | Tracking, candidate selection, and verification |
| `sparc/pipeline/annotation_export.py` | Public output assembly, phase masks, and debug exports |
| `sparc/pipeline/annotation_saver.py` | JSONL writer and HDF5 array shards |
| `sparc/pipeline/publication_schema.py` | Public schema conversion and array loading |
| `sparc/perception/ann_utils.py` | Detection helpers and candidate selection |
| `sparc/perception/tracking_models.py` | AllTracker loading, point sampling, adapter |
| `sparc/scoring/` | Selection scores and signals |
| `sparc/llm/` | Task parsing, prompts, VLM verification |
| `sparc/robot/keystate_utils.py` | Gripper phases |
| `sparc/data/` | Dataset loaders and starVLA config bridge |

The writer emits `sparc.annotation` version `2.0`. Default array storage is HDF5
alongside the JSONL; keep shard directories with their annotations. See README
for coordinate conventions and array loading.

## Version control

After each change, commit and push to GitHub. Make sure to NOT include CLAUDE as
commit user.

## Refactoring guidelines

- Prefer incremental changes over large rewrites and follow SOLID principles.
- Preserve functionality: never change what the retained pipeline does. Keep
  its features, calculations, outputs, and behavior intact unless the user
  explicitly authorizes retiring a feature.
- Improve clarity by reducing complexity/nesting, eliminating redundant code
  and abstractions, using clear names, and consolidating related logic.
- Remove disabled code and comments that only restate obvious operations.
- Do not remove code merely because its name includes `legacy` or `cotracker`:
  internal publication conversion and historical output fields still use these
  names. AllTracker output keys such as `cotracker_tracks` are preserved.
- Keep upstream inference source and licenses when trimming vendor assets.

See `README.md` for release setup and validation commands.
