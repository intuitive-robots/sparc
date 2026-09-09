# Performance tuning

[Back to the workflow](../README.md#workflow)

## Extraction concurrency and vLLM batching

Extraction prepares tasks with CPU workers and sends concurrent requests to the
VLM endpoint. **vLLM continuously batches those requests on the server**; there
is no fixed trajectory batch size in the extractor. For example:

```bash
python extract_task_objects.py --dataset droid_lerobot \
  dataset.root=/path/to/droid_success \
  dataset.task_obj_dict_path=./runs/droid/task_objects.pkl \
  extract_task_objects.n_workers=8 \
  extract_task_objects.vlm_concurrency=64 \
  vllm.base_url=http://your-vlm-host:8000/v1
```

`n_workers` controls CPU preparation (default `16`); `vlm_concurrency` limits
concurrent client requests (default `1500`). Start lower and increase concurrency
if the server is underutilized. In `semantic` or `auto` mode, the semantic
`max_cpu_workers` and `max_vlm_concurrency` settings additionally cap these
values; `max_pending_requests` bounds queued requests.

Configure the GPU batching limits when starting the **VLM server**, for example:

```bash
bash slurm/serve_vlm.sh --max-num-seqs 32 --max-num-batched-tokens 4096
```

These limit sequences and tokens scheduled per iteration, respectively.
Client concurrency and server batch limits do not need to match: extra requests
wait for service. Smaller server limits can reduce memory pressure, at a
throughput cost. These are tuning examples, not guaranteed settings for every
GPU. See the [vLLM tuning guide](https://docs.vllm.ai/en/v0.11.0/configuration/optimization.html).

## Reducing annotation VRAM use

Start with a smaller **MoGe batch**, then reduce sequence sizes or resolution
if needed. AllTracker and RobotSeg do not expose a general `batch_size` override
in this pipeline; use their frame limits below.

| Setting | Example | Effect |
|---|---|---|
| `annotator.pointcloud_batch_size` | `8` (default `64`) | Fewer frames per MoGe inference batch |
| `annotator.n_tracker_instances` | `1` (default) | Avoid extra AllTracker model copies on each GPU |
| `annotator.tracking_max_frames` | `200` (default `800`) | Fewer frames in each AllTracker sequence |
| `annotator.tracking_max_resolution` | `512` | Cap the longest edge of tracking frames |
| `annotator.robotseg_max_frames` | `32` (default `64`) | Fewer sampled frames in RobotSeg's video state |
| `annotator.pointcloud_resolution_level` | `0` (default `2`) | Lower MoGe internal resolution |
| `annotator.max_resolution` | `720` | Cap input resolution across perception stages |

Append the needed overrides to the annotation command, for example:

```bash
python annotate.py --dataset droid_lerobot \
  dataset.root=/path/to/droid_success \
  dataset.task_obj_dict_path=./runs/droid/task_objects.pkl \
  annotator.output_file=./runs/droid/annotations.jsonl \
  annotator.gpu_ids=[0] annotator.n_processes_per_gpu=1 \
  annotator.pointcloud_batch_size=8 \
  annotator.tracking_max_frames=200 annotator.tracking_max_resolution=512 \
  annotator.robotseg_max_frames=32 \
  vllm.base_url=http://your-vlm-host:8000/v1
```

These values are starting points, not a tested VRAM guarantee. CPU workers share the GPU models:
reducing `n_processes_per_gpu` limits work in flight, but does not shrink model
weights or an individual inference request.
Point-cloud **saving** is already off by default; this does not disable MoGe's
scoring computations.

