"""GPU inference server and model proxies for multi-process annotation.

Serves LLMDet, SAM2 image segmentation, AllTracker, MoGe, and RobotSeg requests.
"""

import logging
import os
import threading
import time
import traceback
from collections import defaultdict
from pathlib import Path
from multiprocessing import Process
from multiprocessing.shared_memory import SharedMemory

import torch

import numpy as np

# Instrumentation gated by env var. Enable with PROFILE_PIPELINE=1.
PROFILE_PIPELINE = os.environ.get("PROFILE_PIPELINE", "0") == "1"
SUPPORTED_DETECTION_MODELS = {"llmdet"}
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SAM2_PACKAGE_ROOT = REPOSITORY_ROOT / "sparc" / "detectors" / "sam2"


def _resolve_sam2_checkpoint(repository_root=REPOSITORY_ROOT):
    """Find the SAM2 small checkpoint in current and pre-refactor layouts."""
    repository_root = Path(repository_root)
    configured_path = os.environ.get("SAM2_CHECKPOINT")
    candidates = [
        Path(configured_path).expanduser() if configured_path else None,
        repository_root / "sparc/detectors/sam2/checkpoints/sam2.1_hiera_small.pt",
        repository_root / "checkpoints/sam2/sam2.1_hiera_small.pt",
        repository_root / "annotator/detectors/sam2/checkpoints/sam2.1_hiera_small.pt",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate
    searched = "\n  - ".join(str(path) for path in candidates if path is not None)
    raise FileNotFoundError(
        "SAM2 checkpoint sam2.1_hiera_small.pt was not found. Set "
        f"SAM2_CHECKPOINT or place it in one of:\n  - {searched}"
    )


def _prepare_tracking_frames(frames_torch, device):
    """Move compact video frames to the tracking device, then cast to float32."""
    frames_torch = frames_torch.to(device)
    if not torch.is_floating_point(frames_torch):
        frames_torch = frames_torch.float()
    return frames_torch


class PerfCounter:
    """Thread-safe per-task-type accumulator for timing breakdowns.

    Tracks: count, total wall-time (s), and per-component sub-times (s or ms).
    Use add() to record one call; summary() returns a formatted string.
    """
    def __init__(self, name="perf"):
        self.name = name
        self._lock = threading.Lock()
        self._stats = defaultdict(lambda: defaultdict(float))
        self._counts = defaultdict(int)

    def add(self, task, **fields):
        with self._lock:
            self._counts[task] += 1
            for k, v in fields.items():
                self._stats[task][k] += float(v)

    def snapshot(self):
        with self._lock:
            tasks = list(self._counts.keys())
            return {
                t: (self._counts[t], dict(self._stats[t])) for t in tasks
            }

    def summary(self, header=None):
        snap = self.snapshot()
        if not snap:
            return f"[{self.name}] no samples"
        lines = []
        if header:
            lines.append(header)
        # Collect all field names across tasks
        all_fields = sorted({f for _, (n, fs) in snap.items() for f in fs.keys()})
        col = lambda s: f"{s:>10}"
        lines.append(f"[{self.name}] " + col("task") + " " + col("count") + " " + " ".join(col(f) for f in all_fields))
        for task, (n, fields) in sorted(snap.items()):
            row = [col(task), col(str(n))]
            for f in all_fields:
                tot = fields.get(f, 0.0)
                mean = tot / n if n else 0.0
                # ms-suffixed fields are already in ms; otherwise seconds
                row.append(col(f"{mean:.3f}" if mean < 100 else f"{mean:.1f}"))
            lines.append("[%s] " % self.name + " ".join(row))
        # Also report total wallclock per task for share calculation
        if any("wall_s" in fs for _, (_, fs) in snap.items()):
            lines.append(f"[{self.name}] -- total wall_s share --")
            grand = sum(fs.get("wall_s", 0.0) for _, (_, fs) in snap.items())
            for task, (n, fields) in sorted(snap.items()):
                w = fields.get("wall_s", 0.0)
                pct = 100.0 * w / grand if grand else 0.0
                lines.append(f"[{self.name}] {task:>10}: {w:8.2f}s  ({pct:5.1f}%) over {n} calls — {w/n*1000:.1f} ms/call")
        return "\n".join(lines)


def _get_visible_gpu_devices():
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible_devices:
        return []
    return [dev.strip() for dev in visible_devices.split(",") if dev.strip()]


class SharedTensorManager:
    """Helper to send numpy/torch arrays via SharedMemory to avoid Pickle overhead."""

    @staticmethod
    def send(array_np):
        """
        Allocates shared memory, copies array, returns metadata handle.
        Caller must eventually unlink() this memory!
        """
        if hasattr(array_np, 'numpy'):
            array_np = array_np.numpy()

        # Calculate size
        d_size = np.dtype(array_np.dtype).itemsize * np.prod(array_np.shape)

        # Create block
        shm = SharedMemory(create=True, size=int(d_size))

        # Create a numpy array backed by shared memory
        shm_array = np.ndarray(array_np.shape, dtype=array_np.dtype, buffer=shm.buf)

        # Copy data
        shm_array[:] = array_np[:]

        # Return metadata needed to reconstruct on other side
        return {
            'name': shm.name,
            'shape': array_np.shape,
            'dtype': str(array_np.dtype)
        }, shm

    @staticmethod
    def receive(metadata):
        """
        Attaches to existing shared memory and returns numpy array.
        """
        shm = SharedMemory(name=metadata['name'])
        array = np.ndarray(metadata['shape'], dtype=metadata['dtype'], buffer=shm.buf)
        return array, shm


def _pack_result_shm(obj):
    """Replace large arrays/tensors in a GPU result with SharedMemory handles.

    Called by worker threads before putting results on the reply queue.
    The returned shm objects must NOT be unlinked by the caller — the proxy
    that receives the packed result owns cleanup via _unpack_result_shm.
    Returns (packed_obj, shm_refs); shm_refs only need to stay alive long
    enough for GC to close (not unlink) the file handles.
    """
    import torch
    shm_refs = []

    def pack(x):
        arr, type_tag = None, None
        if isinstance(x, np.ndarray) and x.nbytes > 100_000:
            arr, type_tag = x, 'ndarray'
        elif isinstance(x, torch.Tensor) and not x.is_cuda:
            if x.element_size() * x.numel() > 100_000:
                arr, type_tag = x.numpy(), 'tensor'
        if arr is not None:
            meta, shm = SharedTensorManager.send(arr)
            shm_refs.append(shm)  # proxy will unlink; GC closes our fd
            return {'__shm_result__': meta, '__type__': type_tag}
        if isinstance(x, list):
            return [pack(i) for i in x]
        if isinstance(x, tuple):
            return tuple(pack(i) for i in x)
        if isinstance(x, dict):
            return {k: pack(v) for k, v in x.items()}
        return x

    return pack(obj), shm_refs


def _unpack_result_shm(obj):
    """Reconstruct SharedMemory handles in a packed result back to arrays/tensors.

    Called by ModelProxy._rpc immediately after get(). Copies each array out
    of shared memory, then closes and unlinks the shm block.
    """
    import torch

    def unpack(x):
        if isinstance(x, dict) and '__shm_result__' in x:
            arr, shm = SharedTensorManager.receive(x['__shm_result__'])
            data = arr.copy()   # copy out before closing
            shm.close()
            shm.unlink()
            return torch.from_numpy(data) if x.get('__type__') == 'tensor' else data
        if isinstance(x, list):
            return [unpack(i) for i in x]
        if isinstance(x, tuple):
            return tuple(unpack(i) for i in x)
        if isinstance(x, dict):
            return {k: unpack(v) for k, v in x.items()}
        return x

    return unpack(obj)


class ModelRequest:
    """Standard container for sending jobs to the GPU."""
    def __init__(self, task_type, payload, reply_queue_id):
        self.task_type = task_type
        self.payload = payload
        self.reply_queue_id = reply_queue_id


class GPUInferenceServer(Process):
    """Load the annotation models once per GPU and serve their request queues.

      - detect_queue: LLMDet object detection (sequential, CUDA stream_detect)
      - sam_queue: SAM2 predict (sequential, CUDA stream_sam)
      - tracking_queue: AllTracker point tracking (one stream per tracker instance)
      - pointcloud_queue: MoGe 3D point cloud prediction (sequential, CUDA stream_pc)
      - robotseg_queue: RobotSeg gripper segmentation (sequential, CUDA stream_rs)
    Threads share the same models but use separate CUDA streams so the
    GPU scheduler can overlap them when SM headroom is available.
    """
    def __init__(self, gpu_id, detect_queue, sam_queue, tracking_queue,
                 reply_queues_dict, server_ready_event, tracking_model_name="alltracker",
                 n_tracker_instances=2, detection_model_name="llmdet",
                 detection_config=None,
                 pointcloud_queue=None,
                 robotseg_queue=None, runtime_detection_thresholds=None):
        super().__init__()
        self.gpu_id = gpu_id
        self.detect_queue = detect_queue
        self.sam_queue = sam_queue
        self.tracking_queue = tracking_queue
        self.pointcloud_queue = pointcloud_queue
        self.robotseg_queue = robotseg_queue
        self.reply_queues_dict = reply_queues_dict
        self.server_ready_event = server_ready_event
        self.tracking_model_name = tracking_model_name
        self.n_tracker_instances = n_tracker_instances
        self.detection_model_name = detection_model_name
        self.detection_config = detection_config or {}
        self.runtime_detection_thresholds = runtime_detection_thresholds or {}
        self.daemon = True
        self.gpu_label = str(gpu_id)

        if self.detection_model_name not in SUPPORTED_DETECTION_MODELS:
            raise ValueError(
                f"Unsupported detection_model_name={self.detection_model_name!r}. "
                f"Expected one of {sorted(SUPPORTED_DETECTION_MODELS)}"
            )
        if self.tracking_model_name != "alltracker":
            raise ValueError("tracking_model_name supports only 'alltracker'")


    def _gpu_log_label(self):
        return self.gpu_label

    def _detector_settings(self):
        settings = self.detection_config.get(self.detection_model_name, {})
        return dict(settings or {})

    def _detection_startup_message(self, detector_settings):
        if self.detection_model_name != "llmdet" or not self.runtime_detection_thresholds:
            return (
                f"GPU Server {self._gpu_log_label()}: loading detection model "
                f"{self.detection_model_name} with settings={detector_settings}"
            )

        thresholds = self.runtime_detection_thresholds
        return (
            f"GPU Server {self._gpu_log_label()}: loading detection model llmdet; "
            "effective per-call thresholds: "
            f"standard object/target box_query={thresholds['standard_box_query']:g}, "
            f"text_class={thresholds['standard_text_class']:g}; "
            f"tool box_query={thresholds['tool_box_query']:g}, "
            f"text_class={thresholds['tool_text_class']:g}; "
            f"tool-object box_query={thresholds['tool_object_box_query']:g}, "
            f"text_class={thresholds['tool_object_text_class']:g} "
            f"(model_id={detector_settings.get('model_id', 'iSEE-Laboratory/llmdet_base')}, "
            f"use_slicing={bool(detector_settings.get('use_slicing', False))})"
        )

    def _record_gpu_time(self, stream):
        """Return (start_event, end_event) recorded on `stream`. Use with _gpu_elapsed_ms."""
        if not PROFILE_PIPELINE:
            return None, None
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        return start, end

    def _gpu_elapsed_ms(self, start, end, stream):
        """Caller must have called stream.synchronize() already."""
        if not PROFILE_PIPELINE or start is None:
            return 0.0
        end.record(stream)
        end.synchronize()
        return start.elapsed_time(end)

    def _periodic_perf_logger(self):
        """Background thread: log perf summary every 60s when profiling is enabled."""
        while not self._gpu_dead.is_set():
            time.sleep(60)
            if hasattr(self, "_perf"):
                logging.info("\n" + self._perf.summary(header=f"[GPU-{self._gpu_log_label()}] live perf"))
                try:
                    dev = torch.device(f"cuda:{self.gpu_id}")
                    alloc = torch.cuda.memory_allocated(dev) / 1024 / 1024
                    reserved = torch.cuda.memory_reserved(dev) / 1024 / 1024
                    peak_alloc = torch.cuda.max_memory_allocated(dev) / 1024 / 1024
                    peak_reserved = torch.cuda.max_memory_reserved(dev) / 1024 / 1024
                    free, total = torch.cuda.mem_get_info(dev)
                    free_mb, total_mb = free / 1024 / 1024, total / 1024 / 1024
                    logging.info(
                        f"[GPU-{self._gpu_log_label()}] VRAM live: "
                        f"alloc={alloc:.0f}MB peak_alloc={peak_alloc:.0f}MB "
                        f"reserved={reserved:.0f}MB peak_reserved={peak_reserved:.0f}MB "
                        f"| node-level free={free_mb:.0f}MB / total={total_mb:.0f}MB"
                    )
                except Exception as e:
                    logging.debug(f"VRAM logging failed: {e}")

    def _check_gpu_dead(self, request):
        """If the GPU context is corrupted, immediately return an error to the caller."""
        if self._gpu_dead.is_set():
            self.reply_queues_dict[request.reply_queue_id].put(
                RuntimeError("GPU context is corrupted (prior CUDA assertion failure). Cannot process further requests.")
            )
            return True
        return False

    def _reconstruct_args(self, args, kwargs, device):
        import torch
        import numpy as np

        def reconstruct(arg):
            if isinstance(arg, dict) and '__shm__' in arg:
                meta = arg['__shm__']
                arr, shm = SharedTensorManager.receive(meta)
                try:
                    tensor = torch.from_numpy(arr).to(device)
                    # A CPU .to() aliases the shared-memory buffer, which is about
                    # to close. CUDA transfers are synchronous here and own storage.
                    if tensor.device.type == 'cpu':
                        tensor = tensor.clone()
                except Exception:
                    tensor = arr.copy()
                shm.close()
                return tensor
            elif isinstance(arg, np.ndarray):
                return torch.from_numpy(arg).to(device)
            elif isinstance(arg, torch.Tensor):
                return arg.to(device)
            return arg

        new_args = []
        for a in args:
            if isinstance(a, (list, tuple)):
                 new_args.append([reconstruct(x) for x in a])
            else:
                new_args.append(reconstruct(a))

        new_kwargs = {k: reconstruct(v) for k, v in kwargs.items()}
        return tuple(new_args), new_kwargs

    def _reconstruct_args_cpu(self, args, kwargs):
        """Reconstruct RobotSeg request payloads as CPU NumPy arrays."""
        def reconstruct(arg):
            if isinstance(arg, dict) and '__shm__' in arg:
                meta = arg['__shm__']
                arr, shm = SharedTensorManager.receive(meta)
                result = arr.copy()
                shm.close()
                return result
            return arg

        new_args = []
        for a in args:
            if isinstance(a, (list, tuple)):
                new_args.append([reconstruct(x) for x in a])
            else:
                new_args.append(reconstruct(a))
        new_kwargs = {k: reconstruct(v) for k, v in kwargs.items()}
        return tuple(new_args), new_kwargs

    def _move_to_cpu(self, obj):
        """
        Recursively converts everything to CPU-safe types.
        Tensors are moved to CPU (not converted to numpy) so they can still be used as tensors.
        Custom objects like sv.Detections are converted to dicts with numpy arrays.
        """
        import torch
        import numpy as np

        # 0. Preserve exceptions so RPC callers can re-raise the real failure.
        if isinstance(obj, BaseException):
            return obj

        # 1. Handle None and primitives first
        if obj is None:
            return None
        if isinstance(obj, (int, float, str, bool)):
            return obj

        # 2. Handle Tensors - move to CPU but keep as tensor
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu()

        # 3. Handle Numpy arrays
        elif isinstance(obj, np.ndarray):
            # If the array holds objects, one of them might be a Tensor!
            if obj.dtype == object:
                flat_cleaned = [self._move_to_cpu(x) for x in obj.flat]
                return np.array(flat_cleaned).reshape(obj.shape)
            return obj

        # 4. Handle Lists - recurse into each element
        elif isinstance(obj, list):
            return [self._move_to_cpu(item) for item in obj]

        # 5. Handle Tuples - recurse and return as tuple
        elif isinstance(obj, tuple):
            return tuple(self._move_to_cpu(item) for item in obj)

        # 6. Handle Dicts - recurse into values
        elif isinstance(obj, dict):
            return {k: self._move_to_cpu(v) for k, v in obj.items()}

        # 7. Handle sv.Detections and other custom objects with __dict__
        elif hasattr(obj, '__dict__'):
            try:
                import supervision as sv
                if isinstance(obj, sv.Detections):
                    def to_numpy(x):
                        if isinstance(x, torch.Tensor):
                            return x.detach().cpu().numpy()
                        return x
                    return {
                        'xyxy': to_numpy(obj.xyxy) if obj.xyxy is not None else None,
                        'mask': to_numpy(obj.mask) if obj.mask is not None else None,
                        'confidence': to_numpy(obj.confidence) if obj.confidence is not None else None,
                        'class_id': to_numpy(obj.class_id) if obj.class_id is not None else None,
                        'tracker_id': to_numpy(obj.tracker_id) if hasattr(obj, 'tracker_id') and obj.tracker_id is not None else None,
                        'data': self._move_to_cpu(dict(obj.data)) if hasattr(obj, 'data') and obj.data else {},
                        '_type': 'sv.Detections',
                    }
            except ImportError:
                pass

            return {k: self._move_to_cpu(v) for k, v in obj.__dict__.items()}

        else:
            return obj

    def _run_detect_worker(self, device, detect_stream, detection_model):
        """Process LLMDet requests sequentially on the detection CUDA stream."""
        import torch

        logging.info(f"[detect_worker] GPU {self._gpu_log_label()}: started (LLMDet)")


        while True:
            request = self.detect_queue.get()
            if request is None:
                break


            if self._check_gpu_dead(request):
                continue

            # GPU path: sequential on detect_stream
            result = None
            _t_handler = time.perf_counter()
            _t_recon = _t_gpu = _t_pack = 0.0
            _gpu_ms = 0.0
            try:
                raw_args, raw_kwargs = request.payload
                with torch.cuda.stream(detect_stream):
                    _t0 = time.perf_counter()
                    args, kwargs = self._reconstruct_args(raw_args, raw_kwargs, device)
                    _t_recon = time.perf_counter() - _t0
                    _ev_s, _ev_e = self._record_gpu_time(detect_stream)
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                        result = detection_model.detect_objects(*args, **kwargs)
                    detect_stream.synchronize()
                    _gpu_ms = self._gpu_elapsed_ms(_ev_s, _ev_e, detect_stream)
            except Exception as e:
                logging.error(f"[detect_worker] GPU {self._gpu_log_label()}: {e}")
                traceback.print_exc()
                if "device-side assert" in str(e) or "CUDA error" in str(e):
                    self._gpu_dead.set()
                    logging.critical(f"[detect_worker] GPU {self._gpu_log_label()}: CUDA context corrupted, marking GPU as dead")
                result = RuntimeError(f"GPU Detect Inference Failed: {e}")

            try:
                _t1 = time.perf_counter()
                packed, _ = _pack_result_shm(self._move_to_cpu(result))
                _t_pack = time.perf_counter() - _t1
                self.reply_queues_dict[request.reply_queue_id].put(packed)
            except Exception as e:
                logging.critical(f"[detect_worker] Serialization failed GPU {self._gpu_log_label()}: {e}")
                traceback.print_exc()
                self.reply_queues_dict[request.reply_queue_id].put(
                    RuntimeError(f"Serialization Error: {str(e)}")
                )

            if PROFILE_PIPELINE:
                self._perf.add(
                    "detect",
                    wall_s=time.perf_counter() - _t_handler,
                    gpu_ms=_gpu_ms,
                    recon_s=_t_recon,
                    pack_s=_t_pack,
                )

    def _run_sam_worker(self, device, sam_stream, sam2_predictor):
        """Thread: processes SAM2 predict requests on the SAM CUDA stream."""
        import torch
        logging.info(f"[sam_worker] GPU {self._gpu_log_label()}: started")

        while True:
            request = self.sam_queue.get()
            if request is None:
                break

            if self._check_gpu_dead(request):
                continue

            result = None
            _t_handler = time.perf_counter()
            _t_recon = _t_pack = 0.0
            _gpu_ms = 0.0
            try:
                raw_args, raw_kwargs = request.payload
                with torch.cuda.stream(sam_stream):
                    _t0 = time.perf_counter()
                    args, kwargs = self._reconstruct_args(raw_args, raw_kwargs, device)
                    _t_recon = time.perf_counter() - _t0
                    _ev_s, _ev_e = self._record_gpu_time(sam_stream)
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                        img = args[0]
                        sam2_predictor.set_image(img)
                        result = sam2_predictor.predict(**kwargs)
                    sam_stream.synchronize()
                    _gpu_ms = self._gpu_elapsed_ms(_ev_s, _ev_e, sam_stream)
            except Exception as e:
                logging.error(f"[sam_worker] GPU {self._gpu_log_label()}: {e}")
                traceback.print_exc()
                if "device-side assert" in str(e) or "CUDA error" in str(e):
                    self._gpu_dead.set()
                    logging.critical(f"[sam_worker] GPU {self._gpu_log_label()}: CUDA context corrupted, marking GPU as dead")
                result = RuntimeError(f"GPU SAM Inference Failed: {e}")

            try:
                _t1 = time.perf_counter()
                packed, _ = _pack_result_shm(self._move_to_cpu(result))
                _t_pack = time.perf_counter() - _t1
                self.reply_queues_dict[request.reply_queue_id].put(packed)
            except Exception as e:
                logging.critical(f"[sam_worker] Serialization failed GPU {self._gpu_log_label()}: {e}")
                traceback.print_exc()
                self.reply_queues_dict[request.reply_queue_id].put(
                    RuntimeError(f"Serialization Error: {str(e)}")
                )

            if PROFILE_PIPELINE:
                self._perf.add("sam", wall_s=time.perf_counter() - _t_handler,
                               gpu_ms=_gpu_ms, recon_s=_t_recon, pack_s=_t_pack)

    def _run_tracking_worker(self, device, slow_stream, tracking_model):
        """Thread: processes track_boxes requests on the slow CUDA stream."""
        import torch
        from sparc.perception.tracking_models import sample_tracking_points

        while True:
            request = self.tracking_queue.get()
            if request is None:
                break

            if self._check_gpu_dead(request):
                continue

            result = None
            _t_handler = time.perf_counter()
            _t_recon = _t_pack = 0.0
            _gpu_ms = 0.0
            try:
                raw_args, raw_kwargs = request.payload
                with torch.cuda.stream(slow_stream):
                    _t0 = time.perf_counter()
                    args, kwargs = self._reconstruct_args(raw_args, raw_kwargs, device)
                    _t_recon = time.perf_counter() - _t0
                    _ev_s, _ev_e = self._record_gpu_time(slow_stream)

                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                        if request.task_type == 'track_boxes':
                            frames_torch = _prepare_tracking_frames(args[0], device)
                            pred_boxes_first_frame = args[1]
                            pred_masks_first_frame = args[2] if len(args) > 2 else None
                            grid_size_bbox = kwargs.get('grid_size_bbox', 6)

                            point_samples = sample_tracking_points(
                                pred_boxes_first_frame,
                                pred_masks_first_frame,
                                grid_size_bbox,
                                device,
                                num_points=getattr(tracking_model, "sample_point_count", None),
                            )

                            if len(point_samples) == 0:
                                result = ([], [])
                            else:
                                point_counts = [point_sample.shape[0] for point_sample in point_samples]
                                query_points = torch.cat(point_samples, dim=0)  # [N, 2]

                                pred_tracks, pred_visibility = tracking_model.track_points(
                                    frames_torch, query_points, device, point_counts=point_counts
                                )

                                pred_tracks_per_box = []
                                pred_vis_per_box = []
                                start_idx = 0
                                split_count = getattr(tracking_model, "sample_point_count", None)
                                pad_to_sample_count = getattr(tracking_model, "pad_to_sample_point_count", False)
                                for point_sample in point_samples:
                                    point_count = split_count if pad_to_sample_count and split_count is not None else point_sample.shape[0]
                                    end_idx = start_idx + point_count
                                    pred_tracks_per_box.append(pred_tracks[0][:, start_idx:end_idx])
                                    pred_vis_per_box.append(pred_visibility[0][:, start_idx:end_idx])
                                    start_idx = end_idx

                                result = (pred_tracks_per_box, pred_vis_per_box)


                    slow_stream.synchronize()
                    _gpu_ms = self._gpu_elapsed_ms(_ev_s, _ev_e, slow_stream)

            except Exception as e:
                logging.error(f"[tracking_worker] GPU {self._gpu_log_label()}: {e}")
                traceback.print_exc()
                if "device-side assert" in str(e) or "CUDA error" in str(e):
                    self._gpu_dead.set()
                    logging.critical(f"[tracking_worker] GPU {self._gpu_log_label()}: CUDA context corrupted, marking GPU as dead")
                result = RuntimeError(f"GPU AllTracker Failed: {e}")

            try:
                _t1 = time.perf_counter()
                packed, _ = _pack_result_shm(self._move_to_cpu(result))
                _t_pack = time.perf_counter() - _t1
                self.reply_queues_dict[request.reply_queue_id].put(packed)
            except Exception as e:
                logging.critical(f"[tracking_worker] Serialization failed GPU {self._gpu_log_label()}: {e}")
                traceback.print_exc()
                self.reply_queues_dict[request.reply_queue_id].put(
                    RuntimeError(f"Serialization Error: {str(e)}")
                )

            if PROFILE_PIPELINE:
                self._perf.add("tracking", wall_s=time.perf_counter() - _t_handler,
                               gpu_ms=_gpu_ms, recon_s=_t_recon, pack_s=_t_pack)

    def _run_pointcloud_worker(self, device, pc_stream, pointcloud_model):
        """Thread: processes get_pointcloud requests via MoGe on a dedicated CUDA stream."""
        import torch
        logging.info(f"[pointcloud_worker] GPU {self._gpu_log_label()}: started")

        while True:
            request = self.pointcloud_queue.get()
            if request is None:
                break

            if self._check_gpu_dead(request):
                continue

            result = None
            _t_handler = time.perf_counter()
            _t_recon = _t_pack = 0.0
            _gpu_ms = 0.0
            try:
                raw_args, raw_kwargs = request.payload
                with torch.cuda.stream(pc_stream):
                    _t0 = time.perf_counter()
                    args, kwargs = self._reconstruct_args(raw_args, raw_kwargs, device)
                    _t_recon = time.perf_counter() - _t0
                    _ev_s, _ev_e = self._record_gpu_time(pc_stream)
                    with torch.inference_mode():
                        frame_tensor = args[0]  # (3, H, W) or (B, 3, H, W) float32 in [0, 1]
                        output = pointcloud_model.infer(frame_tensor, **kwargs)
                        result = {
                            'points': output['points'].cpu().numpy(),  # (H, W, 3) or (B, H, W, 3)
                            'mask': output['mask'].cpu().numpy(),      # (H, W) bool or (B, H, W) bool
                        }
                    pc_stream.synchronize()
                    _gpu_ms = self._gpu_elapsed_ms(_ev_s, _ev_e, pc_stream)
            except Exception as e:
                logging.error(f"[pointcloud_worker] GPU {self._gpu_log_label()}: {e}")
                traceback.print_exc()
                if "device-side assert" in str(e) or "CUDA error" in str(e):
                    self._gpu_dead.set()
                    logging.critical(f"[pointcloud_worker] GPU {self._gpu_log_label()}: CUDA context corrupted")
                result = RuntimeError(f"GPU PointCloud Inference Failed: {e}")

            try:
                _t1 = time.perf_counter()
                packed, _ = _pack_result_shm(self._move_to_cpu(result))
                _t_pack = time.perf_counter() - _t1
                self.reply_queues_dict[request.reply_queue_id].put(packed)
            except Exception as e:
                logging.critical(f"[pointcloud_worker] Serialization failed GPU {self._gpu_log_label()}: {e}")
                traceback.print_exc()
                self.reply_queues_dict[request.reply_queue_id].put(
                    RuntimeError(f"Serialization Error: {str(e)}")
                )

            if PROFILE_PIPELINE:
                # Capture batch dimension to show whether you're under-batching MoGe.
                _bsz = 0
                try:
                    ft = args[0]
                    _bsz = int(ft.shape[0]) if ft.dim() == 4 else 1
                except Exception:
                    pass
                self._perf.add("pointcloud", wall_s=time.perf_counter() - _t_handler,
                               gpu_ms=_gpu_ms, recon_s=_t_recon, pack_s=_t_pack,
                               batch_frames=_bsz)


    def run(self):
        self._gpu_dead = threading.Event()
        self._perf = PerfCounter(name=f"GPU-{self.gpu_label}")
        visible_devices = _get_visible_gpu_devices()
        if 0 <= self.gpu_id < len(visible_devices):
            self.gpu_label = visible_devices[self.gpu_id]


        import sys
        import torch
        # Ensure the sam2 package is importable as `sam2` (needed by its internal imports).
        _sam2_root = str(SAM2_PACKAGE_ROOT)
        if _sam2_root not in sys.path:
            sys.path.insert(0, _sam2_root)
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from sam2.build_sam import build_sam2
        from sparc.detectors.llmdet_hf import LLMDet

        logging.basicConfig(level=logging.INFO, format=f'[GPU-{self._gpu_log_label()}] %(asctime)s - %(message)s')
        logging.getLogger("sam2.sam2_image_predictor").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)
        torch.cuda.set_device(self.gpu_id)
        device = torch.device(f"cuda:{self.gpu_id}")

        try:
            from sparc.perception.tracking_models import (
                AllTrackerModel,
                load_alltracker_raw_model,
            )


            logging.info(f"GPU Server {self._gpu_log_label()}: Loading Models (tracking={self.tracking_model_name})...")
            tracking_models = []
            for _ in range(self.n_tracker_instances):
                alltracker_raw = load_alltracker_raw_model(device, tiny=False, window_len=16)
                tracking_models.append(
                    AllTrackerModel(
                        alltracker_raw,
                        inference_iters=4,
                        window_len=16,
                        use_sliding=True,
                    )
                )

            sam2cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
            sam2model = str(_resolve_sam2_checkpoint())
            sam2_base = build_sam2(sam2cfg, sam2model, device=device)
            sam2_predictor = SAM2ImagePredictor(sam2_base)
            detector_settings = self._detector_settings()
            logging.info(self._detection_startup_message(detector_settings))
            detection_model = LLMDet(
                model_id=detector_settings.get("model_id", "iSEE-Laboratory/llmdet_base"),
                box_threshold=float(detector_settings.get("box_threshold", 0.35)),
                text_threshold=float(detector_settings.get("text_threshold", 0.25)),
                use_slicing=bool(detector_settings.get("use_slicing", False)),
            )

            pointcloud_model = None
            if self.pointcloud_queue is not None:
                from moge.model.v2 import MoGeModel
                logging.info(f"GPU Server {self._gpu_log_label()}: Loading MoGe point cloud model...")
                pointcloud_model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitb-normal").to(device)
                pointcloud_model.eval()

            robotseg_predictor = None
            if self.robotseg_queue is not None:
                _vendor = str(REPOSITORY_ROOT / "vendors" / "robotseg" / "RobotSeg")
                if _vendor not in sys.path:
                    sys.path.insert(0, _vendor)
                # SAM2's __init__.py already called initialize_config_module("sam2"), so GlobalHydra
                # is initialized and robotseg's guard would skip registering pkg://robotseg.
                from hydra.core.global_hydra import GlobalHydra
                GlobalHydra.instance().clear()
                import robotseg  # noqa: F401 — triggers hydra.initialize_config_module inside the package
                from robotseg.build_robotseg import build_robotseg_video_predictor
                ckpt = str(REPOSITORY_ROOT / "checkpoints" / "robotseg" / "robotseg.pt")
                logging.info(f"GPU Server {self._gpu_log_label()}: Loading RobotSeg model (ckpt={ckpt})...")
                robotseg_predictor = build_robotseg_video_predictor("configs/robotseg-infer", ckpt, device=device)
                logging.info(f"GPU Server {self._gpu_log_label()}: RobotSeg loaded.")

            self.dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16

            detect_stream = torch.cuda.Stream(device=device)
            sam_stream = torch.cuda.Stream(device=device)
            pc_stream = torch.cuda.Stream(device=device) if pointcloud_model is not None else None
            rs_stream = torch.cuda.Stream(device=device) if robotseg_predictor is not None else None

            # One CUDA stream and model instance per tracking thread; all share tracking_queue
            tracker_instances = tracking_models
            tracker_count = len(tracker_instances)
            tracker_streams = [torch.cuda.Stream(device=device) for _ in range(tracker_count)]

            logging.info(f"GPU Server {self._gpu_log_label()}: Warming up SAM2 image predictor...")
            _dummy_img = np.zeros((64, 64, 3), dtype=np.uint8)
            _dummy_box = np.array([0, 0, 32, 32], dtype=np.float32)
            with torch.autocast("cuda", self.dtype):
                with torch.inference_mode():
                    sam2_predictor.set_image(_dummy_img)
                    sam2_predictor.predict(box=_dummy_box, multimask_output=False)
            torch.cuda.synchronize(device)


            t_detect = threading.Thread(
                target=self._run_detect_worker,
                args=(device, detect_stream, detection_model),
                daemon=True,
            )
            t_sam = threading.Thread(
                target=self._run_sam_worker,
                args=(device, sam_stream, sam2_predictor),
                daemon=True,
            )
            tracker_threads = [
                threading.Thread(
                    target=self._run_tracking_worker,
                    args=(device, tracker_streams[i], tracker_instances[i]),
                    daemon=True,
                )
                for i in range(tracker_count)
            ]
            t_pointcloud = None
            if pointcloud_model is not None:
                t_pointcloud = threading.Thread(
                    target=self._run_pointcloud_worker,
                    args=(device, pc_stream, pointcloud_model),
                    daemon=True,
                )
            t_robotseg = None
            if robotseg_predictor is not None:
                t_robotseg = threading.Thread(
                    target=self._run_robotseg_worker,
                    args=(device, rs_stream, robotseg_predictor),
                    daemon=True,
                )
            if PROFILE_PIPELINE:
                t_perf_logger = threading.Thread(target=self._periodic_perf_logger, daemon=True)
                t_perf_logger.start()
                logging.info(f"[GPU-{self._gpu_log_label()}] PROFILE_PIPELINE=1 — instrumentation enabled")
            t_detect.start()
            t_sam.start()
            for t in tracker_threads:
                t.start()
            if t_pointcloud is not None:
                t_pointcloud.start()
            if t_robotseg is not None:
                t_robotseg.start()

            pc_suffix = ", pointcloud" if pointcloud_model is not None else ""
            rs_suffix = ", robotseg" if robotseg_predictor is not None else ""
            logging.info(f"GPU Server {self._gpu_log_label()}: READY ({tracker_count} AllTracker instance(s), detect, sam{pc_suffix}{rs_suffix}).")
            self.server_ready_event.set()
            t_detect.join()
            t_sam.join()
            for t in tracker_threads:
                t.join()
            if t_pointcloud is not None:
                t_pointcloud.join()
            if t_robotseg is not None:
                t_robotseg.join()

        except Exception as e:
            logging.critical(f"GPU Server CRASHED: {e}")
            traceback.print_exc()
        finally:
            if PROFILE_PIPELINE and hasattr(self, "_perf"):
                logging.info("\n" + self._perf.summary(header=f"[GPU-{self._gpu_log_label()}] FINAL perf"))
                try:
                    peak_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024
                    reserved_mb = torch.cuda.max_memory_reserved(device) / 1024 / 1024
                    logging.info(
                        f"[GPU-{self._gpu_log_label()}] VRAM peak: "
                        f"allocated={peak_mb:.0f} MB, reserved={reserved_mb:.0f} MB"
                    )
                except Exception:
                    pass

    def _run_robotseg_worker(self, device, rs_stream, robotseg_predictor):
        """Thread: processes full-video gripper segmentation via RobotSeg on a dedicated CUDA stream."""
        import torch
        logging.info(f"[robotseg_worker] GPU {self._gpu_log_label()}: started")

        while True:
            request = self.robotseg_queue.get()
            if request is None:
                break

            if self._check_gpu_dead(request):
                continue

            result = None
            try:
                raw_args, raw_kwargs = request.payload
                args, kwargs = self._reconstruct_args_cpu(raw_args, raw_kwargs)
                frames_np = args[0]   # (T, H, W, 3) uint8
                category = args[1]    # str, e.g. "gripper"
                max_frames = args[2] if len(args) > 2 else None
                det_frame_idx = args[3] if len(args) > 3 else None
                requested_frame_indices = args[4] if len(args) > 4 else None
                T, H, W = frames_np.shape[:3]

                # Evenly subsample to at most max_frames, always including det_frame_idx.
                # If explicit frame indices are provided, use them as the source-of-truth
                # frame map so downstream track/mask alignment stays exact.
                if requested_frame_indices is not None:
                    frame_indices = np.asarray(requested_frame_indices, dtype=int).reshape(-1)
                    frame_indices = frame_indices[(frame_indices >= 0) & (frame_indices < T)]
                    frame_indices = np.unique(frame_indices)
                elif max_frames is not None and T > max_frames:
                    base = np.linspace(0, T - 1, max_frames, dtype=int)
                    if det_frame_idx is not None and det_frame_idx not in base:
                        # replace the nearest sampled index with det_frame_idx
                        nearest = int(np.argmin(np.abs(base - det_frame_idx)))
                        base[nearest] = det_frame_idx
                        base.sort()
                    frame_indices = base
                else:
                    frame_indices = np.arange(T, dtype=int)
                frames_to_run = frames_np[frame_indices]
                logging.info(f"[robotseg_worker] GPU {self._gpu_log_label()}: running on {len(frame_indices)}/{T} frames ({H}x{W})")

                with torch.cuda.stream(rs_stream):
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                        state = robotseg_predictor.init_state(
                            video_path=frames_to_run,
                            async_loading_frames=False,
                            offload_video_to_cpu=False,
                            offload_state_to_cpu=False,
                        )
                        robotseg_predictor.add_new_robot(
                            inference_state=state, frame_idx=0, obj_id=0, robot=category
                        )
                        frame_masks = {}
                        for fidx, _obj_ids, logits in robotseg_predictor.propagate_in_video(
                            inference_state=state, robot=category
                        ):
                            mask = (logits[0] > 0.0).squeeze().cpu().numpy()
                            frame_masks[fidx] = mask
                    rs_stream.synchronize()

                N = len(frame_indices)
                masks_out = np.zeros((N, H, W), dtype=np.float32)
                for fidx, m in frame_masks.items():
                    if fidx < N:
                        masks_out[fidx] = m.astype(np.float32)
                result = {'masks': masks_out, 'frame_indices': frame_indices}

            except Exception as e:
                logging.error(f"[robotseg_worker] GPU {self._gpu_log_label()}: {e}")
                traceback.print_exc()
                if "device-side assert" in str(e) or "CUDA error" in str(e):
                    self._gpu_dead.set()
                    logging.critical(f"[robotseg_worker] GPU {self._gpu_log_label()}: CUDA context corrupted")
                result = RuntimeError(f"GPU RobotSeg Inference Failed: {e}")

            try:
                packed, _ = _pack_result_shm(self._move_to_cpu(result))
                self.reply_queues_dict[request.reply_queue_id].put(packed)
            except Exception as e:
                logging.critical(f"[robotseg_worker] Serialization failed GPU {self._gpu_log_label()}: {e}")
                traceback.print_exc()
                self.reply_queues_dict[request.reply_queue_id].put(
                    RuntimeError(f"Serialization Error: {str(e)}")
                )


class ModelProxy:
    """
    Living in the CPU worker. Looks like a model, acts like a model.
    Sends mail to the GPU Server using SharedMemory for large arrays.
    """
    # Class-level shared counter so all proxies in one worker process aggregate together.
    _proxy_perf = PerfCounter(name="proxy") if PROFILE_PIPELINE else None
    _proxy_perf_lock = threading.Lock()
    _proxy_last_log = [0.0]  # mutable for class-method access

    def __init__(self, request_queue, reply_queue, worker_id, task_prefix):
        self.request_queue = request_queue
        self.reply_queue = reply_queue
        self.worker_id = worker_id
        self.task_prefix = task_prefix

    def _rpc(self, task_type, payload):
        shm_objs = [] # Keep track to close/unlink later
        _t_total = time.perf_counter() if PROFILE_PIPELINE else 0.0

        # Helper to process args
        def process_arg(arg):
            # Check if arg is a large numpy array or torch tensor
            is_large = False
            arr = None
            if isinstance(arg, np.ndarray) and arg.nbytes > 100_000: # >100KB
                is_large = True
                arr = arg
            elif hasattr(arg, 'numpy') and hasattr(arg, 'element_size'): # Torch Tensor
                if arg.element_size() * arg.numel() > 100_000:
                    arr = arg.cpu().numpy()
                    is_large = True

            if is_large and arr is not None:
                meta, shm = SharedTensorManager.send(arr)
                shm_objs.append(shm)
                return {'__shm__': meta}

            # Recursive check for lists
            if isinstance(arg, list):
                return [process_arg(x) for x in arg]

            return arg

        # Process payload (args, kwargs) to find arrays
        _t_pack0 = time.perf_counter() if PROFILE_PIPELINE else 0.0
        args, kwargs = payload
        new_args = tuple(process_arg(a) for a in args)
        new_kwargs = {k: process_arg(v) for k, v in kwargs.items()}
        final_payload = (new_args, new_kwargs)
        _t_pack = time.perf_counter() - _t_pack0 if PROFILE_PIPELINE else 0.0

        # Drain stale replies from any previous timed-out requests so they don't
        # contaminate the current call's reply (workers make calls sequentially,
        # so anything already in the queue at this point is a leftover).
        import queue as _queue
        while True:
            try:
                self.reply_queue.get_nowait()
            except _queue.Empty:
                break

        # Send Request
        req = ModelRequest(task_type, final_payload, self.worker_id)
        _t_put0 = time.perf_counter() if PROFILE_PIPELINE else 0.0
        self.request_queue.put(req)
        _t_put = time.perf_counter() - _t_put0 if PROFILE_PIPELINE else 0.0

        # Wait for Result — timeout prevents infinite hang if GPU server deadlocks
        _t_wait0 = time.perf_counter() if PROFILE_PIPELINE else 0.0
        try:
            result = self.reply_queue.get(timeout=600)
        except _queue.Empty:
            raise RuntimeError(
                f"GPU server did not respond within 600s for task '{task_type}' "
                f"(worker={self.worker_id}). GPU server may have deadlocked."
            )
        _t_wait = time.perf_counter() - _t_wait0 if PROFILE_PIPELINE else 0.0

        # CLEANUP: Unlink shared memory now that GPU is done reading
        for shm in shm_objs:
            try:
                shm.close()
                shm.unlink()
            except: pass

        if isinstance(result, Exception):
            raise result

        _t_unp0 = time.perf_counter() if PROFILE_PIPELINE else 0.0
        unpacked = _unpack_result_shm(result)
        _t_unp = time.perf_counter() - _t_unp0 if PROFILE_PIPELINE else 0.0

        if PROFILE_PIPELINE:
            ModelProxy._proxy_perf.add(
                task_type,
                wall_s=time.perf_counter() - _t_total,
                pack_s=_t_pack,
                put_s=_t_put,
                wait_s=_t_wait,     # = GPU work + queue wait + transit
                unpack_s=_t_unp,
            )
            # Periodic flush from any proxy (rate-limited per worker process)
            now = time.perf_counter()
            with ModelProxy._proxy_perf_lock:
                if now - ModelProxy._proxy_last_log[0] > 60.0:
                    ModelProxy._proxy_last_log[0] = now
                    logging.info("\n" + ModelProxy._proxy_perf.summary(
                        header=f"[Worker-{self.worker_id}] live proxy perf"))
        return unpacked

    def __call__(self, *args, **kwargs):
        """Handle calls like model(x)"""
        return self._rpc(self.task_prefix, (args, kwargs))


    def detect_objects(self, *args, **kwargs):
        """Specific method for detection calls."""
        return self._rpc('detect', (args, kwargs))

    # Allow attribute access to fool utils that check for .device
    def __getattr__(self, name):
        if name == 'device':
            import torch
            return torch.device('cpu')  # Return CPU since proxy runs on CPU worker
        if name == 'eval': return lambda: None
        if name == 'to': return lambda x: self
        raise AttributeError(f"Proxy has no attribute {name}")


class SAM2Proxy(ModelProxy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pending_image = None  # Cache for the image

    def set_image(self, image):
        # 1. Do NOT call RPC. Just store locally.
        self._pending_image = image
        return None

    def predict(self, **kwargs):
        if self._pending_image is None:
            raise RuntimeError("You must call set_image before calling predict!")

        # 2. Send ATOMIC request: (image, predict_kwargs)
        # We pass the image as the first positional arg, and predict params as kwargs
        result = self._rpc('sam_predict_atomic', ((self._pending_image,), kwargs))

        return result


class TrackerProxy(ModelProxy):
    """Send AllTracker point-tracking requests to the GPU server."""

    def run_on_boxes(self, frames_torch, pred_boxes_first_frame, pred_masks_first_frame=None, grid_size_bbox=6):
        """Sample and track points for each candidate box and optional mask."""
        result = self._rpc('track_boxes', (
            (frames_torch, pred_boxes_first_frame, pred_masks_first_frame),
            {'grid_size_bbox': grid_size_bbox}
        ))
        return result


class PointCloudProxy(ModelProxy):
    """Proxy for MoGe 3D point cloud prediction."""

    def get_pointcloud(self, frame_rgb: np.ndarray, fov_x: float = None) -> dict:
        """
        Predict a dense 3D point cloud from a single RGB frame.

        Args:
            frame_rgb: (H, W, 3) uint8 numpy array
            fov_x: horizontal field of view in degrees (dataset-specific; None lets MoGe estimate it)
        Returns:
            dict with:
                'points': (H, W, 3) float32 — 3D point map in camera space
                'mask':   (H, W) bool       — valid-depth mask
        """
        frame_f32 = (frame_rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)  # (3, H, W)
        kwargs = {} if fov_x is None else {'fov_x': fov_x}
        return self._rpc('get_pointcloud', ((frame_f32,), kwargs))

    def get_pointcloud_batch(self, frames_rgb: np.ndarray, fov_x: float = None, resolution_level: int = 9) -> dict:
        """
        Predict dense 3D point clouds for a batch of RGB frames.

        Args:
            frames_rgb: (B, H, W, 3) uint8 numpy array
            fov_x: horizontal field of view in degrees (scalar; None lets MoGe estimate it)
            resolution_level: MoGe resolution level 0-9 (controls internal num_tokens); lower
                values reduce the internal spatial resolution → higher safe batch sizes.
        Returns:
            dict with:
                'points': (B, H, W, 3) float32 — 3D point maps in camera space
                'mask':   (B, H, W) bool        — valid-depth masks
        """
        batch_f32 = (frames_rgb.astype(np.float32) / 255.0).transpose(0, 3, 1, 2)  # (B, 3, H, W)
        kwargs = {'resolution_level': resolution_level}
        if fov_x is not None:
            kwargs['fov_x'] = fov_x
        return self._rpc('get_pointcloud_batch', ((batch_f32,), kwargs))


class RobotSegProxy(ModelProxy):
    """Proxy for RobotSeg full-video gripper segmentation."""

    def segment_video(self, frames_np: np.ndarray, category: str = "robot",
                      max_frames: int = None, det_frame_idx: int = None,
                      frame_indices: np.ndarray | None = None) -> dict:
        """Segment gripper/robot through video frames.

        Args:
            frames_np: (T, H, W, 3) uint8 numpy array
            category: RobotSeg prompt — "gripper", "arm", or "robot"
            max_frames: if set, subsample to this many evenly-spaced frames
            det_frame_idx: if set, always included in the subsampled frames
            frame_indices: optional explicit frames_np indices to segment
        Returns:
            dict with 'masks' (N, H, W) float32 and 'frame_indices' (N,) int — original frame positions
        """
        return self._rpc(
            'robotseg_segment_video',
            ((frames_np, category, max_frames, det_frame_idx, frame_indices), {}),
        )
