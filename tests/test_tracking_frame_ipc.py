import numpy as np
import torch
from contextlib import nullcontext
from queue import Queue
from threading import Event
from types import SimpleNamespace

from sparc.perception.bbox_ops import frames_to_tensor
from sparc.pipeline.gpu_inference import (
    GPUInferenceServer,
    SharedTensorManager,
    TrackerProxy,
    _prepare_tracking_frames,
)
from sparc.pipeline.trajectory_annotator import TrajectoryAnnotator


def test_tracker_proxy_round_trip_preserves_candidate_tracks(monkeypatch):
    """Exercise the actual RPC dispatch/splitting with a deterministic CPU model."""
    work_queue, replies = Queue(), Queue()
    server = GPUInferenceServer(0, None, None, work_queue, {"worker": replies}, Event())
    server._gpu_dead = Event()
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(torch, "autocast", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(server, "_record_gpu_time", lambda stream: (None, None))
    monkeypatch.setattr(server, "_gpu_elapsed_ms", lambda *args: 0.0)
    stream = SimpleNamespace(synchronize=lambda: None)
    captured = {}

    class DeterministicTracker:
        sample_point_count = 4

        def track_points(self, frames, queries, device, point_counts=None):
            captured["queries"] = queries.clone()
            captured["counts"] = point_counts
            offsets = torch.arange(frames.shape[1]).reshape(1, -1, 1, 1)
            tracks = queries[None, None] + offsets
            return tracks, torch.ones(tracks.shape[:-1])

    class DispatchQueue:
        def put(self, request):
            work_queue.put(request)
            work_queue.put(None)
            server._run_tracking_worker(torch.device("cpu"), stream, DeterministicTracker())

    proxy = TrackerProxy(DispatchQueue(), replies, "worker", "tracking")
    frames = torch.zeros((1, 3, 3, 16, 16), dtype=torch.uint8)
    boxes = np.array([[1, 1, 7, 7], [8, 8, 14, 14]], dtype=np.float32)
    tracks, visibility = proxy.run_on_boxes(frames, boxes, grid_size_bbox=2)

    assert captured["counts"] == [4, 4]
    assert len(tracks) == len(visibility) == 2
    for index, candidate in enumerate(tracks):
        expected = captured["queries"][index * 4:(index + 1) * 4][None] + torch.arange(3)[:, None, None]
        torch.testing.assert_close(candidate, expected)
        torch.testing.assert_close(visibility[index], torch.ones((3, 4)))


def test_uint8_video_stays_compact_until_tracking_device_conversion():
    frames = np.arange(3 * 4 * 5 * 3, dtype=np.uint8).reshape(3, 4, 5, 3)

    compact = frames_to_tensor(frames)

    assert compact.shape == (1, 3, 3, 4, 5)
    assert compact.dtype == torch.uint8
    assert compact.element_size() * compact.numel() == frames.nbytes

    prepared = _prepare_tracking_frames(compact, torch.device("cpu"))

    assert prepared.dtype == torch.float32
    np.testing.assert_array_equal(
        prepared[0].permute(0, 2, 3, 1).numpy(),
        frames.astype(np.float32),
    )
    assert prepared.element_size() == compact.element_size() * 4


def test_existing_float_tracking_frames_are_not_copied_or_renormalized():
    frames = torch.tensor([0.0, 127.5, 255.0], dtype=torch.float32)

    prepared = _prepare_tracking_frames(frames, torch.device("cpu"))

    assert prepared.data_ptr() == frames.data_ptr()
    torch.testing.assert_close(prepared, frames)


def test_server_reconstructs_from_shared_memory_without_retaining_buffer_alias():
    source = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
    metadata, shared_memory = SharedTensorManager.send(source)
    server = GPUInferenceServer.__new__(GPUInferenceServer)

    try:
        args, kwargs = server._reconstruct_args(
            ({"__shm__": metadata},),
            {},
            torch.device("cpu"),
        )
    finally:
        shared_memory.close()
        shared_memory.unlink()

    assert kwargs == {}
    assert args[0].dtype == torch.uint8
    np.testing.assert_array_equal(args[0].numpy(), source)


def test_optional_cpu_tracking_resize_preserves_float_interpolation_path():
    annotator = TrajectoryAnnotator.__new__(TrajectoryAnnotator)
    annotator.tracking_target_fps = None
    annotator.tracking_max_frames = None
    annotator.tracking_max_resolution = 2
    annotator.fps = 10.0
    frames = frames_to_tensor(np.zeros((3, 4, 6, 3), dtype=np.uint8))

    resized, _, _, _, tracking_scale, _ = annotator._subsample_frames_for_tracking(
        frames,
        det_frame_idx=0,
        task_obj_info={},
        start_frame=0,
    )

    assert resized.shape == (1, 3, 3, 2, 3)
    assert resized.dtype == torch.float32
    assert tracking_scale == (0.5, 0.5)
