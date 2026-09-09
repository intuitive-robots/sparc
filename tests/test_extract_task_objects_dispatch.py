import pickle
import queue
import threading
import time

import extract_task_objects as extractor


def test_vlm_manager_caps_submitted_futures(monkeypatch, tmp_path):
    request_queue = queue.Queue()
    for index in range(5):
        request_queue.put(
            {
                "lang_ann": f"task {index}",
                "grasp_phases": [],
                "cache_key": ("semantic_v9", f"trajectory_{index}"),
                "parsing_mode": "semantic",
            }
        )
    request_queue.put(None)

    release_requests = threading.Event()
    first_batch_started = threading.Event()
    state_lock = threading.Lock()
    state = {"active": 0, "max_active": 0, "started": 0}

    def blocking_vlm_worker(*args, **kwargs):
        cache_key = args[2]
        with state_lock:
            state["active"] += 1
            state["started"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
            if state["started"] >= 2:
                first_batch_started.set()
        release_requests.wait(timeout=5)
        with state_lock:
            state["active"] -= 1
        return cache_key, {"object": "test"}, True, None

    monkeypatch.setattr(extractor, "VLM_CONCURRENCY", 4)
    monkeypatch.setattr(extractor, "vlm_network_worker", blocking_vlm_worker)

    output_path = tmp_path / "task_objects.pkl"
    manager = threading.Thread(
        target=extractor.vlm_manager_thread,
        args=(request_queue, str(output_path), 5, 2),
    )
    manager.start()

    try:
        assert first_batch_started.wait(timeout=5)
        time.sleep(0.1)
        with state_lock:
            assert state["started"] == 2
            assert state["max_active"] == 2
    finally:
        release_requests.set()
        manager.join(timeout=10)

    assert not manager.is_alive()
    with output_path.open("rb") as handle:
        saved = pickle.load(handle)
    assert len(saved) == 5
    assert state["max_active"] <= 2
