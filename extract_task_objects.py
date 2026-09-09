#!/usr/bin/env python3
"""
High-Performance Parallel VLM Extractor
- Producers: Multiprocessing Workers (CPU bound)
- Consumer: Threaded Dispatcher (Network bound)
- Storage: Batched Atomic Writes (I/O bound)
"""

import os
import sys
import pickle
import multiprocessing as mp
import logging
import time
import random
import argparse
from pathlib import Path
from typing import Optional, Dict
from multiprocessing import Queue
from queue import Empty
from threading import Thread
import concurrent.futures

import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from sparc.pipeline.config import load_annotator_config, get_loader_class
from sparc.robot.keystate_utils import (
    get_gripper_close_phases,
    get_phase_sequence_key,
    load_or_compute_gripper_stats
)
from sparc.llm.llm_utils import (
    build_vllm_extra_body,
    extract_task_obj_semantic,
    extract_task_obj_with_phases,
)
from sparc.llm.llm_bimanual import extract_task_obj_bimanual
from sparc.llm.semantic_task_parsing import (
    TASK_OBJECT_PARSING_MODES,
    build_multiview_storyboard_jpeg,
    flatten_storyboard_frame_indices,
    resolve_task_object_parsing_mode,
    semantic_storyboard_samples,
    task_object_cache_key,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.getLogger("httpx").setLevel(logging.WARNING)

# --- CONFIGURATION (set from config in main()) ---
SAVE_BATCH_SIZE = 50
VLM_CONCURRENCY = 1024
# Populated from cfg.vllm at startup, read by vlm_network_worker threads
VLLM_CONFIG = {}


# --- HELPER FOR THREAD POOL ---
def vlm_network_worker(lang_ann, grasp_phases, cache_key,
                       left_phases=None, right_phases=None, is_bimanual=False,
                       parsing_mode="text", episode_samples=None, storyboards=None):
    """
    This function runs in a separate thread.
    It performs the blocking network call.
    """
    base_url = VLLM_CONFIG["base_url"]
    model = VLLM_CONFIG["model_name"]
    temperature = (
        VLLM_CONFIG["semantic_temperature"]
        if parsing_mode == "semantic"
        else VLLM_CONFIG["temperature"]
    )
    extra_body = VLLM_CONFIG.get("extra_body")

    try:
        if is_bimanual:
            res = extract_task_obj_bimanual(
                lang_ann,
                left_phases,
                right_phases,
                model=model,
                base_url=base_url,
                extra_body=extra_body,
            )
        elif parsing_mode == "semantic":
            res = extract_task_obj_semantic(
                lang_ann,
                episode_samples or [],
                storyboards or [],
                model=model,
                base_url=base_url,
                temperature=temperature,
                extra_body=extra_body,
            )
        else:
            res = extract_task_obj_with_phases(
                lang_ann,
                grasp_phases,
                model=model,
                base_url=base_url,
                temperature=temperature,
                extra_body=extra_body,
            )
        if res is None:
            raise ValueError("VLM returned None result")
        return (cache_key, res, True, None)
    except Exception as e:
        return (cache_key, None, False, str(e))


class TaskObjectExtractor:
    """Lightweight extractor running in worker processes."""
    def __init__(self, request_queue: Queue, local_cache: Dict, gripper_stats: Dict,
                 loader=None, parsing_config=None):
        self.request_queue = request_queue
        self.local_cache = local_cache
        self.local_cache_hits = 0
        self._cache_hit_flush_interval = 500
        self.gripper_stats = gripper_stats or {}
        self.loader = loader
        self.parsing_config = parsing_config or {"mode": "text"}
        self.parsing_mode = str(self.parsing_config.get("mode", "text")).lower()
        if self.parsing_mode not in TASK_OBJECT_PARSING_MODES:
            raise ValueError(
                f"Unknown extract_task_objects.mode={self.parsing_mode!r}; "
                f"expected one of {sorted(TASK_OBJECT_PARSING_MODES)}"
            )
        self.closed_threshold = 0.52
        self.hysteresis_offset = 0.05
        self._skip_no_gripper = 0
        self._skip_no_phases = 0
        self._vlm_sent = 0

    def stats_summary(self) -> str:
        return (
            f"cache_hits={self.local_cache_hits} vlm_sent={self._vlm_sent} "
            f"skip_no_gripper={self._skip_no_gripper} skip_no_phases={self._skip_no_phases}"
        )

    def _semantic_view_keys(self):
        semantic_config = self.parsing_config.get("semantic", {}) or {}
        configured = semantic_config.get("view_keys")
        available = list(getattr(self.loader, "_cam_keys", []) or [])
        primary = getattr(self.loader, "view_key", None)
        if configured:
            requested = [str(view) for view in configured]
        else:
            requested = [primary] if primary else []
            requested.extend(view for view in available if "wrist" in view.lower())
            if not requested:
                requested = ["selected_camera"]
        max_views = int(semantic_config.get("max_views", 2))
        result = []
        for view in requested:
            if not view or view in result or (available and view not in available):
                continue
            result.append(view)
            if len(result) >= max_views:
                break
        return result

    def _build_semantic_storyboards(self, trajectory_path, grasp_phases):
        samples = semantic_storyboard_samples(grasp_phases)
        frame_indices = flatten_storyboard_frame_indices(samples)
        semantic_config = self.parsing_config.get("semantic", {}) or {}
        frames_by_view = []
        fallback_frames = None
        for view_key in self._semantic_view_keys():
            try:
                if hasattr(self.loader, "load_frames_at_indices"):
                    frames = self.loader.load_frames_at_indices(
                        trajectory_path,
                        frame_indices,
                        view_key=view_key,
                    )
                else:
                    if fallback_frames is None:
                        visual_trajectory = self.loader.load_trajectory(
                            trajectory_path,
                            load_frames=True,
                            skip_frames=False,
                            skip_observations=True,
                        )
                        fallback_frames = visual_trajectory.frames
                    clipped = [
                        max(0, min(int(index), len(fallback_frames) - 1))
                        for index in frame_indices
                    ]
                    frames = np.asarray(fallback_frames)[clipped]
                frames_by_view.append((view_key, frames))
            except Exception:
                logging.exception(
                    "Could not build semantic storyboard for trajectory=%s view=%s",
                    trajectory_path,
                    view_key,
                )
        if not frames_by_view:
            raise ValueError(f"No semantic camera storyboards available for {trajectory_path}")

        jpeg_bytes = build_multiview_storyboard_jpeg(
            frames_by_view,
            samples,
            cell_width=int(semantic_config.get("cell_width", 320)),
            cell_height=int(semantic_config.get("cell_height", 200)),
            jpeg_quality=int(semantic_config.get("jpeg_quality", 85)),
        )
        view_labels = [view_key for view_key, _ in frames_by_view]
        storyboards = [{
            "view_key": f"combined_multiview[{', '.join(view_labels)}]",
            "jpeg_bytes": jpeg_bytes,
        }]
        return samples, storyboards

    def process(self, lang_ann, gripper_state, traj_name, trajectory_path=None,
                left_gripper_state=None, right_gripper_state=None):
        logging.basicConfig(level=logging.INFO)
        expected_dur = self.gripper_stats.get("avg_interact_duration_frames", None)
        kwargs = dict(
            closed_threshold=self.closed_threshold,
            expected_interact_duration=expected_dur,
            hysteresis_offset=self.hysteresis_offset,
        )

        is_bimanual = (left_gripper_state is not None and right_gripper_state is not None)

        if is_bimanual:
            left_phases = get_gripper_close_phases(left_gripper_state, **kwargs)
            right_phases = get_gripper_close_phases(right_gripper_state, **kwargs)
            if not left_phases and not right_phases:
                self._skip_no_phases += 1
                logging.warning(f"skip no_phases (bimanual): {traj_name!r} lang={lang_ann!r}")
                return
            # Mirror trajectory_annotator.py: single "interact"-only arm is treated as unused ("null")
            if len(left_phases) == 1 and left_phases[0][2] == "interact":
                left_phases[0] = (left_phases[0][0], left_phases[0][1], "null")
            if len(right_phases) == 1 and right_phases[0][2] == "interact":
                right_phases[0] = (right_phases[0][0], right_phases[0][1], "null")
            left_seq_key = get_phase_sequence_key(left_phases)
            right_seq_key = get_phase_sequence_key(right_phases)
            cache_key = ("bimanual", lang_ann.strip(), left_seq_key, right_seq_key)

            if cache_key in self.local_cache:
                self.local_cache_hits += 1
                if self.local_cache_hits % self._cache_hit_flush_interval == 0:
                    self.request_queue.put({"cache_hits": self._cache_hit_flush_interval})
                return

            self._vlm_sent += 1
            self.request_queue.put({
                "lang_ann": lang_ann,
                "left_phases": left_phases,
                "right_phases": right_phases,
                "cache_key": cache_key,
                "trajectory_name": traj_name,
                "is_bimanual": True,
            })
        else:
            if gripper_state is None:
                self._skip_no_gripper += 1
                logging.warning(f"skip no_gripper: {traj_name!r}")
                return
            # CPU Work
            grasp_phases = get_gripper_close_phases(gripper_state, **kwargs)
            if grasp_phases is None or len(grasp_phases) == 0:
                self._skip_no_phases += 1
                logging.warning(f"skip no_phases: {traj_name!r} lang={lang_ann!r}")
                return

            effective_mode = resolve_task_object_parsing_mode(
                self.parsing_mode,
                grasp_phases,
            )
            phase_key = get_phase_sequence_key(grasp_phases)
            cache_key = task_object_cache_key(
                effective_mode,
                traj_name,
                lang_ann,
                phase_key,
            )

            if cache_key in self.local_cache:
                self.local_cache_hits += 1
                if self.local_cache_hits % self._cache_hit_flush_interval == 0:
                    self.request_queue.put({"cache_hits": self._cache_hit_flush_interval})
                return

            episode_samples = None
            storyboards = None
            if effective_mode == "semantic":
                episode_samples, storyboards = self._build_semantic_storyboards(
                    trajectory_path or traj_name,
                    grasp_phases,
                )

            self._vlm_sent += 1
            self.request_queue.put({
                "lang_ann": lang_ann,
                "grasp_phases": grasp_phases,
                "cache_key": cache_key,
                "trajectory_name": traj_name,
                "is_bimanual": False,
                "parsing_mode": effective_mode,
                "episode_samples": episode_samples,
                "storyboards": storyboards,
            })


def vlm_manager_thread(
    request_queue: Queue,
    task_obj_dict_path: str,
    total_trajectories: Optional[int] = 0,
    max_pending_requests: Optional[int] = None,
):
    """
    The High-Performance Dispatcher.
    Manages the ThreadPool for VLM calls and handles all File I/O.
    """
    pending_limit = (
        int(max_pending_requests)
        if max_pending_requests is not None and int(max_pending_requests) > 0
        else None
    )
    logging.info(
        "VLM Manager: Started with %s concurrent threads and max_pending_requests=%s.",
        VLM_CONCURRENCY,
        pending_limit if pending_limit is not None else "unlimited",
    )

    # 1. Load Database
    db_path = Path(task_obj_dict_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if db_path.exists() and db_path.stat().st_size > 0:
        try:
            with open(db_path, "rb") as f:
                full_db = pickle.load(f)
        except:
            full_db = {}
    else:
        full_db = {}

    # 2. Setup Concurrency
    pending_keys = set()
    pending_saves = 0

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=VLM_CONCURRENCY)

    import queue
    completed_queue = queue.Queue()

    def on_future_done(future):
        completed_queue.put(future.result())

    shutdown_signal_received = False

    pb = tqdm(total=total_trajectories, desc="VLM Manager Progress", unit="traj")

    while True:
        # --- A. DRAIN COMPLETED RESULTS (Write Phase) ---
        while not completed_queue.empty():
            key, result, success, err = completed_queue.get()

            if key in pending_keys:
                pending_keys.remove(key)

            if success:
                full_db[key] = result
                pending_saves += 1
                logging.info(f"\033[92mSaved task object for key: {key}\033[0m")
                pb.update(1)
            else:
                logging.warning(f"VLM Request Failed: {err}")
                pb.update(1)

        # Batch Save
        if pending_saves >= SAVE_BATCH_SIZE:
            _atomic_save(full_db, db_path)
            logging.info(f"VLM Manager: Saved {pending_saves} items. DB Size: {len(full_db)}. In-flight: {len(pending_keys)}")
            pending_saves = 0

        # --- B. DISPATCH NEW REQUESTS (Network Phase) ---
        if not shutdown_signal_received:
            try:
                dispatch_budget = 20
                if pending_limit is not None:
                    dispatch_budget = min(
                        dispatch_budget,
                        max(0, pending_limit - len(pending_keys)),
                    )

                for _ in range(dispatch_budget):
                    req = request_queue.get_nowait()

                    #print len of queue
                    #logging.info(f"VLM Manager: Received new request. Queue size: {request_queue.qsize()}. In-flight: {len(pending_keys)}")

                    if req is None:
                        shutdown_signal_received = True
                        logging.info("VLM Manager: Shutdown signal received. Waiting for pending requests...")
                        break

                    if req.get('cache_hits'):
                        pb.update(req['cache_hits'])
                        continue

                    key = req['cache_key']

                    # Deduplication: Check DB AND In-Flight
                    if key in full_db or key in pending_keys:
                        #logging.info(f"VLM Manager: Duplicate request for key {key}. Skipping.")
                        pb.update(1)
                        continue

                    # Submit to Thread Pool
                    pending_keys.add(key)
                    future = executor.submit(
                        vlm_network_worker,
                        req['lang_ann'],
                        req.get('grasp_phases'),
                        key,
                        req.get('left_phases'),
                        req.get('right_phases'),
                        req.get('is_bimanual', False),
                        req.get('parsing_mode', 'text'),
                        req.get('episode_samples'),
                        req.get('storyboards'),
                    )
                    future.add_done_callback(on_future_done)

            except Empty:
                pass

        # --- C. SHUTDOWN LOGIC ---
        if shutdown_signal_received and len(pending_keys) == 0 and completed_queue.empty():
            break

        time.sleep(0.01)

    # Final Save
    if pending_saves > 0:
        _atomic_save(full_db, db_path)

    executor.shutdown()
    logging.info("VLM Manager: Shutdown complete.")


def _atomic_save(data, path):
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(data, f)
    os.replace(tmp_path, path)


def process_chunk_wrapper(args):
    """Worker: Reads files, checks local cache, pushes to queue."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    chunk, config, LoaderCls, root, ds_name, worker_idx = args
    startup_stagger_sec = float(config.get('startup_stagger_sec', 0.5))
    startup_delay = worker_idx * startup_stagger_sec

    logging.info(
        f"Worker {worker_idx} [pid={os.getpid()}]: starting with {len(chunk)} trajectories "
        f"(startup_delay={startup_delay:.2f}s)"
    )

    if startup_delay > 0:
        time.sleep(startup_delay)

    # Load Read-Only Cache (Snapshot at startup)
    initial_cache = {}
    if os.path.exists(config['dict_path']):
        try:
            with open(config['dict_path'], "rb") as f:
                initial_cache = pickle.load(f)
        except Exception as e:
            logging.warning(f"Worker {worker_idx}: could not read cache snapshot: {e}")

    loader_kwargs = config.get('loader_kwargs', {})
    logging.info(f"Worker {worker_idx}: initializing loader {LoaderCls.__name__}")
    loader = LoaderCls(root, dataset_name=ds_name, **loader_kwargs)
    extractor = TaskObjectExtractor(
        config['queue'],
        initial_cache,
        config['stats'],
        loader=loader,
        parsing_config=config.get('parsing_config'),
    )
    logging.info(f"Worker {worker_idx}: loader ready, beginning trajectory loop")

    cnt = 0
    for path in chunk:
        try:
            traj = loader.load_trajectory(path, skip_frames=True, skip_observations=False)
            if traj:
                if cnt == 0:
                    logging.info(f"Worker {worker_idx}: first trajectory loaded successfully: {traj.name}")
                extractor.process(
                    traj.lang_ann,
                    traj.gripper_state,
                    traj.name,
                    trajectory_path=path,
                    left_gripper_state=getattr(traj, 'left_gripper_state', None),
                    right_gripper_state=getattr(traj, 'right_gripper_state', None),
                )
                cnt += 1
                if cnt % 500 == 0:
                    logging.info(
                        f"Worker {worker_idx}: processed {cnt} trajectories | {extractor.stats_summary()}"
                    )
                del traj
        except Exception:
            logging.exception(f"Worker {worker_idx}: failed while processing trajectory {path}")

    remainder = extractor.local_cache_hits % extractor._cache_hit_flush_interval
    if remainder > 0:
        config['queue'].put({"cache_hits": remainder})

    logging.info(f"Worker {worker_idx}: finished | total={cnt} | {extractor.stats_summary()}")
    return cnt


def _check_vllm_reachable(base_url: str):
    """Verify the VLLM server is reachable before spawning workers."""
    import requests
    try:
        resp = requests.get(f"{base_url}/models", timeout=10)
        resp.raise_for_status()
        logging.info(f"VLLM server reachable at {base_url}")
    except Exception as e:
        logging.error(f"VLLM server not reachable at {base_url}: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Extract task objects from trajectories using VLM.",
        epilog="Examples:\n"
               "  python extract_task_objects.py --dataset bridge_lerobot dataset.root=/path/to/bridge\n"
               "  python extract_task_objects.py --dataset droid_lerobot dataset.root=/path/to/droid extract_task_objects.mode=auto\n"
               "  python extract_task_objects.py --dataset agibot_world dataset.root=/path/to/agibot extract_task_objects.n_workers=16\n"
               "  python extract_task_objects.py --config /path/to/config.yaml\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a resolved YAML config file.")
    parser.add_argument("--dataset", type=str, default="agibot_world",
                        help="Dataset name (loads configs/dataset/{name}.yaml).")
    parser.add_argument("overrides", nargs="*",
                        help="OmegaConf dot-list overrides, e.g. extract_task_objects.n_workers=16")
    cli_args = parser.parse_args()

    if not cli_args.config and not cli_args.dataset:
        parser.error("Provide --config or --dataset")

    cfg = load_annotator_config(
        config_path=cli_args.config,
        dataset=cli_args.dataset,
        overrides=cli_args.overrides,
    )

    # --- Extract config values ---
    DATASET_NAME = cfg.dataset.name
    DATASET_TYPE = cfg.dataset.type
    ROOT_PATH = cfg.dataset.root
    loader_kwargs_cfg = cfg.dataset.get("loader_kwargs")
    LOADER_KWARGS = (
        OmegaConf.to_container(loader_kwargs_cfg, resolve=True)
        if loader_kwargs_cfg is not None
        else {}
    )
    TASK_OBJ_DICT_PATH = cfg.dataset.task_obj_dict_path

    eto_cfg = cfg.extract_task_objects
    TASK_OBJECT_MODE = str(eto_cfg.get("mode", "text")).lower()
    if TASK_OBJECT_MODE not in TASK_OBJECT_PARSING_MODES:
        raise ValueError(
            f"Unknown extract_task_objects.mode={TASK_OBJECT_MODE!r}; "
            f"expected one of {sorted(TASK_OBJECT_PARSING_MODES)}"
        )
    semantic_cfg = eto_cfg.get("semantic", {}) or {}
    PARSING_CONFIG = {
        "mode": TASK_OBJECT_MODE,
        "semantic": OmegaConf.to_container(semantic_cfg, resolve=True)
        if OmegaConf.is_config(semantic_cfg)
        else dict(semantic_cfg),
    }
    N_WORKERS = int(eto_cfg.n_workers)
    if TASK_OBJECT_MODE in {"auto", "semantic"}:
        N_WORKERS = min(N_WORKERS, int(semantic_cfg.get("max_cpu_workers", 32)))
    MAX_TRAJECTORIES = eto_cfg.get("max_trajectories", None)
    SHUFFLE_SEED = eto_cfg.get("shuffle_seed", None)
    COMPUTE_GRIPPER_STATS = eto_cfg.get("compute_gripper_stats", False)
    GRIPPER_STATS_SAMPLES = eto_cfg.get("gripper_stats_samples", 200)
    MP_START_METHOD = eto_cfg.get("mp_start_method", "spawn")
    WORKER_STARTUP_STAGGER_SEC = float(eto_cfg.get("worker_startup_stagger_sec", 0.5))

    global VLM_CONCURRENCY
    VLM_CONCURRENCY = int(eto_cfg.get("vlm_concurrency", 1024))
    if TASK_OBJECT_MODE in {"auto", "semantic"}:
        VLM_CONCURRENCY = min(
            VLM_CONCURRENCY,
            int(semantic_cfg.get("max_vlm_concurrency", 64)),
        )

    logging.info(f"Configuration: dataset={DATASET_NAME} type={DATASET_TYPE} root={ROOT_PATH}")
    logging.info(f"Extract Task Objects Config: mode={TASK_OBJECT_MODE} n_workers={N_WORKERS} max_trajectories={MAX_TRAJECTORIES} "
                 f"shuffle_seed={SHUFFLE_SEED} "
                 f"compute_gripper_stats={COMPUTE_GRIPPER_STATS} gripper_stats_samples={GRIPPER_STATS_SAMPLES} "
                 f"mp_start_method={MP_START_METHOD} worker_startup_stagger_sec={WORKER_STARTUP_STAGGER_SEC} "
                 f"VLLM_CONCURRENCY={VLM_CONCURRENCY}"
    )

    # --- VLLM config ---
    stage_temp = eto_cfg.get("temperature", None)
    vllm_temperature = stage_temp if stage_temp is not None else cfg.vllm.temperature
    semantic_temperature = vllm_temperature
    if stage_temp is None and semantic_cfg.get("temperature") is not None:
        semantic_temperature = float(semantic_cfg.get("temperature"))

    global VLLM_CONFIG
    vllm_dict = {
        "base_url": cfg.vllm.base_url,
        "model_name": cfg.vllm.model_name,
        "enable_thinking": cfg.vllm.get("enable_thinking"),
        "repetition_penalty": cfg.vllm.get("repetition_penalty"),
        "top_k": cfg.vllm.get("top_k"),
        "presence_penalty": cfg.vllm.get("presence_penalty"),
    }
    VLLM_CONFIG = {
        "base_url": vllm_dict["base_url"],
        "model_name": vllm_dict["model_name"],
        "temperature": vllm_temperature,
        "semantic_temperature": semantic_temperature,
        "extra_body": build_vllm_extra_body(vllm_dict),
    }

    logging.info(f"VLLM config: {VLLM_CONFIG}")

    # --- Check VLLM server reachability ---
    _check_vllm_reachable(VLLM_CONFIG["base_url"])

    # --- Dataset Setup ---
    loader_cls = get_loader_class(DATASET_TYPE)

    loader_root = ROOT_PATH
    if DATASET_TYPE == "oxe":
        oxe_ds_name = "_".join(DATASET_NAME.split("_")[1:])
        loader_root = os.path.join(ROOT_PATH, oxe_ds_name)
        logging.info(f"Loading OXE dataset from {loader_root}")

    dataset_loader = loader_cls(loader_root, dataset_name=DATASET_NAME, **LOADER_KWARGS)

    # --- Gripper Stats ---
    gripper_stats = {}
    if COMPUTE_GRIPPER_STATS:
        logging.info("Computing gripper statistics...")
        gripper_stats = load_or_compute_gripper_stats(
            dataset_root=ROOT_PATH,
            dataloader=dataset_loader,
            trajectory_paths=dataset_loader.trajectories,
            num_samples=GRIPPER_STATS_SAMPLES
        )

    # --- Trajectories ---
    logging.info("Getting trajectory paths...")
    traj_paths = list(dataset_loader.trajectories)
    selection_rng = random.Random(
        int(SHUFFLE_SEED) if SHUFFLE_SEED is not None else None
    )
    selection_rng.shuffle(traj_paths)
    if MAX_TRAJECTORIES:
        traj_paths = traj_paths[:int(MAX_TRAJECTORIES)]
    logging.info(f"Found {len(traj_paths)} trajectories")

    # --- Multiprocessing Setup ---
    mp_context = mp.get_context(MP_START_METHOD)
    logging.info(
        f"Using multiprocessing start method '{MP_START_METHOD}' "
        f"with worker_startup_stagger_sec={WORKER_STARTUP_STAGGER_SEC}"
    )

    # --- Manager Setup ---
    manager = mp_context.Manager()
    queue_capacity = (
        int(semantic_cfg.get("max_pending_requests", 128))
        if TASK_OBJECT_MODE in {"auto", "semantic"}
        else 0
    )
    request_queue = manager.Queue(maxsize=queue_capacity)

    # Start the Async Dispatcher Thread
    vlm_thread = Thread(
        target=vlm_manager_thread,
        args=(request_queue, TASK_OBJ_DICT_PATH, len(traj_paths), queue_capacity),
        daemon=True
    )
    vlm_thread.start()

    # --- Workers Setup ---
    worker_config = {
        "dict_path": TASK_OBJ_DICT_PATH,
        "queue": request_queue,
        "stats": gripper_stats,
        "startup_stagger_sec": WORKER_STARTUP_STAGGER_SEC,
        "parsing_config": PARSING_CONFIG,
    }

    chunks = [traj_paths[i::N_WORKERS] for i in range(N_WORKERS)]
    chunks = [c for c in chunks if c]

    # Pass n_workers=1 to multi-task loaders inside workers: cache is already warm
    # from the main-process loader, so sequential sub-loader creation is fine and
    # avoids the 16-process × 16-thread thundering herd on the cache JSON files.
    import inspect
    loader_kwargs = dict(LOADER_KWARGS)
    if 'n_workers' in inspect.signature(loader_cls.__init__).parameters and 'n_workers' not in loader_kwargs:
        loader_kwargs['n_workers'] = 1
    worker_config['loader_kwargs'] = loader_kwargs

    process_args = [(c, worker_config, loader_cls, loader_root, DATASET_NAME, i) for i, c in enumerate(chunks)]

    logging.info(f"Starting processing with {N_WORKERS} CPU workers and {VLM_CONCURRENCY} VLM threads...")

    # --- Run Workers ---
    total_files = 0
    with mp_context.Pool(len(chunks)) as pool:
        for count in tqdm(pool.imap_unordered(process_chunk_wrapper, process_args), total=len(chunks)):
            total_files += count

    logging.info("Workers finished. Waiting for VLM Manager to finish pending requests...")

    # Shutdown Manager
    request_queue.put(None)
    vlm_thread.join()

    logging.info("Processing Complete.")


if __name__ == "__main__":
    main()
