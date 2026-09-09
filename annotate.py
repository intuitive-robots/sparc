"""Entry point for the annotation pipeline.

Imports infrastructure from the `sparc` package. Contains process_chunk_wrapper,
CLI arg parsing, and the __main__ block.

Usage:
    python annotate.py --dataset agibot_world
    python annotate.py --dataset bridge_lerobot dataset.root=/path/to/bridge annotator.gpu_ids=[0,1]
    python annotate.py --config /path/to/resolved_config.yaml
"""

import argparse
import logging
import json
import os
import random
import time
import traceback
import pickle
from pathlib import Path
from multiprocessing import Manager, Process

# --- Imports from the sparc package ---
from sparc.pipeline.gpu_inference import (
    GPUInferenceServer,
    ModelProxy,
    SAM2Proxy,
    TrackerProxy,
    PointCloudProxy,
    RobotSegProxy,
)
from sparc.pipeline.annotation_saver import AnnotationSaver
from sparc.pipeline.publication_schema import (
    subtask_index_from_annotation,
    trajectory_id_from_annotation,
)
from sparc.pipeline.trajectory_annotator import TrajectoryAnnotator
from sparc.pipeline.config import load_annotator_config, load_config_from_starvla, get_loader_class
from sparc.perception.ann_utils import (
    DEFAULT_DETECTION_TEXT_THRESHOLD,
    STANDARD_DETECTION_TEXT_THRESHOLD,
    TOOL_OBJECT_DETECTION_THRESHOLD,
)


# --- Process Wrapper ---


def _get_visible_gpu_devices():
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible_devices:
        return []
    return [dev.strip() for dev in visible_devices.split(",") if dev.strip()]


def _physical_gpu_label(local_gpu_id):
    visible_devices = _get_visible_gpu_devices()
    if 0 <= local_gpu_id < len(visible_devices):
        return visible_devices[local_gpu_id]
    return str(local_gpu_id)


def _node_output_files(base_output_file):
    """Return node JSONL files for the configured output stem."""
    base_path = Path(base_output_file)
    return sorted(
        path for path in base_path.parent.glob(f"{base_path.stem}_node*.jsonl")
        if path != base_path
    )


def get_all_processed_trajectories(base_output_file):
    """Read the main output and node files, returning processed trajectory names."""
    processed = set()
    base_path = Path(base_output_file)

    if base_path.exists():
        with open(base_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    trajectory_id = trajectory_id_from_annotation(data)
                    if trajectory_id is not None and "error" not in data:
                        processed.add(trajectory_id)
                except json.JSONDecodeError:
                    continue

    for node_file in _node_output_files(base_output_file):
        logging.info(f"Found node file: {node_file}")
        with open(node_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    trajectory_id = trajectory_id_from_annotation(data)
                    if trajectory_id is not None and "error" not in data:
                        processed.add(trajectory_id)
                except json.JSONDecodeError:
                    continue

    return processed


def _annotation_merge_key(annotation):
    trajectory_id = trajectory_id_from_annotation(annotation)
    if not trajectory_id:
        return None
    window = annotation.get('window') or {}
    task = annotation.get('task') or {}
    return (
        trajectory_id,
        subtask_index_from_annotation(annotation),
        window.get('start_frame'),
        window.get('end_frame_exclusive'),
        task.get('instruction'),
    )


def merge_node_files(base_output_file):
    """Merge node-specific JSONL files into the main output file."""
    base_path = Path(base_output_file)
    node_files = _node_output_files(base_output_file)

    if not node_files:
        logging.info(f"No node files found to merge for {base_path}")
        return

    logging.info(f"Found {len(node_files)} node files to merge")
    base_path.parent.mkdir(parents=True, exist_ok=True)

    existing_annotations = set()
    if base_path.exists():
        with open(base_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    key = _annotation_merge_key(data)
                    if key is not None:
                        existing_annotations.add(key)
                except json.JSONDecodeError:
                    continue

    merged_count = 0
    with open(base_path, 'a') as out_f:
        for node_file in node_files:
            with open(node_file, 'r') as in_f:
                for line in in_f:
                    try:
                        data = json.loads(line)
                        key = _annotation_merge_key(data)
                        if key is not None and key not in existing_annotations:
                            out_f.write(line if line.endswith('\n') else line + '\n')
                            existing_annotations.add(key)
                            merged_count += 1
                    except json.JSONDecodeError:
                        continue

    logging.info(f"Merged {merged_count} new annotations into {base_path}")


def _wait_for_merge_done(done_file, started_at, timeout_s=1800, poll_s=2):
    """Wait for rank 0 to finish the startup merge before reading processed rows."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if done_file.exists() and done_file.stat().st_mtime >= started_at:
                return
        except OSError:
            pass
        time.sleep(poll_s)
    raise TimeoutError(f"Timed out waiting for startup merge marker: {done_file}")


def process_chunk_wrapper(args):
    """
    CPU Worker. Runs logic, asks GPU Server for model outputs.
    """
    (trajectory_path_chunk, gpu_id, worker_id, task_obj_dict, task_obj_dict_path, write_queue,
     gpu_detect_queue, gpu_sam_queue, gpu_tracking_queue, gpu_pointcloud_queue,
     gpu_robotseg_queue, my_reply_queue, loader_cls, root_path, loader_kwargs, debug_config, gripper_stats,
     DATASET_NAME, fps, vllm_config, filter_robot,
     task_object_mode, scoring_mode, verification_config, enable_crop, det_threshold, nms_threshold, tracks_dir,
     dataset_fov_x, pointcloud_batch_size, robotseg_max_frames,
     tracking_target_fps, tracking_max_frames, tracking_max_resolution, max_resolution,
     pointcloud_max_resolution, pointcloud_resolution_level,
     detection_frame_policy, save_pointclouds) = args
    gpu_label = _physical_gpu_label(gpu_id)

    # Set logging level for this worker process
    logging.basicConfig(level=logging.INFO, format=f'[Worker-{worker_id}|GPU-{gpu_label}] %(asctime)s - %(levelname)s - %(message)s')

    try:
        # Create Proxies to talk to GPU Server
        _detection_proxy = ModelProxy(gpu_detect_queue, my_reply_queue, worker_id, 'detect')
        proxies = {
            'detection': _detection_proxy,
            'sam2': SAM2Proxy(gpu_sam_queue, my_reply_queue, worker_id, 'sam'),
            'tracking': TrackerProxy(gpu_tracking_queue, my_reply_queue, worker_id, 'tracking'),
            'pointcloud': PointCloudProxy(gpu_pointcloud_queue, my_reply_queue, worker_id, 'pointcloud') if gpu_pointcloud_queue is not None else None,
            'robotseg': RobotSegProxy(gpu_robotseg_queue, my_reply_queue, worker_id, 'robotseg') if gpu_robotseg_queue is not None else None,
        }

        # Initialize Annotator (Lightweight CPU object)
        annotator = TrajectoryAnnotator(gpu_id, task_obj_dict, proxies, debug_config, gripper_stats,
                                        task_obj_dict_path=task_obj_dict_path, fps=fps,
                                        vllm_config=vllm_config,
                                        filter_robot=filter_robot, scoring_mode=scoring_mode,
                                        verification_config=verification_config,
                                        enable_crop=enable_crop, det_threshold=det_threshold,
                                        nms_threshold=nms_threshold, tracks_dir=tracks_dir,
                                        pointcloud_batch_size=pointcloud_batch_size,
                                        robotseg_max_frames=robotseg_max_frames,
                                        tracking_target_fps=tracking_target_fps,
                                        tracking_max_frames=tracking_max_frames,
                                        tracking_max_resolution=tracking_max_resolution,
                                        max_resolution=max_resolution,
                                        pointcloud_max_resolution=pointcloud_max_resolution,
                                        pointcloud_resolution_level=pointcloud_resolution_level,
                                        task_object_mode=task_object_mode,
                                        detection_frame_policy=detection_frame_policy,
                                        save_pointclouds=save_pointclouds)

        # Load Dataset
        dataloader = loader_cls(root_path, dataset_name=DATASET_NAME, **loader_kwargs)
        # Pre-populate episode metadata so load_trajectory doesn't rebuild it per-call
        if hasattr(dataloader, '_episode_meta') and not dataloader._episode_meta:
            dataloader.get_trajectory_paths()
        annotator.fov_x = dataset_fov_x

        # To mimic AnnotationSaver logic locally
        saver = AnnotationSaver(None, write_queue=write_queue)

        import time as _time
        from tqdm import tqdm
        _worker_start = _time.perf_counter()
        _total_anns = 0
        _total_trajs = 0
        _PROFILE = os.environ.get("PROFILE_PIPELINE", "0") == "1"
        _io_total_s = 0.0
        _ann_total_s = 0.0
        for traj_path in tqdm(trajectory_path_chunk, desc=f"GPU {gpu_label} Worker {worker_id}", position=int(worker_id.split('_w')[-1])):
            try:
                # CPU Loading (Heavy IO)
                _io_t0 = _time.perf_counter()
                trajectory = dataloader.load_trajectory(traj_path, skip_frames=False, skip_observations=False)
                _io_dt = _time.perf_counter() - _io_t0
                _io_total_s += _io_dt
                if trajectory is None:
                    logging.info(f"Skipping {traj_path} as it could not be loaded.")
                    continue

                traj_fps = getattr(trajectory, "fps", None) or fps
                annotator.fps = traj_fps
                _traj_t0 = _time.perf_counter()
                # Run Logic (IPC calls happen here)
                annotations = annotator._annotate_single_trajectory(trajectory)
                _ann_total_s += _time.perf_counter() - _traj_t0
                _total_trajs += 1
                if _PROFILE:
                    logging.info(
                        f"[IO] traj={traj_path} load={_io_dt:.2f}s annotate={_time.perf_counter()-_traj_t0:.2f}s"
                    )

                if annotations:
                    _total_anns += len(annotations)
                    for ann in annotations:
                        saver.save(ann)

            except Exception as e:
                logging.info(f"Error processing {traj_path} on GPU {gpu_label} Worker {worker_id}: {e}")
                traceback.print_exc()
                continue

        _elapsed = _time.perf_counter() - _worker_start
        _ann_per_sec = _total_anns / _elapsed if _elapsed > 0 else float('inf')
        _traj_per_sec = _total_trajs / _elapsed if _elapsed > 0 else float('inf')
        logging.info(
            f"[STATS] Worker {worker_id} done: {_total_trajs} trajs, {_total_anns} anns "
            f"in {_elapsed:.1f}s  ({_traj_per_sec:.3f} traj/s, {_ann_per_sec:.3f} ann/s)"
        )
        if _PROFILE and _total_trajs > 0:
            io_pct = 100.0 * _io_total_s / _elapsed
            ann_pct = 100.0 * _ann_total_s / _elapsed
            logging.info(
                f"[STATS] Worker {worker_id} time split: "
                f"io_load={_io_total_s:.1f}s ({io_pct:.1f}%), "
                f"annotate={_ann_total_s:.1f}s ({ann_pct:.1f}%), "
                f"other={_elapsed-_io_total_s-_ann_total_s:.1f}s "
                f"(mean io={_io_total_s/_total_trajs:.2f}s/traj)"
            )

    except Exception as e:
        logging.critical(f"Worker {worker_id} died: {e}")
        traceback.print_exc()


def _resolve_gpu_ids(cfg_gpu_ids):
    """Resolve configured GPU IDs against any shell-level CUDA visibility mask."""
    gpu_ids = list(cfg_gpu_ids)
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible_devices:
        return gpu_ids

    visible_list = [dev.strip() for dev in visible_devices.split(",") if dev.strip()]
    if not visible_list:
        raise ValueError("CUDA_VISIBLE_DEVICES is set but empty after parsing")

    local_gpu_ids = list(range(len(visible_list)))
    if gpu_ids == local_gpu_ids:
        return gpu_ids

    if len(gpu_ids) == len(visible_list):
        logging.info(
            "CUDA_VISIBLE_DEVICES=%s remaps configured GPU IDs %s to local IDs %s",
            visible_devices,
            gpu_ids,
            local_gpu_ids,
        )
        return local_gpu_ids

    if any(gpu_id >= len(visible_list) or gpu_id < 0 for gpu_id in gpu_ids):
        raise ValueError(
            "Configured annotator.gpu_ids=%s is incompatible with CUDA_VISIBLE_DEVICES=%s. "
            "Use local GPU indices within the visible set."
            % (gpu_ids, visible_devices)
        )

    logging.info(
        "CUDA_VISIBLE_DEVICES=%s active; using local GPU IDs %s within the visible set",
        visible_devices,
        gpu_ids,
    )
    return gpu_ids


# --- Main Execution ---



def _start_gpu_servers(servers, ready_events):
    """Load models concurrently on separate GPUs before starting CPU workers."""
    for server in servers:
        server.start()
    for server, ready in zip(servers, ready_events):
        logging.info(
            "Waiting for GPU local=%s physical=%s to load models...",
            server.gpu_id,
            _physical_gpu_label(server.gpu_id),
        )
        ready.wait()


if __name__ == "__main__":
    # Import torch.multiprocessing here to avoid early CUDA initialization
    import torch.multiprocessing as mp
    # Crucial for PyTorch IPC and CUDA
    mp.set_start_method('spawn', force=True)

    # Configure Logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # --- Config Loading (OmegaConf) ---
    parser = argparse.ArgumentParser(
        description="Annotate trajectories with object bounding boxes.",
        epilog="Examples:\n"
               "  python annotate.py --dataset agibot_world dataset.root=/path/to/agibot\n"
               "  python annotate.py --dataset bridge_lerobot dataset.root=/path/to/bridge annotator.gpu_ids=[0,1]\n"
               "  python annotate.py --dataset agibot_world dataset.root=/path/to/agibot debug.debug_image_freq=1\n"
               "  python annotate.py --config /path/to/config.yaml\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a resolved YAML config file.")
    parser.add_argument("--dataset", type=str, default="agibot_world",
                        help="Dataset name (loads configs/dataset/{name}.yaml).")
    parser.add_argument("--starvla-config", type=str, default=None,
                        help="Path to a starVLA training YAML; annotate the dataset "
                             "defined in its datasets.vla_data block (data_root_dir + data_mix).")
    parser.add_argument("--starvla-root", type=str, default=None,
                        help="Path to the starVLA repo (else $STARVLA_ROOT, else ../starVLA).")
    parser.add_argument("--merge-only", action="store_true",
                        help="Merge node JSONL files into the main output file and exit.")
    parser.add_argument("overrides", nargs="*",
                        help="OmegaConf dot-list overrides, e.g. annotator.gpu_ids=[0]")
    cli_args = parser.parse_args()

    if cli_args.starvla_config:
        cfg = load_config_from_starvla(
            train_yaml=cli_args.starvla_config,
            starvla_root=cli_args.starvla_root,
            overrides=cli_args.overrides,
        )
    else:
        cfg = load_annotator_config(
            config_path=cli_args.config,
            dataset=cli_args.dataset,
            overrides=cli_args.overrides,
        )

    #pretty print the loaded config (config is DictConfig, use OmegaConf to print)
    from omegaconf import OmegaConf
    logging.info("Loaded Config:")
    logging.info("\n" + OmegaConf.to_yaml(cfg))


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
    OUTPUT_FILE = cfg.annotator.output_file

    if cli_args.merge_only:
        logging.info(f"--merge-only: merging node files into {OUTPUT_FILE}")
        merge_node_files(OUTPUT_FILE)
        logging.info("Merge complete.")
        exit(0)

    GPU_IDS = _resolve_gpu_ids(cfg.annotator.gpu_ids)
    PHYSICAL_GPU_IDS = [_physical_gpu_label(gpu_id) for gpu_id in GPU_IDS]
    N_PROCESSES_PER_GPU = cfg.annotator.n_processes_per_gpu
    FPS = cfg.dataset.fps
    TRACKING_MODEL = cfg.annotator.tracking_model
    N_TRACKER_INSTANCES = cfg.annotator.get("n_tracker_instances", 2)
    DETECTION_MODEL = cfg.annotator.get("detection_model", "llmdet")
    detector_cfg = cfg.annotator.get("detectors")
    DETECTION_CONFIG = (
        OmegaConf.to_container(detector_cfg, resolve=True)
        if detector_cfg is not None
        else {}
    )
    DETECTION_FRAME_POLICY = cfg.annotator.get(
        "detection_frame_policy", "pregrasp_midpoint"
    )
    FILTER_ROBOT = cfg.annotator.get("filter_robot", False)
    SCORING_MODE = cfg.annotator.get("scoring_mode", "full")
    TASK_OBJECT_MODE = cfg.extract_task_objects.get("mode", "text")

    # Cross-model verification config
    _cmv = cfg.annotator.get("cross_model_verification", {})
    VERIFICATION_CONFIG = {
        "mode": _cmv.get("mode", "off") if _cmv else "off",
        "weight": float(_cmv.get("weight", 0.15)) if _cmv else 0.15,
        "hard_gate": bool(_cmv.get("hard_gate", False)) if _cmv else False,
    }

    ENABLE_CROP = cfg.annotator.get("enable_crop", False)
    DET_THRESHOLD = cfg.annotator.get("det_threshold", 0.05)
    NMS_THRESHOLD = cfg.annotator.get("nms_threshold", 0.8)
    RUNTIME_DETECTION_THRESHOLDS = {
        "standard_box_query": float(DET_THRESHOLD),
        "standard_text_class": STANDARD_DETECTION_TEXT_THRESHOLD,
        "tool_box_query": float(DET_THRESHOLD),
        "tool_text_class": DEFAULT_DETECTION_TEXT_THRESHOLD,
        "tool_object_box_query": TOOL_OBJECT_DETECTION_THRESHOLD,
        "tool_object_text_class": DEFAULT_DETECTION_TEXT_THRESHOLD,
    }
    POINTCLOUD_BATCH_SIZE = cfg.annotator.get("pointcloud_batch_size", 32)
    POINTCLOUD_MAX_RESOLUTION = cfg.annotator.get("pointcloud_max_resolution", None)
    POINTCLOUD_RESOLUTION_LEVEL = cfg.annotator.get("pointcloud_resolution_level", 9)
    ROBOTSEG_MAX_FRAMES = cfg.annotator.get("robotseg_max_frames", None)
    MAX_RESOLUTION = cfg.annotator.get("max_resolution", None)
    TRACKING_TARGET_FPS = cfg.annotator.get("tracking_target_fps", None)
    TRACKING_MAX_FRAMES = cfg.annotator.get("tracking_max_frames", None)
    TRACKING_MAX_RESOLUTION = cfg.annotator.get("tracking_max_resolution", None)
    DATASET_FOV_X = cfg.dataset.get("fov_x", None)

    logging.info(f"Dataset: {DATASET_NAME} (type={DATASET_TYPE})")
    logging.info(f"Root: {ROOT_PATH}")
    logging.info(
        "GPU IDs: local=%s physical=%s, workers/GPU: %s",
        GPU_IDS,
        PHYSICAL_GPU_IDS,
        N_PROCESSES_PER_GPU,
    )
    logging.info(f"Tracking: model={TRACKING_MODEL}, tracker_instances={N_TRACKER_INSTANCES}")

    # Setup Debug Config (from cfg.debug group)
    debug_config = {
        "save_debug_images": cfg.debug.save_debug_images,
        "debug_image_freq": cfg.debug.debug_image_freq,
        "debug_image_dir": cfg.debug.debug_image_dir,
        "debug_image_format": cfg.debug.debug_image_format,
        "debug_max_per_trajectory": cfg.debug.debug_max_per_traj,
    }

    # Build VLLM config dict (plain dict, safe to pass across processes)
    from sparc.llm.llm_utils import build_vllm_extra_body
    _vllm_raw = {
        "base_url": cfg.vllm.base_url,
        "model_name": cfg.vllm.model_name,
        "temperature": cfg.vllm.temperature,
        "enable_thinking": cfg.vllm.get("enable_thinking"),
        "repetition_penalty": cfg.vllm.get("repetition_penalty"),
        "top_k": cfg.vllm.get("top_k"),
        "presence_penalty": cfg.vllm.get("presence_penalty"),
    }
    vllm_config = {
        "base_url": _vllm_raw["base_url"],
        "model_name": _vllm_raw["model_name"],
        "temperature": _vllm_raw["temperature"],
        "extra_body": build_vllm_extra_body(_vllm_raw),
    }
    logging.info(f"VLLM config: model={vllm_config['model_name']}, base_url={vllm_config['base_url']}")

    # --- Multi-node setup ---
    num_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", 1))
    node_rank = int(os.environ.get("SLURM_NODEID", 0))

    # Merge/read processed trajectories from the base file; write new rows to node files.
    # In multi-node mode only rank 0 merges, then other ranks wait so they all skip
    # against the same base annotations.jsonl instead of their own node shard.
    base_output_file = OUTPUT_FILE
    base_output_path = Path(base_output_file)
    merge_done_file = base_output_path.with_name(f".{base_output_path.name}.startup_merge_done")
    merge_started_at = time.time()
    existing_node_files = _node_output_files(base_output_file)
    if num_nodes > 1:
        if node_rank == 0:
            logging.info(f"Found {len(existing_node_files)} existing node files, merging on rank 0...")
            merge_node_files(base_output_file)
            merge_done_file.parent.mkdir(parents=True, exist_ok=True)
            merge_done_file.write_text(f"{time.time()}\n")
        else:
            logging.info(f"Waiting for rank 0 startup merge marker: {merge_done_file}")
            _wait_for_merge_done(merge_done_file, merge_started_at)
    elif existing_node_files:
        logging.info(f"Found {len(existing_node_files)} existing node files, merging...")
        merge_node_files(base_output_file)

    # For multi-node, each node writes to its own file
    if num_nodes > 1:
        OUTPUT_FILE = base_output_file.replace(".jsonl", f"_node{node_rank}.jsonl")
        logging.info(f"Multi-node mode: Node {node_rank}/{num_nodes} writing to {OUTPUT_FILE}")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # Save resolved config to output root dir (only on rank-0 node to avoid races)
    if node_rank == 0:
        config_save_path = os.path.join(os.path.dirname(OUTPUT_FILE), f"{DATASET_NAME}_config.yaml")
        with open(config_save_path, "w") as _cfg_f:
            _cfg_f.write(OmegaConf.to_yaml(cfg))
        logging.info(f"Saved resolved config to {config_save_path}")

    if cfg.annotator.get("save_tracks_npz", True):
        TRACKS_DIR = cfg.annotator.get("tracks_dir") or os.path.join(
            os.path.dirname(OUTPUT_FILE), "tracks"
        )
    else:
        TRACKS_DIR = None

    # --- Reannotate: backup existing output before starting writer ---
    reannotate = cfg.annotator.get("regenerate", False)
    if reannotate and os.path.exists(base_output_file):
        logging.info(f"Re-annotation enabled. Creating backup of existing file: {base_output_file}")
        backup_file = base_output_file.replace(".jsonl", "_backup.jsonl")
        if node_rank == 0:
            os.rename(base_output_file, backup_file)
        processed = []
    else:
        processed = get_all_processed_trajectories(base_output_file)

    # Load task dictionary
    logging.info("Loading task dictionary...")
    if os.path.exists(TASK_OBJ_DICT_PATH):
        with open(TASK_OBJ_DICT_PATH, "rb") as f:
            task_obj_dict = pickle.load(f)

        logging.info(f"Loaded task dictionary with {len(task_obj_dict)} entries from {TASK_OBJ_DICT_PATH}")
    else:
        logging.warning(f"Task dict not found at {TASK_OBJ_DICT_PATH}, starting with empty dict.")
        task_obj_dict = {}

    gripper_stats = {}

    # Manager for Queues
    manager = Manager()
    write_queue = manager.Queue()

    # --- Start Annotation Saver ---
    saver = AnnotationSaver(
        OUTPUT_FILE,
        write_queue,
        externalize_arrays=cfg.annotator.get("externalize_arrays", True),
        arrays_sidecar_dir=cfg.annotator.get("arrays_sidecar_dir"),
        array_shard_target_bytes=cfg.annotator.get(
            "array_shard_target_bytes", 1 << 30,
        ),
    )
    writer_p = Process(target=saver.start_writer_process)
    writer_p.start()

    # --- Dataset Loading ---
    loader_cls = get_loader_class(DATASET_TYPE)

    # OXE needs sub-dataset path appended
    loader_root = ROOT_PATH
    if DATASET_TYPE == "oxe":
        oxe_ds_name = "_".join(DATASET_NAME.split("_")[1:])
        loader_root = os.path.join(ROOT_PATH, oxe_ds_name)

    dataloader = loader_cls(loader_root, dataset_name=DATASET_NAME, **LOADER_KWARGS)
    all_trajs = dataloader.trajectories

    logging.info(f"Processed {len(processed)} trajectories already (merged from all sources).")

    # Droid .mp4 handling
    if DATASET_NAME == "droid":
        processed = set([t if t.endswith(".mp4") else t + ".mp4" for t in processed])

    remaining = [
        traj for traj in all_trajs if str(traj) not in processed
    ]

    logging.info(f"Total remaining trajectories: {len(remaining)}")

    # --- Debug subsampling ---
    gt_annotations_dir = cfg.debug.get("gt_annotations_dir")
    gt_filter_applied = gt_annotations_dir is not None
    if gt_annotations_dir is not None:
        import json
        gt_annotations_file = Path(gt_annotations_dir) / f"{DATASET_NAME}_gt_annotations.jsonl"
        with open(gt_annotations_file) as _f:
            _gt_trajs = {
                trajectory_id
                for row in (json.loads(line) for line in _f if line.strip())
                if (trajectory_id := trajectory_id_from_annotation(row)) is not None
            }
        remaining = [t for t in remaining if str(t) in _gt_trajs]
        logging.info(f"GT filter: restricted to {len(remaining)} trajectories from {gt_annotations_file}")

    max_per_task = cfg.dataset.get("max_per_task")
    if gt_filter_applied and max_per_task is not None:
        logging.info(
            "Task subsampling skipped: GT filter is active, so all %d matching trajectories will be kept",
            len(remaining),
        )
    elif max_per_task is not None and hasattr(dataloader, "sample_max_per_task"):
        sampled = set(dataloader.sample_max_per_task(max_per_task=int(max_per_task), seed=42))
        remaining = [t for t in remaining if str(t) in sampled]
        logging.info(
            "Task subsampling (max_per_task=%d): %d trajectories across %d unique tasks",
            int(max_per_task), len(remaining), len(dataloader.task_map),
        )

    n_trajectories = cfg.debug.get("n_trajectories")
    if gt_filter_applied and n_trajectories is not None:
        logging.info(
            "Debug subsampling skipped: GT filter is active, so all %d matching trajectories will be kept",
            len(remaining),
        )
    elif n_trajectories is not None:
        n_trajectories = int(n_trajectories)
        if hasattr(dataloader, "sample_diverse_total"):
            sampling_strategy = cfg.debug.get("sampling_strategy", "proportional")
            min_per_subdataset = cfg.debug.get("min_per_subdataset", 1)
            diverse = set(
                dataloader.sample_diverse_total(
                    n_total=n_trajectories,
                    seed=42,
                    strategy=sampling_strategy,
                    min_per_subdataset=min_per_subdataset,
                )
            )
            remaining = [t for t in remaining if str(t) in diverse][:n_trajectories]
            logging.info(
                "Debug subsampling (%s diverse, min_per_subdataset=%s): %s trajectories across %s groups",
                sampling_strategy,
                min_per_subdataset,
                len(remaining),
                len(dataloader.robot_map),
            )
        elif hasattr(dataloader, "sample_diverse"):
            n_per_robot = max(1, n_trajectories // max(len(dataloader.robot_map), 1))
            diverse = set(dataloader.sample_diverse(n_per_robot=n_per_robot, seed=42))
            remaining = [t for t in remaining if str(t) in diverse][:n_trajectories]
            logging.info(f"Debug subsampling (diverse): {len(remaining)} trajectories across {len(dataloader.robot_map)} robots")
        else:
            sample_size = min(n_trajectories, len(remaining))
            remaining = random.Random(42).sample(remaining, sample_size)
            logging.info(f"Debug subsampling: using {len(remaining)} trajectories (seed=42)")

    # --- Multi-node partitioning ---
    if num_nodes > 1:
        remaining = [t for i, t in enumerate(remaining) if i % num_nodes == node_rank]
        logging.info(f"Node {node_rank}/{num_nodes}: Processing {len(remaining)} trajectories (partitioned)")

    if len(remaining) == 0:
        logging.info("Nothing to do.")
        saver.close()
        writer_p.join()
        exit()

    # --- Start GPU Servers & CPU Workers ---
    num_total_workers = len(GPU_IDS) * N_PROCESSES_PER_GPU
    chunks = [remaining[i::num_total_workers] for i in range(num_total_workers)]

    gpu_servers = []
    ready_events = []
    cpu_workers = []

    chunk_idx = 0

    for gpu_id in GPU_IDS:
        gpu_detect_queue      = manager.Queue()
        gpu_sam_queue         = manager.Queue()
        gpu_tracking_queue   = manager.Queue()
        gpu_pointcloud_queue  = manager.Queue() if cfg.annotator.get("enable_pointcloud", False) else None
        gpu_robotseg_queue    = manager.Queue() if cfg.annotator.get("enable_robotseg", False) else None
        gpu_reply_queues = {}
        server_ready = manager.Event()

        for i in range(N_PROCESSES_PER_GPU):
            worker_id = f"g{gpu_id}_w{i}"

            my_reply_queue = manager.Queue()
            gpu_reply_queues[worker_id] = my_reply_queue

            chunk = chunks[chunk_idx]
            chunk_idx += 1
            if not chunk: continue

            p = Process(target=process_chunk_wrapper, args=((
                chunk, gpu_id, worker_id, task_obj_dict, TASK_OBJ_DICT_PATH, write_queue,
                gpu_detect_queue, gpu_sam_queue, gpu_tracking_queue,
                gpu_pointcloud_queue, gpu_robotseg_queue, my_reply_queue, loader_cls, loader_root,
                LOADER_KWARGS, debug_config, gripper_stats, DATASET_NAME, FPS,
                vllm_config, FILTER_ROBOT, TASK_OBJECT_MODE, SCORING_MODE, VERIFICATION_CONFIG,
                ENABLE_CROP, DET_THRESHOLD, NMS_THRESHOLD, TRACKS_DIR,
                DATASET_FOV_X, POINTCLOUD_BATCH_SIZE, ROBOTSEG_MAX_FRAMES,
                TRACKING_TARGET_FPS, TRACKING_MAX_FRAMES, TRACKING_MAX_RESOLUTION, MAX_RESOLUTION,
                POINTCLOUD_MAX_RESOLUTION, POINTCLOUD_RESOLUTION_LEVEL,
                DETECTION_FRAME_POLICY, cfg.annotator.get("save_pointclouds", False),
            ),))
            cpu_workers.append(p)

        server = GPUInferenceServer(
            gpu_id, gpu_detect_queue, gpu_sam_queue, gpu_tracking_queue,
            gpu_reply_queues, server_ready, tracking_model_name=TRACKING_MODEL,
            n_tracker_instances=N_TRACKER_INSTANCES,
            detection_model_name=DETECTION_MODEL,
            detection_config=DETECTION_CONFIG,
            pointcloud_queue=gpu_pointcloud_queue,
            robotseg_queue=gpu_robotseg_queue,
            runtime_detection_thresholds=RUNTIME_DETECTION_THRESHOLDS,
        )
        gpu_servers.append(server)
        ready_events.append(server_ready)

    _start_gpu_servers(gpu_servers, ready_events)

    # Start Workers
    logging.info("GPU Servers ready. Starting CPU workers...")
    for p in cpu_workers:
        p.start()

    for p in cpu_workers:
        p.join()

    logging.info("All workers done.")

    for server in gpu_servers:
        server.terminate()

    saver.close()
    writer_p.join()

    logging.info("Done.")
