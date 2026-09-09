"""Specialized loaders for mobile-manipulation LeRobot datasets.

Ported from the pre-release loader module.  Keeping these loaders separate
avoids coupling the generic LeRobot loaders to Galaxea and RoboCOIN's
subtask-specific schemas.
"""

from collections import defaultdict
import glob
import json
import logging
from pathlib import Path
import re
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as _pq
from tqdm import tqdm

from .dataset_loaders import (
    BaseDatasetLoader,
    LeRobotDatasetLoader,
    Trajectory,
    VideoReader,
    decord_cpu,
)


_GALAXEA_SKIP_TASK_LABELS = {"unqualified", "qualified", "null", ""}

# Chinese prefixes whose tasks are pure robot navigation/posture — not object manipulation.
# Verified across all task folders in Galaxea-Open-World.
_GALAXEA_NAVIGATION_PREFIXES = frozenset({
    "本体",  # robot body movement (move forward/backward, rotate, torso up/down)
    "本机",  # same concept, alternate term
    "底盘",  # chassis-only movement
    "躯体",  # torso elevation / rotation
    "躯干",  # torso rise
    "机器",  # halt / remain stationary
    "恢复",  # reset / return to initial position
    "静止",  # hold still / remain motionless
    "无任",  # do nothing / no meaningful action
    "无动",  # no action
    "体回",  # return to initial position
    "向右",  # move right (directional base navigation)
    "向左",  # turn/move left (directional base navigation)
})

# English label prefixes that always indicate navigation-only tasks (verified across dataset).
# Used as a secondary check independent of Chinese prefix.
_GALAXEA_NAVIGATION_ENGLISH_PREFIXES = (
    "move chassis",          # chassis movement (all 89 verified labels)
    "return to initial",     # reset to start pose
    "reset to initial",
    "reset body to",
)

# English labels that are exact navigation-only phrases (lower-cased, stripped of trailing period).
_GALAXEA_NAVIGATION_ENGLISH_EXACT = frozenset({
    "halt", "halt robot",
    "no action", "do nothing",
    "perform no meaningful actions", "perform a meaningless action",
    "remain stationary", "remain still", "remain motionless", "remain completely still",
    "hold still",
    "machine not moving", "machine, do not move", "main body no action",
    "main body remains stationary", "main body, no action",
    "body remain stationary", "body remains stationary", "keep body stationary",
    "maintain body still", "maintain stationary",
})


def _is_galaxea_navigation(task_str: str) -> bool:
    """Return True if the task is a pure navigation/posture task (no object manipulation)."""
    if "@" not in task_str:
        return False
    chinese, english = task_str.split("@", 1)
    chinese = chinese.strip()
    english = english.strip().lower().rstrip(".")

    if any(chinese.startswith(p) for p in _GALAXEA_NAVIGATION_PREFIXES):
        return True
    if any(english.startswith(p) for p in _GALAXEA_NAVIGATION_ENGLISH_PREFIXES):
        return True
    if english in _GALAXEA_NAVIGATION_ENGLISH_EXACT:
        return True
    return False


class GalaxeaTaskDatasetLoader(BaseDatasetLoader):
    """LeRobot v3.0 loader for a single Galaxea Open World task folder.

    Directory layout::

        <root>/                         (outer task folder, e.g. Make_The_Bed_20250616_001/)
          training_data_set_meta.json
          <task_name>/                  (inner LeRobot dataset)
            meta/info.json
            meta/episodes/chunk-*/file-*.parquet
            meta/tasks.parquet          (index=task_string, col=task_index)
            data/chunk-*/file-*.parquet (frame-level, has task_index per frame)
            videos/<cam_key>/chunk-*/file-*.mp4

    Each episode is segmented into subtask phases via consecutive runs of the same
    ``task_index`` column value. Fine-grained tasks use "Chinese@English" format;
    only tasks containing "@" are kept (coarse/quality labels are skipped).

    Path format: ``"{dc_idx}/{ep_idx}/{run_idx}"``

    Gripper convention: action.left/right_gripper range 0–100 (0=closed, 100=open).
    gripper_state is stored as pos/100 so 0=closed, 1=open.
    """

    GALAXEA_CAM_PRIORITY = [
        "observation.images.head_rgb",
        "observation.images.head_right_rgb",
        "observation.images.left_wrist_rgb",
        "observation.images.right_wrist_rgb",
    ]

    def __init__(self, root_path: str, view_key: Optional[str] = None, dataset_name: str = "galaxea"):
        tmp_root = Path(root_path)
        self._task_name = tmp_root.name
        self._inner_root = tmp_root / self._task_name

        self._chunk_size = 1000
        self._cam_keys: list[str] = []
        self._path_template: Optional[str] = None
        self._data_path_template: Optional[str] = None
        self._task_map: dict[int, str] = {}
        self._fps: float = 15.0
        self._episode_video_meta: dict[int, dict] = {}
        self._subtask_meta: dict[str, dict] = {}
        self.diag: Optional[dict] = None

        self._load_meta(self._inner_root)

        if view_key is None:
            view_key = next(
                (k for k in self.GALAXEA_CAM_PRIORITY if k in self._cam_keys),
                self._cam_keys[0] if self._cam_keys else "observation.images.head_rgb",
            )
        self._view_key = view_key

        BaseDatasetLoader.__init__(self, root_path, dataset_name=dataset_name)

        self._subtask_meta = self.cached.get("subtask_meta", self._subtask_meta)
        self._episode_video_meta = {
            int(k): v for k, v in self.cached.get("episode_video_meta", {}).items()
        }

    def _load_meta(self, inner_root: Path) -> None:
        info_path = inner_root / "meta" / "info.json"
        if info_path.exists():
            try:
                with open(info_path) as f:
                    info = json.load(f)
                self._chunk_size = int(info.get("chunks_size", info.get("chunk_size", 1000)))
                self._fps = float(info.get("fps", 15.0))
                features = info.get("features", {})
                self._cam_keys = [
                    k for k, meta in features.items()
                    if isinstance(meta, dict) and meta.get("dtype") in {"video", "image"}
                ]
                self._path_template = info.get("video_path")
                self._data_path_template = info.get("data_path")
            except Exception as e:
                logging.warning(f"GalaxeaTaskDatasetLoader: could not read info.json: {e}")

        tasks_path = inner_root / "meta" / "tasks.parquet"
        if tasks_path.exists():
            try:
                df = pd.read_parquet(tasks_path)
                for task_str, row in df.iterrows():
                    self._task_map[int(row["task_index"])] = str(task_str)
            except Exception as e:
                logging.warning(f"GalaxeaTaskDatasetLoader: could not read tasks.parquet: {e}")

    @staticmethod
    def _parse_task_lang(task_str: str) -> str:
        """Extract English part from 'Chinese@English' format, lower-cased."""
        if "@" in task_str:
            return task_str.split("@", 1)[1].strip().lower()
        return task_str.strip().lower()

    def _video_path(self, episode_idx: int) -> Path:
        meta = self._episode_video_meta.get(episode_idx, {})
        vid_chunk = meta.get("video_chunk_idx")
        vid_file = meta.get("video_file_idx")
        cam_key = self._view_key
        if vid_chunk is not None and vid_file is not None:
            if self._path_template:
                return self._inner_root / self._path_template.format(
                    video_key=cam_key, chunk_index=vid_chunk, file_index=vid_file
                )
            return self._inner_root / "videos" / cam_key / f"chunk-{vid_chunk:03d}" / f"file-{vid_file:03d}.mp4"
        return self._inner_root / "videos" / cam_key / "chunk-000" / "file-000.mp4"

    def _data_parquet_path(self, dc: int, df_idx: int) -> Path:
        if self._data_path_template:
            return self._inner_root / self._data_path_template.format(
                chunk_index=dc, file_index=df_idx
            )
        return self._inner_root / "data" / f"chunk-{dc:03d}" / f"file-{df_idx:03d}.parquet"

    def get_trajectory_paths(self) -> list:
        self._subtask_meta = {}
        self._episode_video_meta = {}
        paths = []

        n_episodes_info = 0
        n_episodes_discovered = 0
        n_episodes_with_subtasks = 0
        n_trajectories = 0
        n_data_parquets_missing = 0
        n_parquets_read_error = 0

        info_path = self._inner_root / "meta" / "info.json"
        if info_path.exists():
            try:
                with open(info_path) as f:
                    n_episodes_info = json.load(f).get("total_episodes", 0)
            except Exception:
                pass

        episodes_glob = sorted(glob.glob(
            str(self._inner_root / "meta" / "episodes" / "chunk-*" / "file-*.parquet")
        ))

        # Phase 1: read episode metadata (video/data parquet indices)
        data_parquet_episodes: dict[tuple[int, int], list[int]] = defaultdict(list)
        for ep_file in episodes_glob:
            try:
                ep_meta_df = pd.read_parquet(ep_file)
                vid_chunk_col = f"videos/{self._view_key}/chunk_index"
                vid_file_col = f"videos/{self._view_key}/file_index"
                ts_col = f"videos/{self._view_key}/from_timestamp"
                for _, row in ep_meta_df.iterrows():
                    eidx = int(row["episode_index"])
                    dc = int(row["data/chunk_index"]) if "data/chunk_index" in row.index else 0
                    df_idx = int(row["data/file_index"]) if "data/file_index" in row.index else 0
                    self._episode_video_meta[eidx] = {
                        "video_chunk_idx": int(row[vid_chunk_col]) if vid_chunk_col in row.index else None,
                        "video_file_idx": int(row[vid_file_col]) if vid_file_col in row.index else None,
                        "data_chunk_idx": dc,
                        "data_file_idx": df_idx,
                        "from_timestamp": float(row[ts_col]) if ts_col in row.index else 0.0,
                    }
                    data_parquet_episodes[(dc, df_idx)].append(eidx)
            except Exception as e:
                logging.warning(f"GalaxeaTaskDatasetLoader: could not read {ep_file}: {e}")

        # Phase 2: find consecutive task_index runs per episode
        for (dc, df_idx), ep_indices in sorted(data_parquet_episodes.items()):
            data_parquet = self._data_parquet_path(dc, df_idx)
            if not data_parquet.exists():
                n_data_parquets_missing += 1
                continue

            try:
                schema_names = set(_pq.read_schema(data_parquet).names)
                cols = [c for c in ["episode_index", "frame_index", "task_index", "coarse_task_index"] if c in schema_names]
                df = pd.read_parquet(data_parquet, columns=cols)
            except Exception as e:
                logging.warning(f"GalaxeaTaskDatasetLoader: could not read {data_parquet}: {e}")
                n_parquets_read_error += 1
                continue

            ep_set = set(ep_indices)
            n_episodes_discovered += len(ep_set & set(int(e) for e in df["episode_index"].unique()))

            for ep_idx, ep_df in df.groupby("episode_index"):
                ep_idx_int = int(ep_idx)
                if ep_idx_int not in ep_set:
                    continue

                ep_df = ep_df.sort_values("frame_index")

                coarse_task_label = ""
                if "coarse_task_index" in ep_df.columns:
                    coarse_idx = int(ep_df["coarse_task_index"].iloc[0])
                    coarse_task_label = self._parse_task_lang(self._task_map.get(coarse_idx, ""))

                # Find consecutive runs of the same task_index
                runs: list[tuple[int, int, int]] = []  # (task_idx, start_frame, end_frame_excl)
                prev_tidx = None
                run_start = 0
                for _, row in ep_df[["frame_index", "task_index"]].iterrows():
                    frame_idx = int(row["frame_index"])
                    task_idx = int(row["task_index"])
                    if task_idx != prev_tidx:
                        if prev_tidx is not None:
                            runs.append((prev_tidx, run_start, frame_idx))
                        prev_tidx = task_idx
                        run_start = frame_idx
                if prev_tidx is not None:
                    last_frame = int(ep_df["frame_index"].iloc[-1])
                    runs.append((prev_tidx, run_start, last_frame + 1))

                # Keep only fine-grained object-manipulation tasks
                valid_runs = []
                for task_idx, start, end in runs:
                    task_str = self._task_map.get(task_idx, "")
                    if "@" not in task_str:
                        continue
                    label = self._parse_task_lang(task_str)
                    if label in _GALAXEA_SKIP_TASK_LABELS:
                        continue
                    if _is_galaxea_navigation(task_str):
                        continue
                    valid_runs.append((task_idx, start, end, label))

                if not valid_runs:
                    continue

                n_episodes_with_subtasks += 1
                for run_idx, (_, start, end, label) in enumerate(valid_runs):
                    path_str = f"{dc}/{ep_idx_int}/{run_idx}"
                    self._subtask_meta[path_str] = {
                        "ep_idx": ep_idx_int,
                        "data_chunk_idx": dc,
                        "data_file_idx": df_idx,
                        "start_frame": start,
                        "end_frame": end,
                        "lang_ann": label,
                        "lang_ann_top": coarse_task_label,
                        "longtask_id": f"galaxea#{self._task_name}#{ep_idx_int}",
                        "longtask_step": run_idx,
                        "longtask_len": len(valid_runs),
                        "longtask_goal": coarse_task_label,
                    }
                    paths.append(path_str)
                    n_trajectories += 1

        self.diag = {
            "n_episodes_info": n_episodes_info,
            "n_episodes_discovered": n_episodes_discovered,
            "n_episodes_with_subtasks": n_episodes_with_subtasks,
            "n_trajectories": n_trajectories,
            "n_data_parquets_missing": n_data_parquets_missing,
            "n_parquets_read_error": n_parquets_read_error,
        }
        print(f"[GalaxeaTaskDatasetLoader {self._task_name}] "
              f"info={n_episodes_info} discovered={n_episodes_discovered} "
              f"with_subtasks={n_episodes_with_subtasks} trajs={n_trajectories}")

        self.cached["subtask_meta"] = self._subtask_meta
        self.cached["episode_video_meta"] = self._episode_video_meta

        return sorted(paths, key=lambda p: tuple(int(x) for x in p.split("/")))

    def load_trajectory(
        self,
        path: Path,
        load_frames: bool = False,
        skip_frames: bool = True,
        skip_observations: bool = True,
        **kwargs,
    ) -> Optional[Trajectory]:
        path_str = str(path)

        if not self._subtask_meta:
            self.get_trajectory_paths()

        meta = self._subtask_meta.get(path_str)
        if meta is None:
            logging.warning(f"GalaxeaTaskDatasetLoader: unknown path {path_str}")
            return None

        ep_idx: int = meta["ep_idx"]
        video_path = self._video_path(ep_idx)
        split = self.get_split_type(path)

        traj = Trajectory(
            name=path_str,
            lang_ann=meta["lang_ann"],
            base_path=str(self._inner_root),
            dataset_name=self.key,
            media_dir=str(video_path),
            split=split,
        )
        traj.start_frame = meta["start_frame"]
        traj.end_frame = meta["end_frame"]
        traj.longtask_id = meta["longtask_id"]
        traj.longtask_step = meta["longtask_step"]
        traj.longtask_len = meta["longtask_len"]
        traj.longtask_goal = meta["longtask_goal"]
        traj.fps = self._fps
        traj.camera_view_key = self._view_key

        if not skip_observations:
            data_path = self._data_parquet_path(meta["data_chunk_idx"], meta["data_file_idx"])
            if data_path.exists():
                try:
                    schema_names = set(_pq.read_schema(data_path).names)
                    obs_cols = ["episode_index", "frame_index"]
                    for c in [
                        "action.left_gripper", "action.right_gripper",
                        "observation.state.left_arm",
                        "observation.state.left_ee_pose", "observation.state.right_ee_pose",
                    ]:
                        if c in schema_names:
                            obs_cols.append(c)
                    df = pd.read_parquet(data_path, columns=obs_cols)
                    sub_df = df[
                        (df["episode_index"] == ep_idx) &
                        (df["frame_index"] >= meta["start_frame"]) &
                        (df["frame_index"] < meta["end_frame"])
                    ].sort_values("frame_index")

                    if len(sub_df) > 0:
                        if "action.left_gripper" in sub_df.columns:
                            lg = np.asarray(sub_df["action.left_gripper"].tolist(), dtype=np.float32).reshape(-1)
                            traj.left_gripper_state = lg / 100.0
                            traj.gripper_state = traj.left_gripper_state
                        if "action.right_gripper" in sub_df.columns:
                            rg = np.asarray(sub_df["action.right_gripper"].tolist(), dtype=np.float32).reshape(-1)
                            traj.right_gripper_state = rg / 100.0
                        if "observation.state.left_ee_pose" in sub_df.columns:
                            left_pose = LeRobotDatasetLoader._extract_pose7_column(sub_df["observation.state.left_ee_pose"])
                            if left_pose is not None:
                                traj.left_eef_position = left_pose[:, :3]
                        if "observation.state.right_ee_pose" in sub_df.columns:
                            right_pose = LeRobotDatasetLoader._extract_pose7_column(sub_df["observation.state.right_ee_pose"])
                            if right_pose is not None:
                                traj.right_eef_position = right_pose[:, :3]
                        if "observation.state.left_arm" in sub_df.columns:
                            joint_pos = LeRobotDatasetLoader._stack_array_column(sub_df["observation.state.left_arm"])
                            if joint_pos is not None:
                                traj.gripper_velocity = joint_pos.reshape(len(joint_pos), -1)[:, :6]
                except Exception as e:
                    logging.warning(f"GalaxeaTaskDatasetLoader: could not load observations for {path}: {e}")

        if not skip_frames:
            if not video_path.exists():
                raise ValueError(f"Video not found: {video_path}")

            from_ts = self._episode_video_meta.get(ep_idx, {}).get("from_timestamp", 0.0)
            data_path = self._data_parquet_path(meta["data_chunk_idx"], meta["data_file_idx"])
            if not data_path.exists():
                raise ValueError(f"Data parquet not found: {data_path}")

            sub_df = pd.read_parquet(data_path, columns=["episode_index", "frame_index", "timestamp"])
            sub_df = sub_df[
                (sub_df["episode_index"] == ep_idx) &
                (sub_df["frame_index"] >= meta["start_frame"]) &
                (sub_df["frame_index"] < meta["end_frame"])
            ].sort_values("frame_index")

            if sub_df.empty:
                raise ValueError(f"No frames found in parquet for {path}")

            shifted_timestamps = (sub_df["timestamp"].values + from_ts).tolist()
            vr = VideoReader(str(video_path), ctx=decord_cpu(0))
            fps = vr.get_avg_fps()
            frame_indices = [min(max(round(ts * fps), 0), len(vr) - 1) for ts in shifted_timestamps]
            traj.frames = vr.get_batch(frame_indices).asnumpy()

        return traj


class GalaxeaOpenWorldDatasetLoader(BaseDatasetLoader):
    """Wraps multiple GalaxeaTaskDatasetLoader instances (one per task folder).

    Path format: ``"{task_folder}/{dc_idx}/{ep_idx}/{run_idx}"``

    Usage::

        loader = GalaxeaOpenWorldDatasetLoader("/data/Galaxea-Open-World/lerobot")
        traj = loader.load_trajectory(Path("Make_The_Bed_20250616_001/0/3/1"))
    """

    def __init__(self, root_path: str, view_key: Optional[str] = None, dataset_name: str = "galaxea", n_workers: int = 16):
        self._view_key = view_key
        self._loaders: dict[str, GalaxeaTaskDatasetLoader] = {}

        tmp_root = Path(root_path)
        task_dirs = sorted(d for d in tmp_root.iterdir() if d.is_dir())

        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _load_task(task_dir):
            loader = GalaxeaTaskDatasetLoader(
                str(task_dir),
                view_key=view_key,
                dataset_name=f"galaxea_{task_dir.name}",
            )
            return task_dir.name, loader

        if n_workers > 1:
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = {pool.submit(_load_task, td): td for td in task_dirs}
                for future in tqdm(as_completed(futures), total=len(futures), desc="Loading Galaxea tasks"):
                    task_dir = futures[future]
                    try:
                        task_name, loader = future.result()
                        self._loaders[task_name] = loader
                    except Exception as e:
                        logging.warning(f"GalaxeaOpenWorldDatasetLoader: could not load {task_dir}: {e}")
        else:
            for task_dir in tqdm(task_dirs, desc="Loading Galaxea tasks"):
                try:
                    task_name, loader = _load_task(task_dir)
                    self._loaders[task_name] = loader
                except Exception as e:
                    logging.warning(f"GalaxeaOpenWorldDatasetLoader: could not load {task_dir}: {e}")

        total = sum(len(l.trajectories) for l in self._loaders.values())
        print(f"GalaxeaOpenWorldDatasetLoader: {len(self._loaders)} tasks, {total} trajectories total")

        BaseDatasetLoader.__init__(self, root_path, dataset_name=dataset_name)

    def get_trajectory_paths(self) -> list:
        paths = []
        for task_name, loader in self._loaders.items():
            for traj_path in loader.trajectories:
                paths.append(f"{task_name}/{traj_path}")
        return sorted(paths)

    def load_trajectory(
        self,
        path: Path,
        load_frames: bool = False,
        skip_frames: bool = True,
        skip_observations: bool = True,
        **kwargs,
    ) -> Optional[Trajectory]:
        path_str = str(path)
        parts = path_str.split("/", 1)
        if len(parts) != 2:
            logging.warning(f"GalaxeaOpenWorldDatasetLoader: invalid path {path_str}")
            return None

        task_name, sub_path = parts
        loader = self._loaders.get(task_name)
        if loader is None:
            logging.warning(f"GalaxeaOpenWorldDatasetLoader: unknown task '{task_name}'")
            return None

        traj = loader.load_trajectory(
            Path(sub_path),
            load_frames=load_frames,
            skip_frames=skip_frames,
            skip_observations=skip_observations,
            **kwargs,
        )
        if traj is not None:
            traj.name = f"{task_name}/{traj.name}"
        return traj


_ROBOCOIN_SKIP_SUBTASK_LABELS = {"null", "end", "abnormal", "static"}


def _is_robocoin_navigation(label: str) -> bool:
    """Return True if the subtask is robot body navigation (no arm manipulation).

    Verified patterns across all subtask_annotations.jsonl files in RoboCOIN:
    - "Move to <location>" — robot base navigating to a position (no object)
    - "Return to <position>" — robot base returning to start
    Distinct from "Move the X to ..." / "Move X to ..." which move objects with the arm.
    """
    ll = label.strip().lower()
    return ll.startswith("move to ") or ll.startswith("return to ")


class RoboCOINDatasetLoader(BaseDatasetLoader):
    """Dataloader for the RoboCOIN dataset (LeRobot v3.0, multi-robot, multi-task).

    Directory layout:
        <root>/<robot>/<gripper_type>/<task_dataset>/
            annotations/subtask_annotations.jsonl
            meta/info.json
            meta/episodes/chunk-*/file-*.parquet
            data/chunk-*/file-*.parquet
            videos/<cam_key>/chunk-*/file-*.mp4

    Each episode is segmented into subtask phases (via the ``subtask_annotation``
    column in the data parquet, where index 0 = active subtask id).
    Consecutive non-skip subtask segments within an episode are grouped into one
    Trajectory with ``interaction_phases`` (analogous to AgiBotWorldDatasetLoader).

    Path format: ``"{robot}/{gripper_type}/{task_name}/{dc_idx}/{ep_idx}/{pair_idx}"``

    Observations:
        - ``gripper_state``        : 1 - left_gripper_open   (action column 15)
        - ``left_gripper_state``   : 1 - action[:, 15]
        - ``right_gripper_state``  : 1 - action[:, 16]
        - ``left_eef_position``    : eef_sim_pose_state[:, 0:3]
        - ``right_eef_position``   : eef_sim_pose_state[:, 6:9]
        - ``gripper_velocity``     : action[:, :6]  (arm joint velocities proxy)
    """

    CAM_PRIORITY = [
        "observation.images.cam_high_rgb",
        "observation.images.cam_high",
        "observation.images.cam_front_rgb",
        "observation.images.cam_front",
    ]

    def __init__(
        self,
        root_path: str,
        view_key: Optional[str] = None,
        dataset_name: str = "robocoin",
    ):
        self._view_key_pref = view_key
        self._subtask_meta: dict[str, dict] = {}
        self._episode_video_meta: dict[str, dict] = {}  # "{robot}/{gripper}/{task}/{ep_idx}" -> meta
        self._subtask_map_cache: dict[str, dict[int, str]] = {}  # task_root -> {idx: label}
        # robot_map: robot_name -> list of trajectory path strings (for diverse subset selection)
        self.robot_map: dict[str, list[str]] = {}

        super().__init__(root_path, dataset_name=dataset_name)

        self._subtask_meta = self.cached.get("subtask_meta", self._subtask_meta)
        self._episode_video_meta = self.cached.get("episode_video_meta", self._episode_video_meta)
        # Rebuild robot_map from cached subtask_meta if available
        if self._subtask_meta and not self.robot_map:
            self._rebuild_robot_map()

    def _rebuild_robot_map(self) -> None:
        """Rebuild robot_map from _subtask_meta (called after cache restore)."""
        robot_map: dict[str, list[str]] = {}
        for path_str, meta in self._subtask_meta.items():
            robot = meta.get("robot_name", path_str.split("/", 1)[0])
            robot_map.setdefault(robot, []).append(path_str)
        self.robot_map = robot_map

    def sample_diverse(self, n_per_robot: int = 10, seed: int = 42) -> list[str]:
        """Return up to n_per_robot trajectory paths per robot for debugging."""
        import random
        rng = random.Random(seed)
        result = []
        for robot, paths in sorted(self.robot_map.items()):
            sample = rng.sample(paths, min(n_per_robot, len(paths)))
            result.extend(sample)
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _iter_task_roots(self):
        """Yield (robot, gripper_type, task_name, task_root_path) for every task dataset."""
        for robot_dir in sorted(self.root_path.iterdir()):
            if not robot_dir.is_dir():
                continue
            for gripper_dir in sorted(robot_dir.iterdir()):
                if not gripper_dir.is_dir():
                    continue
                for task_dir in sorted(gripper_dir.iterdir()):
                    if not task_dir.is_dir() or task_dir.name == "tasks":
                        continue
                    if (task_dir / "meta" / "info.json").exists():
                        yield robot_dir.name, gripper_dir.name, task_dir.name, task_dir

    def _load_subtask_map(self, task_root: Path) -> dict[int, str]:
        key = str(task_root)
        if key in self._subtask_map_cache:
            return self._subtask_map_cache[key]
        ann_path = task_root / "annotations" / "subtask_annotations.jsonl"
        result: dict[int, str] = {}
        if ann_path.exists():
            with open(ann_path, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line.strip())
                    result[int(item["subtask_index"])] = item["subtask"]
        self._subtask_map_cache[key] = result
        return result

    def _select_view_key(self, task_root: Path, cam_keys: list[str]) -> str:
        if self._view_key_pref and self._view_key_pref in cam_keys:
            return self._view_key_pref
        for k in self.CAM_PRIORITY:
            if k in cam_keys:
                return k
        return cam_keys[0] if cam_keys else "observation.images.cam_high_rgb"

    def _load_cam_keys(self, task_root: Path) -> list[str]:
        info_path = task_root / "meta" / "info.json"
        try:
            with open(info_path, "r", encoding="utf-8") as f:
                info = json.load(f)
            return [
                k for k, meta in info.get("features", {}).items()
                if isinstance(meta, dict) and meta.get("dtype") in {"video", "image"}
            ]
        except Exception:
            return []

    @staticmethod
    def _seg_arm_side(label: str) -> Optional[str]:
        """Extract 'left' or 'right' from a subtask label, or None if absent."""
        ll = label.lower()
        if "left" in ll:
            return "left"
        if "right" in ll:
            return "right"
        return None

    @staticmethod
    def _seg_object_noun(label: str) -> Optional[str]:
        """Extract the primary object noun from a subtask label using heuristic patterns."""
        # Start skills: "Grasp/Pick up/Grab [the] {noun} [from/with/on/into/,/end]"
        # Stop before prepositions but not before "and the" (which continues the noun phrase)
        m = re.search(
            r'(?:simultaneously\s+)?(?:grab\s+and\s+(?:pick\s+up|lift)|pick\s+up|grasp|grab|take|retrieve)'
            r'\s+(?:the\s+)?(.+?)(?:\s+(?:from|with|on\s+the|into)|[,.]|$)',
            label, re.IGNORECASE,
        )
        if m:
            return m.group(1).lower().strip().rstrip('.,')
        # Terminal skills: "Place/Put/Store [the] {noun} into/onto/in/on/with/to/,"
        m = re.search(
            r'(?:place|put|store|insert|return)\s+(?:the\s+)?(.+?)(?:\s+(?:into|onto|in\b|on\b|with\b|to\b)|[,.]|$)',
            label, re.IGNORECASE,
        )
        if m:
            return m.group(1).lower().strip().rstrip('.,')
        return None

    @staticmethod
    def _seg_is_start(label: str) -> bool:
        ll = label.lower()
        return any(w in ll for w in ("grasp", "pick up", "grab", "retrieve", "take "))

    @staticmethod
    def _seg_is_terminal(label: str) -> bool:
        ll = label.lower()
        return any(w in ll for w in ("place", "put ", "store", "release", "drop", "insert", "return", "press", "pass "))

    def _group_subtask_segments(
        self,
        segments: list[dict],
        robot: str,
        gripper: str,
        task_name: str,
        ep_idx: int,
        dc_idx: int,
        lang_ann_top: str,
    ) -> list[dict]:  # each entry also carries robot_name / gripper_type for robot_map
        """Group consecutive subtask segments into one pick-and-place trajectory.

        Grouping follows AgiBotWorld's approach:
        - Segments on the same arm/object stay together (Grasp → Place same object).
        - Break when: previous segment was terminal, next segment is a new start,
          arms differ (when known), or object nouns differ (when both extractable).
        """
        if not segments:
            return []

        groups: list[list[dict]] = []
        i = 0
        while i < len(segments):
            seg_i = segments[i]
            arm_i = self._seg_arm_side(seg_i["label"])
            noun_i = self._seg_object_noun(seg_i["label"])
            is_terminal_i = self._seg_is_terminal(seg_i["label"])

            group = [seg_i]
            j = i + 1
            while j < len(segments):
                seg_j = segments[j]
                arm_j = self._seg_arm_side(seg_j["label"])
                noun_j = self._seg_object_noun(seg_j["label"])

                # When start frames are contiguous, relax unknown arm info (same as AgiBotWorld)
                if seg_j["start_frame"] == seg_i["end_frame"]:
                    arm_j = arm_j if arm_j is not None else arm_i
                    arm_i = arm_i if arm_i is not None else arm_j

                arm_mismatch = (arm_i is not None and arm_j is not None and arm_i != arm_j)
                noun_mismatch = (noun_i is not None and noun_j is not None and noun_i != noun_j)

                should_break = (
                    is_terminal_i
                    or self._seg_is_start(seg_j["label"])
                    or arm_mismatch
                    or noun_mismatch
                )
                if should_break:
                    break

                group.append(seg_j)
                arm_i = arm_j if arm_j is not None else arm_i
                noun_i = noun_j if noun_j is not None else noun_i
                is_terminal_i = self._seg_is_terminal(seg_j["label"])
                seg_i = seg_j
                j += 1

            groups.append(group)
            i = j

        result = []
        for pair_idx, group in enumerate(groups):
            start_frame = group[0]["start_frame"]
            end_frame = group[-1]["end_frame"]
            phases = [
                {
                    "phase_type": seg["label"],
                    "start_frame": seg["start_frame"] - start_frame,
                    "end_frame": seg["end_frame"] - start_frame,
                }
                for seg in group
            ]
            lang_ann = " then ".join(seg["label"].lower().rstrip(".") for seg in group)
            task_prefix = f"{robot}/{gripper}/{task_name}"
            result.append({
                "task_prefix": task_prefix,
                "robot_name": robot,
                "gripper_type": gripper,
                "task_name": task_name,
                "dc_idx": dc_idx,
                "ep_idx": ep_idx,
                "pair_idx": pair_idx,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "lang_ann": lang_ann,
                "lang_ann_top": lang_ann_top,
                "interaction_phases": phases,
                "longtask_id": f"robocoin#{task_prefix}#{ep_idx}",
                "longtask_step": pair_idx,
                "longtask_len": len(groups),
                "longtask_goal": lang_ann_top,
            })
        return result

    # ------------------------------------------------------------------
    # Path discovery
    # ------------------------------------------------------------------

    def get_trajectory_paths(self) -> list[str]:
        self._subtask_meta = {}
        self._episode_video_meta = {}
        self.robot_map = {}
        paths = []

        for robot, gripper, task_name, task_root in self._iter_task_roots():
            task_prefix = f"{robot}/{gripper}/{task_name}"
            cam_keys = self._load_cam_keys(task_root)
            view_key = self._select_view_key(task_root, cam_keys)
            subtask_map = self._load_subtask_map(task_root)
            # subtask indices to skip entirely when forming trajectories
            skip_ids = {
                idx for idx, label in subtask_map.items()
                if label.lower() in _ROBOCOIN_SKIP_SUBTASK_LABELS or _is_robocoin_navigation(label)
            }

            # -- Phase 1: build episode video meta from episode parquets --
            ep_files = sorted(glob.glob(str(task_root / "meta" / "episodes" / "chunk-*" / "file-*.parquet")))
            ep_video_meta: dict[int, dict] = {}
            data_parquet_eps: dict[tuple[int, int], list[int]] = defaultdict(list)

            for ep_file in ep_files:
                try:
                    ep_df = pd.read_parquet(ep_file)
                    vid_chunk_col = f"videos/{view_key}/chunk_index"
                    vid_file_col = f"videos/{view_key}/file_index"
                    ts_col = f"videos/{view_key}/from_timestamp"
                    for _, row in ep_df.iterrows():
                        eidx = int(row["episode_index"])
                        dc = int(row["data/chunk_index"]) if "data/chunk_index" in row.index else 0
                        df_idx = int(row["data/file_index"]) if "data/file_index" in row.index else 0
                        ep_video_meta[eidx] = {
                            "view_key": view_key,
                            "video_chunk_idx": int(row[vid_chunk_col]) if vid_chunk_col in row.index else None,
                            "video_file_idx": int(row[vid_file_col]) if vid_file_col in row.index else None,
                            "from_timestamp": float(row[ts_col]) if ts_col in row.index else 0.0,
                            "data_chunk_idx": dc,
                            "data_file_idx": df_idx,
                        }
                        data_parquet_eps[(dc, df_idx)].append(eidx)
                except Exception as e:
                    logging.warning(f"RoboCOINDatasetLoader: could not read {ep_file}: {e}")

            # -- Phase 2: scan data parquets for subtask segments --
            for (dc_idx, df_idx), ep_indices in sorted(data_parquet_eps.items()):
                data_parquet = task_root / "data" / f"chunk-{dc_idx:03d}" / f"file-{df_idx:03d}.parquet"
                if not data_parquet.exists():
                    continue
                try:
                    schema_names = set(_pq.read_schema(data_parquet).names)
                    needed = [c for c in ["episode_index", "frame_index", "timestamp", "subtask_annotation", "task_index"] if c in schema_names]
                    df = pd.read_parquet(data_parquet, columns=needed)
                except Exception as e:
                    logging.warning(f"RoboCOINDatasetLoader: could not read {data_parquet}: {e}")
                    continue

                if "subtask_annotation" not in df.columns:
                    continue

                # -- Per-episode: extract task text and subtask segments --
                task_text_map: dict[int, str] = {}
                if "task_index" in df.columns:
                    tasks_pq = task_root / "meta" / "tasks.parquet"
                    if tasks_pq.exists():
                        try:
                            t_df = pd.read_parquet(tasks_pq)
                            # tasks.parquet: index=task text, column=task_index
                            task_text_map = {int(v): str(k) for k, v in t_df["task_index"].items()}
                        except Exception:
                            pass

                for ep_idx, ep_df in df.groupby("episode_index"):
                    ep_idx_int = int(ep_idx)
                    if ep_idx_int not in set(ep_indices):
                        continue

                    lang_ann_top = task_text_map.get(
                        int(ep_df["task_index"].iloc[0]) if "task_index" in ep_df.columns else -1, ""
                    )

                    ep_df = ep_df.sort_values("frame_index")
                    ann_col = np.array(ep_df["subtask_annotation"].tolist())  # (T, 5)
                    frame_indices = ep_df["frame_index"].values

                    # Extract contiguous segments where ann[0] != skip_id
                    segments: list[dict] = []
                    prev_active = None
                    seg_start = None
                    for t, (frame, ann_row) in enumerate(zip(frame_indices, ann_col)):
                        active = int(ann_row[0])
                        if active != prev_active:
                            if prev_active is not None and prev_active not in skip_ids:
                                segments.append({
                                    "start_frame": int(seg_start),
                                    "end_frame": int(frame),  # exclusive
                                    "label": subtask_map.get(prev_active, str(prev_active)),
                                })
                            seg_start = frame
                            prev_active = active
                    # flush last segment
                    if prev_active is not None and prev_active not in skip_ids and seg_start is not None:
                        segments.append({
                            "start_frame": int(seg_start),
                            "end_frame": int(frame_indices[-1]) + 1,
                            "label": subtask_map.get(prev_active, str(prev_active)),
                        })

                    subtasks = self._group_subtask_segments(
                        segments, robot, gripper, task_name, ep_idx_int, dc_idx, lang_ann_top
                    )

                    for meta in subtasks:
                        path_str = f"{task_prefix}/{dc_idx}/{ep_idx}/{meta['pair_idx']}"
                        self._subtask_meta[path_str] = meta
                        self._episode_video_meta[f"{task_prefix}/{ep_idx_int}"] = ep_video_meta.get(ep_idx_int, {})
                        self.robot_map.setdefault(robot, []).append(path_str)
                        paths.append(path_str)

        self.cached["subtask_meta"] = self._subtask_meta
        self.cached["episode_video_meta"] = self._episode_video_meta
        return sorted(paths)

    # ------------------------------------------------------------------
    # Trajectory loading
    # ------------------------------------------------------------------

    def _resolve_video_path(self, task_prefix: str, ep_idx: int) -> Path:
        ep_meta = self._episode_video_meta.get(f"{task_prefix}/{ep_idx}", {})
        view_key = ep_meta.get("view_key", self.CAM_PRIORITY[0])
        vid_chunk = ep_meta.get("video_chunk_idx", 0)
        vid_file = ep_meta.get("video_file_idx", 0)
        parts = task_prefix.split("/", 2)  # robot, gripper, task_name
        task_root = self.root_path / parts[0] / parts[1] / parts[2]
        return task_root / "videos" / view_key / f"chunk-{vid_chunk:03d}" / f"file-{vid_file:03d}.mp4"

    def _resolve_data_parquet(self, task_prefix: str, ep_idx: int) -> Path:
        ep_meta = self._episode_video_meta.get(f"{task_prefix}/{ep_idx}", {})
        dc = ep_meta.get("data_chunk_idx", 0)
        df = ep_meta.get("data_file_idx", 0)
        parts = task_prefix.split("/", 2)
        task_root = self.root_path / parts[0] / parts[1] / parts[2]
        return task_root / "data" / f"chunk-{dc:03d}" / f"file-{df:03d}.parquet"

    def load_trajectory(
        self,
        path: Path,
        load_frames: bool = False,
        skip_frames: bool = True,
        skip_observations: bool = True,
        **kwargs,
    ) -> Optional[Trajectory]:
        path_str = str(path)

        if not self._subtask_meta:
            self.get_trajectory_paths()

        meta = self._subtask_meta.get(path_str)
        if meta is None:
            logging.warning(f"RoboCOINDatasetLoader: unknown path {path_str}")
            return None

        task_prefix: str = meta["task_prefix"]
        ep_idx: int = meta["ep_idx"]
        video_path = self._resolve_video_path(task_prefix, ep_idx)
        split = self.get_split_type(path)

        traj = Trajectory(
            name=path_str,
            lang_ann=meta["lang_ann"],
            base_path=str(self.root_path),
            dataset_name=self.key,
            media_dir=str(video_path),
            split=split,
        )
        traj.start_frame = meta["start_frame"]
        traj.end_frame = meta["end_frame"]
        traj.interaction_phases = meta["interaction_phases"]
        traj.longtask_id = meta["longtask_id"]
        traj.longtask_step = meta["longtask_step"]
        traj.longtask_len = meta["longtask_len"]
        traj.longtask_goal = meta["longtask_goal"]
        traj.fps = 30.0
        traj.camera_view_key = self._episode_video_meta.get(
            f"{task_prefix}/{ep_idx}", {}
        ).get("view_key")

        if not skip_observations:
            data_path = self._resolve_data_parquet(task_prefix, ep_idx)
            if data_path.exists():
                try:
                    schema_names = set(_pq.read_schema(data_path).names)
                    obs_cols = [c for c in [
                        "episode_index", "frame_index",
                        "action", "eef_sim_pose_state",
                        "gripper_open_scale_state",
                    ] if c in schema_names]
                    df = pd.read_parquet(data_path, columns=obs_cols)
                    sub_df = df[
                        (df["episode_index"] == ep_idx) &
                        (df["frame_index"] >= meta["start_frame"]) &
                        (df["frame_index"] < meta["end_frame"])
                    ].sort_values("frame_index")

                    if len(sub_df) > 0:
                        # gripper_open_scale_state: 0=fully closed, 1=fully open (feature name).
                        # get_gripper_close_phases expects low=closed → use raw values directly.
                        if "gripper_open_scale_state" in sub_df.columns:
                            gs = np.array(sub_df["gripper_open_scale_state"].tolist(), dtype=np.float32)
                            if gs.ndim == 2 and gs.shape[1] >= 2:
                                traj.left_gripper_state = gs[:, 0]
                                traj.right_gripper_state = gs[:, 1]
                                traj.gripper_state = traj.left_gripper_state

                        # Fallback for robots without gripper_open_scale_state:
                        # synthesize a binary signal from pre-labeled interaction_phases.
                        # 0=closed during Grasp/Pick phases, 1=open otherwise.
                        if traj.gripper_state is None and meta.get("interaction_phases"):
                            T = meta["end_frame"] - meta["start_frame"]
                            synthetic = np.ones(T, dtype=np.float32)
                            grasp_kws = ("grasp", "pick", "grab", "take", "retrieve")
                            for ph in meta["interaction_phases"]:
                                if any(kw in ph["phase_type"].lower() for kw in grasp_kws):
                                    s = max(0, ph["start_frame"])
                                    e = min(T, ph["end_frame"])
                                    synthetic[s:e] = 0.0
                            traj.left_gripper_state = synthetic
                            traj.right_gripper_state = synthetic
                            traj.gripper_state = synthetic

                        # Arm velocity proxy from action[:, :6]
                        if "action" in sub_df.columns:
                            action = np.array(sub_df["action"].tolist(), dtype=np.float32)
                            if action.ndim == 2 and action.shape[1] >= 6:
                                traj.gripper_velocity = action[:, :6]

                    if "eef_sim_pose_state" in sub_df.columns and len(sub_df) > 0:
                        eef = np.array(sub_df["eef_sim_pose_state"].tolist(), dtype=np.float32)
                        if eef.ndim == 2 and eef.shape[1] >= 9:
                            traj.left_eef_position = eef[:, 0:3]
                            traj.right_eef_position = eef[:, 6:9]
                except Exception as e:
                    logging.warning(f"RoboCOINDatasetLoader: could not load observations for {path}: {e}")

        if not skip_frames:
            if not video_path.exists():
                raise ValueError(f"Video not found: {video_path}")

            ep_meta = self._episode_video_meta.get(f"{task_prefix}/{ep_idx}", {})
            from_ts = ep_meta.get("from_timestamp", 0.0)
            data_path = self._resolve_data_parquet(task_prefix, ep_idx)
            if not data_path.exists():
                raise ValueError(f"Data parquet not found: {data_path}")

            frame_df = pd.read_parquet(data_path, columns=["episode_index", "frame_index", "timestamp"])
            frame_df = frame_df[
                (frame_df["episode_index"] == ep_idx) &
                (frame_df["frame_index"] >= meta["start_frame"]) &
                (frame_df["frame_index"] < meta["end_frame"])
            ].sort_values("frame_index")

            if frame_df.empty:
                raise ValueError(f"No frames found in parquet for {path}")

            shifted_timestamps = (frame_df["timestamp"].values + from_ts).tolist()
            vr = VideoReader(str(video_path), ctx=decord_cpu(0))
            fps = vr.get_avg_fps()
            frame_indices = [min(max(round(ts * fps), 0), len(vr) - 1) for ts in shifted_timestamps]
            traj.frames = vr.get_batch(frame_indices).asnumpy()

        return traj
