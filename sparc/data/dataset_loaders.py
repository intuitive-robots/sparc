from abc import abstractmethod
from collections import defaultdict
import hashlib
import io
import logging
from pathlib import Path
import random
import cv2
import json
import numpy as np
from PIL import Image
import glob
import pandas as pd
import pyarrow.parquet as _pq

try:
    import h5py
except ImportError:
    h5py = None

try:
    from decord import VideoReader, cpu as decord_cpu
except ImportError:
    VideoReader = None

    def decord_cpu(*args, **kwargs):
        raise ImportError("decord is required for LeRobot frame decoding")

from typing import Any, Optional
from tqdm import tqdm

from .cache import Cachable
from . import egolive_grasp, egolive_phases


START_SKILLS = {
    "Pick",
    "Grasp",
    "TakeOver",
    "Catch",
    "Open",
    "OpenJar",
    "OpenBox",
    "Uncap",
    "Pull",
    "Unzip",
    "Unscrew",
    "Suction",
    "HeavyObjLift",
    "Scan",
    "Point"
}

TERMINAL_SKILLS = {
    "Place",
    "Release",
    "Drop",
    "Handover",
    "HandOut",
    "Close",
    "CloseJar",
    "CloseBox",
    "Plug",
    "Stamp",
    "Throw",
    "Wipe",
    "Brush",
    "Paint",
    "Mop",
    "Sweep",
    "Clap",
    "Wave",
    "PressButton",
    "Press",
    "Lower"
}

CONTINUATION_SKILLS = {
    "Fold",
    "Push",
    "Carry",
    "Pour",
    "Hang",
    "Move",
    "TurnWheel",
    "Scoop",
    "Unfold",
    "Insert",
    "Hold",
    "Iron",
    "Hammer",
    "Stretch",
    "Scratch",
    "PullApart",
    "Cut",
    "Screw",
    "Pinch",
    "Tap",
    "Stir",
    "Straighten",
    "Beat",
    "Rotate",
    "Remove",
    "TakeOut",
    "RollDough",
    "Turn",
    "Slide",
    "Twist",
    "Untie",
    "Shake",
    "Dip",
    "Rinse",
    "Whisk",
    "Peel",
    "Unstack",
    "Stack",
    "Flip",
    "Water",
    "FitTogether",
    "Tie",
    "Transport",
    "LargeObj",
    "Roll",
    "Chop",
    "Swipe"
}


class Trajectory:
    """A simple data class to hold information for a single trajectory."""
    def __init__(self, name: str, base_path: str, dataset_name, media_dir : str, lang_ann : str, split: str = "train"):

        self.name  : str = name
        self.split : str = split
        self.media_dir : str = media_dir
        self.base_path : str = base_path
        self.dataset_name : str = dataset_name

        self.lang_ann : str = lang_ann
        self.frames : Optional[np.ndarray] = None

        self.gripper_state : Optional[np.ndarray] = None
        self.gripper_velocity : Optional[np.ndarray] = None
        self.left_gripper_state: Optional[np.ndarray] = None   # (T,) left arm, dual-arm only
        self.right_gripper_state: Optional[np.ndarray] = None  # (T,) right arm, dual-arm only
        self.left_eef_position: Optional[np.ndarray] = None    # (T, 3) left arm EEF xyz, dual-arm when available
        self.right_eef_position: Optional[np.ndarray] = None   # (T, 3) right arm EEF xyz, dual-arm when available
        self.camera_extrinsics_cam_to_robot: Optional[np.ndarray] = None  # (T, 7) [tx, ty, tz, qx, qy, qz, qw]
        self.camera_extrinsics_view_key: Optional[str] = None
        self.camera_view_key: Optional[str] = None
        self.fps: Optional[float] = None
        self.source_uuid: Optional[str] = None
        self.source_episode_id: Optional[str] = None
        self.source_episode_index: Optional[int] = None
        self.source_subtrajectory_index: Optional[int] = None
        self.source_episode_frame_start: Optional[int] = None
        self.source_episode_frame_end_exclusive: Optional[int] = None
        self.source_episode_frame_indices: Optional[np.ndarray] = None
        self.frame_timestamps_seconds: Optional[np.ndarray] = None
        self.source_dataset_format: Optional[str] = None

        # Subtask / compound-task fields (populated by AgiBotWorldDatasetLoader)
        self.start_frame: Optional[int] = None
        self.end_frame: Optional[int] = None
        self.interaction_phases: Optional[list] = None
        self.longtask_id: Optional[str] = None
        self.longtask_step: Optional[int] = None
        self.longtask_len: Optional[int] = None
        self.longtask_goal: Optional[str] = None

        # Egocentric human-hand fields (populated by EgoLive loaders)
        self.frame_indices: Optional[np.ndarray] = None        # (T,) source frame index of each kept frame
        self.image_size: Optional[tuple] = None                # (height, width) of the frames as returned
        self.hand_object_boxes: Optional[dict] = None          # slot -> (T, 4) pixel xyxy, NaN where absent
        self.hand_object_track_ids: Optional[dict] = None      # slot -> (T,) track id, -1 where absent
        self.hand_occlusion: Optional[np.ndarray] = None       # (T, 4) [left_vis, left_ratio, right_vis, right_ratio]
        self.observation_state: Optional[np.ndarray] = None    # (T, D) raw state; EgoLive: wrist pose + fingertips
        self.camera_intrinsics: Optional[tuple] = None         # (fx, fy, cx, cy) at the returned resolution
        self.segment_start_frame: Optional[int] = None         # instruction segment inside a padded clip
        self.segment_end_frame: Optional[int] = None
        self.left_hand_kp3d: Optional[np.ndarray] = None       # (T, 21, 3) MANO joints, camera frame
        self.right_hand_kp3d: Optional[np.ndarray] = None      # (T, 21, 3)
        self.left_hand_kp2d: Optional[np.ndarray] = None       # (T, 21, 2) pixels, same scale as frames
        self.right_hand_kp2d: Optional[np.ndarray] = None      # (T, 21, 2)
        self.left_grasp_center_2d: Optional[np.ndarray] = None   # (T, 2) thumb/index midpoint, pixels
        self.right_grasp_center_2d: Optional[np.ndarray] = None  # (T, 2)


class BaseDatasetLoader(Cachable):

    """Abstract base class for dataset loaders."""
    def __init__(self, root_path: str,dataset_name: Optional[str] = None):

        super().__init__(dataset_name=dataset_name)
        self.root_path = Path(root_path)

        self.trajectories = self.cache("trajectories", self.get_trajectory_paths)
        if self.trajectories and isinstance(self.trajectories[0], str):
            self.trajectories = [Path(p) for p in self.trajectories]

        self.save()

        self.data = None

    def __iter__(self):
        """Yields Trajectory objects."""
        print(f"Loading {self.__class__.__name__}")
        loaded, failed = 0, 0
        for traj_path in tqdm(self.trajectories):
            try:
                traj = self.load_trajectory(Path(traj_path), skip_observations=True, skip_frames=True)
                if traj is None:
                    failed += 1
                else:
                    yield traj
                    loaded += 1
            except Exception:
                failed += 1
        print(f"Loaded {loaded}, failed to load {failed} trajectories")

    def is_test_trajectory(self, trajectory : Path) -> bool:
        """Determines if a trajectory is in the test split or not"""
        return False

    def is_validation_trajectory(self, trajectory : Path) -> bool:
        """Determines if a trajectory is in the test split or not"""
        return False

    def get_split_type(self, trajectory : Path) -> str:
        if self.is_test_trajectory(trajectory):
            return "test"
        elif self.is_validation_trajectory(trajectory):
            return "validation"
        else:
            return "train"



    @abstractmethod
    def load_trajectory(self, path : Path, load_frames, skip_frames, skip_observations, **kwargs) -> Trajectory:
        ...


    @abstractmethod
    def get_trajectory_paths(self) -> list[Path]:
        """Returns a list of paths to individual trajectories."""
        pass


class LeRobotDatasetLoader(BaseDatasetLoader):
    """Dataloader for the LeRobot v3.0 format (chunked parquet + concatenated MP4).

    Expected layout:
        <root>/
          meta/info.json
          meta/tasks.jsonl
          meta/episodes/chunk-000/file-000.parquet   (one row per episode)
          data/chunk-000/file-000.parquet             (frame-level rows)
          videos/<cam_key>/chunk-000/file-000.mp4
    """

    _invert_gripper: bool = True  # subclasses override if raw convention is already 0=closed,1=open

    def __init__(self, root_path: str, dataset_name: Optional[str] = None, view_key: Optional[str] = None,
                 starvla_trajectory_names: bool = False):
        self._view_key_pref = view_key
        self._chunk_size = 1000
        self._fps: float = 30.0
        self._cam_keys: list[str] = []
        self._path_template: Optional[str] = None
        self._data_path_template: Optional[str] = None
        self._task_map: dict[int, str] = {}
        self._episode_meta: dict[int, dict] = {}  # episode_index -> {chunk_idx, task_idx}
        # When True, trajectory keys follow the starVLA training convention
        # "{data_chunk_index}/{episode_index % chunks_size}" so annotation
        # trajectory_names match starVLA's CoT lookup key exactly.
        self._starvla_trajectory_names = starvla_trajectory_names
        self._key_to_ep: dict[str, int] = {}  # trajectory key -> global episode_index

        # Load meta before calling super() so caching uses valid state
        self._load_meta(Path(root_path))
        self.view_key = self._select_view_key(view_key)

        super().__init__(root_path, dataset_name=dataset_name or "lerobot")

    def _load_meta(self, root: Path) -> None:
        info_path = root / "meta" / "info.json"
        if info_path.exists():
            try:
                with open(info_path, "r", encoding="utf-8") as f:
                    info = json.load(f)
                self._chunk_size = int(info.get("chunks_size", info.get("chunk_size", 1000)))
                self._fps = float(info.get("fps", self._fps))
                features = info.get("features", {})
                self._cam_keys = [
                    k for k, meta in features.items()
                    if isinstance(meta, dict) and meta.get("dtype") in {"video", "image"}
                ]
                self._path_template = info.get("video_path", None)
                self._data_path_template = info.get("data_path", None)
            except Exception as e:
                logging.warning(f"LeRobotDatasetLoader: could not read info.json: {e}")

        tasks_path = root / "meta" / "tasks.jsonl"
        if tasks_path.exists():
            try:
                with open(tasks_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        item = json.loads(line)
                        idx = item.get("task_index", item.get("id"))
                        text = item.get("task", item.get("instruction", ""))
                        if idx is not None and isinstance(text, str):
                            self._task_map[int(idx)] = text
            except Exception as e:
                logging.warning(f"LeRobotDatasetLoader: could not read tasks.jsonl: {e}")

        tasks_parquet_path = root / "meta" / "tasks.parquet"
        if not self._task_map and tasks_parquet_path.exists():
            try:
                df_tasks = pd.read_parquet(tasks_parquet_path)
                if {"task_index", "task"}.issubset(df_tasks.columns):
                    for _, row in df_tasks.iterrows():
                        self._task_map[int(row["task_index"])] = str(row["task"])
                elif "task_index" in df_tasks.columns:
                    for task_str, row in df_tasks.iterrows():
                        self._task_map[int(row["task_index"])] = str(task_str)
                else:
                    for i, task_str in enumerate(df_tasks.index):
                        self._task_map[i] = str(task_str)
            except Exception as e:
                logging.warning(f"LeRobotDatasetLoader: could not read tasks.parquet: {e}")

    def _select_view_key(self, preferred: Optional[str]) -> str:
        if preferred and preferred in self._cam_keys:
            return preferred
        if self._cam_keys:
            return self._cam_keys[0]
        # Fall back to the preferred view or the default camera
        return preferred or "observation.images.cam_high"

    def get_trajectory_paths(self) -> list[str]:
        from tqdm import tqdm
        episodes_glob = sorted(glob.glob(
            str(self.root_path / "meta" / "episodes" / "chunk-*" / "file*.parquet")
        ))
        paths = []
        _rev_task_map: dict[str, int] = {}
        for ep_file in tqdm(episodes_glob, desc=f"Building episode index ({self.key})", unit="chunk"):
            ep_path = Path(ep_file)
            chunk_idx = int(ep_path.parent.name.replace("chunk-", ""))
            try:
                schema_names = set(_pq.read_schema(ep_file).names)
                cols = [
                    c for c in [
                        "episode_index",
                        "uuid",
                        "task_index",
                        "tasks",
                        "data/chunk_index",
                        "data/file_index",
                    ]
                    if c in schema_names
                ]
                for camera_key in self._cam_keys:
                    for suffix in ("chunk_index", "file_index", "from_timestamp"):
                        column = f"videos/{camera_key}/{suffix}"
                        if column in schema_names:
                            cols.append(column)
                df = pd.read_parquet(ep_file, columns=cols)
            except Exception as e:
                logging.warning(f"Skipping {ep_file}: {e}")
                continue
            for _, row in df.iterrows():
                ep_idx = int(row["episode_index"])
                task_idx = None
                task_text = None
                if "task_index" in row:
                    try:
                        task_idx = int(row["task_index"])
                    except Exception as e:
                        logging.warning(f"ep {ep_idx}: could not parse task_index: {e}")
                elif "tasks" in row:
                    tasks_val = row["tasks"]
                    if isinstance(tasks_val, (list, np.ndarray)) and len(tasks_val) > 0:
                        first = tasks_val[0]
                        try:
                            task_idx = int(first)
                        except Exception:
                            task_text = str(first)
                            if not _rev_task_map and self._task_map:
                                logging.info(f"Building reverse task map ({len(self._task_map)} entries)")
                                _rev_task_map = {v: k for k, v in self._task_map.items()}
                            task_idx = _rev_task_map.get(task_text)
                            if task_idx is None:
                                logging.debug(f"ep {ep_idx}: task text not in map, task_idx=None")
                vid_chunk_col = f"videos/{self.view_key}/chunk_index"
                vid_file_col  = f"videos/{self.view_key}/file_index"
                ts_col = f"videos/{self.view_key}/from_timestamp"
                video_sources = {}
                for camera_key in self._cam_keys:
                    camera_chunk_col = f"videos/{camera_key}/chunk_index"
                    camera_file_col = f"videos/{camera_key}/file_index"
                    camera_ts_col = f"videos/{camera_key}/from_timestamp"
                    if camera_chunk_col not in row.index or camera_file_col not in row.index:
                        continue
                    video_sources[camera_key] = {
                        "chunk_idx": int(row[camera_chunk_col]),
                        "file_idx": int(row[camera_file_col]),
                        "from_timestamp": (
                            float(row[camera_ts_col]) if camera_ts_col in row.index else 0.0
                        ),
                    }
                self._episode_meta[ep_idx] = {
                    "chunk_idx": chunk_idx,
                    "source_uuid": (
                        str(row["uuid"])
                        if "uuid" in row.index and pd.notna(row["uuid"])
                        else None
                    ),
                    "task_idx": task_idx,
                    "task_text": task_text,
                    "video_chunk_idx": int(row[vid_chunk_col]) if vid_chunk_col in row.index else None,
                    "video_file_idx":  int(row[vid_file_col])  if vid_file_col  in row.index else None,
                    "from_timestamp": float(row[ts_col]) if ts_col in row.index else 0.0,
                    "video_sources": video_sources,
                    "data_chunk_idx": int(row["data/chunk_index"]) if "data/chunk_index" in row.index else chunk_idx,
                    "data_file_idx": int(row["data/file_index"]) if "data/file_index" in row.index else ep_idx // self._chunk_size,
                }
                key = self._episode_key(chunk_idx, ep_idx)
                self._key_to_ep[key] = ep_idx
                paths.append(key)
        return paths

    def _episode_key(self, chunk_idx: int, ep_idx: int) -> str:
        """Trajectory key for an episode.

        Default: "{meta_chunk}/{global_episode_index}". With
        ``starvla_trajectory_names`` it matches starVLA's training key
        "{logical_chunk}/{episode_index % chunks_size}".
        """
        if self._starvla_trajectory_names:
            logical_chunk = ep_idx // self._chunk_size
            return f"{logical_chunk}/{ep_idx % self._chunk_size}"
        return f"{chunk_idx}/{ep_idx}"

    def _resolve_episode(self, path: str) -> tuple[int, int, Optional[int]]:
        """Map a trajectory key -> (chunk_idx, global_episode_idx, task_idx)."""
        key = str(path)
        if self._starvla_trajectory_names:
            if not self._key_to_ep:
                self.get_trajectory_paths()
            ep_idx = self._key_to_ep.get(key)
            if ep_idx is None:
                raise KeyError(f"Unknown starVLA trajectory key '{key}' for {self.key}")
            meta = self._episode_meta.get(ep_idx, {})
            chunk_idx = meta.get("chunk_idx", int(key.split("/")[-2]))
            return chunk_idx, ep_idx, meta.get("task_idx")
        parts = key.split("/")
        chunk_idx = int(parts[-2])
        ep_idx = int(parts[-1])
        meta = self._episode_meta.get(ep_idx, {})
        task_idx = meta.get("task_idx")
        return chunk_idx, ep_idx, task_idx

    def _video_path_for_view(self, chunk_idx: int, episode_idx: int, view_key: str) -> Path:
        cam_key = view_key
        meta = self._episode_meta.get(episode_idx, {})
        source = meta.get("video_sources", {}).get(view_key, {})
        vid_chunk = source.get("chunk_idx")
        vid_file = source.get("file_idx")
        if view_key == self.view_key:
            vid_chunk = meta.get("video_chunk_idx") if vid_chunk is None else vid_chunk
            vid_file = meta.get("video_file_idx") if vid_file is None else vid_file
        if vid_chunk is not None and vid_file is not None:
            if self._path_template:
                return self.root_path / self._path_template.format(
                    video_key=cam_key,
                    chunk_index=vid_chunk,
                    file_index=vid_file,
                )
            return self.root_path / "videos" / cam_key / f"chunk-{vid_chunk:03d}" / f"file_{vid_file:03d}.mp4"
        # Fallback for loaders that haven't populated episode meta yet
        file_idx = episode_idx // self._chunk_size
        if self._path_template:
            return self.root_path / self._path_template.format(
                video_key=cam_key,
                chunk_index=chunk_idx,
                file_index=file_idx,
            )
        return self.root_path / "videos" / cam_key / f"chunk-{chunk_idx:03d}" / f"file_{file_idx:03d}.mp4"

    def _video_path(self, chunk_idx: int, episode_idx: int) -> Path:
        return self._video_path_for_view(chunk_idx, episode_idx, self.view_key)

    def load_frames_at_indices(
        self,
        path: Path,
        frame_indices: list[int],
        view_key: Optional[str] = None,
    ) -> np.ndarray:
        """Decode selected trajectory frames from one camera without loading the full video."""
        if not frame_indices:
            return np.empty((0,), dtype=np.uint8)
        if not self._episode_meta:
            self.get_trajectory_paths()

        chunk_idx, ep_idx, _ = self._resolve_episode(str(path))
        selected_view = view_key or self.view_key
        if self._cam_keys and selected_view not in self._cam_keys:
            raise KeyError(f"Camera view {selected_view!r} is unavailable; choices={self._cam_keys}")

        data_path = self._data_parquet_path(chunk_idx, ep_idx)
        frame_df = pd.read_parquet(
            data_path,
            columns=["episode_index", "frame_index", "timestamp"],
        )
        frame_df = frame_df[frame_df["episode_index"] == ep_idx].sort_values("frame_index")
        if frame_df.empty:
            raise ValueError(f"No frames found in parquet for {path}")

        clipped = [max(0, min(int(idx), len(frame_df) - 1)) for idx in frame_indices]
        episode_meta = self._episode_meta.get(ep_idx, {})
        source = episode_meta.get("video_sources", {}).get(selected_view, {})
        default_from_timestamp = (
            episode_meta.get("from_timestamp", 0.0)
            if selected_view == self.view_key
            else 0.0
        )
        from_timestamp = float(source.get("from_timestamp", default_from_timestamp))
        video_path = self._video_path_for_view(chunk_idx, ep_idx, selected_view)
        if not video_path.exists():
            raise ValueError(f"Video not found for view {selected_view!r}: {video_path}")

        timestamps = frame_df.iloc[clipped]["timestamp"].to_numpy() + from_timestamp
        vr = VideoReader(str(video_path), ctx=decord_cpu(0))
        video_fps = vr.get_avg_fps()
        video_indices = [
            min(max(round(float(timestamp) * video_fps), 0), len(vr) - 1)
            for timestamp in timestamps
        ]
        return vr.get_batch(video_indices).asnumpy()

    def _data_parquet_path(self, chunk_idx: int, episode_idx: int) -> Path:
        meta = self._episode_meta.get(episode_idx, {})
        dc = meta.get("data_chunk_idx", chunk_idx)
        df = meta.get("data_file_idx", episode_idx // self._chunk_size)
        if self._data_path_template:
            return self.root_path / self._data_path_template.format(
                chunk_index=dc,
                file_index=df,
            )
        return self.root_path / "data" / f"chunk-{dc:03d}" / f"file_{df:03d}.parquet"

    @staticmethod
    def _stack_array_column(series: pd.Series) -> Optional[np.ndarray]:
        values = [np.asarray(v) for v in series.tolist()]
        if not values:
            return None
        try:
            return np.stack(values)
        except ValueError:
            return np.asarray(values)

    @staticmethod
    def _extract_dual_arm_eef_positions(series: pd.Series) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return left/right EEF xyz arrays from a possibly ragged series of shape-like (2, 3) entries.

        Preserves row alignment by filling malformed rows with NaN instead of dropping them.
        """
        n_rows = len(series)
        if n_rows == 0:
            return None, None

        left_positions = np.full((n_rows, 3), np.nan, dtype=np.float32)
        right_positions = np.full((n_rows, 3), np.nan, dtype=np.float32)
        valid_any = False

        for row_idx, value in enumerate(series.tolist()):
            try:
                arr = np.asarray(value)
                if arr.dtype == object:
                    arr = np.stack([np.asarray(item, dtype=np.float32) for item in arr.tolist()])
                else:
                    arr = arr.astype(np.float32, copy=False)
            except Exception as e:
                    logging.warning(f"Row {row_idx}: could not convert to array: {e}")
            if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 3:
                continue
            left_positions[row_idx] = arr[0, :3]
            right_positions[row_idx] = arr[1, :3]
            valid_any = True

        if not valid_any:
            return None, None
        return left_positions, right_positions

    @staticmethod
    def _extract_pose7_column(series: pd.Series) -> Optional[np.ndarray]:
        """Return a float32 array of shape (T, 7) from a pose-like series, preserving row alignment."""
        n_rows = len(series)
        if n_rows == 0:
            return None

        poses = np.full((n_rows, 7), np.nan, dtype=np.float32)
        valid_any = False

        for row_idx, value in enumerate(series.tolist()):
            try:
                arr = np.asarray(value)
                if arr.dtype == object:
                    arr = np.asarray(arr.tolist(), dtype=np.float32)
                else:
                    arr = arr.astype(np.float32, copy=False)
                arr = arr.reshape(-1)
            except Exception as e:
                logging.warning(f"Row {row_idx}: could not convert pose7 to array: {e}")
                continue
            if arr.shape[0] < 7:
                continue
            poses[row_idx] = arr[:7]
            valid_any = True

        if not valid_any:
            return None
        return poses

    def _observation_columns(self, data_path: Path) -> list[str]:
        schema_names = set(_pq.read_schema(data_path).names)
        cols = ["episode_index"]
        for candidate in (
            "task_index",
            "action",
            "action.gripper_position",
            "action.cartesian_velocity",
            "action.cartesian",
            "action.cartesian_position",
        ):
            if candidate in schema_names:
                cols.append(candidate)
        return cols

    def _populate_task_annotation(
        self,
        traj: Trajectory,
        ep_df: pd.DataFrame,
        ep_idx: int,
    ) -> None:
        """Fill a missing instruction from frame-level LeRobot task metadata."""
        if traj.lang_ann or "task_index" not in ep_df.columns:
            return

        task_indices = ep_df["task_index"].dropna().astype(int).unique()
        if len(task_indices) == 0:
            return
        if len(task_indices) > 1:
            logging.warning(
                "Episode %s contains multiple task indices %s; using the first",
                ep_idx,
                task_indices.tolist(),
            )

        task_idx = int(task_indices[0])
        task_text = self._task_map.get(task_idx, "")
        if not task_text:
            logging.warning("Episode %s references unknown task_index=%s", ep_idx, task_idx)
            return

        traj.lang_ann = str(task_text).lower()
        self._episode_meta.setdefault(ep_idx, {})["task_idx"] = task_idx

    def _populate_observations(self, traj: Trajectory, ep_df: pd.DataFrame) -> None:
        if len(ep_df) == 0:
            return

        if "action" in ep_df.columns:
            action = self._stack_array_column(ep_df["action"])
            if action is not None:
                if action.ndim == 2 and action.shape[1] >= 1:
                    traj.gripper_state = (1 - action[:, -1]) if self._invert_gripper else action[:, -1]
                if action.ndim == 2 and action.shape[1] >= 6:
                    traj.gripper_velocity = action[:, :6]
            return

        if "action.gripper_position" in ep_df.columns:
            gripper = self._stack_array_column(ep_df["action.gripper_position"])
            if gripper is not None:
                gripper = gripper.reshape(len(gripper), -1)
                traj.gripper_state = (1 - gripper[:, 0]) if self._invert_gripper else gripper[:, 0]

        velocity_col = None

        if "action.cartesian" in ep_df.columns:
            velocity_col = "action.cartesian"
        elif "action.cartesian_position" in ep_df.columns:
            velocity_col = "action.cartesian_position"
        elif "action.cartesian_velocity" in ep_df.columns:
            velocity_col = "action.cartesian_velocity"

        if velocity_col is not None:
            velocity = self._stack_array_column(ep_df[velocity_col])
            if velocity is not None:
                velocity = velocity.reshape(len(velocity), -1)
                traj.gripper_velocity = velocity[:, : min(6, velocity.shape[1])]

    def load_trajectory(
        self,
        path: Path,
        load_frames: bool = False,
        skip_frames: bool = True,
        skip_observations: bool = True,
        **kwargs,
    ) -> Optional[Trajectory]:
        path_str = str(path)
        # Rebuild episode_meta if empty (e.g. after loading from cache)
        if not self._episode_meta:
            self.get_trajectory_paths()

        chunk_idx, ep_idx, task_idx = self._resolve_episode(path_str)
        meta = self._episode_meta.get(ep_idx, {})
        lang_ann = meta.get("task_text") or (self._task_map.get(task_idx, "") if task_idx is not None else "")
        lang_ann = lang_ann.lower()
        video_path = self._video_path(chunk_idx, ep_idx)
        split = self.get_split_type(path)

        traj = Trajectory(
            name=path_str,
            lang_ann=lang_ann,
            base_path=str(self.root_path),
            dataset_name=self.key,
            media_dir=str(video_path),
            split=split,
        )
        traj.fps = self._fps
        traj.camera_view_key = self.view_key
        traj.source_uuid = meta.get("source_uuid")
        traj.source_episode_index = ep_idx
        traj.source_dataset_format = "lerobot_v3"

        if not skip_observations:
            data_path = self._data_parquet_path(chunk_idx, ep_idx)
            if data_path.exists():
                try:
                    obs_cols = self._observation_columns(data_path)
                    df = pd.read_parquet(data_path, columns=obs_cols)
                    ep_df = df[df["episode_index"] == ep_idx]
                    self._populate_task_annotation(traj, ep_df, ep_idx)
                    self._populate_observations(traj, ep_df)
                except Exception as e:
                    logging.warning(f"LeRobotDatasetLoader: could not load data parquet for {path}: {e}")

        if not skip_frames:
            if not video_path.exists():
                raise ValueError(f"Video not found: {video_path}")

            from_ts = self._episode_meta.get(ep_idx, {}).get("from_timestamp", 0.0)
            data_path = self._data_parquet_path(chunk_idx, ep_idx)
            if not data_path.exists():
                raise ValueError(f"Data parquet not found: {data_path}")

            frame_df = pd.read_parquet(data_path, columns=["episode_index", "frame_index", "timestamp"])
            frame_df = frame_df[frame_df["episode_index"] == ep_idx].sort_values("frame_index")
            if frame_df.empty:
                raise ValueError(f"No frames found in parquet for {path}")

            traj.source_episode_frame_indices = frame_df["frame_index"].to_numpy(
                dtype=np.int64,
            )
            traj.frame_timestamps_seconds = frame_df["timestamp"].to_numpy(
                dtype=np.float64,
            )

            shifted_timestamps = (frame_df["timestamp"].values + from_ts).tolist()
            vr = VideoReader(str(video_path), ctx=decord_cpu(0))
            fps = vr.get_avg_fps()
            frame_indices = [min(max(round(ts * fps), 0), len(vr) - 1) for ts in shifted_timestamps]
            traj.frames = vr.get_batch(frame_indices).asnumpy()

        return traj


class BridgeLeRobotDatasetLoader(LeRobotDatasetLoader):
    """LeRobot v3 BridgeData loader."""

    _invert_gripper = False  # raw action.gripper_position: 1=open, 0=closed — already matches pipeline convention

    BRIDGE_CAM_PRIORITY = [
        "observation.images.camera_0",
        "observation.images.camera_1",
        "observation.images.camera_2",
        "observation.images.camera_3",
        "observation.images.camera_4",
    ]

    def __init__(self, root_path: str, view_key: Optional[str] = None, dataset_name: str = "bridge_lerobot"):
        super().__init__(root_path, dataset_name=dataset_name, view_key=view_key)

    def _select_view_key(self, preferred: Optional[str]) -> str:
        if preferred and preferred in self._cam_keys:
            return preferred
        for key in self.BRIDGE_CAM_PRIORITY:
            if key in self._cam_keys:
                return key
        return super()._select_view_key(preferred)


class DroidLeRobotDatasetLoader(LeRobotDatasetLoader):
    """LeRobot v3 DROID loader."""

    DROID_CAM_PRIORITY = [
        "observation.images.left_external",
        "observation.images.right_external",
        "observation.images.wrist",
    ]

    def __init__(self, root_path: str, view_key: Optional[str] = None, dataset_name: str = "droid_lerobot"):
        super().__init__(root_path, dataset_name=dataset_name, view_key=view_key)

    def _select_view_key(self, preferred: Optional[str]) -> str:
        if preferred and preferred in self._cam_keys:
            return preferred
        for key in self.DROID_CAM_PRIORITY:
            if key in self._cam_keys:
                return key
        return super()._select_view_key(preferred)

    def _observation_columns(self, data_path: Path) -> list[str]:
        cols = super()._observation_columns(data_path)
        schema_names = set(_pq.read_schema(data_path).names)
        view_to_col = {
            "observation.images.left_external": "left_external_camera_to_robot_extrinsics",
            "observation.images.right_external": "right_external_camera_to_robot_extrinsics",
            "observation.images.wrist": "wrist_camera_to_robot_extrinsics",
        }
        extr_col = view_to_col.get(self.view_key)
        if extr_col in schema_names:
            cols.append(extr_col)
        return cols

    def _populate_observations(self, traj: Trajectory, ep_df: pd.DataFrame) -> None:
        super()._populate_observations(traj, ep_df)
        view_to_col = {
            "observation.images.left_external": "left_external_camera_to_robot_extrinsics",
            "observation.images.right_external": "right_external_camera_to_robot_extrinsics",
            "observation.images.wrist": "wrist_camera_to_robot_extrinsics",
        }
        extr_col = view_to_col.get(self.view_key)
        if extr_col in ep_df.columns:
            traj.camera_extrinsics_cam_to_robot = self._extract_pose7_column(ep_df[extr_col])
            traj.camera_extrinsics_view_key = self.view_key


class RoboMind2DatasetLoader(LeRobotDatasetLoader):
    """LeRobot v3 RoboMIND2 loader."""

    _invert_gripper = True

    ROBOMIND2_CAM_PRIORITY = [
        "observation.images.camera_front",
        "observation.images.camera_top",
        "observation.images.camera_left",
        "observation.images.camera_right",
        "observation.images.camera_wrist_left",
        "observation.images.camera_wrist_right",
    ]
    ROBOMIND2_SAMPLED_CAM_KEYS = [
        "observation.images.camera_front",
        "observation.images.camera_top",
    ]

    def __init__(self, root_path: str, view_key: Optional[str] = None, dataset_name: str = "robomind2"):
        super().__init__(root_path, dataset_name=dataset_name, view_key=view_key)

    def _select_view_key(self, preferred: Optional[str]) -> str:
        if preferred and preferred in self._cam_keys:
            return preferred
        for key in self.ROBOMIND2_CAM_PRIORITY:
            if key in self._cam_keys:
                return key
        return super()._select_view_key(preferred)

    def _available_episode_views(self, row: pd.Series) -> list[str]:
        available = []
        for cam_key in self.ROBOMIND2_CAM_PRIORITY:
            chunk_col = f"videos/{cam_key}/chunk_index"
            file_col = f"videos/{cam_key}/file_index"
            if chunk_col in row.index and file_col in row.index:
                if pd.notna(row[chunk_col]) and pd.notna(row[file_col]):
                    available.append(cam_key)
        return available

    def _choose_episode_view_key(self, ep_idx: int, available_views: list[str]) -> str:
        if self._view_key_pref and self._view_key_pref in available_views:
            return self._view_key_pref

        sampled_views = [
            cam_key for cam_key in self.ROBOMIND2_SAMPLED_CAM_KEYS
            if cam_key in available_views
        ]
        if len(sampled_views) == 2:
            return sampled_views[ep_idx % 2]
        if len(sampled_views) == 1:
            return sampled_views[0]
        if available_views:
            return available_views[0]
        return self.view_key

    def get_trajectory_paths(self) -> list[str]:
        if self._view_key_pref is not None:
            return super().get_trajectory_paths()

        episodes_glob = sorted(glob.glob(
            str(self.root_path / "meta" / "episodes" / "chunk-*" / "file*.parquet")
        ))
        paths = []
        self._episode_meta = {}
        _rev_task_map: dict[str, int] = {}

        for ep_file in tqdm(episodes_glob, desc=f"Building episode index ({self.key})", unit="chunk"):
            ep_path = Path(ep_file)
            chunk_idx = int(ep_path.parent.name.replace("chunk-", ""))
            try:
                schema_names = set(_pq.read_schema(ep_file).names)
                cols = [
                    c for c in [
                        "episode_index",
                        "task_index",
                        "tasks",
                        "data/chunk_index",
                        "data/file_index",
                    ]
                    if c in schema_names
                ]
                for cam_key in self.ROBOMIND2_CAM_PRIORITY:
                    for suffix in ("chunk_index", "file_index", "from_timestamp"):
                        col = f"videos/{cam_key}/{suffix}"
                        if col in schema_names:
                            cols.append(col)
                df = pd.read_parquet(ep_file, columns=cols)
            except Exception as e:
                logging.warning(f"Skipping {ep_file}: {e}")
                continue

            for _, row in df.iterrows():
                ep_idx = int(row["episode_index"])
                task_idx = None
                task_text = None
                if "task_index" in row:
                    try:
                        task_idx = int(row["task_index"])
                    except Exception as e:
                        logging.warning(f"ep {ep_idx}: could not parse task_index: {e}")
                elif "tasks" in row:
                    tasks_val = row["tasks"]
                    if isinstance(tasks_val, (list, np.ndarray)) and len(tasks_val) > 0:
                        first = tasks_val[0]
                        try:
                            task_idx = int(first)
                        except Exception:
                            task_text = str(first)
                            if not _rev_task_map and self._task_map:
                                logging.info(f"Building reverse task map ({len(self._task_map)} entries)")
                                _rev_task_map = {v: k for k, v in self._task_map.items()}
                            task_idx = _rev_task_map.get(task_text)
                            if task_idx is None:
                                logging.debug(f"ep {ep_idx}: task text not in map, task_idx=None")

                available_views = self._available_episode_views(row)
                selected_view_key = self._choose_episode_view_key(ep_idx, available_views)
                vid_chunk_col = f"videos/{selected_view_key}/chunk_index"
                vid_file_col = f"videos/{selected_view_key}/file_index"
                ts_col = f"videos/{selected_view_key}/from_timestamp"

                self._episode_meta[ep_idx] = {
                    "chunk_idx": chunk_idx,
                    "task_idx": task_idx,
                    "task_text": task_text,
                    "selected_view_key": selected_view_key,
                    "video_chunk_idx": int(row[vid_chunk_col]) if vid_chunk_col in row.index else None,
                    "video_file_idx": int(row[vid_file_col]) if vid_file_col in row.index else None,
                    "from_timestamp": float(row[ts_col]) if ts_col in row.index else 0.0,
                    "data_chunk_idx": int(row["data/chunk_index"]) if "data/chunk_index" in row.index else chunk_idx,
                    "data_file_idx": int(row["data/file_index"]) if "data/file_index" in row.index else ep_idx // self._chunk_size,
                }
                paths.append(f"{chunk_idx}/{ep_idx}")

        return paths

    def _video_path(self, chunk_idx: int, episode_idx: int) -> Path:
        meta = self._episode_meta.get(episode_idx, {})
        cam_key = meta.get("selected_view_key", self.view_key)
        vid_chunk = meta.get("video_chunk_idx")
        vid_file = meta.get("video_file_idx")
        if vid_chunk is not None and vid_file is not None:
            if self._path_template:
                return self.root_path / self._path_template.format(
                    video_key=cam_key,
                    chunk_index=vid_chunk,
                    file_index=vid_file,
                )
            return self.root_path / "videos" / cam_key / f"chunk-{vid_chunk:03d}" / f"file-{vid_file:03d}.mp4"

        file_idx = episode_idx // self._chunk_size
        if self._path_template:
            return self.root_path / self._path_template.format(
                video_key=cam_key,
                chunk_index=chunk_idx,
                file_index=file_idx,
            )
        return self.root_path / "videos" / cam_key / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4"

    def _observation_columns(self, data_path: Path) -> list[str]:
        schema_names = set(_pq.read_schema(data_path).names)
        cols = ["episode_index"]
        for candidate in (
            "observation.states.gripper_left.position",
            "observation.states.gripper_right.position",
            "observation.states.arm_left.joint_positions",
            "observation.states.arm_right.joint_positions",
            "observation.states.end.position",
        ):
            if candidate in schema_names:
                cols.append(candidate)
        return cols

    def _populate_observations(self, traj: Trajectory, ep_df: pd.DataFrame) -> None:
        if len(ep_df) == 0:
            return

        left_state = None
        right_state = None

        if "observation.states.gripper_left.position" in ep_df.columns:
            left_state = np.asarray(
                ep_df["observation.states.gripper_left.position"].tolist(),
                dtype=np.float32,
            ).reshape(-1)
            if self._invert_gripper:
                left_state = 1 - left_state
            traj.left_gripper_state = left_state

        if "observation.states.gripper_right.position" in ep_df.columns:
            right_state = np.asarray(
                ep_df["observation.states.gripper_right.position"].tolist(),
                dtype=np.float32,
            ).reshape(-1)
            if self._invert_gripper:
                right_state = 1 - right_state
            traj.right_gripper_state = right_state

        if left_state is not None:
            traj.gripper_state = left_state
        elif right_state is not None:
            traj.gripper_state = right_state

        if "observation.states.end.position" in ep_df.columns:
            left_eef, right_eef = self._extract_dual_arm_eef_positions(
                ep_df["observation.states.end.position"]
            )
            traj.left_eef_position = left_eef
            traj.right_eef_position = right_eef

        primary_arm_col = None
        if "observation.states.arm_left.joint_positions" in ep_df.columns:
            primary_arm_col = "observation.states.arm_left.joint_positions"
        elif "observation.states.arm_right.joint_positions" in ep_df.columns:
            primary_arm_col = "observation.states.arm_right.joint_positions"

        if primary_arm_col is not None:
            joint_pos = self._stack_array_column(ep_df[primary_arm_col])
            if joint_pos is not None:
                joint_pos = joint_pos.reshape(len(joint_pos), -1)
                traj.gripper_velocity = joint_pos[:, : min(6, joint_pos.shape[1])]


class ConfigurableLeRobotDatasetLoader(LeRobotDatasetLoader):
    """Config-driven LeRobot loader for a single dataset root.

    Lets callers specify camera priority and gripper extraction behavior without
    creating a dedicated subclass for each dataset family.
    """

    _invert_gripper = False

    def __init__(
        self,
        root_path: str,
        dataset_name: str,
        view_key: Optional[str] = None,
        camera_priority: Optional[list[str]] = None,
        gripper: Optional[dict[str, Any]] = None,
        starvla_trajectory_names: bool = False,
    ):
        self._camera_priority = list(camera_priority or [])
        self._gripper_cfg = dict(gripper or {})
        super().__init__(root_path, dataset_name=dataset_name, view_key=view_key,
                         starvla_trajectory_names=starvla_trajectory_names)

    def _select_view_key(self, preferred: Optional[str]) -> str:
        if preferred and preferred in self._cam_keys:
            return preferred
        for key in self._camera_priority:
            if key in self._cam_keys:
                return key
        return super()._select_view_key(preferred)

    def _observation_columns(self, data_path: Path) -> list[str]:
        schema_names = set(_pq.read_schema(data_path).names)
        cols = ["episode_index"]
        for candidate in ("task_index", "action", "observation.state"):
            if candidate in schema_names:
                cols.append(candidate)
        return cols

    @staticmethod
    def _pad_velocity(values: np.ndarray, n_rows: int) -> np.ndarray:
        values = values.reshape(n_rows, -1)
        out = np.zeros((n_rows, 6), dtype=np.float32)
        width = min(6, values.shape[1])
        if width > 0:
            out[:, :width] = values[:, :width]
        return out

    @staticmethod
    def _transform_gripper_signal(values: np.ndarray, transform: str) -> np.ndarray:
        values = values.astype(np.float32, copy=False).reshape(-1)
        transform = (transform or "identity").lower()

        if transform in {"identity", "open01"}:
            return values
        if transform in {"invert", "close01"}:
            return 1.0 - values
        if transform in {"neg1pos1_to_01", "minus1_plus1_to_01"}:
            return (values + 1.0) / 2.0
        if transform in {"neg1open_pos1close_to_01", "minus1_open_plus1_close_to_01"}:
            return (1.0 - values) / 2.0
        if transform == "normalize":
            lo = float(np.nanmin(values))
            hi = float(np.nanmax(values))
            if hi - lo < 1e-8:
                return np.zeros_like(values, dtype=np.float32)
            return (values - lo) / (hi - lo)

        raise ValueError(f"Unknown LeRobot gripper transform '{transform}'")

    def _extract_gripper_signal(self, ep_df: pd.DataFrame) -> Optional[np.ndarray]:
        source = self._gripper_cfg.get("source", "action")
        index = int(self._gripper_cfg.get("index", -1))
        transform = self._gripper_cfg.get("transform", "identity")

        if source not in ep_df.columns:
            logging.warning(f"{self.key}: configured gripper source '{source}' not in parquet columns")
            return None

        arr = self._stack_array_column(ep_df[source])
        if arr is None:
            return None
        arr = arr.reshape(len(ep_df), -1)
        signal = arr[:, index]
        return self._transform_gripper_signal(signal, transform)

    def _populate_observations(self, traj: Trajectory, ep_df: pd.DataFrame) -> None:
        if len(ep_df) == 0:
            return

        gripper_state = self._extract_gripper_signal(ep_df)
        if gripper_state is not None:
            traj.gripper_state = gripper_state

        if "action" in ep_df.columns:
            action = self._stack_array_column(ep_df["action"])
            if action is not None:
                traj.gripper_velocity = self._pad_velocity(action, len(ep_df))
        elif "observation.state" in ep_df.columns:
            state = self._stack_array_column(ep_df["observation.state"])
            if state is not None:
                state = state.reshape(len(ep_df), -1)
                width = min(6, state.shape[1])
                velocity = np.diff(
                    state[:, :width],
                    axis=0,
                    prepend=state[0:1, :width],
                )
                traj.gripper_velocity = self._pad_velocity(velocity, len(ep_df))


class OXELeRobotSingleDatasetLoader(ConfigurableLeRobotDatasetLoader):
    """Single Open X-Embodiment subdataset in LeRobot v3 format.

    OXE subdatasets do not share one gripper convention, so gripper extraction is
    driven by the wrapper config instead of a class-level inversion flag.
    The emitted ``gripper_state`` follows the pipeline convention: lower values
    are closed, higher values are open; binary signals are 0=closed, 1=open.
    """


class OXELeRobotDatasetLoader(BaseDatasetLoader):
    """Multi-subdataset loader for LeRobot-converted Open X-Embodiment data.

    Path format: ``"{subdataset}/{chunk_idx}/{episode_idx}"``.
    Per-subdataset camera choice, FPS, and gripper source/transform live in
    ``configs/dataset/oxe_config/oxe_lerobot.yaml`` by default.
    """

    DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "dataset" / "oxe_config" / "oxe_lerobot.yaml"

    def __init__(
        self,
        root_path: str,
        dataset_name: str = "oxe_lerobot",
        config_path: Optional[str] = None,
        subdatasets: Optional[list[str]] = None,
        view_key: Optional[str] = None,
    ):
        self._public_dataset_name = dataset_name
        self._view_key_override = view_key
        self._config_path = Path(config_path) if config_path else self.DEFAULT_CONFIG_PATH
        self._config = self._load_config(self._config_path)
        self._subdataset_specs = self._resolve_subdataset_specs(Path(root_path), subdatasets)
        self._loaders: dict[str, OXELeRobotSingleDatasetLoader] = {}
        self._subdataset_roots: dict[str, Path] = {}
        self.dataset_map: dict[str, list[str]] = {}
        self.robot_map = self.dataset_map  # lets existing diverse debug sampling reuse the RoboCOIN path

        for subdataset, spec in self._subdataset_specs.items():
            sub_root = Path(spec["root"]) if spec.get("root") else Path(root_path) / subdataset
            if not sub_root.exists():
                logging.warning(f"OXELeRobotDatasetLoader: missing subdataset root {sub_root}")
                continue
            self._loaders[subdataset] = OXELeRobotSingleDatasetLoader(
                str(sub_root),
                dataset_name=f"{dataset_name}_{subdataset}",
                view_key=self._view_key_override or spec.get("view_key"),
                camera_priority=spec.get("camera_priority", []),
                gripper=spec.get("gripper", {}),
            )
            self._subdataset_roots[subdataset] = sub_root

        cache_dataset_name = self._cache_dataset_name(dataset_name, subdatasets)
        super().__init__(root_path, dataset_name=cache_dataset_name)

        expected_subdatasets = set(self._loaders)
        cached_subdatasets = {str(traj).split("/", 1)[0] for traj in self.trajectories}
        if cached_subdatasets != expected_subdatasets:
            logging.info(
                "OXELeRobotDatasetLoader: refreshing trajectory cache for subdatasets %s",
                sorted(expected_subdatasets),
            )
            self.cached["trajectories"] = self.get_trajectory_paths()
            self.trajectories = [Path(p) for p in self.cached["trajectories"]]
            self.save()

        self.dataset_map = self.cached.get("dataset_map", self.dataset_map)
        if not self.dataset_map:
            self._rebuild_dataset_map()
        self.robot_map = self.dataset_map

    @staticmethod
    def _cache_dataset_name(dataset_name: str, subdatasets: Optional[list[str]]) -> str:
        if not subdatasets:
            return dataset_name
        safe = "_".join(str(name).replace("/", "_") for name in subdatasets)
        return f"{dataset_name}__{safe}"

    @staticmethod
    def _load_config(path: Path) -> dict[str, Any]:
        if not path.exists():
            logging.warning(f"OXELeRobotDatasetLoader: config not found at {path}; discovering all subdatasets")
            return {}
        from omegaconf import OmegaConf
        return OmegaConf.to_container(OmegaConf.load(path), resolve=True) or {}

    def _resolve_subdataset_specs(self, root: Path, requested: Optional[list[str]]) -> dict[str, dict[str, Any]]:
        configured = self._config.get("subdatasets", {})
        if requested:
            names = requested
        elif configured:
            names = [name for name, spec in configured.items() if spec is None or spec.get("enabled", True)]
        else:
            names = [p.name for p in sorted(root.iterdir()) if (p / "meta" / "info.json").exists()]

        specs: dict[str, dict[str, Any]] = {}
        for name in names:
            spec = dict(configured.get(name, {}) or {})
            if not spec.get("enabled", True):
                continue
            specs[name] = spec
        return specs

    def _rebuild_dataset_map(self) -> None:
        dataset_map: dict[str, list[str]] = {}
        for traj in self.trajectories:
            path_str = str(traj)
            subdataset = path_str.split("/", 1)[0]
            dataset_map.setdefault(subdataset, []).append(path_str)
        self.dataset_map = dataset_map

    def sample_diverse(
        self,
        n_per_dataset: Optional[int] = None,
        n_per_robot: Optional[int] = None,
        n_total: Optional[int] = None,
        seed: int = 42,
        strategy: str = "proportional",
        min_per_subdataset: int = 1,
    ) -> list[str]:
        """Return a deterministic sample across OXE subdatasets."""
        if n_total is not None:
            return self.sample_diverse_total(
                n_total=n_total,
                seed=seed,
                strategy=strategy,
                min_per_subdataset=min_per_subdataset,
            )

        rng = random.Random(seed)
        n = n_per_dataset if n_per_dataset is not None else n_per_robot
        n = 10 if n is None else int(n)
        result = []
        for subdataset, paths in sorted(self.dataset_map.items()):
            result.extend(rng.sample(paths, min(n, len(paths))))
        return result

    def sample_diverse_total(
        self,
        n_total: int,
        seed: int = 42,
        strategy: str = "proportional",
        min_per_subdataset: int = 1,
    ) -> list[str]:
        """Return exactly up to ``n_total`` samples across OXE subdatasets.

        ``proportional`` samples by discovered subdataset size, ``balanced`` gives
        each subdataset equal weight, and ``weighted`` uses config
        ``sample_ratio`` values. Counts are allocated deterministically by largest
        remainder, then paths are sampled with a fixed RNG seed within each
        subdataset.
        """
        rng = random.Random(seed)
        n_total = max(0, int(n_total))
        strategy = (strategy or "proportional").lower()
        min_per_subdataset = max(0, int(min_per_subdataset))
        available = {
            name: paths
            for name, paths in sorted(self.dataset_map.items())
            if paths
        }
        if n_total == 0 or not available:
            return []

        if strategy == "weighted":
            weights = {
                name: max(0.0, float(self._subdataset_specs.get(name, {}).get("sample_ratio", 1.0)))
                for name in available
            }
        elif strategy == "balanced":
            weights = {name: 1.0 for name in available}
        elif strategy == "proportional":
            weights = {name: float(len(paths)) for name, paths in available.items()}
        else:
            raise ValueError(
                "Unknown OXE sampling strategy "
                f"'{strategy}'. Expected one of: proportional, balanced, weighted"
            )
        if sum(weights.values()) <= 0:
            weights = {name: 1.0 for name in available}

        remaining_budget = min(n_total, sum(len(paths) for paths in available.values()))
        counts = {name: 0 for name in available}

        if min_per_subdataset > 0:
            floor_candidates = set(available)
            while remaining_budget > 0 and floor_candidates:
                ranked = sorted(
                    floor_candidates,
                    key=lambda name: (weights[name], len(available[name]), name),
                    reverse=True,
                )
                made_progress = False
                for name in ranked:
                    if remaining_budget <= 0:
                        break
                    if counts[name] >= min_per_subdataset or counts[name] >= len(available[name]):
                        floor_candidates.discard(name)
                        continue
                    counts[name] += 1
                    remaining_budget -= 1
                    made_progress = True
                    if counts[name] >= min_per_subdataset or counts[name] >= len(available[name]):
                        floor_candidates.discard(name)
                if not made_progress:
                    break

        active = {
            name: paths
            for name, paths in available.items()
            if counts[name] < len(paths)
        }

        while remaining_budget > 0 and active:
            active_weight = sum(weights[name] for name in active)
            if active_weight <= 0:
                active_weight = float(len(active))
                active_weights = {name: 1.0 for name in active}
            else:
                active_weights = weights

            raw = {
                name: remaining_budget * active_weights[name] / active_weight
                for name in active
            }
            proposed = {
                name: min(len(active[name]) - counts[name], int(np.floor(value)))
                for name, value in raw.items()
            }
            assigned = sum(proposed.values())
            leftovers = remaining_budget - assigned
            remainders = sorted(
                active,
                key=lambda name: (raw[name] - np.floor(raw[name]), active_weights[name], name),
                reverse=True,
            )
            for name in remainders:
                if leftovers <= 0:
                    break
                capacity = len(active[name]) - counts[name] - proposed[name]
                if capacity <= 0:
                    continue
                proposed[name] += 1
                leftovers -= 1

            made_progress = False
            for name, count in proposed.items():
                if count <= 0:
                    continue
                counts[name] += count
                remaining_budget -= count
                made_progress = True

            active = {
                name: paths
                for name, paths in active.items()
                if counts[name] < len(paths)
            }
            if not made_progress:
                break

        result = []
        for subdataset, paths in sorted(available.items()):
            count = min(counts[subdataset], len(paths))
            if count:
                result.extend(rng.sample(paths, count))
        return result

    def get_trajectory_paths(self) -> list[str]:
        paths = []
        self.dataset_map = {}
        for subdataset, loader in sorted(self._loaders.items()):
            prefixed = [f"{subdataset}/{traj_path}" for traj_path in loader.trajectories]
            self.dataset_map[subdataset] = prefixed
            paths.extend(prefixed)
        self.cached["dataset_map"] = self.dataset_map
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
            logging.warning(f"OXELeRobotDatasetLoader: invalid path {path_str}, expected 'subdataset/chunk/episode'")
            return None

        subdataset, sub_path = parts
        loader = self._loaders.get(subdataset)
        if loader is None:
            logging.warning(f"OXELeRobotDatasetLoader: unknown subdataset '{subdataset}'")
            return None

        traj = loader.load_trajectory(
            Path(sub_path),
            load_frames=load_frames,
            skip_frames=skip_frames,
            skip_observations=skip_observations,
            **kwargs,
        )
        if traj is not None:
            traj.name = f"{subdataset}/{traj.name}"
            traj.base_path = str(self._subdataset_roots.get(subdataset, self.root_path / subdataset))
            traj.dataset_name = self._public_dataset_name
        return traj

class VLAArenaDatasetLoader(BaseDatasetLoader):
    """Dataloader for VLA-Arena (LeRobot v2.1 / openpi-style, per-episode parquet).

    Unlike the v3.0 loaders, there is no ``videos/`` directory: each episode is a
    single parquet file with one row per frame, and camera images are embedded
    per-row as PNG bytes (HuggingFace ``Image`` feature: ``{"bytes": ..., "path": ...}``).

    Directory layout::

        <root>/
          meta/info.json
          meta/tasks.jsonl        {"task_index": int, "task": str}
          meta/episodes.jsonl     {"episode_index": int, "tasks": [str], "length": int}
          data/chunk-*/episode_*.parquet   (columns: image, wrist_image, state, actions, ...)

    Camera keys: ``image`` (third-person, default) and ``wrist_image``.

    Gripper convention: raw ``actions[:, -1]`` follows the LIBERO/openpi convention
    (-1 = open, +1 = close). Transformed to the pipeline convention (0 = closed,
    1 = open) via ``gripper_state = (1 - action) / 2``.
    """

    VIEW_KEYS = ["image", "wrist_image"]

    def __init__(self, root_path: str, dataset_name: Optional[str] = None, view_key: Optional[str] = None):
        self._fps = 10.0
        self._task_map: dict[int, str] = {}
        self._episode_lang: dict[int, str] = {}

        self._load_meta(Path(root_path))
        self.view_key = view_key if view_key in self.VIEW_KEYS else "image"

        super().__init__(root_path, dataset_name=dataset_name or "vla_arena")

    def _load_meta(self, root: Path) -> None:
        info_path = root / "meta" / "info.json"
        if info_path.exists():
            try:
                with open(info_path, "r", encoding="utf-8") as f:
                    info = json.load(f)
                self._fps = float(info.get("fps", self._fps))
            except Exception as e:
                logging.warning(f"VLAArenaDatasetLoader: could not read info.json: {e}")

        tasks_path = root / "meta" / "tasks.jsonl"
        if tasks_path.exists():
            try:
                with open(tasks_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        item = json.loads(line)
                        self._task_map[int(item["task_index"])] = str(item["task"])
            except Exception as e:
                logging.warning(f"VLAArenaDatasetLoader: could not read tasks.jsonl: {e}")

        episodes_path = root / "meta" / "episodes.jsonl"
        if episodes_path.exists():
            try:
                with open(episodes_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        item = json.loads(line)
                        tasks = item.get("tasks") or []
                        if tasks:
                            self._episode_lang[int(item["episode_index"])] = str(tasks[0])
            except Exception as e:
                logging.warning(f"VLAArenaDatasetLoader: could not read episodes.jsonl: {e}")

    def get_trajectory_paths(self) -> list[str]:
        paths = [
            str(p.relative_to(self.root_path).with_suffix(""))
            for p in sorted(self.root_path.glob("data/chunk-*/episode_*.parquet"))
        ]
        return paths

    @staticmethod
    def _episode_index_from_path(path_str: str) -> int:
        return int(Path(path_str).stem.replace("episode_", ""))

    def load_trajectory(
        self,
        path: Path,
        load_frames: bool = False,
        skip_frames: bool = True,
        skip_observations: bool = True,
        **kwargs,
    ) -> Optional[Trajectory]:
        path_str = str(path)
        parquet_path = self.root_path / f"{path_str}.parquet"
        ep_idx = self._episode_index_from_path(path_str)
        lang_ann = self._episode_lang.get(ep_idx, "").lower()
        split = self.get_split_type(path)

        traj = Trajectory(
            name=path_str,
            lang_ann=lang_ann,
            base_path=str(self.root_path),
            dataset_name=self.key,
            media_dir=str(parquet_path),
            split=split,
        )
        traj.fps = self._fps

        cols = []
        if not skip_observations:
            cols.append("actions")
        if not skip_frames:
            cols.append(self.view_key)

        if cols:
            df = pd.read_parquet(parquet_path, columns=cols)

            if not skip_observations:
                actions = np.stack(df["actions"].to_numpy()).astype(np.float32)
                if actions.ndim == 2 and actions.shape[1] >= 1:
                    traj.gripper_state = (1.0 - actions[:, -1]) / 2.0
                    traj.gripper_velocity = actions[:, : min(6, actions.shape[1])]

            if not skip_frames:
                frames = [
                    np.array(Image.open(io.BytesIO(item["bytes"])).convert("RGB"))
                    for item in df[self.view_key]
                ]
                if not frames:
                    raise ValueError(f"No frames decoded for {path}")
                traj.frames = np.stack(frames)

        return traj


class MultiLeRobotDatasetLoader(BaseDatasetLoader):
    """Generic multi-root loader for folders containing many LeRobot datasets.

    Path format: ``"{subdataset}/{chunk_idx}/{episode_idx}"``.
    Per-subdataset camera and gripper behavior comes from a config file similar
    to the OXE wrapper, but without any OXE-specific assumptions.
    """

    DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "dataset" / "multi_lerobot_config" / "default.yaml"
    STARVLA_CACHE_VERSION = "sv2"

    def __init__(
        self,
        root_path: str,
        dataset_name: str = "multi_lerobot",
        config_path: Optional[str] = None,
        subdatasets: Optional[list[str] | dict[str, dict[str, Any]]] = None,
        view_key: Optional[str] = None,
        starvla_trajectory_names: bool = False,
    ):
        self._public_dataset_name = dataset_name
        self._view_key_override = view_key
        self._starvla_trajectory_names = starvla_trajectory_names
        self._config_path = Path(config_path) if config_path else self.DEFAULT_CONFIG_PATH
        self._config = (
            {}
            if isinstance(subdatasets, dict) and config_path is None
            else self._load_config(self._config_path)
        )
        self._subdataset_specs = self._resolve_subdataset_specs(Path(root_path), subdatasets)
        self._loaders: dict[str, ConfigurableLeRobotDatasetLoader] = {}
        self._subdataset_roots: dict[str, Path] = {}
        self.dataset_map: dict[str, list[str]] = {}
        self.robot_map = self.dataset_map

        for subdataset, spec in self._subdataset_specs.items():
            sub_root = Path(spec["root"]) if spec.get("root") else Path(root_path) / subdataset
            if not (sub_root / "meta" / "info.json").exists():
                logging.warning(f"MultiLeRobotDatasetLoader: missing LeRobot dataset root {sub_root}")
                continue
            child_name = f"{dataset_name}_{subdataset}"
            if starvla_trajectory_names:
                child_name += f"_{self.STARVLA_CACHE_VERSION}"
            self._loaders[subdataset] = ConfigurableLeRobotDatasetLoader(
                str(sub_root),
                dataset_name=child_name,
                view_key=self._view_key_override or spec.get("view_key"),
                camera_priority=spec.get("camera_priority", []),
                gripper=spec.get("gripper", {}),
                starvla_trajectory_names=starvla_trajectory_names,
            )
            self._subdataset_roots[subdataset] = sub_root

        cache_dataset_name = self._cache_dataset_name(dataset_name, subdatasets)
        if starvla_trajectory_names:
            cache_dataset_name += f"_{self.STARVLA_CACHE_VERSION}"
        super().__init__(root_path, dataset_name=cache_dataset_name)

        expected_subdatasets = set(self._loaders)
        cached_subdatasets = {str(traj).split("/", 1)[0] for traj in self.trajectories}
        if cached_subdatasets != expected_subdatasets:
            logging.info(
                "MultiLeRobotDatasetLoader: refreshing trajectory cache for subdatasets %s",
                sorted(expected_subdatasets),
            )
            self.cached["trajectories"] = self.get_trajectory_paths()
            self.trajectories = [Path(p) for p in self.cached["trajectories"]]
            self.save()

        self.dataset_map = self.cached.get("dataset_map", self.dataset_map)
        if not self.dataset_map:
            self._rebuild_dataset_map()
        self.robot_map = self.dataset_map

    @staticmethod
    def _cache_dataset_name(dataset_name: str, subdatasets: Optional[list[str]]) -> str:
        if not subdatasets:
            return dataset_name
        safe = "_".join(str(name).replace("/", "_") for name in subdatasets)
        return f"{dataset_name}__{safe}"

    @staticmethod
    def _load_config(path: Path) -> dict[str, Any]:
        if not path.exists():
            logging.warning(f"MultiLeRobotDatasetLoader: config not found at {path}; discovering all subdatasets")
            return {}
        from omegaconf import OmegaConf
        return OmegaConf.to_container(OmegaConf.load(path), resolve=True) or {}

    def _resolve_subdataset_specs(
        self,
        root: Path,
        requested: Optional[list[str] | dict[str, dict[str, Any]]],
    ) -> dict[str, dict[str, Any]]:
        configured = self._config.get("subdatasets", {})
        inline_specs = requested if isinstance(requested, dict) else None
        if inline_specs is not None:
            names = list(inline_specs)
        elif requested:
            names = requested
        elif configured:
            names = [name for name, spec in configured.items() if spec is None or spec.get("enabled", True)]
        else:
            names = [p.name for p in sorted(root.iterdir()) if (p / "meta" / "info.json").exists()]

        specs: dict[str, dict[str, Any]] = {}
        for name in names:
            spec = dict(configured.get(name, {}) or {})
            if inline_specs is not None:
                spec.update(dict(inline_specs.get(name, {}) or {}))
            if not spec.get("enabled", True):
                continue
            specs[name] = spec
        return specs

    def _rebuild_dataset_map(self) -> None:
        dataset_map: dict[str, list[str]] = {}
        for traj in self.trajectories:
            path_str = str(traj)
            subdataset = path_str.split("/", 1)[0]
            dataset_map.setdefault(subdataset, []).append(path_str)
        self.dataset_map = dataset_map

    def get_trajectory_paths(self) -> list[str]:
        paths = []
        self.dataset_map = {}
        for subdataset, loader in sorted(self._loaders.items()):
            prefixed = [f"{subdataset}/{traj_path}" for traj_path in loader.trajectories]
            self.dataset_map[subdataset] = prefixed
            paths.extend(prefixed)
        self.cached["dataset_map"] = self.dataset_map
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
            logging.warning(f"MultiLeRobotDatasetLoader: invalid path {path_str}, expected 'subdataset/chunk/episode'")
            return None

        subdataset, sub_path = parts
        loader = self._loaders.get(subdataset)
        if loader is None:
            logging.warning(f"MultiLeRobotDatasetLoader: unknown subdataset '{subdataset}'")
            return None

        traj = loader.load_trajectory(
            Path(sub_path),
            load_frames=load_frames,
            skip_frames=skip_frames,
            skip_observations=skip_observations,
            **kwargs,
        )
        if traj is not None:
            traj.name = f"{subdataset}/{traj.name}"
            traj.base_path = str(self._subdataset_roots.get(subdataset, self.root_path / subdataset))
            traj.dataset_name = self._public_dataset_name
        return traj

    def load_frames_at_indices(
        self,
        path: Path,
        frame_indices: list[int],
        view_key: Optional[str] = None,
    ) -> np.ndarray:
        """Decode selected frames through the matching child dataset loader."""
        path_str = str(path)
        parts = path_str.split("/", 1)
        if len(parts) != 2:
            raise ValueError(
                f"Invalid multi-LeRobot path {path_str!r}; expected "
                "'subdataset/chunk/episode'"
            )

        subdataset, sub_path = parts
        loader = self._loaders.get(subdataset)
        if loader is None:
            raise KeyError(f"Unknown multi-LeRobot subdataset {subdataset!r}")
        return loader.load_frames_at_indices(
            Path(sub_path),
            frame_indices,
            view_key=view_key,
        )


class AgiBotWorldDatasetLoader(LeRobotDatasetLoader):
    """Dataloader for the AgiBot World Beta dataset (LeRobot v3.0 format).

    Each episode is split into compound subtasks: consecutive actions on the same
    object are grouped into one Trajectory with interaction_phases.
    Path format: "{chunk_idx}/{ep_idx}/{pair_idx}"
    """

    AGIBOT_CAM_PRIORITY = [
        "observation.images.head",
        "observation.images.cam_high",
        "observation.images.cam_front",
        "observation.images.cam_left_wrist",
    ]

    def __init__(self, root_path: str, view_key: Optional[str] = None, dataset_name: str = "agibot_world"):
        tmp_root = Path(root_path)
        self._chunk_size = 1000
        self._cam_keys = []
        self._path_template = None
        self._task_map = {}
        self._episode_meta = {}
        self._subtask_meta: dict = {}
        self._fps: float = 30.0
        self.diag: Optional[dict] = None
        self._load_meta(tmp_root)

        # Try to read FPS from info.json
        info_path = tmp_root / "meta" / "info.json"
        if info_path.exists():
            try:
                with open(info_path) as f:
                    _info = json.load(f)
                self._fps = float(_info.get("fps", 30.0))
            except Exception:
                pass

        if view_key is None:
            view_key = next(
                (k for k in self.AGIBOT_CAM_PRIORITY if k in self._cam_keys),
                self._cam_keys[0] if self._cam_keys else "observation.images.head",
            )

        self._view_key_pref = view_key
        self.view_key = view_key
        self._episode_video_meta: dict[int, dict] = {}
        BaseDatasetLoader.__init__(self, root_path, dataset_name=dataset_name)

        # Load subtask_meta from cache if available (populated during get_trajectory_paths)
        self._subtask_meta = self.cached.get("subtask_meta", self._subtask_meta)
        # JSON round-trips int keys as str — restore to int
        self._episode_video_meta = {
            int(k): v for k, v in self.cached["episode_video_meta"].items()
        } if "episode_video_meta" in self.cached else self._episode_video_meta
        if self._hydrate_source_episode_metadata():
            self.cached["episode_video_meta"] = self._episode_video_meta
            self.save()

        self.task_map: dict[str, list[str]] = self._build_task_map()


    def _build_task_map(self) -> dict[str, list[str]]:
        """Group trajectory paths by their lang_ann (the annotation instruction)."""
        task_map: dict[str, list[str]] = {}
        for path_str, meta in self._subtask_meta.items():
            key = meta.get("lang_ann", "").strip().lower() or "_unknown"
            task_map.setdefault(key, []).append(path_str)
        return task_map

    def sample_max_per_task(self, max_per_task: int, seed: int = 42) -> list[str]:
        """Return at most max_per_task trajectories per unique lang_ann."""
        rng = random.Random(seed)
        result = []
        for paths in sorted(self.task_map.values()):
            result.extend(rng.sample(paths, min(max_per_task, len(paths))))
        return result

    def _video_path(self, chunk_idx: int, episode_idx: int) -> Path:
        ep_meta = self._episode_video_meta[episode_idx]
        vid_chunk = ep_meta["video_chunk_idx"]
        vid_file  = ep_meta["video_file_idx"]
        return self.root_path / "videos" / self.view_key / f"chunk-{vid_chunk:03d}" / f"file-{vid_file:03d}.mp4"

    def _data_parquet_path(self, chunk_idx: int, episode_idx: int) -> Path:
        meta = self._episode_video_meta.get(episode_idx, {})
        dc = meta.get("data_chunk_idx", chunk_idx)
        df = meta.get("data_file_idx", 0)
        return self.root_path / "data" / f"chunk-{dc:03d}" / f"file-{df:03d}.parquet"

    @staticmethod
    def _source_identity_from_episode_row(row) -> dict:
        """Extract stable upstream identifiers from either AgiBot release."""
        source_uuid = row.get("uuid")
        source_episode_id = row.get("agibot_episode_ident")
        return {
            "source_uuid": (
                str(source_uuid) if source_uuid is not None and pd.notna(source_uuid) else None
            ),
            "source_episode_id": (
                str(source_episode_id)
                if source_episode_id is not None and pd.notna(source_episode_id)
                else None
            ),
        }

    def _hydrate_source_episode_metadata(self) -> bool:
        """Add provenance to caches created before source IDs were retained."""
        missing = {
            episode_idx
            for episode_idx, meta in self._episode_video_meta.items()
            if not meta.get("source_uuid") and not meta.get("source_episode_id")
        }
        if not missing:
            return False

        changed = False
        episode_files = sorted(glob.glob(
            str(self.root_path / "meta" / "episodes" / "chunk-*" / "file-*.parquet")
        ))
        for episode_file in episode_files:
            try:
                schema_names = set(_pq.read_schema(episode_file).names)
                columns = [
                    column
                    for column in ("episode_index", "uuid", "agibot_episode_ident")
                    if column in schema_names
                ]
                if "episode_index" not in columns or len(columns) == 1:
                    continue
                episode_df = pd.read_parquet(episode_file, columns=columns)
                for _, row in episode_df.iterrows():
                    episode_idx = int(row["episode_index"])
                    if episode_idx not in missing:
                        continue
                    identity = self._source_identity_from_episode_row(row)
                    if identity["source_uuid"] or identity["source_episode_id"]:
                        self._episode_video_meta[episode_idx].update(identity)
                        missing.remove(episode_idx)
                        changed = True
                if not missing:
                    break
            except Exception as exc:
                logging.warning(
                    "AgiBotWorldDatasetLoader: could not hydrate source identity "
                    f"from {episode_file}: {exc}"
                )
        return changed

    # ------------------------------------------------------------------
    # Subtask grouping
    # ------------------------------------------------------------------

    def _group_subtasks(self, actions: list, ep_idx: int, chunk_idx: int, lang_ann_top: str) -> list:
        """Group consecutive actions on the same arm into one interaction."""
        if not actions:
            return []

        groups = []
        i = 0
        while i < len(actions):


            group = [actions[i]]
            arm_side_i = 0 if "left" in actions[i]["lang_ann_specific"].lower() else 1 if "right" in actions[i]["lang_ann_specific"].lower() else None
            skill_i = actions[i]["skill"]
            j = i + 1
            while j < len(actions):
                arm_side_j = 0 if "left" in actions[j]["lang_ann_specific"].lower() else 1 if "right" in actions[j]["lang_ann_specific"].lower() else None
                skill_j = actions[j]["skill"]

                #relax the arm check. if start and end frame are exactly the same, then we can assume it's the same interaction even if arm info is missing or inconsistent
                if actions[j]["start_frame"] == actions[j-1]["end_frame"]:
                    arm_side_j = arm_side_j if arm_side_j is not None else arm_side_i
                    arm_side_i = arm_side_i if arm_side_i is not None else arm_side_j

                should_break = (
                    # cannot continue across missing arm info
                    arm_side_i is None or arm_side_j is None or
                    # must stay on same arm
                    arm_side_i != arm_side_j or

                    # previous action already ended the interaction
                    skill_i in TERMINAL_SKILLS or
                    # current action starts a new interaction
                    skill_j in START_SKILLS
                )


                if should_break:
                    break

                group.append(actions[j])
                j += 1
            groups.append(group)
            i = j

        result = []
        for group_idx, group in enumerate(groups):
            start_frame = group[0]["start_frame"]
            end_frame = group[-1]["end_frame"]
            phases = [
                {
                    "phase_type": a["skill"] if a["skill"] else a["lang_ann_specific"],
                    "start_frame": a["start_frame"] - start_frame,
                    "end_frame": a["end_frame"] - start_frame,
                }
                for a in group
            ]
            lang_ann = " then ".join(a["lang_ann_specific"].lower().rstrip(".") for a in group)
            result.append({
                "chunk_idx": chunk_idx,
                "ep_idx": ep_idx,
                "pair_idx": group_idx,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "lang_ann": lang_ann,
                "lang_ann_top": lang_ann_top,
                "interaction_phases": phases,
                "longtask_id": f"agibot#{self.key}#{ep_idx}",
                "longtask_step": group_idx,
                "longtask_len": len(groups),
                "longtask_goal": lang_ann_top,
            })
        return result

    # ------------------------------------------------------------------
    # Path discovery
    # ------------------------------------------------------------------

    def get_trajectory_paths(self) -> list:
        self._subtask_meta = {}
        self._episode_video_meta: dict[int, dict] = {}  # ep_idx -> {video_chunk_idx, video_file_idx}
        paths = []

        # --- Diagnostic counters ---
        n_episodes_info = 0
        n_data_parquets_missing = 0
        n_parquets_no_action_col = 0
        n_parquets_read_error = 0
        episodes_discovered: set[int] = set()
        episodes_with_actions: set[int] = set()

        info_path = self.root_path / "meta" / "info.json"
        if info_path.exists():
            try:
                with open(info_path) as f:
                    n_episodes_info = json.load(f).get("total_episodes", 0)
            except Exception:
                pass

        episodes_glob = sorted(glob.glob(
            str(self.root_path / "meta" / "episodes" / "chunk-*" / "file-*.parquet")
        ))

        # --- Phase 1: Read all episode metadata ---
        data_parquet_episodes: dict[tuple[int, int], list[int]] = defaultdict(list)

        for ep_file in episodes_glob:
            try:
                ep_meta_df = pd.read_parquet(ep_file)
                vid_chunk_col = f"videos/{self.view_key}/chunk_index"
                vid_file_col  = f"videos/{self.view_key}/file_index"
                ts_col = f"videos/{self.view_key}/from_timestamp"
                for _, row in ep_meta_df.iterrows():
                    eidx = int(row["episode_index"])
                    dc = int(row["data/chunk_index"]) if "data/chunk_index" in row.index else 0
                    df_idx = int(row["data/file_index"]) if "data/file_index" in row.index else 0
                    self._episode_video_meta[eidx] = {
                        "video_chunk_idx": int(row[vid_chunk_col]) if vid_chunk_col in row.index else None,
                        "video_file_idx":  int(row[vid_file_col])  if vid_file_col  in row.index else None,
                        "data_chunk_idx": dc,
                        "data_file_idx": df_idx,
                        "from_timestamp": float(row[ts_col]) if ts_col in row.index else 0.0,
                        **self._source_identity_from_episode_row(row),
                    }
                    data_parquet_episodes[(dc, df_idx)].append(eidx)
            except Exception as e:
                logging.warning(f"AgiBotWorldDatasetLoader: could not read episodes parquet {ep_file}: {e}")

        # Columns needed for action-index scan — skip all robot-state / image-intrinsic columns
        _SCAN_COLS = ["episode_index", "action_index", "frame_index",
                      "language_instruction", "language_instruction_specific", "skill"]

        # --- Phase 2: Group episodes by data parquet, load each once ---
        for (dc, df_idx), ep_indices in sorted(data_parquet_episodes.items()):
            data_parquet = self.root_path / "data" / f"chunk-{dc:03d}" / f"file-{df_idx:03d}.parquet"
            if not data_parquet.exists():
                n_data_parquets_missing += 1
                continue

            try:
                # Read footer metadata only (microseconds) to know which columns exist,
                # then read just the 6 needed columns — avoids loading 100+ MB of
                # robot-state / camera-extrinsic data that we never use here.
                _schema_names = set(_pq.read_schema(data_parquet).names)
                _cols = [c for c in _SCAN_COLS if c in _schema_names]
                df = pd.read_parquet(data_parquet, columns=_cols)
            except Exception as e:
                logging.warning(f"AgiBotWorldDatasetLoader: could not read {data_parquet}: {e}")
                n_parquets_read_error += 1
                continue

            if "episode_index" in df.columns:
                expected = set(ep_indices)
                episodes_discovered.update(int(e) for e in df["episode_index"].unique() if int(e) in expected)

            # Top-level language instruction per episode
            lang_top_map: dict = {}
            if "language_instruction" in df.columns:
                lang_top_map = (
                    df.groupby("episode_index")["language_instruction"].first().to_dict()
                )

            # Only keep actual action frames
            if "action_index" not in df.columns:
                n_parquets_no_action_col += 1
                continue
            df_actions = df[df["action_index"] >= 0]

            for ep_idx, ep_df in df_actions.groupby("episode_index"):
                ep_idx_int = int(ep_idx)
                if ep_idx_int not in set(ep_indices):
                    continue
                episodes_with_actions.add(ep_idx_int)
                lang_ann_top = str(lang_top_map.get(ep_idx, "")).strip()

                actions = []
                for action_idx, action_df in ep_df.groupby("action_index"):
                    lang_specific = str(action_df["language_instruction_specific"].iloc[0]) \
                        if "language_instruction_specific" in action_df.columns else ""
                    skill = str(action_df["skill"].iloc[0]) \
                        if "skill" in action_df.columns else ""
                    actions.append({
                        "action_idx": int(action_idx),
                        "start_frame": int(action_df["frame_index"].min()),
                        "end_frame": int(action_df["frame_index"].max()) + 1,
                        "lang_ann_specific": lang_specific,
                        "skill": skill,
                    })

                actions.sort(key=lambda x: x["action_idx"])
                subtasks = self._group_subtasks(actions, ep_idx_int, dc, lang_ann_top)

                for meta in subtasks:
                    path_str = f"{dc}/{ep_idx}/{meta['pair_idx']}"
                    self._subtask_meta[path_str] = meta
                    paths.append(path_str)

        # --- Diagnostic summary ---
        self.diag = {
            "n_episodes_info": n_episodes_info,
            "n_episodes_discovered": len(episodes_discovered),
            "n_episodes_with_actions": len(episodes_with_actions),
            "n_trajectories": len(paths),
            "n_data_parquets_missing": n_data_parquets_missing,
            "n_parquets_no_action_col": n_parquets_no_action_col,
            "n_parquets_read_error": n_parquets_read_error,
            "n_episodes_no_actions": len(episodes_discovered) - len(episodes_with_actions),
        }
        print(f"[AgiBotWorldDatasetLoader {self.root_path.name}] Episode funnel:\n"
              f"  info.json total_episodes:          {self.diag['n_episodes_info']}\n"
              f"  Episodes in data parquets:          {self.diag['n_episodes_discovered']}\n"
              f"  Episodes with action frames:        {self.diag['n_episodes_with_actions']}\n"
              f"  Trajectories (after subtask decomp):{self.diag['n_trajectories']}\n"
              f"  --- Loss breakdown ---\n"
              f"  Data parquets missing:              {self.diag['n_data_parquets_missing']}\n"
              f"  Parquets without action_index col:  {self.diag['n_parquets_no_action_col']}\n"
              f"  Parquets with read errors:          {self.diag['n_parquets_read_error']}\n"
              f"  Episodes with no action frames:     {self.diag['n_episodes_no_actions']}")

        # Persist subtask_meta in the cache so it survives across runs
        self.cached["subtask_meta"] = self._subtask_meta
        self.cached["episode_video_meta"] = self._episode_video_meta
        self.task_map = self._build_task_map()

        return sorted(paths, key=lambda p: tuple(int(x) for x in p.split("/")))

    # ------------------------------------------------------------------
    # Trajectory loading
    # ------------------------------------------------------------------

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
            logging.warning(f"AgiBotWorldDatasetLoader: unknown path {path_str}")
            return None

        chunk_idx: int = meta["chunk_idx"]
        ep_idx: int = meta["ep_idx"]
        video_path = self._video_path(chunk_idx, ep_idx)
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
        traj.camera_view_key = self.view_key
        episode_meta = self._episode_video_meta.get(ep_idx, {})
        traj.source_uuid = episode_meta.get("source_uuid")
        traj.source_episode_id = episode_meta.get("source_episode_id")
        traj.source_episode_index = ep_idx
        traj.source_subtrajectory_index = meta["pair_idx"]
        traj.source_episode_frame_start = meta["start_frame"]
        traj.source_episode_frame_end_exclusive = meta["end_frame"]
        traj.source_dataset_format = "lerobot_v3"
        traj.interaction_phases = meta["interaction_phases"]
        traj.longtask_id = meta["longtask_id"]
        traj.longtask_step = meta["longtask_step"]
        traj.longtask_len = meta["longtask_len"]
        traj.longtask_goal = meta["longtask_goal"]

        if not skip_observations:
            data_path = self._data_parquet_path(chunk_idx, ep_idx)
            if data_path.exists():
                try:
                    _OBS_COLS = [
                        "episode_index",
                        "frame_index",
                        "actions.effector.position",
                        "actions.joint.position",
                        "observation.states.end.position",
                        "observation.images.head_extrinsics",
                    ]
                    df = pd.read_parquet(data_path, columns=_OBS_COLS)
                    ep_df = df[df["episode_index"] == ep_idx]
                    sub_df = ep_df[
                        (ep_df["frame_index"] >= meta["start_frame"]) &
                        (ep_df["frame_index"] < meta["end_frame"])
                    ].sort_values("frame_index")

                    if len(sub_df) > 0:
                        if "actions.effector.position" in sub_df.columns:
                            eff = np.array(sub_df["actions.effector.position"].tolist())
                            if eff.ndim == 2 and eff.shape[1] >= 2:
                                # Dual-arm: column 0 = left, column 1 = right
                                traj.left_gripper_state = 1- eff[:, 0]
                                traj.right_gripper_state = 1 - eff[:, 1]
                                traj.gripper_state = 1 - eff[:, 0]  # backward compat primary
                            else:
                                traj.gripper_state = 1 - eff.ravel()
                        if "observation.states.end.position" in sub_df.columns:
                            left_eef, right_eef = self._extract_dual_arm_eef_positions(
                                sub_df["observation.states.end.position"]
                            )
                            traj.left_eef_position = left_eef
                            traj.right_eef_position = right_eef
                        if "observation.images.head_extrinsics" in sub_df.columns:
                            traj.camera_extrinsics_cam_to_robot = self._extract_pose7_column(
                                sub_df["observation.images.head_extrinsics"]
                            )
                            traj.camera_extrinsics_view_key = self.view_key
                        if "actions.joint.position" in sub_df.columns:
                            joint_pos = np.array(sub_df["actions.joint.position"].tolist())
                            if joint_pos.ndim == 2 and joint_pos.shape[1] >= 6:
                                traj.gripper_velocity = joint_pos[:, :6]
                except Exception as e:
                    logging.warning(f"AgiBotWorldDatasetLoader: could not load observations for {path}: {e}")

        if not skip_frames:
            if not video_path.exists():
                raise ValueError(f"Video not found: {video_path}")

            from_ts = self._episode_video_meta.get(ep_idx, {}).get("from_timestamp", 0.0)

            # Load frame indices and episode-relative timestamps from data parquet.
            # Shift timestamps by from_ts (exactly as LeRobot does) then convert to
            # frame indices for decord — avoids integer drift from round(from_ts * fps).
            data_path = self._data_parquet_path(chunk_idx, ep_idx)
            if not data_path.exists():
                raise ValueError(f"Data parquet not found: {data_path}")
            sub_df = pd.read_parquet(data_path, columns=["episode_index", "frame_index", "timestamp"])
            sub_df = sub_df[sub_df["episode_index"] == ep_idx]
            sub_df = sub_df[
                (sub_df["frame_index"] >= meta["start_frame"]) &
                (sub_df["frame_index"] < meta["end_frame"])
            ].sort_values("frame_index")

            if sub_df.empty:
                raise ValueError(f"No frames found in parquet for {path}")

            traj.source_episode_frame_indices = sub_df["frame_index"].to_numpy(
                dtype=np.int64,
            )
            traj.frame_timestamps_seconds = sub_df["timestamp"].to_numpy(
                dtype=np.float64,
            )

            shifted_timestamps = (sub_df["timestamp"].values + from_ts).tolist()

            vr = VideoReader(str(video_path), ctx=decord_cpu(0))
            fps = vr.get_avg_fps()
            frame_indices = [min(max(round(ts * fps), 0), len(vr) - 1) for ts in shifted_timestamps]
            traj.frames = vr.get_batch(frame_indices).asnumpy()

        return traj


class AgiBotWorldMultiTaskLoader(BaseDatasetLoader):
    """Wraps multiple AgiBotWorldDatasetLoader instances (one per task_* folder).

    Trajectory names are prefixed with the task folder name to avoid duplicates
    across sub-loaders: ``"{task_folder}/{chunk}/{ep}/{pair}"``

    Usage:
        loader = AgiBotWorldMultiTaskLoader("/data/AgiBotWorld-Beta")
        traj = loader.load_trajectory(Path("task_0042/0/509/0"))
    """

    def __init__(self, root_path: str, view_key: Optional[str] = None, task_prefix: str = "task_", dataset_name: str = "agibot_world", n_workers: int = 16):
        self._view_key = view_key
        self._task_prefix = task_prefix
        self._loaders: dict[str, AgiBotWorldDatasetLoader] = {}

        tmp_root = Path(root_path)
        task_dirs = sorted(
            d for d in tmp_root.iterdir()
            if d.is_dir() and d.name.startswith(task_prefix)
        )

        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _load_task(task_dir):
            loader = AgiBotWorldDatasetLoader(
                str(task_dir),
                view_key=view_key,
                dataset_name=f"agibot_world_{task_dir.name}",
            )
            return task_dir.name, loader
        if n_workers > 1:
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = {pool.submit(_load_task, td): td for td in task_dirs}
                for future in tqdm(as_completed(futures), total=len(futures), desc="Loading AgiBotWorld tasks"):
                    task_dir = futures[future]
                    try:
                        task_name, loader = future.result()
                        self._loaders[task_name] = loader
                    except Exception as e:
                        logging.warning(f"AgiBotWorldMultiTaskLoader: could not load {task_dir}: {e}")
        else:
            for task_dir in tqdm(task_dirs, desc="Loading AgiBotWorld tasks"):
                try:
                    task_name, loader = _load_task(task_dir)
                    self._loaders[task_name] = loader
                except Exception as e:
                    logging.warning(f"AgiBotWorldMultiTaskLoader: could not load {task_dir}: {e}")

        print(f"AgiBotWorldMultiTaskLoader: loaded {len(self._loaders)} task folders, "
              f"{sum(len(l.trajectories) for l in self._loaders.values())} trajectories total")

        # --- Aggregate diagnostics across sub-loaders ---
        agg = {
            "n_episodes_info": 0,
            "n_episodes_discovered": 0,
            "n_episodes_with_actions": 0,
            "n_trajectories": 0,
            "n_data_parquets_missing": 0,
            "n_parquets_no_action_col": 0,
            "n_parquets_read_error": 0,
            "n_episodes_no_actions": 0,
        }
        n_cached = 0
        for task_name, loader in sorted(self._loaders.items()):
            diag = loader.diag
            if diag is None:
                n_cached += 1
                continue
            for k in agg:
                agg[k] += diag[k]
            if diag["n_episodes_info"] != diag["n_episodes_with_actions"]:
                print(f"  [{task_name}] info={diag['n_episodes_info']} discovered={diag['n_episodes_discovered']} "
                      f"with_actions={diag['n_episodes_with_actions']} trajs={diag['n_trajectories']} "
                      f"missing_parquets={diag['n_data_parquets_missing']} no_action_col={diag['n_parquets_no_action_col']} "
                      f"no_action_frames={diag['n_episodes_no_actions']}")

        if n_cached:
            print(f"  ({n_cached} loaders used cache — no diagnostics available, delete .dataset_cache/ to regenerate)")
        print(f"\n[AgiBotWorldMultiTaskLoader] TOTAL Episode funnel:\n"
              f"  info.json total_episodes:          {agg['n_episodes_info']}\n"
              f"  Episodes in data parquets:          {agg['n_episodes_discovered']}\n"
              f"  Episodes with action frames:        {agg['n_episodes_with_actions']}\n"
              f"  Trajectories (after subtask decomp):{agg['n_trajectories']}\n"
              f"  --- Loss breakdown ---\n"
              f"  Data parquets missing:              {agg['n_data_parquets_missing']}\n"
              f"  Parquets without action_index col:  {agg['n_parquets_no_action_col']}\n"
              f"  Parquets with read errors:          {agg['n_parquets_read_error']}\n"
              f"  Episodes with no action frames:     {agg['n_episodes_no_actions']}")
        self.diag = agg

        BaseDatasetLoader.__init__(self, root_path, dataset_name=dataset_name)

        self.task_map: dict[str, list[str]] = self._build_task_map()

    def _build_task_map(self) -> dict[str, list[str]]:
        """Group all trajectory paths by lang_ann across sub-loaders."""
        task_map: dict[str, list[str]] = {}
        for task_name, loader in self._loaders.items():
            for path_str, meta in loader._subtask_meta.items():
                key = meta.get("lang_ann", "").strip().lower() or "_unknown"
                task_map.setdefault(key, []).append(f"{task_name}/{path_str}")
        return task_map

    def sample_max_per_task(self, max_per_task: int, seed: int = 42) -> list[str]:
        """Return at most max_per_task trajectories per unique lang_ann."""
        rng = random.Random(seed)
        result = []
        for paths in sorted(self.task_map.values()):
            result.extend(rng.sample(paths, min(max_per_task, len(paths))))
        return result

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
            logging.warning(f"AgiBotWorldMultiTaskLoader: invalid path format {path_str}, expected 'task_xxx/chunk/ep/pair'")
            return None

        task_name, sub_path = parts
        loader = self._loaders.get(task_name)
        if loader is None:
            logging.warning(f"AgiBotWorldMultiTaskLoader: unknown task folder '{task_name}'")
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


# --- EgoLive (egocentric human-hand demonstrations, LeRobot v3) -------------

# Column layout of the per-frame `hand_object_info` feature: each slot is
# [x1, y1, x2, y2, obj_id, track_id] and -1 marks "not annotated in this frame".
EGOLIVE_BOX_SLOTS = {
    "left_hand": 0,
    "right_hand": 6,
    "left_obj": 12,
    "right_obj": 18,
    "both_obj": 24,
}


class EgoLiveDatasetLoader(LeRobotDatasetLoader):
    """One EgoLive leaf dataset (LeRobot v3, ``robot_type: human_hands``).

    Trajectories are *instruction segments*, not whole episodes: EgoLive ships
    official subtask boundaries in ``meta/episodes/**.instruction_segments``, so
    a trajectory key is ``"{chunk}/{episode}/{segment}"`` and the segment
    instruction becomes ``lang_ann``.

    Source video is 2160x2160 at 60 fps, which is far more than the annotation
    pipeline needs, so frames are decimated to ``target_fps`` and downscaled to
    ``max_image_size``. Every per-frame array (GT hand/object boxes, 2D
    keypoints, hand state) is decimated with the same indices and the 2D
    quantities are rescaled, so all of them stay aligned with ``traj.frames``.
    """

    EGO_CAM = "observation.images.cam_head_ego_left"

    _OBS_COLUMNS = (
        "episode_index",
        "frame_index",
        "timestamp",
        "observation.state",
        "hand_object_info",
        "hand_occlusion_pred",
        "left_kp3d",
        "right_kp3d",
        "leftcam_left_kp2d",
        "leftcam_right_kp2d",
        "camera_pos",
    )

    def __init__(
        self,
        root_path: str,
        dataset_name: Optional[str] = None,
        view_key: Optional[str] = None,
        target_fps: float = 10.0,
        max_image_size: int = 512,
        min_segment_seconds: float = 0.0,
        max_segment_seconds: Optional[float] = None,
        grasp_signal: str = "contact",
        primary_hand: str = "right",
        phase_gap_seconds: float = 0.25,
        phase_min_seconds: float = 0.3,
        segment_pad_seconds: float = 0.5,
    ):
        self.target_fps = float(target_fps)
        self.max_image_size = int(max_image_size)
        self.min_segment_seconds = float(min_segment_seconds)
        self.max_segment_seconds = max_segment_seconds
        self.grasp_signal = grasp_signal
        self.primary_hand = primary_hand
        self.phase_gap_seconds = float(phase_gap_seconds)
        self.phase_min_seconds = float(phase_min_seconds)
        # Instruction segments are cut on what the person is told to do, not on
        # when they touch things, so a segment often starts and ends mid-grasp.
        # Widening the loaded window past the segment recovers the open frames
        # on either side of the contact, which is what gives the gripper signal
        # its open->closed->open shape and the detector a pre-grasp view.
        self.segment_pad_seconds = float(segment_pad_seconds)
        self._segment_meta: dict[str, dict] = {}
        self._image_hw: tuple[int, int] = (2160, 2160)

        super().__init__(root_path, dataset_name=dataset_name or "egolive",
                         view_key=view_key or self.EGO_CAM)

    def _load_meta(self, root: Path) -> None:
        super()._load_meta(root)
        info_path = root / "meta" / "info.json"
        if info_path.exists():
            try:
                with open(info_path, "r", encoding="utf-8") as f:
                    shape = json.load(f)["features"][self.EGO_CAM]["shape"]
                self._image_hw = (int(shape[0]), int(shape[1]))
            except Exception as e:
                logging.warning(f"EgoLiveDatasetLoader: could not read ego camera shape: {e}")

    @property
    def stride(self) -> int:
        """Number of source frames per kept frame."""
        if self.target_fps <= 0:
            return 1
        return max(1, int(round(self._fps / self.target_fps)))

    @property
    def scale(self) -> float:
        """Factor mapping source pixels onto the returned frame resolution."""
        longest = max(self._image_hw)
        if self.max_image_size <= 0 or longest <= self.max_image_size:
            return 1.0
        return self.max_image_size / longest

    @property
    def output_hw(self) -> tuple[int, int]:
        return (int(round(self._image_hw[0] * self.scale)),
                int(round(self._image_hw[1] * self.scale)))

    def get_trajectory_paths(self) -> list[str]:
        super().get_trajectory_paths()  # populates self._episode_meta / video+data locations
        self._segment_meta = {}
        paths: list[str] = []

        for ep_file in sorted(glob.glob(
            str(self.root_path / "meta" / "episodes" / "chunk-*" / "file*.parquet")
        )):
            chunk_idx = int(Path(ep_file).parent.name.replace("chunk-", ""))
            try:
                columns = ["episode_index", "length", "task", "uuid", "instruction_segments"]
                available = set(_pq.read_schema(ep_file).names)
                if "camera_intrinsics" in available:
                    columns.append("camera_intrinsics")
                df = pd.read_parquet(ep_file, columns=columns)
            except Exception as e:
                logging.warning(f"EgoLiveDatasetLoader: skipping {ep_file}: {e}")
                continue

            for _, row in df.iterrows():
                ep_idx = int(row["episode_index"])
                segments = row["instruction_segments"]
                if segments is None or len(segments) == 0:
                    continue
                for segment in segments:
                    instruction = str(segment.get("instruction") or "").strip()
                    duration = float(segment.get("duration_seconds", 0.0))
                    if not instruction or duration < self.min_segment_seconds:
                        continue
                    if self.max_segment_seconds is not None and duration > self.max_segment_seconds:
                        continue
                    seg_idx = int(segment["subtask_index"])
                    key = f"{chunk_idx}/{ep_idx}/{seg_idx}"
                    self._segment_meta[key] = {
                        "camera_intrinsics": self._scaled_intrinsics(row),
                        "chunk_idx": chunk_idx,
                        "ep_idx": ep_idx,
                        "seg_idx": seg_idx,
                        "lang_ann": instruction,
                        "start_frame": int(segment["start_frame"]),
                        "end_frame": int(segment["end_frame"]) + 1,  # EgoLive end_frame is inclusive
                        "duration_seconds": duration,
                        "episode_task": str(row["task"]),
                        "episode_uuid": str(row["uuid"]),
                        "n_segments": len(segments),
                    }
                    paths.append(key)
        return paths

    def _scaled_intrinsics(self, row) -> Optional[tuple]:
        """(fx, fy, cx, cy) for the ego camera, scaled to the returned frames.

        Read per episode rather than assumed: focal lengths vary 500-517 px
        across episodes at the source 2160x2160, so a single constant is wrong by
        a few percent and wrong by a factor of two if the output size changes.
        """
        raw = row.get("camera_intrinsics") if hasattr(row, "get") else None
        if raw is None:
            return None
        try:
            entry = raw[self.EGO_CAM.rsplit(".", 1)[-1]]
            focal, principal = entry["focal_length"], entry["principal_point"]
            scale = self.scale
            return (float(focal[0]) * scale, float(focal[1]) * scale,
                    float(principal[0]) * scale, float(principal[1]) * scale)
        except Exception as e:
            logging.warning(f"EgoLiveDatasetLoader: could not read camera intrinsics: {e}")
            return None

    def _ensure_segment_meta(self) -> None:
        if not self._segment_meta:
            self.get_trajectory_paths()

    @property
    def pad_frames(self) -> int:
        return max(0, int(round(self.segment_pad_seconds * self._fps)))

    def _padded_bounds(self, meta: dict) -> tuple[int, int]:
        """Segment window widened by the pad, clamped at the episode start."""
        return max(0, meta["start_frame"] - self.pad_frames), meta["end_frame"] + self.pad_frames

    @staticmethod
    def _split_hand_object_info(raw: np.ndarray) -> tuple[dict, dict]:
        """Split the (T, 30) blob into per-slot boxes and track ids.

        Absent annotations are -1 in the raw feature; boxes become NaN so that
        downstream math never treats them as a real corner at (-1, -1).
        """
        boxes, track_ids = {}, {}
        for slot, offset in EGOLIVE_BOX_SLOTS.items():
            box = raw[:, offset: offset + 4].astype(np.float32)
            track = raw[:, offset + 5].astype(np.int32)
            box[(box < 0).all(axis=1)] = np.nan
            boxes[slot] = box
            track_ids[slot] = track
        return boxes, track_ids

    def _decode_frames(self, video_path: Path, timestamps: np.ndarray) -> np.ndarray:
        """Decode and downscale the requested timestamps in small batches.

        Batching matters here: a 5 s segment at full 2160x2160 would be ~700 MB
        if decord materialized every frame before the resize.
        """
        vr = VideoReader(str(video_path), ctx=decord_cpu(0))
        video_fps = vr.get_avg_fps()
        indices = [min(max(int(round(ts * video_fps)), 0), len(vr) - 1) for ts in timestamps]
        out_h, out_w = self.output_hw

        frames = np.empty((len(indices), out_h, out_w, 3), dtype=np.uint8)
        for start in range(0, len(indices), 8):
            batch = vr.get_batch(indices[start:start + 8]).asnumpy()
            for offset, frame in enumerate(batch):
                frames[start + offset] = (
                    frame if frame.shape[:2] == (out_h, out_w)
                    else cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
                )
        return frames

    @staticmethod
    def _blank_untracked(values: Optional[np.ndarray], tracked: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """NaN out rows where the hand was not tracked (EgoLive zeroes them instead)."""
        if values is None or tracked is None:
            return values
        values = values.astype(np.float32, copy=True)
        values[~tracked] = np.nan
        return values

    def _tracked_masks(self, sub_df: pd.DataFrame) -> dict[str, Optional[np.ndarray]]:
        if "observation.state" not in sub_df.columns:
            return {"left": None, "right": None}
        state = self._stack_array_column(sub_df["observation.state"])
        if state is None:
            return {"left": None, "right": None}
        state = state.reshape(len(sub_df), -1)
        return {hand: egolive_grasp.tracked_mask(state, hand) for hand in ("left", "right")}

    def _populate_hand_annotations(self, traj: Trajectory, sub_df: pd.DataFrame) -> None:
        scale = self.scale
        tracked = self._tracked_masks(sub_df)

        if "hand_object_info" in sub_df.columns:
            raw = self._stack_array_column(sub_df["hand_object_info"])
            if raw is not None:
                boxes, track_ids = self._split_hand_object_info(raw.reshape(len(sub_df), -1))
                traj.hand_object_boxes = {slot: box * scale for slot, box in boxes.items()}
                traj.hand_object_track_ids = track_ids

        if "hand_occlusion_pred" in sub_df.columns:
            occlusion = self._stack_array_column(sub_df["hand_occlusion_pred"])
            if occlusion is not None:
                traj.hand_occlusion = occlusion.reshape(len(sub_df), -1)[:, :4].astype(np.float32)

        for column, attribute, hand in (
            ("left_kp3d", "left_hand_kp3d", "left"),
            ("right_kp3d", "right_hand_kp3d", "right"),
        ):
            if column in sub_df.columns:
                kp = self._stack_array_column(sub_df[column])
                if kp is not None:
                    kp = kp.reshape(len(sub_df), -1, 3).astype(np.float32)
                    setattr(traj, attribute, self._blank_untracked(kp, tracked[hand]))

        for column, attribute, hand in (
            ("leftcam_left_kp2d", "left_hand_kp2d", "left"),
            ("leftcam_right_kp2d", "right_hand_kp2d", "right"),
        ):
            if column in sub_df.columns:
                kp = self._stack_array_column(sub_df[column])
                if kp is not None:
                    kp = self._blank_untracked(
                        kp.reshape(len(sub_df), -1, 2).astype(np.float32) * scale, tracked[hand]
                    )
                    setattr(traj, attribute, kp)
                    setattr(traj, f"{hand}_grasp_center_2d", egolive_grasp.grasp_center(kp))

        # camera_pos is the ego camera's pose relative to the episode's first
        # frame — GT camera motion, which a head-mounted camera has plenty of.
        if "camera_pos" in sub_df.columns:
            traj.camera_extrinsics_cam_to_robot = self._extract_pose7_column(sub_df["camera_pos"])
            traj.camera_extrinsics_view_key = self.view_key

    def _populate_phases(self, traj: Trajectory, segment_info: np.ndarray, n_output: int) -> dict:
        """Attach interaction phases and return the matching per-hand gripper states.

        Phases are derived at the full 60 fps and only then decimated: raw
        contact runs are a handful of frames long, so the gap closing has to see
        every source frame to reconstruct anything physical.
        """
        phases = egolive_phases.interaction_phases(
            segment_info,
            fps=self._fps,
            gap_seconds=self.phase_gap_seconds,
            min_seconds=self.phase_min_seconds,
        )
        egolive_phases.annotate_phase_quality(phases, segment_info, self._fps)
        egolive_phases.mark_bimanual_pairs(phases)
        traj.interaction_phases = egolive_phases.rescale_phases(
            phases, self.stride, self.scale, n_output_frames=n_output
        )
        return {
            hand: egolive_phases.gripper_state_from_phases(
                traj.interaction_phases, hand, n_output
            )
            for hand in ("left", "right")
        }

    def _populate_observations(self, traj: Trajectory, sub_df: pd.DataFrame,
                               contact_states: Optional[dict] = None) -> None:
        if "observation.state" not in sub_df.columns:
            return
        state = self._stack_array_column(sub_df["observation.state"])
        if state is None:
            return
        state = state.reshape(len(sub_df), -1).astype(np.float32)
        traj.observation_state = state

        if self.grasp_signal == "contact":
            # Prefer EgoLive's hand-object contact annotations when available.
            if contact_states is None:
                raise ValueError("grasp_signal='contact' needs the full-rate hand_object_info")
            traj.left_gripper_state = contact_states["left"]
            traj.right_gripper_state = contact_states["right"]
        else:
            traj.left_gripper_state = egolive_grasp.gripper_state(state, "left", self.grasp_signal)
            traj.right_gripper_state = egolive_grasp.gripper_state(state, "right", self.grasp_signal)
        traj.gripper_state = (
            traj.right_gripper_state if self.primary_hand == "right" else traj.left_gripper_state
        )
        # Wrist xyz in the ego camera frame (verified: reprojects onto the 2D
        # keypoints with a pinhole model to ~10 px at 2160x2160).
        for hand in ("left", "right"):
            setattr(traj, f"{hand}_eef_position", self._blank_untracked(
                egolive_grasp.wrist_positions(state, hand),
                egolive_grasp.tracked_mask(state, hand),
            ))

        primary_eef = (
            traj.right_eef_position if self.primary_hand == "right" else traj.left_eef_position
        )
        velocity = np.diff(primary_eef, axis=0, prepend=primary_eef[0:1])
        traj.gripper_velocity = np.concatenate(
            [velocity, np.zeros_like(velocity)], axis=1
        ).astype(np.float32)

    def load_trajectory(
        self,
        path: Path,
        load_frames: bool = False,
        skip_frames: bool = True,
        skip_observations: bool = True,
        **kwargs,
    ) -> Optional[Trajectory]:
        path_str = str(path)
        self._ensure_segment_meta()

        meta = self._segment_meta.get(path_str)
        if meta is None:
            logging.warning(f"EgoLiveDatasetLoader: unknown segment {path_str}")
            return None

        chunk_idx, ep_idx = meta["chunk_idx"], meta["ep_idx"]
        video_path = self._video_path(chunk_idx, ep_idx)

        traj = Trajectory(
            name=path_str,
            lang_ann=meta["lang_ann"],
            base_path=str(self.root_path),
            dataset_name=self.key,
            media_dir=str(video_path),
            split=self.get_split_type(path),
        )
        traj.fps = self._fps / self.stride
        traj.camera_view_key = self.view_key
        traj.image_size = self.output_hw
        # Frame bounds describe the clip that is actually returned, i.e. the
        # padded window; the instruction still belongs to the segment inside it.
        traj.start_frame, traj.end_frame = self._padded_bounds(meta)
        traj.segment_start_frame = meta["start_frame"]
        traj.segment_end_frame = meta["end_frame"]
        traj.longtask_id = f"egolive#{self.key}#{ep_idx}"
        traj.longtask_step = meta["seg_idx"]
        traj.longtask_len = meta["n_segments"]
        traj.longtask_goal = meta["episode_task"]
        traj.camera_intrinsics = meta.get("camera_intrinsics")

        data_path = self._data_parquet_path(chunk_idx, ep_idx)
        if not data_path.exists():
            raise ValueError(f"Data parquet not found: {data_path}")

        available = set(_pq.read_schema(data_path).names)
        columns = [c for c in self._OBS_COLUMNS if c in available]
        df = pd.read_parquet(data_path, columns=columns)
        window_start, window_end = self._padded_bounds(meta)
        segment_df = (
            df[(df["episode_index"] == ep_idx)
               & (df["frame_index"] >= window_start)
               & (df["frame_index"] < window_end)]
            .sort_values("frame_index")
            .reset_index(drop=True)
        )
        if segment_df.empty:
            raise ValueError(f"No frames found in parquet for {path_str}")

        sub_df = segment_df.iloc[::self.stride].reset_index(drop=True)
        traj.frame_indices = sub_df["frame_index"].to_numpy()

        contact_states = None
        if "hand_object_info" in segment_df.columns:
            segment_info = self._stack_array_column(segment_df["hand_object_info"])
            if segment_info is not None:
                segment_info = segment_info.reshape(len(segment_df), -1)
                contact_states = self._populate_phases(traj, segment_info, len(sub_df))

        self._populate_hand_annotations(traj, sub_df)
        if not skip_observations:
            self._populate_observations(traj, sub_df, contact_states)

        if not skip_frames:
            if not video_path.exists():
                raise ValueError(f"Video not found: {video_path}")
            from_ts = self._episode_meta.get(ep_idx, {}).get("from_timestamp", 0.0)
            traj.frames = self._decode_frames(
                video_path, sub_df["timestamp"].to_numpy() + from_ts
            )

        return traj


class EgoLiveMultiDatasetLoader(BaseDatasetLoader):
    """Wraps the EgoLive taxonomy tree, whose leaves are LeRobot datasets.

    The tree is ``domain/task-group/task/description``; every leaf holding a
    ``meta/info.json`` becomes a child :class:`EgoLiveDatasetLoader`.
    Trajectory keys are ``"{leaf/relative/path}/{chunk}/{episode}/{segment}"``.
    """

    def __init__(
        self,
        root_path: str,
        dataset_name: str = "egolive",
        subdatasets: Optional[list[str]] = None,
        view_key: Optional[str] = None,
        n_workers: int = 16,
        **loader_kwargs,
    ):
        self._public_dataset_name = dataset_name
        self._loader_kwargs = dict(loader_kwargs)
        self._loaders: dict[str, EgoLiveDatasetLoader] = {}

        root = Path(root_path)
        names = list(subdatasets) if subdatasets else self.discover_subdatasets(root)

        def _build(name: str):
            return name, EgoLiveDatasetLoader(
                str(root / name),
                dataset_name=f"{dataset_name}_{name.replace('/', '_')}",
                view_key=view_key,
                **self._loader_kwargs,
            )

        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=max(1, n_workers)) as pool:
            futures = {pool.submit(_build, name): name for name in names}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Indexing EgoLive leaves"):
                try:
                    name, loader = future.result()
                    self._loaders[name] = loader
                except Exception as e:
                    logging.warning(f"EgoLiveMultiDatasetLoader: could not load {futures[future]}: {e}")

        print(f"EgoLiveMultiDatasetLoader: {len(self._loaders)} leaf datasets, "
              f"{sum(len(l.trajectories) for l in self._loaders.values())} segments")

        super().__init__(root_path, dataset_name=self._cache_name(dataset_name, subdatasets))

    @staticmethod
    def discover_subdatasets(root: Path) -> list[str]:
        """Relative paths of every LeRobot leaf under the EgoLive tree."""
        return sorted(
            str(Path(info).parent.parent.relative_to(root))
            for info in glob.glob(str(root / "**" / "meta" / "info.json"), recursive=True)
        )

    @staticmethod
    def _cache_name(dataset_name: str, subdatasets: Optional[list[str]]) -> str:
        if not subdatasets:
            return dataset_name
        digest = hashlib.md5("|".join(sorted(subdatasets)).encode()).hexdigest()[:10]
        return f"{dataset_name}__{len(subdatasets)}_{digest}"

    def get_trajectory_paths(self) -> list[str]:
        return sorted(
            f"{name}/{traj_path}"
            for name, loader in self._loaders.items()
            for traj_path in loader.trajectories
        )

    def _split_path(self, path: str) -> tuple[EgoLiveDatasetLoader, str]:
        # Leaf names contain slashes, so split off the trailing chunk/ep/segment.
        leaf, _, sub_path = str(path).rpartition("/")
        leaf, _, ep = leaf.rpartition("/")
        leaf, _, chunk = leaf.rpartition("/")
        loader = self._loaders.get(leaf)
        if loader is None:
            raise KeyError(f"Unknown EgoLive leaf dataset {leaf!r}")
        return loader, f"{chunk}/{ep}/{sub_path}"

    def load_trajectory(
        self,
        path: Path,
        load_frames: bool = False,
        skip_frames: bool = True,
        skip_observations: bool = True,
        **kwargs,
    ) -> Optional[Trajectory]:
        try:
            loader, sub_path = self._split_path(str(path))
        except KeyError as e:
            logging.warning(f"EgoLiveMultiDatasetLoader: {e}")
            return None

        traj = loader.load_trajectory(
            Path(sub_path),
            load_frames=load_frames,
            skip_frames=skip_frames,
            skip_observations=skip_observations,
            **kwargs,
        )
        if traj is not None:
            traj.name = str(path)
            traj.base_path = str(loader.root_path)
            traj.dataset_name = self._public_dataset_name
        return traj
