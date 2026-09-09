"""Trajectory orchestration, detection, segmentation, and task parsing."""

import logging
import os
import pickle
import time
from pathlib import Path
import numpy as np
import torch
import supervision as sv
from torchvision.ops import nms
from sparc.perception.bbox_ops import frames_to_tensor, box_from_mask
from sparc.pipeline.debug_images import DebugImageSaver
from sparc.pipeline.pipeline_types import (
    DetectionResult,
    TargetResult,
    CropInfo,
    FinalBoxes,
)
from sparc.pipeline.publication_schema import build_publication_annotation
from sparc.perception.ann_utils import (
    DEFAULT_MAX_BOX_AREA_RATIO,
    STANDARD_DETECTION_TEXT_THRESHOLD,
    TOOL_OBJECT_DETECTION_THRESHOLD,
    get_interacted_obj_boxes,
    filter_boxes_by_size,
)
from sparc.robot.keystate_utils import (
    get_gripper_close_phases,
    apply_phase_offset,
    get_phase_sequence_key,
)
from sparc.llm.llm_utils import extract_task_obj_with_phases
from sparc.llm.llm_bimanual import extract_task_obj_bimanual
from sparc.llm.semantic_task_parsing import (
    TASK_OBJECT_PARSING_MODES,
    resolve_task_object_parsing_mode,
    semantic_episode_window,
    task_object_cache_key,
)
from sparc.pipeline.annotation_scoring import _augment_tracks_with_robotseg
from sparc.pipeline.annotation_scoring import _augment_tracks_with_targets
from sparc.pipeline.annotation_scoring import (
    _augment_tracks_with_tracking_frame_indices,
)
from sparc.pipeline.annotation_scoring import _resolve_emitted_initial_box_confidence

from sparc.pipeline.annotation_coordinates import CoordinateTransforms
from sparc.pipeline.annotation_scoring import AnnotationScoring
from sparc.pipeline.annotation_export import AnnotationExport


OBJ_DET_IDX = 1  # index in task_grasp_phase_frame_indices for object detection keyframe


def _find_bimanual_pairs(task_obj_info_list) -> list:
    """Find (left_idx, right_idx) index pairs whose time windows overlap."""
    def _window(info):
        frames = (
            [gp["start_frame"] for gp in info["grasp_phases"]] +
            [gp["end_frame"]   for gp in info["grasp_phases"]]
        )
        return (min(frames), max(frames)) if frames else (0, 0)

    lefts  = [(i, info) for i, info in enumerate(task_obj_info_list) if info.get("arm") == "left"]
    rights = [(i, info) for i, info in enumerate(task_obj_info_list) if info.get("arm") == "right"]

    pairs = []
    for li, linfo in lefts:
        ls, le = _window(linfo)
        for ri, rinfo in rights:
            rs, re = _window(rinfo)
            if ls < re and rs < le:  # overlap
                pairs.append((li, ri))
    return pairs


class TrajectoryAnnotator(CoordinateTransforms, AnnotationScoring, AnnotationExport):
    def __init__(self, gpu_id, task_obj_dict, proxies, debug_config=None, gripper_stats=None,
                 task_obj_dict_path=None, fps=10.0, vllm_config=None,
                 filter_robot=False, scoring_mode="full", verification_config=None,
                 enable_crop=False, det_threshold=0.05, nms_threshold=0.9, tracks_dir=None,
                 pointcloud_batch_size=32, robotseg_max_frames=None,
                 tracking_target_fps=None, tracking_max_frames=None, tracking_max_resolution=None,
                 max_resolution=None, pointcloud_max_resolution=None, pointcloud_resolution_level=9,
                 task_object_mode="text",
                 detection_frame_policy="pregrasp_midpoint", save_pointclouds=False):
        logging.basicConfig(level=logging.INFO)
        self.detection_model = proxies['detection']
        self.sam2_predictor = proxies['sam2']
        self.tracker = proxies['tracking']
        self.pointcloud = proxies.get('pointcloud')
        self.save_pointclouds = save_pointclouds
        self.robotseg = proxies.get('robotseg')
        self.robotseg_max_frames = robotseg_max_frames
        self.pointcloud_batch_size = pointcloud_batch_size
        self.pointcloud_max_resolution = pointcloud_max_resolution
        self.pointcloud_resolution_level = pointcloud_resolution_level
        self.tracking_target_fps = tracking_target_fps
        self.tracking_max_frames = tracking_max_frames
        self.tracking_max_resolution = tracking_max_resolution
        self.max_resolution = max_resolution
        self.fov_x = None  # set after init from dataloader.fov_x
        self.filter_robot = filter_robot
        self.scoring_mode = scoring_mode
        self.enable_crop = enable_crop
        self.det_threshold = det_threshold
        self.nms_threshold = nms_threshold
        self.detection_frame_policy = str(detection_frame_policy).lower()
        valid_detection_frame_policies = {"pregrasp_midpoint", "subtask_start"}
        if self.detection_frame_policy not in valid_detection_frame_policies:
            raise ValueError(
                f"Unknown detection_frame_policy={detection_frame_policy!r}; "
                f"expected one of {sorted(valid_detection_frame_policies)}"
            )

        self.task_obj_dict = task_obj_dict
        self.task_object_mode = str(task_object_mode).lower()
        if self.task_object_mode not in TASK_OBJECT_PARSING_MODES:
            raise ValueError(
                f"Unknown task_object_mode={task_object_mode!r}; "
                f"expected one of {sorted(TASK_OBJECT_PARSING_MODES)}"
            )
        self.gpu_id = gpu_id
        self.gripper_stats = gripper_stats or {}

        # CRITICAL: Use CPU device because the proxy will handle moving data to GPU.
        self.device = torch.device('cpu')

        self.fps = fps

        # Debug image saving (delegated to DebugImageSaver)
        self.debug_saver = DebugImageSaver(debug_config or {}, gpu_id)

        self.movement_task_types = [
            "pick", "place", "move", "put", "pick_and_place", "relocate",
            "move_to", "put_in", "put_on", "take", "bring", "insert", "push", "stack", "drag"
        ]

        self.vllm_config = vllm_config or {}
        self.verification_config = verification_config or {}


        self.tracks_dir = tracks_dir

        self.lock = None
        if task_obj_dict_path is not None:
            self.task_obj_dict_dir = task_obj_dict_path
            Path(task_obj_dict_path).parent.mkdir(parents=True, exist_ok=True)
        else:
            self.task_obj_dict_dir = None


    def _annotate_single_trajectory(self, trajectory) -> list:
        """Annotates a full trajectory, which may contain multiple subtasks.

        Returns a list of annotation dictionaries, one for each subtask.
        """
        _traj_start = time.perf_counter()
        # --- Detect grasp phases from gripper signal ---
        grasp_phases, grasp_phases_with_offset, left_phases, right_phases = \
            self._detect_gripper_phases(trajectory)

        #check if one of phases is only "interact". for bimanual case, assume that the arm with only "interact" was not used.
        #replace it with "null" phases, which will cause the LLM to ideally ignore that arm and not assign any object to it.
        if left_phases is not None and len(left_phases) == 1 and left_phases[0][2] == "interact":
            left_phases[0] = (left_phases[0][0], left_phases[0][1], "null")
        if right_phases is not None and len(right_phases) == 1 and right_phases[0][2] == "interact":
            right_phases[0] = (right_phases[0][0], right_phases[0][1], "null")


        # --- Get task object info from LLM (with caching) ---
        phase_seq_key = get_phase_sequence_key(grasp_phases)
        _t0 = time.perf_counter()
        task_obj_info_list = self._get_task_obj_info(
            trajectory, grasp_phases, phase_seq_key,
            left_phases=left_phases, right_phases=right_phases,
        )
        logging.info(f"[TIMER] llm_task_parse: {time.perf_counter() - _t0:.2f}s  traj={trajectory.name}")
        if task_obj_info_list is None:
            return []

        traj_len = len(trajectory.gripper_state)
        end_frame_grasp_release = traj_len - 1

        # For bimanual, skip only if ALL arms have no object; for single-arm, check first entry.
        if left_phases is not None:
            all_null = all(
                not info.get("object") or not str(info["object"]).strip()
                for info in task_obj_info_list
            )
            if all_null:
                logging.info(f"Skipping {trajectory.base_path} because no object specified")
                return []
        elif task_obj_info_list[0]["object"] is None or task_obj_info_list[0]["object"].strip() == "":
            logging.info(f"Skipping {trajectory.base_path} because no object specified")
            return []

        # --- Align grasp phases with LLM subtask boundaries ---
        if left_phases is not None:
            self._align_grasp_phase_frames_bimanual(task_obj_info_list, left_phases, right_phases)
        else:
            self._align_grasp_phase_frames(task_obj_info_list, grasp_phases)

        task_obj_info_list = [
        info for info in task_obj_info_list
        if info.get("object") and str(info["object"]).strip()                                                                                                                                                                                           
            ]
        if not task_obj_info_list:
            return []

        # --- Find overlapping bimanual pairs (only for bimanual trajectories) ---
        bimanual_pairs = []
        bimanual_handled = set()
        if left_phases is not None:
            bimanual_pairs = _find_bimanual_pairs(task_obj_info_list)
            for li, ri in bimanual_pairs:
                bimanual_handled.add(li)
                bimanual_handled.add(ri)

        # --- Process each subtask ---
        annotations = []
        emitted_bimanual_pairs = set()

        for subtask_idx, task_obj_info in enumerate(task_obj_info_list):
            if subtask_idx in bimanual_handled:
                # Find the pair this subtask belongs to
                for li, ri in bimanual_pairs:
                    pair_key = (li, ri)
                    if subtask_idx in (li, ri) and pair_key not in emitted_bimanual_pairs:
                        emitted_bimanual_pairs.add(pair_key)
                        ann = self._annotate_bimanual_subtask(
                            trajectory=trajectory,
                            left_subtask_idx=li,
                            left_info=task_obj_info_list[li],
                            right_subtask_idx=ri,
                            right_info=task_obj_info_list[ri],
                            task_obj_info_list=task_obj_info_list,
                            left_phases=left_phases,
                            right_phases=right_phases,
                            grasp_phases_with_offset=grasp_phases_with_offset,
                            traj_len=traj_len,
                            end_frame_grasp_release=end_frame_grasp_release,
                        )
                        if ann is not None:
                            annotations.append(ann)
                        break
                continue  # skip single-arm path for this subtask

            # In a bimanual trajectory a non-paired subtask still knows which
            # arm it belongs to — pass that through for spatial filtering and
            # debug labelling.
            arm = task_obj_info.get("arm") if left_phases is not None else None
            ann = self._annotate_subtask(
                trajectory=trajectory,
                subtask_idx=subtask_idx,
                task_obj_info=task_obj_info,
                task_obj_info_list=task_obj_info_list,
                grasp_phases=grasp_phases,
                grasp_phases_with_offset=grasp_phases_with_offset,
                traj_len=traj_len,
                end_frame_grasp_release=end_frame_grasp_release,
                arm=arm,
            )
            if ann is not None:
                annotations.append(ann)

        _traj_elapsed = time.perf_counter() - _traj_start
        n_anns = len(annotations)
        ann_per_sec = n_anns / _traj_elapsed if _traj_elapsed > 0 else float('inf')
        logging.info(
            f"[TIMER] trajectory_total: {_traj_elapsed:.2f}s  anns={n_anns}  "
            f"ann/s={ann_per_sec:.3f}  traj={trajectory.name}"
        )
        return annotations

    # ------------------------------------------------------------------
    # Phase 1: Gripper phase detection
    # ------------------------------------------------------------------

    def _detect_gripper_phases(self, trajectory):
        """Detect grasp phases from gripper signal and apply offset.

        Returns:
            (grasp_phases, grasp_phases_with_offset, left_phases, right_phases)
            left_phases and right_phases are None for single-arm trajectories.
        """
        kwargs = dict(closed_threshold=0.52, expected_interact_duration=None, hysteresis_offset=0.05)

        if (getattr(trajectory, 'left_gripper_state', None) is not None and
                getattr(trajectory, 'right_gripper_state', None) is not None):
            left_phases  = get_gripper_close_phases(trajectory.left_gripper_state, **kwargs)
            right_phases = get_gripper_close_phases(trajectory.right_gripper_state, **kwargs)
            merged = sorted(left_phases + right_phases, key=lambda p: p[0])
            return merged, apply_phase_offset(merged, offset_ratio=0.3), left_phases, right_phases
        else:
            grasp_phases = get_gripper_close_phases(trajectory.gripper_state, **kwargs)
            return grasp_phases, apply_phase_offset(grasp_phases, offset_ratio=0.3), None, None

    # ------------------------------------------------------------------
    # Phase 2: LLM task object info (with caching)
    # ------------------------------------------------------------------

    def _get_task_obj_info(self, trajectory, grasp_phases, phase_seq_key,
                           left_phases=None, right_phases=None):
        """Get task_obj_info_list from LLM or cache.

        Returns list of task_obj_info dicts, or None to skip.
        """
        effective_mode = None
        if left_phases is not None:
            left_seq_key  = get_phase_sequence_key(left_phases)
            right_seq_key = get_phase_sequence_key(right_phases)
            cache_key = ("bimanual", trajectory.lang_ann.strip(), left_seq_key, right_seq_key)
        else:
            effective_mode = resolve_task_object_parsing_mode(
                self.task_object_mode,
                grasp_phases,
            )
            cache_key = task_object_cache_key(
                effective_mode,
                trajectory.name,
                trajectory.lang_ann,
                phase_seq_key,
            )

        if cache_key not in self.task_obj_dict or self.task_obj_dict[cache_key] is None:
            if left_phases is None and effective_mode == "semantic":
                logging.error(
                    "Semantic task-object cache miss for trajectory %s. Run "
                    "extract_task_objects.py with extract_task_objects.mode=%s first. "
                    "Expected cache_key=%s",
                    trajectory.name,
                    self.task_object_mode,
                    cache_key,
                )
                return None
            logging.info(f"Cache miss for trajectory {trajectory.name} with cache_key={cache_key}. Querying LLM...")
            language_instruction = trajectory.lang_ann

            _t_llm = time.perf_counter()
            _model = self.vllm_config.get("model_name", "qwen3-vl-30b")
            _base  = self.vllm_config.get("base_url", "http://localhost:8000/v1")
            _ebody = self.vllm_config.get("extra_body")
            if left_phases is not None:
                res = extract_task_obj_bimanual(
                    language_instruction,
                    left_phases,
                    right_phases,
                    model=_model,
                    base_url=_base,
                    extra_body=_ebody,
                )
            else:
                res = extract_task_obj_with_phases(
                    language_instruction,
                    grasp_phases,
                    model=_model,
                    base_url=_base,
                    temperature=self.vllm_config.get("temperature", 1.0),
                    extra_body=_ebody,
                )
            logging.info(f"[TIMER] llm_api_call: {time.perf_counter() - _t_llm:.2f}s")
            if not res:
                return None

            # Save to cache (with optional file-based persistence)
            if self.lock:
                self.lock.acquire()
            try:
                if cache_key not in self.task_obj_dict or self.task_obj_dict[cache_key] is None:
                    if self.task_obj_dict_dir and os.path.exists(self.task_obj_dict_dir):
                        with open(self.task_obj_dict_dir, "rb") as f:
                            self.task_obj_dict = pickle.load(f)

                    self.task_obj_dict[cache_key] = res

                    if self.task_obj_dict_dir:
                        with open(self.task_obj_dict_dir, "wb") as f:
                            pickle.dump(self.task_obj_dict, f)
                    logging.info(f"Added cache_key={cache_key} to task index")
            finally:
                if self.lock:
                    self.lock.release()

        task_obj_info_list = self.task_obj_dict[cache_key]
        if not isinstance(task_obj_info_list, list):
            task_obj_info_list = [task_obj_info_list]

        return task_obj_info_list

    # ------------------------------------------------------------------
    # Phase 3: Align grasp phase frames
    # ------------------------------------------------------------------

    def _align_grasp_phase_frames(self, task_obj_info_list, grasp_phases):
        """Assign start_frame/end_frame from grasp_phases to each subtask's grasp_phases."""
        grasp_phase_start_end_list = [(phase[0], phase[1]) for phase in grasp_phases]
        cur_it_idx = 0
        try:
            for idx in range(len(task_obj_info_list)):
                for grasp_phase_idx in range(len(task_obj_info_list[idx]["grasp_phases"])):
                    task_obj_info_list[idx]["grasp_phases"][grasp_phase_idx]["start_frame"] = grasp_phase_start_end_list[cur_it_idx][0]
                    task_obj_info_list[idx]["grasp_phases"][grasp_phase_idx]["end_frame"] = grasp_phase_start_end_list[cur_it_idx][1]
                    cur_it_idx += 1
        except Exception as e:
            logging.error(f"Error occurred while aligning grasp phases: {e}")
            logging.error(f"Current index: {cur_it_idx}")
            logging.error(f"Grasp phase start-end list: {grasp_phase_start_end_list}")
            raise e

    def _align_grasp_phase_frames_bimanual(self, task_obj_info_list, left_phases, right_phases):
        """Assign start_frame/end_frame for bimanual subtasks.

        Left-arm subtasks consume from left_phases; right-arm subtasks from right_phases.
        """
        left_iter  = iter([(p[0], p[1]) for p in left_phases])
        right_iter = iter([(p[0], p[1]) for p in right_phases])
        try:
            for subtask in task_obj_info_list:
                src = left_iter if subtask.get("arm") == "left" else right_iter
                for gp in subtask["grasp_phases"]:
                    start, end = next(src)
                    gp["start_frame"] = start
                    gp["end_frame"]   = end
        except StopIteration as e:
            logging.error(f"_align_grasp_phase_frames_bimanual: ran out of phases: {e}")
            raise

    # ------------------------------------------------------------------
    # Phase 4: Per-subtask annotation
    # ------------------------------------------------------------------

    def _annotate_subtask(self, trajectory, subtask_idx, task_obj_info, task_obj_info_list,
                          grasp_phases, grasp_phases_with_offset, traj_len, end_frame_grasp_release,
                          arm=None):
        """Annotate a single subtask. Returns annotation dict or None.

        arm: "left" | "right" | None.  Set for bimanual trajectories where
             this subtask's arm had no overlapping partner (idle other arm).
             Used for spatial detection filtering and debug labelling.
        """
        obj = task_obj_info.get("object")
        action = task_obj_info.get("action")
        if not obj or not str(obj).strip():
            logging.info(f"Skipping subtask {subtask_idx} of {trajectory.name}: no object specified")
            return None
        if not action or not str(action).strip():
            logging.info(f"Skipping subtask {subtask_idx} of {trajectory.name}: no action specified")
            return None

        # --- Compute frame boundaries ---
        start_frame, end_frame = self._compute_frame_boundaries(
            subtask_idx, task_obj_info, task_obj_info_list,
            grasp_phases_with_offset, traj_len, end_frame_grasp_release,
        )

        frames_subset = trajectory.frames[start_frame:end_frame]
        frames_subset, _global_scale = self._resize_frames_global(frames_subset)
        full_frames_subset = frames_subset

        if len(frames_subset) < 6:
            logging.info(f"Skipping subtask {subtask_idx} of {trajectory.name} because too few frames ({len(frames_subset)}) in range [{start_frame}, {end_frame}]")
            return None

        # --- Build subtask-specific phase annotations ---
        subtask_phase_annotations, subtask_phase_sequence, task_grasp_phase_frame_indices = \
            self._build_subtask_phases(task_obj_info, grasp_phases, grasp_phases_with_offset, start_frame, end_frame, len(frames_subset))
        det_frame_idx = task_grasp_phase_frame_indices[OBJ_DET_IDX]
        phase_mask_keyframes = self._phase_mask_keyframes(
            subtask_phase_annotations,
            start_frame,
            len(frames_subset),
        )
        phase_mask_keyframes.insert(
            0,
            ("detection", det_frame_idx, start_frame + det_frame_idx),
        )

        # --- Convert frames to tensor ---
        frames_torch = frames_to_tensor(frames_subset).to(self.device)

        # --- Detect objects ---
        _t0 = time.perf_counter()
        det_result = self._detect_objects(frames_subset, task_obj_info, task_grasp_phase_frame_indices, arm=arm)
        logging.info(f"[TIMER] detect_objects: {time.perf_counter() - _t0:.2f}s  traj={trajectory.name} subtask={subtask_idx}")

        # In bimanual trajectories with one idle arm, restrict detections to
        # the expected image half so the active arm's objects are not confused
        # with those on the other side.
        if arm is not None:
            det_result = self._filter_detections_by_arm(det_result, arm, frames_subset.shape[2], frames_subset.shape[1])

        # --- Detect targets ---
        _t0 = time.perf_counter()
        target_result = self._detect_targets(frames_subset, task_obj_info, det_result, nms)
        logging.info(f"[TIMER] detect_targets: {time.perf_counter() - _t0:.2f}s  traj={trajectory.name} subtask={subtask_idx}")

        # --- Maybe crop ---
        crop_info = self._maybe_crop(
            frames_subset, frames_torch, det_result, target_result, trajectory.dataset_name,
        )
        # Use cropped frames/tensor if cropping was applied
        if crop_info.did_crop:
            frames_subset = crop_info._frames_subset
            frames_torch = crop_info._frames_torch
            det_result = crop_info._det_result

        # --- Subsample frames for tracking ---
        # Tracking starts from det_frame_idx: object is visible and unoccluded there.
        # Since the object is stationary before the grasp, initial_object_box still represents start_frame.
        frames_torch_tracking, _track_indices, _scaled_phases, _T, _tracking_scale, _tracking_actual_fps = self._subsample_frames_for_tracking(
            frames_torch, det_frame_idx, task_obj_info, start_frame,
            traj_name=trajectory.name, subtask_idx=subtask_idx,
        )
        tracking_frame_indices = (_track_indices + det_frame_idx).astype(np.int32)

        # --- RobotSeg gripper segmentation (replaces box-detection + SAM2 gripper steps) ---
        robotseg_video_masks = None
        robotseg_gripper_mask = None
        if self.robotseg is not None:
            _t0 = time.perf_counter()
            robotseg_frame_indices = self._robotseg_frame_indices_for_tracking(
                n_frames=len(frames_subset),
                det_frame_idx=det_frame_idx,
                tracking_frame_indices=tracking_frame_indices,
                required_frame_indices=[item[1] for item in phase_mask_keyframes],
            )
            gripper_box_rs, robotseg_gripper_mask, robotseg_video_masks = \
                self._segment_gripper_with_robotseg(
                    frames_subset,
                    det_frame_idx,
                    frame_indices=robotseg_frame_indices,
                )
            logging.info(f"[TIMER] robotseg: {time.perf_counter() - _t0:.2f}s  traj={trajectory.name} subtask={subtask_idx}")
            if gripper_box_rs is not None:
                self._set_robotseg_gripper_detection(
                    det_result,
                    gripper_box_rs,
                    robotseg_video_masks,
                    det_frame_idx,
                )
            else:
                logging.info(
                    "RobotSeg produced no usable gripper mask; running detector fallback"
                )
                det_result = self._restore_detector_gripper_fallback(
                    frames_subset,
                    task_obj_info,
                    det_result,
                    arm=arm,
                )

        # --- Filter by SAM2 segmentation ---
        _t0 = time.perf_counter()
        masks, valid_masks, gripper_mask = self._segment_with_sam2(
            frames_subset, det_result, det_frame_idx, robotseg_gripper_mask=robotseg_gripper_mask
        )
        logging.info(f"[TIMER] sam2_segment: {time.perf_counter() - _t0:.2f}s  traj={trajectory.name} subtask={subtask_idx}")
        if masks is None:
            return None
        target_result = self._segment_targets_with_sam2(frames_subset, target_result)

        # --- Track and select best box ---
        _t0 = time.perf_counter()
        _ib_tidx = {}

        tracks_file = None
        tracking_result = self._track_and_select(
            frames_torch_tracking, det_result, target_result, masks, valid_masks, gripper_mask,
            _scaled_phases, trajectory,
            intermediate_detections=det_result.intermediate_detections,
            det_frame_idx=det_frame_idx,
            frames_np=frames_subset[_track_indices + det_frame_idx] if len(_track_indices) > 0 else frames_subset[det_frame_idx:],
            tracking_scale=_tracking_scale,
            tracking_frame_indices=tracking_frame_indices,
        )
        logging.info(f"[TIMER] tracking: {time.perf_counter() - _t0:.2f}s  traj={trajectory.name} subtask={subtask_idx} frames={frames_torch_tracking.shape[1]}")


        if tracking_result.intermediate_boxes:
            _ib_tidx = {
                int(frame_idx): int(np.argmin(
                    np.abs(tracking_frame_indices - int(frame_idx))
                ))
                for frame_idx in tracking_result.intermediate_boxes
            }

        tracking_result.raw_tracks_data = _augment_tracks_with_robotseg(
            _augment_tracks_with_tracking_frame_indices(
                _augment_tracks_with_targets(tracking_result.raw_tracks_data, target_result),
                tracking_frame_indices,
            ),
            robotseg_video_masks,
        )
        if tracking_result.raw_tracks_data is not None:
            tracking_result.raw_tracks_data["tracking_actual_fps"] = np.array(_tracking_actual_fps, dtype=np.float32)
            tracking_result.raw_tracks_data["source_fps"] = np.array(self.fps, dtype=np.float32)

        # Inject fine-grained sub-scores into confidence_breakdown for post-hoc reweighting
        tracking_result = self._augment_confidence_breakdown_with_replay_signals(
            tracking_result,
            _scaled_phases,
            task_obj_info,
            arm=arm,
            scoring_fps=_tracking_actual_fps,
        )
        self._save_phase_keyframe_masks_to_arrays_blob(
            tracking_result,
            frames_subset,
            phase_mask_keyframes,
            robotseg_video_masks,
            full_frames_subset=full_frames_subset,
            intermediate_detections=det_result.intermediate_detections,
            crop_info=crop_info,
            global_scale=_global_scale,
        )

        if tracking_result.best_obj_box is None:
            logging.info(f"Skipping {trajectory.name} because no boxes found after tracking verification")
            return None

        tracks_file = self._save_tracks_npz(
            trajectory.name,
            subtask_idx,
            tracking_result.raw_tracks_data,
            crop_info=crop_info,
            global_scale=_global_scale,
            start_frame=start_frame,
        )

        # Clean up tensors
        del frames_torch, frames_torch_tracking
        torch.cuda.empty_cache()

        # --- Resolve target box ---
        tracking_result = self._resolve_target_box(
            tracking_result, target_result, frames_subset,
        )

        # --- Restore coordinates if cropped ---
        final = self._restore_coordinates(
            tracking_result, crop_info,
            target_candidate_box_confidences=[tb["det_confidence"] for tb in target_result.target_boxes],
        )

        # --- Unscale to original resolution if global resize was applied ---
        if _global_scale != (1.0, 1.0):
            final = self._unscale_final(final, _global_scale)

        # --- Apply tool task logic ---
        final = self._apply_tool_task_logic(final, det_result)

        # --- Align replay/debug boxes with final output coordinates ---
        final = self._final_with_output_coordinate_confidence_breakdown(
            final,
            crop_info,
            global_scale=_global_scale,
        )

        # --- Cross-model verification ---
        final = self._run_cross_model_verification(
            final, frames_subset, start_frame, end_frame, task_obj_info, trajectory,
        )

        obj_traces_cotracker_point, obj_traces_cotracker_point_meta = \
            self._compute_export_centroid_trace(final, tracking_result.raw_tracks_data)
        tool_traces_cotracker_point = (
            obj_traces_cotracker_point.copy()
            if (det_result.is_tool_task and obj_traces_cotracker_point is not None)
            else None
        )

        # --- Build annotation dict ---
        annotation = self._build_annotation_dict(
            trajectory, subtask_idx, start_frame, end_frame,
            final, det_result, task_obj_info, task_obj_info_list,
            grasp_phases, subtask_phase_annotations, subtask_phase_sequence,
            tracks_file=tracks_file, det_frame_idx=det_frame_idx,
            obj_traces_cotracker_point=obj_traces_cotracker_point,
            obj_traces_cotracker_point_meta=obj_traces_cotracker_point_meta,
            tool_traces_cotracker_point=tool_traces_cotracker_point,
            crop_info=crop_info,
            global_scale=_global_scale,
        )
        # Tag arm for bimanual single-arm subtasks
        if arm is not None:
            annotation["arm"] = arm

        self._save_debug(
            trajectory=trajectory, subtask_idx=subtask_idx,
            start_frame=start_frame, end_frame=end_frame, det_frame_idx=det_frame_idx,
            final=final, det_result=det_result, target_result=target_result,
            crop_info=crop_info, tracking_result=tracking_result, ib_tidx=_ib_tidx,
            annotation=annotation, frames_subset=frames_subset,
        )

        return build_publication_annotation(
            annotation,
            original_image_size_hw=tuple(map(int, trajectory.frames[0].shape[:2])),
            embed_arrays=False,
            save_pointclouds=self.save_pointclouds,
        )

    # ------------------------------------------------------------------
    # Sub-methods of _annotate_subtask
    # ------------------------------------------------------------------

    def _subsample_frames_for_tracking(self, frames_torch, det_frame_idx, task_obj_info,
                                       start_frame, traj_name="", subtask_idx=""):
        """Subsample and optionally resize tracking frames.

        Returns:
            (frames_torch_tracking, track_indices, scaled_phases, T, tracking_scale, actual_fps)
            T is the number of frames after slicing from det_frame_idx.
            tracking_scale is (sx, sy) = (new_W/orig_W, new_H/orig_H); (1.0, 1.0) if no resize.
        """
        target_tracking_fps = self.tracking_target_fps or 24.0
        max_tracking_frames = self.tracking_max_frames or 800
        frames_torch = frames_torch[:, det_frame_idx:]
        _T = frames_torch.shape[1]
        src_fps = float(self.fps) if self.fps else target_tracking_fps

        if _T == 0:
            track_indices = np.array([], dtype=int)
            frames_torch_tracking = frames_torch
            actual_fps = src_fps
        else:
            if src_fps > target_tracking_fps:
                target_count = max(2, int(np.ceil(_T * target_tracking_fps / src_fps)))
                target_count = min(target_count, _T)
                track_indices = np.linspace(0, _T - 1, target_count, dtype=int)
                track_indices = np.unique(track_indices)
                frames_torch_tracking = frames_torch[:, track_indices]
            else:
                track_indices = np.arange(_T)
                frames_torch_tracking = frames_torch

            if len(track_indices) > max_tracking_frames:
                track_indices = np.linspace(0, _T - 1, max_tracking_frames, dtype=int)
                track_indices = np.unique(track_indices)
                frames_torch_tracking = frames_torch[:, track_indices]
                actual_fps = len(track_indices) / (_T / src_fps) if _T > 1 else src_fps
                target_tracking_fps = actual_fps

            actual_fps = len(track_indices) / (_T / src_fps) if _T > 1 else src_fps

        logging.info(
            f"[TIMER] frames_for_tracking: {frames_torch_tracking.shape[1]} (orig={_T}, "
            f"src_fps={self.fps}, target_fps={target_tracking_fps}, actual_fps={actual_fps:.3f})  "
            f"traj={traj_name} subtask={subtask_idx}"
        )

        # --- Spatial resolution resize ---
        tracking_scale = (1.0, 1.0)
        if self.tracking_max_resolution is not None and frames_torch_tracking.shape[1] > 0:
            import torch.nn.functional as F
            _orig_H = frames_torch_tracking.shape[3]  # (1, T, C, H, W)
            _orig_W = frames_torch_tracking.shape[4]
            min_edge = min(_orig_H, _orig_W)
            if min_edge > self.tracking_max_resolution:
                scale = self.tracking_max_resolution / min_edge
                new_H = int(round(_orig_H * scale))
                new_W = int(round(_orig_W * scale))
                # Fuse batch+time dims for interpolate, then restore
                _frames_4d = frames_torch_tracking.squeeze(0)  # (T, C, H, W)
                # Preserve the pre-uint8-IPC interpolation behavior for this optional
                # resize path. Default/no-resize tracking remains uint8 until the GPU.
                if not torch.is_floating_point(_frames_4d):
                    _frames_4d = _frames_4d.float()
                _frames_4d = F.interpolate(_frames_4d, size=(new_H, new_W), mode='bilinear', align_corners=False)
                frames_torch_tracking = _frames_4d.unsqueeze(0)  # (1, T, C, H, W)
                tracking_scale = (new_W / _orig_W, new_H / _orig_H)
                logging.info(
                    f"[TIMER] tracking_resize: {_orig_H}x{_orig_W} -> {new_H}x{new_W} (scale={scale:.3f})"
                    f"  traj={traj_name} subtask={subtask_idx}"
                )

        # Scale grasp phase indices into sampled time space via searchsorted on track_indices.
        # Phases are absolute; subtract start_frame + det_frame_idx to get relative to tracking start.
        raw_phases = task_obj_info.get("grasp_phases", [])
        track_start = start_frame + det_frame_idx
        if len(track_indices) != _T and raw_phases:
            scaled_phases = [
                {**p,
                 "start_frame": int(np.searchsorted(track_indices, max(0, p.get("start_frame", 0) - track_start))),
                 "end_frame":   int(np.searchsorted(track_indices, max(0, p.get("end_frame",   0) - track_start)))}
                for p in raw_phases
            ]
        else:
            scaled_phases = raw_phases

        return frames_torch_tracking, track_indices, scaled_phases, _T, tracking_scale, actual_fps

    def _robotseg_frame_indices_for_tracking(
        self,
        n_frames,
        det_frame_idx,
        tracking_frame_indices,
        required_frame_indices=None,
    ):
        """Frame indices to run RobotSeg on, guaranteed to include det_frame_idx.

        Indices are in frames_subset coordinates.  The pool is built from:
          - a sparse prefix covering [0, det_frame_idx) for pre-grasp context
          - tracking_frame_indices for proximity-check alignment in the post-det window
          - required_frame_indices, such as grasp/interact/release keyframes
        The union is then subsampled down to robotseg_max_frames, keeping the
        detection and required keyframes.
        """
        if n_frames <= 0:
            return np.array([], dtype=np.int32)

        if self.robotseg_max_frames is None or n_frames <= self.robotseg_max_frames:
            return None

        cap = int(self.robotseg_max_frames)

        # Sparse prefix: up to 8 evenly-spaced frames before det_frame_idx
        pre_end = int(det_frame_idx) if det_frame_idx is not None else 0
        if pre_end > 0:
            n_prefix = min(8, pre_end)
            prefix = np.linspace(0, pre_end - 1, n_prefix, dtype=int)
        else:
            prefix = np.array([], dtype=int)

        # Post-det pool: the actual tracking frames (sparse on low-FPS, subsampled on high-FPS)
        tracking = np.asarray(tracking_frame_indices, dtype=int).reshape(-1)
        tracking = tracking[(tracking >= 0) & (tracking < n_frames)]

        required = np.asarray(
            [] if required_frame_indices is None else required_frame_indices,
            dtype=int,
        ).reshape(-1)
        required = required[(required >= 0) & (required < n_frames)]
        if det_frame_idx is not None:
            required = np.append(required, int(det_frame_idx))
        required = np.unique(required)

        pool = np.unique(np.concatenate([prefix, tracking, required]))

        # Subsample only the optional pool; required phase keyframes always survive.
        if len(pool) > cap:
            optional = np.setdiff1d(pool, required, assume_unique=True)
            n_optional = max(0, cap - len(required))
            if n_optional > 0 and len(optional) > n_optional:
                sampled_idx = np.linspace(0, len(optional) - 1, n_optional, dtype=int)
                optional = optional[sampled_idx]
            elif n_optional == 0:
                optional = np.array([], dtype=int)
            pool = np.unique(np.concatenate([required, optional]))

        return pool.astype(np.int32)


    def _compute_frame_boundaries(self, subtask_idx, task_obj_info, task_obj_info_list,
                                  grasp_phases_with_offset, traj_len, end_frame_grasp_release,
                                  is_bimanual_arm=False):
        """Compute (start_frame, end_frame) for a subtask."""
        task_grasp_phases = task_obj_info.get("grasp_phases", [])

        if "object_instance" in task_obj_info and task_grasp_phases and not is_bimanual_arm:
            return semantic_episode_window(task_grasp_phases, traj_len)

        if task_grasp_phases:
            start_frame = task_grasp_phases[0].get("start_frame", 0)
            end_frame = task_grasp_phases[-1].get("end_frame", traj_len - 1)
        else:
            num_subtasks = len(task_obj_info_list)
            if num_subtasks == 1:
                start_frame = 0
                end_frame = traj_len - 1
            else:
                phases_per_subtask = max(1, len(grasp_phases_with_offset) // num_subtasks)
                phase_start_idx = subtask_idx * phases_per_subtask
                phase_end_idx = min((subtask_idx + 1) * phases_per_subtask - 1, len(grasp_phases_with_offset) - 1)

                if phase_start_idx < len(grasp_phases_with_offset):
                    start_frame = grasp_phases_with_offset[phase_start_idx][0]
                    end_frame = grasp_phases_with_offset[phase_end_idx][1]
                else:
                    start_frame = 0 if subtask_idx == 0 else traj_len // num_subtasks * subtask_idx
                    end_frame = traj_len - 1 if subtask_idx == num_subtasks - 1 else traj_len // num_subtasks * (subtask_idx + 1)

        # For bimanual arms, skip the sequential subtask boundary logic — they act simultaneously.
        if not is_bimanual_arm:
            if subtask_idx > 0:
                # Walk back to find the nearest previous subtask that has grasp_phases set.
                prev_phases = None
                for prev_idx in range(subtask_idx - 1, -1, -1):
                    phases = task_obj_info_list[prev_idx].get("grasp_phases", [])
                    if phases:
                        prev_phases = phases
                        break
                if prev_phases is not None:
                    prev_end_frame = prev_phases[-1]["end_frame"]
                    cur_grasp_phases = task_obj_info["grasp_phases"]
                    if len(cur_grasp_phases) > 1:
                        next_interact_start = None
                        for phase in cur_grasp_phases:
                            if phase["phase_type"] == "grasp":
                                next_interact_start = phase["end_frame"]
                                break
                        if next_interact_start is not None:
                            start_frame = prev_end_frame + (next_interact_start - prev_end_frame) * 3 // 4
                    else:
                        buffer = min(15, (end_frame - start_frame) // 4)
                        start_frame = prev_end_frame + buffer

            if subtask_idx == len(task_obj_info_list) - 1:
                end_frame = traj_len - 1

        start_frame = max(0, min(start_frame, traj_len - 1))
        end_frame = max(start_frame + 1, min(end_frame, traj_len - 1))
        end_frame = min(end_frame, end_frame_grasp_release)

        return start_frame, end_frame

    def _build_subtask_phases(self, task_obj_info, grasp_phases, grasp_phases_with_offset,
                              start_frame, end_frame, num_frames):
        """Build phase annotations and frame indices for a subtask.

        Returns:
            (subtask_phase_annotations, subtask_phase_sequence,
             task_grasp_phase_frame_indices)
            The detection keyframe is at task_grasp_phase_frame_indices[OBJ_DET_IDX].
        """
        subtask_phase_annotations = []
        subtask_phase_sequence = []
        task_grasp_phases = task_obj_info.get("grasp_phases", [])
        task_grasp_phase_frame_indices = [0, num_frames - 1]

        if task_grasp_phases:
            task_grasp_phase_frame_indices = [
                phase["end_frame"] for phase in task_grasp_phases
                if phase["end_frame"] >= start_frame and phase["start_frame"] <= end_frame
            ]
            task_grasp_phase_frame_indices = [start_frame] + task_grasp_phase_frame_indices
            task_grasp_phase_frame_indices[-1] = task_grasp_phase_frame_indices[-1] - 1

            # Convert absolute frame indices to relative indices for frames_subset
            task_grasp_phase_frame_indices = [
                max(0, min(idx - start_frame, num_frames - 1))
                for idx in task_grasp_phase_frame_indices
            ]

            # Insert the detection keyframe at index [OBJ_DET_IDX]. Some datasets
            # provide first-frame GT boxes, while the default production policy
            # uses a pre-grasp frame where the approached object is most visible.
            if self.detection_frame_policy == "subtask_start":
                detection_frame_abs = start_frame
            else:
                first_grasp = next(
                    (p for p in task_grasp_phases if p.get("phase_type") == "grasp"), None
                )
                if first_grasp is not None:
                    detection_frame_abs = (start_frame + first_grasp["end_frame"]) // 2
                else:
                    detection_frame_abs = start_frame
            det_frame_rel = max(0, min(detection_frame_abs - start_frame, num_frames - 1))
            task_grasp_phase_frame_indices.insert(OBJ_DET_IDX, det_frame_rel)

            for phase_info in task_grasp_phases:
                phase_start = phase_info.get("start_frame", 0)
                phase_end = phase_info.get("end_frame", 0)
                phase_type = phase_info.get("phase_type", "unknown")
                description = phase_info.get("description")

                subtask_phase_annotations.append({
                    "phase": phase_type,
                    "start_frame": phase_start,
                    "end_frame": phase_end,
                    "description": description,
                })
                subtask_phase_sequence.append(phase_type)
        else:
            # No grasp phases: detection frame is frame 0. Insert it at OBJ_DET_IDX so the
            # invariant task_grasp_phase_frame_indices[OBJ_DET_IDX] == detection frame holds.
            task_grasp_phase_frame_indices.insert(OBJ_DET_IDX, 0)
            for phase_start, phase_end, phase_name in grasp_phases_with_offset:
                if phase_end >= start_frame and phase_start <= end_frame:
                    clipped_start = max(phase_start, start_frame)
                    clipped_end = min(phase_end, end_frame)
                    subtask_phase_annotations.append({
                        "phase": phase_name,
                        "start_frame": clipped_start,
                        "end_frame": clipped_end,
                        "description": None,
                    })
                    subtask_phase_sequence.append(phase_name)

        return subtask_phase_annotations, subtask_phase_sequence, task_grasp_phase_frame_indices


    def _detect_objects(self, frames_subset, task_obj_info, task_grasp_phase_frame_indices, arm=None):
        """Run object detection. Returns DetectionResult.

        arm: "left" | "right" | None.  When set, uses arm-specific robot prompts.
        task_grasp_phase_frame_indices[1] is the detection frame (midpoint of first grasp phase).
        """
        tool_name = task_obj_info.get("tool_name_required")
        tool_usage = task_obj_info.get("tool_usage_description")
        is_tool_task = tool_name is not None and str(tool_name).strip() != ""

        tool_boxes_fullres = None
        object_box_for_tool_task = None
        object_box_for_tool_task_confidence = None

        pnp_task = any(mt in (task_obj_info.get("action") or "") for mt in self.movement_task_types)

        all_robot_candidates = None  # will be set below for debug saving

        # task_grasp_phase_frame_indices[1] is the detection frame (midpoint of first grasp phase).
        # Use pred_boxes[1] for objects; pred_box_robot covers all keyframes as before.
        det_frame_idx = task_grasp_phase_frame_indices[1] if len(task_grasp_phase_frame_indices) > 1 else task_grasp_phase_frame_indices[0]
        area_threshold = (
            DEFAULT_MAX_BOX_AREA_RATIO * frames_subset.shape[1] * frames_subset.shape[2]
        )

        if is_tool_task:
            logging.info(f"Tool task detected: tool={tool_name}, object={task_obj_info['object']}, usage={tool_usage}")

            tool_task_obj_info = {
                "object": tool_name,
                "start_location": None,
                "target_location": None,
                "action": task_obj_info["action"],
            }

            pred_boxes, pred_box_robot, all_robot_candidates = get_interacted_obj_boxes(
                self.detection_model, frames_subset[task_grasp_phase_frame_indices],
                tool_task_obj_info, threshold=self.det_threshold, return_robot_det=True,
                return_all_robot_dets=True, arm=arm, nms_threshold=self.nms_threshold,
                skip_robot_detection=self.robotseg is not None,
            )
            valid_tool_indices = filter_boxes_by_size(pred_boxes[1].xyxy, area_threshold)
            pred_boxes[1] = pred_boxes[1][valid_tool_indices]
            tool_boxes_fullres = pred_boxes[1].xyxy.copy() if len(pred_boxes[1].xyxy) > 0 else np.empty((0, 4))

            pred_boxes_object = get_interacted_obj_boxes(
                self.detection_model, frames_subset[[det_frame_idx]], task_obj_info,
                threshold=TOOL_OBJECT_DETECTION_THRESHOLD,
                nms_threshold=self.nms_threshold,
                skip_robot_detection=self.robotseg is not None,
            )
            valid_object_indices = filter_boxes_by_size(pred_boxes_object[0].xyxy, area_threshold)
            pred_boxes_object[0] = pred_boxes_object[0][valid_object_indices]

            pred_boxes_np = pred_boxes[1].xyxy
            all_boxes_np = pred_boxes_np.copy()
            detected_boxes_fullres = pred_boxes[1].xyxy.copy()

            if len(pred_boxes_object[0].xyxy) > 0:
                best_obj_idx = np.argmax(pred_boxes_object[0].confidence)
                object_box_for_tool_task = pred_boxes_object[0].xyxy[[best_obj_idx]]
                object_box_for_tool_task_confidence = float(pred_boxes_object[0].confidence[best_obj_idx])
                all_boxes_np = np.concatenate([all_boxes_np, pred_boxes_object[0].xyxy], axis=0)
        else:
            pred_boxes, pred_box_robot, all_robot_candidates = get_interacted_obj_boxes(
                self.detection_model, frames_subset[task_grasp_phase_frame_indices],
                task_obj_info, threshold=self.det_threshold, return_robot_det=True,
                text_threshold=STANDARD_DETECTION_TEXT_THRESHOLD,
                return_all_robot_dets=True, arm=arm,
                nms_threshold=self.nms_threshold,
                skip_robot_detection=self.robotseg is not None,
            )
            valid_box_indices = filter_boxes_by_size(pred_boxes[1].xyxy, area_threshold)
            pred_boxes[1] = pred_boxes[1][valid_box_indices]
            pred_boxes_np = pred_boxes[1].xyxy
            all_boxes_np = pred_boxes_np.copy()
            detected_boxes_fullres = pred_boxes[1].xyxy.copy()

        # Filter tiny boxes
        box_areas_frac = (pred_boxes[1].xyxy[:, 2] - pred_boxes[1].xyxy[:, 0]) * \
                         (pred_boxes[1].xyxy[:, 3] - pred_boxes[1].xyxy[:, 1]) / \
                         (frames_subset[0].shape[0] * frames_subset[0].shape[1])
        valid_indices = box_areas_frac > 0.001
        pred_boxes[1] = pred_boxes[1][valid_indices]

        # Build candidate_boxes list
        candidate_boxes = []
        for det_idx in range(len(pred_boxes[1].xyxy)):
            candidate_boxes.append({
                "box": [round(coord) for coord in pred_boxes[1].xyxy[det_idx]],
                "det_confidence": float(pred_boxes[1].confidence[det_idx]),
                "frame_index": det_frame_idx,
                "confidence": float(pred_boxes[1].confidence[det_idx]),
            })

        # Robot detections — store one entry per keyframe: {frame_idx: {"box": [...], "confidence": float}}
        try:
            if len(pred_box_robot) == len(task_grasp_phase_frame_indices):
                robot_det_dict = {}
                for i, frame_idx in enumerate(task_grasp_phase_frame_indices):
                    det = pred_box_robot[i]
                    if det is not None and len(det) > 0:
                        robot_det_dict[frame_idx] = {
                            "box": [round(coord) for coord in det.xyxy[0]],
                            "confidence": float(det.confidence[0]),
                        }
            else:
                robot_det_dict = {}
        except Exception:
            robot_det_dict = {}

        gripper_box = None
        gripper_box_confidence = None
        gripper_det = robot_det_dict.get(det_frame_idx)
        if gripper_det is not None and "box" in gripper_det:
            gripper_box = np.asarray(gripper_det["box"], dtype=np.float32)[None]
            gripper_box_confidence = float(gripper_det.get("confidence", 0.0))

        # Extract intermediate detections from pred_boxes[2:] (keyframes after detection frame)
        intermediate_detections = {}
        for kf_list_idx in range(2, len(task_grasp_phase_frame_indices)):
            kf_rel = task_grasp_phase_frame_indices[kf_list_idx]
            if kf_list_idx < len(pred_boxes):
                pb = pred_boxes[kf_list_idx]
                if pb is not None and len(pb.xyxy) > 0:
                    intermediate_detections[kf_rel] = [
                        {"box": [round(float(c)) for c in pb.xyxy[j]], "confidence": float(pb.confidence[j])}
                        for j in range(len(pb.xyxy))
                    ]

        # Normalize so all downstream code (SAM2, filtering, tracking fallback) uses pred_boxes[0].
        pred_boxes[0] = pred_boxes[1]

        return DetectionResult(
            pred_boxes=pred_boxes,
            candidate_boxes=candidate_boxes,
            robot_det_dict=robot_det_dict,
            robot_candidates_all=all_robot_candidates,
            is_tool_task=is_tool_task,
            tool_name=tool_name,
            tool_boxes_fullres=tool_boxes_fullres,
            object_box_for_tool_task=object_box_for_tool_task,
            object_box_for_tool_task_confidence=object_box_for_tool_task_confidence,
            pnp_task=pnp_task,
            all_boxes_np=all_boxes_np,
            detected_boxes_fullres=detected_boxes_fullres,
            arm=arm,
            intermediate_detections=intermediate_detections if intermediate_detections else None,
            keyframe_indices=list(task_grasp_phase_frame_indices),
            gripper_box=gripper_box,
            gripper_box_confidence=gripper_box_confidence,
        )

    def _detect_targets(self, frames_subset, task_obj_info, det_result, nms_fn):
        """Detect target boxes. Returns TargetResult."""
        target_boxes = []
        target_boxes_fullres = None
        pred_boxes_target_from_final_frame = None
        all_boxes_np = det_result.all_boxes_np

        pred_boxes_target_name_from_final_frame = np.empty((0, 4))
        pred_boxes_target_name_from_first_frame = np.empty((0, 4))
        text_labels = []

        if det_result.pnp_task:
            pred_boxes_target_from_final_frame = get_interacted_obj_boxes(
                self.detection_model, frames_subset[[-6]], task_obj_info,
                threshold=self.det_threshold,
                text_threshold=STANDARD_DETECTION_TEXT_THRESHOLD,
                nms_threshold=self.nms_threshold,
                skip_robot_detection=self.robotseg is not None,
            )
            all_boxes_np = np.concatenate([all_boxes_np, pred_boxes_target_from_final_frame[0].xyxy], axis=0)
            target_boxes_fullres = pred_boxes_target_from_final_frame[0].xyxy.copy()

            for i in range(len(pred_boxes_target_from_final_frame[0].xyxy)):
                text_labels.append(task_obj_info["object"])

        reset_target = False
        if task_obj_info["target_location"] is None or task_obj_info["target_location"].strip() == "":
            reset_target = True
            if det_result.is_tool_task:
                task_obj_info["target_location"] = task_obj_info.get("tool_name_required", task_obj_info["object"])
            else:
                task_obj_info["target_location"] = task_obj_info["object"]

        if task_obj_info["target_location"] is not None and task_obj_info["target_location"].strip() != "":
            det_frames = np.stack([frames_subset[0], frames_subset[-3]], axis=0)
            pred_boxes_target = get_interacted_obj_boxes(
                self.detection_model, det_frames, task_obj_info,
                threshold=self.det_threshold, use_target_location=True,
                text_threshold=STANDARD_DETECTION_TEXT_THRESHOLD,
                nms_threshold=self.nms_threshold,
                skip_robot_detection=self.robotseg is not None,
            )

            pred_boxes_target_name_from_first_frame = pred_boxes_target[0]
            for i in range(len(pred_boxes_target_name_from_first_frame.xyxy)):
                text_labels.append(task_obj_info["target_location"])

            pred_boxes_target_name_from_final_frame = pred_boxes_target[1]
            for i in range(len(pred_boxes_target_name_from_final_frame.xyxy)):
                text_labels.append(task_obj_info["target_location"])

            all_boxes_np = np.concatenate([all_boxes_np, pred_boxes_target_name_from_first_frame.xyxy], axis=0)
            all_boxes_np = np.concatenate([all_boxes_np, pred_boxes_target_name_from_final_frame.xyxy], axis=0)
            target_boxes_fullres = pred_boxes_target_name_from_final_frame.xyxy.copy()

        if det_result.pnp_task:
            if (pred_boxes_target_name_from_final_frame is not None and
                    isinstance(pred_boxes_target_name_from_final_frame, sv.Detections) and
                    len(pred_boxes_target_name_from_final_frame.xyxy) > 0):
                if reset_target:
                    pred_boxes_target_from_final_frame[0] = sv.Detections.merge(
                        [pred_boxes_target_from_final_frame[0], pred_boxes_target_name_from_final_frame])
                else:
                    pred_boxes_target_from_final_frame[0] = sv.Detections.merge(
                        [pred_boxes_target_from_final_frame[0], pred_boxes_target_name_from_final_frame,
                         pred_boxes_target_name_from_first_frame])
        else:
            if reset_target:
                pred_boxes_target_from_final_frame = [pred_boxes_target_name_from_final_frame]
            else:
                pred_boxes_target_from_final_frame = sv.Detections.merge(
                    [pred_boxes_target_name_from_final_frame, pred_boxes_target_name_from_first_frame])
                pred_boxes_target_from_final_frame = [pred_boxes_target_from_final_frame]

        # Filter target boxes by size
        threshold = DEFAULT_MAX_BOX_AREA_RATIO * frames_subset.shape[1] * frames_subset.shape[2]
        valid_box_indices = filter_boxes_by_size(pred_boxes_target_from_final_frame[0].xyxy, threshold)
        pred_boxes_target_from_final_frame[0] = pred_boxes_target_from_final_frame[0][valid_box_indices]

        # NMS
        nms_ind = nms(
            torch.tensor(pred_boxes_target_from_final_frame[0].xyxy),
            torch.tensor(pred_boxes_target_from_final_frame[0].confidence),
            0.4,
        ).cpu().numpy()
        pred_boxes_target_from_final_frame[0] = pred_boxes_target_from_final_frame[0][nms_ind]

        for det_idx in range(len(pred_boxes_target_from_final_frame[0].xyxy)):
            target_boxes.append({
                "box": [round(coord) for coord in pred_boxes_target_from_final_frame[0].xyxy[det_idx]],
                "det_confidence": float(pred_boxes_target_from_final_frame[0].confidence[det_idx]),
                "frame_index": -1,
                "label": text_labels.pop(0) if text_labels else "target",
            })

        if reset_target:
            task_obj_info["target_location"] = None

        # Update all_boxes_np in det_result
        det_result.all_boxes_np = all_boxes_np

        return TargetResult(
            target_boxes=target_boxes,
            target_boxes_fullres=target_boxes_fullres,
            pred_boxes_target_merged=pred_boxes_target_from_final_frame,
        )


    @staticmethod
    def _set_robotseg_gripper_detection(
        det_result,
        gripper_box,
        robotseg_result,
        det_frame_idx,
    ):
        """Replace detector gripper metadata with boxes derived from RobotSeg masks."""
        det_result.gripper_box = np.asarray(gripper_box, dtype=np.float32)
        det_result.gripper_box_confidence = 1.0
        det_result.robot_candidates_all = []

        robot_det_dict = {}
        if robotseg_result is not None:
            masks = np.asarray(robotseg_result.get("masks", []))
            frame_indices = np.asarray(
                robotseg_result.get("frame_indices", []), dtype=np.int32,
            ).reshape(-1)
            for mask, frame_idx in zip(masks, frame_indices):
                box = box_from_mask(np.asarray(mask) > 0.0)
                if box is not None:
                    robot_det_dict[int(frame_idx)] = {
                        "box": [round(float(coord)) for coord in box],
                        "confidence": 1.0,
                    }

        if det_frame_idx not in robot_det_dict:
            box = np.asarray(gripper_box, dtype=np.float32).reshape(-1, 4)[0]
            robot_det_dict[int(det_frame_idx)] = {
                "box": [round(float(coord)) for coord in box],
                "confidence": 1.0,
            }
        det_result.robot_det_dict = robot_det_dict

    def _restore_detector_gripper_fallback(
        self,
        frames_subset,
        task_obj_info,
        det_result,
        arm=None,
    ):
        """Run the detector gripper fallback when enabled RobotSeg fails."""
        keyframe_indices = list(det_result.keyframe_indices or [])
        if not keyframe_indices:
            return det_result

        detection_task_info = task_obj_info
        detection_kwargs = {}
        if det_result.is_tool_task:
            detection_task_info = {
                "object": det_result.tool_name,
                "start_location": None,
                "target_location": None,
                "action": task_obj_info.get("action"),
            }
        else:
            detection_kwargs["text_threshold"] = STANDARD_DETECTION_TEXT_THRESHOLD

        _, robot_detections, all_robot_candidates = get_interacted_obj_boxes(
            self.detection_model,
            frames_subset[keyframe_indices],
            detection_task_info,
            threshold=self.det_threshold,
            return_robot_det=True,
            return_all_robot_dets=True,
            arm=arm,
            nms_threshold=self.nms_threshold,
            **detection_kwargs,
        )

        det_result.robot_candidates_all = all_robot_candidates
        det_result.robot_det_dict = {}
        for frame_idx, detection in zip(keyframe_indices, robot_detections):
            if detection is not None and len(detection) > 0:
                det_result.robot_det_dict[frame_idx] = {
                    "box": [round(float(coord)) for coord in detection.xyxy[0]],
                    "confidence": float(detection.confidence[0]),
                }

        if arm is not None:
            self._filter_detections_by_arm(
                det_result,
                arm,
                frames_subset.shape[2],
                frames_subset.shape[1],
            )
        else:
            det_frame_idx = (
                keyframe_indices[OBJ_DET_IDX]
                if len(keyframe_indices) > OBJ_DET_IDX
                else keyframe_indices[0]
            )
            gripper_detection = det_result.robot_det_dict.get(det_frame_idx)
            if gripper_detection is not None:
                det_result.gripper_box = np.asarray(
                    gripper_detection["box"], dtype=np.float32,
                )[None]
                det_result.gripper_box_confidence = float(
                    gripper_detection.get("confidence", 0.0),
                )
            else:
                det_result.gripper_box = None
                det_result.gripper_box_confidence = None
        return det_result

    def _segment_gripper_with_robotseg(self, frames_subset, det_frame_idx, frame_indices=None):
        """Run RobotSeg on (subsampled) video to get per-frame gripper masks.

        Returns (gripper_box, gripper_mask_at_det, robotseg_result) or (None, None, None) on failure.
        gripper_box: (1, 4) float32 pixel coords derived from mask nearest to det_frame_idx
        gripper_mask_at_det: (H, W) uint8 binary mask at the frame nearest to det_frame_idx
        robotseg_result: dict with 'masks' (N, H, W) float32 and 'frame_indices' (N,) int
        """
        try:
            result = self.robotseg.segment_video(
                frames_subset, category="robot",
                max_frames=self.robotseg_max_frames, det_frame_idx=det_frame_idx,
                frame_indices=frame_indices,
            )
            masks = result['masks']           # (N, H, W)
            frame_indices = result['frame_indices']  # (N,) original frame positions

            # det_frame_idx is guaranteed to be in frame_indices
            det_pos = int(np.where(frame_indices == det_frame_idx)[0][0])
            gripper_mask = (masks[det_pos] > 0.0).astype(np.uint8)
            gripper_box_coords = box_from_mask(gripper_mask)
            if gripper_box_coords is None:
                logging.info("RobotSeg: no gripper found at det_frame_idx")
                return None, None, None
            logging.info("RobotSeg: gripper segmented, box=%s", gripper_box_coords)
            return np.array([gripper_box_coords], dtype=np.float32), gripper_mask, result
        except Exception as e:
            logging.warning(f"RobotSeg gripper segmentation failed: {e}")
            return None, None, None

    def _segment_with_sam2(self, frames_subset, det_result, det_frame_idx=0, robotseg_gripper_mask=None):
        """Run SAM2 segmentation on candidate boxes and optional gripper box."""
        candidate_boxes = det_result.candidate_boxes
        pred_boxes = det_result.pred_boxes

        if len(pred_boxes[0].xyxy) == 0:
            logging.info("Skipping because no boxes found")
            return None, None, None

        self.sam2_predictor.set_image(frames_subset[det_frame_idx])

        mask_boxes = np.array([detection["box"] for detection in candidate_boxes])
        has_gripper_box = det_result.gripper_box is not None and len(det_result.gripper_box) > 0
        # When RobotSeg already provides the gripper mask, skip asking SAM2 for the gripper
        use_sam2_gripper = has_gripper_box and robotseg_gripper_mask is None
        if use_sam2_gripper:
            mask_boxes = np.concatenate([mask_boxes, det_result.gripper_box.astype(np.float32)], axis=0)

        masks, scores, _ = self.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=mask_boxes,
            multimask_output=False,
            return_logits=False,
        )
        if mask_boxes.shape[0] == 1:
            masks = masks[None]
        masks = masks[:, 0]
        scores = np.atleast_2d(scores)
        scores = scores[:, 0]

        if robotseg_gripper_mask is not None:
            gripper_mask = robotseg_gripper_mask
        elif use_sam2_gripper:
            gripper_mask = masks[-1].astype(np.uint8)
            masks = masks[:-1]
            scores = scores[:-1]
            if gripper_mask.sum() == 0:
                gripper_mask = None
        else:
            gripper_mask = None

        valid_masks = np.sum(masks, axis=(1, 2)) > 0
        masks_aligned = np.zeros_like(masks, dtype=np.uint8)
        masks_aligned[valid_masks] = masks[valid_masks].astype(np.uint8)

        pred_boxes[0].xyxy = pred_boxes[0].xyxy[valid_masks]
        pred_boxes[0].confidence = pred_boxes[0].confidence[valid_masks]
        pred_boxes[0].class_id = pred_boxes[0].class_id[valid_masks]

        for i in range(len(candidate_boxes)):
            if not valid_masks[i]:
                candidate_boxes[i]["confidence"] = 0.0
            candidate_boxes[i]["mask"] = masks_aligned[i].tolist() if valid_masks[i] else None

        if len(pred_boxes[0].xyxy) == 0:
            logging.info("Skipping because no boxes found after SAM filtering")
            return None, None, None

        return masks_aligned, valid_masks, gripper_mask

    def _segment_targets_with_sam2(self, frames_subset, target_result):
        """Run SAM2 image segmentation on the final frame for each target candidate box.

        Stores resulting masks as target_result.target_masks (M, H, W) uint8.
        Returns target_result (modified in-place).
        """
        target_boxes = target_result.target_boxes
        if not target_boxes:
            return target_result
        try:
            self.sam2_predictor.set_image(frames_subset[-1])
            boxes = np.array([tb["box"] for tb in target_boxes], dtype=np.float32)
            masks, _, _ = self.sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=boxes,
                multimask_output=False,
                return_logits=False,
            )
            if boxes.shape[0] == 1:
                masks = masks[None]
            target_result.target_masks = masks[:, 0].astype(np.uint8)
        except Exception as e:
            logging.warning(f"SAM2 target segmentation failed: {e}")
        return target_result


    def _apply_tool_task_logic(self, final, det_result):
        """Reassign boxes for tool tasks. Returns updated FinalBoxes."""
        if not det_result.is_tool_task:
            return final

        # For tool tasks, the tracked entity IS the tool
        tool_box = final.initial_object_box.copy() if hasattr(final.initial_object_box, 'copy') else final.initial_object_box
        tool_box_confidence = final.initial_object_box_confidence
        tool_traces = final.obj_traces.copy() if final.obj_traces is not None else None

        # initial_object_box becomes the object being acted upon
        if det_result.object_box_for_tool_task is not None and len(det_result.object_box_for_tool_task) > 0:
            initial_box = det_result.object_box_for_tool_task
            initial_confidence = det_result.object_box_for_tool_task_confidence
        else:
            initial_box = np.empty((0, 4))
            initial_confidence = None

        # If no target but we have object, use object as fallback target
        target_box = final.target_object_box
        target_confidence = final.target_object_box_confidence
        if (target_box is None or (hasattr(target_box, 'shape') and target_box.shape[0] == 0)):
            if det_result.object_box_for_tool_task is not None and len(det_result.object_box_for_tool_task) > 0:
                target_box = det_result.object_box_for_tool_task
                target_confidence = det_result.object_box_for_tool_task_confidence

        return FinalBoxes(
            initial_object_box=initial_box,
            initial_object_box_confidence=initial_confidence,
            target_object_box=target_box,
            target_object_box_confidence=target_confidence,
            obj_traces=final.obj_traces,
            obj_traces_target=final.obj_traces_target,
            verified_target=final.verified_target,
            tool_box=tool_box,
            tool_box_confidence=tool_box_confidence,
            tool_traces=tool_traces,
            intermediate_boxes=final.intermediate_boxes,
            target_candidate_box_confidences=final.target_candidate_box_confidences,
            confidence_breakdown=final.confidence_breakdown,
        )


    def _filter_detections_by_arm(self, det_result, arm, img_width, img_height=None):
        """Filter gripper detections to the expected image side for a bimanual arm.

        Only filters robot/gripper detections (robot_det_dict), NOT object
        detections (pred_boxes / candidate_boxes).  Uses a 70% boundary so
        that the left arm keeps boxes whose x-centre is in the left 70% and
        vice-versa.

        For gripper selection, favors larger boxes (likely a better view of the
        gripper) up to a 0.2% image-area cap to avoid background false positives.
        """
        if arm not in ("left", "right"):
            return det_result

        boundary = img_width * 0.7  # 70% line
        img_area = img_width * (img_height or img_width)  # fallback if height unknown
        max_box_area = img_area * 0.2# 0.2% of image area cap

        # --- Filter robot_det_dict (one box per keyframe) ---
        # Re-select from robot_candidates_all (all per-keyframe detections)
        # using arm-side + size preference.  Falls back to filtering the
        # single-entry robot_det_dict if candidates aren't available.
        if det_result.robot_candidates_all is not None and len(det_result.robot_candidates_all) > 0:
            kf_indices = det_result.keyframe_indices or []
            new_robot_dict = {}
            for i, candidates in enumerate(det_result.robot_candidates_all):
                if candidates is None or len(candidates) == 0:
                    continue
                best = self._pick_gripper_box(candidates, arm, boundary, img_width, max_box_area)
                if best is not None and i < len(kf_indices):
                    new_robot_dict[kf_indices[i]] = best
            det_result.robot_det_dict = new_robot_dict
        else:
            # Fallback: filter existing single-box entries by x-centre
            filtered = {}
            for frame_idx, rdet in (det_result.robot_det_dict or {}).items():
                box = rdet["box"]
                x_center = (box[0] + box[2]) / 2.0
                in_region = (x_center < boundary) if arm == "left" else (x_center >= img_width - boundary)
                if in_region:
                    filtered[frame_idx] = rdet
            det_result.robot_det_dict = filtered

        # Re-derive gripper_box from the now-filtered robot_det_dict so the NPZ
        # reflects the correct arm-side box, not the pre-filter argmax pick.
        det_frame_idx = det_result.keyframe_indices[OBJ_DET_IDX] if det_result.keyframe_indices and len(det_result.keyframe_indices) > OBJ_DET_IDX else None
        gripper_det = det_result.robot_det_dict.get(det_frame_idx) if det_frame_idx is not None else None
        if gripper_det is not None and "box" in gripper_det:
            det_result.gripper_box = np.asarray(gripper_det["box"], dtype=np.float32)[None]
            det_result.gripper_box_confidence = float(gripper_det.get("confidence", 0.0))
        else:
            det_result.gripper_box = None
            det_result.gripper_box_confidence = None

        logging.info(
            f"_filter_detections_by_arm: arm={arm} kept {len(det_result.robot_det_dict)} gripper detections "
            f"(object detections unchanged)"
        )
        return det_result

    @staticmethod
    def _pick_gripper_box(candidates, arm, boundary, img_width, max_box_area):
        """Pick the best gripper box from candidates for a given arm side.

        Filters to the arm's 70% region, then scores each box as a weighted
        combination of detection confidence (0.7) and a non-linear size score
        (0.3).  The size score uses sqrt scaling so small boxes contribute
        very little while larger boxes score higher, capped at max_box_area.
        """
        boxes = candidates.xyxy  # (N, 4)
        confidences = candidates.confidence  # (N,)
        if len(boxes) == 0:
            return None

        x_centers = (boxes[:, 0] + boxes[:, 2]) / 2.0
        if arm == "left":
            side_mask = x_centers < boundary
        else:
            side_mask = x_centers >= (img_width - boundary)

        if not side_mask.any():
            return None

        side_boxes = boxes[side_mask]
        side_conf = confidences[side_mask]
        areas = (side_boxes[:, 2] - side_boxes[:, 0]) * (side_boxes[:, 3] - side_boxes[:, 1])

        # Non-linear size score: sqrt(area / cap), clamped to [0, 1].
        # Small boxes get a very small score, large boxes plateau at the cap.
        area_ratio = np.clip(areas / max_box_area, 0.0, 1.0)
        size_score = np.sqrt(area_ratio)

        combined_score = 0.7 * side_conf + 0.3 * size_score
        pick = int(np.argmax(combined_score))

        return {
            "box": [round(float(c)) for c in side_boxes[pick]],
            "confidence": float(side_conf[pick]),
        }

    # ------------------------------------------------------------------
    # Bimanual subtask annotation
    # ------------------------------------------------------------------

    def _annotate_bimanual_subtask(
        self,
        trajectory,
        left_subtask_idx,
        left_info,
        right_subtask_idx,
        right_info,
        task_obj_info_list,
        left_phases,
        right_phases,
        grasp_phases_with_offset,
        traj_len,
        end_frame_grasp_release,
    ):
        """Annotate a pair of simultaneously-acting subtasks (left + right arm).

        Returns a single annotation dict with is_bimanual=True and arm_annotations.
        """
        # --- Compute unified frame window ---
        start_l, end_l = self._compute_frame_boundaries(
            left_subtask_idx, left_info, task_obj_info_list,
            grasp_phases_with_offset, traj_len, end_frame_grasp_release,
            is_bimanual_arm=True,
        )
        start_r, end_r = self._compute_frame_boundaries(
            right_subtask_idx, right_info, task_obj_info_list,
            grasp_phases_with_offset, traj_len, end_frame_grasp_release,
            is_bimanual_arm=True,
        )
        start_frame = min(start_l, start_r)
        end_frame   = max(end_l, end_r)

        frames_subset = trajectory.frames[start_frame:end_frame]
        frames_subset, _global_scale = self._resize_frames_global(frames_subset)
        full_frames_subset = frames_subset
        if len(frames_subset) < 6:
            logging.info(
                f"Skipping bimanual pair ({left_subtask_idx},{right_subtask_idx}) of "
                f"{trajectory.name}: too few frames ({len(frames_subset)})"
            )
            return None

        frames_torch = frames_to_tensor(frames_subset).to(self.device)

        arm_annotations = []
        primary_final = None
        primary_det   = None
        primary_tracks_file = None
        primary_obj_traces_cotracker_point = None
        primary_obj_traces_cotracker_point_meta = None
        primary_det_frame_idx = 0
        right_det_frame_idx = 0

        for arm_label, arm_info, arm_idx, arm_phases in [
            ("left",  left_info,  left_subtask_idx,  left_phases),
            ("right", right_info, right_subtask_idx, right_phases),
        ]:
            # Build phase indices for detection frames
            arm_phase_annotations, _, task_grasp_phase_frame_indices = self._build_subtask_phases(
                arm_info, arm_phases,
                grasp_phases_with_offset, start_frame, end_frame, len(frames_subset),
            )
            phase_mask_keyframes = self._phase_mask_keyframes(
                arm_phase_annotations,
                start_frame,
                len(frames_subset),
            )

            # Detect objects with arm-specific prompts, then restrict to the expected image half
            det_result = self._detect_objects(frames_subset, arm_info, task_grasp_phase_frame_indices, arm=arm_label)
            self.debug_saver.save_robot_candidates(
                trajectory=trajectory,
                subtask_idx=arm_idx,
                frames_subset=frames_subset,
                keyframe_indices=task_grasp_phase_frame_indices,
                all_robot_candidates=det_result.robot_candidates_all,
                arm=arm_label,
            )
            det_result = self._filter_detections_by_arm(det_result, arm_label, frames_subset.shape[2], frames_subset.shape[1])

            # Detect targets
            target_result = self._detect_targets(frames_subset, arm_info, det_result, nms)

            # No cropping for bimanual (cropping is droid-specific and not expected here)
            crop_info = CropInfo(
                did_crop=False, crop_box=None, offsets=None,
                cropped_start_frame=None,
                cropped_detected_boxes=None, cropped_target_boxes=None,
            )

            det_frame_idx = task_grasp_phase_frame_indices[OBJ_DET_IDX]
            phase_mask_keyframes.insert(
                0,
                ("detection", det_frame_idx, start_frame + det_frame_idx),
            )
            frames_torch_tracking, _track_indices, _scaled_phases, _T, _tracking_scale, _tracking_actual_fps = self._subsample_frames_for_tracking(
                frames_torch, det_frame_idx, arm_info, start_frame,
                traj_name=trajectory.name, subtask_idx=f"{left_subtask_idx}_{right_subtask_idx}_{arm_label}",
            )
            tracking_frame_indices = (_track_indices + det_frame_idx).astype(np.int32)

            robotseg_video_masks = None
            robotseg_gripper_mask = None
            if self.robotseg is not None:
                robotseg_frame_indices = self._robotseg_frame_indices_for_tracking(
                    n_frames=len(frames_subset),
                    det_frame_idx=det_frame_idx,
                    tracking_frame_indices=tracking_frame_indices,
                    required_frame_indices=[item[1] for item in phase_mask_keyframes],
                )
                gripper_box_rs, robotseg_gripper_mask, robotseg_video_masks = self._segment_gripper_with_robotseg(
                    frames_subset,
                    det_frame_idx,
                    frame_indices=robotseg_frame_indices,
                )
                if gripper_box_rs is not None:
                    self._set_robotseg_gripper_detection(
                        det_result,
                        gripper_box_rs,
                        robotseg_video_masks,
                        det_frame_idx,
                    )
                else:
                    logging.info(
                        "RobotSeg produced no usable gripper mask for arm=%s; "
                        "running detector fallback",
                        arm_label,
                    )
                    det_result = self._restore_detector_gripper_fallback(
                        frames_subset,
                        arm_info,
                        det_result,
                        arm=arm_label,
                    )

            masks, valid_masks, gripper_mask = self._segment_with_sam2(
                frames_subset,
                det_result,
                det_frame_idx,
                robotseg_gripper_mask=robotseg_gripper_mask,
            )
            if masks is None:
                logging.info(
                    f"Bimanual arm={arm_label}: no SAM2 masks for {trajectory.name}, skipping arm"
                )
                arm_annotations.append(None)
                if arm_label == "left":
                    primary_final = None
                    primary_det   = det_result
                continue
            target_result = self._segment_targets_with_sam2(frames_subset, target_result)

            tracking_result = self._track_and_select(
                frames_torch_tracking, det_result, target_result, masks, valid_masks, gripper_mask,
                _scaled_phases, trajectory,
                intermediate_detections=det_result.intermediate_detections,
                det_frame_idx=det_frame_idx,
                frames_np=frames_subset[_track_indices + det_frame_idx] if len(_track_indices) > 0 else frames_subset[det_frame_idx:],
                tracking_scale=_tracking_scale,
                tracking_frame_indices=tracking_frame_indices,
            )

            tracking_result.raw_tracks_data = _augment_tracks_with_robotseg(
                _augment_tracks_with_tracking_frame_indices(
                    _augment_tracks_with_targets(
                        dict(tracking_result.raw_tracks_data) if tracking_result.raw_tracks_data else None,
                        target_result,
                    ),
                    tracking_frame_indices,
                ),
                robotseg_video_masks,
            )
            if tracking_result.raw_tracks_data is not None:
                tracking_result.raw_tracks_data["tracking_actual_fps"] = np.array(_tracking_actual_fps, dtype=np.float32)
                tracking_result.raw_tracks_data["source_fps"] = np.array(self.fps, dtype=np.float32)
            tracking_result = self._augment_confidence_breakdown_with_replay_signals(
                tracking_result,
                _scaled_phases,
                arm_info,
                arm=arm_label,
                scoring_fps=_tracking_actual_fps,
            )
            self._save_phase_keyframe_masks_to_arrays_blob(
                tracking_result,
                frames_subset,
                phase_mask_keyframes,
                robotseg_video_masks,
                full_frames_subset=full_frames_subset,
                intermediate_detections=det_result.intermediate_detections,
                crop_info=crop_info,
                global_scale=_global_scale,
            )

            tracks_file = self._save_tracks_npz(
                trajectory.name,
                f"{arm_idx}_{arm_label}",
                tracking_result.raw_tracks_data,
                crop_info=crop_info,
                global_scale=_global_scale,
                start_frame=start_frame,
            )

            # Resolve target
            tracking_result = self._resolve_target_box(tracking_result, target_result, frames_subset)

            # Restore coordinates
            final = self._restore_coordinates(
                tracking_result, crop_info,
                target_candidate_box_confidences=[tb["det_confidence"] for tb in target_result.target_boxes],
            )
            if _global_scale != (1.0, 1.0):
                final = self._unscale_final(final, _global_scale)
            final = self._apply_tool_task_logic(final, det_result)
            final = self._final_with_output_coordinate_confidence_breakdown(
                final, crop_info, global_scale=_global_scale,
            )

            # Build per-arm annotation entry
            (
                arm_confidence,
                arm_detection_confidence,
                arm_selection_score,
                arm_selection_score_name,
            ) = _resolve_emitted_initial_box_confidence(final)
            arm_traces = final.obj_traces
            if arm_traces is not None:
                arm_traces = arm_traces.mean(axis=1, keepdims=True)
            arm_point_track, arm_point_meta = self._compute_export_centroid_trace(
                final,
                tracking_result.raw_tracks_data,
            )
            arm_ann = {
                "arm": arm_label,
                "subtask_index": arm_idx,
                "object": arm_info.get("object"),
                "initial_object_box": final.initial_object_box.tolist() if final.initial_object_box is not None else [],
                "initial_object_box_confidence": arm_confidence,
                "initial_object_box_detection_confidence": arm_detection_confidence,
                "initial_object_box_selection_score": arm_selection_score,
                "initial_object_box_selection_score_name": arm_selection_score_name,
                "target_object_box": final.target_object_box.tolist() if final.target_object_box is not None else [],
                "target_object_box_confidence": final.target_object_box_confidence,
                "target_candidate_box_confidences": final.target_candidate_box_confidences,
                "obj_traces": arm_traces.tolist() if arm_traces is not None else None,
                "obj_traces_cotracker_point": arm_point_track.tolist() if arm_point_track is not None else None,
                "obj_traces_cotracker_point_meta": arm_point_meta,
                "robot_detections": det_result.robot_det_dict,
                "confidence_breakdown": final.confidence_breakdown,
                "tracks_file": tracks_file,
                "detection_frame_index": det_frame_idx,
            }
            arm_annotations.append(arm_ann)

            if arm_label == "left":
                primary_final = final
                primary_det   = det_result
                primary_tracks_file = tracks_file
                primary_obj_traces_cotracker_point = arm_point_track
                primary_obj_traces_cotracker_point_meta = arm_point_meta
                primary_det_frame_idx = det_frame_idx
            else:
                right_det_frame_idx = det_frame_idx

        del frames_torch
        torch.cuda.empty_cache()

        # Stop when neither arm produced an annotation
        if primary_final is None:
            # Still no valid arm
            if all(a is None for a in arm_annotations):
                return None

        # Build base annotation using left arm's primary result
        subtask_phase_annotations, subtask_phase_sequence, _ = self._build_subtask_phases(
            left_info, left_phases, grasp_phases_with_offset, start_frame, end_frame, len(frames_subset),
        )

        # Use left arm final boxes for top-level fields (backward compat); fall back to right
        left_has_box = (
            primary_final is not None
            and primary_final.initial_object_box is not None
            and primary_final.initial_object_box.shape[0] > 0
        )
        if left_has_box:
            top_final = primary_final
            top_det   = primary_det
            top_info  = left_info
        else:
            # Right arm only
            right_ann_data = arm_annotations[1]
            top_final = FinalBoxes(
                initial_object_box=np.array(right_ann_data["initial_object_box"]),
                initial_object_box_confidence=right_ann_data[
                    "initial_object_box_detection_confidence"
                ],
                target_object_box=np.array(right_ann_data["target_object_box"]),
                target_object_box_confidence=right_ann_data["target_object_box_confidence"],
                obj_traces=None, obj_traces_target=None, verified_target=False,
                tool_box=None, tool_box_confidence=None, tool_traces=None,
                target_candidate_box_confidences=right_ann_data.get("target_candidate_box_confidences"),
                confidence_breakdown=right_ann_data.get("confidence_breakdown"),
            )
            top_det  = primary_det
            top_info = right_info
            primary_tracks_file = right_ann_data.get("tracks_file")
            primary_obj_traces_cotracker_point = right_ann_data.get("obj_traces_cotracker_point")
            primary_obj_traces_cotracker_point_meta = right_ann_data.get("obj_traces_cotracker_point_meta")
            primary_det_frame_idx = right_det_frame_idx

        top_final = self._run_cross_model_verification(
            top_final, frames_subset, start_frame, end_frame, top_info, trajectory,
        )

        primary_tool_traces_cotracker_point = (
            primary_obj_traces_cotracker_point.copy()
            if (top_det is not None and top_det.is_tool_task and primary_obj_traces_cotracker_point is not None)
            else None
        )

        annotation = self._build_annotation_dict(
            trajectory, left_subtask_idx, start_frame, end_frame,
            top_final, top_det, top_info, task_obj_info_list,
            left_phases + right_phases,
            subtask_phase_annotations, subtask_phase_sequence,
            tracks_file=primary_tracks_file,
            det_frame_idx=primary_det_frame_idx,
            obj_traces_cotracker_point=primary_obj_traces_cotracker_point,
            obj_traces_cotracker_point_meta=primary_obj_traces_cotracker_point_meta,
            tool_traces_cotracker_point=primary_tool_traces_cotracker_point,
            crop_info=crop_info,
            global_scale=_global_scale,
        )

        # Overwrite bimanual fields
        annotation["is_bimanual"] = True
        annotation["arm_annotations"] = [a for a in arm_annotations if a is not None]

        return build_publication_annotation(
            annotation,
            original_image_size_hw=tuple(map(int, trajectory.frames[0].shape[:2])),
            embed_arrays=False,
            save_pointclouds=self.save_pointclouds,
        )
