"""Annotation export for trajectory annotation."""

import logging
import numpy as np
from sparc.scoring.selection_scoring import (
    compute_centroid_traces,
    decode_arrays_blob,
    encode_arrays_blob,
)
from sparc.pipeline.annotation_coordinates import (
    _extract_winner_visibility_from_raw_tracks,
)
from sparc.pipeline.annotation_scoring import _resolve_emitted_initial_box_confidence


def _select_export_centroid_trace(winner_tracks, winner_visibility):
    """Return the winner's visible-point centroid using the detector baseline's reducer.

    Returns a (T, 1, 2) track and metadata, or (None, None) when unavailable.
    """
    if winner_tracks is None:
        return None, None

    tracks = np.asarray(winner_tracks, dtype=np.float64)
    if tracks.ndim != 3 or tracks.shape[0] == 0 or tracks.shape[1] == 0 or tracks.shape[2] < 2:
        return None, None

    T, n_points = tracks.shape[:2]

    visibility = None
    if winner_visibility is not None:
        try:
            visibility = np.asarray(winner_visibility, dtype=np.float64)
        except Exception:
            visibility = None
        if visibility is not None and visibility.shape != (T, n_points):
            visibility = None

    # compute_centroid_traces gates on ``vis > 0``. When point-track visibility is
    # unavailable -- or present but degenerate (all-zero), which has been observed
    # in practice -- fall back to treating every point as visible so the result is
    # the finite-point centroid (the helper independently drops non-finite points).
    # Without this the strict gate yields an all-NaN centroid and the trajectory is
    # dropped outright, which is far worse than a slightly noisier trace. Which
    # path was taken is recorded in meta so it stays auditable.
    visibility_used = visibility is not None
    if visibility is None:
        visibility = np.ones((T, n_points), dtype=np.float64)

    centroid = compute_centroid_traces(tracks[None, ...], visibility[None, ...])[0]  # (T, 2)
    valid_frames = np.flatnonzero(np.all(np.isfinite(centroid), axis=-1))

    if valid_frames.size == 0 and visibility_used:
        visibility = np.ones((T, n_points), dtype=np.float64)
        visibility_used = False
        centroid = compute_centroid_traces(tracks[None, ...], visibility[None, ...])[0]
        valid_frames = np.flatnonzero(np.all(np.isfinite(centroid), axis=-1))

    if valid_frames.size == 0:
        return None, None

    counts = ((visibility > 0) & np.all(np.isfinite(tracks[..., :2]), axis=-1)).sum(axis=1)
    meta = {
        "method": "visible_point_centroid",
        "n_points_averaged_mean": round(float(counts[counts > 0].mean()), 3) if (counts > 0).any() else 0.0,
        "n_points_available": int(n_points),
        "frame_coverage_ratio": round(float(valid_frames.size / max(T, 1)), 6),
        "first_valid_frame": int(valid_frames[0]),
        "last_valid_frame": int(valid_frames[-1]),
        "visibility_used": bool(visibility_used),
    }

    return centroid[:, None, :].copy(), meta


class AnnotationExport:
    """AnnotationExport methods operating on the configured annotator state."""

    def _save_tracks_npz(self, traj_name, subtask_idx, raw_tracks_data, crop_info=None,
                         global_scale=(1.0, 1.0), start_frame=0):
        """Save per-candidate track arrays to a compressed numpy file.

        Returns the absolute path on success, None if tracks_dir is unset or on error.
        Arrays are keyed by type: 'cotracker_tracks', 'cotracker_visibility',
        'winner_idx', 'candidate_boxes', 'candidate_confidences',
        'target_candidate_boxes', 'target_candidate_confidences'.

        Temporal index arrays ('tracking_frame_indices', 'gripper_video_masks_robotseg_frame_indices')
        are in frames_subset space (0 = trajectory frame start_frame).  Add 'start_frame' (stored
        here as a scalar) to convert any of those indices to absolute trajectory frame numbers.

        When cropping or global resize was applied, spatial arrays are restored into
        original full-frame coordinates before saving.
        """
        import re
        if self.tracks_dir is None or not raw_tracks_data:
            return None
        try:
            import os
            os.makedirs(self.tracks_dir, exist_ok=True)
            safe_name = re.sub(r'[/\s\\]+', '_', str(traj_name)).strip('_')
            path = os.path.join(self.tracks_dir, f"{safe_name}_{subtask_idx}.npz")
            # Exclude in-memory segmentation candidates from raw track exports.
            _mask_keys = {'candidate_masks', 'target_candidate_masks'}
            if not getattr(self, "save_pointclouds", False):
                _mask_keys.update({'cotracker_tracks_3d', 'gripper_tracks_3d'})
            raw_tracks_data = {k: v for k, v in raw_tracks_data.items() if k not in _mask_keys}
            raw_tracks_data = self._restore_raw_tracks_data_coordinates(raw_tracks_data, crop_info)
            if global_scale != (1.0, 1.0):
                raw_tracks_data = self._unscale_raw_tracks(raw_tracks_data, global_scale)
            serialized_tracks_data = {"start_frame": np.array(start_frame, dtype=np.int32)}
            for key, value in raw_tracks_data.items():
                if value is None:
                    continue
                array_value = np.asarray(value)
                if np.issubdtype(array_value.dtype, np.floating):
                    serialized_tracks_data[key] = array_value.astype(np.float16)
                else:
                    serialized_tracks_data[key] = value
            np.savez_compressed(path, **serialized_tracks_data)
            return path
        except Exception as e:
            logging.warning(f"Failed to save tracks npz for {traj_name}/{subtask_idx}: {e}")
            return None


    @staticmethod
    def _phase_mask_keyframes(subtask_phase_annotations, start_frame, n_frames):
        """Return named phase-end keyframes in frames_subset coordinates."""
        keyframes = []
        for phase in subtask_phase_annotations or []:
            phase_name = str(phase.get("phase", "")).strip().lower()
            if phase_name not in {"grasp", "interact", "release"}:
                continue
            frame_abs = int(phase.get("end_frame", start_frame))
            frame_rel = max(0, min(frame_abs - int(start_frame), int(n_frames) - 1))
            keyframes.append((phase_name, frame_rel, frame_rel + int(start_frame)))
        return keyframes


    def _save_phase_keyframe_masks_to_arrays_blob(
        self,
        tracking_result,
        frames_subset,
        phase_keyframes,
        robotseg_result,
        full_frames_subset=None,
        intermediate_detections=None,
        crop_info=None,
        global_scale=(1.0, 1.0),
    ):
        """Store winning-object masks, RobotSeg masks, and masked 3D clouds.

        Mask frame indices are saved both relative to ``frames_subset`` and as
        absolute trajectory indices. The arrays are merged into
        ``confidence_breakdown.lightweight.arrays_blob``.
        """
        if (
            tracking_result.best_obj_box is None
            or not phase_keyframes
        ):
            return
        if tracking_result.confidence_breakdown is None:
            tracking_result.confidence_breakdown = {}

        phase_names = np.asarray([item[0] for item in phase_keyframes], dtype="U16")
        phase_rel = np.asarray([item[1] for item in phase_keyframes], dtype=np.int32)
        phase_abs = np.asarray([item[2] for item in phase_keyframes], dtype=np.int32)
        arrays = {}

        # Move the selected initial box with its AllTracker points, refine it
        # against detections at each phase keyframe, and segment the result
        # with the SAM2 image model.
        raw_tracks = tracking_result.raw_tracks_data or {}
        cotracker_tracks = raw_tracks.get("cotracker_tracks")
        cotracker_visibility = raw_tracks.get("cotracker_visibility")
        tracking_frames = raw_tracks.get("tracking_frame_indices")
        winner_idx = int(raw_tracks.get("winner_idx", -1))
        initial_box = np.asarray(
            tracking_result.best_obj_box, dtype=np.float32,
        ).reshape(-1, 4)[0]
        winner_masks_by_frame = {}

        def _box_iou(box_a, box_b):
            x1 = max(float(box_a[0]), float(box_b[0]))
            y1 = max(float(box_a[1]), float(box_b[1]))
            x2 = min(float(box_a[2]), float(box_b[2]))
            y2 = min(float(box_a[3]), float(box_b[3]))
            intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            area_a = max(0.0, float(box_a[2] - box_a[0])) * max(
                0.0, float(box_a[3] - box_a[1]),
            )
            area_b = max(0.0, float(box_b[2] - box_b[0])) * max(
                0.0, float(box_b[3] - box_b[1]),
            )
            union = area_a + area_b - intersection
            return intersection / union if union > 0.0 else 0.0

        for frame_rel in np.unique(phase_rel):
            prompt_box = initial_box.copy()
            if (
                cotracker_tracks is not None
                and cotracker_visibility is not None
                and tracking_frames is not None
                and 0 <= winner_idx < len(cotracker_tracks)
            ):
                candidate_tracks = np.asarray(
                    cotracker_tracks[winner_idx], dtype=np.float32,
                )
                candidate_visibility = np.asarray(
                    cotracker_visibility[winner_idx],
                )
                tracking_frames_arr = np.asarray(
                    tracking_frames, dtype=np.int32,
                ).reshape(-1)
                usable_t = min(
                    len(candidate_tracks),
                    len(candidate_visibility),
                    len(tracking_frames_arr),
                )
                if usable_t > 0:
                    track_t = int(np.argmin(
                        np.abs(tracking_frames_arr[:usable_t] - int(frame_rel)),
                    ))
                    visible = (
                        (candidate_visibility[0] >= 0.5)
                        & (candidate_visibility[track_t] >= 0.5)
                        & np.all(np.isfinite(candidate_tracks[0]), axis=1)
                        & np.all(np.isfinite(candidate_tracks[track_t]), axis=1)
                    )
                    if np.any(visible):
                        delta = np.median(
                            candidate_tracks[track_t, visible]
                            - candidate_tracks[0, visible],
                            axis=0,
                        )
                        prompt_box += np.array(
                            [delta[0], delta[1], delta[0], delta[1]],
                            dtype=np.float32,
                        )

            detections = (intermediate_detections or {}).get(
                int(frame_rel), [],
            )
            if detections:
                best_detection = max(
                    detections,
                    key=lambda det: (
                        _box_iou(prompt_box, det["box"])
                        * float(det.get("confidence", 0.0))
                    ),
                )
                if _box_iou(prompt_box, best_detection["box"]) > 0.0:
                    prompt_box = np.asarray(
                        best_detection["box"], dtype=np.float32,
                    )

            height, width = frames_subset.shape[1:3]
            prompt_box[[0, 2]] = np.clip(
                prompt_box[[0, 2]], 0, width - 1,
            )
            prompt_box[[1, 3]] = np.clip(
                prompt_box[[1, 3]], 0, height - 1,
            )
            try:
                self.sam2_predictor.set_image(frames_subset[int(frame_rel)])
                masks, _, _ = self.sam2_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=prompt_box[None],
                    multimask_output=False,
                    return_logits=False,
                )
                mask = np.asarray(masks)
                while mask.ndim > 2:
                    mask = mask[0]
                if mask.ndim == 2 and np.any(mask):
                    winner_masks_by_frame[int(frame_rel)] = mask.astype(
                        np.uint8,
                    )
            except Exception as exc:
                logging.warning(
                    "SAM2 keyframe mask failed at frame %s: %s",
                    int(frame_rel),
                    exc,
                )
        if winner_masks_by_frame:
            keep = [
                idx for idx, frame_rel in enumerate(phase_rel)
                if int(frame_rel) in winner_masks_by_frame
            ]
            if keep:
                arrays["winner_object_phase_keyframe_masks"] = np.stack([
                    np.asarray(winner_masks_by_frame[int(phase_rel[idx])], dtype=np.uint8)
                    for idx in keep
                ])
                arrays["winner_object_phase_keyframe_frame_indices"] = phase_rel[keep]
                arrays["winner_object_phase_keyframe_absolute_frame_indices"] = phase_abs[keep]
                arrays["winner_object_phase_keyframe_phase_names"] = phase_names[keep]

        if robotseg_result is not None:
            robot_masks = np.asarray(robotseg_result.get("masks"))
            robot_frames = np.asarray(
                robotseg_result.get("frame_indices"), dtype=np.int32
            ).reshape(-1)
            robot_by_frame = {
                int(frame_rel): (robot_masks[idx] > 0).astype(np.uint8)
                for idx, frame_rel in enumerate(robot_frames)
                if idx < len(robot_masks)
            }
            keep = [
                idx for idx, frame_rel in enumerate(phase_rel)
                if int(frame_rel) in robot_by_frame
            ]
            if keep:
                arrays["robotseg_phase_keyframe_masks"] = np.stack([
                    robot_by_frame[int(phase_rel[idx])] for idx in keep
                ])
                arrays["robotseg_phase_keyframe_frame_indices"] = phase_rel[keep]
                arrays["robotseg_phase_keyframe_absolute_frame_indices"] = phase_abs[keep]
                arrays["robotseg_phase_keyframe_phase_names"] = phase_names[keep]

        # MoGe's dense point maps are otherwise discarded after track lifting.
        # Preserve only valid points inside the publishable object/robot masks at
        # grasp, interact, and release keyframes. Ragged clouds are represented
        # by concatenated XYZ/UV arrays plus offsets, keeping the sidecar compact.
        point_map_size_hw = None
        if getattr(self, "save_pointclouds", False) and self.pointcloud is not None and arrays:
            try:
                cloud_frames = sorted({
                    int(frame_rel)
                    for prefix in ("winner_object", "robotseg")
                    for frame_rel in np.asarray(
                        arrays.get(f"{prefix}_phase_keyframe_frame_indices", []),
                        dtype=np.int32,
                    ).reshape(-1)
                })
                if cloud_frames:
                    pointcloud_frames = (
                        full_frames_subset
                        if full_frames_subset is not None
                        else frames_subset
                    )
                    height, width = pointcloud_frames.shape[1:3]
                    pc_height, pc_width = height, width
                    scale = 1.0
                    if (
                        self.pointcloud_max_resolution is not None
                        and max(height, width) > self.pointcloud_max_resolution
                    ):
                        import cv2

                        scale = self.pointcloud_max_resolution / max(height, width)
                        pc_height = max(1, round(height * scale))
                        pc_width = max(1, round(width * scale))
                        cloud_input = np.stack([
                            cv2.resize(
                                pointcloud_frames[frame_rel],
                                (pc_width, pc_height),
                                interpolation=cv2.INTER_AREA,
                            )
                            for frame_rel in cloud_frames
                        ])
                    else:
                        cloud_input = pointcloud_frames[cloud_frames]
                    point_map_size_hw = [int(pc_height), int(pc_width)]

                    pointcloud = self.pointcloud.get_pointcloud_batch(
                        cloud_input,
                        fov_x=self.fov_x,
                        resolution_level=self.pointcloud_resolution_level,
                    )
                    points_by_frame = {
                        frame_rel: (
                            np.asarray(pointcloud["points"][idx], dtype=np.float32),
                            np.asarray(pointcloud["mask"][idx], dtype=bool),
                        )
                        for idx, frame_rel in enumerate(cloud_frames)
                    }

                    global_sx, global_sy = map(float, global_scale)

                    def mask_in_full_processing_frame(mask):
                        mask = (np.asarray(mask) > 0).astype(np.uint8)
                        if mask.shape == (height, width):
                            return mask
                        if (
                            crop_info is not None
                            and crop_info.did_crop
                            and crop_info.crop_box is not None
                        ):
                            x1, y1, x2, y2 = map(int, crop_info.crop_box)
                            canvas = np.zeros((height, width), dtype=np.uint8)
                            paste_height = min(y2 - y1, mask.shape[0], height - y1)
                            paste_width = min(x2 - x1, mask.shape[1], width - x1)
                            if paste_height > 0 and paste_width > 0:
                                canvas[y1:y1 + paste_height, x1:x1 + paste_width] = mask[
                                    :paste_height, :paste_width
                                ]
                            return canvas
                        import cv2

                        return cv2.resize(
                            mask, (width, height), interpolation=cv2.INTER_NEAREST,
                        )

                    def add_masked_cloud(prefix):
                        masks = arrays.get(f"{prefix}_phase_keyframe_masks")
                        rel_frames = arrays.get(
                            f"{prefix}_phase_keyframe_frame_indices"
                        )
                        abs_frames = arrays.get(
                            f"{prefix}_phase_keyframe_absolute_frame_indices"
                        )
                        names = arrays.get(f"{prefix}_phase_keyframe_phase_names")
                        if masks is None or rel_frames is None:
                            return

                        xyz_parts = []
                        uv_parts = []
                        offsets = [0]
                        for mask, frame_rel in zip(masks, rel_frames):
                            points, valid = points_by_frame[int(frame_rel)]
                            mask = mask_in_full_processing_frame(mask)
                            if mask.shape != points.shape[:2]:
                                import cv2

                                mask = cv2.resize(
                                    mask,
                                    (points.shape[1], points.shape[0]),
                                    interpolation=cv2.INTER_NEAREST,
                                )
                            keep_pixels = (mask > 0) & valid & np.all(
                                np.isfinite(points), axis=-1,
                            )
                            v, u = np.nonzero(keep_pixels)
                            xyz = points[v, u]

                            # Convert point-map pixels back to original,
                            # uncropped, unresized camera-image coordinates.
                            frame_u = (u.astype(np.float32) + 0.5) / scale - 0.5
                            frame_v = (v.astype(np.float32) + 0.5) / scale - 0.5
                            original_u = np.rint(frame_u / global_sx).astype(np.int32)
                            original_v = np.rint(frame_v / global_sy).astype(np.int32)
                            uv = np.stack([original_u, original_v], axis=-1)

                            xyz_parts.append(xyz.astype(np.float32, copy=False))
                            uv_parts.append(uv.astype(np.int32, copy=False))
                            offsets.append(offsets[-1] + len(xyz))

                        arrays[f"{prefix}_phase_keyframe_points_xyz"] = (
                            np.concatenate(xyz_parts, axis=0)
                            if xyz_parts else np.empty((0, 3), dtype=np.float32)
                        )
                        arrays[f"{prefix}_phase_keyframe_points_uv"] = (
                            np.concatenate(uv_parts, axis=0)
                            if uv_parts else np.empty((0, 2), dtype=np.int32)
                        )
                        arrays[f"{prefix}_phase_keyframe_point_offsets"] = np.asarray(
                            offsets, dtype=np.int64,
                        )
                        arrays[f"{prefix}_phase_keyframe_point_absolute_frame_indices"] = np.asarray(
                            abs_frames, dtype=np.int32,
                        )
                        arrays[f"{prefix}_phase_keyframe_point_phase_names"] = np.asarray(
                            names, dtype="U16",
                        )

                    add_masked_cloud("winner_object")
                    add_masked_cloud("robotseg")
            except Exception as exc:
                logging.warning("MoGe keyframe point-cloud extraction failed: %s", exc)

        if not arrays:
            return
        lightweight = tracking_result.confidence_breakdown.setdefault("lightweight", {})
        blob_arrays = decode_arrays_blob(lightweight.get("arrays_blob"))
        blob_arrays.update(arrays)
        lightweight["arrays_blob"] = encode_arrays_blob(blob_arrays)
        lightweight["arrays_blob_keys"] = sorted(blob_arrays)
        lightweight["arrays_blob_encoding"] = "base64_encoded_np_savez_compressed"
        lightweight["arrays_blob_schema_version"] = 1
        crop_box = (
            list(map(int, crop_info.crop_box))
            if crop_info is not None and crop_info.did_crop and crop_info.crop_box is not None
            else None
        )
        lightweight["phase_keyframe_mask_metadata"] = {
            "keyframe_rule": (
                "detection_frame_and_end_frame_of_each_grasp_interact_release_phase"
            ),
            "mask_array_shape": "[num_keyframes, image_height, image_width]",
            "mask_dtype": "uint8",
            "mask_values": {"background": 0, "foreground": 1},
            "relative_frame_indices": "zero_based_offset_from_annotation_start_frame",
            "absolute_frame_indices": "zero_based_trajectory_frame_index",
            "phase_names_align_with_first_mask_dimension": True,
            "spatial_coordinate_frame": "segmentation_processing_frame_after_global_resize_and_optional_crop",
            "processing_frame_size_hw": [
                int(frames_subset.shape[1]),
                int(frames_subset.shape[2]),
            ],
            "transform_mask_pixels_to_original_frame": {
                "formula": "original_xy=(mask_xy+crop_xy)/global_scale_xy",
                "crop_xyxy_in_globally_resized_frame": crop_box,
                "crop_origin_xy": crop_box[:2] if crop_box is not None else [0, 0],
                "global_scale_xy": [float(global_scale[0]), float(global_scale[1])],
            },
            "winning_object": {
                "source": "SAM2_image_prompted_by_final_scoring_winner_at_each_keyframe",
                "masks_key": "winner_object_phase_keyframe_masks",
                "relative_frame_indices_key": "winner_object_phase_keyframe_frame_indices",
                "absolute_frame_indices_key": "winner_object_phase_keyframe_absolute_frame_indices",
                "phase_names_key": "winner_object_phase_keyframe_phase_names",
            },
            "robot": {
                "source": "RobotSeg_category_robot",
                "masks_key": "robotseg_phase_keyframe_masks",
                "relative_frame_indices_key": "robotseg_phase_keyframe_frame_indices",
                "absolute_frame_indices_key": "robotseg_phase_keyframe_absolute_frame_indices",
                "phase_names_key": "robotseg_phase_keyframe_phase_names",
            },
        }
        if any(key.endswith("_phase_keyframe_points_xyz") for key in arrays):
            lightweight["phase_keyframe_pointcloud_metadata"] = {
                "keyframe_rule": (
                    "detection_frame_and_end_frame_of_each_grasp_interact_release_phase"
                ),
                "xyz_dtype": "float32",
                "xyz_coordinate_frame": "MoGe_camera_coordinates",
                "uv_dtype": "int32",
                "uv_coordinate_frame": "original_uncropped_unresized_image",
                "ragged_encoding": "concatenated_points_with_offsets",
                "moge_resolution_level": int(self.pointcloud_resolution_level),
                "moge_max_resolution": self.pointcloud_max_resolution,
                "moge_processing_size_hw": list(map(int, (
                    full_frames_subset.shape[1:3]
                    if full_frames_subset is not None else frames_subset.shape[1:3]
                ))),
                "moge_point_map_size_hw": point_map_size_hw,
                "moge_uses_full_uncropped_frame": True,
                "moge_model": "Ruicheng/moge-2-vitb-normal",
                "fov_x_degrees": self.fov_x,
                "object_mask_source": "SAM2_image_final_scoring_winner",
                "robot_mask_source": "RobotSeg_category_robot",
            }


    def _save_debug(self, trajectory, subtask_idx, start_frame, end_frame, det_frame_idx,
                    final, det_result, target_result, crop_info, tracking_result, ib_tidx,
                    annotation, frames_subset):
        """Assemble per-keyframe visualization data and call DebugImageSaver."""
        total_frames = len(trajectory.frames)
        if total_frames > 0:
            debug_start_idx = min(start_frame, max(total_frames - 1, 0))
            debug_end_idx = min(max(end_frame - 1, 0), max(total_frames - 1, 0))
            debug_det_idx = min(start_frame + det_frame_idx, max(total_frames - 1, 0))
            debug_start_frame = trajectory.frames[debug_start_idx]
            debug_end_frame = trajectory.frames[debug_end_idx]
            debug_det_frame = trajectory.frames[debug_det_idx]
        else:
            debug_start_frame = debug_end_frame = debug_det_frame = None

        def _flat_box(box):
            if box is None:
                return None
            arr = np.asarray(box)
            return arr.reshape(-1, 4)[0].tolist() if arr.size > 0 else None

        keyframe_viz = {}

        if debug_start_frame is not None:
            keyframe_viz[0] = {
                "frame": debug_start_frame, "frame_idx": 0,
                "object_box": _flat_box(final.initial_object_box),
                "object_label": "obj",
                "robot_box": det_result.robot_det_dict.get(0),
            }

        for kf_rel, tracked_box in (tracking_result.intermediate_boxes or {}).items():
            if kf_rel == 0:
                continue
            abs_idx = min(start_frame + kf_rel, max(total_frames - 1, 0))
            if abs_idx < total_frames:
                track_pts = None
                if final.obj_traces is not None:
                    t = ib_tidx.get(kf_rel, kf_rel - det_frame_idx)
                    if 0 <= t < len(final.obj_traces):
                        track_pts = final.obj_traces[t]
                display_box = tracked_box
                if kf_rel in keyframe_viz:
                    if keyframe_viz[kf_rel]["object_box"] is None:
                        keyframe_viz[kf_rel]["object_box"] = display_box
                    if keyframe_viz[kf_rel].get("track_points") is None:
                        keyframe_viz[kf_rel]["track_points"] = track_pts
                else:
                    keyframe_viz[kf_rel] = {
                        "frame": trajectory.frames[abs_idx], "frame_idx": kf_rel,
                        "object_box": display_box,
                        "object_label": "obj",
                        "robot_box": det_result.robot_det_dict.get(kf_rel),
                        "track_points": track_pts,
                    }

        end_rel = max(debug_end_idx - start_frame, 0) if debug_end_frame is not None else 0
        if debug_end_frame is not None and end_rel not in keyframe_viz:
            keyframe_viz[end_rel] = {
                "frame": debug_end_frame, "frame_idx": end_rel,
                "object_box": _flat_box(final.target_object_box),
                "object_label": "target",
                "robot_box": det_result.robot_det_dict.get(end_rel),
            }
        elif debug_end_frame is not None:
            existing = keyframe_viz[end_rel]
            if existing["object_box"] is None:
                existing["object_box"] = _flat_box(final.target_object_box)
                existing["object_label"] = "target"

        for kf_rel, rdet in det_result.robot_det_dict.items():
            if kf_rel not in keyframe_viz:
                abs_idx = min(start_frame + kf_rel, max(total_frames - 1, 0))
                if abs_idx < total_frames:
                    keyframe_viz[kf_rel] = {
                        "frame": trajectory.frames[abs_idx], "frame_idx": kf_rel,
                        "object_box": None, "object_label": "",
                        "robot_box": rdet,
                    }

        intermediate_debug = [keyframe_viz[k] for k in sorted(keyframe_viz)]

        self.debug_saver.maybe_save(
            trajectory=trajectory,
            subtask_idx=subtask_idx,
            start_frame=debug_start_frame,
            end_frame=debug_end_frame,
            det_frame=debug_det_frame,
            initial_box=final.initial_object_box,
            target_box=final.target_object_box,
            annotation=annotation,
            detected_boxes_fullres=det_result.detected_boxes_fullres,
            target_boxes_fullres=target_result.target_boxes_fullres,
            crop_box=crop_info.crop_box,
            cropped_start_frame=crop_info.cropped_start_frame,
            cropped_detected_boxes=crop_info.cropped_detected_boxes,
            cropped_target_boxes=crop_info.cropped_target_boxes,
            tool_box=final.tool_box if det_result.is_tool_task else None,
            tool_boxes_fullres=det_result.tool_boxes_fullres if det_result.is_tool_task else None,
            robot_detections=det_result.robot_det_dict,
            obj_traces=final.obj_traces,
            intermediate_debug=intermediate_debug if intermediate_debug else None,
            fps=self.fps,
        )

        self.debug_saver.save_sam2_debug(
            trajectory=trajectory,
            subtask_idx=subtask_idx,
            annotation=annotation,
            frames_subset=frames_subset,
            det_frame_idx=det_frame_idx,
            sam2_det_mask=tracking_result.best_obj_mask,
        )


    def _compute_export_centroid_trace(self, final, raw_tracks_data):
        """Compute the winning candidate's visible-point centroid for JSON export."""
        if final is None or final.obj_traces is None:
            return None, None

        winner_visibility = _extract_winner_visibility_from_raw_tracks(
            raw_tracks_data,
            final.obj_traces,
        )
        track, metadata = _select_export_centroid_trace(
            final.obj_traces,
            winner_visibility,
        )
        tracking_indices = (
            raw_tracks_data.get("tracking_frame_indices")
            if isinstance(raw_tracks_data, dict)
            else None
        )
        if track is not None and metadata is not None and tracking_indices is not None:
            tracking_indices = np.asarray(tracking_indices, dtype=np.int32).reshape(-1)
            if len(tracking_indices) == len(track):
                metadata["frame_indices_relative_to_window"] = tracking_indices.tolist()
        return track, metadata


    def _build_annotation_dict(self, trajectory, subtask_idx, start_frame, end_frame,
                               final, det_result, task_obj_info, task_obj_info_list,
                               grasp_phases, subtask_phase_annotations, subtask_phase_sequence,
                               tracks_file=None, det_frame_idx=0,
                               obj_traces_cotracker_point=None,
                               obj_traces_cotracker_point_meta=None,
                               tool_traces_cotracker_point=None,
                               crop_info=None,
                               global_scale=(1.0, 1.0)):
        """Build the final annotation dictionary."""
        # Collapse obj_traces to centroid (T, N, 2) → (T, 1, 2)
        obj_traces = final.obj_traces
        if obj_traces is not None:
            obj_traces = obj_traces.mean(axis=1, keepdims=True)

        intermediate_boxes = dict(final.intermediate_boxes or {})
        intermediate_boxes = intermediate_boxes if intermediate_boxes else None

        # The top-level confidence must describe the box that is actually emitted.
        # Prefer the live paper-method score and retain the old composite separately.
        confidence_breakdown = final.confidence_breakdown
        (
            annotation_confidence,
            selected_box_detection_confidence,
            selection_score,
            selection_score_name,
        ) = _resolve_emitted_initial_box_confidence(final)
        lightweight = (
            confidence_breakdown.get("lightweight")
            if isinstance(confidence_breakdown, dict)
            else {}
        ) or {}
        target_candidate_boxes = lightweight.get("target_candidate_boxes")
        target_candidate_confidences = (
            final.target_candidate_box_confidences
            or lightweight.get("target_candidate_confidences")
        )
        target_object_box_confidence = final.target_object_box_confidence
        if (
            target_object_box_confidence is None
            and target_candidate_boxes
            and target_candidate_confidences
            and final.target_object_box is not None
            and len(final.target_object_box) > 0
        ):
            target_box = np.asarray(final.target_object_box, dtype=float).reshape(-1, 4)[0]
            for candidate_box, confidence in zip(target_candidate_boxes, target_candidate_confidences):
                if np.allclose(np.asarray(candidate_box, dtype=float).reshape(4), target_box, atol=1e-3):
                    target_object_box_confidence = float(confidence)
                    break

        crop_box = (
            list(map(int, crop_info.crop_box))
            if crop_info is not None and crop_info.did_crop and crop_info.crop_box is not None
            else None
        )
        annotation_format = {
            "schema": "sparc_subtask_annotation",
            "version": 1,
            "selected_box_coordinates": "pixel_xyxy_in_original_uncropped_frame",
            "robot_detection_coordinates": "pixel_xyxy_in_processing_frame",
            "trajectory_frame_indices": "zero_based",
            "subtask_frame_interval": "start_frame_inclusive_end_frame_exclusive",
            "relative_frame_indices": "zero_based_offset_from_start_frame",
            "compressed_arrays_path": "confidence_breakdown.lightweight.arrays_blob",
            "tracks_file_format": "np_savez_compressed_or_null",
            "processing_frame_transform_to_original": {
                "formula": "original_xy=(processing_xy+crop_xy)/global_scale_xy",
                "crop_xyxy_in_globally_resized_frame": crop_box,
                "crop_origin_xy": crop_box[:2] if crop_box is not None else [0, 0],
                "global_scale_xy": [float(global_scale[0]), float(global_scale[1])],
            },
        }

        return {
            "_publication_context": {
                "split": getattr(trajectory, "split", None),
                "fps": getattr(trajectory, "fps", None) or self.fps,
                "image_size_hw": list(map(int, trajectory.frames[0].shape[:2])),
                "camera_view": getattr(trajectory, "camera_view_key", None),
                "robotseg_enabled": self.robotseg is not None,
                "robotseg_category": "robot" if self.robotseg is not None else None,
                "source_dataset_format": getattr(
                    trajectory, "source_dataset_format", None,
                ),
                "source_episode_frame_indices": getattr(
                    trajectory, "source_episode_frame_indices", None,
                ),
                "frame_timestamps_seconds": getattr(
                    trajectory, "frame_timestamps_seconds", None,
                ),
            },
            "annotation_format": annotation_format,
            "trajectory_name": str(trajectory.name),
            "source_uuid": getattr(trajectory, "source_uuid", None),
            "source_episode_id": getattr(trajectory, "source_episode_id", None),
            "source_episode_index": getattr(trajectory, "source_episode_index", None),
            "source_subtrajectory_index": getattr(
                trajectory, "source_subtrajectory_index", None
            ),
            "source_episode_frame_start": getattr(
                trajectory, "source_episode_frame_start", None
            ),
            "source_episode_frame_end_exclusive": getattr(
                trajectory, "source_episode_frame_end_exclusive", None
            ),
            "subtask_index": subtask_idx,
            "language_instruction": trajectory.lang_ann,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "initial_object_box": final.initial_object_box.tolist() if final.initial_object_box is not None else [],
            "initial_object_box_confidence": annotation_confidence,
            "initial_object_box_detection_confidence": selected_box_detection_confidence,
            "initial_object_box_selection_score": selection_score,
            "initial_object_box_selection_score_name": selection_score_name,
            "target_object_box": final.target_object_box.tolist() if final.target_object_box is not None else [],
            "target_object_box_confidence": target_object_box_confidence,
            "target_candidate_boxes": target_candidate_boxes,
            "target_candidate_box_confidences": target_candidate_confidences,
            "obj_traces": obj_traces.tolist() if obj_traces is not None else None,
            "obj_traces_cotracker_point": (
                obj_traces_cotracker_point.tolist()
                if hasattr(obj_traces_cotracker_point, "tolist")
                else obj_traces_cotracker_point
            ),
            "obj_traces_cotracker_point_meta": obj_traces_cotracker_point_meta,
            "gripper_box": det_result.gripper_box.tolist() if det_result.gripper_box is not None else [],
            "gripper_box_confidence": det_result.gripper_box_confidence,
            "detection_frame_index": det_frame_idx,
            "gripper_arm": det_result.arm,
            "robot_detections": det_result.robot_det_dict,
            "movement_type": task_obj_info["action"],
            "num_frames": len(trajectory.frames),
            "is_movement_task": True,
            "media_dir": trajectory.media_dir,
            "dataset_name": trajectory.dataset_name,
            "task_obj_info": task_obj_info,
            "task_obj_info_list": task_obj_info_list,
            "task_object_parsing_mode": (
                "semantic" if "object_instance" in task_obj_info else "text"
            ),
            "verified_target": final.verified_target,
            "grasp_phases": grasp_phases,
            "phase_annotations": subtask_phase_annotations,
            "phase_sequence": subtask_phase_sequence,
            # Tool-specific fields
            "is_tool_task": det_result.is_tool_task,
            "tool_name": det_result.tool_name if det_result.is_tool_task else None,
            "tool_usage_description": task_obj_info.get("tool_usage_description") if det_result.is_tool_task else None,
            "tool_box": (final.tool_box.tolist() if hasattr(final.tool_box, 'tolist') else
                         list(final.tool_box) if final.tool_box is not None else None)
                        if det_result.is_tool_task else None,
            "tool_box_confidence": final.tool_box_confidence if det_result.is_tool_task else None,
            "tool_traces": final.tool_traces.tolist() if (det_result.is_tool_task and final.tool_traces is not None) else None,
            "tool_traces_cotracker_point": (
                tool_traces_cotracker_point.tolist()
                if (det_result.is_tool_task and tool_traces_cotracker_point is not None) else None
            ),
            # Intermediate object boxes derived from point tracks at keyframes
            "intermediate_object_boxes": intermediate_boxes,
            # Per-component confidence breakdown for annotation quality filtering
            "confidence_breakdown": confidence_breakdown,
            # Long-horizon task grouping (populated by AgiBotWorldDatasetLoader / RoboCOINDatasetLoader)
            "longtask_id": getattr(trajectory, 'longtask_id', None),
            "longtask_step": getattr(trajectory, 'longtask_step', None),
            "longtask_len": getattr(trajectory, 'longtask_len', None),
            "longtask_goal": getattr(trajectory, 'longtask_goal', None),
            # Pre-labeled subtask phase boundaries from dataset annotations
            # (e.g. Grasp phase start/end, Place phase start/end within this trajectory clip)
            # Format: list of {"phase_type": str, "start_frame": int, "end_frame": int}
            "dataset_subtask_phases": getattr(trajectory, 'interaction_phases', None),
            # Bimanual fields (None / False for single-arm trajectories)
            "is_bimanual": False,
            "arm_annotations": None,
            # Path to companion .npz file with per-candidate raw track arrays
            "tracks_file": tracks_file,
        }
