"""Annotation coordinates for trajectory annotation."""

import logging
from copy import deepcopy
import numpy as np
from sparc.perception.bbox_ops import shift_boxes, shift_box_dicts
from sparc.pipeline.pipeline_types import CropInfo, FinalBoxes


def _pick_redacted_candidate_idx(candidate_confidences, score_matrix=None):
    """Pick a fallback candidate when tracking verification rejected the match."""
    if score_matrix is not None:
        try:
            scores = np.asarray(score_matrix)
            if scores.ndim == 2 and scores.size > 0:
                per_start = scores.max(axis=1)
                if per_start.size > 0:
                    return int(np.argmax(per_start))
        except Exception:
            pass

    if candidate_confidences is None:
        return None

    try:
        confs = np.asarray(candidate_confidences)
        if confs.size > 0:
            return int(np.argmax(confs))
    except Exception:
        return None
    return None


def _extract_obj_traces_from_raw_tracks(raw_tracks_data, winner_idx):
    """Recover per-winner traces from raw track payloads for annotation/debug saving."""
    if raw_tracks_data is None or winner_idx is None:
        return None

    try:
        cotracker_tracks = raw_tracks_data.get("cotracker_tracks")
        if cotracker_tracks is not None:
            tracks = np.asarray(cotracker_tracks)
            if tracks.ndim == 4 and 0 <= winner_idx < tracks.shape[0]:
                return tracks[winner_idx]
    except Exception:
        pass

    return None


def _extract_winner_visibility_from_raw_tracks(raw_tracks_data, winner_tracks):
    """Return (T, N) visibility for the winner track when point-track data exists."""
    if raw_tracks_data is None or winner_tracks is None:
        return None

    vis = raw_tracks_data.get("cotracker_visibility")
    if vis is None:
        return None

    try:
        vis = np.asarray(vis)
    except Exception:
        return None

    if vis.ndim == 2:
        winner_vis = vis
    elif vis.ndim == 3:
        winner_idx = raw_tracks_data.get("winner_idx")
        try:
            winner_idx = int(np.asarray(winner_idx).item())
        except Exception:
            winner_idx = 0 if vis.shape[0] == 1 else None
        if winner_idx is None or not (0 <= winner_idx < vis.shape[0]):
            return None
        winner_vis = vis[winner_idx]
    else:
        return None

    tracks = np.asarray(winner_tracks)
    if winner_vis.shape[:2] != tracks.shape[:2]:
        return None
    return winner_vis


class CoordinateTransforms:
    """CoordinateTransforms methods operating on the configured annotator state."""

    @staticmethod
    def _shift_raw_track_boxes(boxes, dx, dy):
        """Return boxes shifted into a different image coordinate frame."""
        if boxes is None:
            return None

        arr = np.array(boxes, copy=True)
        if arr.size == 0:
            return arr
        if arr.shape[-1] != 4:
            raise ValueError(f"Expected boxes with last dim 4, got shape {arr.shape}")

        arr[..., 0] += dx
        arr[..., 2] += dx
        arr[..., 1] += dy
        arr[..., 3] += dy
        return arr


    @staticmethod
    def _shift_raw_track_points(points, dx, dy):
        """Return tracked points shifted into a different image coordinate frame."""
        if points is None:
            return None

        arr = np.array(points, copy=True)
        if arr.size == 0:
            return arr
        if arr.shape[-1] != 2:
            raise ValueError(f"Expected points with last dim 2, got shape {arr.shape}")

        arr[..., 0] += dx
        arr[..., 1] += dy
        return arr


    def _restore_raw_tracks_data_coordinates(self, raw_tracks_data, crop_info):
        """Convert saved raw tracking artifacts back to full-frame coordinates."""
        if not raw_tracks_data:
            return raw_tracks_data

        restored = {
            key: np.array(value, copy=True) if isinstance(value, np.ndarray) else value
            for key, value in raw_tracks_data.items()
        }

        if crop_info is None or not crop_info.did_crop or crop_info.offsets is None:
            return restored

        min_x, min_y, min_x_fwd, min_y_fwd = crop_info.offsets

        for key in ("candidate_boxes", "sam2_candidate_boxes", "gripper_box"):
            if key in restored:
                restored[key] = self._shift_raw_track_boxes(restored[key], min_x, min_y)

        if "target_candidate_boxes" in restored:
            restored["target_candidate_boxes"] = self._shift_raw_track_boxes(
                restored["target_candidate_boxes"], min_x_fwd, min_y_fwd
            )

        for key in ("cotracker_tracks", "gripper_tracks"):
            if key in restored:
                restored[key] = self._shift_raw_track_points(restored[key], min_x, min_y)

        if crop_info.original_shape is not None:
            full_h, full_w = map(int, crop_info.original_shape)
            crop_x1, crop_y1, crop_x2, crop_y2 = map(int, crop_info.crop_box)
            for key in ("gripper_mask_first_frame",):
                if key in restored and restored[key] is not None:
                    mask = np.asarray(restored[key])
                    full_mask = np.zeros((full_h, full_w), dtype=mask.dtype)
                    h = min(mask.shape[0], crop_y2 - crop_y1)
                    w = min(mask.shape[1], crop_x2 - crop_x1)
                    full_mask[crop_y1:crop_y1 + h, crop_x1:crop_x1 + w] = mask[:h, :w]
                    restored[key] = full_mask

            if "gripper_video_masks_robotseg" in restored and restored["gripper_video_masks_robotseg"] is not None:
                masks = np.asarray(restored["gripper_video_masks_robotseg"])
                if masks.ndim == 3:
                    full_masks = np.zeros((masks.shape[0], full_h, full_w), dtype=masks.dtype)
                    h = min(masks.shape[1], crop_y2 - crop_y1)
                    w = min(masks.shape[2], crop_x2 - crop_x1)
                    full_masks[:, crop_y1:crop_y1 + h, crop_x1:crop_x1 + w] = masks[:, :h, :w]
                    restored["gripper_video_masks_robotseg"] = full_masks

        restored["crop_box"] = np.asarray(crop_info.crop_box, dtype=np.int32)
        restored["crop_offsets"] = np.asarray(crop_info.offsets, dtype=np.int32)
        return restored


    def _unscale_raw_tracks(self, raw_tracks_data, global_scale):
        """Scale track points, boxes, and masks from resized back to original pixel coordinates."""
        import cv2
        sx, sy = global_scale
        inv_pt  = np.array([1.0 / sx, 1.0 / sy], dtype=np.float64)
        inv_box = np.array([1.0 / sx, 1.0 / sy, 1.0 / sx, 1.0 / sy], dtype=np.float64)

        result = dict(raw_tracks_data)

        for key in ("cotracker_tracks", "gripper_tracks"):
            if key in result and result[key] is not None:
                result[key] = np.asarray(result[key], dtype=np.float64) * inv_pt

        for key in ("candidate_boxes", "sam2_candidate_boxes", "gripper_box",
                    "target_candidate_boxes", "gripper_all_candidate_boxes"):
            if key in result and result[key] is not None:
                result[key] = np.asarray(result[key], dtype=np.float64) * inv_box

        # Spatially resize mask arrays back to original resolution
        if "gripper_mask_first_frame" in result and result["gripper_mask_first_frame"] is not None:
            m = result["gripper_mask_first_frame"]
            orig_H = int(round(m.shape[0] / sy))
            orig_W = int(round(m.shape[1] / sx))
            result["gripper_mask_first_frame"] = cv2.resize(
                m, (orig_W, orig_H), interpolation=cv2.INTER_NEAREST
            )

        if "gripper_video_masks_robotseg" in result and result["gripper_video_masks_robotseg"] is not None:
            masks = result["gripper_video_masks_robotseg"]  # (T, H, W)
            orig_H = int(round(masks.shape[1] / sy))
            orig_W = int(round(masks.shape[2] / sx))
            result["gripper_video_masks_robotseg"] = np.stack([
                cv2.resize(masks[t], (orig_W, orig_H), interpolation=cv2.INTER_NEAREST)
                for t in range(masks.shape[0])
            ])

        return result


    def _resize_frames_global(self, frames_np):
        """Resize all frames so min-edge <= self.max_resolution.

        Returns (resized_frames_np, (sx, sy)) where sx=new_W/orig_W, sy=new_H/orig_H.
        Returns original frames and (1.0, 1.0) when no resize is needed.
        """
        if self.max_resolution is None or len(frames_np) == 0:
            return frames_np, (1.0, 1.0)
        H, W = frames_np.shape[1], frames_np.shape[2]
        min_edge = min(H, W)
        if min_edge <= self.max_resolution:
            return frames_np, (1.0, 1.0)
        import cv2
        scale = self.max_resolution / min_edge
        new_H = int(round(H * scale))
        new_W = int(round(W * scale))
        resized = np.stack([cv2.resize(f, (new_W, new_H), interpolation=cv2.INTER_LINEAR) for f in frames_np])
        logging.info(f"[GLOBAL_RESIZE] {H}x{W} -> {new_H}x{new_W} (scale={scale:.3f}, {len(frames_np)} frames)")
        return resized, (new_W / W, new_H / H)


    @staticmethod
    def _transform_box_list_for_output(boxes, *, dx=0.0, dy=0.0, global_scale=(1.0, 1.0)):
        """Shift/scale a JSON-serializable list of xyxy boxes into output coordinates."""
        if boxes is None:
            return None
        sx, sy = global_scale
        inv = np.array([1.0 / sx, 1.0 / sy, 1.0 / sx, 1.0 / sy], dtype=np.float64)
        shift = np.array([dx, dy, dx, dy], dtype=np.float64)
        out = []
        for box in boxes:
            if box is None:
                out.append(None)
                continue
            arr = np.asarray(box, dtype=np.float64).reshape(-1)
            if arr.size != 4 or not np.all(np.isfinite(arr)):
                out.append(box)
                continue
            out.append(((arr + shift) * inv).tolist())
        return out


    def _confidence_breakdown_boxes_to_output_coordinates(
        self,
        confidence_breakdown,
        crop_info,
        global_scale=(1.0, 1.0),
    ):
        """Convert saved replay box fields to the same full-frame coordinates as outputs/NPZ."""
        if confidence_breakdown is None:
            return None

        transformed = deepcopy(confidence_breakdown)
        start_dx = start_dy = target_dx = target_dy = 0.0
        if crop_info is not None and crop_info.did_crop and crop_info.offsets is not None:
            start_dx, start_dy, target_dx, target_dy = map(float, crop_info.offsets)

        candidate_boxes = [
            (item or {}).get("box")
            for item in transformed.get("all_candidate_breakdowns", [])
        ]
        if candidate_boxes:
            shifted = self._transform_box_list_for_output(
                candidate_boxes,
                dx=start_dx,
                dy=start_dy,
                global_scale=global_scale,
            )
            for item, box in zip(transformed.get("all_candidate_breakdowns", []), shifted):
                if isinstance(item, dict) and box is not None:
                    item["box"] = box

        lightweight = transformed.get("lightweight")
        if isinstance(lightweight, dict):
            if "candidate_boxes" in lightweight:
                lightweight["candidate_boxes"] = self._transform_box_list_for_output(
                    lightweight.get("candidate_boxes"),
                    dx=start_dx,
                    dy=start_dy,
                    global_scale=global_scale,
                )
            if "target_candidate_boxes" in lightweight:
                lightweight["target_candidate_boxes"] = self._transform_box_list_for_output(
                    lightweight.get("target_candidate_boxes"),
                    dx=target_dx,
                    dy=target_dy,
                    global_scale=global_scale,
                )
            per_candidate = lightweight.get("per_candidate_score_breakdown")
            if isinstance(per_candidate, list):
                boxes = [
                    item.get("box") if isinstance(item, dict) else None
                    for item in per_candidate
                ]
                shifted = self._transform_box_list_for_output(
                    boxes,
                    dx=start_dx,
                    dy=start_dy,
                    global_scale=global_scale,
                )
                for item, box in zip(per_candidate, shifted):
                    if isinstance(item, dict) and box is not None:
                        item["box"] = box

        reconstruction = transformed.get("score_reconstruction")
        if not isinstance(reconstruction, dict):
            reconstruction = {}
            transformed["score_reconstruction"] = reconstruction
        reconstruction["box_coordinate_frame"] = "full_frame"
        reconstruction["box_coordinate_transform"] = {
            "crop_offsets_applied": bool(
                crop_info is not None and crop_info.did_crop and crop_info.offsets is not None
            ),
            "global_scale_applied": [float(global_scale[0]), float(global_scale[1])],
        }
        return transformed


    def _final_with_output_coordinate_confidence_breakdown(self, final, crop_info, global_scale=(1.0, 1.0)):
        final.confidence_breakdown = self._confidence_breakdown_boxes_to_output_coordinates(
            final.confidence_breakdown,
            crop_info,
            global_scale=global_scale,
        )
        return final


    def _maybe_crop(self, frames_subset, frames_torch, det_result, target_result, dataset_name):
        """Apply cropping for droid datasets. Returns CropInfo.

        Stores modified frames/boxes as private attributes on CropInfo for the caller to use.
        """
        do_crop = "droid" in dataset_name.lower() or self.enable_crop
        original_shape = tuple(frames_subset.shape[1:3]) if len(frames_subset) > 0 else None

        if not do_crop:
            return CropInfo(
                did_crop=False, crop_box=None, offsets=None,
                cropped_start_frame=None,
                cropped_detected_boxes=None, cropped_target_boxes=None,
                original_shape=original_shape,
            )

        all_boxes_np = det_result.all_boxes_np
        h, w = frames_subset.shape[1:3]

        padding_left = int(w * 0.1)
        padding_right = int(w * 0.1)
        padding_top = int(h * 0.3)
        padding_bottom = int(h * 0.1)

        min_x = int(max(0, all_boxes_np[:, 0].min() - padding_left))
        min_y = int(max(0, all_boxes_np[:, 1].min() - padding_top))
        max_x = int(min(w, all_boxes_np[:, 2].max() + padding_right))
        max_y = int(min(h, all_boxes_np[:, 3].max() + padding_bottom))

        crop_box = (min_x, min_y, max_x, max_y)

        frames_subset = frames_subset[:, min_y:max_y, min_x:max_x]
        frames_torch = frames_torch[:, :, :, min_y:max_y, min_x:max_x]

        cropped_start_frame = frames_subset[0] if frames_subset.shape[0] > 0 else None

        # Shift detection boxes
        det_result.pred_boxes[0].xyxy = shift_boxes(det_result.pred_boxes[0].xyxy, -min_x, -min_y)
        cropped_detected_boxes = det_result.pred_boxes[0].xyxy.copy()

        # Shift target boxes
        cropped_target_boxes = None
        if (target_result.pred_boxes_target_merged is not None and
                len(target_result.pred_boxes_target_merged[0].xyxy) > 0):
            target_result.pred_boxes_target_merged[0].xyxy = shift_boxes(
                target_result.pred_boxes_target_merged[0].xyxy, -min_x, -min_y,
            )
            cropped_target_boxes = target_result.pred_boxes_target_merged[0].xyxy.copy()

        # Shift candidate and target box dicts
        shift_box_dicts(det_result.candidate_boxes, -min_x, -min_y)
        shift_box_dicts(target_result.target_boxes, -min_x, -min_y)

        if det_result.robot_det_dict:
            for robot_det in det_result.robot_det_dict.values():
                if robot_det is not None and robot_det.get("box") is not None:
                    box = np.asarray(robot_det["box"], dtype=np.float32)
                    box = shift_boxes(box[None], -min_x, -min_y)[0]
                    robot_det["box"] = [round(float(coord)) for coord in box]

        if det_result.gripper_box is not None:
            det_result.gripper_box = shift_boxes(det_result.gripper_box, -min_x, -min_y)

        crop_info = CropInfo(
            did_crop=True,
            crop_box=crop_box,
            offsets=(min_x, min_y, min_x, min_y),  # (min_x, min_y, min_x_fwd, min_y_fwd)
            cropped_start_frame=cropped_start_frame,
            cropped_detected_boxes=cropped_detected_boxes,
            cropped_target_boxes=cropped_target_boxes,
            original_shape=original_shape,
        )
        # Attach modified frames for caller
        crop_info._frames_subset = frames_subset
        crop_info._frames_torch = frames_torch
        crop_info._det_result = det_result
        return crop_info


    def _restore_coordinates(self, tracking_result, crop_info, target_candidate_box_confidences=None):
        """Shift boxes back to original coordinates if cropping was applied. Returns FinalBoxes."""
        best_obj_box = tracking_result.best_obj_box
        best_obj_box_target = tracking_result.best_obj_box_target
        obj_traces = tracking_result.obj_traces
        intermediate_boxes = tracking_result.intermediate_boxes

        if best_obj_box_target is None:
            best_obj_box_target = np.empty((0, 4))

        best_obj_box_target_confidence = None
        obj_traces_target = None

        if crop_info.did_crop and crop_info.offsets is not None:
            min_x, min_y, min_x_fwd, min_y_fwd = crop_info.offsets

            if len(best_obj_box.shape) == 1:
                best_obj_box = best_obj_box[None, :]
            best_obj_box = shift_boxes(best_obj_box, min_x, min_y)

            if best_obj_box_target is not None and len(best_obj_box_target) > 0:
                if len(best_obj_box_target.shape) == 1:
                    best_obj_box_target = best_obj_box_target[None, :]
                best_obj_box_target = shift_boxes(best_obj_box_target, min_x_fwd, min_y_fwd)

            if obj_traces is not None:
                obj_traces = obj_traces.copy()
                obj_traces[:, :, 0] += min_x
                obj_traces[:, :, 1] += min_y

            if intermediate_boxes:
                shifted_ib = {}
                for kf, box in intermediate_boxes.items():
                    arr = np.array(box, dtype=float)
                    arr[0] += min_x; arr[2] += min_x
                    arr[1] += min_y; arr[3] += min_y
                    shifted_ib[kf] = arr.tolist()
                intermediate_boxes = shifted_ib

        return FinalBoxes(
            initial_object_box=best_obj_box,
            initial_object_box_confidence=tracking_result.best_obj_box_confidence,
            target_object_box=best_obj_box_target,
            target_object_box_confidence=best_obj_box_target_confidence,
            obj_traces=obj_traces,
            obj_traces_target=obj_traces_target,
            verified_target=tracking_result.verified_target,
            tool_box=None,
            tool_box_confidence=None,
            tool_traces=None,
            intermediate_boxes=intermediate_boxes,
            target_candidate_box_confidences=target_candidate_box_confidences,
            confidence_breakdown=tracking_result.confidence_breakdown,
        )


    def _unscale_final(self, final, global_scale):
        """Unscale FinalBoxes from resized to original pixel coordinates."""
        sx, sy = global_scale
        inv_box = np.array([1.0 / sx, 1.0 / sy, 1.0 / sx, 1.0 / sy], dtype=np.float64)
        inv_pt  = np.array([1.0 / sx, 1.0 / sy], dtype=np.float64)

        def scale_box(b):
            if b is None or (hasattr(b, '__len__') and len(b) == 0):
                return b
            return np.asarray(b, dtype=np.float64) * inv_box

        def scale_traces(t):
            if t is None or (hasattr(t, '__len__') and len(t) == 0):
                return t
            return np.asarray(t, dtype=np.float64) * inv_pt

        def scale_box_dict(d):
            if not d:
                return d
            return {k: (np.asarray(v, dtype=np.float64) * inv_box).tolist() for k, v in d.items()}

        return FinalBoxes(
            initial_object_box=scale_box(final.initial_object_box),
            initial_object_box_confidence=final.initial_object_box_confidence,
            target_object_box=scale_box(final.target_object_box),
            target_object_box_confidence=final.target_object_box_confidence,
            obj_traces=scale_traces(final.obj_traces),
            obj_traces_target=scale_traces(final.obj_traces_target),
            verified_target=final.verified_target,
            tool_box=scale_box(final.tool_box),
            tool_box_confidence=final.tool_box_confidence,
            tool_traces=scale_traces(final.tool_traces),
            arm=final.arm,
            intermediate_boxes=scale_box_dict(final.intermediate_boxes),
            target_candidate_box_confidences=final.target_candidate_box_confidences,
            confidence_breakdown=final.confidence_breakdown,
        )
