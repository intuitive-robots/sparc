"""Debug image saving for annotation inspection."""

import json
import logging
import textwrap
from collections import defaultdict
from pathlib import Path
from re import sub

import numpy as np


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


class DebugImageSaver:
    """Saves annotated debug images for visual inspection of annotation results.

    Extracted from TrajectoryAnnotator._maybe_save_debug_image and helpers.
    """

    def __init__(self, debug_config: dict, gpu_id: int):
        debug_config = debug_config or {}
        self.save_debug_images = bool(debug_config.get("save_debug_images", False))
        self.debug_image_freq = max(1, int(debug_config.get("debug_image_freq", 1)))
        self.debug_image_format = str(debug_config.get("debug_image_format", "png")).lower().lstrip(".")
        self.debug_max_per_traj = debug_config.get("debug_max_per_traj")
        self.debug_image_dir = None
        self._debug_annotation_counter = 0
        self._debug_saves_per_traj = defaultdict(int)

        if self.save_debug_images:
            debug_dir = debug_config.get("debug_image_dir")
            if debug_dir is None:
                debug_dir = Path.cwd() / "annotation_debug"
            self.debug_image_dir = Path(debug_dir) / f"gpu_{gpu_id}"
            self.debug_image_dir.mkdir(parents=True, exist_ok=True)

    def maybe_save(
        self,
        trajectory,
        subtask_idx: int,
        start_frame,
        end_frame,
        initial_box,
        target_box,
        annotation,
        det_frame=None,
        detected_boxes_fullres=None,
        target_boxes_fullres=None,
        crop_box=None,
        cropped_start_frame=None,
        cropped_detected_boxes=None,
        cropped_target_boxes=None,
        tool_box=None,
        tool_boxes_fullres=None,
        robot_detections=None,
        obj_traces=None,
        intermediate_debug=None,
        fps=10.0,
    ) -> None:
        if not self.save_debug_images or self.debug_image_dir is None:
            return

        self._debug_annotation_counter += 1
        if self._debug_annotation_counter != 1 and self._debug_annotation_counter % self.debug_image_freq != 0:
            return

        traj_key = str(trajectory.name)
        if self.debug_max_per_traj:
            if self._debug_saves_per_traj[traj_key] >= self.debug_max_per_traj:
                return

        try:
            ImageCls, ImageDrawCls, ImageFontCls = self._lazy_import_pillow()
        except RuntimeError as exc:
            logging.warning("Skipping debug image saving: %s", exc)
            self.save_debug_images = False
            return

        from sparc.perception.detection_utils import plot_boxes_np

        start_frame_np = self._ensure_rgb(self._to_uint8_image(start_frame))
        end_frame_np = self._ensure_rgb(self._to_uint8_image(end_frame))
        cropped_start_np = self._ensure_rgb(self._to_uint8_image(cropped_start_frame)) if cropped_start_frame is not None else None

        initial_boxes_np = self._to_numpy_boxes(initial_box)
        target_boxes_np = self._to_numpy_boxes(target_box)
        detected_boxes_fullres_np = self._to_numpy_boxes(detected_boxes_fullres)
        target_boxes_fullres_np = self._to_numpy_boxes(target_boxes_fullres)
        cropped_detected_boxes_np = self._to_numpy_boxes(cropped_detected_boxes)
        cropped_target_boxes_np = self._to_numpy_boxes(cropped_target_boxes)

        arm = annotation.get("arm")
        init_conf = annotation.get("initial_object_box_confidence")
        tgt_conf = annotation.get("target_object_box_confidence")

        labels_start = []
        scores_start = []
        box_parts = []
        if initial_boxes_np.size > 0:
            box_parts.append(initial_boxes_np)
            labels_start.extend(["initial"] * len(initial_boxes_np))
            scores_start.extend([init_conf if init_conf is not None else 0.0] * len(initial_boxes_np))
        if target_boxes_np.size > 0:
            box_parts.append(target_boxes_np)
            labels_start.extend(["target"] * len(target_boxes_np))
            scores_start.extend([tgt_conf if tgt_conf is not None else 0.0] * len(target_boxes_np))

        if box_parts:
            boxes_start = np.concatenate(box_parts, axis=0)
        else:
            boxes_start = np.empty((0, 4), dtype=np.float32)

        start_panel = self._render_frame_with_boxes(ImageCls, plot_boxes_np, start_frame_np, boxes_start, labels_start, scores=scores_start)

        # Panel: original start with all detections (initial + target), and crop overlay if available
        det_labels = []
        det_parts = []
        if detected_boxes_fullres_np.size > 0:
            det_parts.append(detected_boxes_fullres_np)
            det_labels.extend(["det_init"] * len(detected_boxes_fullres_np))
        if target_boxes_fullres_np.size > 0:
            det_parts.append(target_boxes_fullres_np)
            det_labels.extend(["det_target"] * len(target_boxes_fullres_np))
        if det_parts:
            det_boxes = np.concatenate(det_parts, axis=0)
        else:
            det_boxes = np.empty((0, 4), dtype=np.float32)

        detections_panel = self._render_frame_with_boxes(ImageCls, plot_boxes_np, start_frame_np, det_boxes, det_labels)
        if crop_box is not None:
            draw_crop = ImageDrawCls.Draw(detections_panel)
            crop_color = (255, 140, 0)
            draw_crop.rectangle(crop_box, outline=crop_color, width=2)

        # Panel: cropped start with detections (if available)
        cropped_panel = None
        if cropped_start_np is not None:
            crop_det_labels = []
            crop_det_parts = []
            if cropped_detected_boxes_np.size > 0:
                crop_det_parts.append(cropped_detected_boxes_np)
                crop_det_labels.extend(["crop_init"] * len(cropped_detected_boxes_np))
            if cropped_target_boxes_np.size > 0:
                crop_det_parts.append(cropped_target_boxes_np)
                crop_det_labels.extend(["crop_target"] * len(cropped_target_boxes_np))
            if crop_det_parts:
                crop_det_boxes = np.concatenate(crop_det_parts, axis=0)
            else:
                crop_det_boxes = np.empty((0, 4), dtype=np.float32)
            cropped_panel = self._render_frame_with_boxes(ImageCls, plot_boxes_np, cropped_start_np, crop_det_boxes, crop_det_labels)

        target_labels = ["target"] * len(target_boxes_np) if target_boxes_np.size > 0 else []
        target_scores = [tgt_conf if tgt_conf is not None else 0.0] * len(target_boxes_np) if target_boxes_np.size > 0 else []
        end_panel = self._render_frame_with_boxes(ImageCls, plot_boxes_np, end_frame_np, target_boxes_np, target_labels, scores=target_scores)

        # Panel: robot detection at first keyframe only (frame 0 = start frame).
        # Each keyframe has its own box; drawing all on one image would be misleading.
        robot_panel = None
        if robot_detections:
            first_key = min(robot_detections.keys())
            det = robot_detections[first_key]
            box = det["box"] if isinstance(det, dict) else det
            conf = det.get("confidence") if isinstance(det, dict) else None
            conf_str = f" {conf:.2f}" if conf is not None else ""
            robot_label = f"{arm}_arm{conf_str}" if arm else f"robot{conf_str}"
            robot_boxes = np.array([box], dtype=np.float32)
            robot_panel = self._render_frame_with_boxes(ImageCls, plot_boxes_np, start_frame_np, robot_boxes, [robot_label])

        # Panel: object traces
        traces_panel = None
        if obj_traces is not None:
            traces_panel = self._render_traces_panel(ImageCls, ImageDrawCls, start_frame_np, obj_traces)

        panels = [start_panel, detections_panel]
        if cropped_panel is not None:
            panels.append(cropped_panel)
        panels.append(end_panel)
        if robot_panel is not None:
            panels.append(robot_panel)
        if traces_panel is not None:
            panels.append(traces_panel)

        # Row 2: same panels but using the detection/tracking start frame
        det_panels = []
        if det_frame is not None:
            det_frame_np = self._ensure_rgb(self._to_uint8_image(det_frame))
            det_panels.append(self._render_frame_with_boxes(ImageCls, plot_boxes_np, det_frame_np, boxes_start, labels_start))
            det_panels.append(self._render_frame_with_boxes(ImageCls, plot_boxes_np, det_frame_np, det_boxes, det_labels))
            if robot_detections:
                det_robot_panel = None
                for key in robot_detections:
                    det = robot_detections[key]
                    box = det["box"] if isinstance(det, dict) else det
                    conf = det.get("confidence") if isinstance(det, dict) else None
                    conf_str = f" {conf:.2f}" if conf is not None else ""
                    arm = annotation.get("arm")
                    robot_label = f"{arm}_arm{conf_str}" if arm else f"robot{conf_str}"
                    rb = np.array([box], dtype=np.float32)
                    det_robot_panel = self._render_frame_with_boxes(ImageCls, plot_boxes_np, det_frame_np, rb, [robot_label])
                    break  # first key (lowest frame index = earliest keyframe)
                if det_robot_panel is not None:
                    det_panels.append(det_robot_panel)

        # Row 3: all annotated keyframes — each frame with its stored object box + robot/gripper box
        intermediate_panels = []
        if intermediate_debug:
            font_kf = ImageFontCls.load_default() if ImageFontCls else None
            for entry in intermediate_debug:
                kf_frame_np = self._ensure_rgb(self._to_uint8_image(entry["frame"]))
                all_boxes_parts = []
                all_labels = []

                # Object box (best stored annotation for this keyframe)
                obj_box = entry.get("object_box")
                if obj_box is not None:
                    all_boxes_parts.append(np.array([obj_box], dtype=np.float32))
                    all_labels.append(entry.get("object_label") or "obj")

                # Robot / gripper box
                robot_box = entry.get("robot_box")
                if robot_box is not None:
                    rb = robot_box["box"] if isinstance(robot_box, dict) else robot_box
                    rconf = robot_box.get("confidence") if isinstance(robot_box, dict) else None
                    all_boxes_parts.append(np.array([rb], dtype=np.float32))
                    rconf_str = f" {rconf:.2f}" if rconf is not None else ""
                    all_labels.append(f"gripper{rconf_str}")

                if all_boxes_parts:
                    combined_boxes = np.concatenate(all_boxes_parts, axis=0)
                else:
                    combined_boxes = np.empty((0, 4), dtype=np.float32)

                panel = self._render_frame_with_boxes(ImageCls, plot_boxes_np, kf_frame_np, combined_boxes, all_labels)
                # Overlay tracked points
                track_points = entry.get("track_points")
                if track_points is not None:
                    panel = self._overlay_track_points(ImageCls, ImageDrawCls, panel, track_points)
                # Overlay frame index
                draw_panel = ImageDrawCls.Draw(panel)
                draw_panel.text((4, 4), f"f{entry['frame_idx']}", fill=(255, 220, 0), font=font_kf)
                intermediate_panels.append(panel)

        caption_height = 32
        row_label_height = 16  # small label to distinguish rows
        panel_height = max(p.height for p in panels)
        row1_width = sum(p.width for p in panels)

        if det_panels:
            det_panel_height = max(p.height for p in det_panels)
            det_row_width = sum(p.width for p in det_panels)
            canvas_width = max(row1_width, det_row_width)
            canvas_height = caption_height + panel_height + row_label_height + det_panel_height
        else:
            det_panel_height = 0
            canvas_width = row1_width
            canvas_height = caption_height + panel_height

        if intermediate_panels:
            intermediate_panel_height = max(p.height for p in intermediate_panels)
            intermediate_row_width = sum(p.width for p in intermediate_panels)
            canvas_width = max(canvas_width, intermediate_row_width)
            canvas_height += row_label_height + intermediate_panel_height

        canvas = ImageCls.new("RGB", (canvas_width, canvas_height), color=(15, 15, 15))

        x_offset = 0
        for panel in panels:
            canvas.paste(panel, (x_offset, caption_height))
            x_offset += panel.width

        if det_panels:
            det_row_y = caption_height + panel_height + row_label_height
            x_offset = 0
            for panel in det_panels:
                canvas.paste(panel, (x_offset, det_row_y))
                x_offset += panel.width

        if intermediate_panels:
            if det_panels:
                int_row_y = caption_height + panel_height + row_label_height + det_panel_height + row_label_height
            else:
                int_row_y = caption_height + panel_height + row_label_height
            x_offset = 0
            for panel in intermediate_panels:
                canvas.paste(panel, (x_offset, int_row_y))
                x_offset += panel.width

        draw = ImageDrawCls.Draw(canvas)
        font = ImageFontCls.load_default() if ImageFontCls else None

        dataset_name = getattr(trajectory, "dataset_name", "unknown") or "unknown"
        arm_tag = f" | arm={arm}" if arm is not None else ""
        caption = f"{dataset_name} | {trajectory.name} | subtask {subtask_idx}{arm_tag}"
        draw.text((8, 6), caption, fill=(255, 255, 255), font=font)
        instruction = annotation.get("language_instruction") or ""
        if instruction:
            instruction_line = textwrap.shorten(instruction, width=90, placeholder="…")
            draw.text((8, 6 + 16), instruction_line, fill=(180, 180, 180), font=font)

        if det_panels:
            row1_label_y = caption_height + panel_height - row_label_height
            draw.rectangle([(0, row1_label_y), (120, caption_height + panel_height)], fill=(40, 40, 40))
            draw.text((4, row1_label_y + 2), "row1: start frame", fill=(180, 180, 180), font=font)
            row2_label_y = caption_height + panel_height
            draw.rectangle([(0, row2_label_y), (140, row2_label_y + row_label_height)], fill=(40, 40, 80))
            draw.text((4, row2_label_y + 2), "row2: det/track frame", fill=(160, 200, 255), font=font)

        if intermediate_panels:
            if det_panels:
                row3_label_y = caption_height + panel_height + row_label_height + det_panel_height
            else:
                row3_label_y = caption_height + panel_height
            draw.rectangle([(0, row3_label_y), (180, row3_label_y + row_label_height)], fill=(40, 60, 40))
            draw.text((4, row3_label_y + 2), "row3: intermediate keyframes", fill=(160, 255, 160), font=font)

        safe_dataset = self._sanitize_for_path(str(dataset_name))
        safe_traj = self._sanitize_for_path(str(trajectory.name))

        traj_dir = self.debug_image_dir / safe_dataset
        traj_dir.mkdir(parents=True, exist_ok=True)

        file_base = f"{safe_traj}_subtask_{subtask_idx:02d}_ann_{self._debug_annotation_counter:06d}"
        image_suffix = f".{self.debug_image_format}" if self.debug_image_format else ".png"
        image_path = traj_dir / f"{file_base}{image_suffix}"

        try:
            canvas.save(image_path)
        except ValueError:
            fallback_path = traj_dir / f"{file_base}.png"
            canvas.save(fallback_path)
            image_path = fallback_path
            logging.info(f"Saved debug image (fallback): {image_path}")

        meta_path = image_path.with_suffix(".json")
        try:
            metadata = json.dumps(annotation, indent=2, default=_json_default)
            meta_path.write_text(metadata)
        except Exception as exc:
            logging.warning("Failed to write debug metadata for %s: %s", image_path, exc)

        self._debug_saves_per_traj[traj_key] += 1

    def save_robot_candidates(
        self,
        trajectory,
        subtask_idx: int,
        frames_subset,
        keyframe_indices,
        all_robot_candidates,
        arm=None,
    ) -> None:
        """Save all robot candidate detections (all keyframes) to a debug image file."""
        if not self.save_debug_images or self.debug_image_dir is None:
            return
        if not all_robot_candidates:
            return

        try:
            ImageCls, ImageDrawCls, ImageFontCls = self._lazy_import_pillow()
        except RuntimeError:
            return

        from sparc.perception.detection_utils import plot_boxes_np

        panels = []
        for i, frame_idx in enumerate(keyframe_indices):
            if i >= len(all_robot_candidates):
                break
            if frame_idx >= len(frames_subset):
                continue
            frame_np = self._ensure_rgb(self._to_uint8_image(frames_subset[frame_idx]))
            dets = all_robot_candidates[i]
            if dets is not None and hasattr(dets, "xyxy") and len(dets.xyxy) > 0:
                boxes = dets.xyxy
                confs = dets.confidence if dets.confidence is not None else [1.0] * len(boxes)
                labels = [f"robot {c:.2f}" for c in confs]
                rendered = plot_boxes_np(frame_np, boxes, labels=labels, scores=list(confs), return_image=True)
                rendered = self._ensure_rgb(rendered)
            else:
                rendered = frame_np
            panel = ImageCls.fromarray(rendered)
            panels.append((frame_idx, panel))

        if not panels:
            return

        caption_height = 32
        panel_height = max(p.height for _, p in panels)
        combined_width = sum(p.width for _, p in panels)
        canvas = ImageCls.new("RGB", (combined_width, panel_height + caption_height), color=(15, 15, 15))
        draw = ImageDrawCls.Draw(canvas)
        font = ImageFontCls.load_default() if ImageFontCls else None

        dataset_name = getattr(trajectory, "dataset_name", "unknown") or "unknown"
        arm_tag = f" | arm={arm}" if arm else ""
        caption = f"{dataset_name} | {trajectory.name} | subtask {subtask_idx} | robot candidates{arm_tag}"
        draw.text((8, 6), caption, fill=(255, 255, 255), font=font)

        x_offset = 0
        for frame_idx, panel in panels:
            canvas.paste(panel, (x_offset, caption_height))
            draw.text((x_offset + 4, caption_height + 4), f"f{frame_idx}", fill=(255, 220, 0), font=font)
            x_offset += panel.width

        safe_dataset = self._sanitize_for_path(str(dataset_name))
        safe_traj = self._sanitize_for_path(str(trajectory.name))
        traj_dir = self.debug_image_dir / safe_dataset
        traj_dir.mkdir(parents=True, exist_ok=True)
        file_base = f"{safe_traj}_subtask_{subtask_idx:02d}_robot_cands"
        if arm:
            file_base += f"_{arm}"
        image_path = traj_dir / f"{file_base}.png"
        try:
            canvas.save(image_path)
        except Exception as exc:
            logging.warning("Failed to save robot candidates debug image: %s", exc)

    def _render_frame_with_boxes(self, image_cls, plot_fn, frame, boxes, labels, scores=None):
        frame_rgb = self._ensure_rgb(frame)
        boxes_np = self._to_numpy_boxes(boxes)
        if boxes_np.size > 0:
            if not labels or len(labels) != len(boxes_np):
                labels = [""] * len(boxes_np)
            if scores is None or len(scores) != len(boxes_np):
                scores = [1.0] * len(boxes_np)
            rendered = plot_fn(frame_rgb, boxes_np, labels=labels, scores=scores, return_image=True)
            rendered = self._ensure_rgb(rendered)
        else:
            rendered = frame_rgb
        return image_cls.fromarray(rendered)

    def _render_traces_panel(self, image_cls, draw_cls, frame_np, obj_traces):
        """Draw tracked point traces ([T, N, 2]) on a copy of frame_np."""
        import cv2
        bg = self._ensure_rgb(frame_np).copy()
        traces = np.asarray(obj_traces, dtype=np.float32)
        # Normalise to [T, N, 2]
        if traces.ndim == 2:
            traces = traces[:, None, :]  # [T, 1, 2]
        if traces.ndim != 3 or traces.shape[2] != 2:
            return image_cls.fromarray(bg)
        T, N, _ = traces.shape
        for n in range(N):
            for t in range(T):
                x, y = int(traces[t, n, 0]), int(traces[t, n, 1])
                alpha = (t + 1) / T
                color = (int(255 * (1 - alpha)), int(255 * alpha), 0)
                cv2.circle(bg, (x, y), 2, color, -1)
        return image_cls.fromarray(bg)

    def _overlay_track_points(self, image_cls, draw_cls, panel, track_points):
        """Draw tracked points ([N, 2]) as small cyan circles on a PIL panel."""
        import cv2
        arr = self._ensure_rgb(np.asarray(panel)).copy()
        pts = np.asarray(track_points, dtype=np.float32)
        if pts.ndim == 2 and pts.shape[1] == 2:
            for x, y in pts:
                cv2.circle(arr, (int(x), int(y)), 2, (0, 220, 255), -1)
        return image_cls.fromarray(arr)

    def _ensure_rgb(self, image):
        arr = np.asarray(image)
        if arr.ndim == 0:
            return np.zeros((1, 1, 3), dtype=np.uint8)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)
        if arr.shape[2] == 4:
            arr = arr[..., :3]
        return arr

    def _to_uint8_image(self, frame):
        arr = np.asarray(frame)
        if arr.dtype.kind == "f":
            arr = np.clip(arr, 0, 255)
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        return arr

    def _to_numpy_boxes(self, boxes):
        if boxes is None:
            return np.empty((0, 4), dtype=np.float32)
        if hasattr(boxes, "detach"):
            boxes = boxes.detach().cpu().numpy()
        boxes_arr = np.asarray(boxes)
        if boxes_arr.size == 0:
            return np.empty((0, 4), dtype=np.float32)
        boxes_arr = boxes_arr.reshape(-1, 4)
        return boxes_arr.astype(np.float32)

    def _lazy_import_pillow(self):
        try:
            from PIL import Image as PILImage, ImageDraw as PILImageDraw, ImageFont as PILImageFont
        except ImportError as exc:
            raise RuntimeError("Pillow is required to save debug images. Install it with `pip install pillow`.") from exc
        return PILImage, PILImageDraw, PILImageFont

    def save_sam2_debug(
        self,
        trajectory,
        subtask_idx: int,
        annotation: dict,
        frames_subset,
        det_frame_idx: int,
        sam2_det_mask=None,
    ) -> None:
        """Save a PNG with the SAM2 image mask overlay.

        sam2_det_mask: (H, W) uint8 — SAM2 image mask at the detection frame
        frames_subset: (T, H, W, C) uint8 numpy array (cropped if crop was applied)
        det_frame_idx: int, index into frames_subset for detection frame
        """
        if not self.save_debug_images or self.debug_image_dir is None:
            return
        if sam2_det_mask is None:
            return
        if frames_subset is None or len(frames_subset) == 0:
            return

        try:
            ImageCls, ImageDrawCls, ImageFontCls = self._lazy_import_pillow()
        except RuntimeError:
            return

        dataset_name = getattr(trajectory, "dataset_name", "unknown") or "unknown"
        safe_dataset = self._sanitize_for_path(str(dataset_name))
        safe_traj = self._sanitize_for_path(str(trajectory.name))
        traj_dir = self.debug_image_dir / safe_dataset
        traj_dir.mkdir(parents=True, exist_ok=True)
        file_base = f"{safe_traj}_subtask_{subtask_idx:02d}_sam2"

        font = ImageFontCls.load_default() if ImageFontCls else None

        # --- Row 1: det frame with SAM2 image mask ---
        det_frame_np = self._ensure_rgb(self._to_uint8_image(frames_subset[det_frame_idx]))
        if sam2_det_mask is not None:
            det_with_mask = self._overlay_mask(det_frame_np, sam2_det_mask, color=(0, 200, 100))
        else:
            det_with_mask = det_frame_np
        panel_det = ImageCls.fromarray(det_with_mask)
        draw_det = ImageDrawCls.Draw(panel_det)
        draw_det.text((4, 4), f"det f{det_frame_idx} (SAM2 img mask)", fill=(255, 255, 80), font=font)

        caption_height = 32
        canvas_w = panel_det.width
        canvas_h = caption_height + panel_det.height

        canvas = ImageCls.new("RGB", (canvas_w, canvas_h), color=(15, 15, 15))
        canvas.paste(panel_det, (0, caption_height))
        draw = ImageDrawCls.Draw(canvas)
        instruction = annotation.get("language_instruction") or ""
        caption = f"{dataset_name} | {trajectory.name} | subtask {subtask_idx} | SAM2 masks"
        draw.text((8, 6), caption, fill=(255, 255, 255), font=font)
        if instruction:
            draw.text((8, 22), instruction[:80], fill=(180, 180, 180), font=font)
        png_path = traj_dir / f"{file_base}_masks.png"
        try:
            canvas.save(png_path)
        except Exception as exc:
            logging.warning(f"Failed to save SAM2 mask debug PNG: {exc}")

    def _overlay_mask(self, frame_np, mask_np, color=(0, 200, 100), alpha=0.45):
        """Return a copy of frame_np with mask_np overlaid as a semi-transparent colour."""
        out = frame_np.copy()
        mask_bool = mask_np.astype(bool)
        color_arr = np.array(color, dtype=np.float32)
        out[mask_bool] = np.clip(
            alpha * color_arr + (1.0 - alpha) * out[mask_bool].astype(np.float32),
            0, 255,
        ).astype(np.uint8)
        # Draw a thin contour around the mask for clarity
        try:
            import cv2
            contours, _ = cv2.findContours(
                mask_np.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(out, contours, -1, tuple(int(c) for c in color_arr), 1)
        except Exception:
            pass
        return out


    def _sanitize_for_path(self, value: str) -> str:
        if value is None:
            value = "unknown"
        return sub(r"[^0-9a-zA-Z._-]+", "_", value)
