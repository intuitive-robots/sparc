"""Annotation scoring for trajectory annotation."""

import logging
import math
import os
import time
import numpy as np
from sparc.pipeline.pipeline_types import TrackingResult, FinalBoxes
from sparc.perception.ann_utils import (
    robust_intermediate_boxes,
    select_best_obj_box_tracking,
)
from sparc.llm.llm_utils import verify_annotation_with_vlm
from sparc.scoring.scoring_signals import (
    compute_scoring_signals,
    compute_lightweight_features,
)
from sparc.scoring.selection_scoring import select_target_box_density
from sparc.pipeline.annotation_coordinates import _extract_obj_traces_from_raw_tracks
from sparc.pipeline.annotation_coordinates import _pick_redacted_candidate_idx


def _compute_log_baseline_subtracted_movement(score_breakdown, best_start, best_target):
    """Match the offline movement-score normalization used in score_debug_analysis."""
    if score_breakdown is None or best_start is None or best_target is None:
        return None

    raw_movements = score_breakdown.get("raw_movements")
    if raw_movements is None:
        raw_movements = score_breakdown.get("movements")
    if raw_movements is None:
        return None

    raw_movements = np.asarray(raw_movements)
    if raw_movements.ndim != 2:
        return None
    if best_start >= raw_movements.shape[0] or best_target >= raw_movements.shape[1]:
        return None

    raw_movement = float(raw_movements[best_start, best_target])
    scene_baseline = float(score_breakdown.get("scene_movement_baseline") or 0.0)
    movement_signal = max(raw_movement - scene_baseline, 0.0)
    return min(math.log1p(movement_signal * 100.0) / 5.0, 1.0)


def _compute_confidence_breakdown(det_confidence, tracking_score, score_breakdown,
                                  best_start, best_target, source, rejection_reason=None,
                                  metadata=None, extra_signals=None):
    """Compute per-annotation confidence breakdown from scoring components.

    Args:
        det_confidence: Detection model confidence for the winning box (0-1).
        tracking_score: Composite tracking score from select_best_obj_box_* (0-1).
        score_breakdown: Dict from metadata['score_breakdown'] with component matrices.
        best_start: Index of winning start box.
        best_target: Index of winning target box.
        source: Serialized selection label; "cotracker" identifies the AllTracker path.
        metadata: Full metadata dict from selection (for debug data).

    Returns:
        dict with per-component scores and composite annotation_confidence.
    """
    breakdown = {"source": source, "detection_confidence": det_confidence}

    if score_breakdown is not None and best_start is not None and best_target is not None:
        # Extract per-component scores for the winning (start, target) pair
        movements = score_breakdown.get("movements")
        movement_score = _compute_log_baseline_subtracted_movement(
            score_breakdown, best_start, best_target
        )
        if movement_score is not None:
            breakdown["movement_score"] = movement_score
        elif movements is not None:
            breakdown["movement_score"] = float(movements[best_start, best_target])

        # Read point-count matching, with an IoU fallback for older metadata.
        target_iou = score_breakdown.get("target_iou")
        point_counts = score_breakdown.get("point_counts")
        if target_iou is not None:
            breakdown["target_match_score"] = float(target_iou[best_start, best_target])
        elif point_counts is not None:
            max_pts = point_counts.max() if point_counts.max() > 0 else 1
            breakdown["target_match_score"] = float(point_counts[best_start, best_target] / max_pts)

        temporal = score_breakdown.get("temporal_scores")
        if temporal is not None:
            breakdown["temporal_alignment_score"] = float(temporal[best_start, best_target])

        intermediate = score_breakdown.get("intermediate_scores")
        if intermediate is not None:
            breakdown["intermediate_match_score"] = float(intermediate[best_start])

        nested = score_breakdown.get("nested_penalties")
        if nested is not None:
            breakdown["nested_penalty"] = float(nested[best_start, best_target])

        cont_penalty = score_breakdown.get("continuous_movement_penalties")
        if cont_penalty is not None:
            breakdown["continuous_movement_penalty"] = float(cont_penalty[best_start])

        breakdown["tracking_score"] = float(tracking_score) if tracking_score is not None else None

        # Composite: 25% detection confidence + 75% tracking score
        annotation_confidence = 0.25 * det_confidence + 0.75 * float(tracking_score)
    elif source == "detect_only":
        breakdown["tracking_score"] = None
        annotation_confidence = float(det_confidence)
    else:
        # Fallback path — tracking failed, only detection confidence available
        breakdown["tracking_score"] = None
        annotation_confidence = 0.25 * det_confidence

    if rejection_reason is not None:
        breakdown['rejection_reason'] = rejection_reason
        breakdown['score_redacted'] = True
        if rejection_reason in ('best_match_insufficient_movement', 'best_match_insufficient_points'):
            annotation_confidence *= 0.4
        else:
            annotation_confidence *= 0.2

    breakdown["annotation_confidence"] = round(annotation_confidence, 4)

    if extra_signals:
        breakdown.update(extra_signals)

    # Heavy debug data for offline re-scoring without re-running the pipeline.
    if score_breakdown is not None and best_start is not None and best_target is not None:
        import numpy as np
        debug = {}
        debug["best_start_idx"] = int(best_start)
        debug["best_target_idx"] = int(best_target)

        # Store all available matrices from score_breakdown
        for key in ("point_counts", "raw_movements", "selection_scores", "final_scores",
                    "temporal_scores", "confidence_movement_scores", "selection_movement_scores",
                    "nested_penalties", "continuous_movement_penalties", "target_iou", "movements",
                    "intermediate_scores", "stability"):
            mat = score_breakdown.get(key)
            if mat is not None:
                arr = np.asarray(mat)
                if arr.ndim == 2:
                    debug[key] = [[round(float(v), 6) for v in row] for row in arr]
                elif arr.ndim == 1:
                    debug[key] = [round(float(v), 6) for v in arr]

        # Scalar metadata
        for scalar_key in ("n_points_per_box", "scene_movement_baseline"):
            val = score_breakdown.get(scalar_key)
            if val is not None:
                debug[scalar_key] = float(val) if isinstance(val, (float, int)) else val

        if metadata:
            for mkey in ("movement_norm_scale", "movement_norm_percentile"):
                val = metadata.get(mkey)
                if val is not None:
                    debug[mkey] = float(val)
            raw_mvt_per_box = metadata.get("raw_movement_per_box")
            if raw_mvt_per_box is not None:
                debug["raw_movement_per_box"] = [
                    round(float(v), 6) if v is not None else None for v in raw_mvt_per_box
                ]

        # Top-8 candidates by final score
        final_scores = score_breakdown.get("final_scores")
        if final_scores is not None:
            sel_arr = np.asarray(final_scores)
            flat = sel_arr.flatten()
            k = min(8, len(flat))
            top_k_flat = np.argsort(flat)[::-1][:k]
            top_k = []
            n_targets = sel_arr.shape[1] if sel_arr.ndim == 2 else 1
            for idx in top_k_flat:
                si, ti = divmod(int(idx), n_targets)
                entry = {"start_idx": si, "target_idx": ti,
                         "selection_score": round(float(sel_arr[si, ti]), 6)}
                top_k.append(entry)
            debug["top_k_candidates"] = top_k

        breakdown["debug"] = debug

    return breakdown


def _resolve_emitted_initial_box_confidence(final):
    """Return confidence fields for the final emitted lightweight winner."""
    confidence_breakdown = final.confidence_breakdown
    detection_confidence = final.initial_object_box_confidence
    annotation_confidence = detection_confidence
    selection_score = None
    selection_score_name = None
    if isinstance(confidence_breakdown, dict):
        annotation_confidence = confidence_breakdown.get(
            "annotation_confidence", annotation_confidence
        )
        winner_override = confidence_breakdown.get("selection_winner_override") or {}
        if winner_override.get("new_final_score") is not None:
            selection_score = float(winner_override["new_final_score"])
            selection_score_name = winner_override.get("signal_name")
            confidence_breakdown.setdefault(
                "preliminary_annotation_confidence", annotation_confidence
            )
            confidence_breakdown["annotation_confidence"] = selection_score
            confidence_breakdown["annotation_confidence_signal_name"] = selection_score_name
            annotation_confidence = selection_score
    return (
        annotation_confidence,
        detection_confidence,
        selection_score,
        selection_score_name,
    )


def _augment_tracks_with_targets(raw_tracks_data, target_result):
    """Add target candidate boxes, confidences, and masks to a raw_tracks_data dict.

    Adds 'target_candidate_boxes' (M, 4), 'target_candidate_confidences' (M,),
    and 'target_candidate_masks' (M, H, W) uint8 when available.
    Returns the original dict (possibly modified in-place) or None.
    """
    if not raw_tracks_data:
        return raw_tracks_data
    try:
        pm = target_result.pred_boxes_target_merged
        if pm is not None and len(pm[0].xyxy) > 0:
            raw_tracks_data['target_candidate_boxes'] = np.asarray(pm[0].xyxy, dtype=np.float32)
            raw_tracks_data['target_candidate_confidences'] = np.asarray(pm[0].confidence, dtype=np.float32)
        elif target_result.target_boxes:
            raw_tracks_data['target_candidate_boxes'] = np.asarray(
                [tb["box"] for tb in target_result.target_boxes],
                dtype=np.float32,
            )
            raw_tracks_data['target_candidate_confidences'] = np.asarray(
                [tb["det_confidence"] for tb in target_result.target_boxes],
                dtype=np.float32,
            )
        if target_result.target_masks is not None and len(target_result.target_masks) > 0:
            raw_tracks_data['target_candidate_masks'] = target_result.target_masks
    except Exception:
        pass
    return raw_tracks_data


def _augment_tracks_with_robotseg(raw_tracks_data, robotseg_result):
    """Add RobotSeg gripper masks and their original frame indices to raw_tracks_data."""
    if raw_tracks_data is None or robotseg_result is None:
        return raw_tracks_data
    raw_tracks_data['gripper_video_masks_robotseg'] = (robotseg_result['masks'] > 0.0).astype(np.uint8)
    raw_tracks_data['gripper_video_masks_robotseg_frame_indices'] = robotseg_result['frame_indices']
    return raw_tracks_data


def _augment_tracks_with_tracking_frame_indices(raw_tracks_data, tracking_frame_indices):
    """Add the full-subtask frame index for each tracking timestep."""
    if raw_tracks_data is None or tracking_frame_indices is None:
        return raw_tracks_data
    raw_tracks_data['tracking_frame_indices'] = np.asarray(tracking_frame_indices, dtype=np.int32).reshape(-1)
    return raw_tracks_data


class AnnotationScoring:
    """AnnotationScoring methods operating on the configured annotator state."""

    def _lift_track_groups_to_3d(self, track_groups_2d, frames_np):
        """Lift multiple 2D track groups while sharing each MoGe inference.

        Track groups may use different numbers of candidates and points, but must
        cover the same frames. Dense point maps are sampled for every group while
        each MoGe batch is resident, then discarded to keep memory bounded.
        """
        if self.pointcloud is None:
            return None
        track_groups = [np.asarray(tracks_2d) for tracks_2d in track_groups_2d]
        if not track_groups:
            return []

        T = len(frames_np)
        for tracks_2d in track_groups:
            if tracks_2d.ndim != 4 or tracks_2d.shape[-1] != 2:
                raise ValueError(
                    "Each 2D track group must have shape (candidates, frames, points, 2)"
                )
            if tracks_2d.shape[1] != T:
                raise ValueError(
                    "2D track groups and frames must have the same frame count "
                    f"({tracks_2d.shape[1]} vs {T})"
                )

        H, W = frames_np.shape[1], frames_np.shape[2]
        results = [
            np.full((*tracks_2d.shape[:-1], 3), np.nan, dtype=np.float32)
            for tracks_2d in track_groups
        ]

        # Compute scale so the longest edge does not exceed pointcloud_max_resolution.
        max_res = self.pointcloud_max_resolution
        if max_res is not None and max(H, W) > max_res:
            scale = max_res / max(H, W)
            pH, pW = max(1, round(H * scale)), max(1, round(W * scale))
        else:
            scale = 1.0
            pH, pW = H, W

        _t0_total = time.perf_counter()
        batch_size = self.pointcloud_batch_size
        for batch_start in range(0, T, batch_size):
            batch_end = min(batch_start + batch_size, T)
            batch_frames = frames_np[batch_start:batch_end]  # (B, H, W, 3)
            if scale < 1.0:
                # Resize CPU-side before sending to GPU — avoids int32 element overflow at large batch
                import cv2
                batch_frames = np.stack([
                    cv2.resize(f, (pW, pH), interpolation=cv2.INTER_AREA)
                    for f in batch_frames
                ])
            try:
                _t0 = time.perf_counter()
                pc = self.pointcloud.get_pointcloud_batch(batch_frames, fov_x=self.fov_x, resolution_level=self.pointcloud_resolution_level)
                logging.debug(f"[TIMER] moge_batch: {time.perf_counter() - _t0:.3f}s  frames={batch_start}-{batch_end}/{T}")
            except Exception as e:
                logging.warning(f"MoGe failed for frames {batch_start}-{batch_end}: {e}")
                continue
            points_batch = pc['points']  # (B, pH, pW, 3)
            valid_batch  = pc['mask']    # (B, pH, pW) bool

            for i, t in enumerate(range(batch_start, batch_end)):
                points = points_batch[i]  # (pH, pW, 3)
                valid  = valid_batch[i]   # (pH, pW) bool
                for tracks_2d, result in zip(track_groups, results):
                    xs = np.clip(
                        np.round(tracks_2d[:, t, :, 0] * scale).astype(np.int32),
                        0,
                        pW - 1,
                    )
                    ys = np.clip(
                        np.round(tracks_2d[:, t, :, 1] * scale).astype(np.int32),
                        0,
                        pH - 1,
                    )
                    pts3d = points[ys, xs].copy()
                    is_valid = valid[ys, xs]
                    pts3d[~is_valid] = np.nan
                    result[:, t] = pts3d

        candidate_counts = [int(tracks.shape[0]) for tracks in track_groups]
        logging.info(
            f"[TIMER] moge_3d_lift: {time.perf_counter() - _t0_total:.2f}s  "
            f"frames={T} groups={candidate_counts} res={pH}x{pW}"
        )
        return results


    def _apply_selection_winner(self, tracking_result, lwf: dict, preliminary_winner_idx: int) -> None:
        """Apply the final selection to the preliminary tracking winner.

        Mutates ``tracking_result`` in place. ``lwf`` is the dict returned by
        ``compute_lightweight_features`` and already contains ``winner_pick``.
        Keeps the preliminary winner when no valid final pick is available,
        recording the rejection reason when supplied. Otherwise refreshes
        boxes, masks, and tracks even when the winner index is unchanged.
        """
        pick = lwf.get('winner_pick')
        rtd = tracking_result.raw_tracks_data
        if not pick or rtd is None:
            return

        new_start = pick.get('best_start_idx')
        new_target = pick.get('best_target_idx')
        if new_start is None:
            tracking_result.confidence_breakdown.setdefault('selection_winner_override', {})
            tracking_result.confidence_breakdown['selection_winner_override'] = {
                'applied': False,
                'reason': pick.get('rejection_reason') or 'no_winner_returned',
                'preliminary_winner_idx': int(preliminary_winner_idx),
                'signal_name': pick.get('signal_name'),
            }
            return

        cand_boxes = rtd.get('candidate_boxes')
        cand_confs = rtd.get('candidate_confidences')
        cand_masks = rtd.get('candidate_masks')
        cot_tracks = rtd.get('cotracker_tracks')
        target_boxes = rtd.get('target_candidate_boxes')
        if cand_boxes is None or new_start < 0 or new_start >= len(cand_boxes):
            return

        # Density-based target re-selection (the paper method): pick the target
        # box by track-point density / sqrt(area) at the winning start candidate
        # instead of the plain argmax. Falls back to the winner-pick target index
        # when the density inputs are missing/degenerate.
        argmax_target = new_target
        density_target = None
        tps_ungated = lwf.get('target_point_score_ungated')
        if (
            target_boxes is not None
            and isinstance(tps_ungated, list)
            and 0 <= new_start < len(tps_ungated)
        ):
            density_target = select_target_box_density(
                tps_ungated[new_start],
                target_boxes,
                start_box=cand_boxes[new_start],
            )
        if density_target is not None:
            new_target = density_target

        preliminary_target = None
        preliminary_breakdown = tracking_result.confidence_breakdown.get('debug') or {}
        if 'best_target_idx' in preliminary_breakdown:
            preliminary_target = int(preliminary_breakdown['best_target_idx'])

        unchanged = (
            int(preliminary_winner_idx) == int(new_start)
            and (new_target is None or preliminary_target is None or int(preliminary_target) == int(new_target))
        )

        # Always record the comparison so the JSONL captures both picks.
        final_scores = pick.get('final_score') or []
        new_final_score = (
            float(final_scores[new_start])
            if isinstance(final_scores, list) and 0 <= new_start < len(final_scores)
               and final_scores[new_start] is not None
            else None
        )
        tracking_result.confidence_breakdown['selection_winner_override'] = {
            'applied': not unchanged,
            'preliminary_winner_idx': int(preliminary_winner_idx),
            'new_winner_idx': int(new_start),
            'new_target_idx': int(new_target) if new_target is not None else None,
            'new_final_score': new_final_score,
            'reason': pick.get('rejection_reason'),
            'signal_name': pick.get('signal_name'),
            'target_source': 'density' if density_target is not None else 'argmax',
            'argmax_target_idx': int(argmax_target) if argmax_target is not None else None,
            'density_target_idx': int(density_target) if density_target is not None else None,
        }
        if (
            new_final_score is not None
            and 'preliminary_annotation_confidence' not in tracking_result.confidence_breakdown
        ):
            tracking_result.confidence_breakdown['preliminary_annotation_confidence'] = (
                tracking_result.confidence_breakdown.get('annotation_confidence')
            )

        # Keep boxes, masks, and tracks aligned with the final scoring winner,
        # even when its candidate index matches the preliminary selection.
        tracking_result.best_obj_box = np.asarray(cand_boxes[new_start], dtype=np.float32)[None]
        if target_boxes is not None and new_target is not None and 0 <= new_target < len(target_boxes):
            tracking_result.best_obj_box_target = np.asarray(target_boxes[new_target], dtype=np.float32)[None]
        else:
            tracking_result.best_obj_box_target = None
        if cot_tracks is not None and 0 <= new_start < len(cot_tracks):
            tracking_result.obj_traces = np.asarray(cot_tracks[new_start])
        if cand_masks is not None and 0 <= new_start < len(cand_masks):
            tracking_result.best_obj_mask = np.asarray(cand_masks[new_start], dtype=np.uint8)
        if cand_confs is not None and 0 <= new_start < len(cand_confs):
            # Confidence reported alongside the box: keep using the detector
            # confidence of the new winner (the preliminary field was a tracking
            # score; the selection final_score is in confidence_breakdown).
            tracking_result.best_obj_box_confidence = float(cand_confs[new_start])

        # Intermediate boxes were produced for the preliminary winner before the
        # final selector ran. Rebuild them from the final winner's
        # own point tensor instead of publishing boxes for a different object.
        cot_visibility = rtd.get('cotracker_visibility')
        tracking_frames = rtd.get('tracking_frame_indices')
        image_hw = rtd.get('tracking_coordinate_image_hw')
        if (
            cot_tracks is not None
            and cot_visibility is not None
            and tracking_frames is not None
            and image_hw is not None
            and 0 <= new_start < len(cot_tracks)
        ):
            tracking_frames_arr = np.asarray(tracking_frames, dtype=np.int32).reshape(-1)
            requested_frames = sorted((tracking_result.intermediate_boxes or {}).keys())
            if not requested_frames and len(tracking_frames_arr) > 2:
                sample_indices = np.linspace(
                    0, len(tracking_frames_arr) - 1, 8, dtype=int,
                )[1:-1]
                requested_frames = tracking_frames_arr[sample_indices].tolist()
            rebuilt_boxes, rebuilt_diagnostics = robust_intermediate_boxes(
                cand_boxes[new_start],
                cot_tracks[new_start],
                cot_visibility[new_start],
                tracking_frames_arr,
                requested_frames,
                np.asarray(image_hw, dtype=np.int32).reshape(-1)[:2],
            )
            tracking_result.intermediate_boxes = rebuilt_boxes or None
            tracking_result.confidence_breakdown['selection_winner_override'][
                'intermediate_boxes_recomputed'
            ] = True
            tracking_result.confidence_breakdown['selection_winner_override'][
                'intermediate_box_diagnostics'
            ] = rebuilt_diagnostics
        rtd['winner_idx'] = np.array(new_start, dtype=np.int32)
        for idx, item in enumerate(
            tracking_result.confidence_breakdown.get('all_candidate_breakdowns') or []
        ):
            if isinstance(item, dict):
                item['is_winner'] = idx == int(new_start)


    def _augment_confidence_breakdown_with_replay_signals(
        self,
        tracking_result,
        grasp_phases,
        task_obj_info,
        *,
        arm=None,
        scoring_fps=None,
    ):
        """Attach per-candidate replay signals to a tracking result in-place."""
        if tracking_result.confidence_breakdown is None or tracking_result.raw_tracks_data is None:
            return tracking_result

        rtd = tracking_result.raw_tracks_data
        scoring_fps = float(scoring_fps) if scoring_fps is not None else float(self.fps)
        winner_idx = int(rtd['winner_idx']) if 'winner_idx' in rtd else -1
        cand_boxes = rtd.get('candidate_boxes')
        cand_confs = rtd.get('candidate_confidences')
        if winner_idx < 0 or cand_boxes is None or cand_confs is None:
            return tracking_result

        try:
            extra = compute_scoring_signals(
                raw_tracks_data=rtd,
                candidate_confidences=cand_confs,
                candidate_boxes=cand_boxes,
                grasp_phases=grasp_phases,
                task_obj_info=task_obj_info,
                movement_type=task_obj_info.get("action"),
                fps=scoring_fps,
                winner_idx=winner_idx,
            )
            tracking_result.confidence_breakdown.update(extra)
        except Exception as exc:
            logging.warning(f"compute_scoring_signals failed: {exc}")

        try:
            scene_baseline = float(
                (tracking_result.confidence_breakdown.get('debug') or {})
                .get('scene_movement_baseline') or 0.0
            )
            verbose_lws = os.environ.get('LIGHTWEIGHT_SCORING_VERBOSE', '0') == '1'
            lwf = compute_lightweight_features(
                raw_tracks_data=rtd,
                candidate_boxes=cand_boxes,
                candidate_confidences=cand_confs,
                target_boxes=rtd.get('target_candidate_boxes'),
                grasp_phases=grasp_phases,
                task_obj_info=task_obj_info,
                movement_type=task_obj_info.get("action"),
                detected_arm=arm,
                scene_movement_baseline_pxps=scene_baseline,
                fps=scoring_fps,
                verbose=verbose_lws,
            )
            if lwf:
                tracking_result.confidence_breakdown['lightweight'] = lwf
                self._apply_selection_winner(
                    tracking_result, lwf, winner_idx,
                )
        except Exception as exc:
            logging.warning(f"compute_lightweight_features failed: {exc}")

        return tracking_result


    def _track_and_select(self, frames_torch, det_result, target_result, masks, valid_masks, gripper_mask,
                          task_grasp_phases, trajectory,
                          intermediate_detections=None, det_frame_idx=0, frames_np=None,
                          tracking_scale=(1.0, 1.0), tracking_frame_indices=None):
        """Run AllTracker and select the best object/target box. Returns TrackingResult."""
        import torch
        candidate_boxes = det_result.candidate_boxes
        pred_boxes = det_result.pred_boxes
        target_boxes = target_result.target_boxes

        indices_to_track = np.array([i for i in range(len(candidate_boxes)) if candidate_boxes[i]["confidence"] > 0.0])
        if len(indices_to_track) == 0:
            logging.info(f"Skipping {trajectory.name} because no valid boxes to track after SAM filtering")
            return TrackingResult(
                best_obj_box=None, best_obj_box_confidence=None,
                best_obj_box_target=None, obj_traces=None, verified_target=False,
            )

        masks_per_box = masks[indices_to_track]
        boxes_to_track = np.array([candidate_boxes[i]["box"] for i in indices_to_track], dtype=np.float32)
        candidate_confidences = np.array(
            [candidate_boxes[i]["det_confidence"] for i in indices_to_track],
            dtype=np.float32,
        )

        tracking_boxes = boxes_to_track
        tracking_masks = np.asarray(masks_per_box)
        tracking_coordinate_image_hw = tuple(int(v) for v in tracking_masks.shape[1:3])
        gripper_tracks = None
        gripper_visibility = None
        gripper_box = det_result.gripper_box.copy() if det_result.gripper_box is not None else None
        gripper_confidence = det_result.gripper_box_confidence
        gripper_arm = det_result.arm

        if gripper_box is not None and gripper_box.shape[0] > 0:
            if gripper_mask is None:
                gripper_mask = np.zeros(tracking_masks.shape[1:], dtype=np.uint8)

            tracking_boxes = np.concatenate([tracking_boxes, gripper_box.astype(np.float32)], axis=0)
            tracking_masks = np.concatenate([tracking_masks, gripper_mask[None]], axis=0)

        # Scale boxes and masks into tracking frame resolution if frames were resized
        sx, sy = tracking_scale
        if tracking_scale != (1.0, 1.0):
            import torch.nn.functional as F
            box_scale = np.array([sx, sy, sx, sy], dtype=np.float32)
            tracking_boxes = tracking_boxes * box_scale
            new_H = frames_torch.shape[3]
            new_W = frames_torch.shape[4]
            tracking_masks = F.interpolate(
                torch.from_numpy(tracking_masks).float().unsqueeze(1),
                size=(new_H, new_W),
                mode='nearest',
            ).squeeze(1).byte().numpy()

        pred_tracks_all, pred_visibility_all = self.tracker.run_on_boxes(
            frames_torch, tracking_boxes, pred_masks_first_frame=tracking_masks,
        )

        # Rescale tracks back to original pixel space
        if tracking_scale != (1.0, 1.0) and len(pred_tracks_all) > 0:
            inv_scale = torch.tensor([1.0 / sx, 1.0 / sy], dtype=torch.float32)
            pred_tracks_all = [t * inv_scale for t in pred_tracks_all]

        if gripper_box is not None and len(pred_tracks_all) > len(boxes_to_track):
            gripper_tracks = pred_tracks_all[-1].numpy()
            gripper_visibility = pred_visibility_all[-1].numpy()
            pred_tracks_per_box = pred_tracks_all[:-1]
            pred_visibility_per_box = pred_visibility_all[:-1]
        else:
            pred_tracks_per_box = pred_tracks_all
            pred_visibility_per_box = pred_visibility_all

        target_boxes_to_track = np.array([tb["box"] for tb in target_boxes])
        target_box_confidences = np.array([tb["det_confidence"] for tb in target_boxes])

        best_start, best_target, scores, meta = select_best_obj_box_tracking(
            candidate_boxes=boxes_to_track,
            candidate_target_boxes=target_boxes_to_track,
            frames_torch=frames_torch,
            pred_tracks_per_box=pred_tracks_per_box,
            pred_visibility_per_box=pred_visibility_per_box,
            grasp_phases=task_grasp_phases,
            robot_tracks=None,
            target_box_confidences=target_box_confidences,
            intermediate_detections=intermediate_detections,
            det_frame_idx=det_frame_idx,
            scoring_mode=self.scoring_mode,
            filter_robot=self.filter_robot,
            fps=self.fps,
            track_coordinate_image_shape=tracking_coordinate_image_hw,
            tracking_frame_indices=tracking_frame_indices,
        )

        best_obj_box_target = None
        best_obj_box = None
        best_obj_box_confidence = None
        obj_traces = None
        intermediate_boxes = None

        raw_tracks_data = None
        if len(pred_tracks_per_box) > 0:
            try:
                raw_tracks_data = {
                    'cotracker_tracks': np.stack(
                        [t.numpy() for t in pred_tracks_per_box], axis=0),  # (n,T,N,2)
                    'cotracker_visibility': np.stack(
                        [v.numpy() for v in pred_visibility_per_box], axis=0),  # (n,T,N)
                    'candidate_boxes': boxes_to_track.astype(np.float32),  # (n,4)
                    'candidate_confidences': candidate_confidences,         # (n,)
                    'source_candidate_indices': indices_to_track.astype(np.int32),
                    'tracking_coordinate_image_hw': np.asarray(
                        tracking_coordinate_image_hw, dtype=np.int32,
                    ),
                    # Held in memory only; needed so the lightweight winner
                    # override can pick a different start's segmentation mask.
                    # Stripped before NPZ persistence.
                    'candidate_masks': np.asarray(masks_per_box, dtype=np.uint8),
                }
                if gripper_tracks is not None:
                    raw_tracks_data['gripper_box'] = gripper_box.astype(np.float32)
                    raw_tracks_data['gripper_det_confidence'] = np.array(gripper_confidence or 0.0, dtype=np.float32)
                    raw_tracks_data['gripper_tracks'] = gripper_tracks.astype(np.float32)
                    raw_tracks_data['gripper_visibility'] = gripper_visibility.astype(np.float32)
                    raw_tracks_data['gripper_mask_first_frame'] = gripper_mask.astype(np.uint8) if gripper_mask is not None else None
                    if gripper_arm is not None:
                        raw_tracks_data['gripper_arm'] = np.array(gripper_arm)
                # Lift 2D tracks to 3D using MoGe (only when pointcloud proxy available)
                if self.pointcloud is not None and frames_np is not None:
                    try:
                        track_groups = [raw_tracks_data['cotracker_tracks']]
                        if gripper_tracks is not None:
                            track_groups.append(gripper_tracks[np.newaxis])
                        lifted_groups = self._lift_track_groups_to_3d(
                            track_groups,
                            frames_np,
                        )
                        tracks_3d = lifted_groups[0]
                        if tracks_3d is not None:
                            raw_tracks_data['cotracker_tracks_3d'] = tracks_3d.astype(np.float16)
                        if gripper_tracks is not None:
                            raw_tracks_data['gripper_tracks_3d'] = lifted_groups[1][0].astype(np.float16)
                    except Exception as e:
                        logging.warning(f"3D track lifting failed: {e}")

                if det_result.robot_candidates_all:
                    all_boxes, all_confs, all_fidxs = [], [], []
                    kf_indices = det_result.keyframe_indices or list(range(len(det_result.robot_candidates_all)))
                    for i, candidates in enumerate(det_result.robot_candidates_all):
                        if candidates is not None and len(candidates) > 0:
                            all_boxes.append(candidates.xyxy)
                            all_confs.append(candidates.confidence)
                            all_fidxs.extend([kf_indices[i]] * len(candidates))
                    if all_boxes:
                        all_boxes_np = np.concatenate(all_boxes, axis=0).astype(np.float32)
                        img_width = frames_torch.shape[-1]
                        x_centers = (all_boxes_np[:, 0] + all_boxes_np[:, 2]) / 2.0
                        arm_labels = (x_centers >= img_width / 2.0).astype(np.int8)  # 0=left, 1=right
                        raw_tracks_data['gripper_all_candidate_boxes'] = all_boxes_np
                        raw_tracks_data['gripper_all_candidate_confidences'] = np.concatenate(all_confs, axis=0).astype(np.float32)
                        raw_tracks_data['gripper_all_candidate_frame_indices'] = np.array(all_fidxs, dtype=np.int32)
                        raw_tracks_data['gripper_all_candidate_arms'] = arm_labels  # 0=left, 1=right
            except Exception as e:
                logging.warning(f"Failed to collect cotracker raw tracks: {e}")

        best_obj_mask = None
        confidence_breakdown = None
        if best_start is None:
            fallback_idx = _pick_redacted_candidate_idx(candidate_confidences, scores)
            rejection_reason = meta.get('rejection_reason', 'tracking_verification_failed')
            if fallback_idx is not None and 0 <= fallback_idx < len(boxes_to_track):
                if raw_tracks_data is not None:
                    raw_tracks_data['winner_idx'] = np.array(fallback_idx, dtype=np.int32)
                obj_traces = _extract_obj_traces_from_raw_tracks(raw_tracks_data, fallback_idx)
                best_obj_box = boxes_to_track[fallback_idx][None]
                best_obj_box_target = None
                best_obj_mask = masks_per_box[fallback_idx].astype(np.uint8)
                det_conf = float(candidate_boxes[indices_to_track[fallback_idx]]["confidence"])
                best_obj_box_confidence = 0.0
                confidence_breakdown = _compute_confidence_breakdown(
                    det_conf, None, meta.get('score_breakdown'), None, None, "fallback",
                    rejection_reason=rejection_reason, metadata=meta)
            elif len(pred_boxes[0].xyxy) == 1 and pred_boxes[0].confidence.max() > 0.23:
                best_obj_box = pred_boxes[0].xyxy
                best_obj_box_confidence = float(pred_boxes[0].confidence[0])
                confidence_breakdown = _compute_confidence_breakdown(
                    best_obj_box_confidence, None, None, None, None, "fallback",
                    rejection_reason=rejection_reason)
            elif len(pred_boxes[0][pred_boxes[0].confidence > 0.6].xyxy) == 1:
                best_obj_box = pred_boxes[0][pred_boxes[0].confidence > 0.6].xyxy
                best_obj_box_confidence = float(pred_boxes[0][pred_boxes[0].confidence > 0.6].confidence[0])
                confidence_breakdown = _compute_confidence_breakdown(
                    best_obj_box_confidence, None, None, None, None, "fallback",
                    rejection_reason=rejection_reason)
        else:
            obj_traces = pred_tracks_per_box[best_start].numpy()
            best_obj_box = boxes_to_track[best_start][None]
            best_obj_box_confidence = scores[best_start][best_target]
            best_obj_box_target = target_boxes_to_track[best_target][None] if len(target_boxes_to_track) > 0 else None
            intermediate_boxes = meta.get('intermediate_tracked_boxes')
            best_obj_mask = masks_per_box[best_start].astype(np.uint8)  # (H, W) uint8
            det_conf = float(candidate_boxes[indices_to_track[best_start]]["confidence"])
            confidence_breakdown = _compute_confidence_breakdown(
                det_conf, best_obj_box_confidence, meta.get('score_breakdown'),
                best_start, best_target, "cotracker",
                rejection_reason=meta.get('rejection_reason'),
                metadata=meta)
            if raw_tracks_data is not None:
                raw_tracks_data['winner_idx'] = np.array(best_start, dtype=np.int32)

        return TrackingResult(
            best_obj_box=best_obj_box,
            best_obj_box_confidence=best_obj_box_confidence,
            best_obj_box_target=best_obj_box_target,
            obj_traces=obj_traces,
            verified_target=False,
            intermediate_boxes=intermediate_boxes if intermediate_boxes else None,
            best_obj_mask=best_obj_mask,
            confidence_breakdown=confidence_breakdown,
            raw_tracks_data=raw_tracks_data,
        )


    def _resolve_target_box(self, tracking_result, target_result, frames_subset):
        """Resolve final target box from tracking result and target detections."""
        best_obj_box_target = tracking_result.best_obj_box_target
        if best_obj_box_target is None:
            best_obj_box_target = np.empty((0, 4))

        verified_target = False

        if best_obj_box_target is not None and best_obj_box_target.shape[0] > 0:
            verified_target = True

        if best_obj_box_target.shape[0] == 0:
            pred_boxes_target = target_result.pred_boxes_target_merged
            if pred_boxes_target is not None and len(pred_boxes_target[0].xyxy) > 0:
                max_conf_idx = np.argmax(pred_boxes_target[0].confidence)
                best_obj_box_target = pred_boxes_target[0].xyxy[[max_conf_idx]]
                verified_target = False

        return TrackingResult(
            best_obj_box=tracking_result.best_obj_box,
            best_obj_box_confidence=tracking_result.best_obj_box_confidence,
            best_obj_box_target=best_obj_box_target,
            obj_traces=tracking_result.obj_traces,
            verified_target=verified_target,
            intermediate_boxes=tracking_result.intermediate_boxes,
            best_obj_mask=tracking_result.best_obj_mask,
            confidence_breakdown=tracking_result.confidence_breakdown,
        )


    def _run_single_verification(self, mode, frames_subset, box_initial, box_target,
                                   task_obj_info, trajectory, vlm_kwargs):
        """Run a single VLM verification call. Returns (result_dict, effective_mode)."""
        effective_mode = mode
        if mode == "spatial_consistency" and box_target is None:
            effective_mode = "highlighted_box"

        frame_start = frames_subset[0]
        frame_end = frames_subset[-1] if effective_mode == "spatial_consistency" else frames_subset[0]

        result = verify_annotation_with_vlm(
            mode=effective_mode,
            frame_start=frame_start,
            frame_end=frame_end,
            box_initial=box_initial,
            box_target=box_target if box_target else box_initial,
            object_name=task_obj_info.get("object", "object"),
            instruction=trajectory.lang_ann,
            task_obj_info=task_obj_info,
            trajectory_name=trajectory.name,
            **vlm_kwargs,
        )
        return result, effective_mode


    def _run_cross_model_verification(self, final, frames_subset, start_frame, end_frame,
                                       task_obj_info, trajectory):
        """Run cross-model VLM verification if configured. Updates confidence_breakdown in-place."""
        mode = self.verification_config.get("mode", "off")
        if mode == "off" or final.initial_object_box is None:
            return final

        if final.initial_object_box.size == 0:
            return final

        _t0 = time.perf_counter()

        vlm_kwargs = {
            "model": self.vllm_config.get("model_name", "qwen3-vl-30b"),
            "base_url": self.vllm_config.get("base_url"),
            "temperature": self.vllm_config.get("temperature", 1.0),
            "extra_body": self.vllm_config.get("extra_body"),
        }

        box_initial = final.initial_object_box[0].tolist() if final.initial_object_box.ndim == 2 else final.initial_object_box.tolist()

        box_target = None
        if final.target_object_box is not None and final.target_object_box.size > 0:
            box_target = final.target_object_box[0].tolist() if final.target_object_box.ndim == 2 else final.target_object_box.tolist()

        verification_weight = self.verification_config.get("weight", 0.15)
        hard_gate = self.verification_config.get("hard_gate", False)

        if mode == "dual":
            # Run both highlighted_box and spatial_consistency, take the minimum score
            result_h, _ = self._run_single_verification(
                "highlighted_box", frames_subset, box_initial, box_target,
                task_obj_info, trajectory, vlm_kwargs)
            result_s, _ = self._run_single_verification(
                "spatial_consistency", frames_subset, box_initial, box_target,
                task_obj_info, trajectory, vlm_kwargs)

            verification_score = min(result_h["verification_score"], result_s["verification_score"])
            vlm_answer = f"h={result_h['vlm_answer']},s={result_s['vlm_answer']}"
            effective_mode = "dual"

            logging.info(
                f"[TIMER] cross_model_verify (dual): {time.perf_counter() - _t0:.2f}s "
                f"traj={trajectory.name} highlighted={result_h['vlm_answer']} "
                f"spatial={result_s['vlm_answer']} combined={verification_score:.2f}"
            )
        else:
            result, effective_mode = self._run_single_verification(
                mode, frames_subset, box_initial, box_target,
                task_obj_info, trajectory, vlm_kwargs)
            verification_score = result["verification_score"]
            vlm_answer = result["vlm_answer"]

            logging.info(
                f"[TIMER] cross_model_verify ({effective_mode}): {time.perf_counter() - _t0:.2f}s "
                f"traj={trajectory.name} answer={vlm_answer}"
            )

        # Integrate verification score into confidence_breakdown
        breakdown = final.confidence_breakdown or {}
        breakdown["verification_mode"] = effective_mode
        breakdown["verification_answer"] = vlm_answer
        breakdown["verification_score"] = verification_score

        # Recompute annotation_confidence with verification weight
        old_confidence = breakdown.get("annotation_confidence", final.initial_object_box_confidence or 0.0)

        if hard_gate and verification_score < 0.5:
            # Hard reject: VLM said "no" or "uncertain" → force confidence to 0
            breakdown["annotation_confidence"] = 0.0
        else:
            adjusted = old_confidence * (1.0 - verification_weight) + verification_score * verification_weight
            breakdown["annotation_confidence"] = round(adjusted, 4)

        # Update FinalBoxes with new breakdown
        return FinalBoxes(
            initial_object_box=final.initial_object_box,
            initial_object_box_confidence=final.initial_object_box_confidence,
            target_object_box=final.target_object_box,
            target_object_box_confidence=final.target_object_box_confidence,
            obj_traces=final.obj_traces,
            obj_traces_target=final.obj_traces_target,
            verified_target=final.verified_target,
            tool_box=final.tool_box,
            tool_box_confidence=final.tool_box_confidence,
            tool_traces=final.tool_traces,
            arm=final.arm,
            intermediate_boxes=final.intermediate_boxes,
            target_candidate_box_confidences=final.target_candidate_box_confidences,
            confidence_breakdown=breakdown,
        )
