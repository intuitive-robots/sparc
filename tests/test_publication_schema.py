import json
import queue

import h5py
import numpy as np
import pytest

from sparc.pipeline.annotation_saver import AnnotationSaver
from sparc.pipeline.publication_schema import (
    SCHEMA_NAME,
    SCHEMA_VERSION,
    build_publication_annotation,
    load_publication_arrays,
    trajectory_id_from_annotation,
)
from sparc.scoring.selection_scoring import decode_arrays_blob, encode_arrays_blob


def _pipeline_mask_payload(fill_column=0):
    object_masks = np.zeros((4, 3, 3), dtype=np.uint8)
    robot_masks = np.zeros((4, 3, 3), dtype=np.uint8)
    object_masks[:, :, fill_column] = 1
    robot_masks[:, :, -1] = 1
    names = np.asarray(["detection", "grasp", "interact", "release"])
    frames = np.asarray([12, 14, 18, 22], dtype=np.int32)
    return {
        "centroid_traces": np.asarray([
            [[0.0, 0.0], [1.0, 2.0], [2.0, 2.0]],
            [[1.0, 1.0], [1.5, 2.0], [2.0, 3.0]],
        ], dtype=np.float32),
        "gripper_rep_tracking_frame_indices": np.asarray([2, 4, 6], dtype=np.int32),
        "winner_object_phase_keyframe_masks": object_masks,
        "winner_object_phase_keyframe_absolute_frame_indices": frames,
        "winner_object_phase_keyframe_phase_names": names,
        "robotseg_phase_keyframe_masks": robot_masks,
        "robotseg_phase_keyframe_absolute_frame_indices": frames,
        "robotseg_phase_keyframe_phase_names": names,
        # This internal diagnostic must not leak into the publication blob.
        "candidate_masks": np.ones((2, 3, 3), dtype=np.uint8),
    }


def _pipeline_annotation():
    arrays = _pipeline_mask_payload()
    return {
        "trajectory_name": "chunk-000/episode_000123",
        "source_uuid": "raw/bridge/source/traj0",
        "source_episode_index": 123,
        "dataset_name": "bridge_lerobot",
        "subtask_index": 2,
        "language_instruction": "put the towel into the basket",
        "start_frame": 10,
        "end_frame": 24,
        "num_frames": 40,
        "detection_frame_index": 2,
        "initial_object_box": [[1, 2, 3, 4]],
        "initial_object_box_confidence": 0.61,
        "initial_object_box_detection_confidence": 0.83,
        "initial_object_box_selection_score": 0.61,
        "initial_object_box_selection_score_name": "adaptive_det_soft_snr_sp8",
        "target_object_box": [[5, 2, 8, 6]],
        "target_object_box_confidence": 0.72,
        "verified_target": True,
        "obj_traces_cotracker_point": [[3.0, 3.0], [4.0, 4.0], [5.0, 5.0]],
        "obj_traces_cotracker_point_meta": {
            "frame_indices_relative_to_window": [2, 4, 6],
            "point_index": 9,
        },
        "movement_type": "place",
        "task_obj_info": {
            "action": "place",
            "object": "towel",
            "start_location": "table",
            "target_location": "basket",
        },
        "phase_annotations": [
            {"phase": name, "start_frame": frame - 2, "end_frame": frame}
            for name, frame in zip(("grasp", "interact", "release"), (14, 18, 22))
        ],
        "confidence_breakdown": {
            "detection_confidence": 0.83,
            "selection_winner_override": {
                "new_winner_idx": 1,
                "new_target_idx": 0,
                "new_final_score": 0.61,
                "signal_name": "adaptive_det_soft_snr_sp8",
                "target_source": "density",
            },
            "lightweight": {
                "arrays_blob": encode_arrays_blob(arrays),
                "candidate_boxes": [[0, 0, 2, 2], [1, 2, 3, 4]],
                "candidate_confidences": [0.95, 0.83],
                "source_candidate_idx": [4, 7],
                "target_candidate_boxes": [[5, 2, 8, 6], [7, 4, 9, 7]],
                "target_candidate_confidences": [0.2, 0.9],
                "final_score": [0.2, 0.61],
                "winner_pick": {
                    "best_start_idx": 1,
                    "best_target_idx": 0,
                    "signal_name": "adaptive_det_soft_snr_sp8",
                    "constants": {"depth_relief": 0.06},
                },
                "per_candidate_score_breakdown": [
                    {"candidate_idx": 0, "final_score": 0.2},
                    {
                        "candidate_idx": 1,
                        "source_candidate_idx": 7,
                        "box": [1, 2, 3, 4],
                        "det_confidence": 0.83,
                        "final_score": 0.61,
                        "phase_snr_a0p6_b0p2": 1.5,
                        "keep": True,
                        "components": {"detection": 0.3, "motion": 0.31},
                    },
                ],
                "phase_keyframe_mask_metadata": {
                    "spatial_coordinate_frame": (
                        "segmentation_processing_frame_after_global_resize_and_optional_crop"
                    ),
                    "transform_mask_pixels_to_original_frame": {
                        "crop_xyxy_in_globally_resized_frame": [1, 1, 4, 4],
                        "global_scale_xy": [0.5, 0.5],
                    },
                },
            },
        },
        "_publication_context": {
            "split": "train",
            "fps": 10.0,
            "image_size_hw": [8, 10],
            "camera_view": "observation.images.left_external",
            "source_dataset_format": "lerobot_v3",
            "source_episode_frame_indices": list(range(40)),
            "frame_timestamps_seconds": [index / 10.0 for index in range(40)],
        },
    }


def test_builds_compact_publication_schema_from_pipeline_state():
    converted = build_publication_annotation(_pipeline_annotation())

    assert converted["schema"] == {"name": SCHEMA_NAME, "version": SCHEMA_VERSION}
    assert converted["annotation_id"].startswith("sha256:")
    assert converted["source"]["trajectory_id"] == "chunk-000/episode_000123"
    assert converted["source"]["source_uuid"] == "raw/bridge/source/traj0"
    assert converted["source"]["episode_index"] == 123
    assert converted["source"]["camera_view"] == "observation.images.left_external"
    assert converted["source"]["dataset_format"] == "lerobot_v3"
    assert converted["source"]["frame_mapping"]["source_episode_frame_indices_array"] == (
        "trajectory_source_episode_frame_indices"
    )
    assert converted["window"]["image_size_hw"] == [8, 10]
    assert converted["object"]["initial"] == {
        "frame_index": 12,
        "box_xyxy_pixels": [1.0, 2.0, 3.0, 4.0],
        "detector_score": 0.83,
    }
    assert converted["object"]["track"] == {
        "array_reference": {
            "xy_pixels_array": "centroid_traces",
            "candidate_index": 1,
            "frame_indices_array": "centroid_trace_frame_indices",
            "validity": "finite_xy",
        },
        "point_selection": {"method": "visible_point_centroid"},
    }
    assert converted["robot"]["segmentation"] == {
        "source": "RobotSeg",
        "category_prompt": "robot",
        "threshold": "logit_gt_0",
        "postprocessing": "none",
        "detector_fallback_used": False,
    }
    assert converted["selection"]["score"] == 0.61
    assert converted["selection"]["target_selection_method"] == "density"
    assert converted["selection"]["breakdown"] == {
        "score_components": {"detection": 0.3, "motion": 0.31},
        "signals": {"phase_snr_a0p6_b0p2": 1.5, "keep": True},
    }
    assert converted["baselines"]["detector"] == {
        "method": "maximum_detector_score",
        "initial": {
            "frame_index": 12,
            "box_xyxy_pixels": [0.0, 0.0, 2.0, 2.0],
            "detector_score": 0.95,
            "tracked_candidate_index": 0,
            "source_detection_index": 4,
        },
        "target": {
            "frame_index": 23,
            "box_xyxy_pixels": [7.0, 4.0, 9.0, 7.0],
            "detector_score": 0.9,
            "candidate_index": 1,
        },
        "track": {
            "array_reference": {
                "xy_pixels_array": "centroid_traces",
                "candidate_index": 0,
                "frame_indices_array": "centroid_trace_frame_indices",
                "validity": "finite_xy",
            },
            "point_selection": {"method": "visible_point_centroid"},
        },
    }
    assert "trajectory_name" not in converted
    assert "confidence_breakdown" not in converted
    assert "gripper_box" not in converted

    arrays = decode_arrays_blob(converted["arrays"]["blob"])
    assert set(arrays) == {
        "centroid_traces", "centroid_trace_frame_indices",
        "object_masks", "object_mask_valid", "robot_masks", "robot_mask_valid",
        "phase_mask_frame_indices",
        "detection_object_mask", "detection_object_mask_valid",
        "detection_robot_mask", "detection_robot_mask_valid",
        "detection_frame_index",
        "trajectory_source_episode_frame_indices", "trajectory_timestamps_seconds",
    }
    assert arrays["object_masks"].shape == (3, 8, 10)
    assert arrays["robot_masks"].shape == (3, 8, 10)
    assert arrays["object_mask_valid"].tolist() == [True, True, True]
    assert arrays["robot_mask_valid"].tolist() == [True, True, True]
    assert bool(arrays["detection_object_mask_valid"])
    assert bool(arrays["detection_robot_mask_valid"])
    assert int(arrays["detection_frame_index"]) == 12
    assert arrays["detection_object_mask"].shape == (8, 10)
    assert arrays["phase_mask_frame_indices"].tolist() == [14, 18, 22]
    assert arrays["trajectory_source_episode_frame_indices"].tolist() == list(range(40))
    assert "gripper_masks" not in arrays


def test_publication_build_is_stable():
    converted = build_publication_annotation(_pipeline_annotation())
    assert converted == build_publication_annotation(_pipeline_annotation())
    assert trajectory_id_from_annotation(converted) == "chunk-000/episode_000123"
    assert converted["annotation_id"] == build_publication_annotation(
        _pipeline_annotation()
    )["annotation_id"]


def test_robotseg_fallback_is_explicit_in_publication_provenance():
    state = _pipeline_annotation()
    arrays = decode_arrays_blob(
        state["confidence_breakdown"]["lightweight"]["arrays_blob"]
    )
    arrays.pop("robotseg_phase_keyframe_masks")
    arrays.pop("robotseg_phase_keyframe_absolute_frame_indices")
    arrays.pop("robotseg_phase_keyframe_phase_names")
    state["confidence_breakdown"]["lightweight"]["arrays_blob"] = (
        encode_arrays_blob(arrays)
    )
    state["_publication_context"]["robotseg_enabled"] = True

    converted = build_publication_annotation(state)

    assert converted["robot"]["segmentation"] == {
        "source": "detector_plus_SAM2_fallback",
        "detector_fallback_used": True,
    }


def test_camera_view_is_part_of_annotation_identity():
    left = _pipeline_annotation()
    right = _pipeline_annotation()
    right["_publication_context"]["camera_view"] = (
        "observation.images.right_external"
    )

    assert (
        build_publication_annotation(left)["annotation_id"]
        != build_publication_annotation(right)["annotation_id"]
    )


def test_explicit_source_episode_index_overrides_chunk_local_trajectory_suffix():
    state = _pipeline_annotation()
    state["trajectory_name"] = "libero/1/102"
    state["source_episode_index"] = 1102

    converted = build_publication_annotation(state)

    assert converted["source"]["trajectory_id"] == "libero/1/102"
    assert converted["source"]["source_uuid"] == "raw/bridge/source/traj0"
    assert converted["source"]["episode_index"] == 1102


def test_agibot_source_identity_preserves_episode_slice_mapping():
    state = _pipeline_annotation()
    state.update({
        "trajectory_name": "task_454/0/0/3",
        "dataset_name": "agibot_world",
        "source_uuid": None,
        "source_episode_id": "688896",
        "source_episode_index": 0,
        "source_subtrajectory_index": 3,
        "source_episode_frame_start": 140,
        "source_episode_frame_end_exclusive": 570,
    })

    converted = build_publication_annotation(state)

    assert converted["source"] == {
        "dataset_id": "agibot_world",
        "dataset_format": "lerobot_v3",
        "trajectory_id": "task_454/0/0/3",
        "source_episode_id": "688896",
        "episode_index": 0,
        "subtrajectory_index": 3,
        "episode_frame_start": 140,
        "episode_frame_end_exclusive": 570,
        "subtask_index": 2,
        "split": "train",
        "fps": 10.0,
        "camera_view": "observation.images.left_external",
        "frame_mapping": {
            "trajectory_frame_index": (
                "array index on trajectory_source_episode_frame_indices and "
                "trajectory_timestamps_seconds"
            ),
            "source_episode_frame_indices_array": (
                "trajectory_source_episode_frame_indices"
            ),
            "timestamps_seconds_array": "trajectory_timestamps_seconds",
            "timestamp_origin": "source_episode_start",
        },
    }


def test_missing_camera_view_is_not_guessed_from_media_path():
    state = _pipeline_annotation()
    state["_publication_context"].pop("camera_view")
    state["media_dir"] = (
        "/dataset/videos/observation.images.right_external/chunk-000/file_000.mp4"
    )

    assert "camera_view" not in build_publication_annotation(state)["source"]


def test_missing_image_size_is_rejected():
    state = _pipeline_annotation()
    state["_publication_context"].pop("image_size_hw")

    with pytest.raises(ValueError, match="original image size"):
        build_publication_annotation(state)


def test_unknown_mask_transform_is_not_guessed():
    state = _pipeline_annotation()
    state["confidence_breakdown"]["lightweight"].pop(
        "phase_keyframe_mask_metadata"
    )

    with pytest.raises(ValueError, match="missing transform"):
        build_publication_annotation(state)



def test_bimanual_masks_use_arm_axis_without_duplicate_top_level_object():
    state = _pipeline_annotation()
    left = {
        "arm": "left",
        "subtask_index": 2,
        "object": "towel",
        "detection_frame_index": 2,
        "initial_object_box": [[1, 2, 3, 4]],
        "initial_object_box_detection_confidence": 0.83,
        "target_object_box": [[5, 2, 8, 6]],
        "confidence_breakdown": state["confidence_breakdown"],
    }
    right = {
        **left,
        "arm": "right",
        "subtask_index": 3,
        "object": "basket",
        "confidence_breakdown": {
            **state["confidence_breakdown"],
            "lightweight": {
                **state["confidence_breakdown"]["lightweight"],
                "arrays_blob": encode_arrays_blob(_pipeline_mask_payload(fill_column=2)),
            },
        },
    }
    state["is_bimanual"] = True
    state["arm_annotations"] = [left, right]

    converted = build_publication_annotation(state)
    arrays = decode_arrays_blob(converted["arrays"]["blob"])

    assert "object" not in converted
    assert "selection" not in converted
    assert [arm["arm"] for arm in converted["arms"]] == ["left", "right"]
    assert arrays["object_masks"].shape == (2, 3, 8, 10)
    assert arrays["object_mask_valid"].shape == (2, 3)
    assert arrays["robot_masks"].shape == (3, 8, 10)
    assert arrays["detection_robot_mask"].shape == (2, 8, 10)
    assert arrays["detection_frame_indices"].tolist() == [12, 12]


def test_resume_reader_accepts_v2_rows(tmp_path):
    annotation = build_publication_annotation(_pipeline_annotation())
    output_path = tmp_path / "annotations.jsonl"
    output_path.write_text(json.dumps(annotation) + "\n", encoding="utf-8")

    saver = AnnotationSaver(str(output_path), write_queue=None)
    assert saver.get_processed_trajectories() == {"chunk-000/episode_000123"}


def test_writer_externalizes_public_arrays_into_shared_shard(tmp_path):
    output_path = tmp_path / "annotations.jsonl"
    write_queue = queue.Queue()
    first = build_publication_annotation(_pipeline_annotation())
    second_state = _pipeline_annotation()
    second_state["trajectory_name"] = "chunk-000/episode_000124"
    second = build_publication_annotation(second_state)
    write_queue.put(first)
    write_queue.put(second)
    write_queue.put(None)

    saver = AnnotationSaver(
        str(output_path),
        write_queue,
        externalize_arrays=True,
        array_shard_target_bytes=1 << 30,
    )
    saver.start_writer_process()

    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert len(rows) == 2
    assert all("blob" not in row["arrays"] for row in rows)
    assert rows[0]["arrays"]["storage"]["path"] == rows[1]["arrays"]["storage"]["path"]
    assert rows[0]["arrays"]["storage"]["path"].endswith(
        "annotation-artifacts-000000.h5"
    )
    first_group = rows[0]["arrays"]["storage"]["group"]
    second_group = rows[1]["arrays"]["storage"]["group"]
    assert first_group == (
        "trajectories/chunk-000%2Fepisode_000123/subtasks/0002/"
        + rows[0]["annotation_id"].removeprefix("sha256:")
    )
    assert second_group == (
        "trajectories/chunk-000%2Fepisode_000124/subtasks/0002/"
        + rows[1]["annotation_id"].removeprefix("sha256:")
    )
    assert "array_keys" not in rows[0]["arrays"]["storage"]
    shard_path = tmp_path / rows[0]["arrays"]["storage"]["path"]
    with h5py.File(shard_path, "r") as shard:
        assert shard.attrs["format"] == "sparc_publication_array_shard"
        assert shard.attrs["version"] == 2
        assert shard.attrs["group_layout"] == "trajectory_subtask_annotation"
        group = shard[first_group]
        assert group.attrs["annotation_id"] == rows[0]["annotation_id"]
        assert group.attrs["trajectory_id"] == "chunk-000/episode_000123"
        assert group.attrs["subtask_index"] == 2
        assert group.attrs["dataset_id"] == "bridge_lerobot"
        assert group.attrs["camera_view"] == "observation.images.left_external"
        assert group.attrs["start_frame"] == 10
        assert group.attrs["end_frame_exclusive"] == 24
    loaded = load_publication_arrays(rows[0], sidecar_root=tmp_path)
    assert loaded["object_masks"].shape == (3, 8, 10)
    assert loaded["robot_mask_valid"].tolist() == [True, True, True]


def test_writer_rotates_descriptively_named_hdf5_shards(tmp_path):
    output_path = tmp_path / "annotations.jsonl"
    write_queue = queue.Queue()
    first = build_publication_annotation(_pipeline_annotation())
    first_arrays = decode_arrays_blob(first["arrays"]["blob"])
    first_arrays["keyframe_phase_names"] = np.asarray(
        ["grasp", "interact", "release"],
    )
    first["arrays"]["blob"] = encode_arrays_blob(first_arrays)
    second_state = _pipeline_annotation()
    second_state["trajectory_name"] = "chunk-000/episode_000124"
    write_queue.put(first)
    write_queue.put(build_publication_annotation(second_state))
    write_queue.put(None)

    saver = AnnotationSaver(
        str(output_path),
        write_queue,
        externalize_arrays=True,
        array_shard_target_bytes=1,
    )
    saver.start_writer_process()

    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    paths = [row["arrays"]["storage"]["path"] for row in rows]
    assert paths == [
        "annotations_artifacts/annotation-artifacts-000000.h5",
        "annotations_artifacts/annotation-artifacts-000001.h5",
    ]
    for index, row in enumerate(rows):
        loaded = load_publication_arrays(row, sidecar_root=tmp_path)
        assert loaded["object_masks"].shape == (3, 8, 10)
        if index == 0:
            assert loaded["keyframe_phase_names"].tolist() == [
                "grasp", "interact", "release",
            ]


@pytest.mark.parametrize("save_pointclouds", [False, True])
def test_publication_pointcloud_export_is_opt_in(save_pointclouds):
    state = _pipeline_annotation()
    lightweight = state["confidence_breakdown"]["lightweight"]
    arrays = decode_arrays_blob(lightweight["arrays_blob"])
    arrays.update({
        "winner_object_phase_keyframe_points_xyz": np.asarray(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32,
        ),
        "winner_object_phase_keyframe_points_uv": np.asarray(
            [[2, 3], [4, 5]], dtype=np.int32,
        ),
        "winner_object_phase_keyframe_point_offsets": np.asarray(
            [0, 1, 1, 2], dtype=np.int64,
        ),
        "winner_object_phase_keyframe_point_absolute_frame_indices": np.asarray(
            [14, 18, 22], dtype=np.int32,
        ),
        "winner_object_phase_keyframe_point_phase_names": np.asarray(
            ["grasp", "interact", "release"],
        ),
        "robotseg_phase_keyframe_points_xyz": np.asarray(
            [[0.0, 0.0, 1.0]], dtype=np.float32,
        ),
        "robotseg_phase_keyframe_points_uv": np.asarray([[7, 6]], dtype=np.int32),
        "robotseg_phase_keyframe_point_offsets": np.asarray(
            [0, 0, 1, 1], dtype=np.int64,
        ),
        "robotseg_phase_keyframe_point_absolute_frame_indices": np.asarray(
            [14, 18, 22], dtype=np.int32,
        ),
        "robotseg_phase_keyframe_point_phase_names": np.asarray(
            ["grasp", "interact", "release"],
        ),
    })
    lightweight["arrays_blob"] = encode_arrays_blob(arrays)

    converted = build_publication_annotation(state, save_pointclouds=save_pointclouds)
    public_arrays = decode_arrays_blob(converted["arrays"]["blob"])

    if not save_pointclouds:
        assert not any("keyframe_point" in key for key in public_arrays)
        assert not converted.get("geometry")
        assert "object_masks" in public_arrays
        return

    assert public_arrays["object_keyframe_points_xyz"].shape == (2, 3)
    assert public_arrays["object_keyframe_point_offsets"].tolist() == [0, 1, 1, 2]
    assert public_arrays["gripper_keyframe_points_uv"].tolist() == [[7, 6]]
    assert public_arrays["gripper_keyframe_point_phase_names"].tolist() == [
        "grasp", "interact", "release",
    ]


@pytest.mark.parametrize("schema", [None, {"name": "robog.annotation", "version": "2.0"},
                                    {"name": SCHEMA_NAME, "version": "1.0"}])
def test_old_formats_are_rejected_at_persisted_annotation_boundaries(tmp_path, schema):
    from annotate import get_all_processed_trajectories, merge_node_files
    from sparc.pipeline.publication_schema import is_publication_annotation

    row = {"trajectory_name": "old-trajectory", "subtask_index": 0}
    if schema is not None:
        row = {"schema": schema, "source": {"trajectory_id": "old-trajectory", "subtask_index": 0}}
    assert not is_publication_annotation(row)
    with pytest.raises(ValueError, match="old annotation formats"):
        load_publication_arrays(row)

    output = tmp_path / "annotations.jsonl"
    output.write_text(json.dumps(row) + "\n", encoding="utf-8")
    saver = AnnotationSaver(str(output), queue.Queue())
    with pytest.raises(ValueError, match="old annotation formats"):
        saver.get_processed_trajectories()
    with pytest.raises(ValueError, match="old annotation formats"):
        get_all_processed_trajectories(str(output))
    with pytest.raises(ValueError, match="old annotation formats"):
        saver.save(row)
    assert saver.write_queue.empty()

    output.unlink()
    node = tmp_path / "annotations_node0.jsonl"
    node.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="old annotation formats"):
        merge_node_files(str(output))
    assert output.read_text() == ""


@pytest.mark.parametrize("externalize_arrays", [False, True])
def test_writer_rejects_old_rows_even_when_put_directly_on_queue(tmp_path, externalize_arrays):
    write_queue = queue.Queue()
    write_queue.put({"trajectory_name": "old-trajectory"})
    write_queue.put(None)
    output = tmp_path / "annotations.jsonl"
    saver = AnnotationSaver(str(output), write_queue, externalize_arrays=externalize_arrays)
    with pytest.raises(ValueError, match="old annotation formats"):
        saver.start_writer_process()
    assert not output.exists()


def test_builder_requires_pipeline_state_instead_of_migrating_stored_rows():
    with pytest.raises(ValueError, match="current pipeline state"):
        build_publication_annotation({"trajectory_name": "old-trajectory"})
    annotation = build_publication_annotation(_pipeline_annotation())
    with pytest.raises(ValueError, match="current pipeline state"):
        build_publication_annotation(annotation)


@pytest.mark.parametrize("prefix", ["", "sample/"])
@pytest.mark.parametrize("requested", [None, ["points"]])
def test_current_publication_npz_arrays_round_trip(tmp_path, prefix, requested):
    annotation = build_publication_annotation(_pipeline_annotation())
    expected = {
        "points": np.arange(12, dtype=np.float32).reshape(4, 3),
        "frame_indices": np.array([10, 12, 14, 16], dtype=np.int32),
    }
    stored = {prefix + key: value for key, value in expected.items()}
    if prefix:
        stored["another-sample/points"] = np.zeros((1, 3), dtype=np.float32)
    np.savez_compressed(tmp_path / "arrays.npz", **stored)
    annotation["arrays"] = {
        "storage": {"path": "arrays.npz", "key_prefix": prefix, "array_keys": requested},
    }
    loaded = load_publication_arrays(annotation, sidecar_root=tmp_path)
    assert loaded.keys() == (expected.keys() if requested is None else set(requested))
    for key, array in loaded.items():
        np.testing.assert_array_equal(array, expected[key])
        assert array.dtype == expected[key].dtype


def test_npz_raw_tracks_export_preserves_frame_and_coordinate_metadata(tmp_path):
    from sparc.pipeline.pipeline_types import CropInfo
    from sparc.pipeline.trajectory_annotator import TrajectoryAnnotator

    annotator = TrajectoryAnnotator.__new__(TrajectoryAnnotator)
    annotator.tracks_dir = str(tmp_path)
    crop = CropInfo(True, (10, 20, 30, 40), (10, 20, 10, 20), None, None, None, None)
    tracks = np.array([[[[1.0, 2.0]], [[3.0, 4.0]]]], dtype=np.float32)
    path = annotator._save_tracks_npz(
        "episode/1", 0,
        {"cotracker_tracks": tracks, "cotracker_visibility": np.ones((1, 2, 1)),
         "tracking_frame_indices": np.array([2, 4]), "candidate_masks": np.ones((1, 3, 3))},
        crop_info=crop, global_scale=(0.5, 0.5), start_frame=10,
    )
    assert path is not None
    with np.load(path, allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["cotracker_tracks"], (tracks + [10, 20]) * 2)
        np.testing.assert_array_equal(arrays["tracking_frame_indices"], [2, 4])
        assert int(arrays["start_frame"]) == 10
        assert "candidate_masks" not in arrays


def test_current_node_outputs_merge_and_resume(tmp_path):
    from annotate import get_all_processed_trajectories, merge_node_files

    row = build_publication_annotation(_pipeline_annotation())
    output = tmp_path / "annotations.jsonl"
    output.with_name("annotations_node0.jsonl").write_text(json.dumps(row) + "\n")
    merge_node_files(str(output))
    merge_node_files(str(output))
    assert [json.loads(line) for line in output.read_text().splitlines()] == [row]
    assert get_all_processed_trajectories(str(output)) == {row["source"]["trajectory_id"]}


def test_publication_manifest_matches_stored_array_dtypes_and_shapes():
    annotation = build_publication_annotation(_pipeline_annotation())
    arrays = load_publication_arrays(annotation)
    for key, spec in annotation['arrays']['manifest'].items():
        if isinstance(spec, dict) and 'shape' in spec:
            assert str(arrays[key].dtype) == spec['dtype'], key
            assert list(arrays[key].shape) == spec['shape'], key
    assert arrays['centroid_traces'].dtype == np.float16


@pytest.mark.parametrize('externalize', [False, True])
def test_deferred_arrays_preserve_storage_values(tmp_path, monkeypatch, externalize):
    import sparc.pipeline.publication_schema as schema_module
    import sparc.pipeline.annotation_saver as saver_module

    state = _pipeline_annotation()
    expected = load_publication_arrays(build_publication_annotation(state))

    def forbidden_encoding(*args, **kwargs):
        raise AssertionError('HDF5 path must not compress an intermediate NPZ blob')

    monkeypatch.setattr(schema_module, 'encode_arrays_blob', forbidden_encoding)
    annotation = build_publication_annotation(state, embed_arrays=False)
    assert '_values' in annotation['arrays']
    if externalize:
        monkeypatch.setattr(saver_module, 'encode_arrays_blob', forbidden_encoding)
    output = tmp_path / 'annotations.jsonl'
    writer = AnnotationSaver(str(output), queue.Queue(), externalize_arrays=externalize)
    writer.save(annotation)
    writer.close()
    writer.start_writer_process()
    row = json.loads(output.read_text())
    assert '_values' not in row['arrays']
    actual = load_publication_arrays(row, sidecar_root=tmp_path)
    assert actual.keys() == expected.keys()
    for key in actual:
        assert actual[key].dtype == expected[key].dtype
        np.testing.assert_array_equal(actual[key], expected[key])


def test_deferred_arrays_fall_back_to_embedded_json_when_hdf5_write_fails(tmp_path, monkeypatch):
    from sparc.pipeline.annotation_saver import _PublicationArrayShardWriter

    def fail_write(*args):
        raise OSError('simulated storage failure')

    monkeypatch.setattr(_PublicationArrayShardWriter, '_write_array', fail_write)
    annotation = build_publication_annotation(_pipeline_annotation(), embed_arrays=False)
    output = tmp_path / 'annotations.jsonl'
    writer = AnnotationSaver(str(output), queue.Queue(), externalize_arrays=True)
    writer.save(annotation)
    writer.close()
    writer.start_writer_process()
    row = json.loads(output.read_text())
    assert '_values' not in row['arrays']
    assert row['arrays']['blob']
    actual = load_publication_arrays(row)
    for key, value in annotation['arrays']['_values'].items():
        np.testing.assert_array_equal(actual[key], value)
