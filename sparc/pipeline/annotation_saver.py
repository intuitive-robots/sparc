"""Queue-based annotation writer for JSONL output."""

import json
import logging
import os
from pathlib import Path
from urllib.parse import quote

import numpy as np

from sparc.pipeline.publication_schema import (
    require_publication_annotation,
    subtask_index_from_annotation,
    trajectory_id_from_annotation,
)
from sparc.scoring.selection_scoring import decode_arrays_blob, encode_arrays_blob


def _embed_arrays(annotation: dict) -> dict:
    """Encode deferred arrays only for JSONL storage or an HDF5 write fallback."""
    arrays = annotation.get("arrays")
    if not isinstance(arrays, dict) or "_values" not in arrays:
        return annotation
    arrays = dict(arrays)
    values = arrays.pop("_values")
    arrays["blob"] = encode_arrays_blob(values)
    return dict(annotation, arrays=arrays)


class _PublicationArrayShardWriter:
    """Stream publication arrays into size-bounded HDF5 shards."""

    def __init__(self, output_file: str, sidecar_dir: str | None,
                 target_bytes: int):
        import h5py

        self.h5py = h5py
        output_path = Path(output_file)
        self.output_dir = output_path.parent
        if sidecar_dir:
            configured_dir = Path(sidecar_dir)
            self.sidecar_dir = (
                configured_dir if configured_dir.is_absolute()
                else self.output_dir / configured_dir
            )
        else:
            self.sidecar_dir = self.output_dir / f"{output_path.stem}_artifacts"
        self.target_bytes = max(1, int(target_bytes))
        self.handle = None
        self.record_index = 0
        existing = []
        if self.sidecar_dir.exists():
            for path in self.sidecar_dir.glob("annotation-artifacts-*.h5"):
                try:
                    existing.append(int(path.stem.rsplit("-", 1)[-1]))
                except ValueError:
                    continue
        self.shard_index = max(existing, default=-1) + 1

    def _open_shard(self):
        if self.handle is not None:
            return
        self.sidecar_dir.mkdir(parents=True, exist_ok=True)
        shard_name = f"annotation-artifacts-{self.shard_index:06d}.h5"
        self.shard_path = self.sidecar_dir / shard_name
        self.relative_path = Path(
            os.path.relpath(self.shard_path, self.output_dir)
        ).as_posix()
        self.handle = self.h5py.File(self.shard_path, "w", libver="latest")
        self.handle.attrs["format"] = "sparc_publication_array_shard"
        self.handle.attrs["version"] = 2
        self.handle.attrs["group_layout"] = "trajectory_subtask_annotation"
        self.record_index = 0

    def _close_shard(self):
        if self.handle is None:
            return
        path = self.shard_path
        record_count = self.record_index
        self.handle.flush()
        self.handle.close()
        self.handle = None
        logging.info(
            "Closed publication artifact shard %s with %d annotations (%.1f MiB)",
            path,
            record_count,
            path.stat().st_size / (1024 * 1024),
        )
        self.shard_index += 1

    def _write_array(self, group, key, value):
        value = np.asarray(value)
        if value.dtype.kind == "U":
            value = value.astype(self.h5py.string_dtype(encoding="utf-8"))
        kwargs = {}
        if value.ndim > 0 and value.size > 0:
            kwargs = {"compression": "gzip", "compression_opts": 4, "shuffle": True}
        group.create_dataset(str(key), data=value, **kwargs)

    @staticmethod
    def _group_name(annotation: dict) -> str:
        trajectory_id = trajectory_id_from_annotation(annotation) or "unknown"
        trajectory_key = quote(str(trajectory_id), safe="-._~")

        subtask_index = subtask_index_from_annotation(annotation)
        try:
            subtask_key = f"{int(subtask_index):04d}"
        except (TypeError, ValueError):
            subtask_key = quote(str(subtask_index or "unknown"), safe="-._~")

        annotation_id = str(annotation.get("annotation_id") or "unknown")
        annotation_key = annotation_id.removeprefix("sha256:")
        annotation_key = quote(annotation_key, safe="-._~")
        return (
            f"trajectories/{trajectory_key}/subtasks/"
            f"{subtask_key}/{annotation_key}"
        )

    @staticmethod
    def _set_group_attributes(group, annotation: dict):
        source = annotation.get("source") or {}
        window = annotation.get("window") or {}
        attributes = {
            "annotation_id": annotation.get("annotation_id"),
            "trajectory_id": trajectory_id_from_annotation(annotation),
            "subtask_index": subtask_index_from_annotation(annotation),
            "dataset_id": source.get("dataset_id"),
            "camera_view": source.get("camera_view"),
            "start_frame": window.get("start_frame"),
            "end_frame_exclusive": window.get("end_frame_exclusive"),
        }
        for key, value in attributes.items():
            if value is not None:
                group.attrs[key] = value

    def add(self, annotation: dict) -> list[dict]:
        require_publication_annotation(annotation)
        arrays = annotation.get("arrays")
        if not isinstance(arrays, dict):
            return [annotation]
        decoded = arrays.get("_values")
        if decoded is None:
            decoded = decode_arrays_blob(arrays.get("blob"))
        if not decoded:
            return [_embed_arrays(annotation)]
        self._open_shard()
        group_name = self._group_name(annotation)
        group_created = False
        try:
            if group_name in self.handle:
                group = self.handle[group_name]
            else:
                group = self.handle.create_group(group_name)
                group_created = True
                self._set_group_attributes(group, annotation)
                for key, value in decoded.items():
                    self._write_array(group, key, value)
                self.handle.flush()
        except Exception:
            logging.exception(
                "Failed to write publication arrays for %s", annotation.get("annotation_id")
            )
            try:
                if group_created and group_name in self.handle:
                    del self.handle[group_name]
                    self.handle.flush()
            except Exception:
                pass
            return [_embed_arrays(annotation)]

        arrays = dict(annotation["arrays"])
        arrays.pop("blob", None)
        arrays.pop("_values", None)
        arrays["encoding"] = "hdf5_shard_v1"
        arrays["storage"] = {
            "path": self.relative_path,
            "group": group_name,
        }
        prepared_annotation = dict(annotation)
        prepared_annotation["arrays"] = arrays
        self.record_index += 1
        if self.shard_path.stat().st_size >= self.target_bytes:
            self._close_shard()
        return [prepared_annotation]

    def close(self):
        self._close_shard()


class AnnotationSaver:
    def __init__(self, output_file: str, write_queue, *, externalize_arrays=False,
                 arrays_sidecar_dir=None, array_shard_target_bytes=1 << 30):
        self.output_file = output_file
        self.write_queue = write_queue
        self.externalize_arrays = bool(externalize_arrays)
        self.arrays_sidecar_dir = arrays_sidecar_dir
        self.array_shard_target_bytes = array_shard_target_bytes

    def get_processed_trajectories(self) -> set:
        processed = set()
        if not os.path.exists(self.output_file):
            return processed
        with open(self.output_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    trajectory_id = trajectory_id_from_annotation(data)
                    if trajectory_id is not None and "error" not in data:
                        processed.add(trajectory_id)
                except json.JSONDecodeError:
                    continue
        return processed

    def start_writer_process(self):
        # Configure logging for this process
        logging.basicConfig(level=logging.INFO, format='[Writer] %(asctime)s - %(levelname)s - %(message)s')
        f = None
        shard_writer = (
            _PublicationArrayShardWriter(
                self.output_file,
                self.arrays_sidecar_dir,
                self.array_shard_target_bytes,
            )
            if self.externalize_arrays else None
        )

        def write_annotations(annotations):
            nonlocal f
            for annotation in annotations:
                require_publication_annotation(annotation)
                if f is None:
                    output_dir = os.path.dirname(self.output_file)
                    if output_dir:
                        os.makedirs(output_dir, exist_ok=True)
                    f = open(self.output_file, 'a')
                f.write(json.dumps(annotation) + '\n')
                f.flush()
                traj_name = trajectory_id_from_annotation(annotation)
                logging.info(f"Wrote annotation for trajectory {traj_name}")

        try:
            while True:
                line = self.write_queue.get()
                if line is None:  # Sentinel for stopping
                    break
                write_annotations(
                    shard_writer.add(line) if shard_writer is not None else [_embed_arrays(line)]
                )
            if shard_writer is not None:
                shard_writer.close()
        finally:
            if shard_writer is not None:
                shard_writer.close()
            if f is not None:
                f.close()

    def save(self, annotation: dict):
        require_publication_annotation(annotation)
        self.write_queue.put(annotation)

    def close(self):
        self.write_queue.put(None)
