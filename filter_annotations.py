#!/usr/bin/env python3
"""Filter SPARC annotations by selection score, preserving array references."""

import argparse
import json
import math
import os
from pathlib import Path
import tempfile


def filter_annotations(source, *, threshold=0.95, output=None):
    """Write retained records to a new file; return (output_path, kept, total)."""
    source = Path(source).resolve()
    output = (Path(output) if output is not None else source.with_name(
        f"{source.stem}_filtered{source.suffix}"
    )).resolve()
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    if output == source:
        raise ValueError("output must differ from the input file")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}; choose another --output")
    kept = total = 0
    temporary = None
    try:
        with source.open() as src, tempfile.NamedTemporaryFile(
            mode="w", dir=output.parent, prefix=f".{output.name}.", delete=False,
        ) as dst:
            temporary = Path(dst.name)
            for line_number, line in enumerate(src, 1):
                if not line.strip():
                    continue
                try:
                    annotation = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_number}: {exc.msg}") from exc
                if not isinstance(annotation, dict):
                    raise ValueError(f"Expected an annotation object on line {line_number}")
                total += 1
                arms = annotation.get("arms")
                parts = arms if isinstance(arms, list) and arms else [annotation]
                scores = []
                for part in parts:
                    selection = part.get("selection") if isinstance(part, dict) else None
                    scores.append(selection.get("score") if isinstance(selection, dict) else None)
                if not all(
                    not isinstance(score, bool) and isinstance(score, (int, float))
                    and math.isfinite(score) and score >= threshold
                    for score in scores
                ):
                    continue
                if output.parent != source.parent:
                    arrays = annotation.get("arrays") or {}
                    storage = arrays.get("storage") or {}
                    if storage.get("path") and not Path(storage["path"]).is_absolute():
                        storage["path"] = os.path.relpath(
                            source.parent / storage["path"], output.parent,
                        )
                    dst.write(json.dumps(annotation, ensure_ascii=False) + "\n")
                else:
                    dst.write(line.rstrip("\n") + "\n")
                kept += 1
        # Publish only a complete file, without replacing an existing output.
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return output, kept, total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input annotation JSONL")
    parser.add_argument("--threshold", type=float, default=0.95,
                        help="Minimum selection.score for every arm (default: 0.95; tune per dataset)")
    parser.add_argument("--output", type=Path,
                        help="New JSONL path (default: <input_stem>_filtered.jsonl)")
    args = parser.parse_args()
    try:
        output, kept, total = filter_annotations(
            args.input, threshold=args.threshold, output=args.output,
        )
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(f"Kept {kept}/{total} annotations at score >= {args.threshold}: {output}")


if __name__ == "__main__":
    main()
