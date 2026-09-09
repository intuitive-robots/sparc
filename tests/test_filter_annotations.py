import json

import pytest

from filter_annotations import filter_annotations


def test_filters_finite_scores_inclusively_without_using_detector_confidence(tmp_path):
    source = tmp_path / "annotations.jsonl"
    rows = [{"selection": {"score": score}, "object": {"initial": {"detector_score": 1}}}
            for score in [0.94, 0.95, 1.2, None, "1", True, float("nan"), float("inf")]]
    rows.append({})
    original = "\n".join(map(json.dumps, rows)) + "\n\n"
    source.write_text(original)
    output, kept, total = filter_annotations(source)
    assert (kept, total) == (2, 9)
    assert [json.loads(line) for line in output.read_text().splitlines()] == rows[1:3]
    assert source.read_text() == original


def test_rebases_sidecar_reference_when_output_directory_changes(tmp_path):
    source = tmp_path / "annotations.jsonl"
    source.write_text(json.dumps({"selection": {"score": 1}, "arrays": {
        "storage": {"path": "annotations_artifacts/shard.h5", "group": "a"}}}) + "\n")
    outdir = tmp_path / "filtered"
    outdir.mkdir()
    output, _, _ = filter_annotations(source, output=outdir / "keep.jsonl")
    storage = json.loads(output.read_text())["arrays"]["storage"]
    assert (output.parent / storage["path"]).resolve() == tmp_path / "annotations_artifacts/shard.h5"
    assert storage["group"] == "a"


def test_bimanual_records_require_every_arm_to_pass(tmp_path):
    source = tmp_path / "annotations.jsonl"
    rows = [{"arms": [
        {"arm": "left", "selection": {"score": left}},
        {"arm": "right", "selection": {"score": right}},
    ]} for left, right in [(0.95, 1.2), (1.2, 0.94), (0.94, 1.2), (1.2, None), (True, 1.2)]]
    source.write_text("\n".join(map(json.dumps, rows)) + "\n")
    output, kept, total = filter_annotations(source)
    assert (kept, total) == (1, 5)
    assert json.loads(output.read_text()) == rows[0]


@pytest.mark.parametrize("tail", ['{bad json}', '[]'])
def test_malformed_input_does_not_leave_partial_output(tmp_path, tail):
    source = tmp_path / "annotations.jsonl"
    source.write_text('{"selection":{"score":1}}\n' + tail + '\n')
    with pytest.raises(ValueError, match="line 2"):
        filter_annotations(source)
    assert sorted(p.name for p in tmp_path.iterdir()) == [source.name]


def test_does_not_overwrite_input_or_existing_output(tmp_path):
    source = tmp_path / "annotations.jsonl"
    source.write_text('{"selection":{"score":1}}\n')
    with pytest.raises(ValueError, match="differ"):
        filter_annotations(source, output=source)
    output = tmp_path / "annotations_filtered.jsonl"
    output.write_text("existing")
    with pytest.raises(FileExistsError):
        filter_annotations(source)
    assert output.read_text() == "existing"
