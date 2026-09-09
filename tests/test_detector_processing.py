from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sparc.detectors import llmdet_hf
from sparc.detectors._batching import create_batches


@pytest.mark.parametrize("as_tensor", [False, True])
@pytest.mark.parametrize(
    ("image_count", "batch_size", "expected_sizes"),
    [(1, 4, [1]), (8, 4, [4, 4]), (9, 4, [4, 4, 1]), (3, 1, [1, 1, 1])],
)
def test_detector_batches_keep_image_order_and_remainder(
    as_tensor, image_count, batch_size, expected_sizes
):
    images = np.arange(image_count * 12).reshape(image_count, 3, 2, 2)
    inputs = torch.from_numpy(images) if as_tensor else images

    batches = create_batches(batch_size, inputs)

    assert [len(batch) for batch in batches] == expected_sizes
    np.testing.assert_array_equal(np.concatenate(batches), images)
    assert all(isinstance(batch, type(inputs)) for batch in batches)


def test_detector_batching_imports_remain_available():
    assert llmdet_hf.create_batches is create_batches


def test_detector_postprocessing_preserves_pixel_boxes_scores_and_class_order():
    probabilities = torch.full((1, 3, 8), 0.001)
    probabilities[0, 0, 1] = 0.8
    probabilities[0, 1, 3] = 0.9
    probabilities[0, 2, 5] = 0.7
    input_ids = torch.tensor([[101, 1001, 1012, 1002, 1012, 1003, 1012, 102]])
    outputs = SimpleNamespace(
        logits=torch.logit(probabilities),
        pred_boxes=torch.tensor([[
            [0.2, 0.2, 0.2, 0.2],
            [0.5, 0.5, 0.2, 0.2],
            [0.8, 0.8, 0.2, 0.2],
        ]]),
        input_ids=input_ids,
    )
    kwargs = dict(input_ids=input_ids, target_sizes=[(50, 100)], token_span_lens=[1, 1, 1])
    processor = llmdet_hf.CustomGroundingDinoProcessor
    kwargs["classes"] = ["cup", "bottle", "robot gripper"]
    expected_labels = ["bottle", "cup", "robot gripper"]
    expected_scores = [0.9, 0.8, 0.35]

    result = processor.post_process_grounded_object_detection(
        SimpleNamespace(), outputs, **kwargs
    )[0]

    assert result["labels"] == expected_labels
    torch.testing.assert_close(result["scores"], torch.tensor(expected_scores))
    torch.testing.assert_close(
        result["boxes"],
        torch.tensor([[40.0, 20.0, 60.0, 30.0], [10.0, 5.0, 30.0, 15.0], [70.0, 35.0, 90.0, 45.0]]),
    )
