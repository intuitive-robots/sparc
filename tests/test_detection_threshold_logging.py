from sparc.pipeline.gpu_inference import GPUInferenceServer


def test_llmdet_startup_message_reports_effective_per_call_thresholds():
    server = object.__new__(GPUInferenceServer)
    server.gpu_label = "0"
    server.detection_model_name = "llmdet"
    server.runtime_detection_thresholds = {
        "standard_box_query": 0.05,
        "standard_text_class": 0.005,
        "tool_box_query": 0.05,
        "tool_text_class": 0.1,
        "tool_object_box_query": 0.25,
        "tool_object_text_class": 0.1,
    }

    message = server._detection_startup_message(
        {
            "model_id": "iSEE-Laboratory/llmdet_base",
            "box_threshold": 0.35,
            "text_threshold": 0.25,
            "use_slicing": False,
        }
    )

    assert "standard object/target box_query=0.05, text_class=0.005" in message
    assert "tool box_query=0.05, text_class=0.1" in message
    assert "tool-object box_query=0.25, text_class=0.1" in message
    assert "box_threshold" not in message
    assert "text_threshold" not in message
