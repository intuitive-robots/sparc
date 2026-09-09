import json
import logging
import os
from openai import OpenAI
from langchain_core.output_parsers import JsonOutputParser

from sparc.llm.prompts.prompts import (
    PROMPT_SYSTEM_TASK_OBJ_WITH_PHASES,
    PROMPT_USER_TASK_OBJ_WITH_PHASES,
    PROMPT_SYSTEM_TASK_OBJ_SEMANTIC,
    PROMPT_USER_TASK_OBJ_SEMANTIC,
)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")


LOCAL_API_KEY = "test"

def build_vllm_extra_body(vllm_config):
    """Build the extra_body dict for OpenAI-compat API calls from a vllm config dict."""
    if vllm_config is None:
        return None
    extra_body = {}
    if vllm_config.get("enable_thinking") is not None:
        extra_body["chat_template_kwargs"] = {"enable_thinking": vllm_config["enable_thinking"]}
    for key in ("repetition_penalty", "top_k", "presence_penalty"):
        if vllm_config.get(key) is not None:
            extra_body[key] = vllm_config[key]
    return extra_body or None


def query_chat_model(prompt, temperature=0.7, model="gpt-4o-mini", base_url=None,
                       extra_body=None):
    if "system" in prompt:
        messages = [{"role": "system", "content": prompt["system"]}, {"role": "user", "content": prompt["user"]}]
    else:
        messages = [{"role": "user", "content": prompt["user"]}]

    if base_url is None:
        client = OpenAI(api_key=OPENAI_API_KEY)
    else:
        client = OpenAI(api_key=LOCAL_API_KEY,
                        base_url=base_url)

    n_retries = 2
    success = False
    while n_retries > 0 and not success:
        try:
            kwargs = {
                "messages": messages,
                "model": model,
                "temperature": temperature,
            }
            if extra_body is not None:
                kwargs["extra_body"] = extra_body
            chat_completion = client.chat.completions.create(**kwargs)
            success = True
        except Exception as e:
            logging.error(f"Error in LLM query: {e}")
            n_retries -= 1
            if n_retries == 0:
                return ["Error in response from LLM"]

    llm_response = chat_completion.choices[0].message.content

    return llm_response

def verify_annotation_with_vlm(mode, frame_start, frame_end, box_initial, box_target,
                                object_name, instruction, task_obj_info,
                                trajectory_name=None,
                                model=None, base_url=None,
                                temperature=1.0, extra_body=None):
    """Cross-model verification of an annotation via VLM.

    Uses the same vllm config as the rest of the pipeline (model, base_url,
    temperature, sampling params) — only thinking mode is disabled by the caller.

    Args:
        mode: "highlighted_box" or "spatial_consistency"
        frame_start: numpy array (H, W, 3) — initial frame
        frame_end: numpy array (H, W, 3) — final frame (used only for spatial_consistency)
        box_initial: [x1, y1, x2, y2] pixel coords of detected object box
        box_target: [x1, y1, x2, y2] pixel coords of target box (spatial_consistency only)
        object_name: name of the object
        instruction: task instruction string
        task_obj_info: dict with start_location, target_location, etc.
        model, base_url, temperature, extra_body: VLM connection params

    Returns:
        dict with keys:
            "vlm_answer": "yes" | "no" | "uncertain" | "error"
            "verification_score": 1.0 | 0.0 | 0.5
    """
    import base64
    import io
    import cv2
    from sparc.llm.prompts.prompts import (
        PROMPT_SYSTEM_VERIFY_HIGHLIGHTED_BOX,
        PROMPT_USER_VERIFY_HIGHLIGHTED_BOX,
        PROMPT_SYSTEM_VERIFY_SPATIAL_CONSISTENCY,
        PROMPT_USER_VERIFY_SPATIAL_CONSISTENCY,
    )

    def _encode_frame(frame):
        from PIL import Image
        img = Image.fromarray(frame)
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _draw_box(frame, box, color=(0, 0, 255), thickness=5):
        """Draw a rectangle on a copy of the frame."""
        img = frame.copy()
        x1, y1, x2, y2 = [int(c) for c in box]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        return img

    def _norm_coords(box, frame):
        """Normalize pixel box coords to 0-1000 scale."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(c) for c in box]
        return f"[{x1*1000//w}, {y1*1000//h}, {x2*1000//w}, {y2*1000//h}]"

    def _extract_text_content(content):
        """Convert OpenAI-compatible message content into a single text string."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text" and item.get("text") is not None:
                        text_parts.append(str(item["text"]))
                    elif item.get("type") == "output_text" and item.get("text") is not None:
                        text_parts.append(str(item["text"]))
                elif item is not None:
                    text_parts.append(str(item))
            return "\n".join(part for part in text_parts if part).strip()
        if content is None:
            return None
        return str(content)

    def _summarize_response(response):
        """Return a compact, log-safe summary of the completion payload."""
        try:
            choice = response.choices[0]
        except Exception:
            return {"response_repr": repr(response)[:800]}

        message = getattr(choice, "message", None)
        raw_content = getattr(message, "content", None) if message is not None else None

        summary = {
            "finish_reason": getattr(choice, "finish_reason", None),
            "message_role": getattr(message, "role", None) if message is not None else None,
            "content_type": type(raw_content).__name__,
            "content_preview": repr(raw_content)[:800],
        }

        refusal = getattr(message, "refusal", None) if message is not None else None
        if refusal is not None:
            summary["refusal"] = refusal

        reasoning = getattr(message, "reasoning_content", None) if message is not None else None
        if reasoning is not None:
            summary["reasoning_preview"] = repr(reasoning)[:400]

        return summary

    if mode == "highlighted_box":
        annotated = _draw_box(frame_start, box_initial, color=(0, 0, 255))
        img_b64 = _encode_frame(annotated)
        user_text = PROMPT_USER_VERIFY_HIGHLIGHTED_BOX.format(
            object_name=object_name, instruction=instruction,
            box_coords=_norm_coords(box_initial, frame_start),
        )
        messages = [
            {"role": "system", "content": PROMPT_SYSTEM_VERIFY_HIGHLIGHTED_BOX},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                {"type": "text", "text": user_text},
            ]},
        ]
    elif mode == "spatial_consistency":
        annotated_start = _draw_box(frame_start, box_initial, color=(0, 0, 255))
        annotated_end = _draw_box(frame_end, box_target, color=(0, 255, 0))
        start_b64 = _encode_frame(annotated_start)
        end_b64 = _encode_frame(annotated_end)
        user_text = PROMPT_USER_VERIFY_SPATIAL_CONSISTENCY.format(
            instruction=instruction,
            object_name=object_name,
            start_location=task_obj_info.get("start_location") or "unknown",
            target_location=task_obj_info.get("target_location") or "unknown",
            box_initial_coords=_norm_coords(box_initial, frame_start),
            box_target_coords=_norm_coords(box_target, frame_end),
        )
        messages = [
            {"role": "system", "content": PROMPT_SYSTEM_VERIFY_SPATIAL_CONSISTENCY},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{start_b64}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{end_b64}"}},
                {"type": "text", "text": user_text},
            ]},
        ]
    else:
        return {"vlm_answer": "error", "verification_score": 0.5}

    if base_url is None:
        client = OpenAI(api_key=OPENAI_API_KEY)
    else:
        client = OpenAI(api_key="test", base_url=base_url)

    try:
        kwargs = {
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "max_tokens": 6000,
        }
        if extra_body is not None:
            kwargs["extra_body"] = extra_body
        response = client.chat.completions.create(**kwargs)
        raw_content = response.choices[0].message.content
        answer_text = _extract_text_content(raw_content)
        if answer_text is None or not answer_text.strip():
            raise ValueError(
                f"Empty VLM verification response content. Summary={_summarize_response(response)}"
            )
        answer = answer_text.strip().lower()
    except Exception:
        verification_context = {
            "trajectory_name": trajectory_name,
            "mode": mode,
            "object_name": object_name,
            "instruction": instruction,
            "start_location": task_obj_info.get("start_location") if task_obj_info else None,
            "target_location": task_obj_info.get("target_location") if task_obj_info else None,
            "box_initial": box_initial,
            "box_target": box_target,
            "frame_start_shape": getattr(frame_start, "shape", None),
            "frame_end_shape": getattr(frame_end, "shape", None),
            "model": model,
            "base_url": base_url,
        }
        logging.exception(
            "VLM verification error. context=%s",
            verification_context,
        )
        return {"vlm_answer": "error", "verification_score": 0.5}

    # Parse verdict — look for "**verdict: X**" first, then fall back to last line
    import re
    verdict_match = re.search(r'\*\*verdict:\s*(yes|no|uncertain)\*\*', answer)
    if verdict_match:
        verdict = verdict_match.group(1)
    else:
        # Fallback: check last non-empty line for yes/no/uncertain
        last_line = [l.strip() for l in answer.splitlines() if l.strip()][-1] if answer.strip() else ""
        if "yes" in last_line:
            verdict = "yes"
        elif "no" in last_line:
            verdict = "no"
        else:
            verdict = "uncertain"

    score_map = {"yes": 1.0, "no": 0.0, "uncertain": 0.5}
    return {"vlm_answer": verdict, "verification_score": score_map[verdict]}


def extract_task_obj_with_phases(task: str, grasp_phases: list[tuple], model="gpt-5-mini", base_url=None, temperature=1.0, extra_body=None):
    """
    Extract task object information with phase descriptions in a single LLM call.
    
    Args:
        task: The task instruction string
        grasp_phases: List of tuples (start_frame, end_frame, phase_type) from gripper detection
                     e.g., [(0, 25, "approach"), (26, 80, "interact"), (81, 100, "release")]
    
    Returns:
        List of dicts, each representing one object interaction with its grasp_phases:
        [
            {
                "object": "object_name",
                "start_location": "...",
                "target_location": "...",
                "action": "...",
                "tool_name_required": "...",
                "tool_usage_description": "...",
                "grasp_phases": [
                    {"start_frame": int, "end_frame": int, "phase_type": str, "description": str}
                ]
            }
        ]
        Returns None on error
    """
    # Format grasp phases as JSON for the prompt
    phases_for_prompt = [
        {"start_frame": int(start), "end_frame": int(end), "phase_type": phase_type}
        for start, end, phase_type in grasp_phases
    ]
    phases_json = json.dumps(phases_for_prompt)

    user_content = f"Task: {task}\nDetected phases: {phases_json}"

    prompt = {
        "system": PROMPT_SYSTEM_TASK_OBJ_WITH_PHASES,
        "user": PROMPT_USER_TASK_OBJ_WITH_PHASES + user_content
    }

    response = query_chat_model(prompt, model=model, base_url=base_url,
                                   temperature=temperature, extra_body=extra_body)

    try:
        result = JsonOutputParser().parse(response)
    except Exception as e:
        logging.error(f"Error parsing task_obj_with_phases response: {e}")
        logging.error(f"Response was: {response}")
        return None

    # Normalize to list
    is_list = isinstance(result, list)
    result_list = result if is_list else [result]

    # Validate and normalize each entry
    normalized = []
    for item in result_list:
        if not isinstance(item, dict):
            logging.error(f"Invalid entry, expected dict got {type(item)}: {item}")
            continue

        normalized_item = {
            "object": item.get("object"),
            "start_location": item.get("start_location"),
            "target_location": item.get("target_location"),
            "action": item.get("action"),
            "tool_name_required": item.get("tool_name_required"),
            "tool_usage_description": item.get("tool_usage_description"),
            "grasp_phases": []
        }

        # Validate and normalize grasp_phases
        raw_phases = item.get("grasp_phases", [])
        if isinstance(raw_phases, list):
            for phase in raw_phases:
                if isinstance(phase, dict):
                    normalized_item["grasp_phases"].append({
                        "start_frame": phase.get("start_frame"),
                        "end_frame": phase.get("end_frame"),
                        "phase_type": phase.get("phase_type"),
                        "description": phase.get("description")
                    })

        normalized.append(normalized_item)

    if not normalized:
        return None

    return normalized if is_list or len(normalized) > 1 else normalized[0]


def extract_task_obj_semantic(
    task: str,
    episode_samples: list[dict],
    storyboards: list[dict],
    model="gpt-5-mini",
    base_url=None,
    temperature=0.2,
    extra_body=None,
):
    """Extract trajectory-specific task episodes from phases and a multiview storyboard."""
    import base64

    episodes_for_prompt = [
        {
            "episode_index": sample["episode_index"],
            "frames": dict(zip(sample["labels"], sample["frames"])),
            "phases": [
                {"start_frame": start, "end_frame": end, "phase_type": phase_type}
                for start, end, phase_type in sample["phases"]
            ],
        }
        for sample in episode_samples
    ]
    user_text = PROMPT_USER_TASK_OBJ_SEMANTIC.format(
        task=task,
        episodes_json=json.dumps(episodes_for_prompt),
    )
    user_content = [{"type": "text", "text": user_text}]
    for storyboard in storyboards:
        view_key = storyboard["view_key"]
        encoded = base64.b64encode(storyboard["jpeg_bytes"]).decode("ascii")
        user_content.extend(
            [
                {
                    "type": "text",
                    "text": (
                        f"Storyboard layout: {view_key}. "
                        "Columns are episodes; rows are camera views labeled in the image."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                },
            ]
        )

    messages = [
        {"role": "system", "content": PROMPT_SYSTEM_TASK_OBJ_SEMANTIC},
        {"role": "user", "content": user_content},
    ]
    client = (
        OpenAI(api_key=OPENAI_API_KEY)
        if base_url is None
        else OpenAI(api_key=LOCAL_API_KEY, base_url=base_url)
    )
    kwargs = {
        "messages": messages,
        "model": model,
        "temperature": temperature,
    }
    if extra_body is not None:
        kwargs["extra_body"] = extra_body

    try:
        response = client.chat.completions.create(**kwargs)
        result = JsonOutputParser().parse(response.choices[0].message.content)
    except Exception as e:
        logging.error(f"Error parsing semantic task-object response: {e}")
        return None

    result_list = result if isinstance(result, list) else [result]
    normalized = []
    for item in result_list:
        if not isinstance(item, dict):
            logging.error(f"Invalid semantic entry, expected dict got {type(item)}: {item}")
            continue
        normalized_item = {
            "object": item.get("object"),
            "object_instance": item.get("object_instance"),
            "instance_relation_to_previous": item.get("instance_relation_to_previous", "uncertain"),
            "start_location": item.get("start_location"),
            "target_location": item.get("target_location"),
            "action": item.get("action"),
            "tool_name_required": item.get("tool_name_required"),
            "tool_usage_description": item.get("tool_usage_description"),
            "grasp_phases": [],
        }
        for phase in item.get("grasp_phases", []):
            if not isinstance(phase, dict):
                continue
            normalized_item["grasp_phases"].append(
                {
                    "start_frame": phase.get("start_frame"),
                    "end_frame": phase.get("end_frame"),
                    "phase_type": phase.get("phase_type"),
                    "description": phase.get("description"),
                }
            )
        normalized.append(normalized_item)

    if not normalized:
        return None
    try:
        from sparc.llm.semantic_task_parsing import validate_and_merge_semantic_entries

        return validate_and_merge_semantic_entries(normalized, episode_samples)
    except ValueError as e:
        logging.error(f"Invalid semantic task-object response: {e}")
        return None
