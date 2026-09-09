"""LLM call for bimanual trajectory task-object extraction.

Returns the same schema as extract_task_obj_with_phases() but each dict
has an extra "arm": "left" | "right" field.
"""

import json
import logging
from typing import Optional


PROMPT_SYSTEM_BIMANUAL = '''You are a robot manipulation expert. Given a task instruction and the DETECTED GRIPPER PHASES for a BIMANUAL robot (separate left-arm and right-arm timelines), extract task information and generate descriptions for each phase.

The grasp phases are detected from each arm's gripper signal.  Each phase has:
- start_frame: when the phase begins
- end_frame: when the phase ends
- phase_type: one of "grasp", "interact", "release", "grasp_failure", "null" (the last indicates no gripper closed for that arm, so likely not used for grasping)

Your task:
1. Analyse the instruction to identify which objects the left arm and right arm interact with.
2. Assign each arm's phases to the corresponding subtask.
3. Generate a natural language description for each phase.

Output a JSON list. Each entry represents one arm's object interaction:
{
  "arm": "left" | "right",
  "object": "object_name",
  "start_location": "location or null",
  "target_location": "location or null",
  "action": "robot action (2 words max)",
  "tool_name_required": "tool name or null",
  "tool_usage_description": "how tool is used or null",
  "grasp_phases": [
    {
      "start_frame": int,
      "end_frame": int,
      "phase_type": "grasp|interact|release|grasp_failure",
      "description": "Natural language description of this phase"
    }
  ]
}

Guidelines:
- Every entry MUST have "arm" set to either "left" or "right"
- Do not merge the two arms into one entry
- If an arm's phases are empty or do not correspond to a valid object interaction, still emit an entry with "grasp_phases": []
- Descriptions should be present tense phrased as an instruction ("Grasp the ...", "Move the ...")'''

PROMPT_USER_BIMANUAL_EXAMPLES = '''Examples:

Task: Pick up the red block with the left arm and place the blue cup with the right arm
Left-arm phases: [{"start_frame": 0, "end_frame": 20, "phase_type": "grasp"}, {"start_frame": 21, "end_frame": 60, "phase_type": "interact"}, {"start_frame": 61, "end_frame": 70, "phase_type": "release"}]
Right-arm phases: [{"start_frame": 5, "end_frame": 25, "phase_type": "grasp"}, {"start_frame": 26, "end_frame": 65, "phase_type": "interact"}, {"start_frame": 66, "end_frame": 75, "phase_type": "release"}]
[{"arm": "left", "object": "red block", "start_location": null, "target_location": null, "action": "pick up", "tool_name_required": null, "tool_usage_description": null, "grasp_phases": [{"start_frame": 0, "end_frame": 20, "phase_type": "grasp", "description": "Move the left gripper to the red block and grasp it."}, {"start_frame": 21, "end_frame": 60, "phase_type": "interact", "description": "Lift the red block."}, {"start_frame": 61, "end_frame": 70, "phase_type": "release", "description": "Release the red block."}]}, {"arm": "right", "object": "blue cup", "start_location": null, "target_location": null, "action": "place", "tool_name_required": null, "tool_usage_description": null, "grasp_phases": [{"start_frame": 5, "end_frame": 25, "phase_type": "grasp", "description": "Move the right gripper to the blue cup and grasp it."}, {"start_frame": 26, "end_frame": 65, "phase_type": "interact", "description": "Move the blue cup to the target location."}, {"start_frame": 66, "end_frame": 75, "phase_type": "release", "description": "Place the blue cup down."}]}]

'''


def extract_task_obj_bimanual(
    instruction: str,
    left_phases: list,
    right_phases: list,
    model: str = "qwen3-vl-30b",
    base_url: str = "http://localhost:8000/v1",
    extra_body: dict = None,
) -> Optional[list]:
    """Extract task-object info for a bimanual trajectory.

    Args:
        instruction: Language instruction string.
        left_phases:  List of (start, end, phase_type) tuples for the left arm.
        right_phases: List of (start, end, phase_type) tuples for the right arm.
        model:    Model name for the OpenAI-compat endpoint.
        base_url: Base URL of the inference server.

    Returns:
        List of dicts (same schema as extract_task_obj_with_phases but with "arm" field),
        or None on error.
    """
    from sparc.llm.llm_utils import query_chat_model
    from langchain_core.output_parsers import JsonOutputParser

    def _fmt(phases):
        return json.dumps([
            {"start_frame": int(s), "end_frame": int(e), "phase_type": pt}
            for s, e, pt in phases
        ])

    user_content = (
        f"Task: {instruction}\n"
        f"Left-arm phases: {_fmt(left_phases)}\n"
        f"Right-arm phases: {_fmt(right_phases)}"
    )

    prompt = {
        "system": PROMPT_SYSTEM_BIMANUAL,
        "user": PROMPT_USER_BIMANUAL_EXAMPLES + user_content,
    }

    response = query_chat_model(prompt, model=model, base_url=base_url, extra_body=extra_body)

    try:
        result = JsonOutputParser().parse(response)
    except Exception as e:
        logging.error(f"extract_task_obj_bimanual: JSON parse error: {e}")
        logging.error(f"Response was: {response}")
        return None

    if not isinstance(result, list):
        result = [result]

    normalized = []
    for item in result:
        if not isinstance(item, dict):
            logging.error(f"extract_task_obj_bimanual: expected dict, got {type(item)}: {item}")
            continue

        arm = item.get("arm")
        if arm not in ("left", "right"):
            logging.warning(f"extract_task_obj_bimanual: missing/invalid arm field: {item}")
            arm = "left"  # safe default

        normalized_item = {
            "arm": arm,
            "object": item.get("object"),
            "start_location": item.get("start_location"),
            "target_location": item.get("target_location"),
            "action": item.get("action"),
            "tool_name_required": item.get("tool_name_required"),
            "tool_usage_description": item.get("tool_usage_description"),
            "grasp_phases": [],
        }

        raw_phases = item.get("grasp_phases", [])
        if isinstance(raw_phases, list):
            for phase in raw_phases:
                if isinstance(phase, dict):
                    normalized_item["grasp_phases"].append({
                        "start_frame": phase.get("start_frame"),
                        "end_frame": phase.get("end_frame"),
                        "phase_type": phase.get("phase_type"),
                        "description": phase.get("description"),
                    })

        normalized.append(normalized_item)

    if not normalized:
        return None

    return normalized
