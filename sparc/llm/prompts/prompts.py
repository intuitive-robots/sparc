"""Prompts used by task parsing and annotation verification."""

PROMPT_SYSTEM_TASK_OBJ_WITH_PHASES = '''You are a robot manipulation expert. Given a task instruction and the DETECTED GRIPPER PHASES from the robot's execution (with frame indices), extract task information and generate descriptions for each phase.

The grasp phases are detected from the robot's gripper signal and represent what ACTUALLY happened during execution. Each phase has:
- start_frame: when the phase begins
- end_frame: when the phase ends  
- phase_type: one of "grasp", "interact", "release", "grasp_failure"

Phase types meaning:
- "grasp": Gripper is approaching and closing on the object
- "interact": Gripper is closed, robot manipulating/moving object
- "release": Gripper opens to release object
- "grasp_failure": Gripper closed but object slipped (short interact followed by premature release)

Your task:
1. Analyze the task instruction to identify distinct object interactions (subtasks)
2. Assign the detected grasp phases to the appropriate subtask based on frame indices and task semantics
3. Generate a natural language description for each phase

Output a JSON list where each entry represents one object interaction:
{
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
      "description": "Detailed natural language description of what happens in this phase"
    }
  ]
}

Guidelines:
- Each grasp_phases entry should have a description referencing the specific object and action
- Descriptions should be present tense phrased as an instruction ("Grasp the ...", "Move the ...")
- For multi-object tasks, split phases between subtasks based on frame order and task semantics
- If only one object but multiple grasp cycles (e.g., slip and retry), keep all phases in one entry
- Frame indices help determine which phases belong to which subtask'''

PROMPT_SYSTEM_TASK_OBJ_SEMANTIC = '''Assign one task object to each robot episode.

Storyboard columns E0, E1, ... match detected episodes. Camera rows are labeled in the image; when both are available, external is on top and wrist is below. Return exactly one JSON item per episode column, in the same order.

Object identity priority: task text and semantic role first, visual evidence second. The images help assign task-mentioned objects to episodes, distinguish instances, and add clear attributes. Do not invent unrelated objects.

Output schema:
[{
  "object": "task-grounded singular object name",
  "object_instance": "instance_0|instance_1|unknown",
  "instance_relation_to_previous": "first_instance|different_instance|same_instance_retry|same_instance_continuation|uncertain",
  "start_location": "location or null",
  "target_location": "location or null",
  "action": "short canonical robot action category",
  "tool_name_required": "tool name or null",
  "tool_usage_description": "how tool is used or null",
  "grasp_phases": [
    {
      "start_frame": 0,
      "end_frame": 10,
      "phase_type": "grasp|interact|release|grasp_failure",
      "description": "imperative phase instruction, starting with a base-form verb"
    }
  ]
}]

Rules:
- Copy each episode's phases exactly. Only add descriptions.
- Do not merge, omit, split, reorder, or invent rows or phases.
- `object` is the task item being moved or affected. If the robot holds an implement to affect it, put that implement in `tool_name_required`, not in `object`.
- "wipe the pan with a towel" -> object: "pan", tool_name_required: "towel". "put the towel in a drawer" -> object: "towel", tool_name_required: null.
- Write `action` as a short canonical category, normally one to three words, for example "pick and place", "pick up", "open", "close", "wipe", or "pour".
- Do not include object names, attributes, locations, or a full instruction in `action`.
- Write every phase `description` as a natural, specific instruction in imperative voice.
- Use diverse, context-appropriate verbs and phrasing. Do not force descriptions into fixed "Grasp ...", "Move ...", or "Release ..." templates based on phase type.
- Start instructions with an imperative base-form verb. Never narrate in third person ("the robot picks up ..."), use a gerund ("picking up ..."), or describe the action as an observation.
- Normally use an object noun from the task. You may add a clear visual attribute: "blocks" can become "yellow block" and "green block".
- Infer a new object class only if the task uses generic words ("object", "item", "clothes") or states only a high-level goal without naming the manipulated item. The item must be clearly held or touched by the gripper; otherwise use the task's generic noun.
- Never select a background object merely because it is visible. If task text and appearance conflict, trust the task. If an attribute is unclear, omit it.
- First row: first_instance.
- Different physical object: different_instance. Same object after a failed grasp: same_instance_retry. Same object continued after release: same_instance_continuation.
- Use same_instance_retry or same_instance_continuation only when the object name and object_instance match the immediately previous episode.
- If the task gives a count but there are extra episodes, treat an extra episode as a retry or continuation unless the images clearly show another physical instance.
- Plural nouns can refer to different instances. If unsure, use uncertain.
- "put the blocks in the bowl", visibly yellow then green -> "yellow block", "green block".
- "put the blue block in the drawer, then close it" -> "blue block", "drawer"; not another visible item.
- Return JSON only.'''

PROMPT_USER_TASK_OBJ_SEMANTIC = '''Task: {task}
Detected episodes: {episodes_json}

Use the attached storyboard.'''

PROMPT_USER_TASK_OBJ_WITH_PHASES = '''Examples:

Task: Place the bag of chips in the bottom drawer
Detected phases: [{"start_frame": 0, "end_frame": 25, "phase_type": "grasp"}, {"start_frame": 26, "end_frame": 80, "phase_type": "interact"}, {"start_frame": 81, "end_frame": 100, "phase_type": "release"}]
{"object": "bag of chips", "start_location": null, "target_location": "bottom drawer", "action": "pick and place", "tool_name_required": null, "tool_usage_description": null, "grasp_phases": [{"start_frame": 0, "end_frame": 25, "phase_type": "grasp", "description": "Move towards the bag, lower the gripper, and close the gripper to grasp the bag."}, {"start_frame": 26, "end_frame": 80, "phase_type": "interact", "description": "Lift the bag and move it toward the bottom drawer."}, {"start_frame": 81, "end_frame": 100, "phase_type": "release", "description": "Place the grasped bag in the drawer."}]}

Task: Pick up the apple
Detected phases: [{"start_frame": 0, "end_frame": 15, "phase_type": "grasp"}, {"start_frame": 16, "end_frame": 22, "phase_type": "interact"}, {"start_frame": 23, "end_frame": 28, "phase_type": "grasp_failure"}, {"start_frame": 29, "end_frame": 40, "phase_type": "grasp"}, {"start_frame": 41, "end_frame": 70, "phase_type": "interact"}]
{"object": "apple", "start_location": null, "target_location": "robot gripper", "action": "pick up", "tool_name_required": null, "tool_usage_description": null, "grasp_phases": [{"start_frame": 0, "end_frame": 15, "phase_type": "grasp", "description": "Move the gripper to the apple and close the gripper to grasp the apple."}, {"start_frame": 16, "end_frame": 22, "phase_type": "interact", "description": "Lift the apple upward from the surface."}, {"start_frame": 23, "end_frame": 28, "phase_type": "grasp_failure", "description": "The apple slips from the gripper and the grasp is lost."}, {"start_frame": 29, "end_frame": 40, "phase_type": "grasp", "description": "Reposition the gripper and close it again to re-grasp the apple."}, {"start_frame": 41, "end_frame": 70, "phase_type": "interact", "description": "Lift and hold the apple securely in the gripper."}]}

Task: Open the fridge and pick up the apple, then put it on the table
Detected phases: [{"start_frame": 0, "end_frame": 10, "phase_type": "grasp"}, {"start_frame": 11, "end_frame": 35, "phase_type": "interact"}, {"start_frame": 36, "end_frame": 45, "phase_type": "release"}, {"start_frame": 46, "end_frame": 60, "phase_type": "grasp"}, {"start_frame": 61, "end_frame": 110, "phase_type": "interact"}, {"start_frame": 111, "end_frame": 125, "phase_type": "release"}]
[{"object": "fridge", "start_location": null, "target_location": null, "action": "open", "tool_name_required": null, "tool_usage_description": null, "grasp_phases": [{"start_frame": 0, "end_frame": 10, "phase_type": "grasp", "description": "Move to the fridge handle and close the gripper around it."}, {"start_frame": 11, "end_frame": 35, "phase_type": "interact", "description": "Pull the handle outward to open the fridge door."}, {"start_frame": 36, "end_frame": 45, "phase_type": "release", "description": "Open the gripper to release the handle."}]}, {"object": "apple", "start_location": "fridge", "target_location": "table", "action": "pick and place", "tool_name_required": null, "tool_usage_description": null, "grasp_phases": [{"start_frame": 46, "end_frame": 60, "phase_type": "grasp", "description": "Move the gripper toward the apple and grasp it."}, {"start_frame": 61, "end_frame": 110, "phase_type": "interact", "description": "Lift the apple and move it from the fridge to the table."}, {"start_frame": 111, "end_frame": 125, "phase_type": "release", "description": "Lower the apple onto the table and open the gripper."}]}]

Task: Stir the soup using the wooden spoon, then place the spoon on the plate
Detected phases: [{"start_frame": 0, "end_frame": 20, "phase_type": "grasp"}, {"start_frame": 21, "end_frame": 90, "phase_type": "interact"}, {"start_frame": 91, "end_frame": 105, "phase_type": "release"}]
[{"object": "soup", "start_location": "pot", "target_location": "pot", "action": "stir", "tool_name_required": "wooden spoon", "tool_usage_description": "use the wooden spoon to stir the soup", "grasp_phases": [{"start_frame": 0, "end_frame": 20, "phase_type": "grasp", "description": "Move to the wooden spoon handle and grasp it with the gripper."}, {"start_frame": 21, "end_frame": 70, "phase_type": "interact", "description": "Move the spoon in a circular motion inside the pot to stir the soup."}]}, {"object": "wooden spoon", "start_location": "robot gripper", "target_location": "plate", "action": "place", "tool_name_required": null, "tool_usage_description": null, "grasp_phases": [{"start_frame": 71, "end_frame": 90, "phase_type": "interact", "description": "Move the spoon toward the plate."}, {"start_frame": 91, "end_frame": 105, "phase_type": "release", "description": "Place the wooden spoon on the plate."}]}]

Task: Knock over the can
Detected phases: [{"start_frame": 0, "end_frame": 30, "phase_type": "grasp"}, {"start_frame": 31, "end_frame": 50, "phase_type": "interact"}]
{"object": "can", "start_location": null, "target_location": null, "action": "knock over", "tool_name_required": null, "tool_usage_description": null, "grasp_phases": [{"start_frame": 0, "end_frame": 30, "phase_type": "grasp", "description": "Move the gripper close to the side of the can."}, {"start_frame": 31, "end_frame": 50, "phase_type": "interact", "description": "Push the can sideways to knock it over."}]}

Task: Wipe the table with the yellow towel
Detected phases: [None]
{"object": "yellow towel", "start_location": null, "target_location": "table", "action": "wipe", "tool_name_required": "yellow towel", "tool_usage_description": "use the yellow towel to wipe the table", "grasp_phases": [{"start_frame": 0, "end_frame": 15, "phase_type": "grasp", "description": "Move to the towel and close the gripper to grasp it."}, {"start_frame": 16, "end_frame": 90, "phase_type": "interact", "description": "Press the towel onto the table and move it back and forth to wipe the surface."}]}
'''

PROMPT_SYSTEM_VERIFY_HIGHLIGHTED_BOX = '''You are a visual verification assistant. You will be shown a robot manipulation scene with a highlighted bounding box drawn on the image. Your task is to verify whether the highlighted box correctly identifies the specified object.

Reason about what you see in the image:
- What object is actually inside the highlighted box?
- Does it match the specified object name?
- Could the box be on the robot arm, background, or a different object?

After reasoning, end your response with your final verdict on a new line in exactly this format:
**verdict: yes** or **verdict: no** or **verdict: uncertain**

- yes: The box clearly contains the named object
- no: The box does not contain the named object (wrong object, empty area, or robot arm)
- uncertain: Cannot determine (occlusion, ambiguous, poor image quality)'''

PROMPT_USER_VERIFY_HIGHLIGHTED_BOX = '''The highlighted box (drawn in red) should contain: "{object_name}"
Box coordinates (x1,y1,x2,y2 normalized 0-1000): {box_coords}
Task instruction: "{instruction}"

Is the highlighted box around the correct object?'''

PROMPT_SYSTEM_VERIFY_SPATIAL_CONSISTENCY = '''You are a spatial verification assistant for robot manipulation. You will be shown two frames from a robot manipulation video: the initial frame and the final frame. Bounding boxes are drawn on both frames showing where the pipeline believes the object is at the start and end.

Your task is to verify whether the object movement shown by the boxes is consistent with the task instruction.

Reason about what you see:
- Is the correct object highlighted in both frames?
- Does the spatial displacement between the two boxes match the expected movement direction?
- Is the start location and end location consistent with the instruction?

After reasoning, end your response with your final verdict on a new line in exactly this format:
**verdict: yes** or **verdict: no** or **verdict: uncertain**

- yes: The boxes show a movement consistent with the instruction (correct object, correct direction)
- no: The movement is clearly wrong (object didn't move there, wrong direction, wrong object)
- uncertain: Cannot determine'''

PROMPT_USER_VERIFY_SPATIAL_CONSISTENCY = '''Task: "{instruction}"
Object: "{object_name}"
Expected movement: from "{start_location}" to "{target_location}"
Initial box (red, 0-1000 scale): {box_initial_coords}
Final box (green, 0-1000 scale): {box_target_coords}

The red box on the first image shows where the object starts. The green box on the second image shows where the object ends. Is this movement consistent with the task?'''
