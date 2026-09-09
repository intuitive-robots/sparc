import numpy as np
import json
import os
import random
import logging
from tqdm import tqdm


def apply_phase_offset(phases: list[tuple], offset_ratio: float = 0.3) -> list[tuple]:
    """
    Apply forward offset to phase boundaries by borrowing frames from the next phase.
    
    This helps capture transition moments better - e.g., extending the grasp phase
    into early interact phase captures the "object just grasped" state.
    
    Args:
        phases: List of tuples (start_idx, end_idx, phase_name) from get_gripper_close_phases
        offset_ratio: Fraction of the *next* phase duration to add to current phase end.
                     Default 0.3 means current phase end extends 30% into next phase.
    
    Returns:
        List of tuples (start_idx, end_idx_with_offset, phase_name) with adjusted boundaries.
        Note: start_idx remains unchanged to avoid overlap issues.
    """
    if not phases or len(phases) < 2:
        return phases

    offset_phases = []

    for i, (start, end, phase_name) in enumerate(phases):
        if i < len(phases) - 1:
            # Get next phase info
            next_start, next_end, _ = phases[i + 1]
            next_duration = next_end - next_start

            # Calculate offset (frames to borrow from next phase)
            offset_frames = int(next_duration * offset_ratio)

            # New end is current end + offset, but not exceeding next phase end
            new_end = min(end + offset_frames, next_end)

            offset_phases.append((start, new_end, phase_name))
        else:
            offset_phases.append((start, end, phase_name))

    return offset_phases


def get_phase_sequence_key(phases: list[tuple]) -> str:
    """
    Generate a string key from phase sequence for caching.
    E.g., [(0, 10, "grasp"), (11, 50, "interact")] -> "grasp_interact"
    """
    phase_names = [p[2] for p in phases]
    return "_".join(phase_names)


def compute_gripper_stats(gripper_signals, closed_threshold=0.5, hysteresis_offset=0.1):
    """
    Compute aggregate statistics from a list of gripper signals.
    Each signal is normalized to [0,1] before computing stats.
                    # Slippage detection: if duration is much shorter than the robust threshold, it's a failure
                    min_valid_duration = min_valid_duration
    (interact) phases, which is more robust than simple fraction of closed time.
    
    Args:
        gripper_signals: List of 1D numpy arrays (raw gripper states per trajectory)
        closed_threshold: Threshold to classify closed vs open (on normalized scale)
        hysteresis_offset: Offset for hysteresis in phase detection
    
    Returns:
        dict with: mean_closed, std_closed, min_closed, mean_open, std_open, 
                   avg_interact_duration_frames, avg_interact_duration_ratio
    """
    all_closed_values = []
    all_open_values = []
    interact_durations_frames = []  # Duration in frames
    interact_durations_ratio = []   # Duration as fraction of trajectory length

    for signal in gripper_signals:
        signal = np.array(signal)
        if len(signal) < 2:
            continue
        # Normalize to [0, 1]
        sig_min, sig_max = np.min(signal), np.max(signal)
        if sig_max - sig_min < 1e-8:
            continue  # Skip constant signals
        normalized = (signal - sig_min) / (sig_max - sig_min)

        closed_mask = normalized < closed_threshold
        open_mask = normalized >= closed_threshold

        if np.any(closed_mask):
            all_closed_values.extend(normalized[closed_mask].tolist())
        if np.any(open_mask):
            all_open_values.extend(normalized[open_mask].tolist())

        # Use phase detection to find interact phases
        # Simplified inline version to avoid circular dependency
        threshold_enter = closed_threshold - hysteresis_offset
        threshold_exit = closed_threshold + hysteresis_offset

        phase = "grasp"
        last_change_idx = 0
        num_frames = len(normalized)

        for step in range(num_frames):
            if phase == "grasp":
                if normalized[step] < threshold_enter:
                    phase = "interact"
                    last_change_idx = step
            elif phase == "interact":
                if normalized[step] >= threshold_exit:
                    # End of interact phase
                    duration = step - last_change_idx
                    if duration > 0:
                        interact_durations_frames.append(duration)
                        interact_durations_ratio.append(duration / num_frames)
                    phase = "grasp"
                    last_change_idx = step

        # Handle case where trajectory ends in interact phase
        if phase == "interact":
            duration = num_frames - last_change_idx
            if duration > 0:
                interact_durations_frames.append(duration)
                interact_durations_ratio.append(duration / num_frames)

    stats = {
        "mean_closed": float(np.mean(all_closed_values)) if all_closed_values else 0.2,
        "std_closed": float(np.std(all_closed_values)) if all_closed_values else 0.1,
        "min_closed": float(np.min(all_closed_values)) if all_closed_values else 0.0,
        "mean_open": float(np.mean(all_open_values)) if all_open_values else 0.8,
        "std_open": float(np.std(all_open_values)) if all_open_values else 0.1,
        "avg_interact_duration_frames": float(np.mean(interact_durations_frames)) if interact_durations_frames else 50.0,
        "std_interact_duration_frames": float(np.std(interact_durations_frames)) if interact_durations_frames else 20.0,
        "avg_interact_duration_ratio": float(np.mean(interact_durations_ratio)) if interact_durations_ratio else 0.3,
        "num_interact_phases": len(interact_durations_frames),
        "num_trajectories_sampled": len(gripper_signals),
    }
    return stats


def load_or_compute_gripper_stats(dataset_root, dataloader, trajectory_paths, num_samples=200):
    """
    Load cached gripper stats from disk, or compute from sampled trajectories.
    
    Args:
        dataset_root: Path to dataset root directory
        dataloader: Dataset loader instance with load_trajectory() method
        trajectory_paths: List of trajectory paths (already computed)
        num_samples: Number of random trajectories to sample for computing stats
    
    Returns:
        dict with gripper statistics
    """
    cache_path = os.path.join(dataset_root, "gripper_stats.json")

    # Try to load from cache
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                stats = json.load(f)
            logging.info(f"Loaded cached gripper stats from {cache_path}")
            return stats
        except (json.JSONDecodeError, IOError) as e:
            logging.warning(f"Failed to load gripper stats cache: {e}")

    # Compute from sampled trajectories
    logging.info(f"Computing gripper stats from {num_samples} sampled trajectories...")

    # Sample random trajectories from provided paths
    sample_size = min(num_samples, len(trajectory_paths))
    sampled_paths = random.sample(list(trajectory_paths), sample_size)

    gripper_signals = []
    for path in tqdm(sampled_paths):
        try:
            trajectory = dataloader.load_trajectory(path, skip_frames=True,skip_observations=False)
            if trajectory is not None and hasattr(trajectory, 'gripper_state'):
                gripper_signals.append(trajectory.gripper_state)
        except Exception as e:
            logging.debug(f"Failed to load trajectory {path} for stats: {e}")
            continue

    if len(gripper_signals) == 0:
        logging.warning("No gripper signals collected, using default stats")
        stats = {
            "mean_closed": 0.2,
            "std_closed": 0.1,
            "min_closed": 0.0,
            "mean_open": 0.8,
            "std_open": 0.1,
            "avg_close_duration_ratio": 0.5,
            "num_trajectories_sampled": 0,
        }
    else:
        stats = compute_gripper_stats(gripper_signals)

    # Save to cache
    try:
        with open(cache_path, "w") as f:
            json.dump(stats, f, indent=2)
        logging.info(f"Saved gripper stats to {cache_path}")
    except IOError as e:
        logging.warning(f"Failed to save gripper stats cache: {e}")

    return stats


def get_gripper_close_phases(gripper_actions, closed_threshold=0.7, expected_interact_duration=None, hysteresis_offset=0.1, min_phase_duration=7):
    """
    Detect gripper phases: grasp, interact, release, grasp_failure.
    
    Two-pass slippage detection: first detect candidate interact phases, then compute a
    robust per-interaction duration (median) and mark only the short outliers as grasp_failure.
    This prevents long multi-pick demos with short but valid grasps from being flagged.
    
    Args:
        gripper_actions: 1D array of gripper states. 0 = fully closed, 1 = fully open (after normalization).
        closed_threshold: Base threshold for closed gripper (on normalized [0,1] scale)
        expected_interact_duration: Expected total interact duration budget (frames) for the
                                   whole trajectory. If None, uses 10% of the trajectory.
        hysteresis_offset: Offset for hysteresis. Enter interact at (closed_threshold - offset),
                          exit interact at (closed_threshold + offset) to prevent oscillation.
        min_phase_duration: Minimum number of consecutive frames required to confirm a phase
                           transition (prevents spurious transitions from noise)
    
    Returns:
        List of tuples (start_idx, end_idx, phase_name)
    """
    num_frames = len(gripper_actions)
    if expected_interact_duration is None:
        # Default: expect interact to be about 10% of the full trajectory
        expected_interact_duration = 0.07 * num_frames


    closed_threshold = 0.6

    # Hysteresis thresholds
    threshold_enter_interact = closed_threshold - hysteresis_offset  # Enter interact when below this
    threshold_exit_interact = closed_threshold + hysteresis_offset   # Exit interact when above this

    # Convert to numpy array
    gripper_actions = np.array(gripper_actions)

    # Check if there's actual gripper activity (closing/opening)
    gripper_min = np.min(gripper_actions)
    gripper_max = np.max(gripper_actions)
    gripper_range = gripper_max - gripper_min

    # If the range is too small, there's no real gripper activity
    # Return a single interact phase for the entire trajectory
    if gripper_range < 0.05:  # Threshold for "no real gripper movement"
        return [(0, num_frames - 1, "interact")]

    # Check if gripper actually closes in absolute terms before normalization
    # If min is very close to max AND both are high, it's just noise while open
    # Example: min=0.95, max=1.0 -> gripper never closed, just minor wiggle
    # Example: min=0.748, max=1.009 -> gripper closed significantly (range=0.26)
    # Use combined check: small range AND high minimum
    if gripper_range < 0.1 and gripper_min > 0.9:
        # Gripper stays open with minimal movement (e.g., 0.95-1.0 range)
        return [(0, num_frames - 1, "interact")]

    # Normalize gripper actions to [0, 1]
    # After normalization: 0 = most closed position, 1 = most open position
    gripper_actions = (gripper_actions - gripper_min) / (gripper_range + 1e-8)

    # ---------- Pass 1: detect candidate interact spans (no failure labeling) ----------
    candidate_interacts = []
    phases = []
    phase = "grasp"
    last_change_idx = 0
    consecutive_counter = 0  # Count consecutive frames meeting transition criteria
    pending_phase = None     # Phase we're considering transitioning to

    min_phase_duration = min(min_phase_duration, int(0.2 * num_frames))

    min_phase_duration_init = min(5, int(0.05 * num_frames))


    for step in range(len(gripper_actions)):
        if phase == "grasp":
            if gripper_actions[step] < threshold_enter_interact:
                if pending_phase != "interact":
                    pending_phase = "interact"
                    consecutive_counter = 1
                else:
                    consecutive_counter += 1

                if consecutive_counter >= min_phase_duration_init:
                    transition_start = step - consecutive_counter + 1

                    # --- NEW: RETROACTIVE ADJUSTMENT ---
                    # Calculate how long the current grasp would be


                    if transition_start > last_change_idx:
                        phases.append((last_change_idx, transition_start - 1, "grasp"))
                    phase = "interact"
                    last_change_idx = transition_start
                    pending_phase = None
                    consecutive_counter = 0
            else:
                pending_phase = None
                consecutive_counter = 0

        elif phase == "interact":
            if gripper_actions[step] >= threshold_exit_interact:
                if pending_phase != "release":
                    pending_phase = "release"
                    consecutive_counter = 1
                else:
                    consecutive_counter += 1


                if consecutive_counter >= min_phase_duration_init:
                    #Confirmed transition to release
                    transition_start = step - consecutive_counter + 1
                    interact_start = last_change_idx
                    interact_end = transition_start - 1
                    if interact_end >= interact_start:
                        duration = interact_end - interact_start + 1
                        phases.append((interact_start, interact_end, "interact"))
                        candidate_interacts.append(duration)
                    phase = "release"
                    last_change_idx = transition_start
                    pending_phase = None
                    consecutive_counter = 0
            else:
                # Reset if condition no longer met
                pending_phase = None
                consecutive_counter = 0

        elif phase == "release":
                # 1. Define your fixed duration (e.g., half the min interaction time)
                # Ensure min_valid_duration is defined in your variables
                release_timeout = min_phase_duration // 2

                # 2. Check how long we have been in release
                current_duration = step - last_change_idx

                # 3. Check for early trigger: If gripper closes BEFORE timeout,
                # we must exit release immediately to catch the next grasp.
                early_trigger = gripper_actions[step] < threshold_enter_interact

                if current_duration >= release_timeout or early_trigger:
                    # Save the Release phase
                    # It ends at the previous step (step - 1)
                    if step > last_change_idx:
                        phases.append((last_change_idx, step - 1, "release"))

                    # Switch back to Grasp (Idle/Approach)
                    phase = "grasp"
                    last_change_idx = step

                    # Reset counters
                    pending_phase = None
                    consecutive_counter = 0

    # Handle final phase
    if phase == "interact":
        phases.append((last_change_idx, len(gripper_actions) - 1, "interact"))
        candidate_interacts.append(len(gripper_actions) - last_change_idx)
    elif pending_phase == "interact":
        #check if gripper was closed until the end, if so append grasp -> interact with consecutive counter
        is_last_frame_closed = gripper_actions[-1] < threshold_enter_interact
        if is_last_frame_closed and consecutive_counter >= min_phase_duration//2:
            transition_start = len(gripper_actions) - consecutive_counter
            if transition_start > last_change_idx:
                phases.append((last_change_idx, transition_start - 1, "grasp"))
            phases.append((transition_start, len(gripper_actions) - 1, "interact"))
            candidate_interacts.append( len(gripper_actions) - transition_start)


    # Compute robust per-interaction expectation
    num_attempts = max(1, len(candidate_interacts))
    per_attempt_budget = expected_interact_duration / num_attempts
    median_interact = float(np.median(candidate_interacts)) if candidate_interacts else None
    # Base threshold: half of per-attempt budget; fall back to median-based if available
    min_valid_duration = per_attempt_budget * 0.5
    if median_interact is not None:
        min_valid_duration = max(min_valid_duration, 0.4 * median_interact)
    # Clamp to avoid extreme values
    per_attempt_span = num_frames / num_attempts
    min_valid_duration = max(4, min_valid_duration)
    min_valid_duration = min(min_valid_duration, 0.3 * per_attempt_span)
    if len(candidate_interacts) == 1:
        min_valid_duration = candidate_interacts[0] -1

    # ---------- Pass 2: re-run with failure labeling using the robust threshold ----------
    phases = []
    phase = "grasp"
    last_change_idx = 0
    consecutive_counter = 0
    pending_phase = None

    if len(candidate_interacts) > 0:
        if min_phase_duration > np.min(candidate_interacts):
            min_phase_duration = np.min(candidate_interacts)

    for step in range(len(gripper_actions)):
        if phase in ["grasp", "grasp_failure"]:
            if gripper_actions[step] < threshold_enter_interact:
                if pending_phase != "interact":
                    pending_phase = "interact"
                    consecutive_counter = 1
                else:
                    consecutive_counter += 1

                if consecutive_counter >= min_phase_duration:
                    transition_start = step - consecutive_counter + 1

                    # --- NEW: RETROACTIVE ADJUSTMENT ---
                    # Calculate how long the current grasp would be
                    current_grasp_duration = transition_start - last_change_idx
                    target_grasp_duration = min_valid_duration # Or set your own int, e.g. 20

                    # If grasp is too short, and previous phase was 'release', steal time
                    if current_grasp_duration < target_grasp_duration and phases and phases[-1][2] == "release":
                        needed = target_grasp_duration - current_grasp_duration
                        prev_start, prev_end, prev_label = phases[-1]
                        prev_len = prev_end - prev_start + 1

                        # We must leave at least 1 frame of release
                        min_release_residue = max(1, int(min_phase_duration) // 2)
                        # Can only steal up to (prev_len - min_release_residue), but also
                        # ensure we don't make end < start (need at least 1 frame: end >= start)
                        max_stealable = prev_len - min_release_residue
                        available_to_steal = max(0, min(max_stealable, prev_end - prev_start))

                        # Convert to int to avoid numpy type issues (e.g., np.int64)
                        steal_amount = int(min(needed, available_to_steal))

                        if steal_amount > 0:
                            new_end = int(prev_end - steal_amount)
                            # Safety check: only steal if resulting phase is valid
                            if new_end >= prev_start:
                                # 1. Shorten the previous release in the list
                                phases[-1] = (prev_start, new_end, prev_label)

                                # 2. Move the start of the current grasp backwards
                                last_change_idx -= steal_amount


                    if transition_start > last_change_idx:
                        phases.append((last_change_idx, transition_start - 1, "grasp"))
                    phase = "interact"
                    last_change_idx = transition_start
                    pending_phase = None
                    consecutive_counter = 0
            else:
                pending_phase = None
                consecutive_counter = 0

        elif phase == "interact":
            if gripper_actions[step] >= threshold_exit_interact:
                if pending_phase != "release":
                    pending_phase = "release"
                    consecutive_counter = 1
                else:
                    consecutive_counter += 1

                if consecutive_counter >= min_phase_duration:
                    transition_start = step - consecutive_counter + 1
                    interact_duration = transition_start - last_change_idx
                    is_valid = interact_duration >= min_valid_duration

                    if is_valid:
                        if transition_start > last_change_idx:
                            phases.append((last_change_idx, transition_start - 1, "interact"))
                        phase = "release"
                    else:
                        if transition_start > last_change_idx:
                            phases.append((last_change_idx, transition_start - 1, "grasp_failure"))
                        phase = "grasp"
                    last_change_idx = transition_start
                    pending_phase = None
                    consecutive_counter = 0
            else:
                pending_phase = None
                consecutive_counter = 0

        elif phase == "release":
                # 1. Define your fixed duration (e.g., half the min interaction time)
                # Ensure min_valid_duration is defined in your variables
                release_timeout = min_phase_duration // 2

                # 2. Check how long we have been in release
                current_duration = step - last_change_idx

                # 3. Check for early trigger: If gripper closes BEFORE timeout,
                # we must exit release immediately to catch the next grasp.
                early_trigger = gripper_actions[step] < threshold_enter_interact

                if current_duration >= release_timeout or early_trigger:
                    # Save the Release phase
                    # It ends at the previous step (step - 1)
                    if step > last_change_idx:
                        phases.append((last_change_idx, step - 1, "release"))

                    # Switch back to Grasp (Idle/Approach)
                    phase = "grasp"
                    last_change_idx = step

                    # Reset counters
                    pending_phase = None
                    consecutive_counter = 0

    # Handle final phase
    if phase == "interact":
        interact_duration = len(gripper_actions) - last_change_idx
        if interact_duration >= min_valid_duration:
            phases.append((last_change_idx, len(gripper_actions) - 1, "interact"))
    elif pending_phase == "interact":
        #check if gripper was closed until the end, if so append grasp -> interact with consecutive counter
        is_last_frame_closed = gripper_actions[-1] < threshold_enter_interact
        if is_last_frame_closed and consecutive_counter >= min_phase_duration//2:
            transition_start = len(gripper_actions) - consecutive_counter
            if transition_start > last_change_idx:
                phases.append((last_change_idx, transition_start - 1, "grasp"))
            phases.append((transition_start, len(gripper_actions) - 1, "interact"))


    elif last_change_idx < len(gripper_actions):
        if phase == "grasp" and pending_phase != "interact":
            pass
        else:
            phases.append((last_change_idx, len(gripper_actions) - 1, phase))

    if len(phases) == 0:
        return [(0, len(gripper_actions) - 1, "interact")]
    #check if last pahse is "release", change last index to frame len -1
    if phases[-1][2] == "release":
        phases[-1] = (phases[-1][0], len(gripper_actions) - 1, "release")

    # If the trajectory ends with a spurious terminal grasp after a release (open gripper
    # after release_timeout fired but no real re-grasp followed), absorb it into the release.
    if (len(phases) >= 2
            and phases[-1][2] == "grasp"
            and phases[-2][2] == "release"):
        phases[-2] = (phases[-2][0], len(gripper_actions) - 1, "release")
        phases.pop()


    return phases
