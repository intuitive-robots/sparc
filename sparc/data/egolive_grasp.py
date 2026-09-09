"""Hand-closure signals for EgoLive (human hands, no gripper).

EgoLive records human demonstrations, so the pipeline's gripper-open scalar has
no direct equivalent. ``observation.state`` does carry named fingertip positions
(5 fingers x (position, direction) per hand, plus wrist pose), which lets us
build aperture signals that play the same role: large value = hand open,
small value = hand closed around something.

The polarity is the opposite of a robot gripper, which is easy to get wrong: a
human hand holding something *spreads* its fingers around the object, while a
free hand relaxes into a curl. So a large aperture means "holding" and
:func:`gripper_state` inverts the aperture before returning it on the
pipeline's gripper convention (1 = open, 0 = closed), which lets the result be
fed to ``get_gripper_close_phases`` unchanged.

These hand-shape signals are heuristic fallbacks; the loader prefers
hand-object contact annotations when available.
"""

from typing import Optional

import numpy as np

# Layout of observation.state (78,), from meta/info.json feature names.
_WRIST = {"left": slice(0, 3), "right": slice(9, 12)}
_FINGER_BLOCK = {"left": 18, "right": 48}
_FINGERS = ("thumb", "index", "middle", "ring", "pinky")

SIGNAL_NAMES = ("thumb_index", "thumb_min", "thumb_mean", "curl", "opposition", "combo")
DEFAULT_SIGNAL = "opposition"

# Joint order of the 21-point hand keypoints (`left_kp3d`, `leftcam_left_kp2d`, ...),
# recovered by matching them against the named fingertips in observation.state:
# joint 0 is the wrist and each finger runs base -> tip in four steps.
WRIST_JOINT = 0
FINGERTIP_JOINTS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}


def grasp_center(keypoints: np.ndarray) -> np.ndarray:
    """Midpoint of the thumb and index tips — the human analogue of a gripper TCP.

    Works for both the 3D keypoints (camera frame, metres) and the 2D ones
    (pixels), returning (T, 2) or (T, 3) to match the input.
    """
    keypoints = np.asarray(keypoints, dtype=np.float32)
    thumb = keypoints[:, FINGERTIP_JOINTS["thumb"]]
    index = keypoints[:, FINGERTIP_JOINTS["index"]]
    return 0.5 * (thumb + index)


def _hand_key(hand: str) -> str:
    hand = hand.lower()
    if hand not in ("left", "right"):
        raise ValueError(f"hand must be 'left' or 'right', got {hand!r}")
    return hand


def tracked_mask(state: np.ndarray, hand: str) -> np.ndarray:
    """(T,) True where the hand was actually tracked.

    EgoLive marks lost tracking by zeroing the whole per-hand block rather than
    with a sentinel, and that happens in ~36% of frames. Left unhandled those
    rows look like a hand collapsed to the origin, which silently corrupts any
    aperture or position computed from them.
    """
    base = _FINGER_BLOCK[_hand_key(hand)]
    state = np.asarray(state, dtype=np.float32).reshape(len(state), -1)
    return ~(state[:, base: base + 30] == 0).all(axis=1)


def fingertip_positions(state: np.ndarray, hand: str) -> np.ndarray:
    """Return (T, 5, 3) fingertip xyz for one hand, ordered as ``_FINGERS``."""
    base = _FINGER_BLOCK[_hand_key(hand)]
    state = np.asarray(state, dtype=np.float32).reshape(len(state), -1)
    tips = [state[:, base + 6 * i: base + 6 * i + 3] for i in range(len(_FINGERS))]
    return np.stack(tips, axis=1)


def wrist_positions(state: np.ndarray, hand: str) -> np.ndarray:
    """Return (T, 3) wrist xyz for one hand."""
    state = np.asarray(state, dtype=np.float32).reshape(len(state), -1)
    return state[:, _WRIST[_hand_key(hand)]]


def hand_scale(state: np.ndarray, hand: str) -> float:
    """Robust per-trajectory hand size, used to make apertures dimensionless."""
    tracked = tracked_mask(state, hand)
    if not tracked.any():
        return 1.0
    tips = fingertip_positions(state, hand)[tracked]
    wrist = wrist_positions(state, hand)[tracked][:, None, :]
    span = np.linalg.norm(tips - wrist, axis=-1)
    scale = float(np.nanpercentile(span, 95)) if np.isfinite(span).any() else 0.0
    return scale if scale > 1e-6 else 1.0


def closure_signals(state: np.ndarray, hand: str) -> dict[str, np.ndarray]:
    """Hand-shape signals for one hand, each (T,) and divided by hand scale.

    Larger = fingers further apart = more likely wrapped around an object.
    ``opposition`` measures how far the thumb tip sits out of the plane of the
    other four tips, which is what distinguishes gripping something from a flat
    or loosely curled hand; ``combo`` averages it with the pinch aperture.
    """
    tips = fingertip_positions(state, hand)
    wrist = wrist_positions(state, hand)[:, None, :]
    scale = hand_scale(state, hand)

    thumb = tips[:, 0:1, :]
    others = tips[:, 1:, :]
    thumb_to_others = np.linalg.norm(others - thumb, axis=-1) / scale  # (T, 4)
    curl = np.linalg.norm(others - wrist, axis=-1).mean(axis=1) / scale

    centroid = others.mean(axis=1, keepdims=True)
    plane_normal = np.linalg.svd(others - centroid)[2][:, 2, :]
    opposition = np.abs(((thumb[:, 0] - centroid[:, 0]) * plane_normal).sum(axis=1)) / scale

    signals = {
        "thumb_index": thumb_to_others[:, 0],
        "thumb_min": thumb_to_others.min(axis=1),
        "thumb_mean": thumb_to_others.mean(axis=1),
        "curl": curl,
        "opposition": opposition,
    }
    signals["combo"] = 0.5 * (
        _unit_normalize(signals["thumb_min"]) + _unit_normalize(opposition)
    )

    untracked = ~tracked_mask(state, hand)
    for values in signals.values():
        values[untracked] = np.nan
    return signals


def _unit_normalize(values: np.ndarray) -> np.ndarray:
    """Map a signal onto [0, 1] using robust percentiles of its own range."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    lo, hi = np.percentile(finite, [2, 98])
    if hi - lo < 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def gripper_state(
    state: np.ndarray,
    hand: str,
    signal: str = DEFAULT_SIGNAL,
    smooth_window: int = 5,
) -> Optional[np.ndarray]:
    """Pipeline-convention gripper signal (1 = open, 0 = closed) for one hand."""
    state = np.asarray(state, dtype=np.float32)
    if state.ndim != 2 or state.shape[0] == 0:
        return None
    signals = closure_signals(state, hand)
    if signal not in signals:
        raise ValueError(f"Unknown EgoLive closure signal {signal!r}; choices={SIGNAL_NAMES}")
    # Inverted: a wide hand shape means the fingers are wrapped around an
    # object, which is the human equivalent of a *closed* gripper.
    values = 1.0 - _unit_normalize(signals[signal])
    return smooth(_interpolate_gaps(values), smooth_window)


def _interpolate_gaps(values: np.ndarray) -> np.ndarray:
    """Bridge untracked frames so the smoother does not spread their NaNs."""
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values, dtype=np.float32)
    indices = np.arange(len(values))
    return np.interp(indices, indices[finite], values[finite]).astype(np.float32)


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average with edge padding; a no-op for window <= 1."""
    if window is None or window <= 1 or len(values) < 2:
        return np.asarray(values, dtype=np.float32)
    window = min(int(window), len(values))
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)
