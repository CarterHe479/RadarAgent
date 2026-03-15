"""
analyze_joint_motion – fine-grained body-part analysis using the 22-joint
SMPL skeleton from HumanML3D new_joints/*.npy files.

Joint layout (from paramUtil.py):
  0: pelvis (root)          1: left_hip       2: right_hip
  3: spine1                 4: left_knee      5: right_knee
  6: spine2                 7: left_ankle     8: right_ankle
  9: spine3 (chest)        10: left_foot     11: right_foot
 12: neck                  13: left_collar   14: right_collar
 15: head                  16: left_shoulder 17: right_shoulder
 18: left_elbow            19: right_elbow
 20: left_wrist            21: right_wrist
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np

from config import JOINTS_DIR, BODY_PART_JOINTS, RADAR_FPS


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_joints(motion_id: str) -> np.ndarray:
    path = JOINTS_DIR / f"{motion_id}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Joint file not found: {path}")
    return np.load(path).astype(np.float32)   # (T, 22, 3)


def _joint_speeds(joints: np.ndarray, fps: int = RADAR_FPS) -> np.ndarray:
    """Per-joint mean speed over the sequence.

    Returns:
        (22,) mean speed in m/s for each joint.
    """
    disp = np.diff(joints, axis=0)              # (T-1, 22, 3)
    speed = np.linalg.norm(disp, axis=2) * fps  # (T-1, 22)
    return speed.mean(axis=0)                   # (22,)


def _part_mean_speed(joint_speeds: np.ndarray, part: str) -> float:
    idxs = BODY_PART_JOINTS[part]
    return float(joint_speeds[idxs].mean())


def _part_max_displacement(joints: np.ndarray, part: str) -> float:
    idxs = BODY_PART_JOINTS[part]
    start = joints[0, idxs, :]              # (k, 3)
    diffs = joints[:, idxs, :] - start[None, :, :]  # (T, k, 3)
    dists = np.linalg.norm(diffs, axis=2)   # (T, k)
    return float(dists.max())


def _activity_label(mean_speed: float) -> str:
    if mean_speed < 0.15:
        return "low"
    if mean_speed < 0.60:
        return "medium"
    return "high"


# ── Action pattern detectors ──────────────────────────────────────────────────

def _detect_walk(joints: np.ndarray, fps: int) -> bool:
    """Walk: alternating left/right foot, periodic, and forward displacement."""
    # foot joints: 10 (left), 11 (right)
    left_foot  = joints[:, 10, :]
    right_foot = joints[:, 11, :]
    pelvis     = joints[:,  0, :]

    # forward displacement of root
    pelvis_disp = np.linalg.norm(pelvis[-1, :2] - pelvis[0, :2])
    if pelvis_disp < 0.3:
        return False

    # alternation: left and right foot should have out-of-phase z oscillation
    left_z  = left_foot[:, 2]  - left_foot[:, 2].mean()
    right_z = right_foot[:, 2] - right_foot[:, 2].mean()
    # cross-correlation at zero lag should be negative (anti-phase)
    if len(left_z) < 4:
        return False
    cross = float(np.corrcoef(left_z, right_z)[0, 1])
    return cross < -0.15


def _detect_run(joints: np.ndarray, fps: int) -> bool:
    """Run: like walk but faster pelvis speed (> 2 m/s)."""
    pelvis = joints[:, 0, :]
    disp = np.diff(pelvis, axis=0)
    speed = np.linalg.norm(disp, axis=1) * fps
    return bool(speed.mean() > 2.0) and _detect_walk(joints, fps)


def _detect_jump(joints: np.ndarray) -> bool:
    """Jump: pelvis z rises by ≥ 0.15 m above its minimum and returns."""
    z = joints[:, 0, 2]
    z_range = z.max() - z.min()
    return z_range >= 0.15 and z.argmax() not in (0, len(z) - 1)


def _detect_squat_crouch(joints: np.ndarray) -> bool:
    """Squat/crouch: pelvis z drops ≥ 0.10 m below starting value."""
    z = joints[:, 0, 2]
    return float(z[0] - z.min()) >= 0.10


def _detect_arm_raise(joints: np.ndarray) -> Tuple[bool, bool]:
    """Detect arm raise (left and/or right).

    Condition: wrist z rises ≥ 0.20 m above shoulder z.
    Returns: (left_raised, right_raised)
    """
    left_shoulder  = joints[:, 16, 2]   # joint 16
    right_shoulder = joints[:, 17, 2]   # joint 17
    left_wrist     = joints[:, 20, 2]   # joint 20
    right_wrist    = joints[:, 21, 2]   # joint 21

    left_raise  = float((left_wrist  - left_shoulder).max())  >= 0.20
    right_raise = float((right_wrist - right_shoulder).max()) >= 0.20
    return left_raise, right_raise


def _detect_kick(joints: np.ndarray) -> Tuple[bool, bool]:
    """Detect kick: foot joint has high velocity spike and z increases."""
    fps = RADAR_FPS
    left_foot  = joints[:, 10, :]
    right_foot = joints[:, 11, :]

    def _has_kick(foot: np.ndarray) -> bool:
        speed = np.linalg.norm(np.diff(foot, axis=0), axis=1) * fps
        z_delta = float(foot[:, 2].max() - foot[:, 2].min())
        return bool(speed.max() > 3.0 and z_delta > 0.20)

    return _has_kick(left_foot), _has_kick(right_foot)


def _detect_turn(joints: np.ndarray) -> bool:
    """Turn: the heading direction of the pelvis→chest vector rotates ≥ 60°."""
    pelvis = joints[:, 0, :2]
    chest  = joints[:, 9, :2]
    heading = chest - pelvis                          # (T, 2)
    norms = np.linalg.norm(heading, axis=1, keepdims=True) + 1e-8
    heading = heading / norms
    # angle between first and last heading
    cos_theta = float(np.clip(np.dot(heading[0], heading[-1]), -1.0, 1.0))
    angle_deg = float(np.degrees(np.arccos(cos_theta)))
    return angle_deg >= 60.0


def _symmetry(joint_speeds: np.ndarray) -> str:
    left_speed  = float(joint_speeds[[13, 16, 18, 20, 1, 4, 7, 10]].mean())
    right_speed = float(joint_speeds[[14, 17, 19, 21, 2, 5, 8, 11]].mean())
    ratio = left_speed / (right_speed + 1e-9)
    if ratio > 1.25:
        return "left_dominant"
    if ratio < 0.80:
        return "right_dominant"
    return "symmetric"


def _root_trajectory_description(joints: np.ndarray) -> str:
    """Human-readable summary of the pelvis path."""
    pelvis = joints[:, 0, :]
    net_xy = np.linalg.norm(pelvis[-1, :2] - pelvis[0, :2])
    path_xy = float(np.linalg.norm(np.diff(pelvis[:, :2], axis=0), axis=1).sum())
    z_range = float(pelvis[:, 2].max() - pelvis[:, 2].min())

    parts = []
    if net_xy < 0.3:
        parts.append("remains approximately stationary")
    elif net_xy / (path_xy + 1e-9) > 0.8:
        parts.append(f"moves in a straight path (~{net_xy:.2f} m)")
    else:
        parts.append(f"follows a curved path (~{net_xy:.2f} m net displacement)")
    if z_range > 0.15:
        parts.append(f"with {z_range:.2f} m vertical variation")
    return "; ".join(parts)


# ── Tool 6: analyze_joint_motion ─────────────────────────────────────────────

def analyze_joint_motion(motion_id: str) -> dict:
    """Fine-grained skeleton analysis using the 22-joint SMPL data.

    Useful when the agent needs detail beyond what radar point cloud statistics
    can reveal (e.g., which specific limb is acting, left-right asymmetry).
    """
    joints = _load_joints(motion_id)   # (T, 22, 3)
    fps    = RADAR_FPS

    joint_speeds = _joint_speeds(joints, fps)

    # ── per body-part stats ───────────────────────────────────────────────────
    body_part_activity = {}
    for part in BODY_PART_JOINTS:
        vel = _part_mean_speed(joint_speeds, part)
        disp = _part_max_displacement(joints, part)
        body_part_activity[part] = {
            "velocity": round(vel, 3),
            "displacement": round(disp, 3),
            "activity": _activity_label(vel),
        }

    # ── most active parts ────────────────────────────────────────────────────
    sorted_parts = sorted(
        body_part_activity.items(),
        key=lambda kv: kv[1]["velocity"],
        reverse=True,
    )
    most_active = [p for p, _ in sorted_parts[:2]]

    # ── action detection ──────────────────────────────────────────────────────
    detected: List[str] = []
    if _detect_run(joints, fps):
        detected.append("running")
    elif _detect_walk(joints, fps):
        detected.append("walking")
    if _detect_jump(joints):
        detected.append("jumping")
    if _detect_squat_crouch(joints):
        detected.append("squatting_or_crouching")
    left_raise, right_raise = _detect_arm_raise(joints)
    if left_raise:
        detected.append("left_arm_raise")
    if right_raise:
        detected.append("right_arm_raise")
    left_kick, right_kick = _detect_kick(joints)
    if left_kick:
        detected.append("left_kick")
    if right_kick:
        detected.append("right_kick")
    if _detect_turn(joints):
        detected.append("turning")

    return {
        "motion_id": motion_id,
        "body_part_activity": body_part_activity,
        "most_active_parts": most_active,
        "detected_actions": detected if detected else ["no_specific_action_detected"],
        "symmetry": _symmetry(joint_speeds),
        "root_trajectory": _root_trajectory_description(joints),
    }
