"""
Core radar processing tools: load_radar_sequence and extract_radar_features.

load_radar_sequence  – loads or synthesises a (T, 128, 4) point cloud and
                       returns lightweight summary statistics.
extract_radar_features – computes the full spatiotemporal feature set that
                         the LLM agent reasons over.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from scipy.signal import find_peaks

from config import (
    JOINTS_DIR,
    SYNTHETIC_POINTS_DIR,
    RADAR_FPS,
    POINTS_PER_FRAME,
    SURFACE_SAMPLE_COUNT,
    SURFACE_JITTER_STD,
    RADAR_POSITION,
)
from tools.radar_synthesis import motion_to_radar_pointcloud


# ── helpers ───────────────────────────────────────────────────────────────────

def _resolve_points_path(motion_id: str) -> Optional[Path]:
    """Return the path to pre-computed synthetic points, or None."""
    candidate = SYNTHETIC_POINTS_DIR / f"rec_{motion_id}.npy"
    if candidate.exists():
        return candidate
    return None


def _load_or_synthesise(motion_id: str) -> np.ndarray:
    """Return (T, 128, 4) point cloud for *motion_id*.

    Priority:
      1. Pre-computed synthetic points in RadarLLM-data/synthetic_points/
      2. On-the-fly synthesis from new_joints/{motion_id}.npy
    """
    pts_path = _resolve_points_path(motion_id)
    if pts_path is not None:
        return np.load(pts_path).astype(np.float32)

    joint_path = JOINTS_DIR / f"{motion_id}.npy"
    if not joint_path.exists():
        raise FileNotFoundError(
            f"No data found for motion_id={motion_id!r}. "
            f"Checked:\n  {SYNTHETIC_POINTS_DIR / f'rec_{motion_id}.npy'}\n  {joint_path}"
        )
    joints = np.load(joint_path).astype(np.float32)   # (T, 22, 3)
    return motion_to_radar_pointcloud(
        joints,
        fps=RADAR_FPS,
        N=POINTS_PER_FRAME,
        num_surface=SURFACE_SAMPLE_COUNT,
        jitter_std=SURFACE_JITTER_STD,
        radar_pos=np.array(RADAR_POSITION, dtype=np.float32),
    )


# ── Tool 1: load_radar_sequence ───────────────────────────────────────────────

def load_radar_sequence(motion_id: str) -> dict:
    """Load or synthesise a radar point cloud and return summary statistics.

    Returns lightweight metadata – NOT raw arrays – so the LLM can read it.
    """
    pts = _load_or_synthesise(motion_id)   # (T, N, 4)
    T, N, _ = pts.shape
    xyz = pts[:, :, :3]                    # (T, N, 3)

    com_start = xyz[0].mean(axis=0).tolist()
    com_end   = xyz[-1].mean(axis=0).tolist()
    displacement = float(np.linalg.norm(
        np.array(com_end) - np.array(com_start)
    ))

    return {
        "motion_id": motion_id,
        "num_frames": int(T),
        "points_per_frame": int(N),
        "fps": RADAR_FPS,
        "duration_sec": round(T / RADAR_FPS, 3),
        "spatial_bounds": {
            "x": {"min": round(float(xyz[:, :, 0].min()), 3),
                  "max": round(float(xyz[:, :, 0].max()), 3)},
            "y": {"min": round(float(xyz[:, :, 1].min()), 3),
                  "max": round(float(xyz[:, :, 1].max()), 3)},
            "z": {"min": round(float(xyz[:, :, 2].min()), 3),
                  "max": round(float(xyz[:, :, 2].max()), 3)},
        },
        "center_of_mass_start": [round(v, 3) for v in com_start],
        "center_of_mass_end":   [round(v, 3) for v in com_end],
        "overall_displacement": round(displacement, 3),
    }


# ── Tool 2: extract_radar_features ────────────────────────────────────────────

def _activity_label(mean_vel: float) -> str:
    if mean_vel < 0.2:
        return "low"
    if mean_vel < 0.8:
        return "medium"
    return "high"


def _complexity_label(vel_std: float) -> str:
    if vel_std < 0.15:
        return "simple"
    if vel_std < 0.40:
        return "moderate"
    return "complex"


def _trajectory_shape(com: np.ndarray) -> str:
    """Classify horizontal trajectory as stationary/linear/curved/back_and_forth."""
    xy = com[:, :2]
    diffs = np.diff(xy, axis=0)                          # (T-1, 2)
    step_lens = np.linalg.norm(diffs, axis=1)
    total_path = float(step_lens.sum())
    net = float(np.linalg.norm(xy[-1] - xy[0]))

    if total_path < 0.3:
        return "stationary"
    ratio = net / (total_path + 1e-9)
    if ratio > 0.80:
        return "linear"
    if ratio < 0.30:
        return "back_and_forth"
    return "curved"


def _periodicity(speed: np.ndarray, fps: int) -> dict:
    """Detect periodicity via autocorrelation of the speed signal."""
    if len(speed) < 6:
        return {"is_periodic": False, "estimated_period_sec": None, "confidence": "low"}

    sig = speed - speed.mean()
    autocorr = np.correlate(sig, sig, mode="full")
    autocorr = autocorr[len(autocorr) // 2:]            # positive lags
    if autocorr[0] == 0:
        return {"is_periodic": False, "estimated_period_sec": None, "confidence": "low"}
    autocorr = autocorr / autocorr[0]

    min_lag = max(int(0.3 * fps), 1)
    if len(autocorr) <= min_lag + 1:
        return {"is_periodic": False, "estimated_period_sec": None, "confidence": "low"}

    peaks, props = find_peaks(autocorr[min_lag:], height=0.0)
    if len(peaks) == 0:
        return {"is_periodic": False, "estimated_period_sec": None, "confidence": "low"}

    best_peak_idx = peaks[np.argmax(props["peak_heights"])]
    peak_val = float(autocorr[best_peak_idx + min_lag])
    period_frames = int(best_peak_idx + min_lag)
    period_sec = round(period_frames / fps, 3)

    if peak_val > 0.5:
        confidence = "high"
    elif peak_val > 0.3:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "is_periodic": peak_val > 0.3,
        "estimated_period_sec": period_sec if peak_val > 0.3 else None,
        "confidence": confidence,
    }


def extract_radar_features(motion_id: str) -> dict:
    """Extract the full spatiotemporal feature set from a radar point cloud.

    This is the primary analytical tool for the agent.  All values are
    human-readable scalars or short strings – no raw arrays.
    """
    pts = _load_or_synthesise(motion_id)   # (T, N, 4)
    T, N, _ = pts.shape
    fps = RADAR_FPS
    xyz = pts[:, :, :3]                    # (T, N, 3)

    # ── centre-of-mass trajectory (T, 3) ─────────────────────────────────────
    com = xyz.mean(axis=1)                 # (T, 3)

    # ── velocity profile ──────────────────────────────────────────────────────
    disp   = np.diff(com, axis=0)          # (T-1, 3)
    speed  = np.linalg.norm(disp, axis=1) * fps  # (T-1,) in m/s
    mean_v = float(speed.mean())
    max_v  = float(speed.max())
    min_v  = float(speed.min())
    std_v  = float(speed.std())

    # ── dominant motion axis ──────────────────────────────────────────────────
    ranges = [
        com[:, 0].max() - com[:, 0].min(),  # x
        com[:, 1].max() - com[:, 1].min(),  # y
        com[:, 2].max() - com[:, 2].min(),  # z
    ]
    axis_names = ["x", "y", "z"]
    dom_axis_idx = int(np.argmax(ranges))
    dom_axis = axis_names[dom_axis_idx]
    dom_range = round(float(ranges[dom_axis_idx]), 3)

    # ── periodicity ───────────────────────────────────────────────────────────
    period_info = _periodicity(speed, fps)

    # ── vertical dynamics ─────────────────────────────────────────────────────
    com_z = com[:, 2]
    vert = {
        "com_height_start_m": round(float(com_z[0]),  3),
        "com_height_end_m":   round(float(com_z[-1]), 3),
        "com_height_min_m":   round(float(com_z.min()), 3),
        "com_height_max_m":   round(float(com_z.max()), 3),
        "vertical_range_m":   round(float(com_z.max() - com_z.min()), 3),
    }

    # ── point-cloud spread (body extent) ─────────────────────────────────────
    spread_per_frame = xyz.std(axis=1).mean(axis=1)   # (T,)  mean std over x/y/z
    mean_spread = float(spread_per_frame.mean())
    spread_start = float(spread_per_frame[:max(T // 4, 1)].mean())
    spread_end   = float(spread_per_frame[-(max(T // 4, 1)):].mean())
    if spread_end > spread_start * 1.1:
        spread_change = "expanding"
    elif spread_end < spread_start * 0.9:
        spread_change = "contracting"
    else:
        spread_change = "stable"

    # ── body-region analysis via z-axis split ─────────────────────────────────
    region_results = {}
    for region, (mask_fn, label) in {
        "upper_body": (lambda z, mz: z > mz,  "upper_body"),
        "lower_body": (lambda z, mz: z <= mz, "lower_body"),
    }.items():
        region_speeds = []
        for t in range(T - 1):
            mz = float(np.median(xyz[t, :, 2]))
            mask_t   = mask_fn(xyz[t,   :, 2], mz)
            mask_t1  = mask_fn(xyz[t+1, :, 2], mz)
            if mask_t.sum() < 2 or mask_t1.sum() < 2:
                continue
            c0 = xyz[t,   mask_t, :].mean(axis=0)
            c1 = xyz[t+1, mask_t1, :].mean(axis=0)
            region_speeds.append(np.linalg.norm(c1 - c0) * fps)
        rv = float(np.mean(region_speeds)) if region_speeds else 0.0
        region_results[region] = {
            "activity_level": _activity_label(rv),
            "mean_velocity_m_per_s": round(rv, 3),
        }

    return {
        "motion_id": motion_id,
        "duration_sec": round(T / fps, 3),
        "velocity": {
            "mean_m_per_s": round(mean_v, 3),
            "max_m_per_s":  round(max_v,  3),
            "min_m_per_s":  round(min_v,  3),
            "std_m_per_s":  round(std_v,  3),
        },
        "dominant_motion_axis":   dom_axis,
        "dominant_axis_range_m":  dom_range,
        "periodicity": period_info,
        "vertical_dynamics": vert,
        "body_spread": {
            "mean_spread_m": round(mean_spread, 3),
            "spread_change": spread_change,
        },
        "upper_body": region_results["upper_body"],
        "lower_body": region_results["lower_body"],
        "motion_complexity": _complexity_label(std_v),
        "trajectory_shape":  _trajectory_shape(com),
    }
