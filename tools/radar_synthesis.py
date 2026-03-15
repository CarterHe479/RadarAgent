"""
Simplified radar point cloud synthesis from SMPL joint positions.

Mirrors the paper's physics-aware pipeline (Section 3):
  1. Surface sampling  ≈ ray-mesh intersection
  2. Intensity model   ≈ IF signal power (simplified RCS: I ∝ 1/r²)
  3. Top-K selection   ≈ peak detection on Doppler-FFT heatmap
  4. Time coordinate   ≈ timestamp per frame

Paper uses 128 points per frame at 20 FPS.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

_RNG = np.random.default_rng(42)


def sample_surface_points(
    joints_frame: np.ndarray,
    num_surface: int = 1024,
    jitter_std: float = 0.02,
) -> np.ndarray:
    """Approximate body-surface sampling from 22 SMPL joints.

    Args:
        joints_frame: (22, 3) joint positions in metres.
        num_surface:  Number of surface points to sample.
        jitter_std:   Gaussian noise std in metres (≈ body thickness).

    Returns:
        (num_surface, 3) approximate surface points.
    """
    J = joints_frame.shape[0]
    idx = _RNG.integers(low=0, high=J, size=num_surface)
    base = joints_frame[idx]                                   # (M, 3)
    noise = _RNG.normal(0.0, jitter_std, size=(num_surface, 3)).astype(np.float32)
    return (base + noise).astype(np.float32)


def compute_intensity(
    points: np.ndarray,
    radar_pos: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute simplified radar intensity: I ∝ 1 / r².

    Args:
        points:    (M, 3) 3-D point positions.
        radar_pos: (3,) radar antenna position; defaults to origin.

    Returns:
        (M,) intensity values (higher = closer to radar).
    """
    if radar_pos is None:
        radar_pos = np.zeros(3, dtype=np.float32)
    vec = points - radar_pos[None, :]
    r = np.linalg.norm(vec, axis=1) + 1e-6
    return (1.0 / (r ** 2)).astype(np.float32)


def select_top_k(
    points: np.ndarray,
    intensities: np.ndarray,
    k: int = 128,
) -> np.ndarray:
    """Select the k highest-intensity points (mimics Doppler-FFT peak picking).

    Args:
        points:      (M, 3) candidate points.
        intensities: (M,) corresponding intensity values.
        k:           Number of points to keep.

    Returns:
        (k, 3) selected points.
    """
    idx = np.argsort(-intensities)[:k]
    return points[idx].astype(np.float32)


def motion_to_radar_pointcloud(
    joints: np.ndarray,
    fps: int = 20,
    N: int = 128,
    num_surface: int = 1024,
    jitter_std: float = 0.02,
    radar_pos: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Convert a sequence of SMPL joint positions to a radar point cloud.

    Implements the simplified pipeline described in the implementation plan
    (Section 3.2), mirroring the paper's physics-aware synthesis.

    Args:
        joints:      (T, 22, 3) SMPL joint positions over T frames.
        fps:         Frames per second (paper uses 20).
        N:           Points to keep per frame (paper uses 128).
        num_surface: Candidate surface points per frame.
        jitter_std:  Gaussian noise std for surface approximation.
        radar_pos:   (3,) radar position; defaults to origin.

    Returns:
        (T, N, 4) point cloud sequence, with each point as (x, y, z, t).
    """
    if radar_pos is None:
        radar_pos = np.zeros(3, dtype=np.float32)

    T = joints.shape[0]
    out = np.zeros((T, N, 4), dtype=np.float32)

    for t in range(T):
        surface = sample_surface_points(joints[t], num_surface, jitter_std)
        intensities = compute_intensity(surface, radar_pos)
        selected = select_top_k(surface, intensities, N)          # (N, 3)
        t_col = np.full((N, 1), t / float(fps), dtype=np.float32)
        out[t] = np.concatenate([selected, t_col], axis=1)        # (N, 4)

    return out
