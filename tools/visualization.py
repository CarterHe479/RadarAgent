"""
visualize_motion – generate matplotlib figures for radar point clouds,
skeleton poses, and centre-of-mass trajectories.

Saves PNG files to outputs/viz/ and returns the file path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401  (registers 3D projection)

from config import VIZ_DIR, JOINTS_DIR, RADAR_FPS, KINEMATIC_CHAINS
from tools.radar_processing import _load_or_synthesise


# ── helpers ───────────────────────────────────────────────────────────────────

def _evenly_spaced_frames(T: int, n: int) -> list[int]:
    if T <= n:
        return list(range(T))
    return [int(round(i * (T - 1) / (n - 1))) for i in range(n)]


# ── point-cloud view ──────────────────────────────────────────────────────────

def _plot_point_cloud(motion_id: str, num_frames: int, out_path: Path) -> None:
    pts = _load_or_synthesise(motion_id)        # (T, 128, 4)
    T = pts.shape[0]
    frames = _evenly_spaced_frames(T, num_frames)

    fig = plt.figure(figsize=(3 * num_frames, 4), facecolor="#111111")
    fig.suptitle(f"Radar Point Cloud  –  motion {motion_id}", color="white", fontsize=11)

    all_z = pts[:, :, 2]
    z_min, z_max = float(all_z.min()), float(all_z.max())

    for i, t in enumerate(frames):
        ax = fig.add_subplot(1, num_frames, i + 1, projection="3d",
                             facecolor="#111111")
        xyz = pts[t, :, :3]
        c = plt.cm.plasma((xyz[:, 2] - z_min) / (z_max - z_min + 1e-9))
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=c, s=4, depthshade=False)
        ax.set_title(f"t={t/RADAR_FPS:.2f}s", color="white", fontsize=7)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(colors="gray", labelsize=5)
        ax.set_xlabel("x", color="gray", fontsize=6)
        ax.set_ylabel("y", color="gray", fontsize=6)
        ax.set_zlabel("z", color="gray", fontsize=6)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ── skeleton view ─────────────────────────────────────────────────────────────

_CHAIN_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]


def _plot_skeleton(motion_id: str, num_frames: int, out_path: Path) -> None:
    joint_path = JOINTS_DIR / f"{motion_id}.npy"
    if not joint_path.exists():
        raise FileNotFoundError(f"Joint file not found: {joint_path}")
    joints = np.load(joint_path).astype(np.float32)    # (T, 22, 3)
    T = joints.shape[0]
    frames = _evenly_spaced_frames(T, num_frames)

    fig = plt.figure(figsize=(3 * num_frames, 4), facecolor="#111111")
    fig.suptitle(f"Skeleton  –  motion {motion_id}", color="white", fontsize=11)

    for i, t in enumerate(frames):
        ax = fig.add_subplot(1, num_frames, i + 1, projection="3d",
                             facecolor="#111111")
        j = joints[t]   # (22, 3)

        # draw bones
        for chain_idx, chain in enumerate(KINEMATIC_CHAINS):
            color = _CHAIN_COLORS[chain_idx % len(_CHAIN_COLORS)]
            for a, b in zip(chain[:-1], chain[1:]):
                ax.plot(
                    [j[a, 0], j[b, 0]],
                    [j[a, 1], j[b, 1]],
                    [j[a, 2], j[b, 2]],
                    color=color, linewidth=1.5,
                )
        # draw joints
        ax.scatter(j[:, 0], j[:, 1], j[:, 2], c="white", s=10, zorder=5)

        ax.set_title(f"t={t/RADAR_FPS:.2f}s", color="white", fontsize=7)
        ax.tick_params(colors="gray", labelsize=5)
        ax.set_xlabel("x", color="gray", fontsize=6)
        ax.set_ylabel("y", color="gray", fontsize=6)
        ax.set_zlabel("z", color="gray", fontsize=6)
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ── trajectory view ───────────────────────────────────────────────────────────

def _plot_trajectory(motion_id: str, out_path: Path) -> None:
    pts = _load_or_synthesise(motion_id)        # (T, 128, 4)
    T = pts.shape[0]
    com = pts[:, :, :3].mean(axis=1)            # (T, 3)
    t_axis = np.arange(T) / RADAR_FPS

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor="#111111")
    fig.suptitle(f"CoM Trajectory  –  motion {motion_id}", color="white", fontsize=11)

    # top-down view (x, y)
    ax = axes[0]
    ax.set_facecolor("#111111")
    sc = ax.scatter(com[:, 0], com[:, 1], c=t_axis, cmap="plasma", s=10)
    ax.plot(com[:, 0], com[:, 1], color="gray", linewidth=0.7, alpha=0.5)
    ax.scatter(com[0, 0], com[0, 1], marker="o", s=60, color="#2ecc71",
               zorder=5, label="start")
    ax.scatter(com[-1, 0], com[-1, 1], marker="X", s=60, color="#e74c3c",
               zorder=5, label="end")
    ax.set_xlabel("x (m)", color="gray", fontsize=8)
    ax.set_ylabel("y (m)", color="gray", fontsize=8)
    ax.set_title("Top-down (xy)", color="white", fontsize=9)
    ax.tick_params(colors="gray", labelsize=7)
    ax.legend(fontsize=7, facecolor="#222", labelcolor="white")
    plt.colorbar(sc, ax=ax, label="time (s)").ax.yaxis.label.set_color("gray")

    # side view with height over time
    ax2 = axes[1]
    ax2.set_facecolor("#111111")
    ax2.plot(t_axis, com[:, 2], color="#3498db", linewidth=1.5)
    ax2.set_xlabel("time (s)", color="gray", fontsize=8)
    ax2.set_ylabel("height z (m)", color="gray", fontsize=8)
    ax2.set_title("CoM height over time", color="white", fontsize=9)
    ax2.tick_params(colors="gray", labelsize=7)
    for spine in ax2.spines.values():
        spine.set_color("gray")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ── Tool 7: visualize_motion ──────────────────────────────────────────────────

def visualize_motion(
    motion_id: str,
    mode: str = "point_cloud",
    num_frames: int = 6,
) -> dict:
    """Generate and save a motion visualisation.

    Args:
        motion_id:  HumanML3D motion identifier.
        mode:       One of "point_cloud", "skeleton", "trajectory".
        num_frames: Number of frames to show (ignored for trajectory mode).

    Returns:
        {"image_path": str}  — absolute path to the saved PNG.
    """
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    out_path = VIZ_DIR / f"viz_{motion_id}_{mode}.png"

    if mode == "point_cloud":
        _plot_point_cloud(motion_id, num_frames, out_path)
    elif mode == "skeleton":
        _plot_skeleton(motion_id, num_frames, out_path)
    elif mode == "trajectory":
        _plot_trajectory(motion_id, out_path)
    else:
        return {"error": f"Unknown mode {mode!r}. Choose from: point_cloud, skeleton, trajectory"}

    return {"image_path": str(out_path)}
