"""
compare_motions – side-by-side comparison of two radar point cloud sequences.

Internally calls extract_radar_features for both motions and produces a
structured natural-language diff the LLM can reason over.
"""

from __future__ import annotations

from tools.radar_processing import extract_radar_features


def _vel_comparison(fa: dict, fb: dict) -> str:
    va = fa["velocity"]["mean_m_per_s"]
    vb = fb["velocity"]["mean_m_per_s"]
    diff = abs(va - vb)
    a_id, b_id = fa["motion_id"], fb["motion_id"]
    if diff < 0.05:
        return f"Both motions have similar average speed (~{va:.2f} m/s)."
    faster, slower = (a_id, va, vb) if va > vb else (b_id, vb, va)
    return (
        f"Motion {faster} is faster ({faster.split('_')[-1] if '_' in faster else faster}: "
        f"{max(va,vb):.2f} m/s vs {min(va,vb):.2f} m/s)."
    )


def _duration_comparison(fa: dict, fb: dict) -> str:
    da, db = fa["duration_sec"], fb["duration_sec"]
    diff = abs(da - db)
    if diff < 0.2:
        return f"Both motions are approximately the same length (~{da:.1f}s)."
    longer = fa["motion_id"] if da > db else fb["motion_id"]
    return f"Motion {longer} is {diff:.1f}s longer ({max(da,db):.1f}s vs {min(da,db):.1f}s)."


def _spatial_comparison(fa: dict, fb: dict) -> str:
    ra = fa["dominant_axis_range_m"]
    rb = fb["dominant_axis_range_m"]
    if abs(ra - rb) < 0.1:
        return "Both motions cover similar ground extent."
    more = fa["motion_id"] if ra > rb else fb["motion_id"]
    return f"Motion {more} covers more ground ({max(ra,rb):.2f} m vs {min(ra,rb):.2f} m on dominant axis)."


def _periodicity_comparison(fa: dict, fb: dict) -> str:
    pa, pb = fa["periodicity"], fb["periodicity"]
    if pa["is_periodic"] and pb["is_periodic"]:
        return (
            f"Both motions are periodic. "
            f"A: ~{pa['estimated_period_sec']}s cycle; "
            f"B: ~{pb['estimated_period_sec']}s cycle."
        )
    if pa["is_periodic"]:
        return f"Motion {fa['motion_id']} is periodic (~{pa['estimated_period_sec']}s); {fb['motion_id']} is not."
    if pb["is_periodic"]:
        return f"Motion {fb['motion_id']} is periodic (~{pb['estimated_period_sec']}s); {fa['motion_id']} is not."
    return "Neither motion shows clear periodicity."


def _complexity_comparison(fa: dict, fb: dict) -> str:
    ca, cb = fa["motion_complexity"], fb["motion_complexity"]
    if ca == cb:
        return f"Both motions have {ca} complexity."
    return f"Motion {fa['motion_id']} is {ca}; motion {fb['motion_id']} is {cb}."


def _overall_similarity(fa: dict, fb: dict) -> str:
    """Rough composite similarity label."""
    score = 0
    # same trajectory shape
    if fa["trajectory_shape"] == fb["trajectory_shape"]:
        score += 1
    # similar velocity (within 30%)
    va, vb = fa["velocity"]["mean_m_per_s"], fb["velocity"]["mean_m_per_s"]
    if max(va, vb) > 0 and min(va, vb) / max(va, vb) > 0.70:
        score += 1
    # both periodic or both not
    if fa["periodicity"]["is_periodic"] == fb["periodicity"]["is_periodic"]:
        score += 1
    # same complexity
    if fa["motion_complexity"] == fb["motion_complexity"]:
        score += 1

    if score >= 3:
        return "very similar"
    if score == 2:
        return "somewhat similar"
    return "different"


# ── Tool 5: compare_motions ───────────────────────────────────────────────────

def compare_motions(motion_id_a: str, motion_id_b: str) -> dict:
    """Compare two motions using extracted radar features."""
    fa = extract_radar_features(motion_id_a)
    fb = extract_radar_features(motion_id_b)

    dur_cmp        = _duration_comparison(fa, fb)
    vel_cmp        = _vel_comparison(fa, fb)
    spatial_cmp    = _spatial_comparison(fa, fb)
    period_cmp     = _periodicity_comparison(fa, fb)
    complexity_cmp = _complexity_comparison(fa, fb)

    # top distinguishing factors
    differences: list[str] = []
    if fa["trajectory_shape"] != fb["trajectory_shape"]:
        differences.append(
            f"Trajectory shape differs: {motion_id_a} is {fa['trajectory_shape']}, "
            f"{motion_id_b} is {fb['trajectory_shape']}."
        )
    if fa["upper_body"]["activity_level"] != fb["upper_body"]["activity_level"]:
        differences.append(
            f"Upper-body activity differs: {motion_id_a}={fa['upper_body']['activity_level']}, "
            f"{motion_id_b}={fb['upper_body']['activity_level']}."
        )
    if fa["lower_body"]["activity_level"] != fb["lower_body"]["activity_level"]:
        differences.append(
            f"Lower-body activity differs: {motion_id_a}={fa['lower_body']['activity_level']}, "
            f"{motion_id_b}={fb['lower_body']['activity_level']}."
        )
    vd = abs(fa["vertical_dynamics"]["vertical_range_m"] - fb["vertical_dynamics"]["vertical_range_m"])
    if vd > 0.1:
        differences.append(
            f"Vertical range differs by {vd:.2f} m "
            f"({motion_id_a}: {fa['vertical_dynamics']['vertical_range_m']:.2f} m, "
            f"{motion_id_b}: {fb['vertical_dynamics']['vertical_range_m']:.2f} m)."
        )

    if not differences:
        differences.append("Motions appear very similar across all measured dimensions.")

    return {
        "motion_a": motion_id_a,
        "motion_b": motion_id_b,
        "duration_comparison":     dur_cmp,
        "velocity_comparison":     vel_cmp,
        "spatial_comparison":      spatial_cmp,
        "periodicity_comparison":  period_cmp,
        "complexity_comparison":   complexity_cmp,
        "key_differences":         differences[:3],
        "overall_similarity":      _overall_similarity(fa, fb),
    }
