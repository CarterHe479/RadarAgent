"""
JSON schemas for all seven RadarAgent tools, compatible with the
Qwen 3 function-calling format (OpenAI-compatible tool spec).
"""

from typing import List

TOOL_SCHEMAS: List[dict] = [
    {
        "type": "function",
        "function": {
            "name": "load_radar_sequence",
            "description": (
                "Load a radar point cloud sequence for a given motion ID. "
                "Returns shape, duration, spatial bounds, start/end centre-of-mass, "
                "and overall displacement. Use this as the first step when analysing "
                "any motion. If pre-computed synthetic points exist they are loaded "
                "directly; otherwise the sequence is synthesised from joint data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "motion_id": {
                        "type": "string",
                        "description": "HumanML3D motion identifier, e.g. '000021'",
                    }
                },
                "required": ["motion_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_radar_features",
            "description": (
                "Extract detailed spatiotemporal features from a radar point cloud "
                "sequence. Returns velocity profile, periodicity analysis, dominant "
                "motion axis, vertical dynamics, body-region (upper/lower) activity "
                "levels, motion complexity, and trajectory shape. "
                "This is the primary analytical tool."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "motion_id": {
                        "type": "string",
                        "description": "HumanML3D motion identifier",
                    }
                },
                "required": ["motion_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_motion_text",
            "description": (
                "Return all text annotations (3–4 natural-language descriptions) "
                "for a motion from the HumanML3D dataset. Useful for checking ground "
                "truth or providing extra context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "motion_id": {
                        "type": "string",
                        "description": "HumanML3D motion identifier",
                    }
                },
                "required": ["motion_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_motions",
            "description": (
                "Search the HumanML3D dataset for motions whose text descriptions "
                "match a natural-language query using sentence-embedding cosine "
                "similarity. Returns the top-k motion IDs with descriptions and "
                "similarity scores."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query, e.g. 'person walking forward'",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default 5)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_motions",
            "description": (
                "Compare two motion sequences side by side using their radar features. "
                "Returns natural-language comparisons of duration, velocity, spatial "
                "extent, periodicity, complexity, key differences, and overall similarity."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "motion_id_a": {
                        "type": "string",
                        "description": "First motion identifier",
                    },
                    "motion_id_b": {
                        "type": "string",
                        "description": "Second motion identifier",
                    },
                },
                "required": ["motion_id_a", "motion_id_b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_joint_motion",
            "description": (
                "Analyse motion at the skeleton joint level using the 22-joint SMPL "
                "data. Reports per-body-part velocities and activity, detects specific "
                "actions (walking, jumping, arm raise, kick, turn, squat), "
                "left-right symmetry, and root trajectory. Use this when you need "
                "fine-grained body-part detail beyond what radar point clouds reveal."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "motion_id": {
                        "type": "string",
                        "description": "HumanML3D motion identifier",
                    }
                },
                "required": ["motion_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "visualize_motion",
            "description": (
                "Generate and save a PNG visualisation of a motion sequence. "
                "mode='point_cloud' shows 3-D scatter frames; "
                "mode='skeleton' shows bone-connected joint frames; "
                "mode='trajectory' shows top-down and height-over-time CoM plots. "
                "Returns the path to the saved image."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "motion_id": {
                        "type": "string",
                        "description": "HumanML3D motion identifier",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["point_cloud", "skeleton", "trajectory"],
                        "description": "Visualisation type (default: point_cloud)",
                    },
                    "num_frames": {
                        "type": "integer",
                        "description": "Number of frames to display, evenly spaced (default 6)",
                    },
                },
                "required": ["motion_id"],
            },
        },
    },
]
