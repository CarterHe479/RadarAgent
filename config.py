from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR.parent / "HumanML3D" / "HumanML3D" / "HumanML3D"

JOINTS_DIR         = DATA_DIR / "new_joints"
TEXTS_DIR          = DATA_DIR / "texts"
SPLITS_DIR         = DATA_DIR          # train.txt, val.txt, test.txt live here
RADAR_DATA_DIR     = DATA_DIR / "RadarLLM-data"
SYNTHETIC_POINTS_DIR = RADAR_DATA_DIR / "synthetic_points"

OUTPUT_DIR  = PROJECT_DIR / "outputs"
VIZ_DIR     = OUTPUT_DIR / "viz"
RESULTS_DIR = OUTPUT_DIR / "results"

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen3-8B"

# ── Radar synthesis (from the paper) ─────────────────────────────────────────
RADAR_FPS            = 20
POINTS_PER_FRAME     = 128
SURFACE_SAMPLE_COUNT = 1024
SURFACE_JITTER_STD   = 0.02    # metres – approximates body-surface spread
RADAR_POSITION       = [0.0, 0.0, 0.0]

# ── SMPL 22-joint skeleton ────────────────────────────────────────────────────
NUM_JOINTS = 22

# Kinematic chains (from paramUtil.py)
KINEMATIC_CHAINS = [
    [0, 2, 5, 8, 11],        # right leg
    [0, 1, 4, 7, 10],        # left leg
    [0, 3, 6, 9, 12, 15],    # spine → head
    [9, 14, 17, 19, 21],     # right arm
    [9, 13, 16, 18, 20],     # left arm
]

# Joint index → body-part groups
BODY_PART_JOINTS = {
    "head":      [12, 15],
    "torso":     [0, 3, 6, 9],
    "left_arm":  [13, 16, 18, 20],
    "right_arm": [14, 17, 19, 21],
    "left_leg":  [1, 4, 7, 10],
    "right_leg": [2, 5, 8, 11],
}

# ── Agent ─────────────────────────────────────────────────────────────────────
MAX_AGENT_ITERATIONS    = 8
GENERATION_MAX_TOKENS   = 2048
GENERATION_TEMPERATURE  = 0.7

# ── Evaluation ────────────────────────────────────────────────────────────────
EVAL_SPLIT = "test"
