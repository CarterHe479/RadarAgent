# RadarAgent

An LLM agent system for radar-based human motion understanding.  
Built on **Qwen 3 8B** with tool-calling — no radar tokeniser training required.

---

## Overview

Instead of training a bespoke VQ-VAE tokeniser and pre-training a language model (as in the RadarLLM paper), RadarAgent exposes radar data processing as **callable tools**.  The LLM decides which tools to invoke, reasons over their structured text outputs, and generates natural-language motion descriptions.

```
User: "What is the person doing in motion 000021?"
  │
  ▼
Qwen 3 8B (agent)
  ├─ load_radar_sequence("000021")       → duration, bounds, displacement
  ├─ extract_radar_features("000021")    → velocity, periodicity, body regions …
  └─ analyze_joint_motion("000021")      → per-part activity, detected actions
  │
  ▼
"The person squats down and springs up into a jump. The motion lasts ~2 seconds
 with strong vertical dynamics and explosive lower-body activity."
```

---

## Project Structure

```
RadarAgent/
├── config.py                   # All paths and hyperparameters
├── main.py                     # CLI entry point
├── requirements.txt
├── IMPLEMENTATION_PLAN.md      # Full design document
│
├── tools/
│   ├── radar_synthesis.py      # Joint positions → radar point cloud (T,128,4)
│   ├── radar_processing.py     # load_radar_sequence, extract_radar_features
│   ├── joint_analysis.py       # analyze_joint_motion
│   ├── data_retrieval.py       # get_motion_text, search_motions
│   ├── comparison.py           # compare_motions
│   ├── visualization.py        # visualize_motion
│   └── registry.py             # ToolRegistry (wires all tools together)
│
├── agent/
│   ├── agent.py                # RadarAgent (ReAct loop)
│   ├── llm.py                  # Qwen 3 8B loading + generation + tool-call parsing
│   ├── prompts.py              # System prompt + few-shot examples
│   └── tool_schemas.py         # JSON schemas for all 7 tools
│
├── eval/
│   ├── metrics.py              # ROUGE, BLEU, METEOR, CIDEr, BERTScore, SimCSE
│   └── evaluate.py             # Batch evaluation loop
│
├── scripts/
│   ├── run_interactive.sh
│   └── run_eval.sh
│
└── outputs/
    ├── viz/                    # Saved visualisation PNGs
    └── results/                # Evaluation JSON files
```

---

## Data

The agent expects the HumanML3D dataset at `../HumanML3D/HumanML3D/HumanML3D/` relative to this directory.

Required sub-directories:

| Path | Content |
|------|---------|
| `new_joints/` | `(T, 22, 3)` SMPL joint positions (`.npy`) |
| `texts/` | Text annotations (`.txt`, 3–4 descriptions per motion) |
| `train.txt`, `val.txt`, `test.txt` | Split lists (one motion ID per line) |
| `RadarLLM-data/synthetic_points/` | Pre-computed `(T, 128, 4)` point clouds (optional) |

If `synthetic_points/rec_{id}.npy` does not exist for a given motion, the agent synthesises it on-the-fly from joint data using the simplified pipeline from the paper.

---

## Setup

### Python requirement

`transformers >= 4.51` (required for Qwen 3) only supports **Python 3.10+**.  
Use [pyenv](https://github.com/pyenv/pyenv) or any other method to get Python 3.10:

```bash
# If pyenv is installed (no sudo needed)
pyenv install 3.10.16   # skip if already installed
```

### Create venv and install dependencies

```bash
cd RadarAgent

# Create venv with Python 3.10
python3.10 -m venv .venv          # or: ~/.pyenv/versions/3.10.16/bin/python -m venv .venv

# Install all dependencies
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

# NLTK data for METEOR (only needed for evaluation)
.venv/bin/python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

GPU with ≥ 16 GB VRAM is recommended for Qwen 3 8B in bfloat16.  
For CPU-only testing, set `MODEL_NAME = "Qwen/Qwen3-1.7B"` in `config.py`.

---

## Usage

All commands are run from the `RadarAgent/` directory. Use `.venv/bin/python` (or activate the venv first with `source .venv/bin/activate`). The shell scripts handle this automatically.

### Interactive chat

```bash
./scripts/run_interactive.sh
# or manually:
source .venv/bin/activate
PYTHONPATH=. python main.py --interactive
```

Example session:

```
You: What is the person doing in motion 000021?
Agent: The person walks forward at a moderate pace (~0.9 m/s), displaying a
       clear periodic stride cycle of about 1.1 seconds. Lower-body activity is
       high with alternating leg movement, while the upper body shows mild
       compensatory arm swing.

You: Compare motion 000001 and 000002
Agent: Motion 000001 shows a brief squat-and-jump (~1.8 s) with strong vertical
       dynamics, while 000002 is a lateral full-body jump to the left (~4.1 s)
       covering more horizontal ground. Both are explosive but differ in
       direction and duration.
```

### Single query

```bash
PYTHONPATH=. .venv/bin/python main.py --query "Describe what the person is doing in motion 000003"
```

### Visualise a motion

```bash
# Radar point cloud (6 frames)
PYTHONPATH=. .venv/bin/python main.py --visualize 000021 --mode point_cloud

# Skeleton pose
PYTHONPATH=. .venv/bin/python main.py --visualize 000021 --mode skeleton

# Centre-of-mass trajectory
PYTHONPATH=. .venv/bin/python main.py --visualize 000021 --mode trajectory
```

### Evaluation

```bash
# Full test split
./scripts/run_eval.sh test

# Quick sanity check (first 50 samples)
./scripts/run_eval.sh test 50
```

Output table (compared against paper baselines):

```
────────────────────────────────────────────────────────────
  RadarAgent – test split (4385 samples)
────────────────────────────────────────────────────────────
Model              ROUGE-1    ROUGE-L     BLEU-1     BLEU-4    METEOR     CIDEr  BERTScore    SimCSE
──────────────────────────────────────────────────────────────────────────────────────────────────
RadarLLM              38.4       36.0       48.0       11.4      33.7       8.3       83.3      89.6
AvatarGPT             32.2       30.0       36.3        5.0      28.3       6.8       82.4      88.7
MotionGPT             31.2       29.4       37.6        5.0      26.1       6.5       82.6      88.9
──────────────────────────────────────────────────────────────────────────────────────────────────
RadarAgent (ours)      5.2        4.3       12.0       4.0       10.0       5.3       79.6      40.2
────────────────────────────────────────────────────────────
```

---

## Available Tools

| Tool | Purpose |
|------|---------|
| `load_radar_sequence` | Load/synthesise point cloud; return shape, duration, bounds |
| `extract_radar_features` | Velocity, periodicity, body regions, trajectory shape |
| `get_motion_text` | Ground-truth text annotations from HumanML3D |
| `search_motions` | Semantic text search over the dataset |
| `compare_motions` | Side-by-side comparison of two motions |
| `analyze_joint_motion` | Per-body-part activity, action detection, symmetry |
| `visualize_motion` | Save point cloud / skeleton / trajectory PNG |

---

## Radar Synthesis

When pre-computed synthetic point clouds are unavailable, the agent generates them from SMPL joint positions using the simplified pipeline from the paper (Section 3):

1. **Surface sampling** — sample 1024 points from 22 joints with Gaussian jitter (σ = 0.02 m)
2. **Intensity model** — I ∝ 1/r² (simplified radar cross-section)
3. **Top-128 selection** — mimics Doppler-FFT peak picking
4. **Time coordinate** — append t = frame_index / fps

Output: `(T, 128, 4)` array with columns (x, y, z, t).

---

## Reference

Paper: **RadarLLM: Empowering Large Language Models to Understand Human Motion from Millimeter-wave Point Cloud Sequence**  
arXiv:2504.09862v1 (April 2025)

Dataset: **HumanML3D** — Guo et al., CVPR 2022  
14,616 motions · 44,970 descriptions · ~28.6 hours of motion data
