# RadarAgent

An LLM agent system for radar-based human motion understanding.  
Built on **Qwen 3 8B** with tool-calling — no radar tokeniser training required.

---

## Overview

Instead of training a bespoke VQ-VAE tokeniser and pre-training a language model (as in the RadarLLM paper), RadarAgent exposes radar data processing as **callable tools**. The LLM decides which tools to invoke, reasons over their structured text outputs, and generates natural-language motion descriptions.

```
User: "Analyse the radar point cloud for motion 000021 and describe what the person is doing."
  |
  v
Qwen 3 8B (agent)
  |- load_radar_sequence("000021")     -> duration, bounds, displacement
  |- extract_radar_features("000021")  -> velocity, periodicity, body regions ...
  `- analyze_joint_motion("000021")    -> per-part activity, detected actions
  |
  v
"a person squats down and then jumps back up."
```

**Evaluation pipeline (recent improvements):**

- **Thinking mode off** — `enable_thinking=False` in the chat template so Qwen 3 does not emit `<think>…</think>` blocks that would contaminate metric scoring. Any residual blocks are stripped in the agent and again in the eval loop.
- **Valid data only** — motions in a split that have no `new_joints/{id}.npy` and no `RadarLLM-data/synthetic_points/rec_{id}.npy` are skipped during evaluation so the agent is not scored on “data not found” error text.
- **Terse captions** — the system prompt targets HumanML3D-style short sentences (about 5–15 words) to align with reference annotations.
- **Calmer eval decoding** — batch evaluation uses `--temperature 0.3` by default (interactive mode still uses the higher temperature in `config.py` unless you override it).
- **Action detectors** — jump / squat heuristics in `analyze_joint_motion` were tightened to reduce false “jumping” labels from ordinary gait or small root motion.

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
│   ├── agent.py                # RadarAgent (ReAct loop; strips thinking + tool tags)
│   ├── llm.py                  # Qwen 3 8B loading + generation + tool-call parsing
│   ├── prompts.py              # System prompt + few-shot examples
│   └── tool_schemas.py         # JSON schemas for all 7 tools
│
├── eval/
│   ├── metrics.py              # ROUGE, BLEU, METEOR, CIDEr, BERTScore, SimCSE
│   └── evaluate.py             # Batch evaluation (filtered IDs, eval temperature)
│
├── scripts/
│   ├── run_interactive.sh
│   ├── run_eval.sh
│   └── rescore.py              # Optional: re-score a saved eval JSON (debugging)
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

### Hugging Face cache / offline runs

If the Hub is blocked by a proxy, ensure the model is cached once, then run with offline flags:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

---

## Usage

All commands are run from the `RadarAgent/` directory. Use `.venv/bin/python` (or activate the venv first with `source .venv/bin/activate`). The shell scripts handle this automatically.

### Interactive chat

```bash
./scripts/run_interactive.sh
# or manually:
source .venv/bin/activate
PYTHONPATH=. python main.py interactive
```

Example session (style matches terse HumanML3D captions):

```
You: What is the person doing in motion 000021?
Agent: a person squats down and then jumps back up.

You: Compare motion 000001 and 000002
Agent: [uses compare_motions / per-motion tools as needed]
```

### Single query

```bash
PYTHONPATH=. .venv/bin/python main.py query "Describe what the person is doing in motion 000003"
```

### Visualise a motion

```bash
# Radar point cloud (6 frames)
PYTHONPATH=. .venv/bin/python main.py visualize 000021 --mode point_cloud

# Skeleton pose
PYTHONPATH=. .venv/bin/python main.py visualize 000021 --mode skeleton

# Centre-of-mass trajectory
PYTHONPATH=. .venv/bin/python main.py visualize 000021 --mode trajectory
```

### Evaluation

```bash
# Full test split (only motions with joint or synthetic data are evaluated)
./scripts/run_eval.sh test

# First N valid samples after filtering
./scripts/run_eval.sh test 50

# Manual: custom temperature and output path
PYTHONPATH=. .venv/bin/python main.py evaluate --split test --max-samples 50 \
  --temperature 0.3 --output outputs/results/eval_test.json
```

**Interpreting the table:** the printed baselines are partial columns from RadarLLM Table 1 (paper). Your run reports **n_samples** after dropping IDs with no data files.

**Example result (50 samples on the test split: first 50 IDs in `test.txt` that have joint or synthetic data):**

| Model | ROUGE-1 | ROUGE-L | BLEU-1 | BLEU-4 | METEOR | CIDEr | BERTScore | SimCSE |
|-------|---------|---------|--------|--------|--------|-------|-----------|--------|
| RadarLLM (paper) | — | 36.0 | 48.0 | 11.4 | 33.7 | — | 83.3 | — |
| AvatarGPT | — | 30.0 | 36.3 | 5.0 | 28.3 | — | 82.4 | — |
| MotionGPT | — | 29.4 | 37.6 | 5.0 | 26.1 | — | 82.6 | — |
| **RadarAgent (ours)** | **32.9** | **28.9** | **36.8** | **6.5** | **24.8** | **0.4** | **88.3** | **60.4** |

Paper baselines are partial columns from RadarLLM Table 1. Semantic metrics (BERTScore, SimCSE) are strong; n-gram metrics still trail the fully trained RadarLLM baseline — see **Future work** below.

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
2. **Intensity model** — intensity scales like 1/r^2 (simplified radar cross-section)
3. **Top-128 selection** — mimics Doppler-FFT peak picking
4. **Time coordinate** — append t = frame_index / fps

Output: `(T, 128, 4)` array with columns (x, y, z, t).

---

## Future work

- **Supervised fine-tuning (SFT / LoRA)** on train split: map tool outputs (or compact feature summaries) to HumanML3D captions — highest leverage for ROUGE / BLEU / METEOR.
- **RL / DPO / GRPO** after SFT to optimise evaluation metrics or human preference.
- **Richer tool features** (e.g. foot-contact heuristics, clearer walk-vs-kick discrimination) to reduce action confusion before any training.

---

## Reference

Paper: **RadarLLM: Empowering Large Language Models to Understand Human Motion from Millimeter-wave Point Cloud Sequence**  
arXiv:2504.09862v1 (April 2025)

Dataset: **HumanML3D** — Guo et al., CVPR 2022  
14,616 motions · 44,970 descriptions · ~28.6 hours of motion data
