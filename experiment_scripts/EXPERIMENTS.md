# RadarAgent – Experiment Guide

This directory (`experiment_scripts/`) lives inside `RadarAgent/` and contains
every experiment needed for the workshop paper.
All scripts are self-contained and share common utilities in `shared/`.

**All commands below are run from the `RadarAgent/` directory unless stated otherwise.**

```bash
cd /home/carter/radar_llm/RadarAgent
```

---

## Prerequisites

Set up the virtual environment once:

```bash
cd /home/carter/radar_llm/RadarAgent
python3.10 -m venv .venv
.venv/bin/pip install -r requirements.txt

# SFT scripts additionally need:
.venv/bin/pip install peft bitsandbytes trl datasets
```

The model must be cached locally (proxy-blocked environments):

```bash
# First time only (needs HF access):
HF_TOKEN=<your_token> huggingface-cli download Qwen/Qwen3-8B

# Then always run with:
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

Results land in `outputs/results/experiments/`.

---

## Quick-start: minimum viable experiment set

```bash
cd /home/carter/radar_llm/RadarAgent
VENV=".venv/bin/python"

# 1. Full test split headline number  (needs GPU, ~40-80 GPU-hrs)
bash experiment_scripts/E1_main_results.sh

# 2. Tool ablation on 500 samples  (~3-4 GPU-hrs each, run A1/A3/A5 at minimum)
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. $VENV experiment_scripts/A1_no_tools.py       --max-samples 500
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. $VENV experiment_scripts/A3_radar_features.py --max-samples 500
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. $VENV experiment_scripts/A5_all_tools_subset.py --max-samples 500

# 3. Qualitative analysis (no model – runs in seconds)
$VENV experiment_scripts/Q1_category_accuracy.py \
  --results outputs/results/experiments/E1_main_results.json
$VENV experiment_scripts/Q4_error_taxonomy.py \
  --results outputs/results/experiments/E1_main_results.json
```

---

## Experiment inventory

### E – Main Results & Backbone Comparison

| Script | Purpose | Samples | GPU-hrs (A100) |
|--------|---------|---------|---------------|
| `E1_main_results.sh` | Headline: full test split, all tools, Qwen3-8B | ~2584 | 40–80 |
| `E3_backbone_qwen3_1b.sh` | Smaller Qwen3-1.7B backbone | 500 | 3–5 |
| `E5_backbone_llama.sh` | Cross-family Llama-3.1-8B-Instruct | 500 | 4–6 |

```bash
# E1 – full test split (or pass N for a quick run)
bash experiment_scripts/E1_main_results.sh
bash experiment_scripts/E1_main_results.sh 50   # sanity check

# E3 and E5 – backbone comparison
bash experiment_scripts/E3_backbone_qwen3_1b.sh 500
bash experiment_scripts/E5_backbone_llama.sh 500
```

For E5 you must first download Llama-3.1-8B-Instruct, or point to a local copy:
```bash
export LLAMA_MODEL_PATH=/path/to/local/llama
bash experiment_scripts/E5_backbone_llama.sh 500
```

---

### A – Tool Ablation (all on 500 samples)

Run these in order; they form the ablation table (Table 2 in the paper).

```bash
cd /home/carter/radar_llm/RadarAgent
VENV=".venv/bin/python"
OPT="HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=."

eval "$OPT $VENV experiment_scripts/A1_no_tools.py         --max-samples 500"
eval "$OPT $VENV experiment_scripts/A2_radar_load_only.py  --max-samples 500"
eval "$OPT $VENV experiment_scripts/A3_radar_features.py   --max-samples 500"
eval "$OPT $VENV experiment_scripts/A4_radar_plus_joints.py --max-samples 500"
eval "$OPT $VENV experiment_scripts/A5_all_tools_subset.py --max-samples 500"
eval "$OPT $VENV experiment_scripts/A6_joints_only.py      --max-samples 500"
```

| Script | Tools available | Paper narrative |
|--------|----------------|-----------------|
| A1 | None (raw LLM) | Lower bound |
| A2 | `load_radar_sequence` | Coarse spatial info only |
| A3 | A2 + `extract_radar_features` | +Radar feature extractor |
| A4 | A3 + `analyze_joint_motion` | +Skeleton analysis |
| A5 | All tools | Full system on ablation subset |
| A6 | `analyze_joint_motion` only | Skeleton only — no radar |

The gap **A5 − A6** quantifies how much the radar signal adds over pure skeleton data.

---

### P – Prompt / Inference Ablation (200 samples)

```bash
cd /home/carter/radar_llm/RadarAgent
VENV=".venv/bin/python"
OPT="HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=."

eval "$OPT $VENV experiment_scripts/P1_no_fewshot.py    --max-samples 200"
eval "$OPT $VENV experiment_scripts/P2_thinking_on.py   --max-samples 200"
eval "$OPT $VENV experiment_scripts/P3_temperature_sweep.py --max-samples 200 --temps 0.0 0.3 0.7"
eval "$OPT $VENV experiment_scripts/P4_verbose_prompt.py --max-samples 200"
```

| Script | What changes | Paper narrative |
|--------|-------------|-----------------|
| P1 | Removes few-shot examples | Are in-context examples necessary? |
| P2 | Re-enables `<think>` blocks | Thinking mode penalty on metrics |
| P3 | Sweeps temperature {0.0, 0.3, 0.7} | Best decoding temperature |
| P4 | Original verbose prompt | Justifies the terse prompt change |

---

### T – Training Extension (LoRA SFT)

Run in three sequential steps.

**Step 1 – Collect training data from tool pipeline (CPU-only, ~20 min for 1000 samples):**
```bash
cd /home/carter/radar_llm/RadarAgent
PYTHONPATH=. .venv/bin/python experiment_scripts/T2a_collect_sft_data.py \
  --split train --max-samples 1000 \
  --output outputs/sft/sft_data_train_n1000.jsonl
```

**Step 2 – Fine-tune with LoRA (~2-3 GPU-hrs on A100):**
```bash
PYTHONPATH=. .venv/bin/python experiment_scripts/T2b_lora_sft.py \
  --data outputs/sft/sft_data_train_n1000.jsonl \
  --output-dir outputs/sft/lora \
  --epochs 3 --batch-size 4 --lora-rank 16

# On 24 GB GPU (tight) or 12 GB GPU, add: --load-4bit
```

**Step 3 – Evaluate the SFT checkpoint:**
```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. .venv/bin/python \
  experiment_scripts/T2c_eval_sft.py \
  --checkpoint outputs/sft/lora/final \
  --max-samples 500
```

Results → `outputs/results/experiments/T2_lora_sft_n500.json`.

---

### Q – Qualitative Analysis (no inference required)

These scripts run in seconds on any existing eval JSON. No GPU needed.

```bash
cd /home/carter/radar_llm/RadarAgent
VENV=".venv/bin/python"
E1="outputs/results/experiments/E1_main_results.json"

# Q1: Per-action-category precision / recall / F1
$VENV experiment_scripts/Q1_category_accuracy.py --results $E1

# Q2: Tool call frequency and per-category usage
$VENV experiment_scripts/Q2_tool_call_freq.py --results $E1

# Q4: Error taxonomy (wrong action / hallucination / multi-action-miss)
$VENV experiment_scripts/Q4_error_taxonomy.py --results $E1

# Q5: Efficiency comparison across multiple experiments
$VENV experiment_scripts/Q5_efficiency.py \
  --results $E1 outputs/results/experiments/T2_lora_sft_n500.json \
  --labels "Zero-shot" "SFT"
```

---

### R – Robustness (200 samples each)

```bash
cd /home/carter/radar_llm/RadarAgent
VENV=".venv/bin/python"
OPT="HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=."

# R1: Force on-the-fly synthesis (no pre-computed point clouds)
eval "$OPT $VENV experiment_scripts/R1_no_precomputed.py --max-samples 200"

# R2: Gaussian noise injection at three levels
eval "$OPT $VENV experiment_scripts/R2_noise_injection.py \
  --max-samples 200 --sigmas 0.02 0.05 0.10"

# R3: Sparse point clouds  (128 → 64 → 32 → 16 points/frame)
eval "$OPT $VENV experiment_scripts/R3_sparse_points.py \
  --max-samples 200 --n-pts 128 64 32 16"
```

---

## Recommended run order (1-week schedule)

| Day | Experiments | Priority |
|-----|------------|---------|
| 1 | E1 (launch overnight, full test split) | Must-have |
| 2 | A1, A3, A5 (ablation core) | Must-have |
| 3 | A2, A4, A6 (complete ablation) + P2 | Must-have |
| 4 | E3, E5 (backbone) + P1, P4 | Should-have |
| 5 | T2a + T2b (SFT training, launch overnight) | Strongly recommended |
| 6 | T2c (eval SFT) + P3 (temp sweep) | Should-have |
| 7 | Q1, Q2, Q4, Q5 (analysis, no GPU) + R1 | Nice-to-have |
| 8 | R2, R3 (robustness sweeps) | Appendix |

---

## Output file schema

All results land in `outputs/results/experiments/`.
Each experiment writes one JSON with this structure:

```json
{
  "experiment":  "A3_radar_features_n500",
  "n_samples":   500,
  "metrics": {
    "ROUGE-1": 30.1, "ROUGE-L": 26.2,
    "BLEU-1": 34.5,  "BLEU-4": 5.8,
    "METEOR": 22.3,  "CIDEr": 0.4,
    "BERTScore": 87.1, "SimCSE": 57.3
  },
  "meta": { "tools": ["load_radar_sequence", "extract_radar_features"] },
  "samples": [
    {
      "motion_id": "009613",
      "generated": "a person walks backwards quickly.",
      "references": ["the man runs back wards", "..."],
      "elapsed_sec": 12.4,
      "n_tokens_approx": 6
    }
  ]
}
```

Analysis scripts (Q1-Q5) produce their own JSONs in the same directory.

---

## Mapping experiments → paper sections

| Paper section | Table/Figure | Experiments |
|---------------|-------------|-------------|
| Sec. 4.1 Main results | Table 1 | E1, E3, E5 |
| Sec. 4.2 Tool ablation | Table 2 | A1–A6 |
| Sec. 4.3 Prompt ablation | Table 3 | P1–P4 |
| Sec. 4.4 Training extension | Table 4 | T2a+T2b+T2c |
| Sec. 4.5 Analysis | Fig. 2, Table 5 | Q1, Q2, Q4, Q5 |
| Appendix: Robustness | Table 6 | R1–R3 |
