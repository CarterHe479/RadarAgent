# RadarAgent – Experiment Results

This file records all experiment results.
Scripts and run instructions are in [`experiment_scripts/EXPERIMENTS.md`](experiment_scripts/EXPERIMENTS.md).
Raw JSON outputs live in `outputs/results/`.

**Status legend:** ✅ Complete · ⏳ Pending GPU · 🔲 Not yet started

---

## Table 1 – Main Results & Backbone Comparison

| Experiment | Model | Tools | n | ROUGE-1 | ROUGE-L | BLEU-1 | BLEU-4 | METEOR | CIDEr | BERTScore | SimCSE | Status |
|------------|-------|-------|---|---------|---------|--------|--------|--------|-------|-----------|--------|--------|
| **E1** RadarAgent | Qwen3-8B | All | 50 | **32.94** | **28.94** | **36.82** | **6.46** | **24.75** | **0.36** | 88.31 | **60.43** | ✅ |
| **E3** Qwen3-1.7B | Qwen3-1.7B | All | 50 | **37.50** | **37.47** | 26.96 | **10.15** | 23.41 | 0.24 | **90.29** | 55.40 | ✅ |
| **E5** Llama-3.1-8B | Llama-3.1-8B-Instruct | Plain-text features | 500 | — | — | — | — | — | — | — | — | ⏳ |
| **E6** Gemini 2.0 Flash | Gemini 2.0 Flash (API) | Plain-text features | 500 | 29.76 | 26.31 | 33.10 | 4.81 | 18.46 | 0.16 | **88.45** | 54.44 | ✅ |

> **E1 notes:** Evaluated on 50 samples from the test split (CPU run, proxy-blocked environment).  
> **E3 notes:** Qwen3-1.7B with full agent + all tools; 50 samples, CPU-only (~23 sec/sample). Qwen3-1.7B surprisingly outperforms 8B on ROUGE-L and BERTScore, likely due to more concise outputs that better match HumanML3D annotation style.  
> **E6 notes:** Gemini 2.0 Flash via Google GenAI API; same feature context as E5. Runtime ~6 min / 500 samples.

---

## Table 2 – Tool Ablation (500 samples, Qwen3-8B)

| Experiment | Tools Available | ROUGE-1 | ROUGE-L | BLEU-1 | BLEU-4 | METEOR | CIDEr | BERTScore | SimCSE | Status |
|------------|----------------|---------|---------|--------|--------|--------|-------|-----------|--------|--------|
| **A1** No tools (raw LLM) | None | — | — | — | — | — | — | — | — | ⏳ |
| **A2** Radar load only | `load_radar_sequence` | — | — | — | — | — | — | — | — | ⏳ |
| **A3** Radar features | + `extract_radar_features` | — | — | — | — | — | — | — | — | ⏳ |
| **A4** Radar + joints | + `analyze_joint_motion` | — | — | — | — | — | — | — | — | ⏳ |
| **A5** All tools | All 7 tools | — | — | — | — | — | — | — | — | ⏳ |
| **A6** Joints only | `analyze_joint_motion` | — | — | — | — | — | — | — | — | ⏳ |

> The gap **A5 − A6** isolates how much the radar signal adds over pure skeleton data.  
> The gap **A5 − A1** shows the total contribution of the tool-calling pipeline.

---

## Table 3 – Prompt & Inference Ablation (200 samples, Qwen3-8B)

| Experiment | Change | ROUGE-1 | ROUGE-L | BLEU-1 | BLEU-4 | METEOR | CIDEr | BERTScore | SimCSE | Status |
|------------|--------|---------|---------|--------|--------|--------|-------|-----------|--------|--------|
| **P1** No few-shot | Removes in-context examples | — | — | — | — | — | — | — | — | ⏳ |
| **P2** Thinking on | Re-enables `<think>` blocks | — | — | — | — | — | — | — | — | ⏳ |
| **P3** Temp = 0.0 | Greedy decoding | — | — | — | — | — | — | — | — | ⏳ |
| **P3** Temp = 0.3 | Low temperature (default) | — | — | — | — | — | — | — | — | ⏳ |
| **P3** Temp = 0.7 | Higher temperature | — | — | — | — | — | — | — | — | ⏳ |
| **P4** Verbose prompt | Reverts to long system prompt | — | — | — | — | — | — | — | — | ⏳ |

---

## Table 4 – Training Extension: LoRA SFT (500 test samples)

| Experiment | Description | ROUGE-1 | ROUGE-L | BLEU-1 | BLEU-4 | METEOR | CIDEr | BERTScore | SimCSE | Status |
|------------|-------------|---------|---------|--------|--------|--------|-------|-----------|--------|--------|
| **T2** LoRA SFT | Qwen3-8B fine-tuned on 1000 train samples | — | — | — | — | — | — | — | — | 🔲 |

> Fine-tuning pipeline: `T2a_collect_sft_data.py` → `T2b_lora_sft.py` → `T2c_eval_sft.py`

---

## Table 5 – Robustness Study

| Experiment | Condition | ROUGE-1 | ROUGE-L | BERTScore | SimCSE | Status |
|------------|-----------|---------|---------|-----------|--------|--------|
| **R1** No pre-computed | On-the-fly radar synthesis | — | — | — | — | ⏳ |
| **R2** Noise σ=0.02 | Gaussian noise on point cloud | — | — | — | — | ⏳ |
| **R2** Noise σ=0.05 | Gaussian noise on point cloud | — | — | — | — | ⏳ |
| **R2** Noise σ=0.10 | Gaussian noise on point cloud | — | — | — | — | ⏳ |
| **R3** 128 pts/frame | Baseline point density | — | — | — | — | ⏳ |
| **R3** 64 pts/frame | 50% sparser | — | — | — | — | ⏳ |
| **R3** 32 pts/frame | 75% sparser | — | — | — | — | ⏳ |
| **R3** 16 pts/frame | 87.5% sparser | — | — | — | — | ⏳ |

---

## Completed Results – Detail

### E1 – RadarAgent (Qwen3-8B, 50 samples)

- **Output file:** `outputs/results/eval_test.json`
- **Model:** Qwen/Qwen3-8B, thinking mode **off**, temperature 0.3
- **Tools:** All 7 tools (load_radar_sequence, extract_radar_features, analyze_joint_motion, get_motion_text, search_motions, compare_motions, visualize_motion)
- **Prompt:** Terse system prompt enforcing 5–15 word output, 2 few-shot examples
- **Environment:** CPU-only (proxy-blocked, offline HF cache)

| Metric | Score |
|--------|-------|
| ROUGE-1 | 32.94 |
| ROUGE-L | 28.94 |
| BLEU-1 | 36.82 |
| BLEU-4 | 6.46 |
| METEOR | 24.75 |
| CIDEr | 0.36 |
| BERTScore | 88.31 |
| SimCSE | 60.43 |

**Key pipeline improvements that led to these scores** (vs. earlier 50-sample baseline):
- Disabled Qwen3 thinking mode (`enable_thinking=False`) — removes `<think>` tokens from output
- Filtered samples with missing data files — eliminates silent empty-output failures
- Tightened action-detection thresholds (jump: 0.25 m + velocity check; squat: 0.20 m)
- Switched system prompt to enforce HumanML3D-style terse single-sentence output
- Reduced `GENERATION_MAX_TOKENS` from 2048 → 512 and eval temperature to 0.3
- Fixed BERTScore overflow: `model_type="roberta-large"`, `use_fast_tokenizer=False`

---

### E3 – Qwen3-1.7B (50 samples)

- **Output file:** `outputs/results/experiments/E3_backbone_qwen3_1b_n50.json`
- **Model:** Qwen/Qwen3-1.7B, thinking mode **off**, temperature 0.3
- **Tools:** All 7 tools (same setup as E1)
- **Environment:** CPU-only, ~23 sec/sample

| Metric | Score |
|--------|-------|
| ROUGE-1 | 37.50 |
| ROUGE-L | **37.47** |
| BLEU-1 | 26.96 |
| BLEU-4 | **10.15** |
| METEOR | 23.41 |
| CIDEr | 0.24 |
| BERTScore | **90.29** |
| SimCSE | 55.40 |

**Observations:**
- ROUGE-L (37.47) and BERTScore (90.29) both exceed E1 (28.94 / 88.31), suggesting the 1.7B model produces tighter, more annotation-aligned outputs.
- BLEU-4 (10.15) is notably higher than E1 (6.46), consistent with shorter, more precise generations.
- BLEU-1 (26.96) is lower than E1 (36.82), which may indicate less vocabulary overlap despite strong sequence-level alignment.
- The smaller model appears to adhere more strictly to the terse output style, which benefits metrics calibrated to short reference sentences.

---

### E6 – Gemini 2.0 Flash (500 samples)

- **Output file:** `outputs/results/experiments/E6_backbone_gemini_n500.json`
- **Model:** `gemini-2.0-flash` via Google GenAI API
- **Approach:** Plain-text feature context (identical to E5 for Llama); no native tool schema
- **Temperature:** 0.3, max output tokens: 64
- **Runtime:** ~6 minutes for 500 samples (~1.4 samples/sec, API-latency bound)

| Metric | Score |
|--------|-------|
| ROUGE-1 | 29.76 |
| ROUGE-L | 26.31 |
| BLEU-1 | 33.10 |
| BLEU-4 | 4.81 |
| METEOR | 18.46 |
| CIDEr | 0.16 |
| BERTScore | **88.45** |
| SimCSE | 54.44 |

**Observations:**
- BERTScore (88.45) matches and slightly edges Qwen3-8B RadarAgent (88.31), confirming the feature extraction pipeline quality rather than model-specific advantage.
- n-gram metrics (ROUGE/BLEU/METEOR) are lower than E1, consistent with Gemini using richer but less HumanML3D-literal phrasing.
- SimCSE (54.44) is lower than E1 (60.43), possibly because E1 has access to native tool calls and iterative reasoning while E6 receives only a flattened feature string.
- No GPU required — fully API-based.

---

## Metric Definitions

| Metric | Description |
|--------|-------------|
| **ROUGE-1** | Unigram recall overlap with references |
| **ROUGE-L** | Longest common subsequence recall |
| **BLEU-1/4** | Precision-based n-gram matching (1-gram / 4-gram) |
| **METEOR** | F-score with stemming and synonym matching |
| **CIDEr** | Consensus-based n-gram weighting (common in captioning) |
| **BERTScore** | Contextual embedding similarity (RoBERTa-large F1) |
| **SimCSE** | Sentence-level cosine similarity (all-MiniLM-L6-v2) |

All metrics are computed against the full set of HumanML3D text annotations for each motion sequence (typically 3–5 references).
