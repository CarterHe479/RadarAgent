# RadarAgent: Implementation Plan

An LLM agent system for radar-based human motion understanding, built on Qwen 3 8B with tool-calling. Instead of the pre-training approach from the RadarLLM paper (arXiv:2504.09862v1), we expose radar data processing as callable tools and let the LLM reason over structured text outputs.

---

## Table of Contents

1. [Background & Motivation](#1-background--motivation)
2. [Data Sources & Formats](#2-data-sources--formats)
3. [Radar Point Cloud Synthesis (from the Paper)](#3-radar-point-cloud-synthesis-from-the-paper)
4. [Architecture Overview](#4-architecture-overview)
5. [Tool Specifications](#5-tool-specifications)
6. [Agent Core](#6-agent-core)
7. [Project Structure](#7-project-structure)
8. [Step-by-Step Build Instructions](#8-step-by-step-build-instructions)
9. [Evaluation](#9-evaluation)
10. [Dependencies](#10-dependencies)

---

## 1. Background & Motivation

### What the Paper Does (RadarLLM)

The paper "RadarLLM: Empowering Large Language Models to Understand Human Motion from Millimeter-wave Point Cloud Sequence" proposes:

1. **Motion-guided Radar Tokenizer**: A VQ-VAE that compresses radar point cloud sequences `(T, 128, 4)` into discrete token indices via template-prior grouping, masked context aggregation, and vector quantization with a codebook of size 512.
2. **Radar-aware Language Model**: A modified T5-Small (60M params) that unifies radar tokens and text tokens in a shared vocabulary, trained via:
   - Unsupervised mask reconstruction (15% span corruption on radar tokens)
   - Supervised cross-modal translation (radar→text and text→radar)
   - Instruction tuning with diversified prompts
3. **Physics-aware Synthesis**: Generates synthetic radar point clouds from the HumanML3D motion-text dataset using IF signal simulation, Range-FFT, Doppler-FFT, and top-128 point selection per frame.

### What We Do Instead (RadarAgent)

We skip the tokenizer training and LLM pre-training entirely. Instead:

- Use **Qwen 3 8B** as the LLM backbone — it already has strong reasoning and native function-calling support.
- Expose radar data loading, feature extraction, and analysis as **tool calls** that return structured text.
- The LLM decides which tools to invoke, reasons over the text-formatted results, and generates motion descriptions.

### Why This Works

| Aspect | RadarLLM | RadarAgent |
|--------|----------|------------|
| Modality bridge | Learned VQ-VAE tokens + pre-training | Feature extraction tools → structured text |
| Training cost | VQ-VAE + 3-stage LLM training | None (zero-shot) or lightweight LoRA |
| LLM backbone | T5-Small (60M) | Qwen 3 8B (8B, native tool-use) |
| Extensibility | Retrain for new capabilities | Add a new tool function |
| Data requirement | Needs radar-text pairs for training | Uses same data, but at inference time via tools |

---

## 2. Data Sources & Formats

All data comes from HumanML3D, located at: `../HumanML3D/HumanML3D/HumanML3D/`

### 2.1 Joint Position Data (`new_joints/`)

Each `.npy` file contains 3D joint positions for one motion sequence.

- **Shape**: `(T, 22, 3)` — T frames, 22 SMPL joints, (x, y, z) in meters
- **FPS**: 20
- **Duration**: 2–10 seconds (T ranges from ~40 to ~200)
- **Naming**: `XXXXXX.npy` (e.g., `000021.npy`); `MXXXXXX.npy` for mirrored versions
- **Path pattern**: `../HumanML3D/HumanML3D/HumanML3D/new_joints/{motion_id}.npy`

The 22 SMPL joints follow this kinematic chain (from `paramUtil.py`):

```
Joint indices and kinematic chains:
  Chain 0 (right leg):  0 → 2 → 5 → 8 → 11
  Chain 1 (left leg):   0 → 1 → 4 → 7 → 10
  Chain 2 (spine/head): 0 → 3 → 6 → 9 → 12 → 15
  Chain 3 (right arm):  9 → 14 → 17 → 19 → 21
  Chain 4 (left arm):   9 → 13 → 16 → 18 → 20

Joint mapping (approximate):
  0: pelvis (root)
  1: left hip, 2: right hip
  3: spine1, 4: left knee, 5: right knee
  6: spine2, 7: left ankle, 8: right ankle
  9: spine3 (chest), 10: left foot, 11: right foot
  12: neck, 13: left collar, 14: right collar
  15: head, 16: left shoulder, 17: right shoulder
  18: left elbow, 19: right elbow
  20: left wrist, 21: right wrist
```

### 2.2 Text Annotations (`texts/`)

Each `.txt` file contains 3–4 natural language descriptions.

- **Path pattern**: `../HumanML3D/HumanML3D/HumanML3D/texts/{motion_id}.txt`
- **Line format**: `original_description#POS_tagged_sentence#start_time#end_time`
  - `start_time=0.0, end_time=0.0` means the description covers the full sequence
- **Example** (file `000001.txt`):
  ```
  a man squats extraordinarily low then bolts up in an unsatisfactory jump.#a/DET man/NOUN squat/VERB ...#0.0#0.0
  a person falls to the ground in a sitting motion and then pops back up in a standing position.#a/DET person/NOUN ...#0.0#0.0
  a person squats down then jumps#a/DET person/NOUN ...#0.0#0.0
  ```

To extract just the natural language description, split each line on `#` and take index 0.

### 2.3 Dataset Splits

Plain text files listing motion IDs (one per line, no extension):

- `train.txt`: ~23,385 IDs
- `val.txt`: ~1,451 IDs
- `test.txt`: ~4,385 IDs
- `all.txt`: all IDs
- `train_val.txt`: train + val combined

Path: `../HumanML3D/HumanML3D/HumanML3D/{split}.txt`

### 2.4 Pre-computed Synthetic Radar Point Clouds (optional)

Already generated and stored in `../HumanML3D/HumanML3D/HumanML3D/RadarLLM-data/`:

- **Point clouds**: `synthetic_points/rec_{motion_id}.npy` — shape `(T, 128, 4)` where 4 = (x, y, z, t_normalized)
- **Manifests**: `train.jsonl`, `val.jsonl`, `test.jsonl` — one JSON object per line:
  ```json
  {
    "id": "rec_000001",
    "text": "full text annotations joined by \\n",
    "points_path": "/absolute/path/to/synthetic_points/rec_000001.npy",
    "split": "train",
    "fps": 20,
    "num_points": 128,
    "source": "synthetic",
    "duration": 1.75
  }
  ```

**Note**: The `points_path` in manifests uses absolute paths from the original author's machine. At runtime, resolve paths relative to the data directory.

### 2.5 Normalization Statistics

- `Mean.npy`: shape `(263,)` — mean of motion feature vectors
- `Std.npy`: shape `(263,)` — std of motion feature vectors

These are for the 263-dim motion representation in `new_joint_vecs/`, not needed for our point-cloud-based approach.

---

## 3. Radar Point Cloud Synthesis (from the Paper)

The paper describes a physics-aware pipeline for generating radar point clouds from motion data. Our project needs this to convert joint data into radar-like point clouds.

### 3.1 Overview of the Paper's Method

From Section 3 of the paper:

1. **Input**: SMPL-X motion sequences from AMASS/HumanML3D — 3D mesh vertices over time.
2. **IF Signal Simulation**: Ray tracing from virtual radar antennas to rendered human meshes, using RF adaptive sampling (edge detection to focus rays on the body), then Physical Optics Integral (POI) to compute intermediate frequency (IF) signals.
3. **Point Cloud Generation**: Range-FFT → Doppler-FFT → static clutter removal (subtract mean Doppler-FFT heatmap) → select top-128 intensity points per frame from the Doppler-FFT heatmap.
4. **Output**: `(T, 128, 4)` point cloud sequences with (x, y, z, t) per point.

### 3.2 Simplified Implementation

Since we have joint positions (not full meshes), and full ray-tracing + FFT simulation is expensive, we implement a simplified but physically-motivated synthesis:

```
Algorithm: motion_to_radar_pointcloud(joints, fps=20, N=128)

Input:  joints — shape (T, 22, 3), 22 SMPL joint positions per frame
Output: points — shape (T, N, 4), N radar-like points per frame

For each frame t in 0..T-1:
    1. SURFACE SAMPLING:
       - From the 22 joints, sample 1024 points with replacement
       - Add Gaussian noise (σ = 0.02 m) to approximate body surface
       → surface_points: (1024, 3)

    2. INTENSITY MODEL (simplified radar cross-section):
       - Radar position: origin (0, 0, 0)
       - For each surface point p:
           distance = ||p - radar_pos||
           intensity = 1.0 / (distance² + ε)     # ε = 1e-6
       → intensities: (1024,)

    3. TOP-K SELECTION (mimics paper's peak picking from Doppler-FFT):
       - Sort by intensity descending
       - Select top N=128 points
       → selected: (128, 3)

    4. APPEND TIME COORDINATE:
       - t_value = t / fps
       - Concatenate to get (x, y, z, t)
       → frame_points: (128, 4)

    points[t] = frame_points

Return points  # (T, 128, 4)
```

This mirrors the paper's pipeline conceptually:
- Surface sampling ≈ ray-mesh intersection
- Intensity model ≈ IF signal power
- Top-K selection ≈ peak detection on Doppler-FFT heatmap
- 128 points per frame matches the paper exactly

### 3.3 When to Synthesize vs. Use Pre-computed

- **Pre-computed available**: If `RadarLLM-data/synthetic_points/rec_{id}.npy` exists, load it directly.
- **On-the-fly**: If not available, load `new_joints/{id}.npy` and run the synthesis algorithm above.

---

## 4. Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                      RadarAgent System                       │
│                                                              │
│  User: "What is this person doing? motion_id=000021"         │
│           │                                                  │
│           ▼                                                  │
│  ┌────────────────────────────────────────────────────┐      │
│  │            Qwen 3 8B  (Agent Core)                 │      │
│  │                                                    │      │
│  │  System prompt: radar domain expert                │      │
│  │  Reasoning: ReAct loop (think → act → observe)     │      │
│  │  Output: function calls or final text answer       │      │
│  └───────┬────────────┬────────────┬──────────────────┘      │
│          │            │            │                          │
│    ┌─────▼─────┐ ┌────▼────┐ ┌────▼──────┐                  │
│    │  Radar    │ │ Motion  │ │ Data      │                   │
│    │  Process  │ │ Analyze │ │ Retrieve  │                   │
│    │  Tools    │ │ Tools   │ │ Tools     │                   │
│    └─────┬─────┘ └────┬────┘ └────┬──────┘                  │
│          │            │            │                          │
│    ┌─────▼────────────▼────────────▼──────┐                  │
│    │          HumanML3D Data Layer         │                  │
│    │   new_joints/  texts/  RadarLLM-data/ │                  │
│    └──────────────────────────────────────┘                  │
│                                                              │
│  Agent: "The person squats down low and then jumps up..."    │
└──────────────────────────────────────────────────────────────┘
```

The agent follows a **ReAct loop**:

1. Receive user query (e.g., "Describe this motion" or "Compare motion A and B")
2. **Think**: reason about what information is needed
3. **Act**: call one or more tools
4. **Observe**: receive structured text results from tools
5. Repeat 2–4 until the agent has enough information
6. **Answer**: synthesize a final natural language response

---

## 5. Tool Specifications

Each tool is a Python function that the LLM can invoke via Qwen 3's function-calling format. Tools return Python dicts that get serialized to JSON strings for the LLM to read.

### Tool 1: `load_radar_sequence`

**Purpose**: Load or synthesize a radar point cloud sequence and return summary statistics (not raw numbers).

**Parameters**:
```json
{
  "name": "load_radar_sequence",
  "description": "Load a radar point cloud sequence for a given motion ID. Returns shape, duration, and per-frame spatial statistics. If pre-computed synthetic points exist, loads them; otherwise synthesizes from joint data.",
  "parameters": {
    "type": "object",
    "properties": {
      "motion_id": {
        "type": "string",
        "description": "HumanML3D motion identifier, e.g. '000021'"
      }
    },
    "required": ["motion_id"]
  }
}
```

**Implementation logic**:
1. Try loading `RadarLLM-data/synthetic_points/rec_{motion_id}.npy`
2. If not found, load `new_joints/{motion_id}.npy` and run the synthesis from Section 3.2
3. Compute and return:
   ```python
   {
     "motion_id": str,
     "num_frames": int,          # T
     "points_per_frame": int,    # 128
     "fps": 20,
     "duration_sec": float,      # T / 20
     "spatial_bounds": {
       "x": {"min": float, "max": float},
       "y": {"min": float, "max": float},
       "z": {"min": float, "max": float}
     },
     "center_of_mass_start": [float, float, float],  # mean of points in frame 0
     "center_of_mass_end": [float, float, float],     # mean of points in last frame
     "overall_displacement": float  # Euclidean distance between start/end CoM
   }
   ```

### Tool 2: `extract_radar_features`

**Purpose**: Core analytical tool. Extracts spatiotemporal features from a radar point cloud and returns them as structured text the LLM can reason about.

**Parameters**:
```json
{
  "name": "extract_radar_features",
  "description": "Extract detailed spatiotemporal features from a radar point cloud sequence. Returns velocity profile, periodicity analysis, spatial extent, motion complexity, and body region activity. This is the primary tool for understanding what motion is occurring.",
  "parameters": {
    "type": "object",
    "properties": {
      "motion_id": {
        "type": "string",
        "description": "HumanML3D motion identifier"
      }
    },
    "required": ["motion_id"]
  }
}
```

**Implementation logic**:

1. Load the point cloud `(T, 128, 4)`.
2. Compute **center of mass trajectory**: for each frame, mean of all 128 points → `(T, 3)`.
3. Compute **velocity profile**: frame-to-frame displacement of CoM × fps → `(T-1,)` in m/s.
4. Compute **point cloud spread**: per-frame standard deviation of point positions → measures how spread out the body is.
5. Compute **periodicity**: autocorrelation of the velocity signal. If a strong peak exists after the zero-lag, report the period.
6. Compute **dominant motion axis**: which of x/y/z has the largest CoM range.
7. Compute **body region analysis** using z-axis height splitting:
   - Points with z > median_z → "upper body region"
   - Points with z ≤ median_z → "lower body region"
   - For each region: compute mean velocity, activity level
8. Compute **motion complexity**: standard deviation of velocity (low = steady, high = complex).
9. Compute **vertical change**: max_z - min_z of CoM over time (detects sitting, jumping, crouching).

Return:
```python
{
  "motion_id": str,
  "duration_sec": float,
  "velocity": {
    "mean_m_per_s": float,
    "max_m_per_s": float,
    "min_m_per_s": float,
    "std_m_per_s": float
  },
  "dominant_motion_axis": str,        # "x", "y", or "z"
  "dominant_axis_range_m": float,     # range of CoM on dominant axis
  "periodicity": {
    "is_periodic": bool,
    "estimated_period_sec": float or null,
    "confidence": str                 # "high", "medium", "low"
  },
  "vertical_dynamics": {
    "com_height_start_m": float,
    "com_height_end_m": float,
    "com_height_min_m": float,
    "com_height_max_m": float,
    "vertical_range_m": float
  },
  "body_spread": {
    "mean_spread_m": float,           # avg std of point positions
    "spread_change": str              # "expanding", "contracting", "stable"
  },
  "upper_body": {
    "activity_level": str,            # "low", "medium", "high"
    "mean_velocity_m_per_s": float
  },
  "lower_body": {
    "activity_level": str,
    "mean_velocity_m_per_s": float
  },
  "motion_complexity": str,           # "simple", "moderate", "complex"
  "trajectory_shape": str             # "stationary", "linear", "curved", "back_and_forth"
}
```

**How to compute each field** (detailed algorithms):

**Velocity profile**:
```
com = mean(points[:, :, :3], axis=1)         # (T, 3)
displacement = np.diff(com, axis=0)          # (T-1, 3)
speed = np.linalg.norm(displacement, axis=1) # (T-1,)
speed_m_per_s = speed * fps                  # convert frame-displacement to m/s
```

**Periodicity** (autocorrelation method):
```
velocity_signal = speed_m_per_s - mean(speed_m_per_s)
autocorr = np.correlate(velocity_signal, velocity_signal, mode='full')
autocorr = autocorr[len(autocorr)//2:]       # keep positive lags only
autocorr = autocorr / autocorr[0]            # normalize
# Find first peak after lag 0 (skip first few frames to avoid zero-lag)
min_lag = int(0.3 * fps)  # at least 0.3 seconds
peaks = find_peaks(autocorr[min_lag:])
if peaks exist and autocorr at peak > 0.3:
    period_frames = peaks[0] + min_lag
    period_sec = period_frames / fps
    is_periodic = True
```

**Dominant motion axis**:
```
com_range_x = max(com[:, 0]) - min(com[:, 0])
com_range_y = max(com[:, 1]) - min(com[:, 1])
com_range_z = max(com[:, 2]) - min(com[:, 2])
dominant = argmax([com_range_x, com_range_y, com_range_z])
```

**Body region analysis**:
```
For each frame t:
    all_z = points[t, :, 2]
    median_z = np.median(all_z)
    upper_mask = all_z > median_z
    lower_mask = ~upper_mask
    upper_points[t] = points[t, upper_mask, :3]
    lower_points[t] = points[t, lower_mask, :3]

For each region, compute mean of per-frame centroids, then velocity the same way.
Activity level: mean_velocity < 0.2 → "low", < 0.8 → "medium", else "high"
```

**Trajectory shape**:
```
start = com[0, :2]  # x,y only (ground plane)
end = com[-1, :2]
net_displacement = np.linalg.norm(end - start)
total_path_length = sum of all frame-to-frame displacements in xy
if total_path_length < 0.3:
    shape = "stationary"
elif net_displacement / total_path_length > 0.8:
    shape = "linear"
elif net_displacement / total_path_length < 0.3:
    shape = "back_and_forth"
else:
    shape = "curved"
```

### Tool 3: `get_motion_text`

**Purpose**: Retrieve the ground-truth text annotations for a motion. Useful for the agent to compare its radar-based analysis with known descriptions (during development), or to provide context.

**Parameters**:
```json
{
  "name": "get_motion_text",
  "description": "Get the text annotations for a motion from the HumanML3D dataset. Returns all available descriptions.",
  "parameters": {
    "type": "object",
    "properties": {
      "motion_id": {
        "type": "string",
        "description": "HumanML3D motion identifier"
      }
    },
    "required": ["motion_id"]
  }
}
```

**Implementation logic**:
1. Read `texts/{motion_id}.txt`
2. Parse each line: split on `#`, take index 0 as the description
3. Return:
   ```python
   {
     "motion_id": str,
     "descriptions": [str, str, str, ...],  # 3-4 descriptions
     "num_descriptions": int
   }
   ```

### Tool 4: `search_motions`

**Purpose**: Search for motions by text query using keyword matching or semantic similarity.

**Parameters**:
```json
{
  "name": "search_motions",
  "description": "Search the HumanML3D dataset for motions whose text descriptions match a query. Returns top matching motion IDs with their descriptions.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Natural language search query, e.g. 'walking forward'"
      },
      "top_k": {
        "type": "integer",
        "description": "Number of results to return (default 5)",
        "default": 5
      }
    },
    "required": ["query"]
  }
}
```

**Implementation logic**:
1. On first call, build an index:
   - Load all `texts/*.txt` files
   - For each motion, extract all descriptions (split on `#`, take index 0)
   - Compute sentence embeddings using `sentence-transformers` (e.g., `all-MiniLM-L6-v2`)
   - Store in memory (or a FAISS index for speed)
2. Embed the query
3. Cosine similarity search, return top_k
4. Return:
   ```python
   {
     "query": str,
     "results": [
       {
         "motion_id": str,
         "descriptions": [str, ...],
         "similarity_score": float
       },
       ...
     ]
   }
   ```

### Tool 5: `compare_motions`

**Purpose**: Compare two motion sequences side by side.

**Parameters**:
```json
{
  "name": "compare_motions",
  "description": "Compare two radar point cloud sequences and return their similarities and differences in velocity, spatial extent, periodicity, and body region activity.",
  "parameters": {
    "type": "object",
    "properties": {
      "motion_id_a": {"type": "string"},
      "motion_id_b": {"type": "string"}
    },
    "required": ["motion_id_a", "motion_id_b"]
  }
}
```

**Implementation logic**:
1. Call `extract_radar_features` internally for both motions
2. Compare each field and produce a summary:
   ```python
   {
     "motion_a": str,
     "motion_b": str,
     "duration_comparison": str,    # e.g., "A is 2.3s longer than B"
     "velocity_comparison": str,    # e.g., "A is faster (1.2 vs 0.4 m/s)"
     "spatial_comparison": str,     # e.g., "A covers more ground"
     "periodicity_comparison": str, # e.g., "Both are periodic; A has shorter cycle"
     "complexity_comparison": str,  # e.g., "B is more complex"
     "key_differences": [str, ...], # top 3 distinguishing factors
     "overall_similarity": str      # "very similar", "somewhat similar", "different"
   }
   ```

### Tool 6: `analyze_joint_motion`

**Purpose**: Analyze motion at the individual joint level (using joint data directly, not radar points). Provides fine-grained body part analysis.

**Parameters**:
```json
{
  "name": "analyze_joint_motion",
  "description": "Analyze motion at the skeleton joint level. Reports which body parts move most, detects arm/leg actions, and identifies poses. Uses the 22-joint SMPL skeleton.",
  "parameters": {
    "type": "object",
    "properties": {
      "motion_id": {"type": "string"}
    },
    "required": ["motion_id"]
  }
}
```

**Implementation logic**:
1. Load `new_joints/{motion_id}.npy` → `(T, 22, 3)`
2. Compute per-joint velocity: `joint_vel[j] = mean(||joints[1:,j] - joints[:-1,j]|| * fps)`
3. Group joints by body part:
   - Head: joints [12, 15]
   - Torso: joints [0, 3, 6, 9]
   - Left arm: joints [13, 16, 18, 20]
   - Right arm: joints [14, 17, 19, 21]
   - Left leg: joints [1, 4, 7, 10]
   - Right leg: joints [2, 5, 8, 11]
4. For each body part, compute:
   - Mean velocity across its joints
   - Max displacement from initial position
   - Whether the part returns to starting position (cyclic motion)
5. Detect specific patterns:
   - **Arm raise**: shoulder/elbow/wrist z-coordinate increases significantly
   - **Kick**: foot joint has high velocity spike + z-coordinate increase
   - **Crouch/squat**: pelvis (joint 0) z drops then recovers
   - **Walk**: alternating left/right foot forward movement, periodic
   - **Turn**: pelvis x/y trajectory shows rotation (use heading angle changes)
6. Return:
   ```python
   {
     "motion_id": str,
     "body_part_activity": {
       "head":      {"velocity": float, "displacement": float, "activity": str},
       "torso":     {"velocity": float, "displacement": float, "activity": str},
       "left_arm":  {"velocity": float, "displacement": float, "activity": str},
       "right_arm": {"velocity": float, "displacement": float, "activity": str},
       "left_leg":  {"velocity": float, "displacement": float, "activity": str},
       "right_leg": {"velocity": float, "displacement": float, "activity": str}
     },
     "most_active_parts": [str, str],    # top 2 most active body parts
     "detected_actions": [str, ...],     # e.g., ["walking", "arm_swing"]
     "symmetry": str,                    # "symmetric", "left_dominant", "right_dominant"
     "root_trajectory": str              # description of pelvis path
   }
   ```

### Tool 7: `visualize_motion`

**Purpose**: Generate a visualization image of the point cloud or skeleton.

**Parameters**:
```json
{
  "name": "visualize_motion",
  "description": "Generate and save a visualization of a motion sequence showing the radar point cloud or skeleton at selected frames.",
  "parameters": {
    "type": "object",
    "properties": {
      "motion_id": {"type": "string"},
      "mode": {
        "type": "string",
        "enum": ["point_cloud", "skeleton", "trajectory"],
        "description": "Visualization type"
      },
      "num_frames": {
        "type": "integer",
        "description": "How many frames to display (evenly spaced, default 6)",
        "default": 6
      }
    },
    "required": ["motion_id"]
  }
}
```

**Implementation logic**:
- `point_cloud`: Show N evenly-spaced frames as 3D scatter plots in a row, color-coded by height
- `skeleton`: Show N frames with 22 joints connected by kinematic chain edges
- `trajectory`: Show CoM trajectory in top-down (xy) and side (xz) views
- Save to `outputs/viz_{motion_id}_{mode}.png`
- Return `{"image_path": str}`

---

## 6. Agent Core

### 6.1 Qwen 3 8B Setup

Use the `transformers` library to load `Qwen/Qwen3-8B` (or the instruction-tuned chat variant). Qwen 3 supports function calling natively in its chat template.

**Model loading**:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
```

**Alternatively**, use `vllm` for faster inference:
```python
from vllm import LLM
llm = LLM(model="Qwen/Qwen3-8B")
```

### 6.2 Function Calling Format

Qwen 3 uses a specific format for tool/function calling in its chat template. The conversation format:

```python
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": user_query},
]

# Tools are passed to the tokenizer's apply_chat_template:
tools = [
    {
        "type": "function",
        "function": {
            "name": "load_radar_sequence",
            "description": "...",
            "parameters": { ... }  # JSON Schema
        }
    },
    # ... more tools
]

text = tokenizer.apply_chat_template(
    messages,
    tools=tools,
    tokenize=False,
    add_generation_prompt=True
)
```

When the model wants to call a tool, it outputs a structured response. Parse the response, execute the tool, and append the result back:

```python
messages.append({"role": "assistant", "content": assistant_response_with_tool_call})
messages.append({"role": "tool", "name": "tool_name", "content": json.dumps(tool_result)})
```

Then continue generation for the next step.

### 6.3 System Prompt

```python
SYSTEM_PROMPT = """You are RadarAgent, an expert system for understanding human motion from millimeter-wave radar point cloud data.

## Your Capabilities
You analyze human motion sequences from the HumanML3D dataset. Each motion consists of radar-like point cloud frames where each frame has 128 3D points representing reflections from a person's body, captured at 20 FPS.

## How to Analyze Motion
When asked to describe or analyze a motion:
1. First call `load_radar_sequence` to get basic info (duration, spatial bounds, displacement).
2. Then call `extract_radar_features` for detailed analysis (velocity, periodicity, body regions, complexity).
3. If you need fine-grained body part information, call `analyze_joint_motion`.
4. Synthesize all tool outputs into a natural language description.

## Domain Knowledge
- Millimeter-wave radar senses motion through sparse 3D point clouds (x, y, z coordinates).
- 128 points per frame at 20 FPS, sequences are 2-10 seconds long.
- The z-axis is vertical (height). Higher z = above ground.
- Point density near a body part indicates stronger radar reflection.
- Periodic velocity patterns suggest repetitive motions (walking, waving).
- High vertical dynamics suggest jumping, crouching, or sitting.
- Asymmetric limb activity suggests one-sided actions (throwing, kicking).

## Output Guidelines
- Describe motions in natural language, focusing on WHAT the person does, not numbers.
- Start with the primary action, then add details about limb movements and trajectory.
- Use concrete action verbs: "walks", "reaches", "kicks", "turns", "squats".
- Mention direction and speed when relevant.
- Do NOT repeat raw numbers from tool outputs; interpret them into human-readable descriptions.
"""
```

### 6.4 Agent Loop Implementation

```python
class RadarAgent:
    def __init__(self, model, tokenizer, tools, data_dir):
        self.model = model
        self.tokenizer = tokenizer
        self.tools = tools           # dict: name → callable
        self.tool_schemas = [...]    # JSON schemas for Qwen 3
        self.data_dir = data_dir
        self.max_iterations = 8      # prevent infinite loops

    def run(self, user_query: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
        ]

        for iteration in range(self.max_iterations):
            # Generate next response
            response = self._generate(messages)

            # Check if the model wants to call a tool
            tool_calls = self._parse_tool_calls(response)

            if not tool_calls:
                # Model produced a final text answer
                return response

            # Execute each tool call
            messages.append({"role": "assistant", "content": response})
            for call in tool_calls:
                result = self.tools[call["name"]](**call["arguments"])
                messages.append({
                    "role": "tool",
                    "name": call["name"],
                    "content": json.dumps(result, default=str)
                })

        # If max iterations reached, force a final answer
        messages.append({"role": "user", "content": "Please provide your final answer now based on the information gathered."})
        return self._generate(messages)

    def _generate(self, messages):
        text = self.tokenizer.apply_chat_template(
            messages,
            tools=self.tool_schemas,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.7,
            do_sample=True
        )
        return self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

    def _parse_tool_calls(self, response: str):
        # Parse Qwen 3's function call format from the response text.
        # Qwen 3 outputs tool calls in a structured format within the response.
        # Return list of {"name": str, "arguments": dict} or empty list.
        ...
```

### 6.5 Tool Registry and Dispatch

```python
class ToolRegistry:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self._search_index = None   # lazy-loaded sentence embedding index

    def load_radar_sequence(self, motion_id: str) -> dict:
        ...

    def extract_radar_features(self, motion_id: str) -> dict:
        ...

    def get_motion_text(self, motion_id: str) -> dict:
        ...

    def search_motions(self, query: str, top_k: int = 5) -> dict:
        ...

    def compare_motions(self, motion_id_a: str, motion_id_b: str) -> dict:
        ...

    def analyze_joint_motion(self, motion_id: str) -> dict:
        ...

    def visualize_motion(self, motion_id: str, mode: str = "point_cloud", num_frames: int = 6) -> dict:
        ...

    def get_tool_map(self) -> dict:
        """Return {name: callable} for all tools."""
        return {
            "load_radar_sequence": self.load_radar_sequence,
            "extract_radar_features": self.extract_radar_features,
            "get_motion_text": self.get_motion_text,
            "search_motions": self.search_motions,
            "compare_motions": self.compare_motions,
            "analyze_joint_motion": self.analyze_joint_motion,
            "visualize_motion": self.visualize_motion,
        }

    def get_tool_schemas(self) -> list:
        """Return list of JSON schemas for Qwen 3 function calling."""
        return [ ... ]  # The schemas from Section 5
```

---

## 7. Project Structure

```
RadarAgent/
├── IMPLEMENTATION_PLAN.md          # This file
├── README.md                       # Usage instructions
├── requirements.txt                # Python dependencies
├── config.py                       # All paths and hyperparameters
├── main.py                         # CLI entry point
│
├── tools/                          # Tool implementations
│   ├── __init__.py
│   ├── registry.py                 # ToolRegistry class
│   ├── radar_processing.py         # load_radar_sequence, extract_radar_features
│   ├── joint_analysis.py           # analyze_joint_motion
│   ├── data_retrieval.py           # get_motion_text, search_motions
│   ├── comparison.py               # compare_motions
│   ├── visualization.py            # visualize_motion
│   └── radar_synthesis.py          # motion_to_radar_pointcloud (Section 3.2)
│
├── agent/                          # Agent core
│   ├── __init__.py
│   ├── agent.py                    # RadarAgent class (ReAct loop)
│   ├── llm.py                      # Qwen 3 8B loading and generation
│   ├── prompts.py                  # System prompt and few-shot examples
│   └── tool_schemas.py             # JSON schemas for all tools
│
├── eval/                           # Evaluation
│   ├── __init__.py
│   ├── evaluate.py                 # Run evaluation on test set
│   └── metrics.py                  # ROUGE, BLEU, METEOR, CIDEr, BERTScore
│
├── outputs/                        # Generated outputs
│   ├── viz/                        # Visualization images
│   └── results/                    # Evaluation results
│
└── scripts/
    ├── run_interactive.sh          # Interactive agent session
    └── run_eval.sh                 # Batch evaluation
```

---

## 8. Step-by-Step Build Instructions

### Step 1: `config.py`

Central configuration file. Every path and hyperparameter lives here.

```python
from pathlib import Path

# === Paths (relative to this project) ===
PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR.parent / "HumanML3D" / "HumanML3D" / "HumanML3D"

JOINTS_DIR = DATA_DIR / "new_joints"
TEXTS_DIR = DATA_DIR / "texts"
SPLITS_DIR = DATA_DIR  # train.txt, val.txt, test.txt are here
RADAR_DATA_DIR = DATA_DIR / "RadarLLM-data"
SYNTHETIC_POINTS_DIR = RADAR_DATA_DIR / "synthetic_points"

OUTPUT_DIR = PROJECT_DIR / "outputs"
VIZ_DIR = OUTPUT_DIR / "viz"
RESULTS_DIR = OUTPUT_DIR / "results"

# === Model ===
MODEL_NAME = "Qwen/Qwen3-8B"

# === Radar synthesis parameters (from the paper) ===
RADAR_FPS = 20
POINTS_PER_FRAME = 128
SURFACE_SAMPLE_COUNT = 1024
SURFACE_JITTER_STD = 0.02       # meters
RADAR_POSITION = [0.0, 0.0, 0.0]

# === SMPL skeleton (22 joints) ===
NUM_JOINTS = 22
KINEMATIC_CHAINS = [
    [0, 2, 5, 8, 11],       # right leg
    [0, 1, 4, 7, 10],       # left leg
    [0, 3, 6, 9, 12, 15],   # spine → head
    [9, 14, 17, 19, 21],    # right arm
    [9, 13, 16, 18, 20],    # left arm
]
BODY_PART_JOINTS = {
    "head":      [12, 15],
    "torso":     [0, 3, 6, 9],
    "left_arm":  [13, 16, 18, 20],
    "right_arm": [14, 17, 19, 21],
    "left_leg":  [1, 4, 7, 10],
    "right_leg": [2, 5, 8, 11],
}

# === Agent ===
MAX_AGENT_ITERATIONS = 8
GENERATION_MAX_TOKENS = 2048
GENERATION_TEMPERATURE = 0.7

# === Evaluation ===
EVAL_SPLIT = "test"
```

### Step 2: `tools/radar_synthesis.py`

Implements the simplified radar point cloud synthesis from Section 3.2.

Must implement these functions:
- `sample_surface_points(joints_frame: np.ndarray, num_surface=1024, jitter_std=0.02) -> np.ndarray`
  - Input: `(22, 3)`, Output: `(1024, 3)`
  - Randomly sample joints with replacement, add Gaussian noise
- `compute_intensity(points: np.ndarray, radar_pos=np.array([0,0,0])) -> np.ndarray`
  - Input: `(M, 3)`, Output: `(M,)` — intensity ∝ 1/r²
- `select_top_k(points: np.ndarray, intensities: np.ndarray, k=128) -> np.ndarray`
  - Sort by intensity descending, take top k
- `motion_to_radar_pointcloud(joints: np.ndarray, fps=20, N=128) -> np.ndarray`
  - Input: `(T, 22, 3)`, Output: `(T, 128, 4)` — the full synthesis pipeline

### Step 3: `tools/radar_processing.py`

Implements `load_radar_sequence` and `extract_radar_features`.

Must implement:
- Loading from pre-computed or on-the-fly synthesis
- All feature extraction algorithms described in Tool 2 (Section 5):
  - CoM trajectory, velocity profile, periodicity (autocorrelation), dominant axis, body region splitting, trajectory shape classification, motion complexity
- Use `scipy.signal.find_peaks` for periodicity detection

### Step 4: `tools/joint_analysis.py`

Implements `analyze_joint_motion`.

Must implement:
- Per-joint velocity computation
- Body part grouping using `BODY_PART_JOINTS` from config
- Action pattern detection (arm raise, kick, crouch, walk, turn)
- Symmetry analysis (compare left vs right side velocities)

### Step 5: `tools/data_retrieval.py`

Implements `get_motion_text` and `search_motions`.

Must implement:
- Text file parsing (split on `#`, take index 0)
- For `search_motions`: build a sentence embedding index using `sentence-transformers`
  - Model: `all-MiniLM-L6-v2` (small, fast, good quality)
  - Build lazily on first call, cache in memory
  - Use cosine similarity for ranking

### Step 6: `tools/comparison.py`

Implements `compare_motions`.

Must implement:
- Call `extract_radar_features` for both motions
- Field-by-field comparison with natural language summaries

### Step 7: `tools/visualization.py`

Implements `visualize_motion`.

Must implement:
- `matplotlib` 3D scatter for point clouds
- Skeleton drawing with kinematic chain edges
- Top-down and side-view trajectory plots
- Save to `outputs/viz/`

### Step 8: `tools/registry.py`

The `ToolRegistry` class that:
- Instantiates all tools with the data directory
- Provides `get_tool_map()` returning `{name: callable}`
- Provides `get_tool_schemas()` returning the JSON schemas list

### Step 9: `agent/prompts.py`

Contains `SYSTEM_PROMPT` (from Section 6.3) and optional few-shot examples.

### Step 10: `agent/tool_schemas.py`

Contains the complete JSON schema list for all 7 tools, exactly as specified in Section 5. These are passed to `tokenizer.apply_chat_template(tools=...)`.

### Step 11: `agent/llm.py`

Handles Qwen 3 8B model loading and text generation.

Must implement:
- `load_model(model_name: str)` → returns `(model, tokenizer)`
- `generate(model, tokenizer, messages, tools, max_new_tokens, temperature)` → returns response string
- Handle GPU memory: use `device_map="auto"`, `torch_dtype=torch.bfloat16`
- Parse Qwen 3 tool call format from generated text

**Qwen 3 tool call parsing**: When Qwen 3 decides to call a function, it generates structured output. The exact format depends on the chat template version, but typically:
- The response contains a function call block that can be parsed
- Use `tokenizer.apply_chat_template` with `tools` parameter to handle formatting
- After tool execution, append the result as a `"tool"` role message

### Step 12: `agent/agent.py`

The `RadarAgent` class implementing the ReAct loop (Section 6.4).

Must implement:
- `__init__`: load model, register tools
- `run(user_query: str) -> str`: the main loop
- `_generate`: call LLM with current messages
- `_parse_tool_calls`: extract tool calls from LLM output
- Handle edge cases: tool errors, max iterations, empty responses

### Step 13: `main.py`

CLI entry point supporting two modes:

1. **Interactive mode**: Chat with the agent in a loop
   ```bash
   python main.py --interactive
   ```
2. **Batch evaluation mode**: Run on test split and compute metrics
   ```bash
   python main.py --evaluate --split test --output results/test_results.json
   ```
3. **Single query mode**:
   ```bash
   python main.py --query "Describe motion 000021 from radar data"
   ```

### Step 14: `eval/metrics.py`

Implements evaluation metrics matching the paper:
- **ROUGE-1, ROUGE-L**: via `rouge-score` library
- **BLEU-1, BLEU-4**: via `nltk.translate.bleu_score`
- **METEOR**: via `nltk.translate.meteor_score`
- **CIDEr**: implement or use `pycocoevalcap`
- **BERTScore**: via `bert-score` library
- **SimCSE**: compute cosine similarity of sentence embeddings (use `princeton-nlp/sup-simcse-roberta-large`)

### Step 15: `eval/evaluate.py`

Evaluation loop:
1. Load test split IDs from `test.txt`
2. For each motion ID:
   - Construct query: `"Analyze the radar point cloud for motion {id} and describe what the person is doing."`
   - Run the agent
   - Collect the agent's generated description
3. Load ground-truth descriptions from `texts/{id}.txt`
4. Compute all metrics (each generated description vs. all ground-truth descriptions, take max)
5. Aggregate and save results

---

## 9. Evaluation

### 9.1 Metrics (matching the paper)

| Metric | Library | What It Measures |
|--------|---------|-----------------|
| ROUGE-1 | `rouge-score` | Unigram overlap |
| ROUGE-L | `rouge-score` | Longest common subsequence |
| BLEU-1 | `nltk` | Unigram precision |
| BLEU-4 | `nltk` | 4-gram precision |
| METEOR | `nltk` | Alignment-based F1 |
| CIDEr | `pycocoevalcap` | TF-IDF weighted n-gram similarity |
| BERTScore | `bert-score` | Contextual embedding similarity |
| SimCSE | `sentence-transformers` | Sentence-level semantic similarity |

### 9.2 Baselines for Comparison

From Table 1 of the paper (virtual test set):

| Model | ROUGE-L | BLEU-1 | BLEU-4 | METEOR | BERTScore |
|-------|---------|--------|--------|--------|-----------|
| MotionGPT* | 29.4 | 37.6 | 5.0 | 26.1 | 82.6 |
| AvatarGPT* | 30.0 | 36.3 | 5.0 | 28.3 | 82.4 |
| **RadarLLM** | **36.0** | **48.0** | **11.4** | **33.7** | **83.3** |

Our goal is not necessarily to beat RadarLLM (which is trained end-to-end) but to demonstrate that an agent-based approach is competitive, especially given zero training cost.

### 9.3 Ablation Studies to Run

1. **Tool ablation**: Disable one tool at a time, measure impact
   - Without `extract_radar_features` (agent can only use `load_radar_sequence`)
   - Without `analyze_joint_motion` (agent can only use radar point cloud tools)
2. **Feature ablation**: Remove specific features from `extract_radar_features`
3. **Prompt ablation**: Minimal system prompt vs. full domain knowledge prompt
4. **Model size**: If feasible, test with Qwen 3 4B or Qwen 3 1.7B for cost comparison

---

## 10. Dependencies

```
# requirements.txt

# LLM
transformers>=4.51.0
torch>=2.3.0
accelerate>=1.0.0

# Data processing
numpy>=1.26.0
scipy>=1.12.0

# Text search and retrieval
sentence-transformers>=3.0.0

# Evaluation metrics
rouge-score>=0.1.2
nltk>=3.9.0
bert-score>=0.3.13

# Visualization
matplotlib>=3.9.0

# Optional: faster LLM inference
# vllm>=0.8.0
```
