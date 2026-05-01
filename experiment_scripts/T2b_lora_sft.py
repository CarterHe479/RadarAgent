#!/usr/bin/env python3
"""
T2b – LoRA SFT: fine-tune Qwen3-8B on collected SFT data.

Training format (instruction-following with tool context):
  <system>: {SYSTEM_PROMPT}
  <user>:   Analyse motion {id}. [Feature context]:
            {feature_context}
            Describe this motion in one sentence.
  <assistant>: {target_description}

Uses LoRA (rank 16, alpha 32) via the `peft` library.
Fine-tuning on 1000 examples with a 4-bit quantised base model requires
~12 GB VRAM (fits on A100 40 GB comfortably; tight on RTX 3090).

Requirements (install into the venv if not present):
    pip install peft bitsandbytes trl

Usage:
    PYTHONPATH=RadarAgent RadarAgent/.venv/bin/python experiment_scripts/T2b_lora_sft.py \
        --data path/to/sft_data_train_n1000.jsonl \
        [--output-dir outputs/sft/lora_checkpoint] \
        [--epochs 3] [--batch-size 4] [--lora-rank 16]
"""

from __future__ import annotations
import argparse, json, logging, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from shared.eval_utils import setup_logging, PROJECT_DIR
os.chdir(PROJECT_DIR)
sys.path.insert(0, str(PROJECT_DIR))

logger = logging.getLogger(__name__)


INSTRUCTION_TEMPLATE = (
    "Analyse motion {motion_id}.\n\n"
    "[Feature context]\n{feature_context}\n\n"
    "Describe this motion in one sentence."
)


def load_sft_data(data_path: str) -> list[dict]:
    samples = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    logger.info("Loaded %d SFT samples from %s", len(samples), data_path)
    return samples


def build_chat_text(sample: dict, tokenizer, system_prompt: str) -> str:
    """Build the full chat-format string for one training example."""
    messages = [
        {"role": "system",    "content": system_prompt},
        {"role": "user",      "content": INSTRUCTION_TEMPLATE.format(
            motion_id=sample["motion_id"],
            feature_context=sample["feature_context"],
        )},
        {"role": "assistant", "content": sample["target"]},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="T2b: LoRA SFT for RadarAgent")
    parser.add_argument("--data",       required=True, help="Path to JSONL from T2a")
    parser.add_argument("--output-dir", default=str(PROJECT_DIR / "outputs" / "sft" / "lora"))
    parser.add_argument("--model-name", default="Qwen/Qwen3-8B")
    parser.add_argument("--epochs",     type=int,   default=3)
    parser.add_argument("--batch-size", type=int,   default=4)
    parser.add_argument("--grad-accum", type=int,   default=4)
    parser.add_argument("--lr",         type=float, default=2e-4)
    parser.add_argument("--lora-rank",  type=int,   default=16)
    parser.add_argument("--lora-alpha", type=int,   default=32)
    parser.add_argument("--max-length", type=int,   default=512)
    parser.add_argument("--load-4bit",  action="store_true", default=False,
                        help="Use 4-bit quantisation (QLoRA) to reduce VRAM")
    args = parser.parse_args()

    setup_logging()

    try:
        from peft import LoraConfig, get_peft_model, TaskType
        from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
        from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
        import torch
        from datasets import Dataset
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install peft bitsandbytes trl datasets")
        raise

    # ── Load tokenizer and model ──────────────────────────────────────────────
    logger.info("Loading tokenizer: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = dict(trust_remote_code=True, device_map="auto")
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        load_kwargs["dtype"] = torch.bfloat16

    logger.info("Loading model: %s  (4-bit=%s)", args.model_name, args.load_4bit)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **load_kwargs)

    # ── LoRA config ───────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Build dataset ─────────────────────────────────────────────────────────
    from agent.prompts import SYSTEM_PROMPT
    samples = load_sft_data(args.data)
    texts   = [build_chat_text(s, tokenizer, SYSTEM_PROMPT) for s in samples]

    dataset = Dataset.from_dict({"text": texts})
    logger.info("Dataset size: %d examples", len(dataset))

    # ── Training arguments ────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        dataloader_num_workers=0,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=args.max_length,
    )

    logger.info("Starting LoRA SFT training …")
    trainer.train()
    trainer.save_model(str(out_dir / "final"))
    tokenizer.save_pretrained(str(out_dir / "final"))
    logger.info("Saved fine-tuned model to %s/final", out_dir)
    print(f"\nLoRA checkpoint saved to: {out_dir}/final")


if __name__ == "__main__":
    main()
