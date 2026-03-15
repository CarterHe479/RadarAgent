"""
Qwen 3 8B model loading and generation.

Qwen 3 function-calling format
───────────────────────────────
When the model wants to call a tool it emits:
    <tool_call>
    {"name": "...", "arguments": {...}}
    </tool_call>

The caller must then append an assistant message containing that text,
followed by a "tool" role message with the result, then continue.

Tool result messages follow the format:
    {"role": "tool", "content": "<json string>", "name": "<tool_name>"}
"""

from __future__ import annotations

import json
import re
import torch
from typing import Any, List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import MODEL_NAME, GENERATION_MAX_TOKENS, GENERATION_TEMPERATURE


# ── Tool-call parsing ─────────────────────────────────────────────────────────

_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)


def parse_tool_calls(response: str) -> List[Dict[str, Any]]:
    """Extract tool calls from a Qwen 3 response string.

    Returns a list of dicts: [{"name": str, "arguments": dict}, ...]
    Empty list means the response is a plain text answer.
    """
    calls = []
    for match in _TOOL_CALL_RE.finditer(response):
        try:
            obj = json.loads(match.group(1))
            name = obj.get("name") or obj.get("function", {}).get("name")
            args = obj.get("arguments") or obj.get("parameters") or {}
            if isinstance(args, str):
                args = json.loads(args)
            if name:
                calls.append({"name": name, "arguments": args})
        except (json.JSONDecodeError, KeyError):
            continue
    return calls


def strip_tool_call_tags(response: str) -> str:
    """Remove <tool_call>…</tool_call> blocks from a response, trimming whitespace."""
    cleaned = _TOOL_CALL_RE.sub("", response).strip()
    return cleaned


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(
    model_name: str = MODEL_NAME,
):
    """Load Qwen 3 8B in bfloat16 with automatic device mapping.

    Returns:
        (model, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


# ── Generation ────────────────────────────────────────────────────────────────

def generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: List[dict],
    tools: List[dict],
    max_new_tokens: int = GENERATION_MAX_TOKENS,
    temperature: float = GENERATION_TEMPERATURE,
) -> str:
    """Run one forward pass and return the decoded response string.

    The tools list is passed to apply_chat_template so the model knows
    which functions are available.
    """
    # Qwen 3 may support a `enable_thinking` flag; disable for tool-use tasks
    text = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)
