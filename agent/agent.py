"""
RadarAgent – ReAct-style agent loop that drives Qwen 3 8B with tool calls.

Loop structure per turn:
  1. Build the message list (system + history + user query).
  2. Generate a response from the LLM.
  3. If the response contains <tool_call> blocks → execute each tool,
     append assistant + tool messages, go to 2.
  4. If the response is plain text → return it as the final answer.
  5. Safety: stop after MAX_AGENT_ITERATIONS rounds.
"""

from __future__ import annotations

import json
import logging
from typing import Any, List, Optional

from config import MAX_AGENT_ITERATIONS, GENERATION_MAX_TOKENS, GENERATION_TEMPERATURE
from agent.llm import load_model, generate, parse_tool_calls, strip_tool_call_tags
from agent.prompts import SYSTEM_PROMPT, FEW_SHOT_EXAMPLES
from agent.tool_schemas import TOOL_SCHEMAS
from tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class RadarAgent:
    """Qwen 3 8B agent with radar-processing tool access."""

    def __init__(
        self,
        model=None,
        tokenizer=None,
        model_name: Optional[str] = None,
    ) -> None:
        """Initialise the agent.

        Either pass an already-loaded (model, tokenizer) pair, or let the
        agent load the model by passing model_name (or use the default from
        config.py).

        Args:
            model:      Pre-loaded HuggingFace model (optional).
            tokenizer:  Pre-loaded tokenizer (optional).
            model_name: Model name/path to load if model is not provided.
        """
        if model is None or tokenizer is None:
            from agent.llm import load_model as _load
            from config import MODEL_NAME
            self.model, self.tokenizer = _load(model_name or MODEL_NAME)
        else:
            self.model = model
            self.tokenizer = tokenizer

        self.registry = ToolRegistry()
        self.tool_schemas = TOOL_SCHEMAS
        self.tool_map = self.registry.get_tool_map()

    # ── public interface ──────────────────────────────────────────────────────

    def run(self, user_query: str) -> str:
        """Run the agent for a single user query and return the final answer."""
        messages = self._build_initial_messages(user_query)

        for iteration in range(MAX_AGENT_ITERATIONS):
            logger.debug("Agent iteration %d", iteration + 1)

            response = generate(
                self.model,
                self.tokenizer,
                messages,
                self.tool_schemas,
                max_new_tokens=GENERATION_MAX_TOKENS,
                temperature=GENERATION_TEMPERATURE,
            )
            logger.debug("Raw response: %s", response[:200])

            tool_calls = parse_tool_calls(response)

            if not tool_calls:
                # Plain text answer – done.
                return strip_tool_call_tags(response)

            # Append the assistant's message (including tool-call blocks)
            messages.append({"role": "assistant", "content": response})

            # Execute each tool call and append results
            for call in tool_calls:
                result = self.registry.call(call["name"], call["arguments"])
                logger.debug("Tool %s → %s", call["name"], str(result)[:200])
                messages.append({
                    "role": "tool",
                    "name": call["name"],
                    "content": json.dumps(result, default=str),
                })

        # Max iterations reached – force a final answer
        logger.warning("Max iterations (%d) reached; requesting final answer.", MAX_AGENT_ITERATIONS)
        messages.append({
            "role": "user",
            "content": (
                "You have gathered enough information. "
                "Please now provide your final answer as a concise motion description."
            ),
        })
        return generate(
            self.model,
            self.tokenizer,
            messages,
            self.tool_schemas,
            max_new_tokens=512,
            temperature=0.3,
        )

    # ── internal helpers ──────────────────────────────────────────────────────

    def _build_initial_messages(self, user_query: str) -> List[dict]:
        """Build the message list with system prompt, optional few-shot, and query."""
        msgs: List[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

        for user_ex, assistant_ex in FEW_SHOT_EXAMPLES:
            msgs.append({"role": "user",      "content": user_ex})
            msgs.append({"role": "assistant", "content": assistant_ex})

        msgs.append({"role": "user", "content": user_query})
        return msgs
