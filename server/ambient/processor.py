"""
Ambient Processor — the core LLM call that analyses each utterance.

Flow:
  1. Build context: relevant memories + capability manifest + transcript
  2. Call LLM with structured output → AmbientAnalysis
  3. Retry once on schema validation failure
  4. Return AmbientAnalysis to caller

The caller (pipeline) is responsible for:
  - Writing memory_operations to mem0 (memory_writer.py)
  - Writing summary/facts to DB (context_writer.py)
  - Spawning an ACTIVE session if confidence > 0.75 and should_act

Only wearer turns should be passed in the transcript.
Bystander speech is filtered upstream (voiceprint_matcher → is_wearer=True only).
"""
import json
import logging
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from server.llm.client import LLMClient
from server.llm.factory import make_llm_client
from server.prompts.ambient import SYSTEM, USER_TEMPLATE, RETRY_SCHEMA_REMINDER

logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.75   # above this → spawn ACTIVE session


# ── Output schema ─────────────────────────────────────────────────────────────

class MemoryOp(BaseModel):
    op:        Literal["add", "update", "delete"]
    fact:      str
    memory_id: str | None = None   # required for update/delete


class AmbientAnalysis(BaseModel):
    memory_operations: list[MemoryOp] = Field(default_factory=list)
    summary:           str
    extracted_facts:   list[str]     = Field(default_factory=list)
    tags:              list[str]     = Field(default_factory=list)
    should_act:        bool
    confidence:        float         = Field(ge=0.0, le=1.0)
    proposed_action:   str           = ""
    consent_prompt:    str           = ""
    reasoning:         str           = ""


# ── Processor ─────────────────────────────────────────────────────────────────

class AmbientProcessor:
    """
    Stateless ambient LLM processor.

    Args:
        client: Optional injected LLMClient (for testing). Uses factory if None.
    """

    def __init__(self, client: LLMClient | None = None):
        self._client = client or make_llm_client("ambient")

    async def analyse(
        self,
        transcript: str,
        memories: list[str] | None = None,
        capability_manifest: str = "",
    ) -> AmbientAnalysis:
        """
        Analyse a wearer transcript and return structured AmbientAnalysis.

        Args:
            transcript:          Wearer-only speech from this utterance.
            memories:            Relevant memories fetched from mem0 (list of strings).
            capability_manifest: One-liner tool descriptions for the system prompt.

        Returns:
            AmbientAnalysis — validated Pydantic model.

        Raises:
            ValueError if schema validation fails after retry.
        """
        memories_text = (
            "\n".join(f"- {m}" for m in memories)
            if memories else "No relevant memories found."
        )

        user_msg = USER_TEMPLATE.format(
            memories=memories_text,
            capability_manifest=capability_manifest or "No tools currently available.",
            transcript=transcript,
        )

        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": user_msg},
        ]

        # ── First attempt ─────────────────────────────────────────────────────
        response = await self._client.complete(
            messages=messages,
            response_format=AmbientAnalysis,
            temperature=0.3,
        )

        analysis = self._parse(response.content)
        if analysis:
            logger.info(
                "[AMBIENT] confidence=%.2f  should_act=%s  memory_ops=%d  tokens=%d",
                analysis.confidence, analysis.should_act,
                len(analysis.memory_operations),
                response.usage_input_tokens + response.usage_output_tokens,
            )
            return analysis

        # ── Retry once with schema reminder ──────────────────────────────────
        logger.warning("[AMBIENT] schema validation failed — retrying with reminder")
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user",      "content": RETRY_SCHEMA_REMINDER})

        retry_response = await self._client.complete(
            messages=messages,
            response_format=AmbientAnalysis,
            temperature=0.1,
        )
        analysis = self._parse(retry_response.content)
        if analysis:
            return analysis

        raise ValueError(
            f"AmbientProcessor: LLM output failed schema validation after retry.\n"
            f"Last output: {retry_response.content[:500]}"
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    def _parse(self, content: str) -> AmbientAnalysis | None:
        """Try to parse and validate LLM output. Returns None on failure."""
        raw = content.strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            lines = raw.splitlines()[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            raw = "\n".join(lines).strip()

        try:
            data = json.loads(raw)
            return AmbientAnalysis(**data)
        except (json.JSONDecodeError, ValidationError, TypeError) as e:
            logger.debug("[AMBIENT] parse error: %s", e)
            return None
