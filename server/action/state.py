"""Action session state — shared across proactive and reactive graphs."""
from typing import Annotated, Any
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class ActionSessionState(TypedDict):
    # ── Session metadata ──────────────────────────────────────────────────────
    session_id:          str
    user_id:             str
    trigger_source:      str          # "proactive_confidence" | "wake_word"

    # ── Task context ──────────────────────────────────────────────────────────
    task_description:    str          # what we're trying to do
    consent_prompt:      str          # spoken to user before starting (proactive only)
    proposed_action:     str          # from AmbientAnalysis

    # ── Conversation history (append-only via add_messages reducer) ───────────
    messages:            Annotated[list[dict], add_messages]

    # ── Execution tracking ────────────────────────────────────────────────────
    step_count:          int
    tool_results:        list[dict[str, Any]]   # accumulated tool outputs
    final_response:      str          # spoken response when done

    # ── Memory ops (flushed to mem0 on termination) ───────────────────────────
    session_memory_ops:  list[dict]   # accumulated during active turns

    # ── Control ───────────────────────────────────────────────────────────────
    user_consented:      bool | None  # None = not yet asked; True/False = answered
    done:                bool
    outcome:             str          # "confirmed" | "declined" | "timeout" | "completed"

    # ── Internal node communication (not persisted externally) ────────────────
    _next_step:             str          # plan → tool_select
    _selected_tools:        list[dict]   # tool_select → execute
    _intent_clear:          bool         # intent_node → intent_router
    _clarified:             bool         # clarification applied
    _clarifying_question:   str          # emitted by intent_node if ambiguous
