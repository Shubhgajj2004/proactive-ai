"""
Reactive Action Graph — triggered by wake word "hey jarvis".

Flow:
  intent → [intent clear?]
    ├── YES → plan → tool_select → execute → plan (loop) → respond → done
    └── NO  → clarifying question → wait_for_user → plan → ...

Key difference from proactive: NO consent gate.
The user explicitly invoked the assistant, so we act immediately.

At most ONE clarifying question is asked before proceeding.
"""
import json
import logging
from typing import Any

from langgraph.graph import StateGraph, END

from server.action.shared_nodes import (
    plan_node, plan_router, tool_select_node, execute_node, respond_node,
)
from server.action.state import ActionSessionState
from server.llm.client import LLMClient
from server.prompts.action import INTENT_SYSTEM, INTENT_USER_TEMPLATE

logger = logging.getLogger(__name__)


# ── Intent node ───────────────────────────────────────────────────────────────

async def intent_node(state: ActionSessionState, llm: LLMClient) -> dict:
    """
    Parse the wake word utterance to extract intent.
    Returns either a clear task description or a clarifying question.
    """
    from pydantic import BaseModel

    class IntentDecision(BaseModel):
        intent_clear:        bool
        parsed_intent:       str = ""
        clarifying_question: str = ""

    transcript = state.get("task_description", "")   # raw wake word utterance

    response = await llm.complete(
        messages=[
            {"role": "system", "content": INTENT_SYSTEM},
            {"role": "user",   "content": INTENT_USER_TEMPLATE.format(transcript=transcript)},
        ],
        response_format=IntentDecision,
        temperature=0.1,
    )

    try:
        decision = IntentDecision(**json.loads(response.content))
    except Exception:
        # If parse fails, treat as clear intent with the raw transcript
        decision = IntentDecision(intent_clear=True, parsed_intent=transcript)

    logger.info(
        "[INTENT] clear=%s  intent=%r  question=%r",
        decision.intent_clear, decision.parsed_intent[:60],
        decision.clarifying_question[:60],
    )

    if decision.intent_clear:
        return {
            "task_description": decision.parsed_intent or transcript,
            "messages": [{"role": "user", "content": transcript}],
            "_intent_clear": True,
            "_clarified":    False,
        }
    else:
        return {
            "messages": [
                {"role": "user",      "content": transcript},
                {"role": "assistant", "content": decision.clarifying_question},
            ],
            "_intent_clear":         False,
            "_clarifying_question":  decision.clarifying_question,
            "_clarified":            False,
        }


def intent_router(state: ActionSessionState) -> str:
    if state.get("_intent_clear"):
        return "plan"
    return "wait_for_clarification"


# ── Clarification handler ─────────────────────────────────────────────────────

async def clarification_node(state: ActionSessionState) -> dict:
    """
    Incorporate the user's clarification reply into task_description.
    Then proceed directly to plan — only one clarification allowed.
    """
    messages = state.get("messages", [])
    # Find the user's clarification reply (last user message after the question)
    user_replies = [
        m for m in messages
        if (isinstance(m, dict) and m.get("role") == "user")
        or (hasattr(m, "type") and m.type == "human")
    ]
    clarification = ""
    if user_replies:
        last = user_replies[-1]
        clarification = last.get("content", "") if isinstance(last, dict) else last.content

    original = state.get("task_description", "")
    combined = f"{original}. {clarification}".strip(". ")

    logger.info("[CLARIFY] combined task: %r", combined[:80])

    return {
        "task_description": combined,
        "_intent_clear":    True,
        "_clarified":       True,
    }


def build_reactive_graph(
    llm:      LLMClient,
    mcp_base: str = "http://localhost:8888/tools",
):
    """
    Build and compile the reactive LangGraph.

    Args:
        llm:      LLMClient for intent/plan/respond nodes
        mcp_base: MCP server base URL

    Returns:
        Compiled LangGraph
    """
    builder = StateGraph(ActionSessionState)

    async def _intent(s):      return await intent_node(s, llm)
    async def _clarify(s):     return await clarification_node(s)
    async def _plan(s):        return await plan_node(s, llm)
    async def _tool_select(s): return await tool_select_node(s)
    async def _execute(s):     return await execute_node(s, mcp_base)
    async def _respond(s):     return await respond_node(s, llm)

    builder.add_node("intent",       _intent)
    builder.add_node("clarification", _clarify)
    builder.add_node("plan",         _plan)
    builder.add_node("tool_select",  _tool_select)
    builder.add_node("execute",      _execute)
    builder.add_node("respond",      _respond)

    builder.set_entry_point("intent")

    builder.add_conditional_edges("intent", intent_router, {
        "plan":                 "plan",
        "wait_for_clarification": END,   # pause for user reply (streaming)
    })

    builder.add_edge("clarification", "plan")

    builder.add_conditional_edges("plan", plan_router, {
        "tool_select":   "tool_select",
        "respond":       "respond",
        "wait_for_user": END,
    })

    builder.add_edge("tool_select", "execute")
    builder.add_edge("execute",     "plan")
    builder.add_edge("respond",     END)

    return builder.compile()


async def run_reactive_session(
    transcript:     str,
    llm:            LLMClient,
    mcp_base:       str = "http://localhost:8888/tools",
    user_id:        str = "",
    session_id:     str = "",
    clarification:  str = "",   # user's reply if intent was ambiguous
) -> ActionSessionState:
    """
    Run a reactive session end-to-end.

    Args:
        transcript:    The full utterance including wake word
        llm:           LLMClient
        mcp_base:      MCP server base URL
        clarification: Optional user clarification if intent needs it
    """
    graph = build_reactive_graph(llm, mcp_base)

    state: dict[str, Any] = {
        "session_id":         session_id or "reactive-test",
        "user_id":            user_id or "test_user",
        "trigger_source":     "wake_word",
        "task_description":   transcript,   # intent_node will refine this
        "consent_prompt":     "",
        "proposed_action":    "",
        "messages":           [],
        "step_count":         0,
        "tool_results":       [],
        "final_response":     "",
        "session_memory_ops": [],
        "user_consented":     True,         # reactive = always consented
        "done":               False,
        "outcome":            "",
        "_next_step":         "",
        "_selected_tools":    [],
        "_intent_clear":      False,
        "_clarified":         False,
        "_clarifying_question": "",
    }

    # If intent was ambiguous and user provided clarification,
    # add it to messages before invoking so clarification_node can pick it up
    if clarification:
        state["messages"] = [{"role": "user", "content": clarification}]
        state["_clarified"] = True

    final = await graph.ainvoke(state)
    return final
