"""
Shared action agent nodes — used by both proactive and reactive graphs.

Nodes:
  plan_node       — LLM decides next step or marks done
  tool_select_node — top-2 tool schemas via pgvector
  execute_node    — calls the tool via executor.py
  respond_node    — composes final spoken response

All nodes receive and return ActionSessionState dicts.
"""
import json
import logging
from typing import Any

from server.action.state import ActionSessionState
from server.llm.client import LLMClient
from server.prompts.action import PLAN_SYSTEM, PLAN_USER_TEMPLATE
from server.tools.executor import call_tool, MCPCallError
from server.tools.selector import select_tools

logger = logging.getLogger(__name__)

# MCP server base URL — overridden in tests
MCP_BASE_URL = "http://localhost:8888/tools"


# ── Plan node ─────────────────────────────────────────────────────────────────

async def plan_node(state: ActionSessionState, llm: LLMClient) -> dict:
    """
    LLM decides the next step:
      - next_step: tool call description → route to tool_select
      - need_clarification: True → TTS question → wait for reply
      - done: True → route to respond
    """
    from pydantic import BaseModel

    class PlanDecision(BaseModel):
        next_step:          str   = ""
        need_clarification: bool  = False
        question:           str   = ""
        done:               bool  = False
        final_response:     str   = ""

    # Build tool schemas context (top-2 for current task)
    task = state.get("task_description", "")
    try:
        tools = await select_tools(task, top_k=2)
        tool_schemas = "\n".join(
            f"- {t.name} ({t.call_type}): {t.description}\n  schema: {json.dumps(t.schema)}"
            for t in tools
        ) or "No tools available."
    except Exception:
        tool_schemas = "No tools available."

    history_text = _format_history(state.get("tool_results", []), state.get("messages", []))

    messages = [
        {"role": "system",  "content": PLAN_SYSTEM},
        {"role": "user",    "content": PLAN_USER_TEMPLATE.format(
            task_description=task,
            tool_schemas=tool_schemas,
            history=history_text,
        )},
    ]

    response = await llm.complete(messages=messages, response_format=PlanDecision, temperature=0.2)

    try:
        decision = PlanDecision(**json.loads(response.content))
    except Exception:
        logger.warning("[PLAN] parse error — treating as done")
        decision = PlanDecision(done=True, final_response="I couldn't complete the task.")

    logger.info(
        "[PLAN] step=%d  done=%s  clarify=%s  next=%r",
        state.get("step_count", 0), decision.done,
        decision.need_clarification, decision.next_step[:60],
    )

    updates: dict = {"step_count": state.get("step_count", 0) + 1}

    if decision.done:
        updates["done"]           = True
        updates["final_response"] = decision.final_response
        updates["outcome"]        = "completed"

    elif decision.need_clarification:
        updates["messages"] = [{"role": "assistant", "content": decision.question}]

    else:
        # Store next_step description for tool_select to use
        updates["messages"] = [{"role": "assistant", "content": f"[PLAN] {decision.next_step}"}]
        updates["_next_step"] = decision.next_step

    return updates


def plan_router(state: ActionSessionState) -> str:
    """Route after plan_node."""
    if state.get("done"):
        return "respond"
    if state.get("step_count", 0) >= 8:   # turn limit
        return "respond"
    # _next_step is set by plan_node when a tool call is needed
    if state.get("_next_step"):
        return "tool_select"
    return "wait_for_user"


# ── Tool select node ──────────────────────────────────────────────────────────

async def tool_select_node(state: ActionSessionState) -> dict:
    """Fetch top-2 tool schemas for the current step."""
    next_step = state.get("_next_step", state.get("task_description", ""))
    try:
        tools = await select_tools(next_step, top_k=2)
        logger.info(
            "[TOOL_SELECT] step=%r → %s",
            next_step[:50], [t.name for t in tools],
        )
        return {"_selected_tools": [
            {"name": t.name, "call_type": t.call_type, "schema": t.schema}
            for t in tools
        ]}
    except Exception as e:
        logger.error("[TOOL_SELECT] failed: %s", e)
        return {"_selected_tools": []}


# ── Execute node ──────────────────────────────────────────────────────────────

async def execute_node(
    state:    ActionSessionState,
    mcp_base: str = MCP_BASE_URL,
) -> dict:
    """Execute the top tool. Surfaces errors back to plan node."""
    tools    = state.get("_selected_tools", [])
    next_step = state.get("_next_step", "")

    if not tools:
        return {"tool_results": state.get("tool_results", []) + [
            {"error": "no tools selected", "step": next_step}
        ]}

    top_tool   = tools[0]
    tool_name  = top_tool["name"]
    call_type  = top_tool.get("call_type", "read")
    endpoint   = f"{mcp_base}/{tool_name}"
    step_count = state.get("step_count", 0)

    # Extract arguments from next_step description (simplified — real impl uses LLM)
    # In production, plan_node would include structured arguments
    arguments: dict[str, Any] = {"query": next_step}

    result = await call_tool(
        tool_name=tool_name,
        call_type=call_type,
        arguments=arguments,
        endpoint=endpoint,
        session_id=state.get("session_id", ""),
        step_count=step_count,
    )

    if isinstance(result, MCPCallError):
        tool_result = {"error": result.to_dict(), "step": next_step}
        logger.warning("[EXECUTE] tool=%s failed: %s", tool_name, result.message)
    else:
        tool_result = {"tool": tool_name, "result": result, "step": next_step}
        logger.info("[EXECUTE] tool=%s success", tool_name)

    existing = state.get("tool_results", [])
    return {
        "tool_results": existing + [tool_result],
        "messages":     [{"role": "assistant", "content": f"[TOOL_RESULT] {json.dumps(tool_result)[:300]}"}],
        "_next_step":   "",   # clear so plan_router doesn't re-trigger tool_select
    }


# ── Respond node ──────────────────────────────────────────────────────────────

async def respond_node(state: ActionSessionState, llm: LLMClient) -> dict:
    """Compose the final spoken response."""
    final = state.get("final_response", "")
    if not final:
        # Fallback: ask LLM to summarise
        from server.prompts.action import RESPOND_SYSTEM, RESPOND_USER_TEMPLATE
        results_summary = json.dumps(state.get("tool_results", []), indent=2)
        response = await llm.complete(
            messages=[
                {"role": "system", "content": RESPOND_SYSTEM},
                {"role": "user",   "content": RESPOND_USER_TEMPLATE.format(
                    task_description=state.get("task_description", ""),
                    tool_results_summary=results_summary,
                )},
            ],
            temperature=0.4,
        )
        final = response.content.strip()

    logger.info("[RESPOND] final=%r", final[:80])
    return {"final_response": final, "done": True, "outcome": "completed"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_history(tool_results: list, messages: list) -> str:
    if not tool_results and not messages:
        return "No steps taken yet."
    parts = []
    for r in tool_results:
        if "error" in r:
            parts.append(f"TOOL ERROR: {r['error']}")
        else:
            parts.append(f"TOOL {r.get('tool')}: {json.dumps(r.get('result', {}))[:200]}")
    return "\n".join(parts) if parts else "No steps taken yet."
