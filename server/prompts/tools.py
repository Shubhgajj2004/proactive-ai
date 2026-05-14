"""
Tool-layer prompts — used by tools/manifest.py and tools/selector.py.

The capability manifest is a single-line summary per tool injected into
the ambient processor and action agent system prompts.
"""

CAPABILITY_MANIFEST_HEADER = (
    "The following tools are available to you. "
    "Use their names exactly as listed when referencing them in your response.\n"
)

# Shown when the tool registry is empty (dev/staging only)
NO_TOOLS_AVAILABLE = (
    "No tools are currently registered. "
    "You can still analyse the conversation and extract memories, "
    "but set should_act=false and confidence=0.0."
)
