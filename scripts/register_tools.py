"""
Register MCP tools into the database.

Reads a JSON file of tool definitions, embeds each description,
and upserts into mcp_tools table.

Usage:
    python scripts/register_tools.py --tools tests/fixtures/mock_tools.json
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


async def main(tools_path: Path) -> None:
    from server.tools.registry import register_tools, ToolDefinition

    if not tools_path.exists():
        print(f"ERROR: {tools_path} not found.")
        sys.exit(1)

    with open(tools_path) as f:
        data = json.load(f)

    tools = [
        ToolDefinition(
            name=t["name"],
            description=t["description"],
            schema=t.get("schema", {}),
            call_type=t.get("call_type", "read"),
            domain=t.get("domain", ""),
        )
        for t in data
    ]

    print(f"Registering {len(tools)} tool(s) from {tools_path.name}…\n")
    await register_tools(tools)

    print(f"\n✓ Done — {len(tools)} tool(s) registered in mcp_tools")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register MCP tools")
    parser.add_argument("--tools", required=True, help="Path to tools JSON file")
    args = parser.parse_args()
    asyncio.run(main(Path(args.tools)))
