"""
Block 10: Tool Registry + Selector test.

Step 1 — registers mock tools from fixtures/mock_tools.json
Step 2 — runs semantic queries and verifies top-2 results are relevant
Step 3 — tests the capability manifest builder

Usage:
    python tests/test_tool_selector.py
"""
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(message)s")

from server.tools.selector import select_tools
from server.tools.manifest import get_manifest


async def run_tests() -> None:
    print(f"\n{'='*60}")
    print("  Block 10: Tool Registry + Selector")
    print(f"{'='*60}")

    # ── Step 1: Register tools ────────────────────────────────────────────────
    print("\n[1] Registering mock tools…")
    import json
    from server.tools.registry import register_tools, ToolDefinition

    with open("tests/fixtures/mock_tools.json") as f:
        data = json.load(f)

    tools = [ToolDefinition(
        name=t["name"], description=t["description"],
        schema=t.get("schema", {}), call_type=t.get("call_type", "read"),
        domain=t.get("domain", ""),
    ) for t in data]

    await register_tools(tools)
    print(f"   ✓ {len(tools)} tools registered")

    results = []

    # ── Step 2: Semantic queries ──────────────────────────────────────────────
    queries = [
        ("book a table at a restaurant for tonight",  ["book_restaurant", "maps_search"]),
        ("send a message to my contact",              ["send_whatsapp", "contacts_lookup"]),
        ("play some jazz music",                      ["play_music"]),
        ("book a ride to the airport",                ["book_cab", "maps_search"]),
        ("what's the weather like tomorrow",          ["weather_fetch"]),
    ]

    print(f"\n[2] Semantic search queries\n")
    for query, expected_any in queries:
        top2 = await select_tools(query, top_k=2)
        names = [t.name for t in top2]
        sims  = [f"{t.similarity:.3f}" for t in top2]
        hit   = any(e in names for e in expected_any)

        print(f"   Query: {query!r}")
        print(f"   Top-2: {names}  sims={sims}")
        print(f"   {'✓' if hit else '✗'} expected one of {expected_any}\n")
        results.append((f"query: {query[:40]}", hit))

    # ── Step 3: Manifest ──────────────────────────────────────────────────────
    print(f"[3] Capability manifest\n")
    manifest = await get_manifest()
    has_tools = "book_restaurant" in manifest and "weather_fetch" in manifest
    print(manifest[:400] + ("…" if len(manifest) > 400 else ""))
    print(f"\n   {'✓' if has_tools else '✗'} manifest contains expected tools")
    results.append(("Capability manifest", has_tools))

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    passed = sum(1 for _, ok in results if ok)
    for name, ok in results:
        print(f"  {'✓' if ok else '✗'} {name}")
    print(f"\n  {passed}/{len(results)} passed")
    print(f"{'='*60}")

    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    asyncio.run(run_tests())
