#!/usr/bin/env python3
"""
extract_claims_batch.py

Input JSON (dict):
{
  "q001": "long text ...",
  "q002": "long text ...",
  ...
}

Output JSON (dict):
{
  "q001": { "claims": [ ... ] },
  "q002": { "claims": [ ... ] },
  ...
}

This script:
- Launches a Claimify/ClaimsMCP server over stdio (python claimify_server.py)
- Connects via MCP ClientSession
- Calls tool `extract_claims` with arg `text_to_process`
- Parses result.content[0].text as JSON list of claims
- Writes the ideal output format
"""

import argparse
import asyncio
import json
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def _result_to_dict(result: Any) -> Dict[str, Any]:
    """Normalize MCP CallToolResult to a plain dict."""
    if hasattr(result, "model_dump"):
        return result.model_dump()
    if isinstance(result, dict):
        return result
    # Best-effort fallback
    return {"raw": result}


def extract_claims_from_result(result_dict: Dict[str, Any]) -> List[str]:
    """
    ClaimsMCP extract_claims result -> List[str]

    Expected structure (based on your observed output):
      result = {
        "isError": false,
        "content": [
          { "type": "text", "text": "[ ...json list string... ]", ... }
        ],
        ...
      }
    """
    if result_dict.get("isError"):
        # Some servers include error text in content; include it in message if present.
        content = result_dict.get("content") or []
        msg = None
        if content and isinstance(content, list):
            msg = content[0].get("text")
        raise RuntimeError(f"Tool execution failed. {msg or ''}".strip())

    content = result_dict.get("content") or []
    if not isinstance(content, list) or len(content) == 0:
        return []

    first = content[0] or {}
    text_block = first.get("text")
    if not text_block or not isinstance(text_block, str):
        return []

    # The server returns JSON array as a *string*
    try:
        claims = json.loads(text_block)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse claims JSON from result.content[0].text: {e}")

    if not isinstance(claims, list):
        raise ValueError("Parsed claims is not a list")

    # Ensure all claims are strings
    cleaned: List[str] = []
    for c in claims:
        if isinstance(c, str):
            cleaned.append(c)
        else:
            cleaned.append(str(c))
    return cleaned


async def run(
    server_script_path: str,
    input_json_path: str,
    output_json_path: str,
    *,
    tool_name: str = "extract_claims",
    python_cmd: str = "python",
) -> None:
    # Load input first (fail fast)
    items = json.load(open(input_json_path, "r", encoding="utf-8"))
    if not isinstance(items, dict):
        raise ValueError(
            "Input JSON must be a dict like {\"q001\": \"...\", \"q002\": \"...\"}"
        )

    exit_stack = AsyncExitStack()
    try:
        # Start MCP server over stdio: python claimify_server.py
        server_params = StdioServerParameters(
            command=python_cmd,
            args=[server_script_path],
            env=None,
        )

        stdio_transport = await exit_stack.enter_async_context(stdio_client(server_params))
        stdio, write = stdio_transport

        session = await exit_stack.enter_async_context(ClientSession(stdio, write))
        await session.initialize()

        # (Optional) verify tool exists
        tools_resp = await session.list_tools()
        available = {t.name for t in tools_resp.tools}
        if tool_name not in available:
            raise ValueError(
                f"Tool '{tool_name}' not found. Available tools: {sorted(available)}"
            )

        out: Dict[str, Dict[str, Any]] = {}

        total = len(items)

        for i, (qid, text) in enumerate(items.items(), start=1):
            if not isinstance(text, str):
                raise ValueError(f"{qid} is not a string")

            tool_args = {"text_to_process": text}
            result = await session.call_tool(tool_name, tool_args)
            result_dict = _result_to_dict(result)

            claims = extract_claims_from_result(result_dict)

            out[qid] = {"claims": claims}

            # ★ ここで毎回書き出す ★
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)

    finally:
        await exit_stack.aclose()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--server", required=True, help="Path to claimify_server.py")
    p.add_argument("--in", dest="inp", required=True, help="Input JSON path (dict)")
    p.add_argument("--out", dest="out", required=True, help="Output JSON path")
    p.add_argument("--tool", default="extract_claims", help="Tool name (default: extract_claims)")
    p.add_argument("--python", dest="python_cmd", default="python", help="Python command (default: python)")
    args = p.parse_args()

    asyncio.run(
        run(
            server_script_path=args.server,
            input_json_path=args.inp,
            output_json_path=args.out,
            tool_name=args.tool,
            python_cmd=args.python_cmd,
        )
    )


if __name__ == "__main__":
    main()
