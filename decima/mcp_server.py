"""MCP server over stdio (standard library only): Decima's decisions as tools for any MCP client.

    decima mcp                                   # each preset uses its recommended model
    decima mcp --model ./export/agent3-int8      # one model for every tool

Tools: `decide` (state + TypeSafe questions → /v1/systemone answers) and one tool per preset (`triage`, `guardrails`,
`agent_gate`, …) that takes `text` (plain text, or a JSON object for presets with structured states).
Protocol: newline-delimited JSON-RPC 2.0 (initialize, tools/list, tools/call, ping). Logs go to stderr only.
"""

from __future__ import annotations

import argparse
import json
import sys

from .router import Router, list_presets, load_preset

PROTOCOL = "2025-06-18"
DECIDE = {
    "name": "decide",
    "description": "Calibrated decision about a state (text or JSON) on CPU. `questions` uses TypeSafe's System One schema: "
                   "{name: {type: 'choice'|'noul'|'score', instructions, criteria}}. choice criteria = {label: description}; "
                   "score criteria = [level 0, level 1, …]; noul = a statement answered with P(true). Returns {answers: {name: …}}.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "state": {"description": "the situation to decide about: a string or a JSON object", "type": ["string", "object"]},
            "questions": {"type": "object", "description": "TypeSafe-style questions keyed by name"},
            "preset": {"type": "string", "description": "optional preset whose questions are asked too"},
        },
        "required": ["state"],
    },
}


def tool_name(preset: str) -> str:
    return preset.replace("-", "_")


def tools() -> list[dict]:
    out = [DECIDE]
    for name, desc in list_presets().items():
        p = load_preset(name)
        qs = ", ".join(f"{q} ({s['type']})" for q, s in p["questions"].items())
        out.append({"name": tool_name(name), "description": f"{desc}. Answers: {qs}.",
                    "inputSchema": {"type": "object", "properties": {"text": {"type": "string", "description":
                                    "the text to judge" + ("; or a JSON object with the keys its questions need" if "state" in p else "")}},
                                    "required": ["text"]}})
    return out


def call(router: Router, name: str, args: dict) -> dict:
    if name == "decide":
        if not args.get("questions") and not args.get("preset"):
            raise ValueError("give `questions`, `preset`, or both")
        return router.predict(args["state"], args.get("questions"), preset=args.get("preset"))
    presets = {tool_name(p): p for p in list_presets()}
    if name not in presets:
        raise ValueError(f"unknown tool {name!r}")
    return router.decide_text(str(args.get("text", "")), preset=presets[name])


def handle(router: Router, msg: dict) -> dict | None:
    mid, method, params = msg.get("id"), msg.get("method"), msg.get("params") or {}
    if mid is None:                                   # notifications (initialized, cancelled) need no answer
        return None
    try:
        if method == "initialize":
            result = {"protocolVersion": params.get("protocolVersion", PROTOCOL), "capabilities": {"tools": {}},
                      "serverInfo": {"name": "decima", "version": "0.1"}}
        elif method == "ping":
            result = {}
        elif method == "tools/list":
            result = {"tools": tools()}
        elif method == "tools/call":
            try:
                r = call(router, params.get("name", ""), params.get("arguments") or {})
                result = {"content": [{"type": "text", "text": json.dumps(r, ensure_ascii=False)}], "structuredContent": r, "isError": False}
            except (ValueError, KeyError, TypeError) as e:
                result = {"content": [{"type": "text", "text": f"error: {e}"}], "isError": True}
        else:
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"method not found: {method}"}}
    except Exception as e:                            # never kill the session on one bad call
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32603, "message": str(e)}}
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def serve(router: Router, stdin=None, stdout=None) -> None:
    stdin, stdout = stdin or sys.stdin, stdout or sys.stdout
    for line in stdin:
        if not line.strip():
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            out = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}
        else:
            out = handle(router, msg) if isinstance(msg, dict) else {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "invalid request"}}
        if out is not None:
            stdout.write(json.dumps(out, ensure_ascii=False) + "\n")
            stdout.flush()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="decima mcp", description="Decima as an MCP server over stdio")
    ap.add_argument("--model", help="one model for every tool (default: each preset's recommended model)")
    ap.add_argument("--precision", default="int8", choices=["int8", "fp32"])
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args(argv)
    print("decima MCP server on stdio", file=sys.stderr, flush=True)
    serve(Router(model=a.model, precision=a.precision, threads=a.threads))
    return 0


if __name__ == "__main__":
    sys.exit(main())
