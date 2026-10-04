"""Shared plumbing for the Decima Claude Code hooks: ask the local /v1/systemone server, answer the hook, log.

Environment:
  DECIMA_URL     default http://127.0.0.1:11436 (`python -m decima.serve` binds to 127.0.0.1 by default)
  DECIMA_MODEL   model name sent to the server (default: decima)
  DECIMA_SHADOW  1 = decide and log, but never enforce: the hook returns nothing and Claude Code asks as usual
  DECIMA_LOG     decision log (JSONL), default ~/.decima/decisions.jsonl — your own labels for a later fine-tune
  DECIMA_TIMEOUT seconds per call, default 3

Every hook fails closed: if the server is down or slow, the answer is "ask", never "allow".
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

URL = os.environ.get("DECIMA_URL", "http://127.0.0.1:11436").rstrip("/") + "/v1/systemone"
MODEL = os.environ.get("DECIMA_MODEL", "decima")
SHADOW = os.environ.get("DECIMA_SHADOW", "") == "1"
LOG = Path(os.environ.get("DECIMA_LOG", Path.home() / ".decima" / "decisions.jsonl"))
TIMEOUT = float(os.environ.get("DECIMA_TIMEOUT", "3"))


class DecimaDown(Exception):
    pass


def ask(state: dict, questions: dict) -> dict:
    """POST one System One request; returns the `answers` object or raises DecimaDown."""
    body = json.dumps({"model": MODEL, "state": state, "questions": questions}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return json.load(r)["answers"]
    except Exception as e:                                    # down, timeout, 422 (state too long), bad JSON
        raise DecimaDown(str(e)) from e


def log(event: dict, hook: str, decision: str, why: str, detail: dict | None = None) -> None:
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with LOG.open("a") as f:
            f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "hook": hook, "decision": decision, "why": why,
                                "shadow": SHADOW, "tool": event.get("tool_name"), "input": event.get("tool_input"),
                                "cwd": event.get("cwd"), "detail": detail or {}}, ensure_ascii=False) + "\n")
    except OSError:
        pass


def answer(event: dict, hook: str, decision: str, why: str, detail: dict | None = None) -> None:
    """Return a PreToolUse permission decision (allow / ask / deny) and exit. In shadow mode, only log."""
    log(event, hook, decision, why, detail)
    if not SHADOW:
        print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": decision,
                                                 "permissionDecisionReason": f"[decima {hook}] {why}"}}))
    sys.exit(0)


def passthrough(event: dict, hook: str, why: str, detail: dict | None = None) -> None:
    """No opinion: log it and let Claude Code's own permission rules decide (the gate never auto-approves on its own)."""
    log(event, hook, "pass", why, detail)
    sys.exit(0)


def read_event() -> dict:
    try:
        return json.load(sys.stdin)
    except ValueError:
        return {}
