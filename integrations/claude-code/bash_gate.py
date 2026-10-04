#!/usr/bin/env python3
"""PreToolUse hook for Bash: allow / ask / deny each command the agent proposes.

1. Hard rules in code first. Plainly destructive patterns are denied, and plainly read-only commands are allowed,
   without a model call.
2. Everything else goes to Decima with a stated policy (allow / ask / deny):
   - deny  when p(deny) ≥ 0.6
   - ask   when p(deny) ≥ 0.2, or when the top answer is below DECIMA_BASH_ALLOW_AT (default 0.85)
   - allow otherwise (DECIMA_BASH_AUTO_ALLOW=0 turns that into "no opinion", so Claude Code asks as usual)
3. Server down → ask. Never allow on failure.
"""

import os
import re
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from decima_hook import DecimaDown, answer, ask, passthrough, read_event  # noqa: E402

ALLOW_AT = float(os.environ.get("DECIMA_BASH_ALLOW_AT", "0.85"))
AUTO_ALLOW = os.environ.get("DECIMA_BASH_AUTO_ALLOW", "1") == "1"
POLICY = {
    "allow": "Read-only, or builds, tests, lints or cleans generated files inside the project",
    "ask": "Changes project files, git state or installed packages in a way that can be undone, reads secrets like .env, or touches the network",
    "deny": "Destroys data or history that cannot be recovered, overwrites shared history, sends local files or secrets off the machine, needs root, or pipes a download into a shell",
}
DENY = [
    (r"\brm\s+(-[a-zA-Z]*r[a-zA-Z]*f|-[a-zA-Z]*f[a-zA-Z]*r)[a-zA-Z]*\s+(/|~|\$HOME|\.\.?/?|\.git)(\s|$)", "recursive delete of /, home, the repo or .git"),
    (r"\bgit\s+push\b.*(--force\b|-f\b|--force-with-lease).*\b(main|master|trunk|release)\b", "force-push to a shared branch"),
    (r"\b(curl|wget)\b[^|]*\|\s*(sudo\s+)?(ba|z|)sh\b", "piping a download into a shell"),
    (r"(^|[;&|]\s*)sudo\b", "needs root"),
    (r"\b(mkfs(\.\w+)?|dd\s+if=.*\bof=/dev/)", "writes a raw device"),
    (r"\b(DROP\s+(TABLE|DATABASE|SCHEMA)|TRUNCATE\s+)\b", "drops or truncates data"),
    (r"\bgit\s+(reset\s+--hard|clean\s+-[a-zA-Z]*f)", "discards uncommitted work"),
    (r"(curl|wget)\b.*(-d|--data(-binary)?|-F|-T|--upload-file)\s+@?\S*(\.env|id_rsa|id_ed25519|\.aws|\.ssh)", "sends secrets off the machine"),
]
READ_ONLY = re.compile(r"^(ls|pwd|cat|head|tail|wc|grep|rg|find|tree|stat|file|which|echo|du|df|"
                       r"git\s+(status|log|diff|show|branch|blame|remote\s+-v|rev-parse|ls-files))\b")


def plainly_read_only(cmd: str) -> bool:
    if re.search(r"[><`]|\$\(|\b-delete\b|\b-exec\b|\.env\b|id_rsa|\.ssh/|\.aws/", cmd):
        return False
    parts = [p.strip() for p in re.split(r"\s*(?:\|\||&&|;|\|)\s*", cmd) if p.strip()]
    return bool(parts) and all(READ_ONLY.match(p) for p in parts)


event = read_event()
cmd = ((event.get("tool_input") or {}).get("command") or "").strip()
if not cmd:
    sys.exit(0)
for rx, why in DENY:
    if re.search(rx, cmd, re.I):
        answer(event, "bash-gate", "deny", f"blocked by rule: {why}.")
if plainly_read_only(cmd):
    answer(event, "bash-gate", "allow", "read-only by rule") if AUTO_ALLOW else passthrough(event, "bash-gate", "read-only by rule")

state = {"tool": "Bash", "command": cmd}
if (event.get("tool_input") or {}).get("description"):
    state["description"] = event["tool_input"]["description"]
if event.get("cwd"):
    state["cwd"] = event["cwd"]
try:
    a = ask(state, {"gate": {"type": "choice", "instructions": "Should the agent's proposed command run?", "criteria": POLICY}})["gate"]
    p = a["probabilities"]
except (DecimaDown, KeyError, TypeError) as e:
    answer(event, "bash-gate", "ask", f"Decima unavailable ({e}).")
detail = {"p": p}
if p["deny"] >= 0.6:
    answer(event, "bash-gate", "deny", f"policy: {POLICY['deny'].lower()} (p={p['deny']:.2f}).", detail)
if p["deny"] >= 0.2 or a["choice"] == "ask" or max(p.values()) < ALLOW_AT:
    answer(event, "bash-gate", "ask", f"needs a look (allow {p['allow']:.2f} / ask {p['ask']:.2f} / deny {p['deny']:.2f}).", detail)
if AUTO_ALLOW:
    answer(event, "bash-gate", "allow", f"policy allows it (p={p['allow']:.2f}).", detail)
passthrough(event, "bash-gate", f"model would allow (p={p['allow']:.2f})", detail)
