#!/usr/bin/env python3
"""PreToolUse hook for Bash: allow / ask / deny each command the agent proposes.

1. Hard rules in code first, on each part of a chain (`cd <dir> && …`, `;`, `|`). Plainly destructive patterns are
   denied; a recursive delete of anything but build output and caches is never auto-allowed (ask, or deny if
   Decima says so); plainly read-only chains are allowed. No model call is needed to allow.
2. Everything else goes to Decima with a stated policy (allow / ask / deny). Leading `cd <dir>` steps become the cwd.
   - deny  when p(deny) ≥ 0.6
   - ask   when p(deny) ≥ 0.2, or when the top answer is below DECIMA_BASH_ALLOW_AT (default 0.85)
   - allow otherwise (DECIMA_BASH_AUTO_ALLOW=0 turns that into "no opinion", so Claude Code asks as usual)
3. Server down → ask. Never allow on failure.

DECIMA_BASH_LOCAL_OK=1 uses a policy where calls to services on this machine (localhost, 127.0.0.1) are allowed
instead of "network → ask". When a rule decides, Decima is still asked (DECIMA_CHECK_RULES=0 turns it off), and a
strong disagreement is logged to ~/.decima/disagreements.jsonl: a rule that keeps disagreeing is usually wrong.
"""

import json
import os
import re
import shlex
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from decima_hook import LOG, DecimaDown, answer, ask, passthrough, read_event  # noqa: E402

ALLOW_AT = float(os.environ.get("DECIMA_BASH_ALLOW_AT", "0.85"))
AUTO_ALLOW = os.environ.get("DECIMA_BASH_AUTO_ALLOW", "1") == "1"
CHECK_RULES = os.environ.get("DECIMA_CHECK_RULES", "1") == "1"
POLICY = {
    "allow": "Read-only, or builds, tests, lints or cleans generated files inside the project",
    "ask": "Changes project files, git state or installed packages in a way that can be undone, reads secrets like .env, or touches the network",
    "deny": "Destroys data or history that cannot be recovered, overwrites shared history, sends local files or secrets off the machine, needs root, or pipes a download into a shell",
}
if os.environ.get("DECIMA_BASH_LOCAL_OK") == "1":
    POLICY = {
        "allow": "Read-only, builds, tests, lints or cleans generated files inside the project, or calls services on this machine (localhost, 127.0.0.1)",
        "ask": "Changes project files, git state or installed packages in a way that can be undone, reads secrets like .env, or touches the network outside this machine",
        "deny": POLICY["deny"],
    }
# build output and caches: the only things a recursive delete may remove without a person looking
GENERATED = re.compile(r"^(\./)?(build|dist|out|target|node_modules(/\.cache)?|__pycache__|\.pytest_cache|\.mypy_cache|\.ruff_cache|"
                       r"coverage|htmlcov|\.next|\.turbo|\.tox|\.parcel-cache|\.gradle|\.cache|[\w.-]+\.egg-info)/?$")
DENY = [
    (r"\brm\s+(-[a-zA-Z]*r[a-zA-Z]*f|-[a-zA-Z]*f[a-zA-Z]*r)[a-zA-Z]*\s+(/|~|\$HOME|\.\.?/?|\.git)(\s|$)", "recursive delete of /, home, the repo or .git"),
    (r"\bgit\s+push\b.*(--force\b|-f\b|--force-with-lease).*\b(main|master|trunk|release)\b", "force-push to a shared branch"),
    (r"\b(curl|wget)\b[^|]*\|\s*(sudo\s+)?(ba|z|)sh\b", "piping a download into a shell"),
    (r"(^|[;&|]\s*)sudo\b", "needs root"),
    (r"\b(mkfs(\.\w+)?|dd\s+if=.*\bof=/dev/)", "writes a raw device"),
    (r"\b(psql|mysql|mariadb|sqlite3|duckdb|clickhouse-client|cockroach\s+sql|mongosh?|redis-cli)\b.*(\b(DROP\s+(TABLE|DATABASE|SCHEMA|COLLECTION)|TRUNCATE|DELETE\s+FROM|dropDatabase|deleteMany|FLUSHALL|FLUSHDB)\b|\.drop\(\))",
     "deletes or drops database data"),
    (r"\bgit\s+(reset\s+--hard|clean\s+-[a-zA-Z]*f)", "discards uncommitted work"),
    (r"(curl|wget)\b.*(-d|--data(-binary)?|-F|-T|--upload-file)\s+@?\S*(\.env|id_rsa|id_ed25519|\.aws|\.ssh)", "sends secrets off the machine"),
]
DB_WRITE = re.compile(r"\b(psql|mysql|mariadb|sqlite3|duckdb|clickhouse-client|mongosh?|redis-cli)\b.*\b(UPDATE\s+\w+\s+SET|INSERT\s+INTO|ALTER\s+TABLE|GRANT|REVOKE|updateMany|insertOne|SET\s+\w+)\b", re.I)
READ_ONLY = re.compile(r"^(ls|pwd|cat|head|tail|wc|grep|rg|find|tree|stat|file|which|echo|du|df|"
                       r"git\s+(status|log|diff|show|branch|blame|remote\s+-v|rev-parse|ls-files))\b")


def split_chain(cmd: str) -> list[str]:
    """Split on &&, ||, ;, | and newlines outside quotes."""
    parts, cur, q, i = [], [], None, 0
    while i < len(cmd):
        c = cmd[i]
        if q:
            cur.append(c)
            if c == "\\" and q == '"' and i + 1 < len(cmd):
                cur.append(cmd[i + 1]); i += 1
            elif c == q:
                q = None
        elif c in "'\"":
            q = c; cur.append(c)
        elif cmd.startswith(("&&", "||"), i):
            parts.append("".join(cur)); cur = []; i += 1
        elif c in ";|\n":
            parts.append("".join(cur)); cur = []
        else:
            cur.append(c)
        i += 1
    parts.append("".join(cur))
    return [p.strip() for p in parts if p.strip()]


def strip_cd(parts: list[str], cwd: str | None) -> tuple[list[str], str | None]:
    """Leading `cd <dir>` steps only set the working directory."""
    while parts and re.fullmatch(r"cd(\s+\S+)?", parts[0]):
        arg = parts.pop(0)[2:].strip().strip("'\"")
        cwd = arg if arg.startswith("/") or not cwd else f"{cwd.rstrip('/')}/{arg}"
    return parts, cwd


def deletes_precious(part: str) -> str | None:
    """A recursive rm (or rmtree) of anything that is not build output or a cache → that path."""
    m = re.match(r"rm\s+(.*)$", part)
    if m:
        try:
            args = shlex.split(m.group(1))
        except ValueError:
            return part
        flags = "".join(a.lstrip("-") for a in args if a.startswith("-") and a != "--")
        if "r" in flags.lower():
            for t in (a for a in args if not a.startswith("-")):
                if not GENERATED.match(t):
                    return t
    m = re.search(r"rmtree\(\s*\\?['\"]([^'\"\\]+)", part)
    if m and not GENERATED.match(m.group(1)):
        return m.group(1)
    return None


LOCAL_HOST = r"(localhost|127\.\d+\.\d+\.\d+|\[::1\]|0\.0\.0\.0)(:\d+)?"
LOCAL_GET = re.compile(rf"(curl|wget)\s+(-[\w-]+\s+)*['\"]?(https?://)?{LOCAL_HOST}(/[^\s'\"]*)?['\"]?(\s+-[\w-]+)*$")


def local_get(parts: list[str]) -> bool:
    """A plain GET to a service on this machine, possibly piped into read-only tools (jq, head, grep)."""
    if not parts or not LOCAL_GET.match(parts[0]) or re.search(r"\s-(X|d|F|T|o|O)\b|--data|--upload|--output|-X\s*(POST|PUT|DELETE)", parts[0]):
        return False
    return all(READ_ONLY.match(p) or re.match(r"(jq|sort|uniq)\b", p) for p in parts[1:])


def plainly_read_only(parts: list[str]) -> bool:
    if any(re.search(r"[><`]|\$\(|\b-delete\b|\b-exec\b|\.env\b|id_rsa|\.ssh/|\.aws/", p) for p in parts):
        return False
    return bool(parts) and all(READ_ONLY.match(p) for p in parts)


def decide(state: dict) -> dict:
    a = ask(state, {"gate": {"type": "choice", "instructions": "Should the agent's proposed command run?", "criteria": POLICY}})["gate"]
    return {"choice": a["choice"], "p": a["probabilities"]}


def by_rule(decision: str, why: str) -> None:
    """A rule decides. Decima is asked anyway and a strong disagreement is logged: it usually means a bad rule."""
    detail = {}
    if CHECK_RULES:
        try:
            m = decide(state)
            detail = {"model": m}
            p = m["p"]
            if (decision == "deny" and p["allow"] >= 0.75) or (decision == "allow" and p["deny"] >= 0.5) or (decision == "ask" and p["allow"] >= 0.9):
                with (LOG.parent / "disagreements.jsonl").open("a") as f:
                    f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "command": cmd, "rule": decision, "why": why,
                                        "model": m}, ensure_ascii=False) + "\n")
                detail["disagree"] = True
                print(f"[decima bash-gate] rule said {decision} ({why}); Decima says {m['choice']} "
                      f"({max(p.values()):.2f}). If this keeps happening, check the rule.", file=sys.stderr)
        except (DecimaDown, KeyError, TypeError, OSError):
            pass
    if decision == "allow" and not AUTO_ALLOW:
        passthrough(event, "bash-gate", why, detail)
    answer(event, "bash-gate", decision, why, detail)


event = read_event()
cmd = ((event.get("tool_input") or {}).get("command") or "").strip()
if not cmd:
    sys.exit(0)
parts, cwd = strip_cd(split_chain(cmd), event.get("cwd"))
body = cmd
while (m := re.match(r"cd(\s+(\"[^\"]*\"|'[^']*'|\S+))?\s*(&&|;)\s*", body)):     # the model sees the command without its cd prefix
    body = body[m.end():]
state = {"tool": "Bash", "command": body}
if (event.get("tool_input") or {}).get("description"):
    state["description"] = event["tool_input"]["description"]
if cwd:
    state["cwd"] = cwd
if not parts:
    by_rule("allow", "only changes directory")
for rx, why in DENY:
    if re.search(rx, cmd, re.I):
        by_rule("deny", f"blocked by rule: {why}.")
precious = next((t for t in map(deletes_precious, parts) if t), None)
if DB_WRITE.search(cmd):
    by_rule("ask", "writes to a database")
if plainly_read_only(parts):
    by_rule("allow", "read-only by rule")
if local_get(parts):
    by_rule("allow" if "127.0.0.1" in POLICY["allow"] else "ask", "a GET to a service on this machine")
try:
    m = decide(state)
    p = m["p"]
except (DecimaDown, KeyError, TypeError) as e:
    answer(event, "bash-gate", "ask", f"Decima unavailable ({e}).")
a = m
detail = {"p": p}
if precious and p["deny"] < 0.6:
    answer(event, "bash-gate", "ask", f"recursive delete of `{precious}`, which is not build output or a cache (deny {p['deny']:.2f}).", detail)
if p["deny"] >= 0.6:
    answer(event, "bash-gate", "deny", f"policy: {POLICY['deny'].lower()} (p={p['deny']:.2f}).", detail)
if p["deny"] >= 0.2 or a["choice"] == "ask" or max(p.values()) < ALLOW_AT:
    answer(event, "bash-gate", "ask", f"needs a look (allow {p['allow']:.2f} / ask {p['ask']:.2f} / deny {p['deny']:.2f}).", detail)
if AUTO_ALLOW:
    answer(event, "bash-gate", "allow", f"policy allows it (p={p['allow']:.2f}).", detail)
passthrough(event, "bash-gate", f"model would allow (p={p['allow']:.2f})", detail)
