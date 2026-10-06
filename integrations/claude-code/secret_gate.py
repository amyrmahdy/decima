#!/usr/bin/env python3
"""PreToolUse hook for Write|Edit|MultiEdit: block real credentials before they reach the disk.

1. Rules: known key formats are denied without asking the model.
2. Decima reads the new text in overlapping chunks and answers one noul question per chunk:
   deny at p ≥ 0.8, ask at p ≥ 0.3 (lopsided on purpose: a missed secret costs more than one prompt).
3. Server down → ask. The deny reason goes back to Claude, so it knows to read the value from the environment.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from decima_hook import DecimaDown, answer, ask, passthrough, read_event  # noqa: E402

DENY_AT, ASK_AT = 0.8, 0.3
CHUNK, OVERLAP = 1500, 200
RULES = [r"sk-(proj-)?[A-Za-z0-9_-]{32,}", r"sk_live_[A-Za-z0-9]{20,}", r"AKIA[0-9A-Z]{16}", r"ghp_[A-Za-z0-9]{36}",
         r"github_pat_[A-Za-z0-9_]{50,}", r"xox[baprs]-[0-9A-Za-z-]{20,}", r"AIza[0-9A-Za-z_-]{35}", r"SG\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}",
         r"-----BEGIN [A-Z ]*PRIVATE KEY-----", r"https://hooks\.slack\.com/services/T[A-Z0-9]+/B[A-Z0-9]+/[A-Za-z0-9]+",
         r"\bkey-[0-9a-f]{32}\b", r"\b\d{8,10}:AA[A-Za-z0-9_-]{33}\b", r"https://(ptb\.|canary\.)?discord(app)?\.com/api/webhooks/\d+/[A-Za-z0-9_-]{60,}",
         r"glpat-[A-Za-z0-9_-]{20}", r"\bnpm_[A-Za-z0-9]{36}\b", r"\bhf_[A-Za-z0-9]{34}\b", r"sk-ant-api03-[A-Za-z0-9_-]{80,}",
         r"\b(mongodb(\+srv)?|postgres(ql)?|mysql|redis|amqp)://[^:/\s\"']+:(?!(?:[Pp]ass(word)?|PASSWORD|[Cc]hangeme|[Ss]ecret|[Ee]xample|x+|X+|\*+|<[^>]*>|\$\{?\w+\}?)@)[^@/\s\"']{4,}@"]
EXAMPLES = ("AKIAIOSFODNN7EXAMPLE", "EXAMPLEKEY", "sk_test_")
QUESTION = {"type": "noul", "instructions": "Does `new_text` contain a real password, API key, token or connection string with credentials, not a placeholder or env lookup?"}

event = read_event()
inp = event.get("tool_input", {}) or {}
path = inp.get("file_path", "")
text = inp.get("content") or inp.get("new_string") or "\n".join(e.get("new_string", "") for e in inp.get("edits", []) or [])
if not text.strip():
    sys.exit(0)

for rx in RULES:
    m = re.search(rx, text)
    if m and not any(x in m.group(0) for x in EXAMPLES):
        answer(event, "secret-gate", "deny", f"blocked by rule {rx!r} in {path}. Read the value from an environment variable or secret store instead.")

chunks = [text[i:i + CHUNK] for i in range(0, max(len(text) - OVERLAP, 1), CHUNK - OVERLAP)] or [text]
try:
    p = max(ask({"file": path, "new_text": c}, {"leak": QUESTION})["leak"]["noul"] for c in chunks)
except (DecimaDown, KeyError, TypeError) as e:
    answer(event, "secret-gate", "ask", f"Decima unavailable ({e}); please check {path} for credentials yourself.")
if p >= DENY_AT:
    answer(event, "secret-gate", "deny", f"{path} looks like it contains a real credential (p={p:.2f}). Use an env variable or secret store.", {"p": p})
if p >= ASK_AT:
    answer(event, "secret-gate", "ask", f"{path} may contain a credential (p={p:.2f}).", {"p": p})
passthrough(event, "secret-gate", f"no credential found (p={p:.2f})", {"p": p})
