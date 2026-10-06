"""Keep the hand-written agent test sets out of training (added 2026-10-06).

On 2026-10-06 we found near-copies of hand-written test cases in training data: the read-only templates of
teacher/agent_proc.py mirrored all 8 ab_read_only cases, and LLM-written bash rows contained ab_bash_gate commands
verbatim. This module flags any row whose *judged* command (or goal) matches a test case exactly or nearly
(token-set Jaccard ≥ 0.8 on 4+ tokens, after masking numbers; same-family commands with other arguments are kept — the
fresh test set, whose command families appear in no generator, is the honest measurement), so mixes and public datasets can drop it.

    python -m teacher.decontam_agent data/agent/*.jsonl            # report
    from teacher.decontam_agent import Decontam; d = Decontam(); d.hit(row) → the matching test string or None
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

GOLD_FILES = ["release/agent/agentbench-gold.toml", "release/agent/agentbench-gold-dynamic.jsonl", "release/agent/agentbench-fresh.toml"]
GENERIC = {"go test ./...", "npm test", "pytest -q", "cargo test", "git status", "git diff", "make"}   # everyone's commands, not test cases


def _norm(s: str) -> str:
    s = s.replace('\\"', '"')
    s = re.sub(r"\d+", "0", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def _tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z_./~$*-]+|\S", _norm(s)))


def _judged(state: str) -> list[str]:
    """The command / goal / proposed step a row asks about (not the surrounding context)."""
    try:
        d = json.loads(state)
    except (ValueError, TypeError):
        return [state] if isinstance(state, str) and len(state) < 400 else []
    if not isinstance(d, dict):
        return []
    out = [d[k] for k in ("command", "goal", "proposed", "task") if isinstance(d.get(k), str)]
    inp = d.get("input")
    if isinstance(inp, dict) and isinstance(inp.get("command"), str):
        out.append(inp["command"])
    return out


class Decontam:
    def __init__(self, root: str | Path = "."):
        import tomllib
        sys.path.insert(0, str(Path(root).resolve()))
        from bench.agent_eval import decode
        texts = []
        for f in GOLD_FILES:
            p = Path(root) / f
            if not p.exists():
                continue
            if p.suffix == ".toml":
                for t in tomllib.loads(decode(p.read_text()))["task"]:
                    texts += [c["state"] for c in t["cases"]]
            else:
                texts += [json.loads(l)["state"] for l in p.open() if l.strip()]
        self.gold = []
        for s in texts:
            for j in _judged(s):
                if len(j) > 10 and _norm(j) not in GENERIC:
                    self.gold.append((_norm(j), _tokens(j), j))

    def hit(self, row: dict) -> str | None:
        for j in _judged(row.get("state", "")):
            n, t = _norm(j), _tokens(j)
            if len(n) <= 10 or n in GENERIC:
                continue
            for gn, gt, raw in self.gold:
                if n == gn or (len(gn) > 18 and gn in n) or (min(len(t), len(gt)) >= 4 and len(t & gt) / len(t | gt) >= 0.8):
                    return raw
        return None


if __name__ == "__main__":
    d = Decontam()
    print(len(d.gold), "test strings")
    for p in sys.argv[1:]:
        n = hits = 0
        for l in open(p):
            if '"empty"' in l:
                continue
            n += 1
            hits += d.hit(json.loads(l)) is not None
        print(f"{p}: {hits}/{n} rows match a test case")
