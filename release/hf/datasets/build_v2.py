"""Assemble the Decima 2 public datasets from the generators' output (all git-ignored inputs).

    uv run python release/hf/datasets/build_v2.py --out ~/Rande/decima-work/ds [--only agent tasks games persian]

agent   → decima-agent-decisions   teacher/agent_proc.py, agent_tools.py (exact labels) and agent_llm.py (Qwen3-Coder
                                   writes, code fills the label or Gemma 4 agrees blind); plus the 130 hand-written
                                   agentbench cases as a test-only config
tasks   → decima-system-one-tasks  teacher/generate_s1.py, generate_s2.py in the raw Jev/TypeSafe schema, decontaminated
                                   and split (5 % of tasks held out) exactly as teacher/render_s1.py did for training
games   → decima-game-decisions    teacher/procedural.py, paraphrase_proc.py, snake_proc.py
persian → jabr-v2-persian          release/jabr-fa/v2-fa*.toml as rows (CC0, like the original)

Every row is machine-generated or written by us; no third-party text is exported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tomllib
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
A = ROOT / "data/agent"

AGENT = {  # config → [(file stem, tasks or None for all)]
    "secret": [("proc", {"secret"}), ("proc-secret2", None), ("llm-files", None)],
    "bash": [("proc", {"bash"}), ("llm-bash", None)],
    "readonly": [("proc-readonly", None)],
    "tool": [("tools", {"tool"}), ("llm-goals", {"tool-para"})],
    "command": [("tools", {"command"}), ("llm-goals", {"command-para"}), ("llm-commands", None)],
    "next_step": [("tools", {"step"}), ("proc", {"action"}), ("llm-action", None)],
    "router": [("llm-router", None)],
    "outcome": [("proc", {"output", "ops"})],
}
ORIGIN = {"agent-proc": "procedural", "agent-llm-files": "llm-file+code-slot", "agent-llm-goals": "llm-paraphrase",
          "agent-llm-commands": "llm+blind-check", "agent-llm-bash": "llm+blind-check", "agent-llm-action": "llm+blind-check",
          "agent-llm-router": "llm+blind-check"}


def jl(path):
    with open(path) as f:
        for l in f:
            try:
                r = json.loads(l)
            except ValueError:
                continue
            if not r.get("empty") and "state" in r:
                yield r


FREE_MAIL = re.compile(r"@(gmail|yahoo|hotmail|outlook|protonmail|icloud|mail)\.(com|ru)\b")


def scrub(v):
    """Invented addresses at real free-mail providers → reserved .example domains (gmail.com → gmail.example)."""
    if isinstance(v, str):
        return FREE_MAIL.sub(lambda m: f"@{m.group(1)}.example", v)
    if isinstance(v, list):
        return [scrub(x) for x in v]
    return v


def write(rows, path, schema=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{k: scrub(v) for k, v in r.items()} for r in rows]
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path, compression="zstd")
    print(f"  {path.relative_to(path.parents[2])}: {len(rows):,}")


def dec_row(r, config, origin):
    return {"id": r["id"], "config": config, "task": r.get("task"), "origin": origin, "kind": r["kind"], "state": r["state"],
            "question": r["question"], "choices": list(r["choices"]), "gold": int(r["gold"]),
            "probs": [float(p) for p in r["probs"]]}


def build_agent(out):
    o = out / "decima-agent-decisions"
    stats = {}
    from teacher.decontam_agent import Decontam
    dc = Decontam(ROOT)                                       # rows that copy a hand-written test case stay out
    dropped = Counter()
    for cfg, parts in AGENT.items():
        split_rows = {}
        for split in ("train", "test"):
            rows, seen = [], set()
            for stem, tasks in parts:
                p = A / f"{stem}-{split}.jsonl"
                if not p.exists():
                    continue
                for r in jl(p):
                    if tasks and r.get("task") not in tasks:
                        continue
                    if split == "train" and dc.hit(r):
                        dropped[cfg] += 1
                        continue
                    k = (r["state"], r["question"], tuple(r["choices"]))
                    if r["id"] in seen or k in seen:
                        continue
                    seen.update((r["id"], k))
                    rows.append(dec_row(r, cfg, ORIGIN.get(r.get("source"), "procedural")))
            split_rows[split] = rows
        test_states = {r["state"] for r in split_rows["test"]}
        before = len(split_rows["train"])
        split_rows["train"] = [r for r in split_rows["train"] if r["state"] not in test_states]
        print(f"[agent/{cfg}] dropped {before - len(split_rows['train'])} train rows whose state is in test")
        for split, rows in split_rows.items():
            write(rows, o / cfg / f"{split}.parquet")
        stats[cfg] = {s: len(v) for s, v in split_rows.items()} | {
            "origin": dict(Counter(r["origin"] for r in split_rows["train"] + split_rows["test"]))}

    print("[agent] train rows dropped as copies of a test case:", dict(dropped))
    from bench.agent_eval import GOLD, decode  # hand-written sets, decoded to plain text
    fresh = tomllib.loads(decode((ROOT / "release/agent/agentbench-fresh.toml").read_text()))
    frows = []
    for t in fresh["task"]:
        crit = t["question"].get("criteria")
        keys = ["yes", "no"] if t["type"] == "noul" else list(crit)
        for i, c in enumerate(t["cases"]):
            exp = c["expected"]
            exp = "yes" if exp is True else "no" if exp is False else exp
            frows.append({"id": f"{t['id']}-{i}", "task": t["id"], "type": t["type"], "state": c["state"],
                          "instructions": t["question"]["instructions"], "criteria": json.dumps(crit) if crit else None,
                          "choices": keys, "expected": exp, "gold": keys.index(exp)})
    write(frows, o / "agentbench_fresh" / "test.parquet")
    stats["agentbench_fresh"] = {"test": len(frows)}
    data = tomllib.loads(decode(GOLD.read_text()))
    rows = []
    for t in data["task"]:
        crit = t["question"].get("criteria")
        keys = ["yes", "no"] if t["type"] == "noul" else list(crit)
        for i, c in enumerate(t["cases"]):
            exp = c["expected"]
            exp = "yes" if exp is True else "no" if exp is False else exp
            rows.append({"id": f"{t['id']}-{i}", "task": t["id"], "type": t["type"], "state": c["state"],
                         "instructions": t["question"]["instructions"], "criteria": json.dumps(crit) if crit else None,
                         "choices": keys, "expected": exp, "gold": keys.index(exp)})
    for r in jl(ROOT / "release/agent/agentbench-gold-dynamic.jsonl"):
        rows.append({"id": r["id"], "task": r["task"], "type": "choice", "state": r["state"], "instructions": r["question"],
                     "criteria": None, "choices": r["choices"], "expected": r["choices"][r["gold"]], "gold": r["gold"]})
    write(rows, o / "agentbench" / "test.parquet")
    stats["agentbench"] = {"test": len(rows), "tasks": dict(Counter(r["task"] for r in rows))}
    return stats


def build_tasks(out):
    o = out / "decima-system-one-tasks"
    from teacher.gold import Contam
    from teacher.render_s1 import _h
    contam = [Contam(ROOT / "runs/items"), Contam(ROOT / "data/decontam")]  # the same rule the training mix used
    raw = [("s1", "data/teacher/s1-rande-fast-local.jsonl"), ("s2", "data/teacher/s2-rande-fast-local.jsonl"),
           ("s2", "data/teacher/s2q-rande-strong-local.jsonl")]
    writer = {"rande-fast-local": "gemma-4-26b-a4b-it", "rande-strong-local": "qwen3-coder-next"}
    stats = {}
    by = defaultdict(list)
    dropped = Counter()
    for gen, p in raw:
        for r in jl(ROOT / p):
            if r.get("source") not in ("s1gen", "s2gen"):
                continue
            if any(c.hit([r["state"], r["instructions"]]) for c in contam):
                dropped[gen] += 1
                continue
            crit = r.get("criteria")
            split = "test" if (_h(r["group"]) % 10_000) / 10_000 < 0.05 else "train"  # held-out tasks, as in training
            by[gen, split].append({
                "id": r["id"], "generation": gen, "task_group": r["group"], "state_group": r.get("sgroup"),
                "question_id": r.get("qid"), "type": r["type"], "instructions": r["instructions"],
                "criteria": json.dumps(crit, ensure_ascii=False) if crit else None,  # score: the ordered levels
                "state": r["state"],
                "p_true": r.get("p_true"), "probs": json.dumps(r["probs"], ensure_ascii=False) if r.get("probs") is not None else None,
                "domain": r.get("domain") or r.get("topic"), "cluster": r.get("cluster"), "long_state": bool(r.get("long")),
                "state_lang": r.get("state_lang"), "question_lang": r.get("q_lang"),
                "writer": writer.get(r.get("model"), r.get("model")), "labeler": "gemma-4-26b-a4b-it"
                if r.get("model") == "rande-fast-local" else writer.get(r.get("model"), r.get("model"))})
    print("[tasks] dropped rows matching an evaluation item:", dict(dropped))
    for (gen, split), rows in by.items():
        write(rows, o / gen / f"{split}.parquet")
        stats[f"{gen}/{split}"] = {"rows": len(rows), "tasks": len({r["task_group"] for r in rows}), "types": dict(Counter(r["type"] for r in rows)),
                      "langs": dict(Counter(r["state_lang"] for r in rows).most_common()),
                      "writer": dict(Counter(r["writer"] for r in rows))}
    return stats


def build_games(out):
    o = out / "decima-game-decisions"
    files = {"grid": "proc-grid", "side_scroller": "proc-side", "reworded": "proc-para", "snake": "snake-room"}
    stats = {}
    for cfg, stem in files.items():
        rows = []
        for r in jl(ROOT / f"data/proc/{stem}.jsonl"):
            d = dec_row(r, cfg, "llm-reworded" if cfg == "reworded" else "procedural")
            d["context"] = r.get("context")
            rows.append(d)
        # hold out 3 % by state hash; the games used for evaluation (release/games) never appear here
        test = [r for r in rows if int(hashlib.sha1(r["state"].encode()).hexdigest(), 16) % 100 < 3]
        tids = {id(r) for r in test}
        write([r for r in rows if id(r) not in tids], o / cfg / "train.parquet")
        write(test, o / cfg / "test.parquet")
        stats[cfg] = {"train": len(rows) - len(test), "test": len(test), "kinds": dict(Counter(r["kind"] for r in rows))}
    return stats


def build_persian(out):
    o = out / "jabr-v2-persian"
    stats = {}
    for cfg, f in (("fa", "v2-fa.toml"), ("fa_state", "v2-fa-state.toml")):
        data = tomllib.loads((ROOT / "release/jabr-fa" / f).read_text())
        rows = []
        for t in data["task"]:
            q = t["question"]
            for i, c in enumerate(t["cases"]):
                exp = c["expected"]
                rows.append({"id": f"{t['id']}-{i}", "task": t["id"], "type": t["type"], "instructions": q.get("instructions"),
                             "criteria": json.dumps(q["criteria"], ensure_ascii=False) if q.get("criteria") else None,
                             "state": c["state"], "expected": json.dumps(exp, ensure_ascii=False) if not isinstance(exp, str) else exp})
        write(rows, o / cfg / "test.parquet")
        (o / "toml").mkdir(parents=True, exist_ok=True)
        (o / "toml" / f).write_text((ROOT / "release/jabr-fa" / f).read_text())
        stats[cfg] = {"tasks": len(data["task"]), "cases": len(rows)}
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", default=["agent", "tasks", "games", "persian"])
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    stats = {}
    for name in a.only:
        print(f"== {name}")
        stats[name] = globals()[f"build_{name}"](out)
    (out / "stats.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False))
    print(json.dumps(stats, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
