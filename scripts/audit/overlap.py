"""Exact-text overlap between Decima-small's training states and every evaluation suite's states.

    uv run python scripts/audit/overlap.py            # → runs/audit/overlap.json + a table

Training data = the union of every file in v1i's lineage (V0 → v1a → v1a2 → v1e → v1f → v1g → v1i):
teacher decisions (data/teacher/generate-*, label-*), Jev-style cases (data/jevgen-snap), gold NLI/BoolQ
(data/gold) and gold intent/topic/sentiment (data/gold-cls). Texts are compared after unwrapping JSON states (Laya's protocol), lower-casing and
collapsing whitespace; states shorter than 8 characters are ignored (stock phrases like "yes", "ok").
Reported per suite: share of eval states found verbatim in training, and which training source they
came from. For every suite with any overlap it also recomputes accuracy on the non-overlapping items
("clean") for every system with predictions in runs/preds/<set>--<system>.jsonl, to show what the
overlap is worth. docs/EVAL.md §2 quotes this table; it is the only overlap measure release documents use.
"""

from __future__ import annotations

import glob
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

TRAIN = {
    "teacher-invented": ["data/teacher/generate-*.jsonl"],
    "teacher-relabelled": ["data/teacher/label-*.jsonl"],
    "jev-style": ["data/jevgen-snap/*.jsonl"],
    "gold-nli": ["data/gold/*.jsonl"],
    "gold-cls": ["data/gold-cls/*.jsonl"],
}
SYSTEMS = ("v1i", "v0", "kev-0.5b", "kev-0.8b", "laya", "laya-multilingual")
EVAL = ["runs/items/kev.jsonl", "runs/items/laya.jsonl", "runs/items/decima.jsonl", "runs/items/jevtyped.jsonl",
        "runs/items/btzsc.jsonl"]
MIN_CHARS = 8
_ws = re.compile(r"\s+")


def unwrap(text: str) -> str:
    """Laya's protocol wraps states as JSON ({"utterance": ...}); compare the text inside."""
    t = (text or "").strip()
    if t.startswith("{"):
        try:
            d = json.loads(t)
            if isinstance(d, dict):
                return " ".join(str(v) for v in d.values())
        except ValueError:
            pass
    return t


def key(text: str) -> bytes | None:
    t = _ws.sub(" ", unwrap(text).lower())
    return hashlib.blake2b(t.encode(), digest_size=12).digest() if len(t) >= MIN_CHARS else None


def main() -> None:
    seen: dict[bytes, str] = {}
    rows = Counter()
    for source, pats in TRAIN.items():
        for f in sorted(p for pat in pats for p in glob.glob(pat)):
            with open(f) as fh:
                for line in fh:
                    r = json.loads(line)
                    k = key(r.get("state", ""))
                    rows[source] += 1
                    if k is not None:
                        seen.setdefault(k, source)
    print("training rows:", dict(rows), "· distinct states:", len(seen))

    out = {}
    for f in EVAL:
        per = defaultdict(lambda: [0, 0, Counter(), [], set()])   # n, hits, sources, items, overlapping ids
        for line in open(f):
            r = json.loads(line)
            if r["split"] != "eval" or "group" in r:
                continue
            k = key(r["state"])
            s = per[r["suite"]]
            s[0] += 1
            s[3].append((r["id"], r["gold"]))
            if k is not None and k in seen:
                s[1] += 1; s[2][seen[k]] += 1; s[4].add(r["id"])
        set_name = Path(f).stem
        preds = {}
        for sysname in SYSTEMS:
            pf = Path(f"runs/preds/{set_name}--{sysname}.jsonl")
            if any(v[1] for v in per.values()) and pf.exists():
                preds[sysname] = {p["id"]: p["probs"] for p in map(json.loads, open(pf))}
        for suite, (n, hit, src, its, bad) in per.items():
            out[suite] = {"n": n, "overlap": hit / n if n else 0.0, "sources": dict(src)}
            if hit:
                clean = {}
                for sysname, P in preds.items():
                    if not all(i in P for i, _ in its):
                        continue
                    right = {i: max(range(len(P[i])), key=P[i].__getitem__) == g for i, g in its}
                    clean[sysname] = {"all": sum(right.values()) / len(its),
                                      "clean": sum(v for i, v in right.items() if i not in bad) / (len(its) - len(bad))}
                out[suite]["accuracy_all_vs_clean"] = clean
    Path("runs/audit").mkdir(parents=True, exist_ok=True)
    Path("runs/audit/overlap.json").write_text(json.dumps({"min_chars": MIN_CHARS, "train_rows": dict(rows), "suites": out}, indent=2))
    for suite, v in sorted(out.items(), key=lambda kv: -kv[1]["overlap"]):
        if v["overlap"] > 0:
            d = v.get("accuracy_all_vs_clean", {}).get("v1i")
            dd = f"  v1i {d['all']:.3f} → {d['clean']:.3f}" if d else ""
            print(f"  {suite:34s} {v['overlap']:6.1%}  of {v['n']:5d}  {v['sources']}{dd}")
    print(f"{sum(v['overlap'] == 0 for v in out.values())} of {len(out)} suites have 0 % exact overlap → runs/audit/overlap.json")


if __name__ == "__main__":
    main()
