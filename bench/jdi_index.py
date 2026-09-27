"""Jev Decision Index 0.1 (JDI) scorer — the kit's index maths over our items/preds format.

    uv run python -m bench.jdi_index --items runs/items/jdi.jsonl --preds decima=runs/p/jdi-decima.jsonl \
        [random=... ...] [--unsupported decima=ids.txt] [--out runs/jdi-decima.json]

    # Decima's no-truncation rule (its layout: state alone ≤ 256 tokens, each "question choice" ≤ 48):
    uv run python -m bench.jdi_index --items ... --preds decima=... \
        --budget-tokens 256 --choice-budget-tokens 48 --tokenizer checkpoints/v0/best/encoder \
        --budget-systems decima --write-unsupported runs/jdi-decima-unsupported.txt

Items come from ``python -m bench.items --suites 'jdi/*' --no-flips --calib 0 --limit 1000000000``
(the --limit matters: bench.items defaults to 1000 items per suite for loaders without their own
default, which would silently cut the suite). Preds: {"id", "probs"} aligned with the item's choices;
for "rank" items (ToolRet, BRIGHT) probs are independent P(yes) per candidate.

What is computed, per system, mirrors ``decision_index/scoring/index.py::static_score`` and
``recompute`` at the pinned kit commit (``JDI_KIT_COMMIT``):

  * The engine's choice is argmax(probs) (first on ties). A pred is valid when it has one finite,
    non-negative number per choice (choice items: sum > 0, renormalised; rank items: each in [0, 1]).
  * Unanswered = wrong against the full denominator. A case group (meta.group within a benchmark:
    ToolRet/BRIGHT chunks, RouterBench 0/5-shot pairs, ACOS groups, …) counts only when every item
    in it has a valid pred and none is unsupported; otherwise every item in the group scores 0
    (conservative F1: pred = missing). Same as the kit's per-group ``good`` flag.
  * Panel metric per benchmark (19): BFCL case exact over the group; ToolRet/BRIGHT nDCG@10 with the
    kit's (−p, doc id) ordering, and the queries our loader dropped (no judged-relevant candidate in
    the pool, nDCG 0 for everyone) put back into the denominator (``RETRIEVAL_QUERIES``); RouterBench
    realised quality of the pick for question "quality", per track, tracks averaged; ContractNLI /
    ESCI / VAST conservative macro-F1 2TP/(2TP+FP+FN+missing) over the fixed label universe (every
    option description seen in the track, so a class that is never gold caps the oracle below 1);
    iSarcasmEval conservative F1 of "yes" on track A-En (the headline; A-Ar and C tracks shown, B never
    scores); MMLU subject-macro, BPoMP variant-macro, POP909 song-macro, cfcolor user-macro accuracy;
    ChessBench / Habermas accuracy with ties accepted (meta.accepted); GSM8K tracks averaged; the rest
    plain accuracy.
  * Areas = plain means (knowledge includes ChessBench, folded from "games"); Index = 100 × mean of
    the five areas (``balanced_raw``). Also ``balanced_skill`` (clip((raw − chance)/(1 − chance))
    per track first) and the chance level of each benchmark, computed from the items the way the
    kit does (conservative-F1 chance levels are the kit's Monte-Carlo values, ``F1_CHANCE``).
  * Display-only benchmarks: the kit's ``report.score`` native metric (accuracy / macro-F1 over
    observed labels / case exact / Brier) on complete groups only, with coverage beside it.

Budget emulation (``--budget-tokens``): JDI forbids truncation, so a request over the engine's
budget is "unsupported" and scores as wrong. Two layouts:
  * combined (default): tokens(state) + tokens(question) + tokens(longest choice) + specials > N;
  * split (``--choice-budget-tokens M``, Decima's layout): tokens("query: " + state) > N, or any
    tokens("passage: " + question + " " + choice) > M (with special tokens, after decima.normalize),
    exactly what Decima would otherwise truncate.
``--decima-checkpoint DIR`` takes N, M, the tokenizer and the text layout (question with the
state or with each choice, prefixes) from that checkpoint's decima.json.
The flagged ids apply to ``--budget-systems`` (default: every system) and can be saved with
``--write-unsupported`` for reuse as ``--unsupported name=file``.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

from bench.suites_public import JDI_BENCHMARKS, JDI_REFERENCE

# kit constants.AREAS + FOLD (ChessBench → knowledge), interactive environments dropped
AREAS = {
    "knowledge": [24, 25, 30, 43, 44, 31],
    "language": [11, 40, 41],
    "retrieval": [36, 37],
    "tools": [1, 2, 6],
    "arts": [20, 21, 22, 23, 50],
}
PANEL = [n for ids in AREAS.values() for n in ids]
assert sorted(PANEL) == sorted(n for n, b in JDI_BENCHMARKS.items() if b[2]) and all(JDI_BENCHMARKS[n][2] == a for a, ids in AREAS.items() for n in ids)
HEADLINE = {40: "iSarcasmEval-A-En"}
CONS_F1 = {11, 37, 41}                          # conservative macro-F1 over option descriptions
MACRO_KEY = {24: "subject", 20: "variant", 22: "song_id", 23: "user_id"}
ACCEPTED = {31, 50}
F1_REPORT = {4, 5, 10, 11, 12, 37, 39, 40, 41, 42}      # kit report.F1_BENCHMARKS
CASE_EXACT_REPORT = {1, 9, 33, 38}
LABELS_NEEDED = F1_REPORT
# Scoreable queries after the kit's exclusions (data/public/jdi/rows/{2,36}.jsonl minus
# hub/excluded-questions.json); our loader keeps only the ones with a judged-relevant candidate.
RETRIEVAL_QUERIES = {2: 7704, 36: 1297}
RETRIEVAL_KEPT = {2: 5329, 36: 519}
# kit data/chance-baselines.json (Monte Carlo, 1000 trials) for the conservative-F1 tracks
F1_CHANCE = {11: 0.30851893087083776, 37: 0.20273782411694521, 41: 0.3333245556439576,
             "iSarcasmEval-A-En": 0.2226862339379383, "iSarcasmEval-A-Ar": 0.22250656045660705}
# eval items per suite in the full build (bench.items --suites 'jdi/*' --limit 1000000000), for a partial-build warning
FULL_ITEMS = {
    "jdi/bfcl": 4768, "jdi/toolret": 5329, "jdi/apibank": 508, "jdi/banking77": 3080, "jdi/clinc150": 5500,
    "jdi/routerbench-0shot": 10006, "jdi/routerbench-5shot": 9994, "jdi/home-appliance": 2895, "jdi/sgd": 2500,
    "jdi/contractnli": 2091, "jdi/anli": 3200, "jdi/bpomp": 5000, "jdi/humicroedit": 2628, "jdi/pop909": 2000,
    "jdi/cfcolor": 5000, "jdi/mmlu": 14033, "jdi/gpqa-diamond": 196, "jdi/arc-easy": 2376, "jdi/arc-challenge": 1172,
    "jdi/winogrande": 1267, "jdi/hellaswag": 10042, "jdi/gsm8k-4choice": 1319, "jdi/gsm8k-10choice": 1319,
    "jdi/chessbench": 5000, "jdi/musr": 752, "jdi/sata-bench": 15517, "jdi/simplebench": 10, "jdi/bright": 519,
    "jdi/esci": 5000, "jdi/acos": 318945, "jdi/finentity": 2129, "jdi/isarcasmeval-a-en": 1400,
    "jdi/isarcasmeval-a-ar": 1400, "jdi/isarcasmeval-b-en": 8400, "jdi/isarcasmeval-c-en": 200,
    "jdi/isarcasmeval-c-ar": 200, "jdi/vast": 3006, "jdi/nli4ct": 5500, "jdi/cruxeval": 570, "jdi/cladder": 5000,
    "jdi/forecastbench": 10139, "jdi/habermas": 1676,
}
REF_SHORT = {"Jev 1.13.0": "Jev1.13", "Decider 35B-A3B": "Dec35B", "Kev 9B": "Kev9B", "Kev 0.5B": "Kev0.5B",
             "Decision 1.0 Kai (mmBERT-base)": "Kai", "GLiNER 2.5 small (74M)": "GLiN-s", "Laya": "Laya"}


@dataclass(slots=True)
class It:
    id: str
    n: int
    track: str
    group: str
    fld: str | None
    kind: str
    gold: int
    nc: int
    keys: list | None = None
    labels: list | None = None          # choice texts = the kit's semantic labels (checked 1:1 on the data)
    macro: str = "all"
    accepted: list | None = None
    quality: list | None = None
    doc_ids: list | None = None
    qrels: list | None = None
    ideal: list | None = None


def load_items(path: str, budget=None) -> tuple[list[It], dict[str, int], set[str]]:
    """Stream the (≈1 GB) items file keeping only what scoring needs; optionally flag budget overruns."""
    items, per_suite, over = [], collections.Counter(), set()
    chunk = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if not r["suite"].startswith("jdi/") or r.get("split") != "eval":
                continue
            per_suite[r["suite"]] += 1
            m, n = r["meta"], r["meta"]["catalog_id"]
            it = It(r["id"], n, m["track"], str(m["group"]), m.get("field"), r["kind"], r["gold"], len(r["choices"]),
                    keys=m.get("keys"), macro=str(m.get(MACRO_KEY.get(n), "all")) if n in MACRO_KEY else "all",
                    accepted=m.get("accepted"), quality=m.get("quality"))
            if n in LABELS_NEEDED:
                it.labels = r["choices"]
            if r["kind"] == "rank":
                it.doc_ids, it.qrels, it.ideal = m["doc_ids"], m["qrels"], m["ideal"]
            items.append(it)
            if budget is not None:
                chunk.append(r)
                if len(chunk) >= 20000:
                    over |= budget(chunk)
                    chunk = []
    if budget is not None and chunk:
        over |= budget(chunk)
    return items, dict(per_suite), over


# ── budget ───────────────────────────────────────────────────────────────────

def make_budget(tokenizer: str, n: int, m: int | None, cfg=None):
    """→ fn(list of item rows) → ids over budget. Token counts cached by string hash.
    `cfg` (a DecimaConfig) renders state/choice texts exactly as that checkpoint does."""
    os.environ.setdefault("RAYON_NUM_THREADS", "4")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    from transformers import AutoTokenizer

    from decima.normalize import normalize

    tok = AutoTokenizer.from_pretrained(tokenizer)
    tok.model_max_length = 10 ** 9                # we count, never feed: silence the >512 warning
    cache: dict[tuple[int, bool], int] = {}
    specials = tok.num_special_tokens_to_add()

    def lengths(texts: list[str], special: bool) -> list[int]:
        todo = list({t for t in texts if (hash(t), special) not in cache})
        for i in range(0, len(todo), 2048):
            b = todo[i:i + 2048]
            enc = tok(b, add_special_tokens=special, truncation=False)["input_ids"]
            for t, ids in zip(b, enc):
                cache[(hash(t), special)] = len(ids)
        return [cache[(hash(t), special)] for t in texts]

    def check(rows: list[dict]) -> set[str]:
        if m is None:                            # combined: state + question + longest choice
            s = lengths([r["state"] or "" for r in rows], False)
            q = lengths([r["question"] or "" for r in rows], False)
            flat = [c for r in rows for c in r["choices"]]
            cl, out, i = lengths(flat, False), set(), 0
            for r, a, b in zip(rows, s, q):
                k = len(r["choices"])
                if a + b + max(cl[i:i + k], default=0) + specials > n:
                    out.add(r["id"])
                i += k
            return out
        if cfg is not None:                      # the checkpoint's own layout (question placement, prefixes)
            s = lengths([cfg.state_of(r["state"] or "", r["question"] or "", r["lang"]) for r in rows], True)
            flat = [cfg.choice_of(r["question"] or "", c, r["lang"]) for r in rows for c in r["choices"]]
        else:
            s = lengths(["query: " + normalize(r["state"] or "", r["lang"]) for r in rows], True)
            flat = ["passage: " + normalize(f"{r['question']} {c}".strip(), r["lang"]) for r in rows for c in r["choices"]]
        cl, out, i = lengths(flat, True), set(), 0
        for r, a in zip(rows, s):
            k = len(r["choices"])
            if a > n or max(cl[i:i + k], default=0) > m:
                out.add(r["id"])
            i += k
        return out

    return check


# ── preds ────────────────────────────────────────────────────────────────────

def load_preds(path: str, want: set[str]) -> dict[str, list[float]]:
    if path.endswith(".parquet"):                 # published decima-bench-predictions layout
        import pyarrow.parquet as pq

        return {r["id"]: r["probs"] for r in pq.read_table(path).to_pylist() if r["id"] in want}
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r["id"] in want:
                    out[r["id"]] = r["probs"]
    return out


def valid(it: It, p) -> list[float] | None:
    if not isinstance(p, list) or len(p) != it.nc or not all(isinstance(x, (int, float)) and math.isfinite(x) and x >= 0 for x in p):
        return None
    if it.kind == "rank":
        return p if all(x <= 1 for x in p) else None
    s = sum(p)
    return [x / s for x in p] if s > 0 else None


def argmax(p: list[float]) -> int:
    return max(range(len(p)), key=lambda i: (p[i], -i))


# ── metrics (kit scoring/metrics.py) ─────────────────────────────────────────

def avg(xs):
    return statistics.mean(xs) if xs else 0.0


def conservative_f1(golds, preds, classes, positive=None):
    missing = sum(p is None for p in preds)
    vals = []
    for k in [positive] if positive is not None else classes:
        tp = sum(g == k and p == k for g, p in zip(golds, preds))
        fp = sum(g != k and p == k for g, p in zip(golds, preds))
        fn = sum(g == k and p is not None and p != k for g, p in zip(golds, preds))
        d = 2 * tp + fp + fn + missing
        vals.append(2 * tp / d if d else 0.0)
    return avg(vals)


def macro_f1(pairs):
    labels = set(x for p in pairs for x in p)
    res = []
    for lab in labels:
        tp = sum(g == lab and p == lab for g, p in pairs)
        fp = sum(g != lab and p == lab for g, p in pairs)
        fn = sum(g == lab and p != lab for g, p in pairs)
        res.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0)
    return statistics.mean(res) if res else None


def ndcg10(it: It, p: list[float]) -> float:
    order = sorted(range(it.nc), key=lambda i: (-p[i], it.doc_ids[i]))
    dcg = sum((2 ** it.qrels[j] - 1) / math.log2(i + 2) for i, j in enumerate(order[:10]))
    idcg = sum((2 ** v - 1) / math.log2(i + 2) for i, v in enumerate(it.ideal))
    return dcg / idcg if idcg else 0.0


def random_ndcg(it: It) -> float:
    idcg = sum((2 ** v - 1) / math.log2(i + 2) for i, v in enumerate(it.ideal))
    gain = avg([2 ** v - 1 for v in it.qrels])
    return gain * sum(1 / math.log2(i + 2) for i in range(min(10, it.nc))) / idcg if idcg else 0.0


def skill(x, b):
    return min(1.0, max(0.0, (x - b) / (1 - b))) if b < 1 else 0.0


# ── scoring ──────────────────────────────────────────────────────────────────

def panel_benchmark(n: int, items: list[It], P: dict[str, list[float]], good: dict, full: bool) -> dict:
    """kit index.static_score for one benchmark (P: id → valid probs; good: group → all answered)."""
    tracks = collections.defaultdict(list)
    for it in items:
        t = it.track if n in (30, 40, 6) else "all"
        if n == 40 and "-B-" in t:
            continue
        tracks[t].append(it)
    parts = []
    for t, rr in tracks.items():
        ok = lambda it: good[it.group]
        if n in (2, 36):
            values = [ndcg10(it, P[it.id]) if ok(it) else 0.0 for it in rr]
            base = [random_ndcg(it) for it in rr]
            pad = RETRIEVAL_QUERIES[n] - len(rr) if full else 0          # dropped queries: 0 for everyone
            raw, chance = sum(values) / (len(rr) + pad), sum(base) / (len(rr) + pad)
        elif n == 6:
            q = [it for it in rr if it.fld == "quality"]
            raw = avg([it.quality[argmax(P[it.id])] if ok(it) else 0.0 for it in q])
            chance = avg([avg(it.quality) for it in q])
        elif n == 1:
            by = collections.defaultdict(list)
            for it in rr:
                by[it.group].append(it)
            raw = avg([float(good[g] and all(argmax(P[it.id]) == it.gold for it in its)) for g, its in by.items()])
            chance = avg([math.prod(1 / it.nc for it in its) for its in by.values()])
        elif n in CONS_F1 or (n == 40 and "-A-" in t):
            if n == 40:
                golds = [it.keys[it.gold] for it in rr]
                preds = [it.keys[argmax(P[it.id])] if ok(it) else None for it in rr]
                raw, chance = conservative_f1(golds, preds, None, "yes"), F1_CHANCE[t]
            else:
                golds = [it.labels[it.gold] for it in rr]
                preds = [it.labels[argmax(P[it.id])] if ok(it) else None for it in rr]
                raw, chance = conservative_f1(golds, preds, sorted({l for it in rr for l in it.labels})), F1_CHANCE[n]
        else:
            macro = collections.defaultdict(list)
            for it in rr:
                acc = it.accepted if n in ACCEPTED else [it.gold]
                v = float(ok(it) and argmax(P[it.id]) in acc)
                macro[it.macro].append((v, len(acc) / it.nc))
            raw = avg([avg([v for v, _ in xs]) for xs in macro.values()])
            chance = avg([avg([c for _, c in xs]) for xs in macro.values()])
        gids = {it.group for it in rr}
        parts.append({"track": t, "raw": raw, "random": chance, "skill": skill(raw, chance),
                      "coverage": avg([int(good[g]) for g in gids]), "cases": len(gids)})
    out = {"raw": avg([p["raw"] for p in parts]), "skill": avg([p["skill"] for p in parts]),
           "random": avg([p["random"] for p in parts]), "coverage": avg([p["coverage"] for p in parts]),
           "tracks": sorted(parts, key=lambda p: p["track"])}
    h = next((p for p in parts if p["track"] == HEADLINE.get(n)), None)
    if h:                                         # kit index.headline: iSarcasmEval = track A-En
        out.update(raw=h["raw"], skill=h["skill"], random=h["random"], coverage=h["coverage"])
    return out


def display_benchmark(n: int, items: list[It], P: dict[str, list[float]], good: dict) -> dict:
    """kit report.score primary metric on complete groups only (native, not coverage-adjusted)."""
    groups = collections.defaultdict(list)
    for it in items:
        groups[it.group].append(it)
    done = [it for it in items if good[it.group]]
    cov = avg([int(good[g]) for g in groups])
    if not done:
        return {"metric": None, "score": None, "coverage": cov}
    if n == 48:
        yes = [it.keys.index("yes") for it in done]
        return {"metric": "Brier (lower is better)", "coverage": cov,
                "score": avg([(P[it.id][y] - (it.gold == y)) ** 2 for it, y in zip(done, yes)])}
    if n in CASE_EXACT_REPORT:
        score = avg([float(all(argmax(P[it.id]) == it.gold for it in its)) for g, its in groups.items() if good[g]])
        return {"metric": "case exact accuracy", "score": score, "coverage": cov}
    if n in F1_REPORT:
        score = macro_f1([(it.labels[it.gold], it.labels[argmax(P[it.id])]) for it in done])
        return {"metric": "macro-F1", "score": score, "coverage": cov}
    return {"metric": "accuracy", "score": avg([float(argmax(P[it.id]) == it.gold) for it in done]), "coverage": cov}


def score_system(items_by_n: dict[int, list[It]], preds: dict[str, list[float]], unsupported: set[str], full_ret: dict[int, bool]) -> dict:
    P, bad = {}, collections.defaultdict(set)
    n_missing = n_invalid = n_unsup = 0
    for n, items in items_by_n.items():
        for it in items:
            if it.id in unsupported:
                bad[n].add(it.group)
                n_unsup += 1
                continue
            p = preds.get(it.id)
            v = valid(it, p) if p is not None else None
            if v is None:
                bad[n].add(it.group)
                n_missing += p is None
                n_invalid += p is not None
            else:
                P[it.id] = v
    good = {n: collections.defaultdict(lambda: True, {g: False for g in bad[n]}) for n in items_by_n}
    bench = {n: panel_benchmark(n, items_by_n[n], P, good[n], full_ret.get(n, False)) for n in PANEL if n in items_by_n}
    areas = {a: avg([bench[n]["raw"] for n in ids if n in bench]) for a, ids in AREAS.items()}
    askill = {a: avg([bench[n]["skill"] for n in ids if n in bench]) for a, ids in AREAS.items()}
    chance = {a: avg([bench[n]["random"] for n in ids if n in bench]) for a, ids in AREAS.items()}
    display = {n: display_benchmark(n, items_by_n[n], P, good[n]) for n in sorted(items_by_n) if n not in PANEL}
    if 40 in items_by_n:                          # B track: never scored by the kit; show its per-field positive F1 mean
        b = [it for it in items_by_n[40] if "-B-" in it.track and good[40][it.group]]
        if b:
            by = collections.defaultdict(list)
            for it in b:
                by[it.fld].append((it.keys[it.gold], it.keys[argmax(P[it.id])]))
            f1 = [conservative_f1([g for g, _ in v], [p for _, p in v], None, "yes") for v in by.values()]
            display["40-B"] = {"metric": "category macro-F1 (yes per field)", "score": avg(f1),
                               "coverage": avg([int(good[40][g]) for g in {it.group for it in items_by_n[40] if "-B-" in it.track}])}
    return {"index": 100 * avg(list(areas.values())), "balanced_skill": 100 * avg(list(askill.values())),
            "chance_index": 100 * avg(list(chance.values())), "areas": areas, "area_skill": askill,
            "benchmarks": bench, "display": display,
            "counts": {"items": sum(map(len, items_by_n.values())), "scored": len(P), "missing": n_missing,
                       "invalid": n_invalid, "unsupported": n_unsup}}


# ── printing ─────────────────────────────────────────────────────────────────

def fmt(x, w=7, d=3):
    return f"{'—':>{w}}" if x is None else f"{x:>{w}.{d}f}"


def report(res: dict[str, dict]) -> None:
    names = list(res)
    refs = list(JDI_REFERENCE)
    first = res[names[0]]
    head = f"{'benchmark':24s}{'chance':>7s} " + "".join(f"{s[:9]:>10s}" for s in names) + " │" + "".join(f"{REF_SHORT.get(r, r)[:8]:>9s}" for r in refs)
    print("\nDecision Index 0.1 — panel (raw index metric, unanswered = 0)")
    print(head)
    print("─" * len(head))
    for a, ids in AREAS.items():
        for n in ids:
            if n not in first["benchmarks"]:
                continue
            b = first["benchmarks"][n]
            print(f"{JDI_BENCHMARKS[n][0]:24s}{fmt(b['random'])} " + "".join(fmt(res[s]["benchmarks"][n]["raw"], 10) for s in names)
                  + " │" + "".join(fmt(JDI_REFERENCE[r].get(n), 9) for r in refs))
            if len(b["tracks"]) > 1:
                for i, t in enumerate(b["tracks"]):
                    tag = t["track"].split("-", 1)[-1] if n != 40 else t["track"].replace("iSarcasmEval-", "")
                    mark = "*" if t["track"] == HEADLINE.get(n) else " "
                    print(f"  {mark}{tag:21s}{fmt(t['random'])} " + "".join(fmt(res[s]['benchmarks'][n]['tracks'][i]['raw'], 10) for s in names))
        ref_area = [avg([JDI_REFERENCE[r][n] for n in ids]) for r in refs]
        print(f"{'= ' + a:24s}{fmt(avg([first['benchmarks'][n]['random'] for n in ids if n in first['benchmarks']]))} "
              + "".join(fmt(res[s]["areas"][a], 10) for s in names) + " │" + "".join(fmt(v, 9) for v in ref_area))
        print()
    ref_idx = [100 * avg([avg([JDI_REFERENCE[r][n] for n in ids]) for ids in AREAS.values()]) for r in refs]
    print(f"{'INDEX (balanced_raw)':24s}{fmt(first['chance_index'], 7, 2)} " + "".join(fmt(res[s]["index"], 10, 2) for s in names)
          + " │" + "".join(fmt(v, 9, 2) for v in ref_idx))
    print(f"{'balanced_skill':24s}{'':7s} " + "".join(fmt(res[s]["balanced_skill"], 10, 2) for s in names))
    print("(* = headline track; reference columns: JDI_REFERENCE per-benchmark raw, areas/index recomputed from them;"
          " the board's own index values are in the suites_public docstring)")

    low = [(s, n, res[s]["benchmarks"][n]["coverage"]) for s in names for n in res[s]["benchmarks"] if res[s]["benchmarks"][n]["coverage"] < 1]
    if low:
        print("\ncoverage < 1 (answered case groups / all):")
        for s, n, c in low:
            print(f"  {s:12s} {JDI_BENCHMARKS[n][0]:18s} {c:.3f}")

    print("\nDisplay-only (kit report metric on complete groups; coverage in brackets)")
    for n in first["display"]:
        slug = JDI_BENCHMARKS[n][0] + (" [contaminated]" if n in (4, 5) else "") if isinstance(n, int) else "isarcasmeval-b-en"
        d0 = first["display"][n]
        print(f"  {slug:28s}{(d0['metric'] or ''):34s}" + "".join(
            f"{fmt(res[s]['display'][n]['score'], 8)} ({res[s]['display'][n]['coverage']:.2f})" for s in names))

    for s in names:
        c = res[s]["counts"]
        print(f"\n{s}: {c['scored']}/{c['items']} items scored, {c['missing']} missing, {c['invalid']} invalid, {c['unsupported']} unsupported")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", required=True)
    ap.add_argument("--preds", nargs="+", required=True, metavar="NAME=PATH")
    ap.add_argument("--unsupported", nargs="*", default=[], metavar="NAME=IDS.txt", help="ids (one per line) over the system's budget")
    ap.add_argument("--budget-tokens", type=int, help="state budget (split layout) or total budget (combined)")
    ap.add_argument("--choice-budget-tokens", type=int, help="per-choice budget; switches to the split (Decima) layout")
    ap.add_argument("--tokenizer", help="HF id or path, e.g. checkpoints/v0/best/encoder")
    ap.add_argument("--budget-systems", nargs="*", help="systems the budget applies to (default: all)")
    ap.add_argument("--decima-checkpoint", help="derive the split budget (lengths, layout, tokenizer) from a Decima checkpoint dir")
    ap.add_argument("--write-unsupported", help="save the budget-flagged ids here")
    ap.add_argument("--out")
    args = ap.parse_args()

    budget = None
    if args.decima_checkpoint:
        from decima.model import DecimaConfig

        raw = json.loads((Path(args.decima_checkpoint) / "decima.json").read_text())
        c = DecimaConfig(**{k: v for k, v in raw.items() if k in DecimaConfig.__dataclass_fields__})
        args.budget_tokens, args.choice_budget_tokens = c.max_state_tokens, c.max_choice_tokens
        args.tokenizer = str(Path(args.decima_checkpoint) / "encoder")
        budget = make_budget(args.tokenizer, c.max_state_tokens, c.max_choice_tokens, c)
    elif args.budget_tokens:
        if not args.tokenizer:
            ap.error("--budget-tokens needs --tokenizer")
        budget = make_budget(args.tokenizer, args.budget_tokens, args.choice_budget_tokens)
    items, per_suite, over = load_items(args.items, budget)
    if not items:
        sys.exit("no jdi/* eval items in " + args.items)
    partial = {s: (per_suite.get(s, 0), k) for s, k in FULL_ITEMS.items() if per_suite.get(s, 0) != k}
    if partial:
        print("WARNING: not the full suite (build with --limit 1000000000); the index is not comparable to the board:", file=sys.stderr)
        for s, (a, b) in partial.items():
            print(f"  {s}: {a} items, full build has {b}", file=sys.stderr)
    by_n = collections.defaultdict(list)
    for it in items:
        by_n[it.n].append(it)
    full_ret = {n: per_suite.get(f"jdi/{JDI_BENCHMARKS[n][0]}") == RETRIEVAL_KEPT[n] for n in RETRIEVAL_QUERIES}
    print(f"{len(items)} items over {len(by_n)} benchmarks from {args.items}")
    if budget is not None:
        print(f"budget: {len(over)} items over ({args.budget_tokens}" + (f"/{args.choice_budget_tokens}" if args.choice_budget_tokens else "") + f" tokens, {args.tokenizer})")
        if args.write_unsupported:
            Path(args.write_unsupported).write_text("".join(i + "\n" for i in sorted(over)))

    unsup = collections.defaultdict(set)
    for spec in args.unsupported:
        name, path = spec.split("=", 1)
        unsup[name] |= {l.strip() for l in open(path) if l.strip()}
    want = {it.id for it in items}
    res = {}
    for spec in args.preds:
        name, path = spec.split("=", 1)
        u = unsup[name] | (over if budget is not None and (not args.budget_systems or name in args.budget_systems) else set())
        res[name] = score_system(by_n, load_preds(path, want), u, full_ret)
    report(res)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"items": args.items, "kit_commit": "see bench.suites_public.JDI_KIT_COMMIT",
                                              "per_suite": per_suite, "systems": res}, indent=1, default=str) + "\n")
        print(f"→ {args.out}")


if __name__ == "__main__":
    main()
