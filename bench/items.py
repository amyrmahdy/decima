"""Scoreboard items — the one format every system is judged on.

One JSONL row per decision:

    {"id", "suite", "lang", "kind", "question", "choices", "gold", "split": "eval"|"calib", "state"}
    + for option-order copies: "group" (id of the original) and "perm" (new choice i = original choice perm[i])

Each system (ours in-process, competitors in their own venvs under bench/external/) reads the same
items and writes preds {"id", "probs"}; bench.score does all the maths. That keeps competitor
dependencies out of this project and makes every comparison literally the same questions.

    uv run python -m bench.items --suites 'kev/*' 'laya/*' --out runs/items/phase0.jsonl
    uv run python -m bench.items --list

Suites come from four registries: decima/* (our own bench, full label sets), laya/* (Laya's
published protocol), kev/* (Kev's published protocol), and public ones (btzsc/*, jevbench/*, typed/*).
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import importlib
import json
import random
from pathlib import Path
from typing import Callable

Loader = Callable[[int, int], list[dict]]


def _decima_suites() -> dict[str, Loader]:
    from bench.datasets import REGISTRY

    def wrap(key: str) -> Loader:
        def load(limit: int, calib: int) -> list[dict]:
            t = REGISTRY[key](limit, calib)
            q = t.question
            base = {"suite": f"decima/{key}", "lang": t.lang, "kind": q.kind, "question": q.text, "choices": list(q.choices)}
            ev = [{**base, "id": f"decima/{key}/e{i}", "state": s, "gold": g, "split": "eval"} for i, (s, g) in enumerate(zip(t.states, t.gold))]
            ca = [{**base, "id": f"decima/{key}/c{i}", "state": s, "gold": g, "split": "calib"} for i, (s, g) in enumerate(zip(t.calib_states, t.calib_gold))]
            return ev + ca
        return load

    return {f"decima/{k}": wrap(k) for k in REGISTRY}


def registry() -> dict[str, Loader]:
    out = _decima_suites()
    for mod in ("bench.suites_laya", "bench.suites_kev", "bench.suites_public"):
        try:
            out.update(importlib.import_module(mod).SUITES)
        except ModuleNotFoundError as e:
            if e.name != mod:
                raise
    return out


def select(patterns: list[str]) -> list[str]:
    reg = registry()
    names = [n for n in reg if any(fnmatch.fnmatch(n, p) for p in patterns)]
    missing = [p for p in patterns if not any(fnmatch.fnmatch(n, p) for n in reg)]
    if missing:
        raise SystemExit(f"no suite matches {missing}; see --list")
    return names


def flip_copy(item: dict) -> dict | None:
    """Same decision with the choices in a different order. Ordered levels (score) keep their
    order by definition, so they get no copy. The permutation is seeded by the item id."""
    n = len(item["choices"])
    if item["kind"] == "score" or n < 2:
        return None
    rng = random.Random(int(hashlib.sha1(item["id"].encode()).hexdigest()[:8], 16))
    perm = list(range(n))
    while perm == list(range(n)):
        rng.shuffle(perm)
    return {
        **item, "id": item["id"] + "#flip", "group": item["id"], "perm": perm,
        "choices": [item["choices"][j] for j in perm], "gold": perm.index(item["gold"]),
    }


def build(patterns: list[str], limit: int | None, calib: int, flips: bool = True) -> list[dict]:
    reg = registry()
    rows = []
    for name in select(patterns):
        items = reg[name](limit, calib) if limit is not None else reg[name](*_defaults(reg[name], calib))
        rows.extend(items)
        if flips:
            rows.extend(f for f in (flip_copy(i) for i in items if i["split"] == "eval") if f)
        n_ev = sum(i["split"] == "eval" for i in items)
        print(f"  {name:28s} eval={n_ev:5d} calib={len(items) - n_ev:5d} choices={len(items[0]['choices']) if items else 0}", flush=True)
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate item ids"
    return rows


def _defaults(fn: Loader, calib: int) -> tuple[int, int]:
    """Suites that mirror a published protocol carry their own default size (e.g. Laya's 300)."""
    import inspect

    p = inspect.signature(fn).parameters
    first = next(iter(p.values()), None)
    limit = first.default if first is not None and first.default is not inspect.Parameter.empty else 1000
    return limit, calib


def write(rows: list[dict], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read(path: str | Path) -> list[dict]:
    """JSONL rows; `.parquet` files (the published decima-bench-predictions layout) are read with pyarrow."""
    if Path(path).suffix == ".parquet":
        import pyarrow.parquet as pq

        return pq.read_table(path).to_pylist()
    return [json.loads(l) for l in Path(path).open() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suites", nargs="*", default=["*"], help="glob patterns, e.g. 'kev/*' 'laya/massive-*'")
    ap.add_argument("--limit", type=int, default=None, help="eval items per suite (default: the suite's own)")
    ap.add_argument("--calib", type=int, default=500)
    ap.add_argument("--no-flips", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out", default="runs/items/items.jsonl")
    args = ap.parse_args()
    if args.list:
        print("\n".join(registry()))
        return
    rows = build(args.suites, args.limit, args.calib, flips=not args.no_flips)
    write(rows, args.out)
    print(f"\n{len(rows)} items → {args.out}")


if __name__ == "__main__":
    main()
