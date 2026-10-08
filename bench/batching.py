"""Batched multi-question inference: parity with one-by-one `decide`, and latency before / after.

    uv run python -m bench.batching --check export/agent3-int8 export/base2-int8       # max |Δp| ≤ 1e-5 or exit 1
    uv run python -m bench.batching --latency export/agent3-int8 --threads 1 4 --out runs/bench-batching.json
    uv run python -m bench.batching --chunk export/base2-int8 --threads 4               # long="chunk" timings
    uv run python -m bench.batching --scaling export/agent3-int8 --threads 4            # 1 question, 256 → 2048 tokens

Inputs are self-contained: a synthetic service agreement of a requested token length (numbered clauses
with varied parties, amounts and deadlines) and TypeSafe-style questions about it (noul, choice, score),
rendered by decima.systemone. "before" is what serve.py did until now: `decide` once per question;
"after" is `decide_many`. Choice encodings are warm (cached) unless the row says cold. Times exclude any
stretch the process spent stopped (the GX10's thermal guard SIGSTOPs bench jobs when the CPU runs hot).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import threading
import time

import numpy as np

from decima.runtime import DecimaOnnx
from decima.systemone import to_question
from decima.types import Question

PARTIES = ["Northwind Freight", "Halcyon Labs", "Brightwater Utilities", "Kestrel Logistics", "Orchid Health", "Atlas Mills"]
GOODS = ["steel brackets", "lab reagents", "server racks", "cold-chain vaccines", "printed circuit boards", "office furniture"]
SHORT_STATE = "Hi, I was charged twice for my order #4471 last week and nobody answers my emails. I want a refund today."


def make_doc(tok, n_tokens: int, seed: int = 0) -> str:
    """Deterministic contract-like text of about `n_tokens` tokens (never more)."""
    rng = np.random.default_rng(seed)
    doc, i = "", 1
    while True:
        a, g = PARTIES[rng.integers(len(PARTIES))], GOODS[rng.integers(len(GOODS))]
        clause = (f"Clause {i}. {a} shall deliver the {g} listed in schedule {i} within {int(rng.integers(2, 30))} business days of each "
                  f"purchase order. Late delivery entitles the customer to a credit of {int(rng.integers(1, 9))} percent of the order value "
                  f"per week of delay, capped at {int(rng.integers(10, 40))} percent. Invoices under this clause are payable within "
                  f"{int(rng.choice([15, 30, 45, 60]))} days. ")
        if rng.random() < 0.3:
            clause += f"Either party may terminate clause {i} with {int(rng.choice([14, 30, 90]))} days written notice.\n"
        if len(tok(doc + clause)["input_ids"]) > n_tokens:
            return doc.strip()
        doc += clause
        i += 1


def questions(n: int) -> list[Question]:
    """n TypeSafe-style questions, cycling through noul / choice / score templates with varied wording."""
    specs = []
    for k in range(n):
        p, g = PARTIES[k % len(PARTIES)], GOODS[(k // 2) % len(GOODS)]
        t = k % 3
        if t == 0:
            specs.append({"type": "noul", "instructions": f"The document says {p} delivers {g}."})
        elif t == 1:
            specs.append({"type": "choice", "instructions": f"Which payment term applies to {g}?",
                          "criteria": {"15 days": "invoices due in 15 days", "30 days": "due in 30 days", "45 days": None, "60 days": None}})
        else:
            specs.append({"type": "score", "instructions": f"How strict are the delivery penalties for {p} (question {k})?",
                          "criteria": ["lenient", "moderate", "strict", "very strict"]})
    return [to_question(f"q{k}", s)[1] for k, s in enumerate(specs)]


def kinds() -> list[Question]:
    """One question of every kind, rank included, for the parity check."""
    return questions(6) + [Question("Which topics does the text touch?", ["billing", "shipping", "refunds", "legal terms"], kind="rank"),
                           Question("Is this urgent?", ["yes", "no"], kind="verify")]


def check(path: str, threads: int) -> float:
    rt = DecimaOnnx(path, threads=threads)
    limit = rt.cfg["max_state_tokens"]
    states = [SHORT_STATE, make_doc(rt.tok, 120, 1), make_doc(rt.tok, 400, 2), make_doc(rt.tok, limit - 80, 3), make_doc(rt.tok, limit + 300, 4)]
    qs = kinds()
    worst = 0.0
    for s in states:                                      # one state × many questions (the last state is cut, as in decide)
        ref = [rt.decide(s, q).probs for q in qs]
        rt._cache.clear()                                 # decide_many encodes the choice sets itself
        got = [d.probs for d in rt.decide_many(s, qs, batch_size=8)]
        worst = max(worst, max(float(np.abs(np.subtract(a, b)).max()) for a, b in zip(ref, got)))
    for q in qs:                                          # many states × one question
        ref = [rt.decide(s, q).probs for s in states]
        got = [d.probs for d in rt.decide_batch(states, q, batch_size=8)]
        worst = max(worst, max(float(np.abs(np.subtract(a, b)).max()) for a, b in zip(ref, got)))
    print(f"{path}  threads={threads}  states={len(states)} questions={len(qs)}  max |dp| = {worst:.2e}", flush=True)
    return worst


class Clock:
    """Wall time minus time spent stopped: a ticker thread (ONNX Runtime and the tokenizer release the GIL)
    counts any gap far longer than its period as a stop."""

    def __init__(self, period: float = 0.02, gap: float = 0.5):
        self.period, self.gap, self.stalled, self.ms = period, gap, 0.0, 0.0

    def _tick(self):
        while not self._done.is_set():
            t = time.perf_counter(); time.sleep(self.period); dt = time.perf_counter() - t
            if dt > self.gap:
                self.stalled += dt - self.period

    def __enter__(self):
        self._done = threading.Event()
        self._t = threading.Thread(target=self._tick, daemon=True); self._t.start()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        wall = time.perf_counter() - self.t0
        self._done.set(); self._t.join()
        self.ms = (wall - self.stalled) * 1e3
        if self.stalled:
            print(f"  (excluded {self.stalled:.1f} s stopped)", flush=True)


def timed(fn, reps: int) -> float:
    out = []
    for _ in range(reps):
        with Clock() as c:
            fn()
        out.append(c.ms)
    return float(np.median(out))


def latency(path: str, threads: list[int], long_reps: int) -> list[dict]:
    rows = []
    for th in threads:
        rt = DecimaOnnx(path, threads=th)
        doc = make_doc(rt.tok, 2000 if rt.cfg["max_state_tokens"] >= 2048 else rt.cfg["max_state_tokens"] - 64)
        cases = [("1 question, short state", SHORT_STATE, questions(1), 20), ("6 questions, short state", SHORT_STATE, questions(6), 10),
                 ("24 questions, long state", doc, questions(24), long_reps), ("48 questions, long state", doc, questions(48), long_reps)]
        for name, s, qs, reps in cases:
            n_tok = len(rt.tok(s)["input_ids"])
            if name.startswith("6"):                       # cold: a first request, choice sets not cached yet
                rt._cache.clear(); cold_b = timed(lambda: [rt.decide(s, q) for q in qs], 1)
                rt._cache.clear(); cold_a = timed(lambda: rt.decide_many(s, qs), 1)
                rows.append(row(path, th, name + " (cold)", n_tok, len(qs), cold_b, cold_a))
            rt.decide_many(SHORT_STATE, qs)                # warm the choice cache
            before = timed(lambda: [rt.decide(s, q) for q in qs], reps)
            after = timed(lambda: rt.decide_many(s, qs), reps)
            rows.append(row(path, th, name, n_tok, len(qs), before, after))
    return rows


def chunk(path: str, threads: list[int]) -> list[dict]:
    rows = []
    for th in threads:
        rt = DecimaOnnx(path, threads=th)
        limit = rt.cfg["max_state_tokens"]
        for n_tok, nq in ((2000, 6), (2000, 24), (8000, 6)) if limit < 2048 else ((8000, 6),):
            doc, qs = make_doc(rt.tok, n_tok, 7), questions(nq)
            rt.decide_many(SHORT_STATE, qs)
            with Clock() as c:
                ds = rt.decide_many(doc, qs, long="chunk")
            ms = c.ms
            r = row(path, th, f"{nq} questions, chunked", len(rt.tok(doc)["input_ids"]), nq, None, ms)
            r["windows"] = len(ds[0].meta.get("windows", [None]))
            rows.append(r)
            print(f"  windows per question: {r['windows']}", flush=True)
    return rows


def scaling(path: str, threads: list[int]) -> list[dict]:
    """One question, state length doubling: how the encoder's cost grows with the state."""
    rows = []
    for th in threads:
        rt = DecimaOnnx(path, threads=th)
        q = questions(1)[0]
        rt.decide(SHORT_STATE, q)
        n = 256
        while n <= rt.cfg["max_state_tokens"]:
            doc = make_doc(rt.tok, n - 40, 3)
            rows.append(row(path, th, "1 question, scaling", len(rt.tok(doc)["input_ids"]), 1, None, timed(lambda: rt.decide(doc, q), 3)))
            n *= 2
    return rows


def row(path, th, name, n_tok, nq, before, after) -> dict:
    r = {"model": path, "threads": th, "case": name, "state_tokens": n_tok, "questions": nq,
         "before_ms": None if before is None else round(before, 1), "after_ms": round(after, 1),
         "after_ms_per_question": round(after / nq, 1), "speedup": None if before is None else round(before / after, 2)}
    print(json.dumps(r), flush=True)
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", nargs="*", default=[])
    ap.add_argument("--latency", nargs="*", default=[])
    ap.add_argument("--chunk", nargs="*", default=[])
    ap.add_argument("--scaling", nargs="*", default=[])
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--long-reps", type=int, default=1)
    ap.add_argument("--out")
    a = ap.parse_args()
    worst = max([check(p, t) for p in a.check for t in a.threads] or [0.0])
    rows = [r for p in a.latency for r in latency(p, a.threads, a.long_reps)] + [r for p in a.chunk for r in chunk(p, a.threads)] \
        + [r for p in a.scaling for r in scaling(p, a.threads)]
    if a.out and rows:
        with open(a.out, "w") as f:
            json.dump({"machine": platform.machine(), "processor": platform.processor(), "rows": rows}, f, indent=1)
    if a.check:
        print(f"parity: max |dp| = {worst:.2e} ({'OK' if worst <= 1e-5 else 'FAIL'}, bound 1e-5)")
        sys.exit(0 if worst <= 1e-5 else 1)


if __name__ == "__main__":
    main()
