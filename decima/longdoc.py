"""Long documents (opt-in, `long="chunk"`): overlapping, sentence-aware windows and per-kind aggregation.

A state longer than the model's window is split into character spans whose rendering (question + window)
fits `max_state_tokens`. Windows end at a sentence boundary when there is one in their second half, and
the next one starts `overlap` tokens earlier, snapped to a sentence start when one is close. Each window
is decided like an ordinary state; the per-window answers are then combined:

  max   (default) the strongest window. verify: the window with the highest P(first choice, "yes"), so
        the answer is "yes" if any window supports it — right for "does the document mention / contain
        X", wrong for statements about the document as a whole or negated ones ("never mentions X").
        choose / score: the single most confident window, whose distribution is returned unchanged.
        rank: per choice, its highest probability over windows.
  mean  the average distribution over windows: every window is an equal witness. Evidence that sits in
        one window of n is diluted n-fold, so use it for questions about the document as a whole.
  sum   summed log-probabilities, renormalised (product of experts): windows are treated as independent
        evidence. Sharp, and over-confident when windows overlap or agree for the same reason.

Chunking cannot do what needs the whole document at once: comparing a clause on page 1 with one on
page 9, counting across windows, or a fact that only holds in context outside the window.
"""

from __future__ import annotations

import re
from bisect import bisect_left, bisect_right
from typing import Callable

import numpy as np

from .types import Decision, Question

AGGREGATES = ("max", "mean", "sum")
_BOUNDARY = re.compile(r"(?<=[.!?;。！？؟])\s+|\n\s*")


def probs(lp: np.ndarray, kind: str) -> np.ndarray:
    """Log-probabilities → probabilities, exactly as `decide` computes them (independent sigmoids for rank)."""
    return np.exp(lp) if kind == "rank" else np.exp(lp - lp.max()) / np.exp(lp - lp.max()).sum()


def windows(tok, state: str, render: Callable[[str], str], limit: int, overlap: int = 64) -> list[tuple[tuple[int, int], list[int]]]:
    """[(character span of `state`, token ids of its rendering)], each at most `limit` tokens.
    A state that fits comes back whole, with the ids an ordinary decision would use."""
    ids = tok(render(state))["input_ids"]
    if len(ids) <= limit:
        return [((0, len(state)), ids)]
    offs = tok(state, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    n = len(offs)
    budget = max(8, limit - len(tok(render(""))["input_ids"]) - 2)
    overlap = min(overlap, budget // 4)
    ts = [o[0] for o in offs]
    starts = sorted({t for t in (bisect_left(ts, m.start()) for m in _BOUNDARY.finditer(state)) if 0 < t < n})
    out, a = [], 0
    while True:
        b = min(a + budget, n)
        if b < n:
            i = bisect_right(starts, b) - 1
            if i >= 0 and starts[i] > a + budget // 2:
                b = starts[i]
        while True:                                     # the window's own tokenization can differ from the slice
            lo, hi = offs[a][0], offs[b - 1][1]
            ids = tok(render(state[lo:hi]))["input_ids"]
            if len(ids) <= limit:
                break
            if b - a <= 1:
                ids = tok(render(state[lo:hi]), truncation=True, max_length=limit)["input_ids"]
                break
            b = max(a + 1, b - (len(ids) - limit))
        out.append(((lo, hi), ids))
        if b >= n:
            return out
        nxt = b - overlap
        i = bisect_right(starts, nxt) - 1
        if i >= 0 and starts[i] > a and starts[i] >= b - 2 * overlap:
            nxt = starts[i]
        a = max(nxt, a + 1)


def aggregate(kind: str, lps: list[np.ndarray], how: str = "max") -> tuple[np.ndarray, int, np.ndarray]:
    """Per-window log-probabilities → (probabilities, deciding window, per-window probabilities [W, n]).
    The deciding window is the one chosen (max on verify / choose / score), otherwise the one that gives the
    final top choice its highest probability."""
    if how not in AGGREGATES:
        raise ValueError(f"aggregate must be one of {AGGREGATES}, not {how!r}")
    P = np.stack([probs(lp, kind) for lp in lps])
    if how == "max" and kind != "rank":
        w = int(np.argmax(P[:, 0] if kind == "verify" else P.max(1)))
        return P[w], w, P
    if how == "mean":
        p = P.mean(0)
    elif how == "sum":
        if kind == "rank":
            raise ValueError("rank choices are independent sigmoids: aggregate with max or mean")
        p = probs(np.sum(lps, 0), kind)
    else:
        p = P.max(0)
    return p, int(np.argmax(P[:, int(np.argmax(p))])), P


def plan(tok, pairs: list[tuple[str, Question]], render: Callable[[str, Question], str], limit: int, long: str | None,
         overlap: int = 64) -> list[tuple[int, tuple[int, int] | None, list[int]]]:
    """Encoder inputs for (state, question) pairs: [(pair index, window span or None, token ids)].
    Without `long` an over-long state is cut at its end, as an ordinary decision does."""
    if long not in (None, "chunk"):
        raise ValueError(f'long must be None or "chunk", not {long!r}')
    jobs = []
    for i, (s, q) in enumerate(pairs):
        if long == "chunk":
            ws = windows(tok, s, lambda w, q=q: render(w, q), limit, overlap)
            jobs += [(i, span if len(ws) > 1 else None, ids) for span, ids in ws]
        else:
            jobs.append((i, None, tok(render(s, q), truncation=True, max_length=limit)["input_ids"]))
    return jobs


def collect(pairs: list[tuple[str, Question]], jobs: list[tuple], lps: list[np.ndarray], how: str = "max",
            min_confidence: float | None = None, latency_ms: float = 0.0) -> list[Decision]:
    """Per-job log-probabilities (aligned with `plan`'s jobs) → one Decision per pair."""
    per: list[list[np.ndarray]] = [[] for _ in pairs]
    spans: list[list[tuple[int, int]]] = [[] for _ in pairs]
    for (i, span, _), lp in zip(jobs, lps):
        per[i].append(lp)
        if span is not None:
            spans[i].append(span)
    out = []
    for (_, q), lp, sp in zip(pairs, per, spans):
        if sp:
            p, w, P = aggregate(q.kind, lp, how)
            meta = {"window": w, "windows": sp, "window_probs": P.tolist()}
        else:
            p, meta = probs(lp[0], q.kind), {}
        d = Decision(probs=[float(x) for x in p], choices=list(q.choices), latency_ms=latency_ms, meta=meta)
        if min_confidence is not None:
            d.abstain(min_confidence)
        out.append(d)
    return out
