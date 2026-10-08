"""long="chunk": windows over a synthetic long text, aggregation rules, and the deciding window."""

from __future__ import annotations

import numpy as np
import pytest

from bench.batching import make_doc
from decima.longdoc import aggregate, windows
from decima.types import Question

NEEDLE = "The vault access code for the Lisbon office is PURPLE-ELEPHANT-42, and only the night manager may use it."


def _render(rt, q):
    return lambda w: rt._state(w, q)


def test_windows_cover_fit_and_overlap(base2):
    doc = make_doc(base2.tok, 3000, 11)
    q = Question("Does the agreement mention a vault access code?", ["yes", "no"], kind="verify")
    limit = base2.cfg["max_state_tokens"]
    ws = windows(base2.tok, doc, _render(base2, q), limit, overlap=64)
    assert len(ws) >= 6
    spans = [s for s, _ in ws]
    assert spans[0][0] == 0 and spans[-1][1] == len(doc)
    for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
        assert b0 < a1 < b1                                   # overlapping, moving forward, no gap
    for span, ids in ws:
        assert len(ids) <= limit
        assert ids == base2.tok(base2._state(doc[span[0]:span[1]], q))["input_ids"]
    ends = [doc[b - 1] for (_, b) in spans[:-1]]
    assert sum(c in ".\n" for c in ends) >= len(ends) - 1    # windows end at sentence boundaries


def test_windows_short_state_is_whole(base2):
    q = Question("Is it urgent?", ["yes", "no"], kind="verify")
    s = "I need this fixed before the board meeting at noon."
    [(span, ids)] = windows(base2.tok, s, _render(base2, q), base2.cfg["max_state_tokens"])
    assert span == (0, len(s)) and ids == base2.tok(base2._state(s, q))["input_ids"]


def test_windows_without_sentences(base2):
    doc = " ".join(f"item{i}" for i in range(3000))           # no sentence boundary anywhere
    q = Question("Is item 7 listed?", ["yes", "no"], kind="verify")
    ws = windows(base2.tok, doc, _render(base2, q), 512, overlap=32)
    assert len(ws) > 1 and all(len(ids) <= 512 for _, ids in ws)
    assert ws[-1][0][1] == len(doc)


def test_aggregate_rules():
    lp = lambda *p: np.log(np.array(p))
    # verify: any-window evidence, the highest P(yes)
    p, w, P = aggregate("verify", [lp(0.1, 0.9), lp(0.8, 0.2), lp(0.3, 0.7)], "max")
    assert w == 1 and np.allclose(p, [0.8, 0.2]) and P.shape == (3, 2)
    p, w, _ = aggregate("verify", [lp(0.1, 0.9), lp(0.8, 0.2), lp(0.3, 0.7)], "mean")
    assert np.allclose(p, [0.4, 0.6]) and w == 0                # top is "no": the window with the highest P(no)
    # choose: the single most confident window, unchanged
    p, w, _ = aggregate("choose", [lp(0.5, 0.3, 0.2), lp(0.1, 0.05, 0.85), lp(0.6, 0.2, 0.2)], "max")
    assert w == 1 and np.allclose(p, [0.1, 0.05, 0.85])
    # sum: product of experts, renormalised
    p, _, _ = aggregate("choose", [lp(0.5, 0.5), lp(0.8, 0.2)], "sum")
    assert np.allclose(p, [0.8, 0.2])
    # rank: per-choice max of independent sigmoids
    p, _, _ = aggregate("rank", [lp(0.2, 0.9), lp(0.7, 0.1)], "max")
    assert np.allclose(p, [0.7, 0.9])
    with pytest.raises(ValueError):
        aggregate("rank", [lp(0.2, 0.9)], "sum")
    with pytest.raises(ValueError):
        aggregate("choose", [lp(0.2, 0.8)], "vote")


def test_chunk_finds_needle(base2):
    """A fact far past the 512-token window: cut, it is gone; chunked, the deciding window holds it."""
    doc = make_doc(base2.tok, 1500, 21)
    cut = len(doc) * 3 // 4
    cut = doc.index(". ", cut) + 2
    doc = doc[:cut] + NEEDLE + " " + doc[cut:]
    q = Question("Does the text mention a vault access code?", ["yes", "no"], kind="verify")
    plain = base2.decide(doc, q)
    d = base2.decide(doc, q, long="chunk")
    lo, hi = d.meta["windows"][d.meta["window"]]
    assert len(d.meta["windows"]) >= 3 and "windows" not in plain.meta
    assert NEEDLE in doc[lo:hi]
    assert d.probs[0] == max(p[0] for p in d.meta["window_probs"])
    assert d.probs[0] > 0.5 > plain.probs[0]


def test_verify_max_reports_the_chosen_window():
    """max on verify picks the highest P(yes) even when that window still says "no", and reports it."""
    lp = lambda *p: np.log(np.array(p))
    p, w, _ = aggregate("verify", [lp(0.17, 0.83), lp(0.42, 0.58), lp(0.23, 0.77)], "max")
    assert w == 1 and np.allclose(p, [0.42, 0.58])


def test_chunk_when_it_fits_is_plain(base2):
    q = Question("Is the customer angry?", ["yes", "no"], kind="verify")
    s = "This is the third time my order is late. I am done with you."
    a, b = base2.decide(s, q), base2.decide(s, q, long="chunk")
    assert a.probs == b.probs and b.meta == {}


def test_chunk_choice_and_mean(base2):
    doc = make_doc(base2.tok, 1200, 5)
    q = Question("Which payment term appears in the agreement?", ["15 days", "30 days", "45 days", "60 days"])
    mx, mean = (base2.decide_many(doc, [q], long="chunk", aggregate=a)[0] for a in ("max", "mean"))
    P = np.array(mx.meta["window_probs"])
    assert np.allclose(mx.probs, P[int(np.argmax(P.max(1)))])
    assert np.allclose(mean.probs, P.mean(0), atol=1e-12)
    assert abs(sum(mx.probs) - 1) < 1e-6 and abs(sum(mean.probs) - 1) < 1e-6      # float32 window probabilities


def test_bad_options(base2):
    q = Question("Is it urgent?", ["yes", "no"], kind="verify")
    with pytest.raises(ValueError):
        base2.decide_many("x", [q], long="split")
    with pytest.raises(ValueError):
        base2.decide_many("x", [q], aggregate="vote")
