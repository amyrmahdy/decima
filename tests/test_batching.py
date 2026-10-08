"""decide_many / decide_batch give the numbers of decide, one by one."""

from __future__ import annotations

import numpy as np
import pytest

from bench.batching import SHORT_STATE, kinds, make_doc
from decima.systemone import system_one

TOL = 1e-5


def _states(rt) -> list[str]:
    limit = rt.cfg["max_state_tokens"]
    return [SHORT_STATE, make_doc(rt.tok, 90, 1), make_doc(rt.tok, 300, 2), make_doc(rt.tok, min(limit, 512) - 60, 3)]


def _maxdiff(a, b) -> float:
    return max(float(np.abs(np.subtract(x.probs, y.probs)).max()) for x, y in zip(a, b))


@pytest.mark.parametrize("model", ["base2", "agent3"])
def test_decide_many_equals_decide(model, request):
    rt = request.getfixturevalue(model)
    qs = kinds()
    for s in _states(rt):
        ref = [rt.decide(s, q) for q in qs]
        rt._cache.clear()                        # the batched path encodes the choice sets itself
        got = rt.decide_many(s, qs, batch_size=4)
        assert [d.choices for d in got] == [d.choices for d in ref]
        assert _maxdiff(ref, got) <= TOL


def test_decide_batch_equals_decide(base2):
    states = _states(base2) + [make_doc(base2.tok, 700, 5)]      # the last one is cut at 512, as decide does
    for q in kinds():
        ref = [base2.decide(s, q) for s in states]
        assert _maxdiff(ref, base2.decide_batch(states, q, batch_size=3)) <= TOL


def test_padding_is_exercised(base2, monkeypatch):
    """The parity above must come from padded batches, not from one row per call."""
    shapes = []
    run = base2.enc.run

    class Spy:
        def run(self, out, feeds):
            shapes.append(feeds["attention_mask"].copy())
            return run(out, feeds)

    monkeypatch.setattr(base2, "enc", Spy())
    base2.decide_batch(_states(base2), kinds()[0], batch_size=8)
    assert any(m.shape[0] > 1 and (m.sum(1) < m.shape[1]).any() for m in shapes)


def test_system_one_unchanged(base2):
    """serve.py's answers (now batched) equal the per-question path they replaced."""
    qs = {"team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "charges, refunds", "tech": "bugs"}},
          "urgent": {"type": "noul", "instructions": "The customer needs an answer today."},
          "anger": {"type": "score", "instructions": "How upset is the customer?", "criteria": ["calm", "annoyed", "furious"]}}

    class OneByOne:                              # no decide_many: system_one falls back to decide per question
        def __init__(self, rt):
            self.rt = rt

        def decide(self, s, q):
            return self.rt.decide(s, q)

    assert system_one(base2, SHORT_STATE, qs) == system_one(OneByOne(base2), SHORT_STATE, qs)


def test_torch_reference_decide_many(base2_torch):
    qs = kinds()
    for s in (SHORT_STATE, make_doc(base2_torch.tok, 200, 1)):
        ref = [base2_torch.decide(s, q) for q in qs]
        assert _maxdiff(ref, base2_torch.decide_many(s, qs, batch_size=4)) <= 1e-4    # torch fp32: padded GEMMs may round differently
