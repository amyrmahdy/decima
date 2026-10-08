"""Confidence gating: Decision.abstain, decide(..., min_confidence=…)."""

from __future__ import annotations

from decima.types import Decision, Question


def test_decision_abstain():
    d = Decision(probs=[0.55, 0.45], choices=["yes", "no"])
    assert not d.abstained and d.answer == "yes"
    assert d.abstain(0.6) and d.abstained and d.answer is None
    assert not d.abstain(0.5) and d.answer == "yes"


def test_decide_min_confidence(base2):
    q = Question("Which team should handle this?", ["billing", "tech support", "sales"])
    s = "I was charged twice for my subscription."
    d = base2.decide(s, q)
    assert not d.abstained
    hi, lo = base2.decide(s, q, min_confidence=1.0), base2.decide(s, q, min_confidence=0.0)
    assert hi.abstained and hi.answer is None and hi.probs == d.probs
    assert not lo.abstained and lo.answer == d.top
    ds = base2.decide_many(s, [q, Question("Is the customer happy?", ["yes", "no"], kind="verify")], min_confidence=0.999)
    assert [x.abstained for x in ds] == [x.confidence < 0.999 for x in ds]
