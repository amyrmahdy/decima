"""TypeSafe System One wire format on top of Decima: the same request and answer shapes as Jev's API
(and Kev's and Ollaya's `/v1/systemone`), so code written for Jev runs against a local Decima.

    from decima import Decima
    from decima.systemone import system_one
    d = Decima.from_pretrained("amyrmahdy/decima-small")
    system_one(d, "My card was charged twice", {
        "team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "charges, refunds", "tech": "bugs"}},
        "urgent": {"type": "noul", "instructions": "The customer needs an answer today."},
        "anger": {"type": "score", "instructions": "How upset is the customer?", "criteria": ["calm", "annoyed", "furious"]},
    })

Mapping (the renderings Decima 1.1 was trained on):
  choice → choose over "label: description" (the label alone when there is no description);
  noul   → verify: the statement itself, followed by "True if: … / False if: …" when criteria give them;
           options yes / no; the answer is P(yes);
  score  → score over the level descriptions, level 0 first; the answer is the expected level Σ i·pᵢ.
A question without instructions uses its id, as Ollaya does. Answer fields, their order, the
4-decimal rounding and `confidence = (K·p_max − 1) / (K − 1)` follow TypeSafe's schema.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from .types import Question

MAX_CHOICES, MIN_LEVELS, MAX_LEVELS, MAX_QUESTIONS = 255, 2, 10, 256


class SystemOneError(ValueError):
    """A request that does not fit TypeSafe's schema; `issues` holds FastAPI-style validation items."""

    def __init__(self, msg: str, code: str = "INVALID_REQUEST", issues: list[dict] | None = None):
        super().__init__(msg)
        self.code, self.issues = code, issues or []


def _text(v: Any) -> str:
    if v is None:
        return ""
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def _confidence(p: list[float]) -> float:
    k = len(p)
    return max(0.0, min(1.0, (k * max(p) - 1) / (k - 1))) if k > 1 else 1.0


def _r(x: float) -> float:
    return round(float(x), 4)


def to_question(qid: str, spec: dict, lang: str = "en") -> tuple[str, Question, list[str] | list[str]]:
    """TypeSafe question → (type, Decima Question, answer keys). Raises SystemOneError."""
    if not isinstance(spec, dict):
        raise SystemOneError(f"questions.{qid}: must be an object", issues=[{"loc": ["body", "questions", qid], "msg": "must be an object", "type": "dict_type"}])
    t = spec.get("type")
    ins = spec.get("instructions")
    text = qid if ins is None else _text(ins)
    crit = spec.get("criteria")
    loc = ["body", "questions", qid, str(t), "criteria"]
    if t == "choice":
        if isinstance(crit, list):
            crit = {str(c): None for c in dict.fromkeys(crit)}
        if not isinstance(crit, dict) or not (2 <= len(crit) <= MAX_CHOICES):
            raise SystemOneError(f"questions.{qid}: choice needs 2–{MAX_CHOICES} criteria", issues=[{"loc": loc, "msg": "choice needs 2-255 labels", "type": "value_error"}])
        labels = [str(k) for k in crit]
        opts = [f"{k}: {_text(v)}" if _text(v).strip() else k for k, v in zip(labels, crit.values())]
        return t, Question(text, opts, kind="choose", lang=lang), labels
    if t == "score":
        if not isinstance(crit, list) or not (MIN_LEVELS <= len(crit) <= MAX_LEVELS):
            raise SystemOneError(f"questions.{qid}: score needs {MIN_LEVELS}–{MAX_LEVELS} levels", issues=[{"loc": loc, "msg": "score needs 2-10 levels", "type": "value_error"}])
        levels = [_text(v) for v in crit]
        return t, Question(text, levels, kind="score", lang=lang), levels
    if t == "noul":
        if isinstance(crit, dict):
            tr, fa = _text(crit.get("true")).strip(), _text(crit.get("false")).strip()
            if tr or fa:
                text = f"{text}\nTrue if: {tr or '—'}\nFalse if: {fa or '—'}"
        return t, Question(text, ["yes", "no"], kind="verify", lang=lang), ["yes", "no"]
    raise SystemOneError(f"questions.{qid}: unknown type {t!r}", issues=[{"loc": ["body", "questions", qid, "type"], "msg": "must be choice, score or noul", "type": "literal_error"}])


def system_one(model, state: Any, questions: dict, lang: str = "en", long: str | None = None, aggregate: str = "max",
               min_confidence: float | None = None) -> dict:
    """Answer TypeSafe-style questions about one state. Returns the `answers` object.

    All questions go through `model.decide_many` (batched, same numbers as one by one). Opt-in extras, absent
    from the answers unless asked for: `long="chunk"` decides an over-long state window by window
    (decima/longdoc.py) and adds "window": {"index", "start", "end", "count"} (character span of the state)
    to answers that needed more than one window; `min_confidence` adds "abstain": true / false, comparing it
    with the answer's TypeSafe confidence (for noul, |2·noul − 1|)."""
    if not isinstance(questions, dict) or not (1 <= len(questions) <= MAX_QUESTIONS):
        raise SystemOneError("questions: 1–256 questions required", issues=[{"loc": ["body", "questions"], "msg": "1-256 questions", "type": "value_error"}])
    if isinstance(state, (bool, int, float)) or state is None:
        raise SystemOneError("state: must be a string, object or array", issues=[{"loc": ["body", "state"], "msg": "must be a string, object or array", "type": "state_type"}])
    s = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    parsed = [(qid, *to_question(qid, spec, lang)) for qid, spec in questions.items()]
    if hasattr(model, "decide_many"):
        decisions = model.decide_many(s, [q for _, _, q, _ in parsed], long=long, aggregate=aggregate)
    else:
        if long is not None:
            raise SystemOneError(f"{type(model).__name__} cannot chunk long states")
        decisions = [model.decide(s, q) for _, _, q, _ in parsed]
    answers = {}
    for (qid, t, _, keys), d in zip(parsed, decisions):
        p = d.probs
        if t == "choice":
            i = int(np.argmax(p))
            a = {"type": "choice", "choice": keys[i], "confidence": _r(_confidence(p)),
                 "probabilities": {k: _r(v) for k, v in zip(keys, p)}}
        elif t == "score":
            a = {"type": "score", "score": _r(sum(i * v for i, v in enumerate(p))), "confidence": _r(_confidence(p)),
                 "legend": {str(i): lv for i, lv in enumerate(keys)}, "probabilities": {str(i): _r(v) for i, v in enumerate(p)}}
        else:
            a = {"type": "noul", "noul": _r(p[0])}
        if "windows" in d.meta:
            lo, hi = d.meta["windows"][d.meta["window"]]
            a["window"] = {"index": d.meta["window"], "start": lo, "end": hi, "count": len(d.meta["windows"])}
        if min_confidence is not None:
            a["abstain"] = _confidence(p) < min_confidence
        answers[qid] = a
    return answers


def state_tokens(model, state: Any) -> int:
    s = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    return len(model.tok(s)["input_ids"])
