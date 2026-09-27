"""Decima-small as a Decision Index engine (`decision_index.engines.Engine`), in-process, CPU, ONNX int8.

    pip install -e ".[rebuild]"  (the kit)  and  pip install "git+https://github.com/amyrmahdy/decima"
    python -m decision_index pipeline --engine decima_engine:DecimaEngine \
        --option model=amyrmahdy/decima-small --option precision=int8 --option threads=1 \
        --out runs/decima-small --upload <you>/decision-index-results

(`decima_engine.py` must be importable: run from this directory or put it on PYTHONPATH.)

The kit's rules, and how this engine keeps them:
  * No truncation. Decima's trained budget is 512 tokens for the state (with the question, see below) and
    64 tokens per option (with the question). Anything longer raises `Unsupported` — the row is recorded as
    unsupported and scores as wrong. Nothing is cut. (The e5 backbone has 512 positions, so this limit cannot
    be lifted the way the board lifts some entrants' caps.)
  * No option filtering: every option in `criteria` gets a probability.
  * No prompt tuning: one fixed rendering for every request, decided before any run:
      state     string as given; an object -> compact JSON (the kit's `text()`).
      question  the instructions (an object -> compact JSON).
      empty state ("" or {}): the instructions become the state and the question is the fixed line
                "Choose the option that best answers the request." (20 benchmarks put the whole input in the
                instructions; Decima's state slot is where it reads context).
      options   "<key>: <description>", or the description alone when the key is a placeholder
                (A, B, ..., option_N, text_N, headline_N) or equals the description.
      kinds     a choice whose keys are exactly {yes, no} -> "verify"; every other choice -> "choose"
                (softmax); noul -> "verify" over ["no", "yes"] (criteria false/true when given) -> noul = P(yes).
  * Probabilities are Decima's own softmax at the exported temperature; nothing is fitted on the suite.
"""

from __future__ import annotations

import json
import re
import time

from decision_index.engines import Engine, Unsupported

QUESTION_WHEN_EMPTY = "Choose the option that best answers the request."
_PLAIN_KEY = re.compile(r"^(?:[A-Z]|option_\d+|text_\d+|headline_\d+)$")


def _text(x) -> str:
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, separators=(",", ":"))


def _option(key: str, desc) -> str:
    desc = _text(desc) if desc is not None else ""
    if not desc:
        return key
    return desc if _PLAIN_KEY.match(key) or key.lower() == desc.lower() else f"{key}: {desc}"


class DecimaEngine(Engine):
    name = "decima"
    latency = "In-process wall time of one request (all its questions, one after another), tokenisation included; excludes model loading."

    def __init__(self, model="amyrmahdy/decima-small", precision="int8", threads=1, **options):
        super().__init__(**options)
        from decima import Decima

        t0 = time.perf_counter()
        self.d = Decima.from_pretrained(model, precision=precision, threads=int(threads))
        self.load_s = time.perf_counter() - t0
        self.max_state = int(self.d.cfg["max_state_tokens"])
        self.max_choice = int(self.d.cfg["max_choice_tokens"])
        self.provenance = {"kind": "in-process", "model": model, "precision": precision, "threads": int(threads),
                           "runtime": "onnxruntime CPU", "budget": {"state_tokens": self.max_state,
                                                                     "option_tokens": self.max_choice},
                           "policy": "fixed rendering; over-budget requests raise Unsupported (maximum context length); no truncation"}

    def _ntok(self, s: str) -> int:
        return len(self.d.tok(s, truncation=False)["input_ids"])

    def _question(self, state: str, q: dict):
        from decima import Question

        instr = _text(q.get("instructions", ""))
        if not state:
            state, text = instr, QUESTION_WHEN_EMPTY
        else:
            text = instr
        crit = q.get("criteria") or {}
        if q["type"] == "noul":
            choices = [_option("no", crit.get("false")), _option("yes", crit.get("true"))]
            return state, Question(text, choices, kind="verify"), ["no", "yes"]
        if q["type"] != "choice":
            raise Unsupported(f"Unsupported question type {q['type']}")
        keys = list(crit)
        if len(keys) < 2:
            raise Unsupported("a choice needs at least two options")
        kind = "choose"
        if {k.lower() for k in keys} == {"no", "yes"}:
            keys = sorted(keys, key=lambda k: k.lower() != "no")
            kind = "verify"
        return state, Question(text, [_option(k, crit[k]) for k in keys], kind=kind), keys

    def __call__(self, state, questions):
        st = _text(state) if state not in (None, "", {}) else ""
        answers, used = {}, 0
        for key, q in questions.items():
            s, question, keys = self._question(st, q)
            n_state = self._ntok(self.d._state(s, question))
            if n_state > self.max_state:
                raise Unsupported(f"maximum context length: state is {n_state} tokens, Decima reads {self.max_state}")
            qt = question.text if self.d.cfg.get("question_in_choices", True) else ""
            n_opt = [self._ntok(self.d.cfg.get("choice_prefix", "passage: ") + f"{qt} {c}".strip()) for c in question.choices]
            if max(n_opt) > self.max_choice:
                raise Unsupported(f"maximum context length: an option is {max(n_opt)} tokens, Decima reads {self.max_choice}")
            used += n_state + sum(n_opt)
            p = self.d.decide(s, question).probs
            z = sum(p)
            probs = {k: float(v) / z for k, v in zip(keys, p)}
            if q["type"] == "noul":
                answers[key] = {"type": "noul", "noul": probs["yes"]}
            else:
                answers[key] = {"type": "choice", "choice": max(probs, key=probs.get), "probabilities": probs}
        return {"model": "decima-small", "answers": answers, "usage": {"input_tokens": used}}, None

    def runtime(self):
        return {"load_s": round(self.load_s, 2), **self.provenance}
