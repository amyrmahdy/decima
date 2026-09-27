"""Decima-small behind JevBench's TypeSafe wire format (`POST /v1/systemone`), CPU only, ONNX int8.

JevBench's stock `typesafe` adapter talks to this server unchanged, the same way the maintainer ran
verdict-small, watt-flash and jeff:

    pip install "git+https://github.com/amyrmahdy/decima"          # or: uv pip install -e .
    python decima_typesafe_server.py --model amyrmahdy/decima-small --precision int8 --threads 1 --port 8942

    # from a JevBench checkout
    python -m jevbench.cli run \
      --tasks datasets/public/easy.jsonl,datasets/public/original.jsonl,datasets/public/hard.jsonl \
      --adapter typesafe --endpoint http://127.0.0.1:8942 --key-env '' --model decima-small \
      --cost-basis self_hosted_cpu --reserve-usd 0 \
      --results out/decima-small-public.jsonl --raw-dir /tmp/decima-raw

Request  {"state": str | object, "model"?: str, "questions": {qid: {"type", "instructions", "criteria"?}}}
Response {"model", "answers": {qid: answer}, "usage": {"input_tokens", "output_tokens": 0}, "runtime": {...}}

Mapping (fixed before any run; identical to the mapping behind our self-measured public numbers):
  state      str as given; an object is serialised with json.dumps(ensure_ascii=False)
  question   Question.text = instructions
  noul       kind "verify", choices ["no: <criteria.false>", "yes: <criteria.true>"] ("no"/"yes" when no criteria)
             -> {"type": "noul", "noul": P(yes)}
  choice     kind "choose", one choice per criteria key "<label with _ as space>: <criterion>"
             (a list of labels, or no criteria, gives the bare labels) -> {"type": "choice", "choice", "probabilities"}
  score      kind "score", choice k = "<k>: <criteria[k]>" (Decima's ordinal head)
             -> {"type": "score", "score": argmax level, "probabilities": {"0": p0, "1": p1, ...}}
All probabilities are Decima's own softmax (ordinal head for score), temperature fixed at export
(decima.json), nothing fitted on JevBench. Decima is option-order invariant by construction, so the order of
criteria keys does not matter.

Input budget: 512 state tokens and 64 tokens per option (the model's trained limits). By default a longer state
is truncated by the tokenizer (keeps the head), exactly what a user of `Decima.decide` gets. `--refuse-over-budget`
answers such requests with HTTP 422 instead (the harness's refusal rule: counted wrong, not an outage).

Several questions in one request are answered one after another against the same state. No network access after
the weights are loaded (`HF_HUB_OFFLINE=1` works once the snapshot is cached, or pass a local export dir).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from decima import Decima, Question

MODEL_ID = "decima-small"


def _pretty(label: str) -> str:
    return str(label).replace("_", " ").strip()


def state_text(state) -> str:
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def to_question(q: dict) -> tuple[Question, list[str]]:
    """JevBench/TypeSafe question -> (Decima Question, answer keys aligned with its choices)."""
    t, text, crit = q.get("type"), str(q.get("instructions") or ""), q.get("criteria")
    if t == "noul":
        crit = crit if isinstance(crit, dict) else {}
        choices = [f"no: {crit['false']}" if crit.get("false") else "no",
                   f"yes: {crit['true']}" if crit.get("true") else "yes"]
        return Question(text, choices, kind="verify"), ["no", "yes"]
    if t == "choice":
        if isinstance(crit, dict) and crit:
            keys = [str(k) for k in crit]
            return Question(text, [f"{_pretty(k)}: {crit[k]}" if crit[k] else _pretty(k) for k in keys], "choose"), keys
        if isinstance(crit, list) and crit:
            keys = [str(k) for k in crit]
            return Question(text, [_pretty(k) for k in keys], "choose"), keys
        raise ValueError("choice question needs criteria (a label->description map or a list of labels)")
    if t == "score":
        if not isinstance(crit, list) or len(crit) < 2:
            raise ValueError("score question needs criteria as a list of at least two levels")
        return Question(text, [f"{k}: {c}" for k, c in enumerate(crit)], "score"), [str(k) for k in range(len(crit))]
    raise ValueError(f"unsupported question type {t!r}")


class Engine:
    def __init__(self, model: str, precision: str, threads: int, refuse_over_budget: bool):
        t0 = time.perf_counter()
        self.d = Decima.from_pretrained(model, precision=precision, threads=threads)
        self.load_s = time.perf_counter() - t0
        self.model, self.precision, self.threads, self.refuse = model, precision, threads, refuse_over_budget
        self.max_state = int(self.d.cfg["max_state_tokens"])
        self.max_choice = int(self.d.cfg["max_choice_tokens"])

    def _n_tokens(self, text: str) -> int:
        return len(self.d.tok(text, truncation=False)["input_ids"])

    def answer(self, body: dict) -> tuple[int, dict]:
        questions = body.get("questions")
        if not isinstance(questions, dict) or not questions:
            return 400, {"error": "questions must be a non-empty object"}
        state = state_text(body.get("state", ""))
        answers, used, truncated = {}, 0, False
        for qid, q in questions.items():
            try:
                question, keys = to_question(q)
            except (ValueError, TypeError) as e:
                return 400, {"error": f"{qid}: {e}"}
            n_state = self._n_tokens(self.d._state(state, question))
            if n_state > self.max_state:
                if self.refuse:
                    # "maximum context length" is a capacity marker both JevBench (422 = refusal) and the
                    # Decision Index http engine (-> Unsupported) recognise
                    return 422, {"error": f"{qid}: maximum context length exceeded: state is {n_state} tokens, "
                                          f"Decima reads {self.max_state}"}
                truncated = True
            used += min(n_state, self.max_state) + sum(min(self._n_tokens(c), self.max_choice) for c in question.choices)
            p = self.d.decide(state, question).probs
            probs = {k: float(v) for k, v in zip(keys, p)}
            top = max(probs, key=probs.get)
            if q["type"] == "noul":
                answers[qid] = {"type": "noul", "noul": probs["yes"]}
            elif q["type"] == "choice":
                answers[qid] = {"type": "choice", "choice": top, "probabilities": probs}
            else:
                answers[qid] = {"type": "score", "score": int(top), "probabilities": probs}
        return 200, {"model": MODEL_ID, "answers": answers,
                     "usage": {"input_tokens": used, "output_tokens": 0},
                     "runtime": {"weights": self.model, "precision": self.precision, "threads": self.threads,
                                 "device": "cpu", "state_truncated": truncated,
                                 "budget": {"state_tokens": self.max_state, "option_tokens": self.max_choice},
                                 "probability_origin": "native-softmax"}}


def make_handler(engine: Engine):
    class H(BaseHTTPRequestHandler):
        def _send(self, code: int, obj: dict):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.rstrip("/") in ("", "/health", "/v1/models"):
                return self._send(200, {"status": "ok", "model": MODEL_ID, "load_s": round(engine.load_s, 2)})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path.rstrip("/") != "/v1/systemone":
                return self._send(404, {"error": "POST /v1/systemone"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            except json.JSONDecodeError as e:
                return self._send(400, {"error": f"bad JSON: {e}"})
            try:
                code, out = engine.answer(body)
            except Exception as e:  # noqa: BLE001 - report, never crash the server
                code, out = 500, {"error": f"{type(e).__name__}: {str(e)[:300]}"}
            self._send(code, out)

        def log_message(self, *a):  # quiet: one line per request is noise at 20 ms/decision
            pass

    return H


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="amyrmahdy/decima-small",
                    help="HF repo (onnx/<precision>/ is downloaded) or a local export dir")
    ap.add_argument("--precision", default="int8", choices=["int8", "fp32"])
    ap.add_argument("--threads", type=int, default=1, help="ONNX Runtime intra-op threads (1 is fastest at batch 1)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8942)
    ap.add_argument("--refuse-over-budget", action="store_true",
                    help="HTTP 422 for a state over 512 tokens instead of truncating it")
    a = ap.parse_args(argv)
    engine = Engine(a.model, a.precision, a.threads, a.refuse_over_budget)
    print(f"[decima] {a.model} {a.precision} threads={a.threads} loaded in {engine.load_s:.1f}s; "
          f"POST http://{a.host}:{a.port}/v1/systemone", file=sys.stderr, flush=True)
    HTTPServer((a.host, a.port), make_handler(engine)).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
