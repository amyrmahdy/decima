"""A local, TypeSafe-compatible System One server for Decima (standard library only).

    python -m decima.serve                          # amyrmahdy/decima-small, int8, http://127.0.0.1:11436
    python -m decima.serve --model ./export/int8 --port 8000 --threads 2

    export TYPESAFE_BASE_URL=http://127.0.0.1:11436 TYPESAFE_API_KEY=local TYPESAFE_DEFAULT_MODEL=decima-small
    # …and existing TypeSafe SDK code now runs on Decima.

Endpoints: POST /v1/systemone (alias /v1/decisions), GET /v1/models, GET / (liveness). Wire format:
decima/systemone.py. As on TypeSafe's API, a state that would be cut to fit the model's context is an
error (422 STATE_TRUNCATED), not a silent truncation. Errors use the body {"error", "code", "detail"?}.
Set DECIMA_API_KEY to require `Authorization: Bearer <key>`; otherwise any key is accepted.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .runtime import DecimaOnnx
from .systemone import SystemOneError, system_one, to_question

MAX_BODY = 8 * 1024 * 1024


def make_handler(model: DecimaOnnx, name: str):
    lock = threading.Lock()          # one ONNX Runtime session pair, one decision at a time
    key = os.environ.get("DECIMA_API_KEY")

    class H(BaseHTTPRequestHandler):
        server_version = "decima"

        def log_message(self, fmt, *args):
            pass

        def _send(self, status: int, body, ctype: str = "application/json"):
            data = body.encode() if isinstance(body, str) else json.dumps(body, ensure_ascii=False).encode()
            rid = self.headers.get("X-Request-Id") or uuid.uuid4().hex
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Request-Id", rid)
            self.send_header("x-typesafe-request-id", rid)
            self.end_headers()
            self.wfile.write(data)

        def _err(self, status: int, code: str, msg: str, detail: list | None = None):
            body = {"error": msg, "code": code}
            if detail:
                body["detail"] = detail
            self._send(status, body)

        def _authorized(self) -> bool:
            if key and self.headers.get("Authorization", "") != f"Bearer {key}":
                self._err(401, "UNAUTHORIZED", "missing or wrong API key")
                return False
            return True

        def do_GET(self):
            if self.path in ("/", ""):
                return self._send(200, "Decima is running", "text/plain; charset=utf-8")
            if self.path == "/v1/models":
                return self._send(200, {"models": [{"name": name, "description": "Decima-small: open Jev-style decision model (122M, CPU)",
                                                    "release_date": time.strftime("%Y-%m-%d")}]})
            self._err(404, "NOT_FOUND", f"no endpoint {self.path}")

        def do_POST(self):
            if self.path not in ("/v1/systemone", "/v1/decisions"):
                return self._err(404, "NOT_FOUND", f"no endpoint {self.path}")
            if not self._authorized():
                return
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_BODY:
                return self._err(413, "REQUEST_TOO_LARGE", "body over 8 MiB")
            try:
                req = json.loads(self.rfile.read(n) or b"null")
                if not isinstance(req, dict):
                    raise ValueError
            except ValueError:
                return self._err(400, "INVALID_JSON", "body must be a JSON object")
            if "state" not in req:
                return self._err(422, "INVALID_REQUEST", "state: field required",
                                 [{"loc": ["body", "state"], "msg": "Field required", "type": "missing"}])
            state, questions = req["state"], req.get("questions")
            try:
                s = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
                limit = model.cfg["max_state_tokens"]
                for qid, spec in (questions or {}).items():
                    _, q, _ = to_question(qid, spec)
                    if len(model.tok(model._state(s, q))["input_ids"]) > limit:
                        return self._err(422, "STATE_TRUNCATED", f"state: part of state would be dropped to fit the context of {name}",
                                         [{"loc": ["body", "state"], "msg": f"part of state would be dropped to fit the context of {name}",
                                           "type": "state_truncated", "ctx": {"model": name, "max_state_tokens": limit}}])
                with lock:
                    answers = system_one(model, state, questions)
                used = len(model.tok(s)["input_ids"])
            except SystemOneError as e:
                return self._err(422, e.code, str(e), e.issues)
            except Exception:                  # never leak internals
                return self._err(500, "INFERENCE_FAILED", "decision failed")
            self._send(200, {"model": name, "answers": answers, "usage": {"input_tokens": used, "output_tokens": 0}})

    return H


def main() -> None:
    ap = argparse.ArgumentParser(description="TypeSafe-compatible System One server for Decima")
    ap.add_argument("--model", default="amyrmahdy/decima-small", help="HF repo id or local export dir")
    ap.add_argument("--precision", default="int8", choices=["int8", "fp32"])
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=11436)
    ap.add_argument("--name", default="decima-small", help="model name reported in responses")
    a = ap.parse_args()
    model = DecimaOnnx.from_pretrained(a.model, precision=a.precision, threads=a.threads)
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(model, a.name))
    print(f"Decima System One server on http://{a.host}:{a.port}  (POST /v1/systemone)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
