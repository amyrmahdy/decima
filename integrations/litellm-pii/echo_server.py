"""A stand-in LLM for trying the guardrail: an OpenAI-compatible /v1/chat/completions that replies with exactly the
messages it received, so you can see what reached "the model". Standard library only.

    python echo_server.py --port 9000        # then point a LiteLLM model at http://127.0.0.1:9000/v1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Echo(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        msgs = body.get("messages", [])
        print("received:", json.dumps(msgs or body.get("input", []), ensure_ascii=False), file=sys.stderr, flush=True)
        seen = "\n".join(f"{m.get('role')}: {m.get('content') if isinstance(m.get('content'), str) else json.dumps(m.get('content'), ensure_ascii=False)}"
                         for m in msgs)
        reply, model = f"The model received:\n{seen}", body.get("model", "echo")
        if body.get("stream"):                       # server-sent events: one content chunk, then a stop chunk
            chunk = lambda delta, stop: "data: " + json.dumps({"id": "echo-1", "object": "chat.completion.chunk", "created": int(time.time()),
                                                               "model": model, "choices": [{"index": 0, "delta": delta, "finish_reason": stop}]},
                                                              ensure_ascii=False) + "\n\n"
            out = (chunk({"role": "assistant", "content": reply}, None) + chunk({}, "stop") + "data: [DONE]\n\n").encode()
            ctype = "text/event-stream"
        else:
            out = json.dumps({"id": "echo-1", "object": "chat.completion", "created": int(time.time()), "model": model,
                              "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": reply}}],
                              "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}, ensure_ascii=False).encode()
            ctype = "application/json"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a) -> None:
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9000)
    port = ap.parse_args().port
    print(f"echo model on http://127.0.0.1:{port}/v1", file=sys.stderr, flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Echo).serve_forever()
