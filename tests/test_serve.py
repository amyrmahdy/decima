"""serve.py: 422 for over-long states stays the default; "long", "aggregate", "min_confidence" options."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from bench.batching import make_doc
from decima.serve import make_handler

QS = {"code": {"type": "noul", "instructions": "The document gives a payment deadline."},
      "term": {"type": "choice", "instructions": "Which payment term applies?", "criteria": ["15 days", "30 days", "45 days", "60 days"]},
      "strict": {"type": "score", "instructions": "How strict are the penalties?", "criteria": ["lenient", "moderate", "strict"]}}


@pytest.fixture(scope="module")
def url(base2):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(base2, "decima-test", max_doc_tokens=3000))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/v1/systemone"
    srv.shutdown()


def post(url: str, body: dict) -> tuple[int, dict]:
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_long_state_refused_by_default(url, base2):
    status, body = post(url, {"state": make_doc(base2.tok, 900), "questions": QS})
    assert status == 422 and body["code"] == "STATE_TRUNCATED"
    assert body["detail"][0]["ctx"]["max_state_tokens"] == 512


def test_long_state_chunked(url, base2):
    status, body = post(url, {"state": make_doc(base2.tok, 900), "questions": QS, "long": "chunk"})
    assert status == 200
    for qid, a in body["answers"].items():
        assert a["type"] == QS[qid]["type"]
        w = a["window"]
        assert w["count"] >= 2 and 0 <= w["index"] < w["count"] and w["start"] < w["end"]
        assert "abstain" not in a


def test_chunk_mean_and_cap(url, base2):
    status, body = post(url, {"state": make_doc(base2.tok, 900), "questions": QS, "long": "chunk", "aggregate": "mean"})
    assert status == 200 and set(body["answers"]) == set(QS)
    status, body = post(url, {"state": make_doc(base2.tok, 3200), "questions": QS, "long": "chunk"})
    assert status == 422 and body["code"] == "STATE_TOO_LONG"


def test_short_state_answers_unchanged(url):
    s = "Invoices are payable within 30 days; late delivery costs a 5 percent credit per week."
    status, body = post(url, {"state": s, "questions": QS})
    assert status == 200 and all("window" not in a and "abstain" not in a for a in body["answers"].values())
    status2, body2 = post(url, {"state": s, "questions": QS, "long": "chunk"})
    assert status2 == 200 and body2["answers"] == body["answers"]


def test_min_confidence(url):
    s = "Invoices are payable within 30 days."
    _, body = post(url, {"state": s, "questions": QS, "min_confidence": 1.0})
    assert all(a["abstain"] is True for a in body["answers"].values())
    _, body = post(url, {"state": s, "questions": QS, "min_confidence": 0})
    assert all(a["abstain"] is False for a in body["answers"].values())


@pytest.mark.parametrize("extra", [{"long": "split"}, {"aggregate": "vote"}, {"min_confidence": 2}, {"min_confidence": "high"}])
def test_bad_options(url, extra):
    status, body = post(url, {"state": "hello", "questions": QS, **extra})
    assert status == 422 and body["code"] == "INVALID_REQUEST"
