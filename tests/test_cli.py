import io
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest
from conftest import THREADS, needs_models

from decima.cli import bar, main, render

T = ["--threads", str(THREADS)]


def run(capsys, *argv, stdin: str | None = None, monkeypatch=None) -> tuple[int, str, str]:
    if stdin is not None:
        monkeypatch.setattr(sys, "stdin", io.StringIO(stdin))
    code = main(list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err


def test_bar():
    assert bar(0) == " " * 20 and bar(1) == "█" * 20 and len(bar(0.37)) == 20


def test_render_all_types():
    r = {"answers": {
        "team": {"type": "choice", "choice": "billing", "confidence": 0.9, "probabilities": {"billing": 0.93, "tech": 0.05, "a": 0.01, "b": 0.005, "c": 0.005}},
        "urgent": {"type": "noul", "noul": 0.81},
        "anger": {"type": "score", "score": 1.7, "confidence": 0.5, "legend": {"0": "calm", "1": "annoyed", "2": "furious"},
                  "probabilities": {"0": 0.05, "1": 0.2, "2": 0.75}}}}
    text = render(r)
    assert "billing" in text and "… 1 more" in text and "p(yes)" in text and "score 1.70" in text and "2 furious" in text


def test_presets_command(capsys):
    code, out, _ = run(capsys, "presets")
    assert code == 0 and all(p in out for p in ("triage", "agent-gate", "kg-judge", "decima-agent"))
    code, out, _ = run(capsys, "presets", "kg-judge")
    assert code == 0 and "forward_looking" in out and "needs mention_a, mention_b" in out


@needs_models
def test_preset_json(capsys):
    code, out, _ = run(capsys, "My card was charged twice, please refund", "--preset", "triage", "--json", *T)
    r = json.loads(out)
    assert code == 0 and r["model"] == "decima-base" and r["answers"]["category"]["choice"] == "billing"


@needs_models
def test_pretty_output(capsys):
    code, out, _ = run(capsys, "git push --force origin main", "-p", "agent-gate", *T)
    assert code == 0 and "decima-agent · agent-gate" in out and "deny" in out and "█" in out


@needs_models
def test_own_questions(capsys):
    code, out, _ = run(capsys, "Can I get my money back for the annual plan?", "-q", "The customer asks for a refund.", "--json", *T)
    assert code == 0 and json.loads(out)["answers"]["answer"]["noul"] > 0.5
    code, out, _ = run(capsys, "The app crashes when I upload a photo", "-q", "Which team?", "-c", "billing,tech=bugs and crashes,sales", "--json", *T)
    a = json.loads(out)["answers"]["answer"]
    assert a["choice"] == "tech" and list(a["probabilities"]) == ["billing", "tech", "sales"]
    code, out, _ = run(capsys, "THIS IS UNACCEPTABLE, FIX IT NOW", "-q", "How angry is the writer?", "--levels", "calm,annoyed,furious", "--json", *T)
    assert json.loads(out)["answers"]["answer"]["score"] > 1


@needs_models
def test_questions_file_and_stdin(capsys, monkeypatch, tmp_path):
    f = tmp_path / "q.json"
    f.write_text(json.dumps({"bird": {"type": "noul", "instructions": "The text is about birds."}}))
    code, out, _ = run(capsys, "--questions", str(f), "--json", *T, stdin="The heron waded through the marsh.\n", monkeypatch=monkeypatch)
    assert code == 0 and json.loads(out)["answers"]["bird"]["noul"] > 0.5


@needs_models
def test_file_lines(capsys, tmp_path):
    f = tmp_path / "t.txt"
    f.write_text("I love the new update!\n\nMy card was charged twice, refund me.\n")
    code, out, _ = run(capsys, "--file", str(f), "--lines", "-p", "triage", "--json", *T)
    rows = [json.loads(x) for x in out.splitlines()]
    assert code == 0 and [r["answers"]["category"]["choice"] for r in rows] == ["other", "billing"]


@needs_models
def test_json_state_text(capsys):
    code, out, _ = run(capsys, json.dumps({"task": "Fix the typo 'recieve' in the signup email"}), "-p", "agent-gate", "--json", *T)
    assert code == 0 and set(json.loads(out)["answers"]) == {"tier"}


def test_errors(capsys, monkeypatch, tmp_path):
    code, _, err = run(capsys, "hello", "-p", "nope")
    assert code == 2 and "unknown preset" in err
    code, _, err = run(capsys, "hello", "--questions", str(tmp_path / "missing.json"))
    assert code == 2 and "missing.json" in err
    code, _, err = run(capsys, "-p", "triage", stdin="  \n", monkeypatch=monkeypatch)
    assert code == 2 and "empty" in err


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@needs_models
def test_module_entry_and_serve():
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).parents[1])}
    out = subprocess.run([sys.executable, "-m", "decima", "presets"], capture_output=True, text=True, env=env, timeout=60)
    assert out.returncode == 0 and "triage" in out.stdout
    port = _free_port()
    p = subprocess.Popen([sys.executable, "-m", "decima", "serve", "--model", "decima-small", "--port", str(port), "--threads", "1"],
                         env=env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    try:
        assert "System One server" in p.stdout.readline()
        body = json.dumps({"state": "Someone logged into my account from another country.",
                           "questions": {"security": {"type": "noul", "instructions": "This is an account security problem."}}}).encode()
        for _ in range(50):
            try:
                req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/systemone", data=body, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=10) as r:
                    res = json.load(r)
                break
            except OSError:
                time.sleep(0.2)
        assert res["model"] == "decima-small" and res["answers"]["security"]["noul"] > 0.5
    finally:
        p.terminate()
        p.wait(timeout=10)

