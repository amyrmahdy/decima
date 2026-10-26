"""decima.pii: span decoding, post-filter and windowing (no model needed), then the model itself when a local export
(export/pii2, DECIMA_PII_EXPORT) or a cached amyrmahdy/decima-pii is available, and parity with the standalone
release/hf/decima-pii/decima_pii.py on 50 rows of data/pii/test.jsonl (private data; skipped when absent)."""

from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
from pathlib import Path

import pytest
from conftest import ROOT

from decima.pii import DecimaPII, decode, merge_windows, postfilter, redact_spans, window_starts

THREADS = 2


# --- decoding --------------------------------------------------------------------------------------------------------

def test_decode_bridges_space_tokens_inside_an_entity():
    text = "card 5185 5272 2255 5565 ok"
    # the tokenizer gives bare space tokens before ASCII digit groups; they are tagged O but stay inside the entity
    offsets = [(0, 4), (4, 5), (5, 9), (9, 10), (10, 14), (14, 15), (15, 19), (19, 20), (20, 24), (24, 27)]
    tags = ["O", "O", "B-CARD", "O", "I-CARD", "O", "I-CARD", "O", "I-CARD", "O"]
    assert decode(text, offsets, tags) == [{"start": 5, "end": 24, "label": "CARD"}]


def test_decode_new_entity_on_B_or_type_change_and_trims_punctuation():
    text = "«Sara», 0912 123 4567."
    offsets = [(0, 1), (1, 5), (5, 6), (6, 7), (7, 8), (8, 12), (12, 13), (13, 16), (16, 17), (17, 21), (21, 22)]
    tags = ["B-PERSON_NAME", "I-PERSON_NAME", "I-PERSON_NAME", "O", "O", "B-PHONE", "O", "I-PHONE", "O", "I-PHONE", "I-PHONE"]
    assert decode(text, offsets, tags) == [{"start": 1, "end": 5, "label": "PERSON_NAME"},
                                           {"start": 8, "end": 21, "label": "PHONE"}]
    # an I- tag of another type starts a new span; a non-space O token ends the current one
    assert decode("ab cd", [(0, 2), (3, 5)], ["B-PERSON_NAME", "I-ADDRESS"]) == [
        {"start": 0, "end": 2, "label": "PERSON_NAME"}, {"start": 3, "end": 5, "label": "ADDRESS"}]
    assert decode("ab x cd", [(0, 2), (3, 4), (5, 7)], ["B-ADDRESS", "O", "I-ADDRESS"]) == [
        {"start": 0, "end": 2, "label": "ADDRESS"}, {"start": 5, "end": 7, "label": "ADDRESS"}]


def test_decode_skips_special_tokens_and_drops_empty_spans():
    assert decode("x", [(0, 0), (0, 1), (1, 1)], ["B-PHONE", "O", "B-PHONE"]) == []
    assert decode(" . ", [(0, 3)], ["B-PERSON_NAME"]) == []          # trimmed to nothing


def test_postfilter():
    t = "call 123 or 0912 345 6789, mail a.b at x, Ali in Tehran"
    spans = [{"start": 5, "end": 8, "label": "PHONE"},          # 3 digits: dropped
             {"start": 12, "end": 25, "label": "PHONE"},         # kept
             {"start": 32, "end": 35, "label": "EMAIL"},         # no '@': dropped
             {"start": 42, "end": 43, "label": "PERSON_NAME"},   # one character: dropped
             {"start": 43, "end": 45, "label": "PERSON_NAME"},   # starts inside "Ali": dropped
             {"start": 49, "end": 55, "label": "ADDRESS"}]       # kept
    assert [s["start"] for s in postfilter(t, spans)] == [12, 49]
    # Persian / Arabic-Indic digits count as digits
    assert postfilter("۰۹۱۲۳۴۵۶۷۸۹", [{"start": 0, "end": 11, "label": "PHONE"}]) != []
    assert postfilter("٠٥٥١٢٣٤٥٦٧", [{"start": 0, "end": 10, "label": "PHONE"}]) != []


def test_postfilter_allows_a_one_letter_clitic_before_a_name():
    t = "جاء ومحمد"                                               # و + محمد: the name starts after the clitic
    assert postfilter(t, [{"start": 5, "end": 9, "label": "PERSON_NAME"}]) != []
    t2 = "جاء سومحمد"                                            # inside a longer word: dropped
    assert postfilter(t2, [{"start": 6, "end": 10, "label": "PERSON_NAME"}]) == []


def test_redact_spans():
    t = "Ali, 0912 345 6789"
    spans = [{"start": 0, "end": 3, "label": "PERSON_NAME"}, {"start": 5, "end": 18, "label": "PHONE"}]
    assert redact_spans(t, spans) == "[PERSON_NAME], [PHONE]"
    assert redact_spans(t, spans, "<{label}:{text}>") == "<PERSON_NAME:Ali>, <PHONE:0912 345 6789>"
    assert redact_spans(t, []) == t


# --- windowing -------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("n,w,stride", [(255, 254, 192), (300, 254, 192), (1000, 254, 192), (17, 5, 3), (11, 5, 5)])
def test_windows_cover_every_token(n, w, stride):
    starts = window_starts(n, w, stride)
    assert starts[0] == 0 and starts[-1] == n - w
    assert all(b - a <= w for a, b in zip(starts, starts[1:]))       # no gaps
    assert set().union(*(range(s, s + w) for s in starts)) == set(range(n))


def test_merge_windows_takes_the_most_central_window():
    n, w, stride = 12, 6, 3
    calls = []

    def tag_fn(a, b):                                    # tag = the window's start, so we can see which window won
        calls.append((a, b))
        return [a] * (b - a)

    pred = merge_windows(n, w, stride, tag_fn)
    assert calls == [(0, 6), (3, 9), (6, 12)]
    # token 0-2: only window 0 or window 0 most central; token 4 is central in window 3 (k=1 vs k=4 -> min 1 vs 1: first wins)
    for i, src in enumerate(pred):
        best = max((min(i - s, w - 1 - (i - s)), -k) for k, s in enumerate((0, 3, 6)) if s <= i < s + w)
        assert src == (0, 3, 6)[-best[1]]


# --- the model -------------------------------------------------------------------------------------------------------

def _model_dir() -> Path | None:
    for p in (os.environ.get("DECIMA_PII_EXPORT"), ROOT / "export" / "pii2"):
        if p and (Path(p) / "onnx" / "model_nbits8.onnx").exists():
            return Path(p)
    return None


@pytest.fixture(scope="module")
def pii():
    p = _model_dir()
    if p is not None:
        return DecimaPII.from_pretrained(p, threads=THREADS)
    try:
        return DecimaPII.from_pretrained(local_files_only=True, threads=THREADS)
    except Exception as e:                               # noqa: BLE001 - not cached, offline
        pytest.skip(f"decima-pii not available locally ({type(e).__name__}); set DECIMA_PII_EXPORT")


def test_model_finds_arabic_persian_english(pii):
    t = "اسمي محمود الشمري ورقم جوالي ٠٥٥١٢٣٤٥٦٧، وایمیل من sara.ahmadi@gmail.com است"
    got = {(s["label"], s["text"]) for s in pii.find(t)}
    assert ("PHONE", "٠٥٥١٢٣٤٥٦٧") in got and ("EMAIL", "sara.ahmadi@gmail.com") in got
    assert any(lab == "PERSON_NAME" and "الشمري" in v for lab, v in got)
    red = pii.redact(t)
    assert "٠٥٥١٢٣٤٥٦٧" not in red and "[PHONE]" in red and "[EMAIL]" in red
    assert pii.find("") == [] and pii.find("The weather is nice today.") == []
    assert pii("Call 0912 345 6789") == pii.find("Call 0912 345 6789")


def test_model_long_text_is_windowed(pii):
    filler = "This sentence carries no personal data at all. " * 60
    t = filler + "Write to omid.karimi@example.com please. " + filler + "Phone: +98 912 345 6789"
    assert len(pii.tok.encode(t, add_special_tokens=False).ids) > 3 * pii.window
    got = {(s["label"], s["text"]) for s in pii.find(t)}
    assert ("EMAIL", "omid.karimi@example.com") in got
    assert any(lab == "PHONE" and "912 345 6789" in v for lab, v in got)
    for s in pii.find(t):
        assert t[s["start"]:s["end"]] == s["text"]


def test_cli_pii(pii, capsys, monkeypatch):
    from decima.cli import main

    args = ["pii", "-m", str(pii.path), "-t", str(THREADS)]
    assert main(args + ["--json", "Ali Rezaei, 0912 345 6789"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["redacted"] == "[PERSON_NAME], [PHONE]" and {s["label"] for s in out["spans"]} == {"PERSON_NAME", "PHONE"}
    monkeypatch.setattr(sys, "stdin", io.StringIO("ایمیل من sara@example.com است\n"))
    assert main(args + ["--redact-only"]) == 0
    assert capsys.readouterr().out == "ایمیل من [EMAIL] است\n"


STANDALONE = ROOT / "release" / "hf" / "decima-pii" / "decima_pii.py"
TEST_DATA = ROOT / "data" / "pii" / "test.jsonl"


@pytest.mark.skipif(not (STANDALONE.exists() and TEST_DATA.exists()), reason="standalone script or data/pii/test.jsonl not present")
def test_parity_with_standalone_script(pii):
    spec = importlib.util.spec_from_file_location("decima_pii_standalone", STANDALONE)
    ref_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref_mod)
    ref = ref_mod.DecimaPII(pii.path, threads=THREADS)
    rows = [json.loads(line) for line in TEST_DATA.read_text().splitlines()[:50]]
    assert len(rows) == 50
    n_spans = 0
    for r in rows:
        ours = pii.find(r["text"])
        assert ours == ref.find(r["text"]), r["id"]
        assert pii.redact(r["text"]) == ref.redact(r["text"])
        n_spans += len(ours)
    assert n_spans > 50                                  # the rows do contain personal data
