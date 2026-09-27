"""Laya's own MASSIVE-intent and XNLI protocol, rebuilt item-for-item.

Source: github.com/NandhaKishorM/laya, branch `research`,
`research/scripts/build_benchmark_nb.py` (section 5b). The published numbers
(MASSIVE intent en 0.783 / 0.657, 13 other languages 0.306 / 0.451; XNLI en 0.860 / 0.843,
14 other languages 0.521 / 0.731 for laya / laya-multilingual) are in
`research/results/t4_colab_benchmark.json` and come from exactly this construction.

MASSIVE intent (`mteb/amazon_massive_intent`, config = language, split `test`)
    * rows: the first 300 rows in file order (`list(d)[:300]`), no shuffle, no filtering.
    * label space: `sorted(set(d["label_text"]))` of that language's test split (59 intents;
      label_text is the English intent id in every language).
    * options: gold + 19 distractors. One `random.Random(13)` per language, consumed row by row:
      `rng.sample(pool, 19)` over the sorted labels minus gold, then `rng.shuffle(keys)`, so the
      gold position is random. We replay the same RNG calls, so option sets and order are
      byte-identical to Laya's.
    * option text: Laya passes `criteria = {key: key.replace("_", " ").replace(".", ": ")}` and
      its renderer emits "key: description", e.g. "alarm_set: alarm set". `choices` here are
      those rendered strings, so both models read the same option text.
    * question (choice type): "What is the user asking for in `utterance`?"
    * state: `json.dumps({"utterance": text}, ensure_ascii=False)`, which is what Laya
      serialises a dict state to.
    * languages: en + de fr es pt ru tr ar hi ta zh-CN ja ko sw (13 non-English).

XNLI (`facebook/xnli`, config = language, split `test`)
    * rows: first 300 in file order. 3 options in fixed label order (no shuffle), gold = label
      (0 entailment, 1 neutral, 2 contradiction).
    * options: "entailment: the premise implies the hypothesis is true",
      "neutral: the premise neither implies nor contradicts the hypothesis",
      "contradiction: the premise implies the hypothesis is false".
    * question (choice type): "What is the relationship between `premise` and `hypothesis`?"
    * state: `json.dumps({"premise": p, "hypothesis": h}, ensure_ascii=False)`.
    * languages: en + de fr es ru tr ar hi ur vi th el bg zh sw (14 non-English).

Their metric is plain accuracy (argmax). Temperature does not change it. Laya's headline
"other languages" numbers are the unweighted mean over the per-language suites (see GROUPS).

Where we differ from Laya's code:
    * `limit` is honoured (default 300 = Laya's PER_LANG). Rows and the RNG sequence are the
      same, so limit=300 reproduces their items exactly and a smaller limit gives a prefix.
    * calib items are new (Laya has no calibration split): the first `calib` rows of the
      `validation` split, built the same way with a separate `random.Random(13)` stream so the
      eval items do not depend on `calib`. MASSIVE calib distractors come from the test label
      space; any calib row whose gold intent is not in it is skipped.
    * suite names use ISO codes: MASSIVE "zh-CN" becomes "laya/massive-zh" (lang "zh").
"""

from __future__ import annotations

import json
import random
from typing import Callable

SEED = 13          # Laya's SEED
N_OPTS = 20        # gold + 19 distractors
PER_LANG = 300     # Laya's PER_LANG

MASSIVE_LANGS = ["en", "de", "fr", "es", "pt", "ru", "tr", "ar", "hi", "ta", "zh-CN", "ja", "ko", "sw"]
XNLI_LANGS = ["en", "de", "fr", "es", "ru", "tr", "ar", "hi", "ur", "vi", "th", "el", "bg", "zh", "sw"]

MASSIVE_Q = "What is the user asking for in `utterance`?"
XNLI_Q = "What is the relationship between `premise` and `hypothesis`?"
NLI_CRIT = {
    "entailment": "the premise implies the hypothesis is true",
    "neutral": "the premise neither implies nor contradicts the hypothesis",
    "contradiction": "the premise implies the hypothesis is false",
}
XNLI_CHOICES = ["%s: %s" % (k, v) for k, v in NLI_CRIT.items()]


def _iso(cfg: str) -> str:
    return cfg.split("-")[0].lower()


def _render(key: str) -> str:
    # laya.common.render_options for criteria {key: key.replace("_"," ").replace(".",": ")}
    return "%s: %s" % (key, key.replace("_", " ").replace(".", ": "))


def _massive_items(cfg: str, rows, labels: list[str], n: int, split: str, suite: str) -> list[dict]:
    rng = random.Random(SEED)
    lab = set(labels)
    out = []
    for i, r in enumerate(rows):
        if len(out) >= n:
            break
        gold = r["label_text"]
        if gold not in lab:
            continue
        pool = [x for x in labels if x != gold]
        keys = [gold] + rng.sample(pool, min(N_OPTS - 1, len(pool)))
        rng.shuffle(keys)
        out.append({
            "id": "%s/%s/%d" % (suite, split, i),
            "suite": suite,
            "lang": _iso(cfg),
            "kind": "choose",
            "question": MASSIVE_Q,
            "choices": [_render(k) for k in keys],
            "gold": keys.index(gold),
            "split": split,
            "state": json.dumps({"utterance": r["text"]}, ensure_ascii=False),
        })
    return out


def massive(cfg: str) -> Callable[[int, int], list[dict]]:
    suite = "laya/massive-%s" % _iso(cfg)

    def build(limit: int = PER_LANG, calib: int = 0) -> list[dict]:
        from datasets import load_dataset

        d = load_dataset("mteb/amazon_massive_intent", cfg, split="test")
        labels = sorted(set(d["label_text"]))
        items = _massive_items(cfg, list(d.select(range(min(limit, len(d))))), labels, limit, "eval", suite)
        if calib > 0:
            v = load_dataset("mteb/amazon_massive_intent", cfg, split="validation")
            items += _massive_items(cfg, list(v), labels, calib, "calib", suite)
        return items

    return build


def _xnli_items(cfg: str, rows, split: str, suite: str) -> list[dict]:
    return [{
        "id": "%s/%s/%d" % (suite, split, i),
        "suite": suite,
        "lang": _iso(cfg),
        "kind": "choose",
        "question": XNLI_Q,
        "choices": list(XNLI_CHOICES),
        "gold": int(r["label"]),
        "split": split,
        "state": json.dumps({"premise": r["premise"], "hypothesis": r["hypothesis"]}, ensure_ascii=False),
    } for i, r in enumerate(rows)]


def xnli(cfg: str) -> Callable[[int, int], list[dict]]:
    suite = "laya/xnli-%s" % _iso(cfg)

    def build(limit: int = PER_LANG, calib: int = 0) -> list[dict]:
        from datasets import load_dataset

        d = load_dataset("facebook/xnli", cfg, split="test")
        items = _xnli_items(cfg, d.select(range(min(limit, len(d)))), "eval", suite)
        if calib > 0:
            v = load_dataset("facebook/xnli", cfg, split="validation")
            items += _xnli_items(cfg, v.select(range(min(calib, len(v)))), "calib", suite)
        return items

    return build


SUITES: dict[str, Callable[[int, int], list[dict]]] = {}
for _c in MASSIVE_LANGS:
    SUITES["laya/massive-%s" % _iso(_c)] = massive(_c)
for _c in XNLI_LANGS:
    SUITES["laya/xnli-%s" % _iso(_c)] = xnli(_c)

# How Laya aggregates its headline rows: unweighted mean of per-language accuracy.
GROUPS: dict[str, list[str]] = {
    "laya/massive-en": ["laya/massive-en"],
    "laya/massive-xx": ["laya/massive-%s" % _iso(c) for c in MASSIVE_LANGS if c != "en"],
    "laya/xnli-en": ["laya/xnli-en"],
    "laya/xnli-xx": ["laya/xnli-%s" % _iso(c) for c in XNLI_LANGS if c != "en"],
}
