"""Every benchmark is reshaped into the same thing Decima sees at inference:

    state (text) + question + choices (natural language) → gold choice index

Label names are turned into readable text ("alarm_set" → "alarm set") because the
model is meant to match *descriptions*, never label ids. MASSIVE fa/ar/ru keep their
English intent names on purpose: that is the cross-lingual case (foreign state,
English choice set) agents hit in practice.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from decima.types import Question

SEED = 0


@dataclass
class Task:
    name: str
    lang: str
    question: Question
    states: list[str]
    gold: list[int]
    calib_states: list[str]        # held-out slice used only to fit temperature
    calib_gold: list[int]

    @property
    def n_choices(self) -> int:
        return len(self.question.choices)


_RENAME = {"oos": "none of the above (out of scope)"}


def _pretty(label: str) -> str:
    label = _RENAME.get(label, label)
    return label.replace("_", " ").replace(".", " ").replace("-", " ").strip()


def _split(rows: list[tuple[str, int]], limit: int, calib: int) -> tuple[list, list, list, list]:
    rng = random.Random(SEED)
    rows = rows[:]
    rng.shuffle(rows)
    ev, ca = rows[:limit], rows[limit : limit + calib]
    return [r[0] for r in ev], [r[1] for r in ev], [r[0] for r in ca], [r[1] for r in ca]


def _from_hf(name, lang, question_text, kind, ds_id, config, text_col, label_col, split, calib_split, limit, calib, choices=None):
    from datasets import load_dataset

    ds = load_dataset(ds_id, config, split=split) if config else load_dataset(ds_id, split=split)
    cds = load_dataset(ds_id, config, split=calib_split) if config else load_dataset(ds_id, split=calib_split)
    feat = ds.features[label_col]
    if choices is None:
        if hasattr(feat, "names"):
            names = feat.names
            choices = [_pretty(n) for n in names]
            to_idx = lambda v: int(v)
        elif "label_text" in ds.column_names and feat.dtype.startswith("int"):  # int labels + parallel text column (mteb/banking77)
            m = {int(i): t for i, t in zip(ds[label_col], ds["label_text"])}
            m.update({int(i): t for i, t in zip(cds[label_col], cds["label_text"])})
            choices = [_pretty(m[i]) for i in range(max(m) + 1)]
            to_idx = lambda v: int(v)
        else:  # string labels → vocabulary is the union across splits (MASSIVE test lacks one intent)
            names = sorted(set(ds[label_col]) | set(cds[label_col]))
            choices = [_pretty(n) for n in names]
            lut = {n: i for i, n in enumerate(names)}
            to_idx = lambda v: lut[v]
    else:
        to_idx = lambda v: int(v)

    rows = [(t, to_idx(l)) for t, l in zip(ds[text_col], ds[label_col])]
    crow = [(t, to_idx(l)) for t, l in zip(cds[text_col], cds[label_col])]
    rng = random.Random(SEED)
    rng.shuffle(rows); rng.shuffle(crow)
    return Task(
        name=name, lang=lang,
        question=Question(text=question_text, choices=choices, kind=kind, lang=lang),
        states=[r[0] for r in rows[:limit]], gold=[r[1] for r in rows[:limit]],
        calib_states=[r[0] for r in crow[:calib]], calib_gold=[r[1] for r in crow[:calib]],
    )


def _nli(name, lang, ds_id, config, split, calib_split, limit, calib, binary: bool):
    """NLI → verify/choose. State carries both sentences; choices are the relations."""
    from datasets import load_dataset

    def load(s):
        ds = load_dataset(ds_id, config, split=s) if config else load_dataset(ds_id, split=s)
        cols = ds.column_names
        p = "premise" if "premise" in cols else "sentence1"
        h = "hypothesis" if "hypothesis" in cols else "sentence2"
        l = "label" if "label" in cols else "labels"
        return [(f"Premise: {a}\nHypothesis: {b}", int(c)) for a, b, c in zip(ds[p], ds[h], ds[l])]

    if binary:   # mteb/FarsTail: 0 = not entailed, 1 = entailed
        choices, kind = ["the hypothesis does not follow from the premise", "the hypothesis follows from the premise"], "verify"
    else:        # xnli: 0 entail, 1 neutral, 2 contradiction
        choices, kind = ["entailment", "neutral", "contradiction"], "choose"
    rows, crow = load(split), load(calib_split)
    rng = random.Random(SEED)
    rng.shuffle(rows); rng.shuffle(crow)
    return Task(
        name=name, lang=lang,
        question=Question(text="Does the hypothesis follow from the premise?", choices=choices, kind=kind, lang=lang),
        states=[r[0] for r in rows[:limit]], gold=[r[1] for r in rows[:limit]],
        calib_states=[r[0] for r in crow[:calib]], calib_gold=[r[1] for r in crow[:calib]],
    )


# name → loader(limit, calib). Ordered roughly by cardinality so the table reads well.
REGISTRY = {
    "sst5/en":       lambda L, C: _from_hf("sst5", "en", "What is the sentiment of this review?", "score",
                                           "SetFit/sst5", None, "text", "label", "test", "validation", L, C,
                                           choices=["very negative", "negative", "neutral", "positive", "very positive"]),
    "agnews/en":     lambda L, C: _from_hf("agnews", "en", "What is the topic of this news article?", "choose",
                                           "fancyzhx/ag_news", None, "text", "label", "test", "train", L, C),
    "xnli/en":       lambda L, C: _nli("xnli", "en", "facebook/xnli", "en", "test", "validation", L, C, binary=False),
    "xnli/ar":       lambda L, C: _nli("xnli", "ar", "facebook/xnli", "ar", "test", "validation", L, C, binary=False),
    "xnli/ru":       lambda L, C: _nli("xnli", "ru", "facebook/xnli", "ru", "test", "validation", L, C, binary=False),
    "farstail/fa":   lambda L, C: _nli("farstail", "fa", "mteb/FarsTail", None, "test", "test", L, C, binary=True),
    "massive/en":    lambda L, C: _from_hf("massive", "en", "What does the user want the assistant to do?", "choose",
                                           "mteb/amazon_massive_intent", "en", "text", "label", "test", "validation", L, C),
    "massive/fa":    lambda L, C: _from_hf("massive", "fa", "What does the user want the assistant to do?", "choose",
                                           "mteb/amazon_massive_intent", "fa", "text", "label", "test", "validation", L, C),
    "massive/ar":    lambda L, C: _from_hf("massive", "ar", "What does the user want the assistant to do?", "choose",
                                           "mteb/amazon_massive_intent", "ar", "text", "label", "test", "validation", L, C),
    "massive/ru":    lambda L, C: _from_hf("massive", "ru", "What does the user want the assistant to do?", "choose",
                                           "mteb/amazon_massive_intent", "ru", "text", "label", "test", "validation", L, C),
    "banking77/en":  lambda L, C: _from_hf("banking77", "en", "What is the customer asking about?", "choose",
                                           "mteb/banking77", None, "text", "label", "test", "train", L, C),
    "clinc150/en":   lambda L, C: _from_hf("clinc150", "en", "What is the user's intent? (may be out of scope)", "choose",
                                           "clinc/clinc_oos", "plus", "text", "intent", "test", "validation", L, C),
}
