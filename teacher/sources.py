"""States the label generator re-asks questions about.

Only TRAIN splits of the bench datasets are used — never test or validation, which the
bench evaluates on and fits temperature on respectively. The bench's own loaders are
deliberately not reused here: they are wired to the eval splits, and one misplaced
argument would leak test text into training. Explicit split names, in one place.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

# key → (hf id, config, text col, label col, train split, state lang, domain description)
TRAIN_SPLITS = {
    "banking77/en": ("mteb/banking77", None, "text", "label", "train", "en", "retail banking customer queries"),
    "clinc150/en": ("clinc/clinc_oos", "plus", "text", "intent", "train", "en", "virtual assistant requests across banking, travel, home, and small talk, some out of scope"),
    "massive/en": ("mteb/amazon_massive_intent", "en", "text", "label", "train", "en", "voice assistant commands"),
    "massive/fa": ("mteb/amazon_massive_intent", "fa", "text", "label", "train", "fa", "voice assistant commands"),
    "massive/ar": ("mteb/amazon_massive_intent", "ar", "text", "label", "train", "ar", "voice assistant commands"),
    "massive/ru": ("mteb/amazon_massive_intent", "ru", "text", "label", "train", "ru", "voice assistant commands"),
    "agnews/en": ("fancyzhx/ag_news", None, "text", "label", "train", "en", "news headlines and leads"),
    "sst5/en": ("SetFit/sst5", None, "text", "label", "train", "en", "movie review sentences"),
    "xnli/en": ("facebook/xnli", "en", None, "label", "train", "en", "premise/hypothesis sentence pairs"),
    "xnli/ar": ("facebook/xnli", "ar", None, "label", "train", "ar", "premise/hypothesis sentence pairs"),
    "xnli/ru": ("facebook/xnli", "ru", None, "label", "train", "ru", "premise/hypothesis sentence pairs"),
    "farstail/fa": ("mteb/FarsTail", None, None, "label", "train", "fa", "Persian premise/hypothesis sentence pairs"),
}


def _pretty(label: str) -> str:
    return label.replace("_", " ").replace(".", " ").replace("-", " ").strip()


def train_states(key: str, n: int, seed: int = 0) -> tuple[list[str], list[str], str, str]:
    """→ (states, label names for the source's own catalogue, lang, domain description).

    Label names come along so a "subset of the real catalogue + none of the above"
    question can be built; the gold label itself is not used — the teacher labels."""
    from datasets import load_dataset

    ds_id, cfg, text_col, label_col, split, lang, desc = TRAIN_SPLITS[key]
    ds = load_dataset(ds_id, cfg, split=split) if cfg else load_dataset(ds_id, split=split)
    if text_col is None:   # NLI-style pairs
        cols = ds.column_names
        p = "premise" if "premise" in cols else "sentence1"
        h = "hypothesis" if "hypothesis" in cols else "sentence2"
        texts = [f"Premise: {a}\nHypothesis: {b}" for a, b in zip(ds[p], ds[h])]
        names = []
    else:
        texts = list(ds[text_col])
        feat = ds.features[label_col]
        if hasattr(feat, "names"):
            names = [_pretty(x) for x in feat.names]
        elif "label_text" in ds.column_names:
            names = sorted({_pretty(x) for x in ds["label_text"]})
        else:
            names = sorted({_pretty(str(x)) for x in ds[label_col]})
    rng = random.Random(seed)
    idx = rng.sample(range(len(texts)), min(n, len(texts)))
    return [texts[i] for i in idx], names, lang, desc


def generated_states(path: str | Path, n_per_group: int, seed: int = 0) -> dict[tuple[str, str], list[str]]:
    """Our own generated states, grouped by (domain, state_lang), to be re-asked new questions."""
    groups: dict[tuple[str, str], list[str]] = {}
    p = Path(path)
    if not p.exists():
        return groups
    for line in p.open():
        r = json.loads(line)
        if r.get("source") != "generate":
            continue
        groups.setdefault((r["domain"], r["state_lang"]), []).append(r["state"])
    rng = random.Random(seed)
    for k, v in groups.items():
        v = list(dict.fromkeys(v))
        rng.shuffle(v)
        groups[k] = v[:n_per_group]
    return groups
