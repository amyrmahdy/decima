"""Kev's published kev-0.5b protocol, rebuilt item-for-item.

Source: github.com/jaredpalmer/kev at the v0.1.0 release (commit 2bfec1c, 2026-09-17; code identical to the
first commit d0e2b1f that produced the committed `runs/kev/eval.json`). Numbers: README / MODEL_CARD.md
"Evaluation" (Banking77 0.860, AG News 0.940, BoolQ 0.753, MNLI 0.747, SST-5 0.533, Yelp 0.553,
AG News yes/no 0.960, Yelp yes/no 0.887; "all" 0.799 / ECE 0.065 over 1,350 questions).

Their eval: `python -m kev.evaluate --run runs/kev --n_per_source 150` (seed 1) ->
`reqs = build(150, "test", seed=1)` from `kev/data.py`:

    SOURCES = {"banking77": (_banking, "train", "test"), "boolq": (_boolq, "train", "validation"),
               "agnews": (_agnews, "train", "test"), "mnli": (_mnli, "train", "validation_matched"),
               "sst5": (_sst5, "train", "test"), "yelp": (_yelp, "train", "test")}
    def build(n_per_source, split="train", seed=0, ...):
        rng = random.Random(seed)            # ONE rng shared by all six sources, in this order
        for name, (fn, tr, te) in SOURCES.items(): reqs += fn(te, n_per_source, rng)
        rng.shuffle(reqs)
    def _sample(ds, n, rng): return [ds[i] for i in rng.sample(range(len(ds)), min(n, len(ds)))]

Datasets (HF id, eval split): legacy-datasets/banking77 test, google/boolq validation, fancyzhx/ag_news test,
nyu-mll/multi_nli validation_matched (label >= 0), SetFit/sst5 test, Yelp/yelp_review_full test (text cut
to 220 words). 150 records per source. Each record is a TypeSafe request; the same rng also randomises the
rendering: state wrapped as {"document"} 15% / {"ticket": {channel, body}} 10% / [{"role","content"}] 7%;
instructions as {"question", "focus"} 15%; option descriptions null 30% / {"what"} 10% (Banking77: templated
description, null 50%). Questions (verbatim):

    banking77  choice K=77  "Which banking intent best describes this customer message?"  options = intent ids
    agnews     choice K=4   "What is the topic of this article?"  world/sports/business/scitech + descriptions
    agnews_yn  noul x2      "Is this article about {world news|sports|business|science and technology}?"
    boolq      noul         "<question>?"  40%: criteria {true: "The passage supports a yes answer", false: ...}
    mnli       choice K=3   'Hypothesis: "<h>" How does it relate to the premise?'  (state = premise)
    sst5       score L=5    "What is the sentiment of this review sentence?"  very negative .. very positive
    yelp       score L=5    "How many stars did this reviewer give?"  "1 star: terrible experience" .. "5 stars: excellent"
    yelp_yn    noul         "Would this reviewer recommend the business?"  label = stars >= 4

`kev.api.to_record` renders every request to text: state/instructions through `render()` (dicts become
"key: value" lines), noul options `["no[: desc]", "yes[: desc]"]`, choice options "key" or "key: desc",
score options the level texts. Evaluation (`test_accuracy`) then applies
`augment(r, random.Random(1), p_none=0, p_distract=0)` record by record in shuffled order, which only
permutes Choice option order. We replay every rng call, so `state`, `question` and `choices` here are the
exact strings kev-0.5b read, in the exact option order. Scoring is argmax accuracy; ECE is 10 equal-width
bins on the top probability (`kev.evaluate.ece`).

Mapping to our items: choice -> "choose", score -> "score" (choices = the 5 level texts, in order), noul ->
"verify" with choices [yes-option, no-option] (Kev's order is [no, yes]; gold 0 = yes). Option text keeps
Kev's rendering ("yes: The passage supports a yes answer", "sports: Sports: games, ...").

Suites: kev/banking77, kev/agnews, kev/boolq, kev/mnli, kev/sst5, kev/yelp (the six per-source numbers) plus
kev/agnews-yn (2 questions per record -> 300 at the default) and kev/yelp-yn, the derived yes/no questions
that are one third of Kev's 1,350-question "all" number.

Deviations:
    * `limit` counts records per source, like Kev's --n_per_source (default 150). limit <= 150 returns the
      first `limit` records of the published sample (a subset of Kev's items); limit > 150 re-runs the chain
      with n = limit, a different sample.
    * calib items are ours (Kev fitted T on the even-indexed half of the eval records). Same construction,
      one `random.Random("kev-calib:<source>")` per source, drawn from a split that is neither eval nor
      Kev's training sample: SST-5 `validation`, MNLI `validation_mismatched`; Banking77 / BoolQ / AG News
      / Yelp `train` minus the exact 1,500 rows kev-0.5b trained on (we replay `build(1500, "train", 0)`).
      Choice option order is permuted as in Kev's eval.
    * Kev's model truncates the state to 384 tokens; items keep the full rendered state.
    * In-distribution: kev-0.5b trained on these six datasets' train splits with these exact templates,
      instructions and option sets; these are held-out rows, not held-out tasks.
"""

from __future__ import annotations

import random
from functools import lru_cache
from typing import Callable

N_PER_SOURCE = 150     # kev.evaluate --n_per_source
EVAL_SEED = 1          # kev.evaluate --seed
TRAIN_N, TRAIN_SEED = 1500, 0   # kev.train --n_per_source 1500 --seed 0 (kev-0.5b)

# ---- verbatim from kev/data.py @ 2bfec1c -------------------------------------------------------------
AG = {"world": "World news: politics, international affairs, conflicts", "sports": "Sports: games, athletes, teams, results",
      "business": "Business: companies, markets, economy, finance", "scitech": "Science and technology: research, gadgets, software, space"}
MNLI = {"entailment": "The hypothesis follows from the premise", "neutral": "The hypothesis may or may not be true given the premise", "contradiction": "The hypothesis contradicts the premise"}
SST5 = ["very negative", "negative", "neutral", "positive", "very positive"]
YELP = ["1 star: terrible experience", "2 stars: poor", "3 stars: average", "4 stars: good", "5 stars: excellent"]
BANK_TEMPLATES = ["Customer asks about {}", "Issue concerning {}", "Request related to {}", "{}"]


def _wrap_state(text, rng):
    r = rng.random()
    if r < 0.15: return {"document": text}
    if r < 0.25: return {"ticket": {"channel": rng.choice(["email", "chat", "web form"]), "body": text}}
    if r < 0.32: return [{"role": "customer", "content": text}]
    return text


def _instr(text, rng):
    return {"question": text, "focus": rng.choice(["Use only the information given.", "Pick the single best fit.", "Consider the whole message."])} if rng.random() < 0.15 else text


def _desc(desc, rng, p_null=0.3, p_struct=0.1):
    r = rng.random()
    if r < p_null: return None
    if r < p_null + p_struct: return {"what": desc}
    return desc


def _sample(ds, n, rng, pool=None):
    # Kev: rng.sample(range(len(ds)), min(n, len(ds))). `pool` (ours, calib only) restricts the candidate rows.
    idx = range(len(ds)) if pool is None else pool
    return [{**ds[i], "_idx": i} for i in rng.sample(idx, min(n, len(idx)))]


def _banking(ds, n, rng, pool=None):
    names = ds.features["label"].names
    out = []
    for ex in _sample(ds, n, rng, pool):
        t = rng.choice(BANK_TEMPLATES)
        crit = {k: _desc(t.format(k.replace("_", " ")), rng, p_null=0.5, p_struct=0.0) for k in names}
        out.append({"idx": ex["_idx"], "state": _wrap_state(ex["text"], rng), "questions": {"intent": {"type": "choice", "instructions": _instr("Which banking intent best describes this customer message?", rng), "criteria": crit, "label": names[ex["label"]], "src": "banking77"}}})
    return out


def _boolq(ds, n, rng, pool=None):
    out = []
    for ex in _sample(ds, n, rng, pool):
        q = {"type": "noul", "instructions": _instr(ex["question"].strip().rstrip("?") + "?", rng), "label": bool(ex["answer"]), "src": "boolq"}
        if rng.random() < 0.4: q["criteria"] = {"true": "The passage supports a yes answer", "false": "The passage supports a no answer or does not say"}
        out.append({"idx": ex["_idx"], "state": _wrap_state(ex["passage"], rng), "questions": {"answer": q}})
    return out


def _agnews(ds, n, rng, pool=None):
    keys = list(AG)
    out = []
    for ex in _sample(ds, n, rng, pool):
        y = keys[ex["label"]]
        qs = {"topic": {"type": "choice", "instructions": _instr("What is the topic of this article?", rng), "criteria": {k: _desc(v, rng) for k, v in AG.items()}, "label": y, "src": "agnews"}}
        for k in rng.sample(keys, 2):
            qs[f"is_{k}"] = {"type": "noul", "instructions": f"Is this article about {AG[k].split(':')[0].lower()}?", "label": k == y, "src": "agnews_yn"}
        out.append({"idx": ex["_idx"], "state": _wrap_state(ex["text"], rng), "questions": qs})
    return out


def _mnli(ds, n, rng, pool=None):
    keys = list(MNLI)
    return [{"idx": ex["_idx"], "state": _wrap_state(ex["premise"], rng), "questions": {"relation": {"type": "choice", "instructions": _instr(f'Hypothesis: "{ex["hypothesis"]}" How does it relate to the premise?', rng), "criteria": {k: _desc(v, rng) for k, v in MNLI.items()}, "label": keys[ex["label"]], "src": "mnli"}}}
            for ex in _sample(ds, n, rng, pool) if ex["label"] >= 0]


def _sst5(ds, n, rng, pool=None):
    return [{"idx": ex["_idx"], "state": _wrap_state(ex["text"], rng), "questions": {"sentiment": {"type": "score", "instructions": _instr("What is the sentiment of this review sentence?", rng), "criteria": list(SST5), "label": ex["label"], "src": "sst5"}}} for ex in _sample(ds, n, rng, pool)]


def _yelp(ds, n, rng, pool=None):
    out = []
    for ex in _sample(ds, n, rng, pool):
        text = " ".join(ex["text"].split()[:220])
        qs = {"rating": {"type": "score", "instructions": _instr("How many stars did this reviewer give?", rng), "criteria": list(YELP), "label": ex["label"], "src": "yelp"},
              "recommend": {"type": "noul", "instructions": "Would this reviewer recommend the business?", "criteria": {"true": "Clearly positive overall", "false": "Negative or mixed"}, "label": ex["label"] >= 3, "src": "yelp_yn"}}
        out.append({"idx": ex["_idx"], "state": _wrap_state(text, rng), "questions": qs})
    return out


# name -> (constructor, HF id, Kev train split, Kev eval split)
SOURCES = {"banking77": (_banking, "legacy-datasets/banking77", "train", "test"), "boolq": (_boolq, "google/boolq", "train", "validation"),
           "agnews": (_agnews, "fancyzhx/ag_news", "train", "test"), "mnli": (_mnli, "nyu-mll/multi_nli", "train", "validation_matched"),
           "sst5": (_sst5, "SetFit/sst5", "train", "test"), "yelp": (_yelp, "Yelp/yelp_review_full", "train", "test")}
# ours: where calib rows come from (never the eval split); None pool = whole split, "train" = minus Kev's training rows
CALIB_SPLIT = {"banking77": "train", "boolq": "train", "agnews": "train", "mnli": "validation_mismatched", "sst5": "validation", "yelp": "train"}


# kev/api.py: render / option_text / to_record
def render(v, indent=0):
    pad = "  " * indent
    if v is None: return ""
    if isinstance(v, (str, int, float, bool)): return str(v)
    if isinstance(v, list): return "\n".join(f"{pad}- {render(x, indent + 1).lstrip()}" for x in v)
    return "\n".join(f"{pad}{k}:\n{render(x, indent + 1)}" if isinstance(x, (dict, list)) else f"{pad}{k}: {render(x)}" for k, x in v.items())


def option_text(name, desc):
    return name if desc is None or desc == "" else f"{name}: {render(desc)}"


def _augment_eval(req, rng):
    """kev.data.augment(req, rng, p_none=0, p_distract=0): replays its rng calls; only permutes Choice options."""
    out = {**req, "questions": {}}
    for qid, q in req["questions"].items():
        if q["type"] != "choice":
            out["questions"][qid] = q; continue
        crit = dict(q["criteria"])
        if len(crit) > 2 and rng.random() < 0:
            pass
        elif rng.random() < 0:
            pass
        keys = list(crit); rng.shuffle(keys)
        out["questions"][qid] = {**q, "criteria": {k: crit[k] for k in keys}}
    return out
# ------------------------------------------------------------------------------------------------------


def _load(repo, split):
    from datasets import load_dataset

    return load_dataset(repo, split=split)


@lru_cache(maxsize=None)
def _chain(n: int, which: str, seed: int) -> dict[str, list[dict]]:
    """kev.data.build(n, which, seed) without the final shuffle: requests per source, in sample order.
    `order` gives each request's position after build's rng.shuffle (the order evaluation walks)."""
    rng = random.Random(seed)
    per, flat = {}, []
    for name, (fn, repo, tr, te) in SOURCES.items():
        reqs = fn(_load(repo, tr if which == "train" else te), n, rng)
        for r in reqs:
            r["src_name"] = name
        per[name] = reqs
        flat += reqs
    rng.shuffle(flat)       # build()'s shuffle; evaluation walks this order with its own Random(seed)
    if which != "train":
        ev = random.Random(seed)
        for pos, r in enumerate(flat):
            aug = _augment_eval(r, ev)
            r["questions"] = aug["questions"]; r["order"] = pos
    return per


@lru_cache(maxsize=None)
def _kev_train_rows() -> dict[str, frozenset]:
    """Row indices kev-0.5b trained on: build(1500, "train", seed=0)."""
    return {name: frozenset(r["idx"] for r in reqs) for name, reqs in _chain(TRAIN_N, "train", TRAIN_SEED).items()}


def _calib_reqs(source: str, n: int) -> list[dict]:
    fn, repo, tr, _ = SOURCES[source]
    split = CALIB_SPLIT[source]
    ds = _load(repo, split)
    pool = None
    if split == tr:
        used = _kev_train_rows()[source]
        pool = [i for i in range(len(ds)) if i not in used]
    rng = random.Random(f"kev-calib:{source}")
    reqs = fn(ds, n, rng, pool)
    ev = random.Random(f"kev-calib-order:{source}")
    return [{**r, "questions": _augment_eval(r, ev)["questions"], "split_name": split} for r in reqs]


def _items(req: dict, src: str, suite: str, split: str, tag: str) -> list[dict]:
    """Every question of `req` whose src == `src`, rendered exactly as kev.api.to_record renders it."""
    out = []
    state = render(req["state"])
    for qid, q in req["questions"].items():
        if q["src"] != src:
            continue
        question = render(q["instructions"])
        if q["type"] == "noul":
            c = q.get("criteria") or {}
            choices = [option_text("yes", c.get("true")), option_text("no", c.get("false"))]
            kind, gold = "verify", 0 if q["label"] else 1
        elif q["type"] == "choice":
            keys = list(q["criteria"])
            choices = [option_text(k, v) for k, v in q["criteria"].items()]
            kind, gold = "choose", keys.index(q["label"])
        else:
            choices = [render(x) for x in q["criteria"]]
            kind, gold = "score", int(q["label"])
        qtag = "" if len([x for x in req["questions"].values() if x["src"] == src]) == 1 else f"-{qid}"
        out.append({"id": f"{suite}/{tag}{req['idx']}{qtag}", "suite": suite, "lang": "en", "kind": kind,
                    "question": question, "choices": choices, "gold": gold, "split": split, "state": state})
    return out


def suite(source: str, src: str, name: str) -> Callable[[int, int], list[dict]]:
    def build(limit: int = N_PER_SOURCE, calib: int = 0) -> list[dict]:
        reqs = _chain(max(limit, N_PER_SOURCE), "eval", EVAL_SEED)[source][:limit]
        items = [it for r in reqs for it in _items(r, src, name, "eval", f"{SOURCES[source][3]}-")]
        if calib > 0:
            per_req = 2 if src == "agnews_yn" else 1
            cal = [it for r in _calib_reqs(source, -(-calib // per_req)) for it in _items(r, src, name, "calib", f"calib-{CALIB_SPLIT[source]}-")]
            items += cal[:calib]
        return items

    return build


SUITES: dict[str, Callable[[int, int], list[dict]]] = {
    "kev/banking77": suite("banking77", "banking77", "kev/banking77"),
    "kev/agnews": suite("agnews", "agnews", "kev/agnews"),
    "kev/boolq": suite("boolq", "boolq", "kev/boolq"),
    "kev/mnli": suite("mnli", "mnli", "kev/mnli"),
    "kev/sst5": suite("sst5", "sst5", "kev/sst5"),
    "kev/yelp": suite("yelp", "yelp", "kev/yelp"),
    "kev/agnews-yn": suite("agnews", "agnews_yn", "kev/agnews-yn"),
    "kev/yelp-yn": suite("yelp", "yelp_yn", "kev/yelp-yn"),
}

# Kev's published kev-0.5b numbers (runs/kev/eval.json @ v0.1.0): accuracy, ECE (10 bins, raw), n questions.
PUBLISHED = {
    "kev/banking77": (0.860, 0.057, 150), "kev/agnews": (0.940, 0.028, 150), "kev/boolq": (0.753, 0.136, 150),
    "kev/mnli": (0.747, 0.100, 150), "kev/sst5": (0.533, 0.121, 150), "kev/yelp": (0.553, 0.118, 150),
    "kev/agnews-yn": (0.960, 0.017, 300), "kev/yelp-yn": (0.887, 0.084, 150),
}
