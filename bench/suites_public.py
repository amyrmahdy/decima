"""Public benchmark suites, reshaped into Decima items for a fair scoreboard.

    from bench.suites_public import SUITES
    items = SUITES["btzsc/agnews"](limit=500, calib=0)

Every loader is ``fn(limit, calib) -> list[dict]``: up to ``limit`` eval items, then up to
``calib`` calibration items. One dict per decision:

    {"id", "suite", "lang", "kind", "question", "choices", "gold", "split", "state"}
    + optional "gold_probs" (soft gold aligned with choices) and "meta" (tier/family/workflow …)

``calib`` items come only from a split the benchmark does not score. BTZSC and JevBench
have no such split, so they return **no** calib items whatever ``calib`` says: never fit a
temperature on their test items. Report raw ECE there, or carry a temperature fitted
elsewhere. Sampling is deterministic (seed 0) whenever ``limit`` is below the suite size.


A. BTZSC (ICLR 2026, arXiv 2603.11991)
--------------------------------------
Paper https://openreview.net/forum?id=IxMryAz2p3 · data https://huggingface.co/datasets/btzsc/btzsc
· harness https://github.com/IliasAarab/btzsc · board https://huggingface.co/spaces/btzsc/btzsc-leaderboard

22 English single-label datasets, one ``test`` split each, stored as (text, hypothesis,
labels∈{0,1}) pairs, one positive per text. The harness (``btzsc.data``) regroups
consecutive rows into (text, [hypotheses], gold). Models see only text + the hypotheses
("This example news text is about sports"), so **the hypotheses are the choices here,
verbatim and in the dataset's order**. Metric: macro-F1 per dataset (sklearn
``average="macro"``, ``zero_division=0``), unweighted mean over the 22; accuracy and
macro precision/recall are secondary. Strictly zero-shot: no training or tuning on the
evaluation datasets.

    sentiment  amazonpolarity 10000·2  imdb 10000·2  appreviews 4000·2  yelpreviews 10000·2
               rottentomatoes 1066·2  financialphrasebank 690·3
    emotion    emotiondair 2000·6  empathetic 2542·32
    intent     banking77 3080·72  biasframes_intent 3648·2  massive 2974·59
    topic      agnews 7600·4  yahootopics 45002·10  trueteacher 8955·2  manifesto 17018·56
               capsotu 3355·21  biasframes_offensive 3838·2  biasframes_sex 4404·2
               wikitoxic_toxicaggregated 10000·2  wikitoxic_obscene 8691·2
               wikitoxic_threat 5211·2  wikitoxic_insult 8427·2          (samples·classes)

Replicated harness details: ``limit`` below the dataset size uses the harness's own
``max_samples`` draw (groups 0 and 1, then ``random.Random(0).sample(range(2, N), limit-2)``),
so a subset here is the same subset ``btzsc --max-samples`` scores. yahootopics drops every
text whose gold is "Business & Finance" except the first two groups (harness quirk, kept).

Deviations:
  * Unlabelled groups are skipped: 200 banking77 texts (gold is one of the 5 intents with
    no hypothesis: 77 intents, 72 hypotheses) and 169 massive texts have no positive row.
    The harness keeps them with gold=0, which makes them near-certain errors for everyone.
    Set ``BTZSC_HARNESS_EXACT = True`` to keep them with gold 0 and match the published setup
    exactly. So banking77 and massive can return slightly fewer than ``limit`` items.
  * Decima needs a question; every BTZSC item gets the same neutral one (``BTZSC_QUESTION``).
    It adds no label information. Harness baselines get no question at all.
  * kind is "choose", except financialphrasebank ("score": negative < neutral < positive).
    Binary sets are two contrasting hypotheses, not yes/no propositions, so they stay "choose".
  * No calib split (the benchmark is test-only).

Published macro-F1, mean over the 22 (computed from the harness's packaged baselines,
``btzsc/_baselines/f1_scores.csv``, which seed the leaderboard):
    Qwen3-Reranker-8B 0.722 · Mistral-Nemo-Instruct 0.670 · Qwen3-8B 0.665 · Qwen3-4B 0.649
    gte-large-en-v1.5 0.617 · Qwen3-Reranker-0.6B 0.606 · e5-large-v2 0.597 · e5-base-v2 0.597
    deberta-v3-large-nli-triplet 0.596 (best NLI cross-encoder) · deberta-v3-large-nli 0.591
    Qwen3-Embedding-0.6B 0.580 · bart-large-mnli 0.508 · ms-marco-MiniLM-L6-v2 0.422
    all-MiniLM-L6-v2 0.366 · deberta-v3-large (no NLI) 0.270
Per-dataset numbers for five of them are in ``BTZSC_REFERENCE_F1``.

Overlap with Decima V0 training (TRAIN splits of banking77, MASSIVE, AG News, SST-5,
CLINC150, XNLI, FarsTail): see ``BTZSC_CONTAMINATED``. rottentomatoes is contaminated
outright: 821 of its 1066 texts appear verbatim in SetFit/sst5 train (SST was built from the
same Rotten Tomatoes sentences). banking77, massive and agnews are exactly our own bench's
test splits (3080 / 2974 en / 7600) and share label spaces we trained on, so they are
in-distribution supervised results, not zero-shot. Report a "clean 18" mean next to the
all-22 mean. Exact-match overlap of other BTZSC texts with those train splits: 0.


B. JevBench v1.4 (Benchmark Heaven)
-----------------------------------
Board https://benchmarkheaven.com/jev-models · repo https://github.com/fstandhartinger/jevbench
(MIT) · method docs/METHOD-v1.4.md

534 frozen v1.2 decisions plus 308 sealed v1.4 decisions. **Only 231 of the 534 are public**:
datasets/public/{easy (48), original (72, the "standard" tier, 36 paraphrase pairs), hard (111)}.jsonl.
Not public: heldout 24, easy-heldout 24, hard held-out 109, and the whole 146-item judge
tier (imported, text not redistributable). Files are fetched from the repo pinned at
``JEVBENCH_COMMIT`` into data/public/jevbench/ and checked against datasets/manifest.json.

Item mapping: ``question.type`` noul → "verify" (labels ["no", "yes"], choice text
"no: <false criterion>" / "yes: <true criterion>"); choice → "choose" ("<label>: <criterion>",
in ``labels`` order); score → "score" ("<level>: <criterion>"). ``state`` is the item's
state (35 hard states are JSON objects, serialised with json.dumps). Hard states run 2k–6k
tokens, far past a 512-token encoder, so expect truncation.
10 hard "probability" items carry an exact gold distribution → ``gold_probs``.
meta = {tier, family, group}. Paraphrase pairs share a ``group``; resample by group for CIs.

Scoring (jevbench/composite_v13.py, composite_v14.py):
  Intelligence I13 = Σ_tier w·100·max(0, (acc−chance)/(1−chance)), w = easy .14,
      standard .28, judge .28, hard .30; chance = mean of 1/options per tier (missing tiers
      renormalised). v1.4: I = (0.8·I13 + 0.2·Isealed)·(1 − max(0, gap−25)/100), where
      Isealed = 100·max(0, (a_sealed − 0.293)/0.707) and gap = public-minus-sealed acc in points.
  Calibration C13 = 100·(1 − ECE/0.5) (top-label ECE, 10 equal-width bins, all decisions
      pooled), averaged with 100·(1 − mean TVD to the exact gold distributions) on the hard
      probability items. v1.4 blends in the sealed-inclusive calibration: C = C13 + (C14−C13)·0.571.
      Label-only systems get 0.
  Speed = mean(score(p50), score(p95)), score(s) = 100 − 20·log10(s/0.1 s), serial, network
      included, from Germany. Self-hosted/demo endpoints: s×2 (+0.15 s on their own servers).
  Cost = 100 − 30·log10(USD per 1000 decisions / 0.001). A positive price is mandatory. For
      self-hosted systems they use a hosted-price estimate for the same size class.
  Score = 4/(1/I + 1/C + 1/S + 1/K), then ×(X/50)² for each of I, S, K that is below 50.
  The composite needs sealed and judge items, so it cannot be reproduced locally. From these
  suites you can reproduce public-tier accuracy, a public-only I13 over easy/standard/hard,
  pooled ECE, Brier and hard-tier TVD.

Leaderboard v1.4.0 (scored 2026-09-23): 1 Jev 1.13.0 63.29 (I 53.1 C 76.3 S 83.3 K 52.0,
  $0.040/1k) · 2 JevK5 v0.2.0 62.04 · 3 Hopper 59.43 · 4 Winnow-12B Q8 55.58 · 5 reflex 4B 53.99 ·
  6 djev 52.23 · 7 metask-jev-4b 47.78 · 8 SemIf (Qwen3.5-4B) 47.69 · … small encoders: jeff
  (400M) 30.58, Laya (ModernBERT-large) 30.25 (I 36.1), OpenDecision (ModernBERT-large) 21.64,
  GLiNER2.5-small (74M) 7.19; rerankers bge-reranker-v2-m3 0.17 (I 5.0). Honorable mention
  (not ranked): classifier.dev fast tier 70.82. Jev 1.13.0 tier accuracy: easy 1.000,
  standard 0.990, judge 0.945, hard 0.741; sealed 0.367.
Submission: open an issue on the GitHub repo with a reproducible endpoint or runnable code,
  the exact model and licence, and whether public JevBench items were used in development.
  The maintainer runs every decision (sealed included) serially on their own RunPod pods or
  CPU, or against a production API. Rows from operator APIs carry an "API" exposure flag.
  Self-reported numbers are not ranked.


C. typed-decisions
------------------
Data https://huggingface.co/datasets/LocalLLaMA/typed-decisions (Apache-2.0, synthetic)

4 workflows (agent_trace_observability, customer_service, invoice_processing,
security_incidents). train 1200 cases, test 400 cases, 5 questions per case, so 2,000 test
decisions. Columns: id, workflow, state (JSON), questions (JSON {qid: {type, instructions,
criteria}}), gold (JSON {qid: {label, confidence, probabilities, score?, noul?}}), factors,
label_agreement. Gold is the mean of 3 teacher samples at T=0.7 (a ~4B teacher). It measures
agreement with that teacher, not correctness.

Item mapping: noul → "verify" (["no: <false criterion>", "yes: <true criterion>"]; plain
"no"/"yes" for invoice/duplicate and security/credential_compromise, which ship no criteria); choice → "choose"
("<label>: <criterion>"); score → "score" ("<level>: <criterion>"). question = instructions;
state = compact JSON of the state. gold = discrete gold label; ``gold_probs`` = the teacher
distribution. calib items come from the **train** split (a separate generation run; no shared
ids or states). Fitting a temperature there is a light adaptation to these four workflows, so
report raw ECE too if you claim the zero-shot "generalist" mode.

Scoring (Luni/laya-jev-benchmark bench/eval.py, which reproduces the card's rows):
  acc = argmax == gold label over all 2,000 decisions (noul: p(true) ≥ 0.5)
  soft acc = Σ_k p_k·g_k, Brier = Σ_k (p_k − g_k)² against the soft gold, KL/TV likewise,
      all averaged over **choice + noul decisions only** (score questions are excluded)
  ECE = top-label confidence vs argmax-correct over all decisions
  score MAE = |E[level] − gold expected score|, within-1 = MAE ≤ 1 (score questions)
Reference (test split): uniform 0.308 · prior 0.470 (ECE 0.088) · majority 0.461–0.520 ·
  MiniLM-L6 specialist 0.587 · ModernBERT-base specialist 0.646 · factor ceiling 0.704 ·
  **TypeSafe Jev 1.13.0 0.727** (soft 0.580, Brier 0.148, ECE 0.144, KL 1.442, MAE 0.391, 710 ms/case) ·
  **teacher self-agreement ceiling 0.735** · meraGPT Decider 1 0.768 (card-listed, zero-shot,
  soft 0.608, Brier 0.052, ECE 0.180) · Laya fine-tuned on train (specialist)
  0.766 (convaiinnovations/laya-typed-decisions: soft 0.471, Brier 0.062, ECE 0.213) and
  0.789 (agk4444/laya-typed-decisions: soft 0.513, Brier 0.061, ECE 0.232) · Laya zero-shot 0.362.
  The card warns that specialist (fitted on train) and generalist (zero-shot) numbers are not
  comparable. Decima V0 never saw these workflows, so it is a generalist.


D. Jev Decision Index 0.1 (JDI)
-------------------------------
Board https://huggingface.co/spaces/multimodalart/jev-decision-index (static Space; numbers in
its data/index.json + data/methodology.json, built 2026-09-22) · kit https://github.com/apolinario/decision-index
(MIT, pinned at ``JDI_KIT_COMMIT``). Community-run, not affiliated with TypeSafe.

What it is: a frozen suite of 132,422 requests (117,764 source cases, 775,202 answer fields)
over 37 static benchmarks, reshaped by the lab into Jev's typed-decision API. An engine gets
exactly ``state`` (string, JSON object or empty) and ``questions`` ({key: {"type": "choice",
"instructions", "criteria": {option_key: description}}}, 2–255 options, every question a
choice) and must return, per question, a choice plus a probability for every option. 442
questions are dropped for everyone at scoring time (27 defective: duplicate options/gold; 380
ToolRet/BRIGHT rows longer than nearly every context window; 35 of their sibling chunks).
Sampling: all prepared cases, except case caps (SGD 2500, POP909 2000, cfcolor/ChessBench/
CLadder/ESCI 5000, ESCI stratified locale × relevance) and request caps (RouterBench 10000,
BPoMP 5000), in hash order sha256("20260919:<catalog id>:<group id>").

Hard rules: no truncation (a request over an engine's budget is "unsupported" and scores as
wrong), no option pruning, no per-benchmark prompts, labels never in the payload, transport
retries only, native abstentions stand, **unanswered = wrong** against the full denominator.
A run is ranked only if complete (≥ 131,980 rows) and re-scored from its results file.

Index (``balanced_raw``, the one number shown): each panel benchmark scored with its own
metric (``JDI_BENCHMARKS``) over all scoreable cases, unanswered = 0, so coverage is already
inside the number; the five areas are plain means of their benchmarks; Index = 100 × mean of
the five areas (0.2 each). Panel (19): Knowledge & Reasoning MMLU (subject-macro), GPQA-D,
GSM8K (4/10-choice averaged), CRUXEval, CLadder, ChessBench (ties accepted) · Language
ContractNLI, iSarcasmEval (track A English, sarcastic-class F1), VAST (conservative macro-F1 =
2TP/(2TP+FP+FN+missing)) · Retrieval & Classification BRIGHT (nDCG@10) and Amazon ESCI
(conservative macro-F1) · Tools BFCL (case exact), ToolRet (nDCG@10), RouterBench
(quality of the picked model) · Arts & Human Judgment BPoMP (perturbation-macro), Humicroedit,
POP909 (song-macro), cfcolor (user-macro), Habermas (ties accepted). The other 18 static
benchmarks (BANKING77, CLINC150, SGD, ANLI, ARC-E/C, WinoGrande, HellaSwag, MuSR, SATA, SimpleBench,
HLE, ACOS, FinEntity, NLI4CT, API-Bank, Home appliance, ForecastBench) are display-only. Six
interactive environments are in the lab's frozen 25-panel but no reproduction ran them, so
they are left out for everyone. Hidden variants: ``balanced_skill`` maps each benchmark to
clip((raw − random)/(1 − random)) first; ``breadth_skill`` is a shifted geometric mean of the
area skills. A uniform-random engine that answers everything scores ≈ 27 on the shown index.
Calibration (top-label ECE, Brier, reliability bins over 33 benchmarks, a 1-in-6 request
sample), latency (CUDA-synchronised in-process on 1× RTX PRO 6000; Jev over HTTPS) and cost
(Jev's API bill: $12.26 for the whole suite) are reported per row but **not** in the index.
No bootstrap intervals yet (point estimates).

Leaderboard, Decision Index 0.1 (32 rows; index · ECE · median/p95 ms · params):
   1 Jev 1.13.0 (TypeSafe API) 59.51 · 0.065 · 253/437 HTTP   2 Jevfire (Qwen3.8-27B, decoding) 55.74
   3 JoshuaSP diffusiongemma 26B 55.56   4 Decider 35B-A3B 54.34 · 0.031   5 razorback openjev 51.52
   6 Kev 9B 50.48 · 0.164 · 54/948   7 Solomon v1.1 (27B) 47.51   8 Kev 4B 47.43   9 Kev 8B 46.55
  10 open-jev (pngwn, 4B) 46.15  11 openvons 45.59  12 Decision 1.0 Nox 45.31  13 SemIf (Qwen3.5-4B) 44.77
  14 Decider 2B 44.00  15 mini-jev 43.48  16 Decision 1.0 Sol 40.41  17 Bespoke Nimble 9B 34.21
  18 Kev 0.6B 31.30  19 Kev 0.5B 30.34 · 0.194 · 17/86  20 Qwen-2.5-1B-RLCD 28.81  21 LFM2.5-2.6B-RLCD 27.27
  22 jeff (gliformer 576M) 27.23  23 NanoJev 26.19  24 LFM2.5-350M-RLCD 25.79  25 GLiNER2.5 base 24.70
  26 GLiNER2.5 small (74M) 23.93 · 0.162 · 15/38  27 GLiNER2.5 multi 22.42  28 Decision Lex (mmBERT) 19.57
  29 Decision Kai (mmBERT-base 308M) 18.37  30 system-one-gemma (270M) 17.09
  31 **Laya (421M) 16.39** · 0.131 · 19/75, coverage 51 % (50 % of requests over its instruction,
     option or context budget)  32 Verdict (gliclass 151M) 13.38.
  Jev per-area raw: knowledge .689 language .623 retrieval .370 tools .734 arts .560; pooled
  ECE 0.065 at acc 0.747. Per-benchmark raw scores for a few rows are in ``JDI_REFERENCE``.
  Not on the Space: surogate Rune 26B-A4B self-reports 57.24 (Hub dataset
  surogate/decision-index-results).

Submission (kit README): run the full suite with the kit (``python -m decision_index pipeline``
or ``hf-job`` on one RTX PRO 6000), upload the run directory to a Hub dataset, open a PR on
apolinario/decision-index adding a line to submissions/README.md (model, results dataset with
runs/<name>/scores.json and "complete": true, engine + commit, hardware, declared capacity
limits). An engine is ``Engine.__call__(state, questions) → ({"answers": {key: {"type": "choice",
"choice", "probabilities"}}}, raw)`` and raises ``Unsupported`` for capacity limits.

Data here: the frozen rows (Hub dataset ``JDI_SUITE_DATASET``) are **not public** (404 on
2026-09-23); the loader uses them if they appear (data/public/jdi/selected-rows.jsonl.gz,
hash-checked). Otherwise it rebuilds each benchmark from its pinned public sources with the
kit's own normalizers (``_jdi_rebuild``, niced, 4 threads), checks every normalized file
against the kit manifest's sha256 (the frozen file was built from exactly those bytes), and
keeps the manifest's selected group ids. Verified byte-identical for all 36 non-gated
benchmarks. HLE (cais/hle) is gated: accept its terms on the Hub and set
``JDI_INCLUDE_GATED = True``. Rebuild: ≈ 3 GB of downloads, ≈ 11 GB on disk in data/public/jdi/work (BM25 indexes
included), about 40 min on 4 niced cores, BRIGHT and ToolRet indexing dominate.

Item mapping: one item per (request, question); run id "<catalog>:<track>:<id>" + "/<key>".
choices = option descriptions in ``criteria`` order ("<key>: <description>" when the key
itself carries meaning, e.g. "yes: Invoke this tool."). yes/no questions → "verify" with
choices reordered [no, yes]; FinEntity → "score"; everything else "choose". lang is "ar" for
iSarcasmEval Arabic tracks, en/es/ja for ESCI locales, else "en". Retrieval (ToolRet, BRIGHT):
one "rank" item per query with every scorable candidate as a choice, question = the task
line, state = the query, gold = first judged-relevant candidate; meta.qrels / meta.ideal give
nDCG@10 exactly as the kit computes it. meta = {benchmark, catalog_id, track, group, field,
keys (JDI option keys, aligned with choices), area, in_index, moved} plus subject (MMLU),
variant (BPoMP), song_id (POP909), user_id (cfcolor), locale (ESCI), accepted (tied-best
choice indices: ChessBench, Habermas, RouterBench) and quality (RouterBench per-choice
outcome), so every index metric, including case-exact over ``group``, can be recomputed.
Multi-track benchmarks get one suite per track (jdi/gsm8k-4choice, jdi/isarcasmeval-a-en …);
suites never overlap, so 'jdi/*' is the whole suite once.

Deviations:
  * ``JDI_MOVE_INPUT``: when a request's state is empty (BANKING77, CLINC, MMLU, ANLI, MuSR,
    CLadder, VAST, ARC …, 20 benchmarks) the instructions carry the whole input, so they
    become the state and the question is the neutral ``JDI_QUESTION`` (meta.moved). One
    mechanical rule for every benchmark, nothing dropped or added; set it False for verbatim.
  * Decima truncates at its token budgets; JDI forbids truncation. To be JDI-faithful, count
    an item whose state or choices exceed the budget as unsupported (wrong), not truncated.
  * Retrieval queries with no judged-relevant candidate in the pool are dropped (nDCG 0 for
    every system); scale by kept/total to compare with the board.
  * limit below the suite size takes whole groups in the kit's own hash order (seed
    20260919, same subset as ``decision_index suite sample``), then cuts at limit.
  * calib: none for uncapped benchmarks (every prepared case is scored). For the eight capped
    ones (RouterBench, SGD, BPoMP, POP909, cfcolor, ChessBench, ESCI, CLadder) calib items
    come from the unselected rows of the same pool (never scored by JDI, disjoint groups),
    first ``JDI_POOL_MAX`` rows in hash order. Same upstream split as the eval rows, so it is
    in-distribution calibration.

Overlap with Decima V0 training (TRAIN splits of banking77, MASSIVE, AG News, SST-5, CLINC150,
XNLI; FarsTail has no Persian counterpart here): see ``JDI_CONTAMINATED``. Checked by exact
match (lower-cased, whitespace-folded, ≥ 20 chars) of every state/instruction line and sentence
of all 117,764 selected cases against those train texts. Two benchmarks are contaminated by
design: BANKING77 (3,080) and CLINC150+OOS (5,500) are the very test splits of our own bench,
with the label spaces we trained on (18 BANKING77 test texts also sit verbatim in its train
split, upstream duplicates), so they are in-distribution supervised results, not zero-shot.
Both are display-only, so neither enters the Decision Index. Everything else is clean: the
only other hits are stock phrases ("artificial intelligence", "thanks for the help!",
"house of representatives") in ≤ 13 cases of SGD, VAST, ACOS, SATA, BRIGHT, API-Bank, ToolRet
and ContractNLI. No ANLI premise/hypothesis matches XNLI (MNLI-derived) train, AG News and
SST-5 match nothing, and nothing is Persian or Russian.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import urllib.request
from functools import partial
from pathlib import Path
from typing import Callable

SEED = 0
ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "public"          # data/ is git-ignored

Item = dict
Loader = Callable[[int, int], list[Item]]


def _pretty(label: str) -> str:
    return str(label).replace("_", " ").strip()


def _item(id, suite, kind, question, choices, gold, split, state, lang="en", **extra) -> Item:
    assert 0 <= gold < len(choices), (id, gold, choices)
    if kind == "verify":
        assert len(choices) == 2, id
    d = {"id": id, "suite": suite, "lang": lang, "kind": kind, "question": question,
         "choices": list(choices), "gold": int(gold), "split": split, "state": state}
    d.update({k: v for k, v in extra.items() if v is not None})
    return d


# ── A. BTZSC ─────────────────────────────────────────────────────────────────

BTZSC_DATASETS = [
    "amazonpolarity", "imdb", "appreviews", "yelpreviews", "rottentomatoes", "financialphrasebank",
    "emotiondair", "empathetic",
    "banking77", "biasframes_intent", "massive",
    "agnews", "yahootopics", "trueteacher", "manifesto", "capsotu", "biasframes_offensive",
    "biasframes_sex", "wikitoxic_toxicaggregated", "wikitoxic_obscene", "wikitoxic_threat", "wikitoxic_insult",
]
BTZSC_TASK = {d: t for t, ds in {
    "sentiment": BTZSC_DATASETS[:6], "emotion": BTZSC_DATASETS[6:8],
    "intent": BTZSC_DATASETS[8:11], "topic": BTZSC_DATASETS[11:]}.items() for d in ds}
BTZSC_QUESTION = "Which statement best describes this example?"
BTZSC_SCORE_KIND = {"financialphrasebank"}          # ordered levels: negative < neutral < positive
BTZSC_HARNESS_EXACT = False                         # True: keep unlabelled groups with gold 0, as the harness does

BTZSC_CONTAMINATED = {
    "rottentomatoes": "821/1066 texts verbatim in SetFit/sst5 train (same RT sentences)",
    "banking77": "same test split as our banking77/en; trained on banking77 train + its label space",
    "massive": "same test split as our massive/en; trained on MASSIVE train + its label space",
    "agnews": "same test split as our agnews/en; trained on AG News train + its label space",
}
BTZSC_CLEAN = [d for d in BTZSC_DATASETS if d not in BTZSC_CONTAMINATED]

# macro-F1 per dataset from btzsc/_baselines/f1_scores.csv (harness v. 2026-09, commit on main)
BTZSC_REFERENCE_F1 = {
    "Qwen3-Reranker-0.6B": {"amazonpolarity": 0.912, "imdb": 0.884, "appreviews": 0.887, "yelpreviews": 0.946, "rottentomatoes": 0.783, "financialphrasebank": 0.414, "emotiondair": 0.487, "empathetic": 0.408, "banking77": 0.627, "biasframes_intent": 0.496, "massive": 0.531, "agnews": 0.788, "yahootopics": 0.546, "trueteacher": 0.336, "manifesto": 0.268, "capsotu": 0.529, "biasframes_offensive": 0.571, "biasframes_sex": 0.078, "wikitoxic_toxicaggregated": 0.791, "wikitoxic_obscene": 0.799, "wikitoxic_threat": 0.502, "wikitoxic_insult": 0.741},
    "gte-large-en-v1.5": {"amazonpolarity": 0.952, "imdb": 0.942, "appreviews": 0.907, "yelpreviews": 0.931, "rottentomatoes": 0.873, "financialphrasebank": 0.490, "emotiondair": 0.401, "empathetic": 0.344, "banking77": 0.632, "biasframes_intent": 0.565, "massive": 0.573, "agnews": 0.740, "yahootopics": 0.561, "trueteacher": 0.405, "manifesto": 0.276, "capsotu": 0.549, "biasframes_offensive": 0.467, "biasframes_sex": 0.208, "wikitoxic_toxicaggregated": 0.822, "wikitoxic_obscene": 0.816, "wikitoxic_threat": 0.379, "wikitoxic_insult": 0.752},
    "deberta-v3-large-nli-triplet": {"amazonpolarity": 0.929, "imdb": 0.927, "appreviews": 0.930, "yelpreviews": 0.981, "rottentomatoes": 0.837, "financialphrasebank": 0.789, "emotiondair": 0.423, "empathetic": 0.415, "banking77": 0.238, "biasframes_intent": 0.699, "massive": 0.404, "agnews": 0.832, "yahootopics": 0.276, "trueteacher": 0.413, "manifesto": 0.060, "capsotu": 0.246, "biasframes_offensive": 0.644, "biasframes_sex": 0.273, "wikitoxic_toxicaggregated": 0.823, "wikitoxic_obscene": 0.842, "wikitoxic_threat": 0.437, "wikitoxic_insult": 0.691},
    "e5-base-v2": {"amazonpolarity": 0.926, "imdb": 0.898, "appreviews": 0.927, "yelpreviews": 0.951, "rottentomatoes": 0.837, "financialphrasebank": 0.463, "emotiondair": 0.432, "empathetic": 0.371, "banking77": 0.616, "biasframes_intent": 0.601, "massive": 0.473, "agnews": 0.761, "yahootopics": 0.552, "trueteacher": 0.436, "manifesto": 0.209, "capsotu": 0.529, "biasframes_offensive": 0.587, "biasframes_sex": 0.197, "wikitoxic_toxicaggregated": 0.672, "wikitoxic_obscene": 0.642, "wikitoxic_threat": 0.307, "wikitoxic_insult": 0.739},
    "all-MiniLM-L6-v2": {"amazonpolarity": 0.353, "imdb": 0.340, "appreviews": 0.413, "yelpreviews": 0.328, "rottentomatoes": 0.338, "financialphrasebank": 0.312, "emotiondair": 0.111, "empathetic": 0.154, "banking77": 0.434, "biasframes_intent": 0.465, "massive": 0.334, "agnews": 0.495, "yahootopics": 0.363, "trueteacher": 0.395, "manifesto": 0.150, "capsotu": 0.479, "biasframes_offensive": 0.476, "biasframes_sex": 0.512, "wikitoxic_toxicaggregated": 0.513, "wikitoxic_obscene": 0.502, "wikitoxic_threat": 0.264, "wikitoxic_insult": 0.322},
}


def _btzsc(name: str, limit: int, calib: int) -> list[Item]:
    from datasets import load_dataset

    ds = load_dataset("btzsc/btzsc", name=name, split="test")
    labels: list[int] = ds.data.column("labels").to_pylist()     # arrow → list; Column indexing is slow
    rows = list(range(len(ds)))
    if name == "yahootopics":        # harness: drop gold "Business & Finance" but keep the first 20 rows
        lt = ds.data.column("label_text").to_pylist()
        rows = list(range(min(20, len(ds)))) + [i for i in rows if lt[i] != "Business & Finance"]
    n = next((i for i in range(1, len(rows)) if labels[rows[i]] == labels[rows[0]]), len(rows))
    total = len(rows) // n
    if limit < total:                # the harness's own max_samples draw
        rest = random.Random(SEED).sample(range(2, total), k=max(0, limit - 2)) if limit > 2 else []
        groups = [0, 1][:max(limit, 0)] + rest
    else:
        groups = list(range(total))

    hyp = ds.select(rows[:n])[:]["hypothesis"]
    sel = ds.select([rows[g * n + j] for g in groups for j in range(n)])[:]   # one gather, not one per group
    H, L, T = sel["hypothesis"], sel["labels"], sel["text"]
    kind = "score" if name in BTZSC_SCORE_KIND else "choose"
    suite = f"btzsc/{name}"
    out = []
    for k, g in enumerate(groups):
        if H[k * n:(k + 1) * n] != hyp:            # never observed; guard against silent reorderings
            continue
        pos = [j for j, l in enumerate(L[k * n:(k + 1) * n]) if l == 1]
        if len(pos) != 1 and not (BTZSC_HARNESS_EXACT and not pos):
            continue                               # no hypothesis for this gold (banking77/massive) or multi-label
        out.append(_item(f"{suite}/{g}", suite, kind, BTZSC_QUESTION, hyp, pos[0] if pos else 0, "eval",
                         T[k * n], meta={"task": BTZSC_TASK[name], "contaminated": name in BTZSC_CONTAMINATED}))
    return out                                     # no calib: BTZSC is test-only


# ── B. JevBench ──────────────────────────────────────────────────────────────

JEVBENCH_COMMIT = "2fa63fa3226cb369795525ed011800f57dcbd894"   # v1.4.0 era, 2026-09-23
JEVBENCH_FILES = {   # public file → (tier, sha256 from datasets/manifest.json)
    "easy": ("easy", "231df3c2c8e88a1a8c137ebe85de96ba70fabd330849098ac7b3c52c70b7172b"),
    "original": ("standard", "5c2414edb3006b8bfcb70fda433f0f9ca015759433849f8d3104328a1f7c4180"),
    "hard": ("hard", "89e9e6becb33ed88c1de7d42dcc87531b2fb64cfaef4e1986faf7c37b3f80ebb"),
}
JEVBENCH_TIER_WEIGHTS = {"easy": 0.14, "standard": 0.28, "judge": 0.28, "hard": 0.30}


def _jev_file(name: str) -> Path:
    path = CACHE / "jevbench" / JEVBENCH_COMMIT[:12] / f"{name}.jsonl"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://raw.githubusercontent.com/fstandhartinger/jevbench/{JEVBENCH_COMMIT}/datasets/public/{name}.jsonl"
        with urllib.request.urlopen(url, timeout=60) as r:
            data = r.read()
        if hashlib.sha256(data).hexdigest() != JEVBENCH_FILES[name][1]:
            raise RuntimeError(f"jevbench {name}.jsonl hash mismatch; refusing to cache")
        path.write_bytes(data)
    return path


def _jev_item(r: dict, suite: str) -> Item | None:
    q, labels, exp = r["question"], [str(l) for l in r["labels"]], str(r["expected"])
    t, crit = q["type"], q.get("criteria")
    if t == "noul":        # labels are ["no", "yes"]; criteria keyed "false"/"true"
        crit = crit or {}
        choices = [f"{l}: {crit.get('true' if l == 'yes' else 'false', l)}" for l in labels]
        kind = "verify"
    elif t == "choice":
        choices = [f"{_pretty(l)}: {crit[l]}" if isinstance(crit, dict) and l in crit else _pretty(l) for l in labels]
        kind = "choose"
    elif t == "score":
        choices = [f"{l}: {crit[int(l)]}" if isinstance(crit, list) else l for l in labels]
        kind = "score"
    else:
        return None
    if exp not in labels:
        return None
    gp = r.get("provenance", {}).get("gold_probs")
    tier = JEVBENCH_FILES[r["_file"]][0]
    state = r["state"] if isinstance(r["state"], str) else json.dumps(r["state"], ensure_ascii=False)  # 35 hard states are JSON
    return _item(f"jevbench/{r['id']}", suite, kind, q["instructions"], choices, labels.index(exp), "eval",
                 state, gold_probs=[float(gp.get(l, 0.0)) for l in labels] if gp else None,
                 meta={"tier": tier, "family": r["family"], "group": r.get("group") or r["id"]})


def _jevbench(files: list[str], suite: str, limit: int, calib: int) -> list[Item]:
    rows = []
    for f in files:
        for line in _jev_file(f).read_text().splitlines():
            if line.strip():
                rows.append({**json.loads(line), "_file": f})
    items = [it for r in rows if (it := _jev_item(r, suite))]
    if limit < len(items):
        items = random.Random(SEED).sample(items, limit)
    return items                                   # no calib: every public item is scored


# ── C. typed-decisions ───────────────────────────────────────────────────────

TYPED_ID = "LocalLLaMA/typed-decisions"
TYPED_WORKFLOWS = ["agent_trace_observability", "customer_service", "invoice_processing", "security_incidents"]


def _typed_items(row: dict, suite: str, split: str) -> list[Item]:
    state = json.dumps(json.loads(row["state"]), ensure_ascii=False)
    qs, gold = json.loads(row["questions"]), json.loads(row["gold"])
    out = []
    for qid, q in qs.items():
        g, t, crit = gold.get(qid), q["type"], q.get("criteria")
        if g is None:
            continue
        if t == "noul":                  # 2 of the 10 noul questions ship without criteria
            keys, crit = ["false", "true"], crit or {}
            choices = [f"{yn}: {crit[k]}" if k in crit else yn for yn, k in zip(("no", "yes"), keys)]
            kind = "verify"
        elif t == "choice" and isinstance(crit, dict):
            keys = list(crit)
            choices = [f"{_pretty(k)}: {crit[k]}" for k in keys]
            kind = "choose"
        elif t == "score" and isinstance(crit, list):
            keys = [str(i) for i in range(len(crit))]
            choices = [f"{i}: {c}" for i, c in enumerate(crit)]
            kind = "score"
        else:
            continue
        label = str(g["label"]).lower() if t == "noul" else str(g["label"])
        if label not in keys or (kind == "verify" and len(keys) != 2):
            continue
        probs = g.get("probabilities") or {}
        out.append(_item(f"typed/{row['id']}/{qid}", suite, kind, q["instructions"], choices, keys.index(label),
                         split, state, gold_probs=[float(probs.get(k, 0.0)) for k in keys],
                         meta={"workflow": row["workflow"], "question_id": qid, "type": t,
                               "gold_score": g.get("score")}))
    return out


def _typed(workflow: str | None, limit: int, calib: int) -> list[Item]:
    from datasets import load_dataset

    suite = f"typed/{workflow or 'test'}"

    def take(split: str, n: int, tag: str) -> list[Item]:
        if n <= 0:
            return []
        rows = [r for r in load_dataset(TYPED_ID, "all", split=split) if workflow is None or r["workflow"] == workflow]
        items = [it for r in rows for it in _typed_items(r, suite, tag)]
        if n < len(items):               # shuffle whole cases (seed 0), keep each case's questions together
            ids = [r["id"] for r in rows]
            random.Random(SEED).shuffle(ids)
            rank = {cid: i for i, cid in enumerate(ids)}
            items.sort(key=lambda it: rank[it["id"].split("/")[1]])
            items = items[:n]
        return items

    return take("test", limit, "eval") + take("train", calib, "calib")


# ── D. Jev Decision Index ────────────────────────────────────────────────────

JDI_KIT_URL = "https://github.com/apolinario/decision-index"
JDI_KIT_COMMIT = "52a698928a9ae5bdf16b75687c903871db29c6e5"      # 2026-09-22, headline = plain area mean
JDI_SUITE_DATASET = "multimodalart/decision-index-suite"          # frozen rows; not public on 2026-09-23
JDI_ROWS_SHA256 = "288d37207a9581187bdf83eada1983aa63de6fc50b0108e2badb229547a57f99"      # uncompressed
JDI_ROWS_GZ_SHA256 = "750d353a3a83af615c67cfe9752e005bf09e6c28d9c4ba28d3a9f57ba8536cfd"
JDI_SEED = 20260919                   # the kit's hash order, so a subset here = `decision_index suite sample`
JDI_DIR = CACHE / "jdi"
JDI_POOL_MAX = 20000                  # unselected rows kept per capped benchmark (calib source)
JDI_MOVE_INPUT = True                 # empty state → the instructions become the state, question = JDI_QUESTION
JDI_QUESTION = "Choose the option that best answers the request."
JDI_INCLUDE_GATED = False             # HLE is gated on the Hub: accept cais/hle's terms, then flip this

# catalog id → (slug, normalized track files, area (None = display only), index metric)
JDI_BENCHMARKS = {
    1: ("bfcl", ["BFCL-tool-selection"], "tools", "case exact: every candidate-tool yes/no right"),
    2: ("toolret", ["ToolRet-retrieval"], "tools", "nDCG@10 over P(yes), chunks combined per query"),
    3: ("apibank", ["API-Bank-tool-selection"], None, "accuracy"),
    4: ("banking77", ["BANKING77"], None, "macro-F1"),
    5: ("clinc150", ["CLINC150+OOS"], None, "macro-F1"),
    6: ("routerbench", ["RouterBench-0shot", "RouterBench-5shot"], "tools", "realised quality of the chosen model (question 'quality'), tracks averaged"),
    9: ("home-appliance", ["Home-Appliance"], None, "case exact accuracy"),
    10: ("sgd", ["SGD-service-given-intent"], None, "macro-F1"),
    11: ("contractnli", ["ContractNLI"], "language", "conservative macro-F1"),
    12: ("anli", ["ANLI"], None, "macro-F1"),
    20: ("bpomp", ["BPoMP-original-limerick"], "arts", "perturbation-macro accuracy"),
    21: ("humicroedit", ["Humicroedit"], "arts", "accuracy"),
    22: ("pop909", ["POP909-chord-pitch-class"], "arts", "song-macro accuracy"),
    23: ("cfcolor", ["CFColor-preference"], "arts", "user-macro accuracy"),
    24: ("mmlu", ["MMLU"], "knowledge", "subject-macro accuracy"),
    25: ("gpqa-diamond", ["GPQA-Diamond"], "knowledge", "accuracy"),
    26: ("arc-easy", ["ARC-Easy"], None, "accuracy"),
    27: ("arc-challenge", ["ARC-Challenge"], None, "accuracy"),
    28: ("winogrande", ["WinoGrande"], None, "accuracy"),
    29: ("hellaswag", ["HellaSwag"], None, "accuracy"),
    30: ("gsm8k", ["GSM8K-4choice", "GSM8K-10choice"], "knowledge", "accuracy, tracks averaged"),
    31: ("chessbench", ["ChessBench-legal-move"], "knowledge", "best-move accuracy, ties accepted"),
    32: ("musr", ["MuSR"], None, "accuracy"),
    33: ("sata-bench", ["SATA-Bench"], None, "case exact: every option yes/no right"),
    34: ("simplebench", ["SimpleBench"], None, "accuracy"),
    36: ("bright", ["BRIGHT-retrieval"], "retrieval", "nDCG@10 over P(yes), chunks combined per query"),
    37: ("esci", ["Amazon-ESCI"], "retrieval", "conservative macro-F1"),
    38: ("acos", ["ACOS-category-sentiment"], None, "case exact: every category×sentiment yes/no right"),
    39: ("finentity", ["FinEntity-entity-given"], None, "macro-F1"),
    40: ("isarcasmeval", ["iSarcasmEval-A-En", "iSarcasmEval-A-Ar", "iSarcasmEval-B-En", "iSarcasmEval-C-En", "iSarcasmEval-C-Ar"],
         "language", "conservative F1 of 'yes'; the index uses track A-En only, B never scores"),
    41: ("vast", ["VAST"], "language", "conservative macro-F1"),
    42: ("nli4ct", ["NLI4CT-2024"], None, "macro-F1"),
    43: ("cruxeval", ["CRUXEval-output-choice"], "knowledge", "accuracy"),
    44: ("cladder", ["CLadder"], "knowledge", "accuracy"),
    45: ("hle", ["HLE-text-MC"], None, "accuracy"),
    48: ("forecastbench", ["ForecastBench-binary"], None, "Brier of P(yes), lower is better"),
    50: ("habermas", ["Habermas-consensus"], "arts", "group-preference accuracy, ties accepted"),
}
JDI_GATED = {45}
JDI_CONTAMINATED = {
    4: "same test split as our banking77/en; trained on banking77 train + its label space",
    5: "same test split as our clinc150/en; trained on CLINC150 train + its label space",
}
JDI_RETRIEVAL = {2, 36}                 # one "rank" item per query (all chunks merged)
JDI_SCORE_KIND = {39}                   # FinEntity: Negative < Neutral < Positive (cf. financialphrasebank)
JDI_BUILD_GROUPS = [{24, 26, 27, 28, 29}, {11, 42}, {21, 25}, {41, 44}, {30, 43}]   # kit builders that write siblings
JDI_LANG = {"iSarcasmEval-A-Ar": "ar", "iSarcasmEval-C-Ar": "ar"}
JDI_ESCI_LANG = {"us": "en", "es": "es", "jp": "ja"}
_JDI_PLAIN_KEY = re.compile(r"^(?:[A-Z]|option_\d+|text_\d+|headline_\d+)$")

# Coverage-adjusted index metric ("raw") per panel benchmark, Decision Index 0.1 (data/index.json, 2026-09-22)
JDI_REFERENCE = {
    "Jev 1.13.0": {24: 0.92, 25: 0.786, 30: 0.799, 43: 0.73, 44: 0.726, 31: 0.172, 11: 0.717, 40: 0.505, 41: 0.646, 36: 0.188, 37: 0.552, 1: 0.958, 2: 0.446, 6: 0.799, 20: 0.909, 21: 0.619, 22: 0.166, 23: 0.644, 50: 0.459},
    "Decider 35B-A3B": {24: 0.833, 25: 0.408, 30: 0.461, 43: 0.598, 44: 0.667, 31: 0.109, 11: 0.75, 40: 0.548, 41: 0.545, 36: 0.168, 37: 0.506, 1: 0.921, 2: 0.44, 6: 0.796, 20: 0.892, 21: 0.62, 22: 0.064, 23: 0.63, 50: 0.464},
    "Kev 9B": {24: 0.762, 25: 0.388, 30: 0.487, 43: 0.512, 44: 0.62, 31: 0.112, 11: 0.578, 40: 0.471, 41: 0.554, 36: 0.154, 37: 0.492, 1: 0.945, 2: 0.445, 6: 0.8, 20: 0.672, 21: 0.557, 22: 0.101, 23: 0.554, 50: 0.394},
    "Kev 0.5B": {24: 0.372, 25: 0.189, 30: 0.165, 43: 0.395, 44: 0.497, 31: 0.088, 11: 0.366, 40: 0.274, 41: 0.338, 36: 0.078, 37: 0.209, 1: 0.388, 2: 0.31, 6: 0.497, 20: 0.517, 21: 0.486, 22: 0.013, 23: 0.519, 50: 0.291},
    "Decision 1.0 Kai (mmBERT-base)": {24: 0.314, 25: 0.0, 30: 0.334, 43: 0.365, 44: 0.499, 31: 0.075, 11: 0.049, 40: 0.0, 41: 0.387, 36: 0.005, 37: 0.198, 1: 0.388, 2: 0.099, 6: 0.0, 20: 0.423, 21: 0.0, 22: 0.0, 23: 0.493, 50: 0.308},
    "GLiNER 2.5 small (74M)": {24: 0.26, 25: 0.076, 30: 0.235, 43: 0.39, 44: 0.492, 31: 0.081, 11: 0.0, 40: 0.237, 41: 0.257, 36: 0.0, 37: 0.208, 1: 0.407, 2: 0.0, 6: 0.565, 20: 0.427, 21: 0.499, 22: 0.0, 23: 0.508, 50: 0.307},
    "Laya": {24: 0.278, 25: 0.163, 30: 0.216, 43: 0.402, 44: 0.479, 31: 0.034, 11: 0.005, 40: 0.149, 41: 0.271, 36: 0.0, 37: 0.126, 1: 0.362, 2: 0.0, 6: 0.0, 20: 0.178, 21: 0.472, 22: 0.0, 23: 0.509, 50: 0.0},
}


def _sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _jdi_stems() -> dict[str, int]:
    return {s: n for n, (_, stems, _, _) in JDI_BENCHMARKS.items() for s in stems}


def _jdi_kit() -> Path:
    """The reproduction kit at the pinned commit: manifest (selected group ids, normalized-file
    hashes), exclusions, and the per-benchmark normalizers used to rebuild the suite."""
    import subprocess

    kit = JDI_DIR / "kit"
    if not (kit / "hub" / "manifest.json").exists():
        kit.mkdir(parents=True, exist_ok=True)
        git = lambda *a: subprocess.run(["git", *a], cwd=kit, check=True)
        git("init", "-q")
        git("remote", "add", "origin", JDI_KIT_URL)
        git("fetch", "-q", "--depth", "1", "origin", JDI_KIT_COMMIT)
        git("checkout", "-q", "FETCH_HEAD")
    return kit


def _jdi_manifest() -> dict:
    global _JDI_MANIFEST
    if _JDI_MANIFEST is None:
        kit = _jdi_kit()
        m = json.loads((kit / "hub" / "manifest.json").read_text())
        _JDI_MANIFEST = {
            "selected": {int(k): set(v) for k, v in m["selected_group_ids"].items()},
            "sha": {Path(s["path"]).stem: s["sha256"] for b in m["benchmarks"] for s in b["sources"]},
            "caps": {b["catalog_id"] for b in m["benchmarks"] if b["case_cap"] or b["request_cap"]},
            "excluded": set(json.loads((kit / "hub" / "excluded-questions.json").read_text())["rows"]),
        }
    return _JDI_MANIFEST


_JDI_MANIFEST: dict | None = None


def _jdi_rebuild(n: int) -> None:
    """Rebuild benchmark n's normalized files from the pinned public sources with the kit's own
    normalizers (niced, 4 threads). Two kit bugs are patched around: acquire() writes Hub/HTTP
    sources under <work>/raw while the normalizers read <work>/artifacts/benchmark-suite/raw, and
    PolyAI/banking77 is a script dataset whose CSVs live on GitHub; the Habermas parquet is a
    separate GCS download."""
    import os
    import shutil
    import subprocess

    kit, work = _jdi_kit(), JDI_DIR / "work"
    py = kit / ".venv" / "bin" / "python"
    if not py.exists():
        uv = shutil.which("uv") or str(Path.home() / ".local" / "bin" / "uv")
        subprocess.run([uv, "venv", "-q", "--python", "3.12", str(kit / ".venv")], check=True)
        subprocess.run([uv, "pip", "install", "-q", "--python", str(py), "-e", f"{kit}[rebuild]"], check=True)
    raw = work / "artifacts" / "benchmark-suite" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    if not (work / "raw").exists():
        (work / "raw").symlink_to("artifacts/benchmark-suite/raw")
    if n == 4:
        for f in ("test.csv", "categories.json"):
            dst = raw / "banking77" / f
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                with urllib.request.urlopen(f"https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/{f}", timeout=120) as r:
                    dst.write_bytes(r.read())
    if n == 50:          # the repo README says to wget this; acquire() does not
        dst = raw / "repos" / "habermas_machine" / "hm_all_candidate_comparisons.parquet"
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(f"https://storage.googleapis.com/habermas_machine/datasets/{dst.name}", timeout=1800) as r:
                dst.write_bytes(r.read())
    ids = [n] + sorted(next((g for g in JDI_BUILD_GROUPS if n in g), {n}) - {n})
    code = ("import sys; from decision_index.suite.build import acquire as a; from decision_index.suite.build.rebuild import normalize; "
            "from decision_index.suite.build.layout import Layout; L = Layout(sys.argv[1]); ids = [int(x) for x in sys.argv[2:]]; "
            "a.acquire(L, ids); normalize(L, ids[:1])")
    caps = {k: "4" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_MAX_THREADS",
                             "RAYON_NUM_THREADS", "POLARS_MAX_THREADS", "ARROW_NUM_THREADS")}
    env = {**os.environ, **caps, "HF_HUB_DISABLE_XET": "1", "HF_HUB_DOWNLOAD_MAX_WORKERS": "2", "GIT_TERMINAL_PROMPT": "0"}
    subprocess.run(["nice", "-n", "19", str(py), "-c", code, str(work), *map(str, ids)], cwd=kit, env=env, check=True)


def _jdi_normalized(n: int) -> list[Path]:
    """Normalized track files for n, rebuilt if missing, each checked against the manifest hash
    (the frozen file was built from exactly these bytes)."""
    sha, out = _jdi_manifest()["sha"], []
    for stem in JDI_BENCHMARKS[n][1]:
        p = JDI_DIR / "work" / "artifacts" / "benchmark-suite" / "normalized" / f"{stem}.jsonl"
        if not p.exists() or _sha256(p) != sha[stem]:
            _jdi_rebuild(n)
        if _sha256(p) != sha[stem]:
            raise RuntimeError(f"jdi {stem}.jsonl rebuilt with sha {_sha256(p)[:12]}, manifest says {sha[stem][:12]}; "
                               "an upstream source drifted from its pin")
        out.append(p)
    return out


def _jdi_frozen() -> Path | None:
    """The published frozen rows, if they can be had (a local copy, or the Hub dataset once public)."""
    p = JDI_DIR / "selected-rows.jsonl.gz"
    if not p.exists():
        try:
            from huggingface_hub import hf_hub_download

            src = hf_hub_download(JDI_SUITE_DATASET, p.name, repo_type="dataset")
        except Exception:
            return None
        p.write_bytes(Path(src).read_bytes())
    if _sha256(p) != JDI_ROWS_GZ_SHA256:
        import gzip

        with gzip.open(p, "rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != JDI_ROWS_SHA256:
                raise RuntimeError(f"{p} is not the frozen Decision Index suite (sha mismatch)")
    return p


def _jdi_rows(n: int) -> tuple[Path, Path]:
    """→ (selected rows, unselected pool) for benchmark n, cached as JSONL under data/public/jdi/rows/.
    Every row carries _run_id / _track / _group; pool rows (capped benchmarks only) come in the
    kit's hash order after the selected ones and never overlap them."""
    d = JDI_DIR / "rows"
    sel_p, pool_p = d / f"{n}.jsonl", d / f"{n}.pool.jsonl"
    if sel_p.exists() and pool_p.exists():
        return sel_p, pool_p
    d.mkdir(parents=True, exist_ok=True)
    m = _jdi_manifest()
    prio = lambda g: hashlib.sha256(f"{JDI_SEED}:{n}:{g}".encode()).hexdigest()
    frozen = _jdi_frozen()
    sel, pool = [], []
    if frozen is not None:                       # frozen rows only; no pool to draw calib from
        import gzip

        with gzip.open(frozen, "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                e = r.pop("_evaluation")
                if e["catalog_id"] == n:
                    sel.append({**r, "_run_id": e["run_id"], "_track": e["track"], "_group": e["group_id"]})
    else:
        keep = m["selected"][n]
        for p in _jdi_normalized(n):
            for line in p.open(encoding="utf-8"):
                r = json.loads(line)
                g = str((r.get("metadata") or {}).get("group_id", r["id"]))
                r.update(_run_id=f"{n}:{p.stem}:{r['id']}", _track=p.stem, _group=g)
                (sel if g in keep else pool).append(r)
        if n in m["caps"]:
            order = {g: i for i, g in enumerate(sorted({r["_group"] for r in pool}, key=prio))}
            pool.sort(key=lambda r: order[r["_group"]])
            pool = pool[:JDI_POOL_MAX]
        else:
            pool = []                            # uncapped: every prepared case is scored
    for path, rows in ((sel_p, sel), (pool_p, pool)):
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        tmp.replace(path)
    return sel_p, pool_p


def _jdi_choice(key: str, text) -> str:
    text = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False)
    if not text:
        return key
    return text if _JDI_PLAIN_KEY.match(key) or key.lower() == text.lower() else f"{key}: {text}"


def _jdi_items(rows: list[dict], n: int, suite: str, split: str) -> list[Item]:
    """One item per (row, question). expected=None (unjudged retrieval candidate) never becomes gold."""
    name, _, area, _ = JDI_BENCHMARKS[n]
    out = []
    for r in rows:
        st = r["state"]
        state = st if isinstance(st, str) else json.dumps(st, ensure_ascii=False) if st else ""
        md, prov, sc = r.get("metadata") or {}, r.get("provenance") or {}, r.get("scoring") or {}
        lang = JDI_LANG.get(r["_track"]) or JDI_ESCI_LANG.get(md.get("locale"), "en")
        for key, q in r["questions"].items():
            gold_key = r["expected"].get(key)
            if gold_key is None:
                continue
            crit = q["criteria"]
            keys = list(crit)
            kind = "choose"
            if {k.lower() for k in keys} == {"no", "yes"}:
                keys = sorted(keys, key=lambda k: k.lower() != "no")        # [no, yes], as elsewhere here
                kind = "verify"
            elif n in JDI_SCORE_KIND:
                kind = "score"
            question = q["instructions"] if isinstance(q["instructions"], str) else json.dumps(q["instructions"], ensure_ascii=False)
            moved = JDI_MOVE_INPUT and not state
            item_state = question if moved else state
            meta = {"benchmark": name, "catalog_id": n, "track": r["_track"], "group": r["_group"], "field": key,
                    "keys": keys, "area": area, "in_index": area is not None and not (n == 40 and r["_track"] != "iSarcasmEval-A-En")
                    and not (n == 6 and key != "quality"), "moved": moved, "contaminated": n in JDI_CONTAMINATED}
            if n == 24:
                meta["subject"] = prov.get("subject")
            for k in ("variant", "song_id", "user_id", "locale"):
                if k in md:
                    meta[k] = md[k]
            if n in (31, 50) or (n == 6 and isinstance(sc.get("accepted"), dict)):
                acc = sc["accepted"][key] if n == 6 else sc["accepted"]
                meta["accepted"] = [keys.index(a) for a in acc if a in keys]
            if n == 6:
                meta["quality"] = [sc["outcomes"][k]["quality"] for k in keys]
            out.append(_item(f"jdi/{r['_run_id']}/{key}", suite, kind, JDI_QUESTION if moved else question,
                             [_jdi_choice(k, crit[k]) for k in keys], keys.index(gold_key), split, item_state,
                             lang=lang, meta=meta))
    return out


def _jdi_rank_items(rows: list[dict], n: int, suite: str) -> list[Item]:
    """Retrieval: merge a query's chunks into one "rank" item (choices = every scorable candidate,
    gold = first judged-relevant one). Queries with no relevant candidate are dropped: nDCG@10 is 0
    for every system there. meta.qrels aligns with choices (unjudged = 0, as the kit's nDCG does)."""
    name, _, area, _ = JDI_BENCHMARKS[n]
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r["_group"], []).append(r)
    out = []
    for g, rs in groups.items():
        docs, texts = [], []
        for r in rs:
            f2d = r["scoring"]["field_to_document"]
            for key, q in r["questions"].items():
                docs.append(f2d[key])
                texts.append(q["instructions"]["candidate"])
        qrels = rs[0]["scoring"]["qrels"]
        rel = [int(qrels.get(d, 0)) for d in docs]
        if not any(rel):
            continue
        st = rs[0]["state"]
        out.append(_item(f"jdi/{n}:{rs[0]['_track']}:{g}", suite, "rank", st["task"], texts,
                         next(i for i, v in enumerate(rel) if v > 0), "eval", st["query"],
                         meta={"benchmark": name, "catalog_id": n, "track": rs[0]["_track"], "group": g, "area": area,
                               "in_index": True, "contaminated": False, "doc_ids": docs, "qrels": rel,
                               "ideal": sorted(map(int, qrels.values()), reverse=True)[:10],
                               "candidate_recall": rs[0]["scoring"].get("candidate_recall")}))
    return out


def _jdi(n: int, track: str | None, suite: str, limit: int, calib: int) -> list[Item]:
    sel_p, pool_p = _jdi_rows(n)
    excluded = _jdi_manifest()["excluded"]
    prio = lambda g: hashlib.sha256(f"{JDI_SEED}:{n}:{g}".encode()).hexdigest()

    def load(path: Path) -> list[dict]:
        rows = [json.loads(l) for l in path.open(encoding="utf-8")]
        return [r for r in rows if (track is None or r["_track"] == track) and r["_run_id"] not in excluded]

    def take(items: list[Item], k: int) -> list[Item]:
        if k < len(items):                       # whole groups in the kit's hash order, cut at k
            items = sorted(items, key=lambda it: prio(it["meta"]["group"]))[:k]
        return items

    sel = load(sel_p)
    ev = _jdi_rank_items(sel, n, suite) if n in JDI_RETRIEVAL else _jdi_items(sel, n, suite, "eval")
    ca = _jdi_items(load(pool_p), n, suite, "calib") if calib > 0 and n not in JDI_RETRIEVAL else []
    return take(ev, limit) + ca[:max(calib, 0)]           # pool is already in hash order


# ── registry ─────────────────────────────────────────────────────────────────

SUITES: dict[str, Loader] = {f"btzsc/{d}": partial(_btzsc, d) for d in BTZSC_DATASETS}
SUITES.update({
    "jevbench/public": partial(_jevbench, ["easy", "original", "hard"], "jevbench/public"),
    "jevbench/easy": partial(_jevbench, ["easy"], "jevbench/easy"),
    "jevbench/standard": partial(_jevbench, ["original"], "jevbench/standard"),
    "jevbench/hard": partial(_jevbench, ["hard"], "jevbench/hard"),
    "typed/test": partial(_typed, None),
})
SUITES.update({f"typed/{w}": partial(_typed, w) for w in TYPED_WORKFLOWS})
for _n, (_slug, _stems, _, _) in JDI_BENCHMARKS.items():   # multi-track benchmarks: one suite per track, no overlap
    if _n in JDI_GATED and not JDI_INCLUDE_GATED:
        continue
    for _t in (_stems if len(_stems) > 1 else [None]):
        _s = f"jdi/{_slug}" + (f"-{_t.split('-', 1)[1].lower()}" if _t else "")
        SUITES[_s] = partial(_jdi, _n, _t, _s)
