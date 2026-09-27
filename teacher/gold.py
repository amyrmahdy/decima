"""Gold-labelled NLI + yes/no reading data for Decima, rendered as decisions.

Why: Decima is weak when the deciding information sits INSIDE THE QUESTION (Kev's MNLI
format: state = premise, question = 'Hypothesis: "<h>" How does it relate to the
premise?'; BoolQ: state = passage, question = the yes/no question). These rows carry
human (or machine-translated human) gold labels, so no teacher is involved.

Sources — TRAIN SPLITS ONLY (validation/test are our eval sets; never read here):
  en  mnli      nyu-mll/multi_nli                       train
  en  snli      stanfordnlp/snli                        train (label -1 dropped, subsampled)
  en  anli      facebook/anli                           train_r1, train_r2, train_r3
  en  wanli     alisawuffles/WANLI                      train.jsonl
  en  boolq     google/boolq                            train (each question rendered twice)
  xx  xnli      facebook/xnli  <lang>/train             (machine-translated MNLI train)
                ar, ru + de fr es tr zh hi vi bg el sw ur th
  fa  farstail  azarijafari/FarsTail  data/Train-word.csv  (original train TSV; each pair
                rendered twice. mteb/FarsTail only ships test)
  xx  nli26     MoritzLaurer/multilingual-NLI-26lang-2mil7  <lang>_{mnli,anli,fever,wanli,ling}
                (machine translations of the TRAIN splits of those five datasets)
                fa, ar, ru + de fr es tr zh hi uk pl it pt ja ko id he vi

Formats (chosen per row by a seeded RNG; the format is not stored in the rows):
  A  info in question: state = premise (sometimes wrapped as `document: …`, a ticket, JSON…),
     question = hypothesis embedded in one of ≥12 templates (+ fa/ar/ru/de/fr/es/tr ones),
     kind choose, choices = relation labels (3-way, or 2-way follows / does-not-follow).
  B  both in state: state = "Premise: …\\nHypothesis: …" (or JSON, yaml, Sentence 1/2,
     Text/Claim, Context/Statement, native-language keys), question generic, kind choose.
  C  verify: yes/no, asking entailment (mostly), contradiction or undetermined-ness; the
     hypothesis is in the question (Cq, ~55 %) or in the state (Cs). Yes/no order varies,
     bare or "yes: …"/"no: …" descriptive choices.
  S  score (small share): ordered truth levels, lowest→highest (false / undetermined / true).
  Q  BoolQ: state = passage (≤1,500 chars, cut at a word), question = the question in one
     of ≥10 templates ("Answer yes or no: …", "question: …\\nfocus: …", …), kind verify.
  Choice wording: bare labels, descriptive sentences, Laya-style "key: description", Kev-style
  mixes; ~30 % of non-English rows get translated choices (fa/ar/ru/de/fr/es/tr); choose
  choices are shuffled.

Targets: probs = 0.92 on gold, 0.08 spread evenly over the rest.

Size (targets, after filtering): ~400k rows. en ~160k (40 %); fa/ar/ru ~56k each (14 %);
~72k over 17 other languages.

Decontamination: every eval item file runs/items/*.jsonl is read ("state" and "question");
both are split into segments (whole text, lines with a leading `key:` stripped, JSON string
values, quoted spans «…» "…" “…”) and normalized (lowercase, collapsed whitespace, trailing
punctuation/quotes stripped); segments shorter than 8 chars are ignored. A source pair is
dropped if its premise, hypothesis, passage or BoolQ question (or, for nli26, the English
originals) exactly matches a segment, or if a ≥200-char text shares its first 200 normalized
chars with a ≥200-char segment (catches truncated passages). Drops are counted over the
whole train pool of each source and written to <out>/stats.json.

Usage:  uv run python -m teacher.gold --out data/gold/ --seed 0
"""

from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

E, N, C = 0, 1, 2  # entailment, neutral, contradiction (MNLI/XNLI/ANLI/SNLI convention)
REPO = Path(__file__).resolve().parents[1]

# ─────────────────────────────── quotas ───────────────────────────────
XNLI_OTHER = ["de", "fr", "es", "tr", "zh", "hi", "vi", "bg", "el", "sw", "ur", "th"]
NLI26_OTHER = ["de", "fr", "es", "tr", "zh", "hi", "uk", "pl", "it", "pt", "ja", "ko", "id", "he", "vi"]
# (domain, lang) → number of source pairs to keep (boolq/farstail pairs are rendered twice)
QUOTA: dict[tuple[str, str], int] = {
    ("mnli", "en"): 66_000, ("snli", "en"): 20_000, ("anli", "en"): 30_000, ("wanli", "en"): 25_000,
    ("boolq", "en"): 10_000,                       # all of train (9,427) → ×2 renders
    ("farstail", "fa"): 8_000,                     # all of train (7,266) → ×2 renders
    ("nli26", "fa"): 42_000,
    ("xnli", "ar"): 26_000, ("nli26", "ar"): 30_000,
    ("xnli", "ru"): 26_000, ("nli26", "ru"): 30_000,
    **{("xnli", l): 3_000 for l in XNLI_OTHER},
    **{("nli26", l): 2_400 for l in NLI26_OTHER},
}
RENDERS = {"boolq": 2, "farstail": 2}
MAX_TEXT = 1500

# ─────────────────────────────── loading ───────────────────────────────


def _hub(repo: str, path: str) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, path, repo_type="dataset")


def _parquet(repo: str, path: str, cols: list[str]) -> list[dict]:
    import pyarrow.parquet as pq
    return pq.read_table(_hub(repo, path), columns=cols).to_pylist()


def _pairs(rows, lang, domain, sub, lab_map=None):
    out = []
    for r in rows:
        lab = r["label"] if lab_map is None else lab_map.get(r["label"], -1)
        p, h = (r.get("premise") or "").strip(), (r.get("hypothesis") or "").strip()
        if lab not in (E, N, C) or not p or not h:
            continue
        orig = [x for x in (r.get("premise_original"), r.get("hypothesis_original")) if x]
        out.append({"domain": domain, "lang": lang, "sub": sub, "p": p, "h": h, "y": lab, "orig": orig})
    return out


@functools.lru_cache(maxsize=1)
def _nli26_files() -> tuple[str, ...]:
    from huggingface_hub import HfApi
    return tuple(s.rfilename for s in HfApi().dataset_info("MoritzLaurer/multilingual-NLI-26lang-2mil7").siblings)


def load_source(domain: str, lang: str) -> list[dict]:
    if domain == "mnli":
        return _pairs(_parquet("nyu-mll/multi_nli", "data/train-00000-of-00001.parquet", ["premise", "hypothesis", "label"]), lang, domain, "train")
    if domain == "snli":
        return _pairs(_parquet("stanfordnlp/snli", "plain_text/train-00000-of-00001.parquet", ["premise", "hypothesis", "label"]), lang, domain, "train")
    if domain == "anli":
        out = []
        for r in ("r1", "r2", "r3"):
            out += _pairs(_parquet("facebook/anli", f"plain_text/train_{r}-00000-of-00001.parquet", ["premise", "hypothesis", "label"]), lang, domain, f"train_{r}")
        return out
    if domain == "wanli":
        rows = [json.loads(l) for l in open(_hub("alisawuffles/WANLI", "train.jsonl"))]
        for r in rows:
            r["label"] = r["gold"]
        return _pairs(rows, lang, domain, "train", {"entailment": E, "neutral": N, "contradiction": C})
    if domain == "xnli":
        return _pairs(_parquet("facebook/xnli", f"{lang}/train-00000-of-00001.parquet", ["premise", "hypothesis", "label"]), lang, domain, "train")
    if domain == "farstail":
        with open(_hub("azarijafari/FarsTail", "data/Train-word.csv"), encoding="utf-8") as f:
            rows = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
        return _pairs(rows, lang, domain, "train", {"e": E, "n": N, "c": C})
    if domain == "nli26":
        files = _nli26_files()
        out = []
        for sub in ("mnli", "anli", "fever", "wanli", "ling"):
            fn = [f for f in files if f.startswith(f"data/{lang}_{sub}-")]
            for f in fn:
                out += _pairs(_parquet("MoritzLaurer/multilingual-NLI-26lang-2mil7", f,
                                       ["premise_original", "hypothesis_original", "label", "premise", "hypothesis"]), lang, domain, sub)
        return out
    if domain == "boolq":
        rows = _parquet("google/boolq", "data/train-00000-of-00001.parquet", ["question", "answer", "passage"])
        return [{"domain": "boolq", "lang": "en", "sub": "train", "p": r["passage"].strip(), "h": r["question"].strip(),
                 "y": 0 if r["answer"] else 1, "orig": []} for r in rows if r["passage"] and r["question"]]
    raise KeyError(domain)


# ─────────────────────────────── decontamination ───────────────────────────────
_WS = re.compile(r"\s+")
_KEY = re.compile(r"^[\w`'\" \-]{1,30}:\s+")
_QUOTED = re.compile(r"\"([^\"]{8,})\"|«([^»]{8,})»|“([^”]{8,})”")
MIN_SEG, PREFIX = 8, 200


def norm(s: str) -> str:
    return _WS.sub(" ", s.lower()).strip().strip("\"'`“”«»").strip().rstrip(".!?。؟").strip()


def _json_strings(o, out):
    if isinstance(o, str):
        out.append(o)
    elif isinstance(o, dict):
        for v in o.values():
            _json_strings(v, out)
    elif isinstance(o, list):
        for v in o:
            _json_strings(v, out)


def _segments(text: str) -> list[str]:
    segs = [text]
    t = text.strip()
    if t[:1] in "{[":
        try:
            _json_strings(json.loads(t), segs)
        except Exception:
            pass
    more = []
    for s in segs:
        for line in s.split("\n"):
            line = line.strip()
            more.append(line)
            m = _KEY.match(line)
            if m:
                more.append(line[m.end():])
        for m in _QUOTED.finditer(s):
            more.append(next(g for g in m.groups() if g))
    return segs + more


class Contam:
    def __init__(self, item_dir: Path):
        self.exact: set[int] = set()
        self.prefix: set[int] = set()
        self.files = sorted(item_dir.glob("*.jsonl"))
        for f in self.files:
            for line in f.open():
                r = json.loads(line)
                for field in ("state", "question"):
                    v = r.get(field)
                    if not isinstance(v, str):
                        continue
                    for s in _segments(v):
                        n = norm(s)
                        if len(n) >= MIN_SEG:
                            self.exact.add(hash(n))
                            if len(n) >= PREFIX:
                                self.prefix.add(hash(n[:PREFIX]))

    def hit(self, texts: list[str]) -> bool:
        for t in texts:
            n = norm(t)
            if len(n) < MIN_SEG:
                continue
            if hash(n) in self.exact or (len(n) >= PREFIX and hash(n[:PREFIX]) in self.prefix):
                return True
        return False


# ─────────────────────────────── label wording ───────────────────────────────
# 3-way sets in E, N, C order
EN3 = [
    ["entailment", "neutral", "contradiction"],
    ["entails", "neutral", "contradicts"],
    ["the premise implies the hypothesis is true", "the premise neither implies nor contradicts the hypothesis", "the premise implies the hypothesis is false"],
    ["The hypothesis follows from the premise", "The hypothesis may or may not be true given the premise", "The hypothesis contradicts the premise"],
    ["true", "undetermined", "false"],
    ["definitely true", "might be true", "definitely false"],
    ["supported", "not enough information", "refuted"],
    ["implied by the text", "not addressed by the text", "ruled out by the text"],
    ["yes, it follows", "can't tell", "no, it is contradicted"],
    ["consistent and implied", "unrelated or undetermined", "inconsistent"],
]
KEYS3 = ["entailment", "neutral", "contradiction"]
DESC3 = [
    ["the premise implies the hypothesis is true", "the premise neither implies nor contradicts the hypothesis", "the premise implies the hypothesis is false"],
    ["The hypothesis follows from the premise", "The hypothesis may or may not be true given the premise", "The hypothesis cannot be true if the premise is"],
    ["the statement must be true", "the statement may or may not be true", "the statement must be false"],
    ["supported by the text", "neither supported nor refuted", "refuted by the text"],
]
EN2 = [
    ["entailment", "not entailment"],
    ["the hypothesis follows from the premise", "the hypothesis does not follow from the premise"],
    ["entailed", "not entailed"],
    ["follows", "does not follow"],
    ["implied by the text", "not implied by the text"],
]
TR3 = {
    "fa": [["استلزام", "خنثی", "تناقض"], ["فرضیه از مقدمه نتیجه می‌شود", "مقدمه نه فرضیه را تأیید می‌کند نه رد", "فرضیه با مقدمه در تناقض است"], ["درست", "نامشخص", "نادرست"]],
    "ar": [["استلزام", "محايد", "تناقض"], ["الفرضية تنتج عن المقدمة", "المقدمة لا تؤكد الفرضية ولا تنفيها", "الفرضية تتناقض مع المقدمة"], ["صحيح", "غير محدد", "خطأ"]],
    "ru": [["следование", "нейтрально", "противоречие"], ["гипотеза следует из посылки", "посылка не подтверждает и не опровергает гипотезу", "гипотеза противоречит посылке"], ["верно", "неизвестно", "неверно"]],
    "de": [["Folgerung", "neutral", "Widerspruch"], ["wahr", "unbestimmt", "falsch"]],
    "fr": [["implication", "neutre", "contradiction"], ["vrai", "indéterminé", "faux"]],
    "es": [["implicación", "neutral", "contradicción"], ["verdadero", "indeterminado", "falso"]],
    "tr": [["çıkarım", "nötr", "çelişki"], ["doğru", "belirsiz", "yanlış"]],
}
TR2 = {
    "fa": [["فرضیه از مقدمه نتیجه می‌شود", "فرضیه از مقدمه نتیجه نمی‌شود"], ["استلزام", "عدم استلزام"]],
    "ar": [["الفرضية تنتج عن المقدمة", "الفرضية لا تنتج عن المقدمة"], ["استلزام", "لا استلزام"]],
    "ru": [["гипотеза следует из посылки", "гипотеза не следует из посылки"], ["следует", "не следует"]],
    "de": [["folgt", "folgt nicht"]], "fr": [["découle", "ne découle pas"]], "es": [["se deduce", "no se deduce"]], "tr": [["çıkar", "çıkmaz"]],
}
YESNO = {"fa": ("بله", "خیر"), "ar": ("نعم", "لا"), "ru": ("да", "нет"), "de": ("ja", "nein"), "fr": ("oui", "non"),
         "es": ("sí", "no"), "tr": ("evet", "hayır"), "zh": ("是", "否"), "hi": ("हाँ", "नहीं")}
EN_YN = [("yes", "no"), ("Yes", "No"), ("true", "false")]
YN_DESC = [
    ("yes: the hypothesis follows from the premise", "no: the hypothesis does not follow from the premise"),
    ("yes: The text supports this", "no: The text does not support this"),
    ("yes: it is implied", "no: it is not implied"),
]
YN_DESC_POL = {  # polarity-neutral descriptions
    "E": [("yes: The text supports this", "no: The text does not support this")],
    "C": [("yes: The text rules this out", "no: The text does not rule this out")],
    "N": [("yes: The text leaves it open", "no: The text settles it one way or the other")],
}
SCORE3 = [["definitely false", "could be either", "definitely true"], ["false", "undetermined", "true"],
          ["contradicted", "neutral", "entailed"], ["very unlikely", "uncertain", "certain"]]

# ─────────────────────────────── templates ───────────────────────────────
A_EN = [
    'Hypothesis: "{h}" How does it relate to the premise?',
    'Given the text, is the following statement true, false, or undetermined? "{h}"',
    "Claim: {h}\nWhat does the text say about this claim?",
    'How does the statement "{h}" relate to the passage?',
    'Consider the statement: "{h}". Based on the text, which relation holds?',
    "Hypothesis: {h}\nRelation to the premise?",
    'Does the text support, contradict, or say nothing about: "{h}"?',
    'If the text is true, what can we say about "{h}"?',
    'Statement: "{h}" Is it implied by, contradicted by, or neutral with respect to the text?',
    "Read the text, then judge this hypothesis: {h}",
    "hypothesis: {h}\nfocus: relation to the premise",
    '"{h}": entailed, neutral, or contradicted by the premise?',
    "What is the relationship between the text and this sentence? {h}",
    'Premise is given. Hypothesis: "{h}". Classify the relation.',
]
A_NATIVE = {
    "fa": ["فرضیه: «{h}» این فرضیه چه نسبتی با مقدمه دارد؟", "با توجه به متن، آیا این جمله درست است، نادرست است یا نامشخص؟ «{h}»",
           "ادعا: {h}\nمتن درباره این ادعا چه می‌گوید؟", "رابطه جمله «{h}» با متن بالا چیست؟"],
    "ar": ["الفرضية: «{h}» ما علاقتها بالمقدمة؟", "بناءً على النص، هل العبارة التالية صحيحة أم خاطئة أم غير محددة؟ «{h}»",
           "الادعاء: {h}\nماذا يقول النص عن هذا الادعاء؟", "ما العلاقة بين الجملة «{h}» والنص أعلاه؟"],
    "ru": ["Гипотеза: «{h}» Как она соотносится с посылкой?", "Исходя из текста, является ли утверждение верным, ложным или неопределённым? «{h}»",
           "Утверждение: {h}\nЧто говорит текст об этом утверждении?", "Как утверждение «{h}» соотносится с текстом выше?"],
    "de": ["Hypothese: „{h}“ Wie verhält sie sich zur Prämisse?", "Ist die Aussage „{h}“ laut Text wahr, falsch oder unbestimmt?"],
    "fr": ["Hypothèse : « {h} » Quel est son rapport avec la prémisse ?", "D'après le texte, l'énoncé « {h} » est-il vrai, faux ou indéterminé ?"],
    "es": ["Hipótesis: «{h}» ¿Qué relación tiene con la premisa?", "Según el texto, ¿la afirmación «{h}» es verdadera, falsa o indeterminada?"],
    "tr": ["Hipotez: \"{h}\" Öncülle ilişkisi nedir?", "Metne göre \"{h}\" ifadesi doğru mu, yanlış mı, belirsiz mi?"],
}
WRAP = ["Premise: {p}", "Text: {p}", "document: {p}", "ticket:\n  channel: email\n  body: {p}", "message:\n  from: user\n  body: {p}",
        "context: {p}", "passage: {p}", "JSON"]
WRAP_NATIVE = {"fa": ["مقدمه: {p}", "متن: {p}"], "ar": ["المقدمة: {p}", "النص: {p}"], "ru": ["Посылка: {p}", "Текст: {p}"]}

# B layouts: (state template, name of first, name of second)
B_LAYOUT = [
    ("Premise: {p}\nHypothesis: {h}", "the premise", "the hypothesis"),
    ("premise: {p}\nhypothesis: {h}", "the premise", "the hypothesis"),
    ("JSON", "`premise`", "`hypothesis`"),
    ("Sentence 1: {p}\nSentence 2: {h}", "sentence 1", "sentence 2"),
    ("Text: {p}\nClaim: {h}", "the text", "the claim"),
    ("Context: {p}\nStatement: {h}", "the context", "the statement"),
    ("{p}\n\nHypothesis: {h}", "the passage", "the hypothesis"),
    ("Classify the relationship between the premise and hypothesis.\nPremise: {p}\nHypothesis: {h}", "the premise", "the hypothesis"),
]
B_NATIVE = {"fa": ("مقدمه: {p}\nفرضیه: {h}", "the premise", "the hypothesis"),
            "ar": ("المقدمة: {p}\nالفرضية: {h}", "the premise", "the hypothesis"),
            "ru": ("Посылка: {p}\nГипотеза: {h}", "the premise", "the hypothesis")}
B_Q = [
    "What is the relationship between {A} and {B}?",
    "Does {B} follow from {A}?",
    "How does {B} relate to {A}?",
    "Classify the relationship between {A} and {B}.",
    "Given {A}, is {B} true, false, or undetermined?",
    "Which label describes how {A} bears on {B}?",
    "Choose the inference relation from {A} to {B}.",
    "Is {B} entailed by {A}, contradicted by it, or neither?",
    "Choose the option that best answers the request.",
    "Natural language inference: label the pair.",
]
B_Q_NATIVE = {"fa": ["رابطه مقدمه و فرضیه چیست؟", "آیا فرضیه از مقدمه نتیجه می‌شود؟"],
              "ar": ["ما العلاقة بين المقدمة والفرضية؟", "هل تنتج الفرضية عن المقدمة؟"],
              "ru": ["Каково отношение между посылкой и гипотезой?", "Следует ли гипотеза из посылки?"]}

# C: question asks polarity P ∈ {E, C, N}; answer yes iff label == P
CQ_EN = {
    "E": ['Does the text imply that "{h}"?', 'Is the following statement supported by the text? "{h}"',
          'Hypothesis: "{h}" Does it follow from the premise?', 'Can we conclude that "{h}"?',
          "Statement: {h}\nIs this statement entailed by the text?", 'Given the text, must it be true that "{h}"?',
          'Claim: "{h}"\nDoes the text back up this claim?', "Based only on the passage, is this true? {h}",
          "question: does the premise entail \"{h}\"?\nfocus: Consider the whole message."],
    "C": ['Does the text contradict this statement: "{h}"?', "Statement: {h}\nIs this statement ruled out by the text?",
          'Hypothesis: "{h}" Is it contradicted by the premise?', 'Does the passage make "{h}" false?',
          'Is "{h}" incompatible with the text?'],
    "N": ['Is "{h}" left undetermined by the text (neither implied nor contradicted)?',
          "Statement: {h}\nIs the text silent on whether this is true?"],
}
CQ_NATIVE = {
    "fa": {"E": ["آیا از متن نتیجه می‌شود که «{h}»؟", "فرضیه: «{h}» آیا از مقدمه نتیجه می‌شود؟"], "C": ["آیا متن با این جمله در تناقض است: «{h}»؟"]},
    "ar": {"E": ["هل يستنتج من النص أن «{h}»؟", "الفرضية: «{h}» هل تنتج عن المقدمة؟"], "C": ["هل يتناقض النص مع هذه العبارة: «{h}»؟"]},
    "ru": {"E": ["Следует ли из текста, что «{h}»?", "Гипотеза: «{h}» Следует ли она из посылки?"], "C": ["Противоречит ли текст утверждению «{h}»?"]},
    "de": {"E": ["Folgt aus dem Text, dass „{h}“?"]}, "fr": {"E": ["Peut-on déduire du texte que « {h} » ?"]},
    "es": {"E": ["¿Se deduce del texto que «{h}»?"]}, "tr": {"E": ["Metinden \"{h}\" sonucu çıkar mı?"]},
}
CS_EN = {
    "E": ["Does {B} follow from {A}?", "Is {B} entailed by {A}?", "Can {B} be concluded from {A}?",
          "Does {A} support {B}?", "Given {A}, must {B} be true?"],
    "C": ["Does {A} contradict {B}?", "Is {B} ruled out by {A}?", "Are {A} and {B} incompatible?"],
    "N": ["Is {B} neither implied nor contradicted by {A}?"],
}
CS_NATIVE = {"fa": {"E": ["آیا فرضیه از مقدمه نتیجه می‌شود؟"], "C": ["آیا مقدمه با فرضیه در تناقض است؟"]},
             "ar": {"E": ["هل تنتج الفرضية عن المقدمة؟"], "C": ["هل تتناقض المقدمة مع الفرضية؟"]},
             "ru": {"E": ["Следует ли гипотеза из посылки?"], "C": ["Противоречит ли посылка гипотезе?"]}}
S_Q = ['How true is "{h}", given the text?', "Statement: {h}\nHow well does the text establish this?",
       'Rate the truth of "{h}" in light of the premise.']
S_QS = ["How true is {B}, given {A}?", "Rate how strongly {A} establishes {B}."]

BQ = [
    "{q}?", "Answer yes or no: {q}?", "question: {q}?\nfocus: Consider the whole message.", "Based on the passage, {q}?",
    "Q: {q}?", "According to the text, {q}?", "{Q}?", "Question: {Q}? Answer yes or no.",
    "Read the passage and answer: {q}?", "Is the answer to this question yes or no? {q}?", "{q}",
    "Does the passage say yes or no to this: {q}?",
]
BQ_CHOICES = [("yes", "no"), ("Yes", "No"), ("true", "false"),
              ("yes: The passage supports a yes answer", "no: The passage supports a no answer or does not say"),
              ("yes: the answer is yes", "no: the answer is no")]
BOOLQ_WRAP = ["{p}", "{p}", "{p}", "document: {p}", "passage: {p}", "Passage:\n{p}"]


# ─────────────────────────────── rendering ───────────────────────────────


def _cut(t: str, n: int = MAX_TEXT) -> str:
    if len(t) <= n:
        return t
    t = t[:n]
    sp = t.rfind(" ")
    return (t[:sp] if sp > n * 0.8 else t).rstrip() + " …"


def smooth(n: int, gold: int) -> list[float]:
    rest = round(0.08 / (n - 1), 6)
    p = [rest] * n
    p[gold] = round(1 - rest * (n - 1), 6)
    return p


def _wrap_premise(rng, p, lang):
    r = rng.random()
    if r < 0.65:
        return p
    if lang in WRAP_NATIVE and r < 0.72:
        return rng.choice(WRAP_NATIVE[lang]).format(p=p)
    w = rng.choice(WRAP)
    return json.dumps({"text": p}, ensure_ascii=False) if w == "JSON" else w.format(p=p)


def _nli_choices(rng, y, lang, allow2=True):
    """→ (choices, gold, translated?) — shuffled."""
    two = allow2 and rng.random() < 0.12
    translate = lang != "en" and rng.random() < 0.30 and lang in TR3
    if two:
        labs = rng.choice(TR2[lang]) if translate else rng.choice(EN2)
        items = [(labs[0], y == E), (labs[1], y != E)]
    else:
        r = rng.random()
        if translate:
            labs = rng.choice(TR3[lang])
        elif r < 0.45:
            labs = rng.choice(EN3)
        elif r < 0.75:
            d = rng.choice(DESC3)
            labs = [f"{k}: {v}" for k, v in zip(KEYS3, d)]
        else:  # Kev-style mix of bare keys and "key: description"
            d = rng.choice(DESC3)
            labs = [f"{k}: {v}" if rng.random() < 0.5 else k for k, v in zip(KEYS3, d)]
        items = [(labs[i], y == i) for i in range(3)]
    rng.shuffle(items)
    return [c for c, _ in items], next(i for i, (_, g) in enumerate(items) if g), translate


def _yn(rng, lang, pol, yes_gold):
    translate = lang in YESNO and lang != "en" and rng.random() < 0.30
    r = rng.random()
    if translate:
        y, n = YESNO[lang]
    elif r < 0.55:
        y, n = rng.choice(EN_YN)
    else:
        y, n = rng.choice(YN_DESC if pol == "E" else YN_DESC_POL[pol])
    items = [(y, yes_gold), (n, not yes_gold)]
    if rng.random() < 0.5:
        items.reverse()
    return [c for c, _ in items], next(i for i, (_, g) in enumerate(items) if g), translate


def _b_state(rng, p, h, lang):
    lay = B_NATIVE[lang] if lang in B_NATIVE and rng.random() < 0.15 else rng.choice(B_LAYOUT)
    st = json.dumps({"premise": p, "hypothesis": h}, ensure_ascii=False) if lay[0] == "JSON" else lay[0].format(p=p, h=h)
    return st, lay[1], lay[2]


def _native(rng, lang, pool_native, share=0.25):
    return lang in pool_native and rng.random() < share


def render_nli(pair: dict, rng: random.Random, fmt: str | None = None) -> tuple[dict, str]:
    p, h, y, lang = _cut(pair["p"]), _cut(pair["h"], 600), pair["y"], pair["lang"]
    if fmt is None:
        r = rng.random()
        fmt = "A" if r < 0.42 else "B" if r < 0.64 else "C" if r < 0.94 else "S"
    if fmt == "C":
        fmt = "Cq" if rng.random() < 0.55 else "Cs"
    kind, native_q = "choose", False
    if fmt == "A":
        state = _wrap_premise(rng, p, lang)
        native_q = _native(rng, lang, A_NATIVE)
        question = rng.choice(A_NATIVE[lang] if native_q else A_EN).format(h=h)
        choices, gold, tr = _nli_choices(rng, y, lang)
    elif fmt == "B":
        state, A, B = _b_state(rng, p, h, lang)
        native_q = _native(rng, lang, B_Q_NATIVE, 0.2)
        question = rng.choice(B_Q_NATIVE[lang]) if native_q else rng.choice(B_Q).format(A=A, B=B)
        choices, gold, tr = _nli_choices(rng, y, lang)
    elif fmt in ("Cq", "Cs"):
        kind = "verify"
        pol = "E" if rng.random() < 0.65 else "C" if rng.random() < 0.75 else "N"
        yes_gold = y == {"E": E, "C": C, "N": N}[pol]
        if fmt == "Cq":
            state = _wrap_premise(rng, p, lang)
            nat = CQ_NATIVE.get(lang, {}).get(pol)
            native_q = bool(nat) and rng.random() < 0.25
            question = rng.choice(nat if native_q else CQ_EN[pol]).format(h=h)
        else:
            state, A, B = _b_state(rng, p, h, lang)
            nat = CS_NATIVE.get(lang, {}).get(pol)
            native_q = bool(nat) and rng.random() < 0.2
            question = rng.choice(nat) if native_q else rng.choice(CS_EN[pol]).format(A=A, B=B)
        choices, gold, tr = _yn(rng, lang, pol, yes_gold)
    else:  # S: ordered truth levels, lowest → highest (C, N, E)
        kind = "score"
        levels = rng.choice(SCORE3)
        gold = {C: 0, N: 1, E: 2}[y]
        tr = False
        choices = list(levels)
        if rng.random() < 0.6:
            state = _wrap_premise(rng, p, lang)
            question = rng.choice(S_Q).format(h=h)
            fmt = "Sq"
        else:
            state, A, B = _b_state(rng, p, h, lang)
            question = rng.choice(S_QS).format(A=A, B=B)
            fmt = "Ss"
    h_in_q = fmt in ("A", "Cq", "Sq")
    choice_lang = lang if (native_q or tr or h_in_q) else "en"
    return _row(pair, kind, state, question, choices, gold, lang, choice_lang), fmt


def render_boolq(pair: dict, rng: random.Random) -> tuple[dict, str]:
    q = pair["h"].rstrip("?").strip()
    state = rng.choice(BOOLQ_WRAP).format(p=_cut(pair["p"]))
    question = rng.choice(BQ).format(q=q, Q=q[:1].upper() + q[1:])
    y, n = rng.choice(BQ_CHOICES)
    if "yes or no" in question and y == "true":
        y, n = "yes", "no"
    items = [(y, pair["y"] == 0), (n, pair["y"] == 1)]
    if rng.random() < 0.35:
        items.reverse()
    choices = [c for c, _ in items]
    gold = next(i for i, (_, g) in enumerate(items) if g)
    return _row(pair, "verify", state, question, choices, gold, "en", "en"), "Q"


def _row(pair, kind, state, question, choices, gold, lang, choice_lang):
    dom = pair["domain"]
    key = json.dumps([dom, lang, state, question, choices], ensure_ascii=False)
    return {
        "id": hashlib.sha1(key.encode()).hexdigest()[:20], "source": "gold", "domain": dom, "kind": kind,
        "state": state, "question": question, "choices": choices, "gold": gold, "probs": smooth(len(choices), gold),
        "state_lang": lang, "choice_lang": choice_lang, "none_idx": None, "batch": f"{dom}/{lang}|gold",
    }


# ─────────────────────────────── main ───────────────────────────────


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="data/gold/")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--items", default=str(REPO / "runs/items"))
    ap.add_argument("--scale", type=float, default=1.0, help="multiply every quota (for quick tests)")
    a = ap.parse_args(argv)
    try:
        import pyarrow as pa
        pa.set_cpu_count(4); pa.set_io_thread_count(4)
    except Exception:
        pass

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    print("decontamination index …", file=sys.stderr, flush=True)
    cont = Contam(Path(a.items))
    print(f"  {len(cont.files)} item files, {len(cont.exact):,} segments, {len(cont.prefix):,} prefixes", file=sys.stderr)

    drops: dict[str, dict] = {}
    by_domain: dict[str, list[dict]] = defaultdict(list)
    fmt_of: dict[str, str] = {}
    seen_rows: set[str] = set()
    for (dom, lang), quota in QUOTA.items():
        quota = max(1, int(quota * a.scale))
        key = f"{dom}/{lang}"
        print(f"{key}: loading …", file=sys.stderr, flush=True)
        pool = load_source(dom, lang)
        n_pool = len(pool)
        clean, n_cont, n_dup, seen_pairs = [], 0, 0, set()
        for pr in pool:
            texts = [pr["p"], pr["h"]] + pr["orig"]
            if cont.hit(texts):
                n_cont += 1
                continue
            k = (norm(pr["p"]), norm(pr["h"]))
            if k in seen_pairs:
                n_dup += 1
                continue
            seen_pairs.add(k)
            clean.append(pr)
        rng = random.Random(f"{a.seed}/{key}")
        rng.shuffle(clean)
        take = clean[:quota]
        del pool
        n_rows = 0
        for pr in take:
            reps = RENDERS.get(dom, 1)
            fmts_used = set()
            for rep in range(reps):
                rrng = random.Random(f"{a.seed}/{key}/{pr['p'][:64]}/{pr['h']}/{rep}")
                for _ in range(8):  # re-draw so the second render differs in format
                    row, fmt = render_boolq(pr, rrng) if dom == "boolq" else render_nli(pr, rrng)
                    if dom == "boolq" or fmt[0] not in fmts_used or rep == 0:
                        break
                fmts_used.add(fmt[0])
                if row["id"] in seen_rows:
                    continue
                seen_rows.add(row["id"])
                by_domain[dom].append(row); fmt_of[row["id"]] = fmt; n_rows += 1
        drops[key] = {"pool": n_pool, "contaminated": n_cont, "duplicate_pairs": n_dup, "clean": len(clean),
                      "pairs_used": len(take), "rows": n_rows}
        print(f"  {key}: pool {n_pool:,}  contaminated {n_cont:,}  dup {n_dup:,}  used {len(take):,}  rows {n_rows:,}", file=sys.stderr, flush=True)

    # write, one file per domain, shuffled
    total = 0
    for dom, rows in sorted(by_domain.items()):
        random.Random(f"{a.seed}/write/{dom}").shuffle(rows)
        with open(out / f"{dom}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        total += len(rows)

    # stats
    rows = [r for rs in by_domain.values() for r in rs]
    lang_c = Counter(r["state_lang"] for r in rows)
    fmt_c = Counter(fmt_of[r["id"]] for r in rows)
    kind_c = Counter(r["kind"] for r in rows)
    cell = Counter((r["domain"], r["state_lang"], r["kind"], fmt_of[r["id"]]) for r in rows)
    info_in_q = sum(fmt_of[r["id"]] in ("A", "Cq", "Sq", "Q") for r in rows)
    stats = {
        "total": total, "by_lang": dict(lang_c.most_common()), "by_format": dict(fmt_c.most_common()), "by_kind": dict(kind_c),
        "info_in_question_share": round(info_in_q / max(total, 1), 3),
        "cross_lang_rows": sum(r["state_lang"] != r["choice_lang"] for r in rows),
        "sources": drops,
        "cells": [{"domain": d, "lang": l, "kind": k, "format": f, "n": n} for (d, l, k, f), n in sorted(cell.items())],
    }
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1))

    print(f"\n=== {total:,} rows → {out}/ ===")
    print("by lang:  " + "  ".join(f"{l} {n:,} ({n / total:.1%})" for l, n in lang_c.most_common()))
    print("by format:" + "  ".join(f" {f} {n:,} ({n / total:.1%})" for f, n in fmt_c.most_common()))
    print("by kind:  " + "  ".join(f"{k} {n:,}" for k, n in kind_c.most_common()))
    print(f"info in question (A/Cq/Sq/BoolQ): {info_in_q / total:.1%}")
    print("\ndecontamination / pool:")
    for k, d in drops.items():
        print(f"  {k:14s} pool {d['pool']:>8,}  contaminated {d['contaminated']:>6,}  dup {d['duplicate_pairs']:>6,}  used {d['pairs_used']:>7,}  rows {d['rows']:>7,}")
    print("\ndomain × lang × kind × format:")
    for (d, l, k, f), n in sorted(cell.items()):
        print(f"  {d:9s} {l:3s} {k:7s} {f:3s} {n:>7,}")
    print("\nsamples:")
    srng = random.Random(a.seed)
    by_fmt = defaultdict(list)
    for r in rows:
        by_fmt[fmt_of[r["id"]]].append(r)
    for f in sorted(by_fmt):
        print(f"\n--- format {f} ---")
        for r in srng.sample(by_fmt[f], min(3, len(by_fmt[f]))):
            s = dict(r); s["state"] = s["state"][:300]
            print(json.dumps(s, ensure_ascii=False))


if __name__ == "__main__":
    main()
