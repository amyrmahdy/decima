"""Exact-label data for knowledge-graph judgments: entity matching, mention pairs, pair relations, modality.

    uv run python -m teacher.kg_proc --n 120000 --out data/kg/proc-train.jsonl
    uv run python -m teacher.kg_proc --n 4000 --split test --exclude data/kg/proc-train.jsonl --out data/kg/proc-test.jsonl

The System One questions a KG pipeline asks (William Lyon's TypeSafe KG notebooks, TypeSafe's entity-alignment
cookbook), with labels computed from the generated case, never from a model:

- em       : two structured records ({"entity_a": {...}, "entity_b": {...}}) → 3-level link_state score
             (different / related variant / same) and per-field nouls (same name / maker / category / …).
             Same entity rendered with case, punctuation, abbreviations, legal suffixes, typos, units, scraped-text
             mojibake, dropped fields, and often byte-identical; variants differ in size / version / edition /
             vintage / model / location; different = same maker other product, shared-word names, or unrelated.
- mention  : two mentions with context ({"mention_a": {"text", "type", "context"}, ...}) → link_state score and
             same_name / same_context / abbreviation nouls; aliases, legal suffixes, honorifics, possessives,
             acronyms, tickers, shared-word different entities, product variants, and an appended ontology rule
             ("In this domain, a {type} is: …") that can flip the answer.
- relpair  : {"a": ..., "b": ...} → alternatives noul and alternatives / complementary / same / unrelated choice.
- modality : {"sentence": ...} or a short {"document": ...} → asserted / hypothetical / negated / forward_looking,
             the `factual` noul (true only when asserted) and the edge gate's `supported` noul.

Every row renders exactly as decima.systemone builds it from a TypeSafe request (choice = "key: description",
noul with criteria = "instructions\\nTrue if: …\\nFalse if: …" over yes / no, score = the ordered level texts).
The test split draws half its cases from families held out of training: two record domains, some mention kinds,
category lists, modality cue words, and the last two instruction phrasings of every question type. The Beer
benchmark, the KG notebooks' corpora and their npm / pnpm, Python / Go, … pairs are never used for training.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import Counter
from pathlib import Path

from teacher.agent_proc import choice_opts, noul_text, sid

SOURCE = "kg-proc"
DEBUG = bool(os.environ.get("KG_DEBUG"))          # tag rows with the generating case, for audits

# Names in the kgx corpora (documents, conversations, shopping, travel) and the Beer benchmark: never generated.
BLOCK = {"northwind", "aurora", "halcyon", "priya", "raman", "meridian", "vasquez", "kestrel", "webb", "ferreira",
         "duarte", "cascade", "skyline", "vellum", "torrent", "shah", "vantage", "brightline", "tidewater", "kanda",
         "compass", "typhoon", "meilin", "whitlock", "rheinblick", "summit", "sandoval", "zephyr", "cobalt", "delgado",
         "longacre", "rowe", "okada", "ashcroft", "aldgate", "solent", "blackmere", "redwood", "okonkwo", "iyer",
         "ironbridge", "ingersoll", "horizon", "harbour", "bayfront", "marina", "quay", "magellan", "kinetic",
         "kuhnhenn", "kniksen", "boneyard", "beer", "brewery", "brewing", "ale", "lager", "stout", "ipa", "porter",
         "marcus", "elena", "torrent", "nwl", "hlcn", "telematics", "berlin", "confluence", "cypher", "neo4j", "jenkins"}


def ok(s: str) -> bool:
    return not any(w in BLOCK for w in re.findall(r"[a-z0-9]+", s.lower()))


def row(kind, state, question, choices, probs, task, i, split) -> dict:
    s = sum(probs)
    probs = [round(p / s, 5) for p in probs]
    st = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    return {"kind": kind, "state": st, "question": question, "choices": choices, "probs": probs,
            "gold": max(range(len(probs)), key=probs.__getitem__), "state_lang": "en", "choice_lang": "en",
            "id": sid("kg", split, task, i, st, question), "source": SOURCE, "task": task}


def pg(rng, lo=0.92, hi=0.96):
    return round(rng.uniform(lo, hi), 3)


def verify(rng, state, ins, truth, task, i, split, crit=None, p=None):
    p = p or pg(rng)
    return row("verify", state, noul_text(ins, crit), ["yes", "no"], [p, 1 - p] if truth else [1 - p, p], task, i, split)


def score(rng, state, ins, levels, gold, task, i, split, p=None, nb=None):
    """Ordered levels; a little mass on the neighbour(s) of the gold level."""
    k = len(levels)
    p = p or pg(rng)
    nb = (1 - p) * 0.8 if nb is None else nb
    probs = [0.0] * k
    probs[gold] = p
    nbs = [j for j in (gold - 1, gold + 1) if 0 <= j < k]
    for j in nbs:
        probs[j] = nb / len(nbs)
    far = [j for j in range(k) if probs[j] == 0.0]
    rest = max(1 - p - nb, 0.004)
    for j in far:
        probs[j] = rest / len(far)
    return row("score", state, ins, levels, probs, task, i, split)


def choose(rng, state, ins, crit: dict, gold_key, task, i, split, p=None, soft: dict | None = None):
    p = p or pg(rng)
    keys = list(crit)
    probs = [p if k == gold_key else (1 - p) / (len(keys) - 1) for k in keys]
    if soft:                                                   # move some of the gold's mass to a named neighbour
        for k, m in soft.items():
            probs[keys.index(gold_key)] -= m
            probs[keys.index(k)] += m
    return row("choose", state, ins, choice_opts(crit), probs, task, i, split)


def pool(rng, lst, test):
    """Training phrasings are all but the last two; the test split uses the held-out two half of the time."""
    return rng.choice(lst[-2:] if test and rng.random() < 0.5 else lst[:-2])


# ================================================================== shared name material

ONS = ["b", "br", "c", "cr", "d", "dr", "f", "fl", "g", "gr", "h", "k", "l", "m", "n", "p", "pr", "qu", "r", "s",
       "sk", "st", "t", "tr", "v", "w", "z", "j", "sh", "th", "pl", "gl"]
VOW = ["a", "e", "i", "o", "u", "a", "e", "o", "ai", "ea", "io", "y"]
COD = ["n", "r", "x", "l", "s", "nd", "rk", "th", "", "", "m", "v", "nt", "st"]


def coin(rng, syl=None) -> str:
    while True:
        n = syl or rng.choice([2, 2, 2, 3])
        w = "".join(rng.choice(ONS) + rng.choice(VOW) for _ in range(n)) + rng.choice(COD)
        w = w.capitalize()
        if 4 <= len(w) <= 10 and ok(w):
            return w


P1 = ["Bramble", "Stone", "Silver", "Copper", "Ash", "Elm", "Fern", "Granite", "Hollow", "Juniper", "Larch", "Maple",
      "Oak", "Pine", "Raven", "Saddle", "Thistle", "Willow", "Wren", "Amber", "Briar", "Cedar", "Clover", "Falcon",
      "Heron", "Lark", "Moss", "Otter", "Pebble", "Rook", "Sparrow", "Tansy", "Bell", "Crane", "Dune", "Ember", "Flint",
      "Gale", "Hazel", "Marten", "Quill", "Rush", "Sorrel", "Teal", "Vale", "Yarrow", "Alder", "Birch", "Holly", "Kite"]
P2 = ["wood", "field", "gate", "brook", "ridge", "crest", "ford", "haven", "stone", "moor", "wick", "ley", "hurst",
      "combe", "bury", "dale", "fell", "marsh", "well", "worth", "ton", "by", "thorpe", "mere"]


def stem(rng) -> str:
    """A company / brand stem: compound English, coined, or two words."""
    while True:
        k = rng.random()
        if k < 0.4:
            s = rng.choice(P1) + rng.choice(P2)
        elif k < 0.75:
            s = coin(rng)
        else:
            s = rng.choice(P1) + " " + rng.choice(["Ridge", "Point", "Bay", "Hill", "Lake", "Valley", "Creek", "Peak", "Grove", "Field", "Crossing", "Cove"])
        if ok(s):
            return s


FIRST = ["Lena", "Daniel", "Amara", "Jonas", "Sofia", "Mateo", "Ingrid", "Kwame", "Yuki", "Rafael", "Nadia", "Tobias",
         "Leila", "Henrik", "Chiara", "Samuel", "Aisha", "Viktor", "Mei", "Oscar", "Farah", "Lukas", "Imani", "Pablo",
         "Hana", "Arjun", "Greta", "Emeka", "Sara", "Felix", "Noor", "Adrian", "Zara", "Kenji", "Olga", "Diego",
         "Maren", "Tariq", "Elif", "Bruno", "Ayla", "Stefan", "Rosa", "Idris", "Clara", "Nikolai", "Anika", "Omar",
         "Beatriz", "Hugo", "Laila", "Ravi", "Josefine", "Malik", "Selin", "Teodor", "Wanjiru", "Erik", "Lucía", "René"]
LAST = ["Okafor", "Lindqvist", "Moreau", "Haddad", "Nakamura", "Petrov", "Abara", "Castellano", "Brandt", "Novak",
        "Asante", "Kovacs", "Mensah", "Takahashi", "Rossi", "Eriksen", "Farouk", "Gallagher", "Hoang", "Ivanova",
        "Jansen", "Kariuki", "Lemaire", "Mbeki", "Nordin", "Osei", "Pereira", "Quintero", "Rahimi", "Sato",
        "Tanaka", "Ulrich", "Varga", "Wojcik", "Yilmaz", "Zielinski", "Achebe", "Bianchi", "Cohen", "Dimitrov",
        "Engel", "Fischer", "Grunwald", "Hayashi", "Ibrahim", "Jaramillo", "Keller", "Laurent", "Mwangi", "Nilsen",
        "Olsen", "Park", "Reyes", "Schulz", "Toivonen", "Usman", "Vogel", "Weiss", "Müller", "Lefèvre"]
FIRST = [x for x in FIRST if ok(x)]
LAST = [x for x in LAST if ok(x)]


def person(rng) -> tuple:
    return (rng.choice(FIRST), rng.choice("ABCDEFGHJKLMNPRSTW") if rng.random() < 0.3 else None, rng.choice(LAST))


def r_person(rng, p, initials=True) -> str:
    f, m, l = p
    forms = [f"{f} {l}", f"{f} {l}", f"{l}, {f}"]
    if m:
        forms += [f"{f} {m}. {l}", f"{l}, {f} {m}."]
    if initials:
        forms += [f"{f[0]}. {l}", f"{l}, {f[0]}."] + ([f"{f[0]}. {m}. {l}"] if m else [])
    s = rng.choice(forms)
    return s.upper() if rng.random() < 0.04 else s


# ================================================================== string noise

ABBR = [("Company", "Co."), ("Incorporated", "Inc."), ("Limited", "Ltd."), ("Corporation", "Corp."), ("and", "&"),
        ("Street", "St."), ("Avenue", "Ave."), ("Road", "Rd."), ("Boulevard", "Blvd."), ("Saint", "St."),
        ("Mount", "Mt."), ("International", "Intl."), ("Brothers", "Bros."), ("Edition", "Ed."), ("Volume", "Vol."),
        ("University", "Univ."), ("Laboratories", "Labs"), ("Doctor", "Dr."), ("Professor", "Prof."), ("Number", "No."),
        ("Department", "Dept."), ("Institute", "Inst."), ("Manufacturing", "Mfg."), ("Associates", "Assoc."),
        ("Technologies", "Tech."), ("Lane", "Ln."), ("Drive", "Dr."), ("Square", "Sq."), ("Place", "Pl.")]
MOJI = {"é": "Ã©", "è": "Ã¨", "ü": "Ã¼", "ö": "Ã ¶", "á": "Ã¡", "í": "Ã­", "ô": "Ã´", "â": "Ã¢", "ç": "Ã§", "ñ": "Ã±"}
KEYB = {"a": "s", "s": "a", "e": "r", "r": "e", "i": "o", "o": "i", "n": "m", "m": "n", "t": "r", "l": "k", "c": "v", "u": "y"}
PROTECT = {"Pro", "Max", "Plus", "Mini", "Lite", "Ultra", "Reserve", "Decaf", "Live", "Acoustic", "Remix", "Edition",
           "Special", "Limited", "Cask", "Strength", "Extended", "Version", "XR", "ER", "Junior", "Jr", "Sr", "Europe",
           "Holdings", "Gen", "Radio", "Edit", "Remastered", "Late", "Harvest"}


def abbr_toggle(rng, s: str) -> str:
    for lo, sh in rng.sample(ABBR, len(ABBR)):
        if rng.random() < 0.5:
            if re.search(rf"\b{re.escape(lo)}\b", s):
                s = re.sub(rf"\b{re.escape(lo)}\b", sh, s, count=1)
            elif sh != "&" and re.search(rf"(?<!\w){re.escape(sh)}(?!\w)", s):
                s = re.sub(rf"(?<!\w){re.escape(sh)}(?!\w)", lo, s, count=1)
            elif sh == "&" and " & " in s:
                s = s.replace(" & ", " and ", 1)
    return s


def typo(rng, s: str) -> str:
    words = s.split(" ")
    idx = [j for j, w in enumerate(words) if w.isalpha() and len(w) >= 5 and w not in PROTECT]
    if not idx:
        return s
    j = rng.choice(idx)
    w = words[j]
    k = rng.randint(1, len(w) - 2)
    op = rng.choice(["swap", "drop", "dup", "key"])
    if op == "swap":
        w = w[:k] + w[k + 1] + w[k] + w[k + 2:]
    elif op == "drop":
        w = w[:k] + w[k + 1:]
    elif op == "dup":
        w = w[:k] + w[k] + w[k:]
    else:
        w = w[:k] + KEYB.get(w[k].lower(), w[k]) + w[k + 1:]
    words[j] = w
    out = " ".join(words)
    return out if ok(out) else s


def dirty(rng, s: str) -> str:
    """Scraped-catalogue damage: HTML entities, split apostrophes, mis-decoded UTF-8."""
    if "(" in s and rng.random() < 0.6:
        s = s.replace("(", "&#40; ").replace(")", " &#41;")
    if "'" in s and rng.random() < 0.6:
        s = s.replace("'", " '")
    if "&" in s and "&#" not in s and rng.random() < 0.5:
        s = s.replace("&", "&amp;")
    if any(c in s for c in MOJI) and rng.random() < 0.7:
        for a, b in MOJI.items():
            s = s.replace(a, b)
    return s


def restyle(rng, s: str, strength: float = 1.0) -> str:
    """A formatting-only change: case, punctuation, whitespace, abbreviation."""
    k = rng.random() / max(strength, 1e-6)
    if k < 0.18:
        s = s.lower()
    elif k < 0.24:
        s = s.upper()
    elif k < 0.34:
        s = " ".join(w if any(c.isdigit() for c in w) else w.capitalize() for w in s.split(" "))
    elif k < 0.48:
        s = abbr_toggle(rng, s)
    elif k < 0.56:
        s = re.sub(r"[.,']", "", s)
    elif k < 0.62:
        s = s.replace("-", " ") if "-" in s else s.replace(" ", "  ", 1)
    elif k < 0.68:
        s = dirty(rng, s)
    return re.sub(r"\s+$", "", s)


# ================================================================== unit renderers (canonical value → surface)

def r_pct(rng, v):
    return rng.choice([f"{v:.1f}%", f"{v:.1f} %", f"{v:.2f} %", f"{v:.1f}% ABV", f"{v:.1f} % vol", f"{v:g}%", f"ABV {v:.1f}%"])


def r_money(rng, v):
    return rng.choice([f"${v:.2f}", f"{v:.2f} USD", f"USD {v:.2f}", f"{v:.2f}", f"$ {v:.2f}", f"US${v:.2f}"])


WEIGHT = {250: ["250 g", "250g", "0.25 kg", "8.8 oz"], 340: ["340 g", "340g", "12 oz", "12oz"],
          454: ["454 g", "1 lb", "16 oz", "454g"], 500: ["500 g", "500g", "0.5 kg", "1.1 lb"],
          907: ["907 g", "2 lb", "2 lbs", "32 oz"], 1000: ["1 kg", "1000 g", "1kg", "2.2 lb"]}
VOLUME = {375: ["375 ml", "375ml", "37.5 cl", "0.375 L"], 700: ["700 ml", "70 cl", "0.7 L", "70cl"],
          750: ["750 ml", "750ml", "75 cl", "0.75 L", "750 mL"], 1000: ["1 L", "1000 ml", "100 cl", "1 litre"],
          1500: ["1.5 L", "1500 ml", "150 cl", "1.5 litre"]}


def r_storage(rng, gb):
    if gb >= 1024:
        t = gb // 1024
        return rng.choice([f"{t}TB", f"{t} TB", f"{gb} GB", f"{t} tb"])
    return rng.choice([f"{gb}GB", f"{gb} GB", f"{gb} gb", f"{gb}gb"])


def r_phone(rng, d):
    a, b, c = d[:3], d[3:6], d[6:]
    return rng.choice([f"{a}-{b}-{c}", f"({a}) {b}-{c}", f"{a}/{b}-{c}", f"+1 {a} {b} {c}", f"{a}{b}{c}", f"{a}.{b}.{c}"])


STREET_T = {"Street": ["Street", "St.", "St"], "Avenue": ["Avenue", "Ave.", "Ave"], "Road": ["Road", "Rd.", "Rd"],
            "Boulevard": ["Boulevard", "Blvd.", "Blvd"], "Lane": ["Lane", "Ln."], "Drive": ["Drive", "Dr."], "Place": ["Place", "Pl."]}


def r_addr(rng, a):
    num, st, t = a
    s = f"{num} {st} {rng.choice(STREET_T[t])}"
    return restyle(rng, s, 0.5) if rng.random() < 0.3 else s


CITY = {"nyc": ["New York", "New York City", "NYC", "New York, NY"], "la": ["Los Angeles", "LA", "Los Angeles, CA"],
        "sf": ["San Francisco", "SF", "San Francisco, CA"], "stpaul": ["Saint Paul", "St. Paul", "St Paul, MN"],
        "vienna": ["Vienna", "Wien"], "munich": ["Munich", "München"], "prague": ["Prague", "Praha"],
        "lyon": ["Lyon", "Lyon, France"], "austin": ["Austin", "Austin, TX"], "denver": ["Denver", "Denver, CO"],
        "boston": ["Boston", "Boston, MA"], "toronto": ["Toronto", "Toronto, ON"], "madrid": ["Madrid"],
        "copenhagen": ["Copenhagen", "København"], "milan": ["Milan", "Milano"], "zurich": ["Zurich", "Zürich"],
        "montreal": ["Montreal", "Montréal"], "melbourne": ["Melbourne", "Melbourne, VIC"], "osaka": ["Osaka"],
        "nashville": ["Nashville", "Nashville, TN"], "portland_or": ["Portland, OR", "Portland, Oregon"],
        "portland_me": ["Portland, ME", "Portland, Maine"], "stlouis": ["Saint Louis", "St. Louis"],
        "mumbai": ["Mumbai", "Bombay"], "hcmc": ["Ho Chi Minh City", "Saigon"], "krakow": ["Kraków", "Krakow", "Cracow"]}


def r_city(rng, c):
    return rng.choice(CITY[c])


def r_year(rng, y):
    return y if rng.random() < 0.3 else str(y)


def r_version(rng, v):
    return rng.choice([v, v, "v" + v, f"version {v}", f"{v} (stable)"])


def r_strength(rng, mg):
    f = [f"{mg:g} mg", f"{mg:g}mg", f"{mg:g} MG"]
    if mg >= 100 and mg % 100 == 0:
        f.append(f"{mg / 1000:g} g")
    return rng.choice(f)


def r_engine(rng, l):
    return rng.choice([f"{l:.1f}L", f"{l:.1f} L", f"{l:.1f}-liter", f"{int(round(l * 1000))} cc", f"{l:.1f} litre"])


def r_stars(rng, s):
    return rng.choice([str(s), f"{s}-star", f"{s} stars", "★" * s, s, f"{s}/5"])


def r_age(rng, a):
    return rng.choice([f"{a} Year Old", f"{a} YO", f"{a}-year-old", f"Aged {a} years", f"{a} yrs", f"{a}yo"])


def r_time(rng, s):
    m, ss = divmod(s, 60)
    return rng.choice([f"{m}:{ss:02d}", f"{m:02d}:{ss:02d}", f"{m}m {ss}s", f"{m}:{ss:02d} min"])


def r_url(rng, d):
    return rng.choice([d, "www." + d, "https://" + d, f"https://www.{d}/", "http://" + d])


def r_edition(rng, n):
    o = {1: ("1st", "First"), 2: ("2nd", "Second"), 3: ("3rd", "Third"), 4: ("4th", "Fourth"), 5: ("5th", "Fifth")}[n]
    return rng.choice([f"{o[0]} edition", f"{o[1]} Edition", f"{o[0]} ed.", f"{n}e", f"{o[0]} Ed."])


ORG_SUF = {"inc": ["Inc.", "Inc", "Incorporated", ", Inc."], "ltd": ["Ltd.", "Ltd", "Limited"], "co": ["Co.", "Company"],
           "corp": ["Corp.", "Corporation", "Corp"], "gmbh": ["GmbH"], "llc": ["LLC", "L.L.C."], "plc": ["plc", "PLC"],
           "sa": ["S.A.", "SA"], "ag": ["AG"], "bv": ["B.V.", "BV"]}


def r_org(rng, o, drop_suffix=0.3):
    """o = (stem, word, suffix id or None). The suffix may be dropped; the stem and word never change."""
    st, w, suf = o
    s = f"{st} {w}".strip()
    if suf and rng.random() > drop_suffix:
        f = rng.choice(ORG_SUF[suf])
        s = s + f if f.startswith(",") else s + " " + f
    return restyle(rng, s, 0.6)


def org_eq(a, b):
    return a is not None and b is not None and (a[0], a[1]) == (b[0], b[1])


# ================================================================== A. entity matching (record pairs)

class F:
    """A record field: canonical key, renderer kind, JSON key options, question word, role, optional?"""

    def __init__(self, c, kind, keys, word, role="other", opt=False, table=None):
        self.c, self.kind, self.keys, self.word, self.role, self.opt, self.table = c, kind, keys, word, role, opt, table


def render_val(rng, f: F, v):
    k = f.kind
    if k == "name":
        return restyle(rng, v)
    if k == "text":
        return restyle(rng, v, 0.6)
    if k == "org":
        return r_org(rng, v)
    if k == "person":
        return r_person(rng, v)
    if k == "persons":                                                  # one style for the whole list, so the separators stay unambiguous
        style = rng.choice(["full", "init", "lastfirst"])
        one = {"full": lambda p: f"{p[0]} {p[2]}", "init": lambda p: f"{p[0][0]}. {p[2]}", "lastfirst": lambda p: f"{p[2]}, {p[0][0]}."}[style]
        if len(v) >= 3 and rng.random() < 0.2:
            return one(v[0]) + " et al."
        names = [one(p) for p in v]
        if style == "lastfirst":
            return "; ".join(names)
        return rng.choice([", ", " and "]).join(names) if len(names) < 3 or rng.random() < 0.5 else ", ".join(names[:-1]) + " and " + names[-1]
    if k == "cat":
        s = rng.choice(f.table[v])
        return s.lower() if rng.random() < 0.15 else s
    return {"pct": r_pct, "money": r_money, "storage": r_storage, "phone": r_phone, "addr": r_addr, "city": r_city,
            "year": r_year, "version": r_version, "strength": r_strength, "engine": r_engine, "stars": r_stars,
            "age": r_age, "time": r_time, "url": r_url, "edition": r_edition,
            "weight": lambda r, x: r.choice(WEIGHT[x]), "volume": lambda r, x: r.choice(VOLUME[x]),
            "ticker": lambda r, x: r.choice([x[0], x[0], f"{x[1]}: {x[0]}", f"{x[1]}:{x[0]}"])}[k](rng, v)


def eqv(f: F, a, b) -> bool:
    if f.kind == "org":
        return org_eq(a, b)
    if f.kind == "person":
        return (a[0], a[2]) == (b[0], b[2])
    if f.kind == "persons":
        return [(p[0], p[2]) for p in a] == [(p[0], p[2]) for p in b]
    if f.kind in ("name", "text"):
        return a.casefold() == b.casefold()
    return a == b


STOP = {"the", "and", "for", "of", "on", "in", "with", "over", "les", "des"}


class Dom:
    name, noun, nouns, generic, maker, cat, test = "", "product", "products", True, "brand", "category", False
    fields: list = []

    def new(self, rng) -> dict: ...

    def variant(self, rng, e) -> dict: ...

    def sibling(self, rng, e) -> dict | None:
        return None

    def name_forms(self, rng, e) -> str:
        return e["name"]

    def near(self, rng, e):
        """A different entity whose name shares a word with e's."""
        for _ in range(20):
            o = self.new(rng)
            ws = [w for w in e["name"].split() if w.isalpha() and len(w) >= 3 and w not in PROTECT and w.lower() not in STOP]
            ow = o["name"].split()
            js = [j for j, w in enumerate(ow) if w.isalpha()]
            if ws and len(ow) >= 2 and js:
                j = rng.choice(js)
                ow[j] = rng.choice(ws)
                o["name"] = " ".join(ow)
            if o["name"].casefold() != e["name"].casefold():
                return o
        return self.new(rng)

    @property
    def maker_field(self):
        return next((f for f in self.fields if f.role == "maker"), None)


ELEC_CAT = {"laptop": ["Laptop", "Notebook computer", "Laptops", "Notebook"], "phone": ["Smartphone", "Mobile phone", "Cell phone", "Smartphones"],
            "headphones": ["Headphones", "Headphone", "Head phones"], "tablet": ["Tablet", "Tablet computer", "Tablets"],
            "monitor": ["Monitor", "Computer monitor", "Monitors", "Display"], "router": ["Wi-Fi router", "Wireless router", "Router"],
            "camera": ["Digital camera", "Camera", "Cameras"], "speaker": ["Bluetooth speaker", "Portable speaker", "Speaker"],
            "watch": ["Smartwatch", "Smart watch", "Smart Watch"], "keyboard": ["Keyboard", "Computer keyboard", "Keyboards"],
            "printer": ["Printer", "Printers"], "drive": ["External SSD", "Portable SSD", "External solid-state drive"]}
STORAGE_CATS = {"laptop", "phone", "tablet", "drive"}
COLORS = ["Black", "Silver", "Graphite", "Midnight Blue", "White", "Rose Gold", "Forest Green", "Sand"]
SUFFIX_V = ["Pro", "Plus", "Max", "Mini", "Lite", "Ultra"]


class Electronics(Dom):
    name, noun, nouns = "electronics", "product", "products"
    fields = [F("name", "name", ["name", "title", "product"], "name", "name"),
              F("brand", "org", ["brand", "manufacturer", "maker"], "brand", "maker"),
              F("cat", "cat", ["category", "type", "product_type"], "category", "cat", table=ELEC_CAT),
              F("storage", "storage", ["storage", "capacity"], "storage capacity", opt=True),
              F("color", "text", ["color", "colour", "finish"], "color", opt=True),
              F("price", "money", ["price"], "price", opt=True)]
    maker = "brand"

    def new(self, rng):
        c = rng.choice(list(ELEC_CAT))
        model = rng.choice([f"{rng.choice('ABCDEFGHKMNPQRSTVXZ')}{rng.randint(1, 99)}", str(rng.randint(2, 99)),
                            f"{rng.choice('ABCDEFGHKMNPQRSTVXZ')}{rng.randint(100, 990)}", f"{rng.choice('ABCDEFGHKMNPQRSTVXZ')}{rng.choice('ABCDEFGHKMNPQRSTVXZ')}-{rng.randint(10, 99)}"])
        line = coin(rng) if rng.random() < 0.6 else rng.choice(["Nova", "Pulse", "Echo", "Flux", "Vista", "Orbit", "Prism", "Arc", "Volt", "Strata", "Lumen", "Sonic", "Glide", "Core"])
        e = {"name": f"{line} {model}", "brand": (stem(rng), "", rng.choice([None, None, "inc", "corp", "ltd"])), "cat": c,
             "color": rng.choice(COLORS), "price": round(rng.uniform(19, 2400), 2)}
        if c in STORAGE_CATS:
            e["storage"] = rng.choice([64, 128, 256, 512, 1024, 2048])
        return e

    def name_forms(self, rng, e):
        return f"{e['brand'][0]} {e['name']}" if rng.random() < 0.35 else e["name"]

    def variant(self, rng, e):
        v = dict(e)
        ops = ["suffix", "model", "color"] + (["storage"] if "storage" in e else [])
        op = rng.choice(ops)
        if op == "suffix":
            w = e["name"].split()
            if w[-1] in SUFFIX_V:
                w[-1] = rng.choice([s for s in SUFFIX_V if s != w[-1]])
            else:
                w.append(rng.choice(SUFFIX_V))
            v["name"] = " ".join(w)
        elif op == "model":
            m = re.search(r"\d+", e["name"])
            n = int(m.group()) + rng.choice([-2, -1, 1, 2, 10])
            v["name"] = e["name"][:m.start()] + str(max(n, 1) if n != int(m.group()) else n + 1) + e["name"][m.end():]
            if v["name"] == e["name"]:
                v["name"] = e["name"] + " (2nd gen)"
        elif op == "color":
            v["color"] = rng.choice([c for c in COLORS if c != e["color"]])
        else:
            v["storage"] = rng.choice([s for s in [64, 128, 256, 512, 1024, 2048] if s != e["storage"]])
        return v

    def sibling(self, rng, e):
        o = self.new(rng)
        o["brand"] = e["brand"]
        return o


BOOK_GENRE = {"mystery": ["Mystery", "Mystery fiction", "Mysteries"], "scifi": ["Science fiction", "Sci-fi", "SF", "Science Fiction"],
              "history": ["History", "Nonfiction / History"], "romance": ["Romance", "Romance novels"],
              "fantasy": ["Fantasy", "Fantasy fiction"], "biography": ["Biography", "Biographies & Memoirs", "Biography/Memoir"],
              "cooking": ["Cooking", "Cookbooks", "Cookery"], "poetry": ["Poetry", "Poems"], "business": ["Business", "Business & Economics"],
              "children": ["Children's books", "Children's fiction", "Kids"]}
BOOK_FMT = {"hc": ["Hardcover", "Hardback", "HC"], "pb": ["Paperback", "Softcover", "PB", "Trade paperback"], "eb": ["Ebook", "E-book", "Digital edition"]}
T_ADJ = ["Quiet", "Silver", "Hidden", "Northern", "Broken", "Golden", "Crimson", "Distant", "Painted", "Wandering", "Bright",
         "Salt", "Paper", "Glass", "Velvet", "Winter", "Summer", "Last", "Lost", "Burning", "Silent", "Little", "Wild", "Hollow"]
T_NOUN = ["Orchard", "Lantern", "River", "Garden", "Archive", "Meadow", "Station", "Island", "Letters", "Atlas", "Engine",
          "Tide", "Feather", "Mirror", "Signal", "Forest", "Canyon", "Lighthouse", "Map", "Bridge", "Season", "Clock", "Kitchen",
          "Cartographer", "Orchestra", "Weaver", "Sparrow", "Ledger", "Voyage", "Shore"]


def pl(w: str) -> str:
    return w if w.endswith("s") else w + "s"


def book_title(rng):
    return rng.choice([f"The {rng.choice(T_ADJ)} {rng.choice(T_NOUN)}", f"{rng.choice(T_NOUN)} of {pl(rng.choice(T_NOUN))}",
                       f"A {rng.choice(T_NOUN)} in {coin(rng)}", f"The {rng.choice(T_NOUN)}'s {rng.choice(T_NOUN)}",
                       f"{rng.choice(T_ADJ)} {pl(rng.choice(T_NOUN))}", f"The {rng.choice(T_NOUN)} and the {rng.choice(T_NOUN)}",
                       f"{rng.choice(T_ADJ)} {rng.choice(T_NOUN)}: A Novel"])


class Books(Dom):
    name, noun, nouns, generic, maker, cat = "books", "book", "books", False, "author", "genre"
    fields = [F("name", "name", ["title", "name", "book_title"], "title", "name"),
              F("author", "person", ["author", "authors", "writer"], "author", "maker"),
              F("cat", "cat", ["genre", "category", "subject"], "genre", "cat", table=BOOK_GENRE),
              F("publisher", "org", ["publisher", "imprint"], "publisher", opt=True),
              F("fmt", "cat", ["format", "binding"], "format", opt=True, table=BOOK_FMT),
              F("edition", "edition", ["edition"], "edition", opt=True),
              F("year", "year", ["year", "published", "pub_year"], "publication year", opt=True)]

    def new(self, rng):
        return {"name": book_title(rng), "author": person(rng), "cat": rng.choice(list(BOOK_GENRE)),
                "publisher": (stem(rng), rng.choice(["Press", "Books", "Publishing", "House"]), rng.choice([None, None, "ltd", "inc"])),
                "fmt": rng.choice(list(BOOK_FMT)), "edition": rng.choice([1, 1, 1, 2, 3]), "year": rng.randint(1975, 2025)}

    def variant(self, rng, e):
        v = dict(e)
        if rng.random() < 0.5:
            v["edition"] = rng.choice([n for n in (1, 2, 3, 4, 5) if n != e["edition"]])
            v["year"] = e["year"] + rng.randint(2, 9)
        else:
            v["fmt"] = rng.choice([f for f in BOOK_FMT if f != e["fmt"]])
        return v

    def sibling(self, rng, e):
        o = self.new(rng)
        o["author"] = e["author"]
        if rng.random() < 0.3:
            o["name"] = e["name"] + rng.choice([": Book Two", ", Volume 2", " II"])
        return o


GRAPE = {"pinot_noir": ["Pinot Noir", "Pinot noir", "Spätburgunder"], "cab": ["Cabernet Sauvignon", "Cab. Sauvignon", "Cabernet-Sauvignon"],
         "chardonnay": ["Chardonnay", "Chard."], "riesling": ["Riesling"], "syrah": ["Syrah", "Shiraz"], "grenache": ["Grenache", "Garnacha"],
         "pinot_gris": ["Pinot Gris", "Pinot Grigio"], "sauv_blanc": ["Sauvignon Blanc", "Sauv. Blanc"], "tempranillo": ["Tempranillo", "Tinto Fino"],
         "malbec": ["Malbec", "Côt"], "zinfandel": ["Zinfandel", "Primitivo"], "merlot": ["Merlot"], "chenin": ["Chenin Blanc", "Steen"]}
REGIONS = ["Côte de Vrell", "Valle Norte", "Hollow Hills AVA", "Sankt Ilm", "Coastal Ranges", "Alta Serra", "Rive Gauche", "Kettle Valley"]


class Wine(Dom):
    name, noun, nouns, generic, maker, cat = "wine", "wine", "wines", False, "winery", "grape variety"
    fields = [F("name", "name", ["name", "wine", "label"], "name", "name"),
              F("winery", "org", ["winery", "producer", "estate"], "winery", "maker"),
              F("cat", "cat", ["varietal", "grape", "variety"], "grape variety", "cat", table=GRAPE),
              F("vintage", "year", ["vintage", "year"], "vintage", opt=True),
              F("abv", "pct", ["abv", "alcohol"], "alcohol content", opt=True),
              F("region", "text", ["region", "appellation"], "region", opt=True),
              F("bottle", "volume", ["bottle", "size", "volume"], "bottle size", opt=True)]

    def new(self, rng):
        return {"name": rng.choice([f"{stem(rng)} {rng.choice(['Old Vine', 'Hillside', 'Estate', 'Valley Floor', 'Block 7', 'Les Pierres', 'Clos', 'Terraces', 'Single Vineyard', 'Cuvée Anna'])}",
                                    f"Château {coin(rng)}", f"Domaine {coin(rng)} {rng.choice(['Rouge', 'Blanc', 'Vieilles Vignes', 'Côte'])}", f"{coin(rng)} {rng.choice(T_NOUN)}"]),
                "winery": (stem(rng), rng.choice(["Estate", "Vineyards", "Winery", "Cellars", "Wines", ""]), rng.choice([None, None, None, "ltd", "sa"])),
                "cat": rng.choice(list(GRAPE)), "vintage": rng.randint(2008, 2023), "abv": round(rng.uniform(11.5, 15.5) * 2) / 2,
                "region": rng.choice(REGIONS), "bottle": rng.choice([750, 750, 750, 375, 1500])}

    def variant(self, rng, e):
        v = dict(e)
        k = rng.random()
        if k < 0.5:
            v["vintage"] = e["vintage"] + rng.choice([-3, -2, -1, 1, 2, 3])
        elif k < 0.8:
            v["name"] = e["name"] + " " + rng.choice(["Reserve", "Special Edition", "Late Harvest", "Limited Release"])
        else:
            v["bottle"] = rng.choice([b for b in (375, 750, 1500) if b != e["bottle"]])
        return v

    def sibling(self, rng, e):
        o = self.new(rng)
        o["winery"] = e["winery"]
        return o


ORIGIN = {"ethiopia": ["Ethiopia", "Ethiopian"], "colombia": ["Colombia", "Colombian"], "kenya": ["Kenya", "Kenyan"],
          "guatemala": ["Guatemala", "Guatemalan"], "brazil": ["Brazil", "Brazilian"], "sumatra": ["Sumatra", "Indonesia (Sumatra)"],
          "costa_rica": ["Costa Rica", "Costa Rican"], "rwanda": ["Rwanda", "Rwandan"], "peru": ["Peru", "Peruvian"], "yemen": ["Yemen", "Yemeni"]}
ROAST = {"light": ["Light", "Light roast", "Cinnamon"], "medium": ["Medium", "Medium roast", "City"], "dark": ["Dark", "Dark roast", "French roast"]}
PROCESS = {"washed": ["Washed", "Wet-processed", "Fully washed"], "natural": ["Natural", "Dry-processed", "Sun-dried natural"], "honey": ["Honey", "Honey process", "Pulped natural"]}


class Coffee(Dom):
    name, noun, nouns, generic, maker, cat = "coffee", "coffee", "coffees", False, "roaster", "origin"
    fields = [F("name", "name", ["name", "coffee", "blend"], "name", "name"),
              F("roaster", "org", ["roaster", "brand", "roastery"], "roaster", "maker"),
              F("cat", "cat", ["origin", "country"], "origin", "cat", table=ORIGIN),
              F("roast", "cat", ["roast", "roast_level"], "roast level", opt=True, table=ROAST),
              F("process", "cat", ["process", "processing"], "processing method", opt=True, table=PROCESS),
              F("weight", "weight", ["weight", "size", "net_weight"], "bag size", opt=True),
              F("price", "money", ["price"], "price", opt=True)]

    def new(self, rng):
        return {"name": rng.choice([f"{rng.choice(T_ADJ)} {rng.choice(T_NOUN)}", f"{coin(rng)} {rng.choice(['Estate', 'Single Origin', 'Microlot', 'Reserve Lot', 'Blend', 'Espresso'])}",
                                    f"{rng.choice(['Morning', 'House', 'Night Owl', 'Harvest', 'Sunrise', 'Hearth'])} {rng.choice(['Blend', 'Roast', 'Espresso'])}"]),
                "roaster": (stem(rng), rng.choice(["Coffee Roasters", "Roasting Company", "Coffee Company", "Roastery", "Coffee"]), rng.choice([None, None, "llc", "ltd"])),
                "cat": rng.choice(list(ORIGIN)), "roast": rng.choice(list(ROAST)), "process": rng.choice(list(PROCESS)),
                "weight": rng.choice(list(WEIGHT)), "price": round(rng.uniform(9, 45), 2)}

    def variant(self, rng, e):
        v = dict(e)
        k = rng.random()
        if k < 0.45:
            v["weight"] = rng.choice([w for w in WEIGHT if w != e["weight"]])
        elif k < 0.75:
            v["name"] = e["name"] + " " + rng.choice(["Decaf", "Special Edition", "Holiday Edition"])
        else:
            v["roast"] = rng.choice([r for r in ROAST if r != e["roast"]])
        return v

    def sibling(self, rng, e):
        o = self.new(rng)
        o["roaster"] = e["roaster"]
        return o


CUISINE = {"italian": ["Italian", "Italian cuisine"], "mexican": ["Mexican", "Mexican food"], "japanese": ["Japanese", "Japanese cuisine"],
           "thai": ["Thai"], "indian": ["Indian", "Indian cuisine"], "french": ["French", "French cuisine"], "american": ["American", "American cuisine"],
           "chinese": ["Chinese", "Chinese food"], "vietnamese": ["Vietnamese"], "greek": ["Greek", "Greek / Mediterranean"],
           "ethiopian": ["Ethiopian"], "korean": ["Korean", "Korean BBQ"], "lebanese": ["Lebanese", "Middle Eastern (Lebanese)"]}
STREETS = ["Maple", "Juniper", "Harrow", "Lindell", "Mercer", "Beacon", "Calloway", "Prescott", "Garnet", "Wexford",
           "Orchard", "Halsey", "Kingsley", "Dorset", "Fairmount", "Larkin", "Union", "Elmwood", "Tremont", "Belmont"]


def addr(rng):
    return (rng.randint(2, 2999), rng.choice(STREETS), rng.choice(list(STREET_T)))


class Restaurants(Dom):
    name, noun, nouns, generic, maker, cat = "restaurants", "restaurant", "restaurants", False, None, "cuisine"
    fields = [F("name", "name", ["name", "restaurant"], "name", "name"),
              F("addr", "addr", ["address", "addr", "street"], "street address"),
              F("city", "city", ["city"], "city"),
              F("phone", "phone", ["phone", "telephone", "tel"], "phone number", opt=True),
              F("cat", "cat", ["cuisine", "type", "category"], "cuisine", "cat", table=CUISINE)]

    def new(self, rng):
        kind = rng.choice(["Trattoria", "Bistro", "Kitchen", "Grill", "Noodle House", "Taqueria", "Brasserie", "Diner", "Cantina",
                           "Tavern", "Cafe", "Supper Club", "Eatery", "Oyster Bar"])
        return {"name": rng.choice([f"{rng.choice(FIRST)}'s {kind}", f"The {rng.choice(T_ADJ)} {rng.choice(T_NOUN)}", f"{coin(rng)} {kind}",
                                    f"{rng.choice(T_NOUN)} & {rng.choice(T_NOUN)}", f"Casa {coin(rng)}", f"Chez {rng.choice(FIRST)}"]),
                "addr": addr(rng), "city": rng.choice(list(CITY)), "phone": f"{rng.randint(201, 989)}{rng.randint(200, 999)}{rng.randint(0, 9999):04d}",
                "cat": rng.choice(list(CUISINE))}

    def variant(self, rng, e):          # another branch of the same restaurant
        v = dict(e)
        v["addr"] = addr(rng)
        v["phone"] = f"{e['phone'][:3]}{rng.randint(200, 999)}{rng.randint(0, 9999):04d}"
        if rng.random() < 0.4:
            v["city"] = rng.choice([c for c in CITY if c != e["city"]])
        return v


ECO = {"pypi": ["PyPI", "pypi", "Python Package Index"], "npm": ["npm", "npmjs", "npm registry"], "crates": ["crates.io", "Cargo (crates.io)"],
       "maven": ["Maven Central", "Maven"], "gems": ["RubyGems", "rubygems.org"], "nuget": ["NuGet", "nuget.org"]}
LICENSE = {"mit": ["MIT", "MIT License"], "apache": ["Apache-2.0", "Apache License 2.0", "Apache 2.0"], "gpl3": ["GPL-3.0", "GPLv3", "GNU GPL v3"],
           "bsd3": ["BSD-3-Clause", "BSD 3-Clause", "New BSD"], "mpl": ["MPL-2.0", "Mozilla Public License 2.0"]}
PK_W = ["json", "yaml", "http", "cache", "log", "queue", "date", "color", "path", "test", "mock", "cli", "config", "graph",
        "plot", "crypto", "image", "mail", "task", "schema", "retry", "token", "geo", "csv", "diff", "glob", "lint", "trace"]


class Packages(Dom):
    name, noun, nouns, generic, maker, cat = "software", "package", "packages", False, "maintainer", "registry"
    fields = [F("name", "name", ["name", "package"], "package name", "name"),
              F("maintainer", "org", ["maintainer", "publisher", "owner"], "maintainer", "maker"),
              F("cat", "cat", ["registry", "ecosystem"], "registry", "cat", table=ECO),
              F("version", "version", ["version", "latest_version"], "version"),
              F("license", "cat", ["license", "licence"], "license", opt=True, table=LICENSE)]

    def new(self, rng):
        a, b = rng.sample(PK_W, 2)
        nm = rng.choice([f"{a}{rng.choice(['kit', 'lib', 'ify', 'er', 'io', 'x', 'ly'])}", f"{rng.choice(['tiny', 'fast', 'easy', 'micro', 'lit', 'super', 'smart'])}{a}",
                         f"{a}-{b}", f"{a}_{b}", f"{coin(rng).lower()}-{a}"])
        return {"name": nm, "maintainer": (rng.choice([coin(rng).lower(), stem(rng), f"{rng.choice(FIRST).lower()}{rng.choice(LAST).lower()[:4]}"]), "", None),
                "cat": rng.choice(list(ECO)), "version": f"{rng.randint(0, 9)}.{rng.randint(0, 30)}.{rng.randint(0, 20)}",
                "license": rng.choice(list(LICENSE))}

    def name_forms(self, rng, e):
        n = e["name"]
        if e["cat"] == "pypi" and rng.random() < 0.4:               # PyPI normalises - _ . and case
            n = re.sub(r"[-_.]", rng.choice(["-", "_", "."]), n)
        return n

    def variant(self, rng, e):
        v = dict(e)
        if rng.random() < 0.7:
            ma, mi, pa = map(int, e["version"].split("."))
            v["version"] = rng.choice([f"{ma + 1}.0.0", f"{ma}.{mi + 1}.0", f"{ma}.{mi}.{pa + 1}", f"{max(ma - 1, 0)}.{mi + 3}.{pa}"])
            if v["version"] == e["version"]:
                v["version"] = f"{ma + 2}.0.0"
        else:                                                           # the same maintainer's port to another registry
            v["cat"] = rng.choice([c for c in ECO if c != e["cat"]])
        return v

    def sibling(self, rng, e):
        o = self.new(rng)
        o["maintainer"] = e["maintainer"]
        return o

    def near(self, rng, e):
        if rng.random() < 0.4:                                          # same name, another registry, another maintainer
            o = self.new(rng)
            o["name"] = e["name"]
            o["cat"] = rng.choice([c for c in ECO if c != e["cat"]])
            return o
        return super().near(rng, e)


INDUSTRY = {"logistics": ["Logistics", "Freight", "Shipping"], "pharma": ["Pharma", "Therapeutics", "Biosciences"],
            "software": ["Software", "Systems", "Labs"], "energy": ["Energy", "Power", "Renewables"], "food": ["Foods", "Farms", "Provisions"],
            "bank": ["Bank", "Capital", "Financial"], "mining": ["Mining", "Resources", "Minerals"], "retail": ["Retail", "Stores", "Outfitters"],
            "telecom": ["Telecom", "Networks", "Communications"], "aero": ["Aerospace", "Aviation", "Dynamics"],
            "semis": ["Semiconductor", "Microdevices", "Circuits"], "insurance": ["Insurance", "Assurance", "Mutual"],
            "media": ["Media", "Studios", "Broadcasting"], "construction": ["Construction", "Builders", "Infrastructure"],
            "chemicals": ["Chemicals", "Materials", "Polymers"]}
IND_CAT = {"logistics": ["Logistics", "Transportation & Logistics", "Freight"], "pharma": ["Pharmaceuticals", "Pharma", "Biotech & Pharma"],
           "software": ["Software", "Enterprise software", "IT / Software"], "energy": ["Energy", "Utilities & Energy"],
           "food": ["Food & Beverage", "Food production"], "bank": ["Banking", "Financial services", "Banks"], "mining": ["Mining", "Metals & Mining"],
           "retail": ["Retail", "Consumer retail"], "telecom": ["Telecommunications", "Telecom"], "aero": ["Aerospace & Defense", "Aerospace"],
           "semis": ["Semiconductors", "Chips"], "insurance": ["Insurance"], "media": ["Media & Entertainment", "Media"],
           "construction": ["Construction", "Engineering & Construction"], "chemicals": ["Chemicals", "Specialty chemicals"]}
HQ = ["austin", "denver", "boston", "toronto", "madrid", "copenhagen", "milan", "zurich", "montreal", "melbourne", "osaka", "nashville",
      "vienna", "munich", "prague", "lyon", "nyc", "sf", "la", "stlouis", "mumbai", "krakow"]


def ticker_of(rng, name):
    letters = [c for c in name.upper() if c.isalpha()]
    while True:
        idx = sorted(rng.sample(range(1, len(letters)), min(len(letters) - 1, rng.choice([2, 3]))))   # letters keep their order
        t = letters[0] + "".join(letters[j] for j in idx)
        if ok(t):
            return t


class Companies(Dom):
    name, noun, nouns, generic, maker, cat = "companies", "company", "companies", False, None, "industry"
    fields = [F("org", "org", ["name", "company", "legal_name"], "company name", "name"),
              F("hq", "city", ["headquarters", "hq", "city"], "headquarters city"),
              F("cat", "cat", ["industry", "sector"], "industry", "cat", table=IND_CAT),
              F("ticker", "ticker", ["ticker", "symbol"], "ticker symbol", opt=True),
              F("url", "url", ["website", "url", "domain"], "website", opt=True),
              F("founded", "year", ["founded", "founded_year"], "founding year", opt=True)]

    def new(self, rng):
        ind = rng.choice(list(INDUSTRY))
        st = stem(rng)
        o = (st, rng.choice(INDUSTRY[ind]), rng.choice(["inc", "corp", "ltd", "plc", "gmbh", "sa", "llc", "ag", None]))
        return {"org": o, "name": f"{o[0]} {o[1]}", "hq": rng.choice(HQ), "cat": ind,
                "ticker": (ticker_of(rng, st), rng.choice(["NYSE", "NASDAQ", "LSE", "TSX", "XETRA", "SIX"])),
                "url": re.sub(r"[^a-z]", "", st.lower()) + rng.choice([".com", ".io", ".co", ".de", ".net"]), "founded": rng.randint(1890, 2020)}

    def variant(self, rng, e):          # a subsidiary or regional arm of the same group
        v = dict(e)
        tag = rng.choice(["Europe", "Holdings", "UK", "Americas", "Asia Pacific", "Nordic", "International"])
        v["org"] = (e["org"][0], f"{e['org'][1]} {tag}", rng.choice(["gmbh", "ltd", "bv", "llc", "inc"]))
        v["name"] = f"{v['org'][0]} {v['org'][1]}"
        v["hq"] = rng.choice([c for c in HQ if c != e["hq"]])
        v.pop("ticker", None)
        return v

    def near(self, rng, e):             # shares the stem, another industry: "Apex Logistics" vs "Apex Pharma"
        o = self.new(rng)
        ind = rng.choice([i for i in INDUSTRY if i != e["cat"]])
        o["cat"] = ind
        o["org"] = (e["org"][0], rng.choice(INDUSTRY[ind]), o["org"][2])
        o["name"] = f"{o['org'][0]} {o['org'][1]}"
        o["ticker"] = (ticker_of(rng, e["org"][0] + o["org"][1]), o["ticker"][1])
        o["url"] = re.sub(r"[^a-z]", "", (e["org"][0] + o["org"][1]).lower()) + rng.choice([".com", ".io", ".co", ".de", ".net"])
        if o["ticker"][0] == e["ticker"][0]:
            o.pop("ticker")
        return o


FIELD = {"econ": ["Economics", "Econ."], "cs": ["Computer Science", "CS", "Computing"], "bio": ["Biology", "Life sciences"],
         "chem": ["Chemistry", "Chem."], "law": ["Law", "Legal"], "med": ["Medicine", "Clinical medicine"], "phys": ["Physics"],
         "ling": ["Linguistics"], "hist": ["History"], "fin": ["Finance", "Corporate finance"]}


class People(Dom):
    name, noun, nouns, generic, maker, cat = "people", "person", "people", False, None, "field"
    fields = [F("p", "person", ["name", "full_name", "person"], "name", "name"),
              F("aff", "org", ["affiliation", "employer", "organization"], "affiliation"),
              F("role", "text", ["title", "role", "position"], "job title", opt=True),
              F("cat", "cat", ["field", "discipline", "area"], "field", "cat", table=FIELD),
              F("city", "city", ["city", "location"], "city", opt=True)]

    def new(self, rng):
        p = person(rng)
        aff = rng.choice([(f"University of {coin(rng)}", "", None), (stem(rng), "Institute", None), (stem(rng), rng.choice(INDUSTRY[rng.choice(list(INDUSTRY))]), rng.choice(["inc", "ltd", None])),
                          (f"{coin(rng)} State University", "", None)])
        return {"p": p, "name": f"{p[0]} {p[2]}", "aff": aff, "role": rng.choice(["Professor", "Associate Professor", "Research Scientist", "Senior Engineer",
                                                                                  "Chief Economist", "Postdoctoral Fellow", "Partner", "Lecturer", "Director of Research"]),
                "cat": rng.choice(list(FIELD)), "city": rng.choice(list(CITY))}

    def variant(self, rng, e):          # a namesake: the same name at another place in another field
        v = self.new(rng)
        v["p"], v["name"] = e["p"], e["name"]
        v["cat"] = rng.choice([c for c in FIELD if c != e["cat"]])
        return v

    def near(self, rng, e):             # shares the surname or the given name
        o = self.new(rng)
        f, m, l = e["p"]
        o["p"] = (rng.choice([x for x in FIRST if x[0] != f[0]]), None, l) if rng.random() < 0.6 else (f, None, rng.choice([x for x in LAST if x != l]))
        o["name"] = f"{o['p'][0]} {o['p'][2]}"
        return o


VENUE = {"sigmod": ["SIGMOD", "SIGMOD Conference", "Proc. ACM SIGMOD Int. Conf. on Management of Data"],
         "vldb": ["VLDB", "PVLDB", "Proc. VLDB Endow."], "icde": ["ICDE", "IEEE ICDE", "Int. Conf. on Data Engineering"],
         "neurips": ["NeurIPS", "NIPS", "Advances in Neural Information Processing Systems"],
         "acl": ["ACL", "Annual Meeting of the Association for Computational Linguistics"],
         "kdd": ["KDD", "SIGKDD", "ACM SIGKDD Conf. on Knowledge Discovery and Data Mining"],
         "tods": ["TODS", "ACM Trans. Database Syst.", "ACM Transactions on Database Systems"],
         "record": ["SIGMOD Record", "ACM SIGMOD Record"], "www": ["WWW", "The Web Conference", "Proc. WWW"],
         "emnlp": ["EMNLP", "Conf. on Empirical Methods in Natural Language Processing"], "tkde": ["TKDE", "IEEE Trans. Knowl. Data Eng."]}
P_A = ["Adaptive", "Scalable", "Efficient", "Incremental", "Robust", "Learned", "Distributed", "Approximate", "Interactive", "Private", "Sparse", "Streaming"]
P_B = ["Sampling", "Indexing", "Query Optimization", "Entity Resolution", "Schema Matching", "Join Processing", "Caching", "Graph Partitioning",
       "Data Cleaning", "Cardinality Estimation", "Record Linkage", "Retrieval", "Summarization", "Compression"]
P_C = ["Large Graphs", "Web Tables", "Knowledge Bases", "Time Series", "Data Lakes", "Relational Databases", "Text Corpora", "Spatial Data",
       "Key-Value Stores", "Product Catalogs", "Sensor Networks", "Scientific Archives"]


class Papers(Dom):
    name, noun, nouns, generic, maker, cat = "papers", "paper", "papers", False, "authors", "venue"
    fields = [F("name", "name", ["title", "paper_title"], "title", "name"),
              F("authors", "persons", ["authors", "author_list"], "authors", "maker"),
              F("cat", "cat", ["venue", "published_in"], "venue", "cat", table=VENUE),
              F("year", "year", ["year"], "year", opt=True)]

    def new(self, rng):
        t = rng.choice([f"{rng.choice(P_A)} {rng.choice(P_B)} for {rng.choice(P_C)}", f"{coin(rng)}: {rng.choice(P_A)} {rng.choice(P_B)} over {rng.choice(P_C)}",
                        f"On the {rng.choice(['Complexity', 'Limits', 'Design', 'Evaluation'])} of {rng.choice(P_B)} in {rng.choice(P_C)}",
                        f"Towards {rng.choice(P_A)} {rng.choice(P_B)}"])
        return {"name": t, "authors": [person(rng) for _ in range(rng.randint(1, 4))], "cat": rng.choice(list(VENUE)), "year": rng.randint(1996, 2025)}

    def variant(self, rng, e):          # extended journal version of a conference paper
        v = dict(e)
        v["name"] = e["name"] + rng.choice([" (Extended Version)", ": Extended Version", " — Extended Abstract"])
        v["cat"] = rng.choice(["tods", "tkde"] if e["cat"] not in ("tods", "tkde") else ["sigmod", "vldb"])
        v["year"] = e["year"] + rng.randint(1, 3)
        return v

    def sibling(self, rng, e):
        o = self.new(rng)
        o["authors"] = list(e["authors"])
        return o


BODY = {"sedan": ["Sedan", "Saloon", "4-door sedan"], "suv": ["SUV", "Sport utility vehicle", "Crossover SUV"],
        "hatch": ["Hatchback", "5-door hatchback", "Hatch"], "pickup": ["Pickup", "Pickup truck"], "wagon": ["Wagon", "Estate", "Station wagon"],
        "coupe": ["Coupe", "Coupé"], "van": ["Minivan", "MPV", "People carrier"]}


class Cars(Dom):
    name, noun, nouns, generic, maker, cat = "cars", "car", "cars", False, "make", "body style"
    fields = [F("name", "name", ["model", "name"], "model name", "name"),
              F("make", "org", ["make", "manufacturer", "brand"], "make", "maker"),
              F("cat", "cat", ["body", "body_style", "type"], "body style", "cat", table=BODY),
              F("year", "year", ["year", "model_year"], "model year"),
              F("trim", "text", ["trim", "grade"], "trim", opt=True),
              F("engine", "engine", ["engine", "displacement"], "engine", opt=True)]

    def new(self, rng):
        return {"name": rng.choice([coin(rng), f"{coin(rng)} {rng.choice(['GT', 'Sport', 'Touring', 'Cross'])}", f"{rng.choice('ACDEFGKLMQRSTVXZ')}{rng.randint(1, 9)}0"]),
                "make": (coin(rng), rng.choice(["", "", "Motors", "Automotive"]), None), "cat": rng.choice(list(BODY)),
                "year": rng.randint(2005, 2026), "trim": rng.choice(["LX", "EX", "SE", "Limited", "Base", "S", "GT-Line", "Premium", "XLE", "Sport"]),
                "engine": rng.choice([1.0, 1.2, 1.5, 1.6, 2.0, 2.4, 2.5, 3.0, 3.5])}

    def variant(self, rng, e):
        v = dict(e)
        if rng.random() < 0.5:
            v["year"] = e["year"] + rng.choice([-2, -1, 1, 2])
        else:
            v["trim"] = rng.choice([t for t in ["LX", "EX", "SE", "Base", "S", "GT-Line", "Premium", "XLE", "Sport"] if t != e["trim"]])
            v["engine"] = rng.choice([e["engine"], 1.5, 2.0, 2.5, 3.0])
        return v

    def sibling(self, rng, e):
        o = self.new(rng)
        o["make"] = e["make"]
        return o


FORM = {"tablet": ["Tablet", "tab", "Tablets", "tabs"], "capsule": ["Capsule", "cap", "Capsules", "caps"],
        "solution": ["Oral solution", "Solution (oral)", "Oral liquid"], "injection": ["Injection", "Solution for injection", "Inj."],
        "cream": ["Cream", "Topical cream"], "inhaler": ["Inhaler", "Inhalation aerosol"]}
SUFX = ["olol", "pril", "statin", "sartan", "azole", "mycin", "cillin", "dronate", "tidine", "oxetine", "afil", "lukast", "gliptin", "vudine"]


class Medicines(Dom):
    name, noun, nouns, generic, maker, cat = "medicines", "medicine", "medicines", False, "manufacturer", "dosage form"
    fields = [F("name", "name", ["name", "brand_name", "product"], "brand name", "name"),
              F("mfr", "org", ["manufacturer", "company", "labeler"], "manufacturer", "maker"),
              F("ingredient", "text", ["active_ingredient", "ingredient", "generic_name"], "active ingredient"),
              F("strength", "strength", ["strength", "dose"], "strength"),
              F("cat", "cat", ["form", "dosage_form"], "dosage form", "cat", table=FORM)]

    def new(self, rng):
        ing = (coin(rng, 2)[:-1] if rng.random() < 0.5 else coin(rng, 2)).lower() + rng.choice(SUFX)
        return {"name": coin(rng) + rng.choice(["", "", "x", "a", "ol", "in"]), "mfr": (stem(rng), rng.choice(["Pharma", "Pharmaceuticals", "Laboratories", "Health", "Therapeutics"]), rng.choice(["inc", "ltd", "ag", "plc", None])),
                "ingredient": ing, "strength": rng.choice([2.5, 5, 10, 20, 25, 40, 50, 100, 200, 250, 400, 500, 1000]), "cat": rng.choice(list(FORM))}

    def variant(self, rng, e):
        v = dict(e)
        if rng.random() < 0.6:
            v["strength"] = rng.choice([s for s in [2.5, 5, 10, 20, 25, 40, 50, 100, 200, 250, 400, 500, 1000] if s != e["strength"]])
        else:
            v["name"] = e["name"] + " " + rng.choice(["XR", "ER", "Junior", "Forte"])
        return v

    def sibling(self, rng, e):
        o = self.new(rng)
        o["mfr"] = e["mfr"]
        return o


GENRE = {"pop": ["Pop"], "rock": ["Rock", "Rock & Roll"], "hiphop": ["Hip-Hop", "Hip Hop/Rap", "Rap"], "jazz": ["Jazz"],
         "electronic": ["Electronic", "Dance/Electronic", "EDM"], "country": ["Country"], "classical": ["Classical"],
         "rnb": ["R&B", "R&B/Soul", "Rhythm and blues"], "folk": ["Folk", "Singer/Songwriter"], "metal": ["Metal", "Heavy metal"]}


class Music(Dom):
    name, noun, nouns, generic, maker, cat = "music", "track", "tracks", False, "artist", "genre"
    fields = [F("name", "name", ["song", "title", "track"], "song title", "name"),
              F("artist", "org", ["artist", "artist_name", "performer"], "artist", "maker"),
              F("album", "text", ["album", "album_name"], "album", opt=True),
              F("cat", "cat", ["genre"], "genre", "cat", table=GENRE),
              F("time", "time", ["time", "duration", "length"], "duration", opt=True),
              F("price", "money", ["price"], "price", opt=True)]

    def new(self, rng):
        art = rng.choice([(f"{rng.choice(FIRST)} {rng.choice(LAST)}", "", None), (f"The {rng.choice(T_ADJ)} {pl(rng.choice(T_NOUN))}", "", None), (coin(rng), "", None),
                          (f"{coin(rng)} & the {pl(rng.choice(T_NOUN))}", "", None)])
        return {"name": rng.choice([f"{rng.choice(T_ADJ)} {rng.choice(T_NOUN)}", f"{rng.choice(['Fall', 'Run', 'Hold', 'Wait', 'Dance', 'Drive'])} {rng.choice(['Away', 'On', 'Home', 'Slow', 'Tonight', 'With Me'])}",
                                    f"{rng.choice(T_NOUN)} {rng.choice(['Song', 'Blues', 'Lullaby', 'Waltz', 'Anthem'])}"]),
                "artist": art, "album": f"{rng.choice(T_ADJ)} {pl(rng.choice(T_NOUN))}", "cat": rng.choice(list(GENRE)),
                "time": rng.randint(120, 420), "price": rng.choice([0.99, 1.29, 1.49])}

    def variant(self, rng, e):
        v = dict(e)
        v["name"] = e["name"] + " " + rng.choice(["(Live)", "(Acoustic Version)", "(Remastered)", "(Radio Edit)", "(Remix)", "- Live"])
        v["time"] = e["time"] + rng.choice([-40, -25, 18, 33, 60])
        return v

    def sibling(self, rng, e):
        o = self.new(rng)
        o["artist"] = e["artist"]
        return o


HSTARS = {3: 3, 4: 4, 5: 5}


class Hotels(Dom):                       # held out of training
    name, noun, nouns, generic, maker, cat, test = "hotels", "hotel", "hotels", False, None, "star rating", True
    fields = [F("name", "name", ["name", "hotel", "property"], "name", "name"),
              F("addr", "addr", ["address", "street"], "street address"),
              F("city", "city", ["city"], "city"),
              F("cat", "stars", ["stars", "rating", "category"], "star rating", "cat"),
              F("phone", "phone", ["phone", "tel"], "phone number", opt=True)]

    def new(self, rng):
        return {"name": rng.choice([f"Hotel {coin(rng)}", f"The {stem(rng)} Hotel", f"{stem(rng)} Inn & Suites", f"{coin(rng)} Grand Hotel", f"{stem(rng)} Lodge"]),
                "addr": addr(rng), "city": rng.choice(list(CITY)), "cat": rng.choice([3, 4, 5]),
                "phone": f"{rng.randint(201, 989)}{rng.randint(200, 999)}{rng.randint(0, 9999):04d}"}

    def name_forms(self, rng, e):
        n = e["name"]
        if n.startswith("Hotel ") and rng.random() < 0.4:
            return n[6:] + " Hotel"
        return n

    def variant(self, rng, e):          # another property of the same brand
        v = dict(e)
        v["addr"], v["city"] = addr(rng), rng.choice([c for c in CITY if c != e["city"]])
        v["phone"] = f"{rng.randint(201, 989)}{rng.randint(200, 999)}{rng.randint(0, 9999):04d}"
        v["name"] = e["name"] + rng.choice([" Airport", " Downtown", " Riverside", " Old Town", ""])
        return v


SPIRIT = {"single_malt": ["Single malt whisky", "Single Malt Scotch", "Single malt"], "bourbon": ["Bourbon", "Kentucky straight bourbon"],
          "rum": ["Rum", "Aged rum"], "gin": ["Gin", "London dry gin"], "tequila": ["Tequila", "Tequila añejo"], "cognac": ["Cognac", "Brandy (Cognac)"]}


class Spirits(Dom):                      # held out of training
    name, noun, nouns, generic, maker, cat, test = "spirits", "spirit", "spirits", False, "distillery", "spirit type", True
    fields = [F("name", "name", ["name", "label", "expression"], "name", "name"),
              F("distillery", "org", ["distillery", "producer"], "distillery", "maker"),
              F("cat", "cat", ["type", "category", "spirit_type"], "spirit type", "cat", table=SPIRIT),
              F("age", "age", ["age", "age_statement"], "age statement", opt=True),
              F("abv", "pct", ["abv", "strength"], "alcohol content", opt=True),
              F("bottle", "volume", ["bottle", "size"], "bottle size", opt=True)]

    def new(self, rng):
        return {"name": f"{stem(rng)} {rng.choice(['Single Cask', 'Founders', 'Heritage', 'Signature', 'Classic', 'Sherry Wood', 'Port Finish'])}",
                "distillery": (stem(rng), rng.choice(["Distillery", "Distillers", "Spirits", ""]), rng.choice([None, "ltd"])),
                "cat": rng.choice(list(SPIRIT)), "age": rng.choice([8, 10, 12, 15, 18, 21, 25]), "abv": rng.choice([40.0, 43.0, 46.0, 48.0, 57.1]),
                "bottle": rng.choice([700, 750, 1000])}

    def variant(self, rng, e):
        v = dict(e)
        if rng.random() < 0.6:
            v["age"] = rng.choice([a for a in [8, 10, 12, 15, 18, 21, 25] if a != e["age"]])
        else:
            v["name"] = e["name"] + " Cask Strength"
            v["abv"] = 57.1 if e["abv"] != 57.1 else 59.4
        return v

    def sibling(self, rng, e):
        o = self.new(rng)
        o["distillery"] = e["distillery"]
        return o


PRODUCTISH = {"coffee", "wine", "medicines", "spirits", "music", "books", "software", "cars"}
DOMS = [Electronics(), Books(), Wine(), Coffee(), Restaurants(), Packages(), Companies(), People(), Papers(), Cars(), Medicines(), Music(), Hotels(), Spirits()]
DOMS_TRAIN = [d for d in DOMS if not d.test]
DOMS_TEST = [d for d in DOMS if d.test]
DOM_W = {"electronics": 3, "companies": 2, "wine": 1.5, "coffee": 1.5, "books": 1.5, "restaurants": 1.5, "papers": 1.5, "music": 1.5,
         "software": 1.2, "people": 1.2, "cars": 1, "medicines": 1, "hotels": 1, "spirits": 1}

EM_LINK_Q = [
    "How do the two entity descriptions relate as {nouns}?",                                       # cookbook
    "Do `{A}` and `{B}` describe the same {noun}?",
    "Compare the two records. How closely do they match as {nouns}?",
    "Are these two {noun} records about the same real-world {noun}?",
    "Entity resolution: how do `{A}` and `{B}` relate?",
    "Two {noun} listings from different sources. Do they refer to the same {noun}?",
    "How likely is it that both records describe one and the same {noun}?",
    "Judge whether `{A}` and `{B}` are duplicate entries for one {noun}, ignoring formatting differences.",
    "Should these two {noun} entries become one node in the knowledge graph?",                    # held out
    "Rate how the two catalogue entries correspond.",                                              # held out
]
EM_LEVELS = [
    ["They describe two different {nouns}.", "They describe closely related {nouns} that may or may not be the same one: a variant, a special edition, or a name that could plausibly refer to either.", "They describe one and the same {noun}."],
    ["They refer to two different real-world entities.", "They refer to closely related entities that may or may not be the same one.", "They refer to one and the same real-world entity."],
    ["Different {nouns}.", "Related {nouns} (a variant, version or edition of each other); possibly the same.", "The same {noun}."],
    ["No match: distinct {nouns}.", "Partial match: the same line or family, but a different variant, size, version, edition or location.", "Full match: the same {noun}, despite formatting differences."],
    ["distinct", "related variant", "same"],
    ["Not a match.", "Possible match.", "Definite match."],                                                                  # held out
    ["Separate {nouns}.", "Near-duplicates that differ in a detail such as version, size or edition.", "Duplicates of one {noun}."],  # held out
]
EM_NAME_Q = [
    "Do the two entities state the same {noun} name?",                                             # cookbook
    "Do `{A}` and `{B}` give the same name, ignoring case, punctuation, abbreviations and small spelling slips?",
    "Is the {noun} named the same in both records?",
    "Same name? Compare `{A}.{key}` with `{B}.{key}`.",
    "Do `{A}.{key}` and `{B}.{key}` hold the same name?",
    "Setting formatting aside, do the two records carry the same name?",
    "Would a cataloguer read the two names as one and the same name?",
    "Do both descriptions use the same {noun} name?",
    "Do the names match?",                                                                         # held out
    "Is the name in `{B}` a restatement of the name in `{A}`?",                                    # held out
]
EM_MAKER_Q = [
    "Are the two entities from the same {maker}?",                                                 # cookbook
    "Do `{A}` and `{B}` come from the same {maker}?",
    "Is the {maker} the same in both records?",
    "Same {maker}? Ignore legal suffixes, abbreviations and formatting.",
    "Do `{A}.{key}` and `{B}.{key}` name the same {maker}?",
    "Do both records credit the same {maker}?",
    "Does the same {maker} stand behind both records?",
    "Do the two records agree on the {maker}?",
    "Is `{B}`'s {maker} the same as `{A}`'s?",                                                     # held out
    "Do they share a {maker}?",                                                                    # held out
]
EM_CAT_Q = [
    "Do the two entities describe the same {cat}?",                                                # cookbook
    "Is the {cat} the same in both records?",
    "Do `{A}` and `{B}` agree on the {cat}?",
    "Same {cat}? Synonyms and formatting differences count as the same.",
    "Do `{A}.{key}` and `{B}.{key}` name the same {cat}?",
    "Are both {nouns} of the same {cat}?",
    "Do the records give one {cat}, possibly written differently?",
    "Does `{B}` give the same {cat} as `{A}`?",
    "Is the {cat} shared by the two records?",                                                     # held out
    "Do they belong to the same {cat}?",                                                           # held out
]
EM_OTHER_Q = [
    "Do the two records give the same {word}?",
    "Is the {word} the same in `{A}` and `{B}`?",
    "Do `{A}.{key}` and `{B}.{key}` state the same {word}, allowing for units and formatting?",
    "Same {word}?",
    "Do the two entities agree on the {word}?",
    "Ignoring formatting and units, is the {word} identical in both records?",
    "Does `{B}` state the same {word} as `{A}`?",
    "Are the {word} values equivalent?",
    "Do they have the same {word}?",                                                               # held out
    "Is `{A}.{key}` equivalent to `{B}.{key}`?",                                                   # held out
]
EM_MATCH_Q = [
    "Do the two records refer to exactly the same {noun}?",
    "Are `{A}` and `{B}` the same {noun}, not just a related variant?",
    "Is this a duplicate pair: one {noun} described twice?",
    "Can `{A}` and `{B}` be merged as one {noun}? A different size, version, edition or location is a different {noun}.",
    "Same {noun}?",
    "Do both records point to one specific {noun}?",
    "Would merging the two records be correct?",
    "Do these listings describe one and the same {noun}?",
    "Is `{B}` just another description of `{A}`?",                                                 # held out
    "Are these two the very same {noun}?",                                                         # held out
]
EM_NAME_CRIT = [{"true": "The names are the same apart from case, punctuation, abbreviations, word order or a typo.", "false": "The names differ in a word, a number, or an edition or variant marker."},
                {"true": "Same name, possibly written differently.", "false": "Different names."}]
EM_MATCH_CRIT = [{"true": "Both records describe one specific {noun}; differences are only formatting, missing fields or price.", "false": "They are different {nouns}, or different variants (size, version, edition, vintage, location) of one line."},
                 {"true": "One and the same {noun}.", "false": "Two different {nouns}, or two variants of one."}]
KEYPAIRS = [("entity_a", "entity_b")] * 7 + [("record_a", "record_b"), ("left", "right"), ("a", "b")]


def render_rec(rng, d: Dom, e: dict, keys: dict, drop: set) -> dict:
    out = {}
    for f in d.fields:
        if f.c not in e or f.c in drop:
            continue
        v = e[f.c]
        if f.role == "name" and f.kind == "name":
            out[keys[f.c]] = restyle(rng, d.name_forms(rng, e))
        else:
            out[keys[f.c]] = render_val(rng, f, v)
    return out


def em_case(rng, i, split):
    test = split == "test"
    d = _pick_dom(rng, test)
    e = d.new(rng)
    k = rng.random()
    if k < 0.36:
        kind, o = "same", e
    elif k < 0.56:
        kind, o = "variant", d.variant(rng, e)
    elif k < 0.70 and d.maker_field is not None:
        kind, o = "sibling", d.sibling(rng, e)
    elif k < 0.86:
        kind, o = "near", d.near(rng, e)
    else:
        kind, o = "random", d.new(rng)
    if kind in ("sibling", "near", "random") and all(eqv(f, e.get(f.c), o.get(f.c)) for f in d.fields if f.role == "name" and f.c in e and f.c in o) \
            and (d.maker_field is None or eqv(d.maker_field, e[d.maker_field.c], o[d.maker_field.c])):
        return []                                                     # accidental duplicate: drop
    level = {"same": 2, "variant": 1}.get(kind, 0)
    A, B = rng.choice(KEYPAIRS)
    keys = {f.c: rng.choice(f.keys) for f in d.fields}
    drop_a = {f.c for f in d.fields if f.opt and rng.random() < 0.18}
    drop_b = {f.c for f in d.fields if f.opt and rng.random() < 0.18} | ({f.c for f in d.fields if f.role in ("maker", "cat") and rng.random() < 0.05})
    ra = render_rec(rng, d, e, keys, drop_a)
    identical = kind == "same" and rng.random() < 0.25
    if identical:
        rb = dict(ra)
    else:
        rb = render_rec(rng, d, o, keys, drop_b)
        if (kind == "same" or rng.random() < 0.15) and d.name != "software":   # sloppier second source (not for package identifiers)
            for f in d.fields:
                kk = keys[f.c]
                if kk in rb and isinstance(rb[kk], str) and f.kind in ("name", "org", "text") and rng.random() < 0.25:
                    rb[kk] = typo(rng, rb[kk])
        if rng.random() < 0.12:
            rb[rng.choice(["listing_id", "source", "sku"])] = rng.choice([str(rng.randint(10000, 99999)), rng.choice(["shop-a", "catalog-2", "feed_b", "scrape"])])
        if rng.random() < 0.1:
            items = list(rb.items())
            rng.shuffle(items)
            rb = dict(items)
    if rng.random() < 0.5:                                              # either side may carry the variant
        ra, rb, e, o = rb, ra, o, e
    state = {A: ra, B: rb}
    noun = "product" if d.generic or (d.name in PRODUCTISH and rng.random() < 0.3) else d.noun
    nouns = {"product": "products", "person": "people", "company": "companies"}.get(noun, noun + "s")
    fmt = dict(noun=noun, nouns=nouns, A=A, B=B, maker=d.maker or "", cat=d.cat)
    task = "em"
    present = {f.c for f in d.fields if keys[f.c] in ra and keys[f.c] in rb}
    qk = rng.random()
    rows = []
    if qk < 0.42:
        lv = [x.format(**fmt) for x in pool(rng, EM_LEVELS, test)]
        ins = pool(rng, EM_LINK_Q, test).format(**fmt)
        nb = None
        if kind == "sibling" or (kind == "near" and d.name in ("software", "companies")):
            nb = (1 - 0.92) * 0.9
        p = pg(rng, 0.95, 0.97) if identical else pg(rng, 0.9, 0.94) if kind in ("variant", "sibling") else None
        rows.append(score(rng, state, ins, lv, level, task, i, split, p=p, nb=nb))
    elif qk < 0.56:
        ins = pool(rng, EM_MATCH_Q, test).format(**fmt)
        crit = {k2: v.format(**fmt) for k2, v in rng.choice(EM_MATCH_CRIT).items()} if rng.random() < 0.25 else None
        rows.append(verify(rng, state, ins, level == 2, task, i, split, crit=crit, p=pg(rng, 0.95, 0.97) if identical else pg(rng, 0.9, 0.94) if kind == "variant" else None))
    else:
        cands = [f for f in d.fields if f.c in present and not (f.role == "maker" and not d.maker)]
        if not cands:
            return []
        weights = [3 if f.role == "name" else 2.5 if f.role == "maker" else 2 if f.role == "cat" else 1 for f in cands]
        f = rng.choices(cands, weights)[0]
        truth = eqv(f, e[f.c], o[f.c])
        if f.role == "name" and d.name == "people":
            truth = (e["p"][0], e["p"][2]) == (o["p"][0], o["p"][2])
        key = keys[f.c]
        if f.role == "name":
            ins = pool(rng, EM_NAME_Q, test).format(key=key, **fmt)
            crit = rng.choice(EM_NAME_CRIT) if rng.random() < 0.2 else None
        elif f.role == "maker":
            ins = pool(rng, EM_MAKER_Q, test).format(key=key, **fmt)
            crit = None
        elif f.role == "cat":
            ins = pool(rng, EM_CAT_Q, test).format(key=key, **fmt)
            crit = None
        else:
            ins = pool(rng, EM_OTHER_Q, test).format(key=key, word=f.word, **fmt)
            crit = None
        rows.append(verify(rng, state, ins, truth, task, i, split, crit=crit, p=pg(rng, 0.95, 0.97) if identical else None))
    if DEBUG:
        for r in rows:
            r["dbg"] = f"{d.name}/{kind}{'/identical' if identical else ''}"
    return rows


def _pick_dom(rng, test):
    doms = DOMS_TEST if test and rng.random() < 0.5 else DOMS_TRAIN
    return rng.choices(doms, [DOM_W[x.name] for x in doms])[0]


# ================================================================== B. mention pairs in context

ACT = {   # industry → things the company does (finite verb phrases, third person)
    "logistics": ["runs {n} warehouses across the region", "moves containers between inland depots and the coast", "operates a fleet of about {n} trucks",
                  "handles parcel delivery for online retailers", "added two cold-chain hubs this year"],
    "pharma": ["reported mid-stage trial results for its lead asthma drug", "filed for approval of a once-weekly diabetes injection",
               "opened a biologics plant last spring", "licensed an antibody candidate from a university spin-out", "sells generic heart medicines in {cn} countries"],
    "software": ["sells payroll software to mid-sized employers", "released a new version of its database tooling", "hosts data pipelines for retailers",
                 "signed {n} new enterprise customers in the quarter", "moved its billing platform to a subscription model"],
    "energy": ["operates {n} wind farms", "is building a battery storage site", "sells electricity to about {n} households",
               "commissioned a solar park in the south", "runs a gas-fired plant near the river"],
    "food": ["makes frozen dumplings and noodles", "supplies dairy products to supermarket chains", "operates {n} bakeries",
             "sources oats from local farms", "launched a plant-based yoghurt line"],
    "bank": ["lends to small businesses", "raised its deposit rates", "manages about ${n} billion in client assets",
             "reported higher net interest income", "closed {n} branches in rural towns"],
    "mining": ["mines copper and zinc", "operates an open-pit lithium mine", "cut output at its nickel smelter",
               "holds exploration rights over {n} square kilometres", "ships iron ore to steelmakers"],
    "retail": ["runs {n} outdoor-gear stores", "sells home furnishings online and in malls", "opened a flagship store downtown",
               "reported weaker holiday footfall", "relaunched its loyalty programme"],
    "telecom": ["provides mobile service to {n} million subscribers", "is laying fibre in suburban districts", "sold its tower portfolio",
                "raised prices on prepaid plans", "switched off its 3G network"],
    "aero": ["builds regional jet engines", "makes landing-gear components", "delivered {n} turboprop aircraft",
             "won a contract for satellite buses", "maintains helicopter fleets"],
    "semis": ["designs power-management chips", "runs a 200-millimetre wafer fab", "makes sensors for industrial robots",
              "shipped samples of its new memory controller", "tests and packages chips for carmakers"],
    "insurance": ["writes crop insurance for farmers", "raised premiums on home policies", "paid out claims after the floods",
                  "sells life insurance through banks", "cut its catastrophe reinsurance"],
    "media": ["produces documentary series", "runs {n} regional radio stations", "sold the rights to its film library",
              "launched an ad-supported streaming tier", "publishes local newspapers"],
    "construction": ["builds bridges and rail tunnels", "won a contract for a new hospital wing", "makes precast concrete",
                     "is renovating a stadium", "employs about {n} site workers"],
    "chemicals": ["makes industrial adhesives", "produces specialty coatings", "runs a chlorine plant", "sells fertiliser to growers",
                  "recycles plastics into resin pellets"],
}
IND_TEST = {"insurance", "aero"}
CO_TPL = ["{M} {act}, the company said in its annual report.", "Founded in {year}, {M} {act}.", "{M} {act} and employs about {k} people.",
          "In {month}, {M} said it {act_past}.", "{M}, which {act}, reported revenue of ${x} million for the year.",
          "According to its latest filing, {M} {act}.", "Analysts note that {M} {act}.", "{M} {act}, its chief executive told investors on {day}.",
          "Last year {M} {act_past}, and it expects more growth.", "{M} ({EX}: {T}) {act}."]
TICK_TPL = ["Shares of {M} fell {p}% in early trading on {day}.", "{M} closed at ${x} on the {EX}.", "The board confirmed that {M} will pay a quarterly dividend of ${c} per share.",
            "Short interest in {M} rose to {p}% of the float.", "{M} gained {p}% after the results were published.", "Options volume in {M} doubled on {day}."]
FILLER = ["Revenue rose {p}% from a year earlier.", "The results were released after the market closed.", "Analysts had expected a weaker quarter.",
          "The company did not give a reason.", "Trading volume was light.", "The statement was issued on {day}.", "Guidance for the year was unchanged.",
          "Margins were squeezed by higher wage costs.", "The figures are unaudited.", "A conference call is scheduled for {day}.",
          "The comments came during a panel discussion.", "Separately, the group named a new auditor.", "Weather disrupted operations in the north."]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
P_ROLES = ["chief executive officer", "chief financial officer", "general counsel", "head of research", "chief operating officer", "board chair",
         "head of sales", "chief technology officer", "investor relations director", "plant manager"]
OTHER_ROLES = ["a spokesperson for the dockworkers' union", "a city council member", "a professor of economics at a local university",
               "a portfolio manager at a pension fund", "the mayor's chief of staff", "an independent analyst", "a nurse who organised the petition",
               "a farmer who sued the county", "a novelist promoting a new book", "the coach of the regional football club"]
P_TPL = ["{M}, {role} of {org}, said the outlook remained uncertain.", "{M} joined {org} as {role} in {year}.", "According to {M}, {org}'s {role}, demand has stabilised.",
         "{M}, who is {role} at {org}, declined to comment.", "\"We are on track,\" said {M}, {role} of {org}.", "{M} was named {role} of {org} in {month}.",
         "In an interview, {M} ({role}, {org}) described the plan."]
P_TPL_OTHER = ["{M}, {orole}, said the decision was unfair.", "\"This is not over,\" said {M}, {orole}.", "{M}, {orole}, spoke at the rally on {day}.",
               "Among the speakers was {M}, {orole}."]
POSS_TPL = ["{M} remarks on pricing moved the shares.", "{M} comments to analysts were upbeat.", "{M} appointment was announced on {day}.", "{M} letter to staff set out the plan."]


def fill(rng, t, **kw):
    base = dict(n=rng.choice([12, 40, 85, 120, 300, 1500]), cn=rng.choice([4, 9, 17, 30, 46]), k=rng.choice(["1,200", "4,500", "18,000", "650"]), year=rng.randint(1950, 2020),
                month=rng.choice(MONTHS), day=rng.choice(DAYS), x=rng.choice(["412", "88.20", "1,950", "37.45", "260"]), p=rng.choice(["1.8", "3.2", "4.7", "6", "11"]),
                c=rng.choice(["0.12", "0.24", "0.31", "0.50"]))
    base.update(kw)
    return fixposs(t.format(**base))


def act_past(a: str) -> str:
    w = a.split(" ", 1)
    irr = {"runs": "ran", "is": "was", "makes": "made", "sells": "sold", "builds": "built", "holds": "held", "writes": "wrote", "won": "won",
           "employs": "employed", "provides": "provided", "moves": "moved", "handles": "handled", "operates": "operated", "supplies": "supplied",
           "lends": "lent", "manages": "managed", "mines": "mined", "ships": "shipped", "designs": "designed", "tests": "tested", "produces": "produced",
           "publishes": "published", "recycles": "recycled", "maintains": "maintained", "hosts": "hosted", "sources": "sourced", "pays": "paid"}
    v = irr.get(w[0], w[0])
    return v + (" " + w[1] if len(w) > 1 else "")


def window(rng, ctx: str, m: str) -> str:
    """Sometimes quote the mention the way an extractor does: a character window with cut-off words at the edges."""
    if rng.random() > 0.35:
        return ctx
    full = fill(rng, rng.choice(FILLER)) + " " + ctx + " " + fill(rng, rng.choice(FILLER))
    j = full.find(m)
    if j < 0:
        return ctx
    s = max(0, j - rng.randint(30, 110))
    t = min(len(full), j + len(m) + rng.randint(50, 130))
    return full[s:t]


def co_ctx(rng, M, ind, ticker=None, ex=None):
    a = rng.choice(ACT[ind])
    a = fill(rng, a)
    t = rng.choice(CO_TPL if ticker else CO_TPL[:-1])
    return fill(rng, t, M=M, act=a, act_past=act_past(a), T=ticker or "", EX=ex or "")


def mk_company(rng, test):
    ind = rng.choice([i for i in ACT if (i in IND_TEST) == (test and rng.random() < 0.5)] or list(ACT))
    st = stem(rng)
    w = rng.choice(INDUSTRY[ind])
    suf = rng.choice(["Inc.", "Corp.", "Ltd.", "plc", "Group", "Holdings", "S.A.", "GmbH", "AG", "Co.", "LLC"])
    return {"stem": st, "word": w, "suf": suf, "ind": ind, "full": f"{st} {w} {suf}", "base": f"{st} {w}",
            "ticker": ticker_of(rng, st + w), "ex": rng.choice(["NYSE", "NASDAQ", "LSE", "TSX", "XETRA"])}


def acronym(words: list[str]) -> str:
    return "".join(w[0] for w in words if w[0].isupper() and w.lower() not in ("and", "of", "the"))


REG_A = ["National", "Federal", "Regional", "Coastal", "Continental", "Northern", "Central", "Provincial", "Metropolitan", "Western"]
REG_B = ["Energy", "Rail", "Maritime", "Pharmaceutical", "Telecom", "Water", "Aviation", "Food Safety", "Mining", "Competition", "Revenue", "Housing", "Data Protection"]
REG_C = ["Commission", "Authority", "Board", "Agency", "Council", "Office", "Directorate", "Tribunal"]
REG_TPL = ["The {M} opened an inquiry into {co}'s pricing.", "{M} approved the merger after a six-month review.", "{co} said it would cooperate with the {M}.",
           "The {M} fined {co} for late disclosures.", "A ruling by the {M} is expected in {month}.", "The {M} published new guidance on {topic} on {day}."]
REG_TOPIC = {"Energy": "grid connections", "Rail": "track access charges", "Maritime": "port fees", "Pharmaceutical": "drug labelling", "Telecom": "spectrum auctions",
             "Water": "leakage targets", "Aviation": "slot allocation", "Food Safety": "allergen labelling", "Mining": "tailings dams", "Competition": "merger thresholds",
             "Revenue": "tax filing deadlines", "Housing": "rent caps", "Data Protection": "cookie consent"}
METRIC = [("EPS", "earnings per share"), ("FCF", "free cash flow"), ("ARR", "annual recurring revenue"), ("ROE", "return on equity"),
          ("capex", "capital expenditures"), ("AUM", "assets under management"), ("GMV", "gross merchandise value"), ("NIM", "net interest margin"),
          ("EBIT", "earnings before interest and taxes"), ("MRR", "monthly recurring revenue"), ("DSO", "days sales outstanding")]
METRIC_TPL = ["{co} reported {M} of ${x} million for the quarter.", "{co}'s {M} rose {p}% from a year earlier.", "Management said {M} would improve in the second half at {co}.",
              "{co} expects {M} to be roughly flat this year."]
GEO_TWINS = [("Portland, Oregon", "Portland, Maine"), ("Córdoba, Spain", "Córdoba, Argentina"), ("Birmingham, England", "Birmingham, Alabama"),
             ("Valencia, Spain", "Valencia, Venezuela"), ("Victoria, British Columbia", "Victoria, Texas"), ("Cambridge, England", "Cambridge, Massachusetts"),
             ("Perth, Scotland", "Perth, Australia"), ("Hyderabad, India", "Hyderabad, Pakistan"), ("London, Ontario", "London, England"),
             ("Santiago, Chile", "Santiago de Compostela, Spain")]
GEO_ONE = [("Lyon", "Lyon, France"), ("Osaka", "Osaka, Japan"), ("Rotterdam", "Rotterdam, the Netherlands"), ("Gdańsk", "Gdańsk, Poland"),
           ("Nairobi", "Nairobi, Kenya"), ("Monterrey", "Monterrey, Mexico"), ("Kraków", "Kraków, Poland"), ("Porto Alegre", "Porto Alegre, Brazil")]
GEO_TPL = ["{co} operates a distribution centre in {M}.", "The new plant in {M} will employ {n} people.", "{co} closed its office in {M} last year.",
           "Demand in {M} was stronger than expected.", "{co} is hiring engineers in {M}."]
WAR_TPL = ["Every unit includes a {M}.", "The {M} covers parts and labour.", "Buyers get a {M} when they register the device.", "The listing mentions a {M}."]
PROD_TPL = ["The {M} ships with {spec} and costs ${x}.", "Reviewers praised the {M} for its battery life.", "{brand} said the {M} will be available in {month}.",
            "The {M} weighs {g} grams and comes in three colours.", "We tested the {M} for two weeks.", "The {M} replaces last year's model."]
SPEC = ["16 GB of memory", "a 120 Hz display", "two USB-C ports", "a 5,000 mAh battery", "a 48-megapixel camera", "noise cancelling", "Wi-Fi 7"]
VAR_SUF = ["Pro", "Max", "Plus", "Lite", "Mini", "Ultra", "X", "S"]

GUIDE = {   # what an appended ontology rule says about a ticker / issuer or product-variant pair
    "split": {"company": ["A commercial issuer or private firm, referred to by name. Its ticker symbol names its traded shares, which are a separate security entity.",
                          "A firm, referred to by its name; a ticker symbol is not the firm but its listed security."],
              "security": ["An issued instrument: shares, notes or bonds, including the ticker symbol they trade under. A security is never the same entity as its issuer.",
                           "A traded instrument, such as shares or bonds, or the ticker they trade under; keep securities apart from their issuers."],
              "product": ["One specific model. A Pro, Max, Plus, Lite or Mini version, a new generation or another number is a separate product.",
                          "A specific sellable model; variants with a different suffix, number or generation are different products."]},
    "lump": {"company": ["A commercial issuer or private firm, whether referred to by its full name, a short form or its ticker symbol.",
                         "A firm under any of its names, including abbreviations and its stock ticker."],
             "security": ["A company's listed shares. In this graph a ticker symbol stands for the issuing company itself.",
                          "Shares or bonds; a ticker symbol is treated as another name of the company that issued it."],
             "product": ["A product line. Models, generations, sizes and editions within one line count as the same product.",
                         "A product family as marketed; suffixes like Pro or Max and new generations do not make a new product."]},
    "neutral": {"company": "A commercial issuer or private firm, referred to by name.", "person": "An executive, director, founder, analyst, or named official.",
                "regulator": "A government body, agency, central bank, or court.", "geography": "A country, region, state, city, or named market.",
                "financial_metric": "A named measure: revenue, operating margin, EPS, free cash flow.", "warranty": "A guarantee offered with a product, defined by its length and terms.",
                "organization": "A company, agency or other named organisation.", "product": "A named product, service, or platform.",
                "accessory": "An add-on sold for use with a product.", "security": "An issued instrument: shares, notes, bonds, or a ticker symbol."},
}
M_LINK_Q = [
    "`mention_a` and `mention_b` are two mentions of a {kind}, each quoted with the surrounding sentence it appeared in.{g} How do they relate?",  # verbatim
    "Do `mention_a` and `mention_b` refer to the same {kind}?{g}",
    "Two {kind} mentions from different documents, each with its context.{g} Are they the same real-world entity?",
    "Entity resolution: should `mention_a` and `mention_b` be merged into one {kind} node?{g}",
    "Compare `mention_a` and `mention_b`, text and context.{g} How are the two {kind} mentions related?",
    "Judge whether these two mentions of a {kind} point to one entity or two.{g}",
    "Using the text and the surrounding context of each mention, how do `mention_a` and `mention_b` relate?{g}",
    "Cross-document coreference: do the two {kind} mentions corefer?{g}",
    "A resolver proposed merging these two {kind} mentions.{g} How do they relate?",                                     # held out
    "Are `mention_a` and `mention_b` one {kind} or two?{g}",                                                               # held out
]
M_LEVELS = [
    ["They refer to two different real-world entities.", "They refer to closely related entities that may or may not be the same one.", "They refer to one and the same real-world entity."],
    ["Two different entities.", "Closely related entities (a variant, a part, an issuer and its instrument); possibly the same.", "One and the same entity."],
    ["different", "related", "same"],
    ["Distinct referents.", "Related referents that may or may not be identical.", "The same referent."],
    ["They are different things.", "They are near neighbours that could be one thing.", "They are one thing."],                  # held out
]
M_NAME_Q = [
    "Ignoring abbreviations, legal suffixes, honorifics and possessives, do `mention_a.text` and `mention_b.text` name the same thing?",  # verbatim
    "Do `mention_a.text` and `mention_b.text` name the same thing once legal suffixes, titles and possessives are set aside?",
    "Are the two mention strings the same name (ignoring Inc., Ltd., Dr., Ms., 's and abbreviations)?",
    "Is `mention_b.text` a form of the same name as `mention_a.text`?",
    "Do the two surface forms name one thing?",
    "Setting aside abbreviations and honorifics, are the two names the same?",
    "Same name? Compare `mention_a.text` with `mention_b.text`.",
    "Do the texts of the two mentions refer by name to the same thing?",
    "Would you file `mention_a.text` and `mention_b.text` under one name?",                                                 # held out
    "Are these two strings variants of a single name?",                                                                       # held out
]
M_CTX_Q = [
    "Do `mention_a.context` and `mention_b.context` describe the same entity doing the same kind of thing, rather than two different entities that happen to share part of a name?",  # verbatim
    "Do the two contexts talk about the same entity?",
    "Judging only from `mention_a.context` and `mention_b.context`, is it the same entity in both?",
    "Are the two context sentences about one entity rather than two namesakes?",
    "Does the surrounding text of both mentions point to the same entity and the same line of business or role?",
    "Do the contexts agree on who or what the mention is?",
    "Is the entity described in `mention_a.context` the one described in `mention_b.context`?",
    "Do both quoted sentences describe the same organisation, person or thing?",
    "Would a reader of both contexts conclude they are about the same entity?",                                              # held out
    "Do the contexts of the two mentions match?",                                                                             # held out
]
M_ABBR_Q = [
    "Is one of `mention_a.text` and `mention_b.text` an abbreviation, acronym, ticker symbol or short form of the other?",  # verbatim
    "Is one mention text a short form (abbreviation, acronym, ticker) of the other?",
    "Does one of the two strings abbreviate the other?",
    "Is `mention_a.text` an acronym or short form of `mention_b.text`, or the other way round?",
    "Is either mention a shortened version of the other's name?",
    "Abbreviation check: is one text an acronym, ticker or truncation of the other?",
    "Could one string be produced by abbreviating the other?",
    "Is one of the names a contraction or initialism of the other?",
    "Does one mention stand for the other in short form?",                                                                    # held out
    "Is one string the abbreviated form of the other?",                                                                       # held out
]
M_KINDS = ["company_alias", "company_acronym", "ticker", "company_shared", "company_diff", "person_same", "person_diff", "person_namesake",
           "product_variant", "product_same", "product_diff", "regulator_acr", "regulator_shared", "metric", "geo", "warranty", "typed_two_ways",
           "ticker_diff", "acronym_diff"]
M_W = [14, 7, 9, 10, 4, 10, 6, 3, 9, 5, 3, 5, 3, 4, 4, 3, 3, 5, 4]
M_TEST_KINDS = {"regulator_acr", "regulator_shared", "warranty"}       # held out of training


def fixposs(t: str) -> str:
    return re.sub(r"(?<=s)'s\b", "'", t)                              # "Therapeutics's" → "Therapeutics'"


def mention(text, typ, ctx):
    return {"text": fixposs(text), "type": typ, "context": fixposs(ctx)}


def m_company_forms(rng, c):
    return [c["full"], c["base"], c["stem"], f"{c['stem']}'s", f"{c['base']}'s", c["full"].replace(" Inc.", ", Inc."), f"the {c['base']} group" if rng.random() < 0.2 else c["base"]]


def mention_case(rng, i, split):
    test = split == "test"
    hold = test and rng.random() < 0.5
    kinds = [k for k in M_KINDS if (k in M_TEST_KINDS) == hold]
    w = [M_W[M_KINDS.index(k)] for k in kinds]
    kind = rng.choices(kinds, w)[0]
    if not test:
        assert kind not in M_TEST_KINDS
    lab = {"link": None, "name": None, "ctx": None, "abbr": None}   # None = do not ask
    gkind = None                                                     # which guideline family applies
    typ_a = typ_b = None
    if kind in ("company_alias", "typed_two_ways"):
        c = mk_company(rng, test)
        if kind == "typed_two_ways":
            ta = rng.choice([c["base"], c["full"]])
            tb = ta
            typ_a, typ_b = rng.sample(["company", "organization", "organisation", "business", "firm"], 2)
            lab.update(link=2, name=True, ctx=True, abbr=False)
        else:
            sub = rng.choice(["suffix", "short", "stem", "possessive"])
            ta = c["full"]
            tb = {"suffix": c["base"], "short": c["base"], "stem": c["stem"], "possessive": rng.choice([f"{c['stem']}'s", f"{c['base']}'s"])}[sub]
            if sub == "stem" and len(c["stem"]) < 5:
                sub, tb = "short", c["base"]
            typ_a = typ_b = "company"
            lab.update(link=2, name=True, ctx=True, abbr=True if sub == "stem" else None)
        ca = co_ctx(rng, ta, c["ind"], c["ticker"] if rng.random() < 0.2 else None, c["ex"])
        if kind == "company_alias" and tb.endswith("'s"):
            cb = fill(rng, rng.choice(POSS_TPL), M=tb)
        else:
            cb = co_ctx(rng, tb, c["ind"])
        gkind = "neutral"
    elif kind == "company_acronym":
        c = mk_company(rng, test)
        st_words = c["stem"].split() if " " in c["stem"] else [c["stem"]]
        long_name = " ".join(st_words + [c["word"], rng.choice(["Group", "Holdings", "Partners", "Systems", "International"])])
        ac = acronym(long_name.split())
        ta, tb = long_name, ac
        typ_a = typ_b = "company"
        ca = co_ctx(rng, ta, c["ind"])
        cb = co_ctx(rng, tb, c["ind"]) if rng.random() < 0.7 else fill(rng, "{M} ({A}) {act}.", M=long_name, A=ac, act=fill(rng, rng.choice(ACT[c["ind"]])))
        lab.update(link=2, name=True, ctx=True, abbr=True)
        gkind = "neutral"
    elif kind == "ticker":
        c = mk_company(rng, test)
        ta = rng.choice([c["full"], c["base"]])
        tb = c["ticker"]
        typ_a, typ_b = "company", rng.choice(["security", "security", "company"])
        ca = co_ctx(rng, ta, c["ind"], c["ticker"] if rng.random() < 0.3 else None, c["ex"])
        cb = fill(rng, rng.choice(TICK_TPL), M=tb, EX=c["ex"])
        lab.update(link=2, name=None, ctx=None, abbr=True)
        if rng.random() < 0.5:
            lab["name"] = True
        gkind = rng.choice(["split", "lump", None, None])
    elif kind in ("ticker_diff", "acronym_diff"):                      # the short form belongs to another company
        c = mk_company(rng, test)
        d = mk_company(rng, test)
        while d["stem"][0] == c["stem"][0] or d["ind"] == c["ind"]:
            d = mk_company(rng, test)
        if kind == "ticker_diff":
            ta, tb = rng.choice([c["full"], c["base"]]), d["ticker"]
            typ_a, typ_b = "company", rng.choice(["security", "security", "company"])
            ca = co_ctx(rng, ta, c["ind"])
            cb = fill(rng, rng.choice(TICK_TPL), M=tb, EX=d["ex"]) if rng.random() < 0.6 else co_ctx(rng, tb, d["ind"])
            gkind = rng.choice(["split", "lump", None, None])
        else:
            words = c["stem"].split() + [c["word"], rng.choice(["Group", "Holdings", "Partners", "Systems", "International"])]
            dwords = d["stem"].split() + [d["word"], rng.choice(["Group", "Holdings", "Partners", "Systems", "International"])]
            ta, tb = " ".join(words), acronym(dwords)
            typ_a = typ_b = "company"
            ca, cb = co_ctx(rng, ta, c["ind"]), co_ctx(rng, tb, d["ind"])
            gkind = "neutral"
        lab.update(link=0, name=False, ctx=False, abbr=False)
    elif kind in ("company_shared", "company_diff"):
        c = mk_company(rng, test)
        d = mk_company(rng, test)
        if kind == "company_shared":
            ind2 = rng.choice([x for x in ACT if x != c["ind"]])
            d.update(stem=c["stem"], ind=ind2, word=rng.choice(INDUSTRY[ind2]))
            d["base"] = f"{d['stem']} {d['word']}"
            d["full"] = f"{d['base']} {d['suf']}"
            ta = rng.choice([c["full"], c["base"]])
            tb = rng.choice([d["full"], d["base"]])
            if rng.random() < 0.3:                                    # a bare stem whose context shows the other industry
                ta = c["stem"]
                lab.update(link=0, name=None, ctx=False, abbr=None)
            else:
                lab.update(link=0, name=False, ctx=False, abbr=False)
        else:
            ta, tb = rng.choice([c["full"], c["base"]]), rng.choice([d["full"], d["base"]])
            if d["ind"] == c["ind"]:
                d["ind"] = rng.choice([x for x in ACT if x != c["ind"]])
            lab.update(link=0, name=False, ctx=False, abbr=False)
        typ_a = typ_b = "company"
        ca, cb = co_ctx(rng, ta, c["ind"]), co_ctx(rng, tb, d["ind"])
        gkind = "neutral"
    elif kind in ("person_same", "person_diff", "person_namesake"):
        f, m, l = person(rng)
        org = mk_company(rng, test)["base"]
        role = rng.choice(P_ROLES)
        full = f"{f} {l}"
        typ_a = typ_b = "person"
        if kind == "person_same":
            sub = rng.choice(["hon", "surname", "initial", "poss", "middle"])
            ta = rng.choice([full, f"{rng.choice(['Dr.', 'Ms.', 'Mr.', 'Prof.'])} {full}"])
            tb = {"hon": f"{rng.choice(['Dr.', 'Ms.', 'Mr.', 'Prof.'])} {l}", "surname": l, "initial": f"{f[0]}. {l}", "poss": f"{rng.choice([l, full])}'s",
                  "middle": f"{f} {rng.choice('ABCDEFGHJKLMNPRSTW')}. {l}"}[sub]
            ca = fill(rng, rng.choice(P_TPL), M=ta, role=role, org=org)
            cb = fill(rng, rng.choice(POSS_TPL), M=tb) if sub == "poss" else fill(rng, rng.choice(P_TPL), M=tb, role=role, org=org)
            if sub == "poss" and rng.random() < 0.5:
                cb = cb[:-1] + f", {org}'s {role} said."
            lab.update(link=2, name=True, ctx=True if sub != "poss" else None, abbr=True if sub in ("surname", "initial") else None)
        elif kind == "person_diff":
            f2 = rng.choice([x for x in FIRST if x[0] != f[0]])
            ta, tb = full, f"{f2} {l}"
            ca = fill(rng, rng.choice(P_TPL), M=ta, role=role, org=org)
            cb = fill(rng, rng.choice(P_TPL_OTHER), M=tb, orole=rng.choice(OTHER_ROLES))
            if rng.random() < 0.35:                                   # the bare surname, but its context is the other person
                tb = rng.choice([l, f"Mr. {l}", f"Ms. {l}"])
                cb = fill(rng, rng.choice(P_TPL_OTHER), M=f"{f2} {l}", orole=rng.choice(OTHER_ROLES)) + " " + rng.choice([f"{tb} said the plan would cost jobs.", f"{tb} has led the campaign since {rng.randint(2010, 2024)}."])
                lab.update(link=0, name=None, ctx=False, abbr=None)
            else:
                lab.update(link=0, name=False, ctx=False, abbr=False)
        else:                                                         # same full name, unmistakably different people
            ta = tb = full
            ca = fill(rng, rng.choice(P_TPL), M=ta, role=role, org=org)
            cb = fill(rng, rng.choice(P_TPL_OTHER), M=tb, orole=rng.choice(OTHER_ROLES))
            lab.update(link=0, name=None, ctx=False, abbr=False)
        gkind = "neutral"
    elif kind in ("product_variant", "product_same", "product_diff"):
        brand = stem(rng)
        line = coin(rng) if rng.random() < 0.5 else rng.choice(["Nova", "Pulse", "Echo", "Flux", "Vista", "Orbit", "Prism", "Arc", "Volt", "Lumen", "Glide"])
        num = rng.choice([str(rng.randint(2, 99)), f"{rng.choice('ABCDNPQRSTVX')}{rng.randint(100, 990)}"])
        base = f"{line} {num}"
        typ_a = typ_b = "product"
        if kind == "product_variant":
            sub = rng.random()
            if sub < 0.6:
                tb = f"{base} {rng.choice(VAR_SUF)}" if not num[0].isalpha() or rng.random() < 0.5 else f"{base}{rng.choice(['X', 'S', 'e'])}"
            elif sub < 0.85:
                tb = f"{line} {int(re.sub(r'[^0-9]', '', num)) + rng.choice([1, 2, 10])}" if num.isdigit() else f"{base} {rng.choice(VAR_SUF)}"
            else:
                tb = f"{base} ({rng.choice(['2nd', '3rd'])} generation)"
            ta = base
            lab.update(link=1, name=False, ctx=None, abbr=None)
            gkind = rng.choice(["split", "lump", None, None])
        elif kind == "product_same":
            ta = rng.choice([base, f"{brand} {base}"])
            tb = rng.choice([base.replace(" ", "-"), f"{brand} {base}", base, f"{brand}'s {base}", f"the {base}"])
            if tb == ta:
                tb = f"{brand} {base}" if ta == base else base
            lab.update(link=2, name=True, ctx=True, abbr=None)
            gkind = "neutral"
        else:
            other = f"{coin(rng)} {rng.choice(['Dock', 'Hub', 'Charger', 'Stand', 'Case', 'Buds', 'Pad'])}"
            ta, tb = base, other
            lab.update(link=0, name=False, ctx=False, abbr=False)
            typ_b = rng.choice(["product", "accessory"])
            gkind = "neutral"
        ca = fill(rng, rng.choice(PROD_TPL), M=ta, brand=brand, spec=rng.choice(SPEC), g=rng.randint(150, 1900))
        cb = fill(rng, rng.choice(PROD_TPL), M=tb, brand=brand if kind != "product_diff" else stem(rng), spec=rng.choice(SPEC), g=rng.randint(150, 1900))
    elif kind in ("regulator_acr", "regulator_shared"):
        a, b, cc = rng.choice(REG_A), rng.choice(REG_B), rng.choice(REG_C)
        full = f"{a} {b} {cc}"
        co = mk_company(rng, test)["base"]
        typ_a = typ_b = "regulator"
        if kind == "regulator_acr":
            ta, tb = full, acronym(full.split())
            ca = fill(rng, rng.choice(REG_TPL), M=full, co=co, topic=REG_TOPIC[b])
            cb = fill(rng, rng.choice(REG_TPL), M=tb, co=co, topic=REG_TOPIC[b])
            lab.update(link=2, name=True, ctx=True, abbr=True)
        else:
            b2 = rng.choice([x for x in REG_B if x != b])
            ta, tb = full, f"{a} {b2} {cc}"
            ca = fill(rng, rng.choice(REG_TPL), M=ta, co=co, topic=REG_TOPIC[b])
            cb = fill(rng, rng.choice(REG_TPL), M=tb, co=mk_company(rng, test)["base"], topic=REG_TOPIC[b2])
            lab.update(link=0, name=False, ctx=False, abbr=False)
        ca, cb = ca.replace("The The", "The"), cb.replace("The The", "The")
        gkind = "neutral"
    elif kind == "metric":
        co = mk_company(rng, test)["base"]
        ab, full = rng.choice(METRIC)
        typ_a = typ_b = "financial_metric"
        if rng.random() < 0.6:
            ta, tb = full, ab
            lab.update(link=2, name=True, ctx=True, abbr=True)
        else:
            ab2, full2 = rng.choice([x for x in METRIC if x[0] != ab and {x[0], ab} != {"ARR", "MRR"}])
            ta, tb = rng.choice([full, ab]), rng.choice([full2, ab2])
            lab.update(link=0, name=False, ctx=None, abbr=False)
        ca = fill(rng, rng.choice(METRIC_TPL), M=ta, co=co)
        cb = fill(rng, rng.choice(METRIC_TPL), M=tb, co=co)
        gkind = "neutral"
    elif kind == "geo":
        co = mk_company(rng, test)["base"]
        typ_a = typ_b = "geography"
        if rng.random() < 0.5:
            ta, tb = rng.choice(GEO_TWINS)
            lab.update(link=0, name=False, ctx=None, abbr=False)
        else:
            ta, tb = rng.choice(GEO_ONE)
            lab.update(link=2, name=True, ctx=None, abbr=None)
        if rng.random() < 0.5:
            ta, tb = tb, ta
        ca = fill(rng, rng.choice(GEO_TPL), M=ta, co=co)
        cb = fill(rng, rng.choice(GEO_TPL), M=tb, co=mk_company(rng, test)["base"] if rng.random() < 0.5 else co)
        gkind = "neutral"
    elif kind == "warranty":
        y = rng.choice([1, 2, 3, 5])
        q = rng.choice(["limited warranty", "manufacturer's warranty", "warranty"])
        words = {1: "one", 2: "two", 3: "three", 5: "five"}
        ta = rng.choice([f"{y}-year {q}", f"{words[y]}-year {q}"])
        typ_a = typ_b = "warranty"
        if rng.random() < 0.5:
            tb = rng.choice([f"{12 * y}-month {q}", f"{words[y]}-year {q}", f"{y} year {q}"])
            if tb == ta:
                tb = f"{12 * y}-month {q}"
            lab.update(link=2, name=True, ctx=None, abbr=None)
        else:
            y2 = rng.choice([x for x in (1, 2, 3, 5) if x != y])
            tb = rng.choice([f"{12 * y2}-month {q}", f"{y2}-year {q}"])
            lab.update(link=0, name=False, ctx=None, abbr=False)
        ca, cb = fill(rng, rng.choice(WAR_TPL), M=ta), fill(rng, rng.choice(WAR_TPL), M=tb)
        gkind = "neutral"
    # ---- assemble
    ma, mb = mention(ta, typ_a, window(rng, ca, ta)), mention(tb, typ_b, window(rng, cb, tb))
    if rng.random() < 0.5:
        ma, mb = mb, ma
    state = {"mention_a": ma, "mention_b": mb}
    tk = ma["type"] if ma["type"] == mb["type"] else f"{ma['type']} or {mb['type']}"
    rows, task = [], "mention"
    asks = [q for q in ("link", "name", "ctx", "abbr") if lab[q] is not None]
    q = rng.choices(asks, [5 if a == "link" else 2 for a in asks])[0]
    if q == "link":
        g, level, p = "", lab["link"], None
        types = tk.split(" or ")
        if gkind in ("split", "lump"):
            g = f" In this domain, a {tk} is: " + " / ".join(rng.choice(GUIDE[gkind][t]) for t in types)
            if kind in ("ticker", "product_variant"):                # the rule decides what the pair is
                level = 0 if gkind == "split" else 2
                if gkind == "split":
                    p = pg(rng, 0.88, 0.92)
        elif gkind == "neutral" and rng.random() < 0.3 and all(t in GUIDE["neutral"] for t in types):
            g = f" In this domain, a {tk} is: " + " / ".join(GUIDE["neutral"][t] for t in types)
        if kind == "person_namesake":
            p = pg(rng, 0.88, 0.92)
        ins = pool(rng, M_LINK_Q, test).format(kind=tk, g=g)
        rows.append(score(rng, state, ins, pool(rng, M_LEVELS, test), level, task, i, split, p=p))
    elif q == "name":
        rows.append(verify(rng, state, pool(rng, M_NAME_Q, test), lab["name"], task, i, split, p=pg(rng, 0.88, 0.92) if kind in ("ticker", "warranty") else None))
    elif q == "ctx":
        rows.append(verify(rng, state, pool(rng, M_CTX_Q, test), lab["ctx"], task, i, split))
    else:
        rows.append(verify(rng, state, pool(rng, M_ABBR_Q, test), lab["abbr"], task, i, split))
    if DEBUG:
        for r in rows:
            r["dbg"] = f"{kind}/{gkind}"
    return rows


# ================================================================== C. pair relations (alternatives / complementary / same / unrelated)

# role → items; an item is "Name" or "Name|alias|alias" (aliases are the same thing)
ROLES = {
    "pkg_js": ["npm", "pnpm", "Yarn", "Bun"],
    "pkg_py": ["pip", "Poetry", "uv", "Pipenv", "PDM", "Conda"],
    "lang": ["Java", "C#|C Sharp", "Go|Golang", "Rust", "Python", "Ruby", "Elixir", "Scala", "PHP", "Kotlin", "Clojure", "Erlang", "Haskell", "OCaml", "Perl", "Crystal"],
    "rdbms": ["PostgreSQL|Postgres", "MySQL", "MariaDB", "Microsoft SQL Server|SQL Server|MSSQL", "Oracle Database", "CockroachDB", "IBM Db2"],
    "docdb": ["MongoDB|Mongo", "Couchbase", "CouchDB|Apache CouchDB", "RavenDB", "Firestore|Cloud Firestore"],
    "cache": ["Redis", "Memcached", "Valkey", "Dragonfly|DragonflyDB"],
    "graphdb": ["Neo4j", "Memgraph", "TigerGraph", "Amazon Neptune|Neptune", "JanusGraph"],
    "search": ["Elasticsearch", "OpenSearch", "Apache Solr|Solr", "Meilisearch", "Typesense"],
    "warehouse": ["Snowflake", "Google BigQuery|BigQuery", "Amazon Redshift|Redshift", "ClickHouse", "Azure Synapse|Synapse Analytics"],
    "ci": ["Jenkins", "GitHub Actions|GHA", "GitLab CI|GitLab CI/CD", "CircleCI", "Travis CI|Travis", "Buildkite", "TeamCity", "Drone CI", "Azure Pipelines", "Bitbucket Pipelines"],
    "cloud": ["Amazon Web Services|AWS", "Microsoft Azure|Azure", "Google Cloud Platform|GCP|Google Cloud", "Oracle Cloud|OCI", "IBM Cloud", "DigitalOcean", "Hetzner"],
    "editor": ["Visual Studio Code|VS Code|VSCode", "Vim", "Neovim", "Emacs|GNU Emacs", "Sublime Text", "IntelliJ IDEA|IntelliJ", "Zed", "Helix", "Notepad++"],
    "web_py": ["Django", "Flask", "FastAPI", "Pyramid", "Falcon"],
    "web_js": ["Express|Express.js", "Fastify", "Koa", "NestJS", "Hapi"],
    "web_rb": ["Ruby on Rails|Rails|RoR", "Sinatra", "Hanami"],
    "web_jvm": ["Spring Boot", "Quarkus", "Micronaut"],
    "frontend": ["React|React.js|ReactJS", "Vue.js|Vue", "Angular", "Svelte", "SolidJS", "Ember.js|Ember", "Preact"],
    "orch": ["Kubernetes|K8s", "Docker Swarm", "HashiCorp Nomad|Nomad", "Apache Mesos|Mesos", "Amazon ECS|ECS"],
    "container": ["Docker", "Podman"],
    "mq": ["Apache Kafka|Kafka", "RabbitMQ", "Amazon SQS|SQS", "NATS", "Apache Pulsar|Pulsar", "ActiveMQ"],
    "apm": ["Datadog", "New Relic", "Dynatrace", "AppDynamics", "Honeycomb"],
    "vcs": ["GitHub", "GitLab", "Bitbucket", "Gitea", "SourceHut"],
    "wiki": ["Confluence", "Notion", "BookStack", "MediaWiki", "Wiki.js"],
    "desktop_os": ["Windows|Microsoft Windows", "macOS|Mac OS|OS X", "Linux|GNU/Linux", "ChromeOS|Chrome OS"],
    "phone_os": ["Android", "iOS", "HarmonyOS"],
    "browser": ["Google Chrome|Chrome", "Mozilla Firefox|Firefox", "Safari", "Microsoft Edge|Edge", "Brave", "Opera", "Vivaldi"],
    "payments": ["Stripe", "Adyen", "PayPal", "Braintree", "Checkout.com", "Mollie", "Razorpay"],
    "airline": ["Lufthansa", "Air France", "KLM|KLM Royal Dutch Airlines", "Delta Air Lines|Delta", "United Airlines", "Qantas", "Turkish Airlines", "Ryanair", "easyJet", "Southwest Airlines|Southwest", "Iberia", "Finnair"],
    "chat": ["Slack", "Microsoft Teams|Teams|MS Teams", "Discord", "Mattermost", "Zulip", "Rocket.Chat"],
    "video": ["Zoom", "Google Meet", "Webex|Cisco Webex", "Jitsi Meet|Jitsi", "GoTo Meeting"],
    "streaming": ["Netflix", "Disney+|Disney Plus", "Hulu", "Apple TV+", "Amazon Prime Video|Prime Video", "Paramount+"],
    "ride": ["Uber", "Lyft", "DiDi", "Cabify", "FreeNow"],
    "maps": ["Google Maps", "Apple Maps", "Waze", "HERE WeGo", "Citymapper"],
    "password": ["1Password", "Bitwarden", "LastPass", "Dashlane", "KeePass", "Proton Pass"],
    "search_engine": ["Google Search", "Bing|Microsoft Bing", "DuckDuckGo", "Kagi", "Ecosia"],
    "transport": ["train|rail", "bus", "plane|airplane|aeroplane", "car"],
    "hot_drink": ["coffee", "tea", "hot chocolate|cocoa"],
    "plant_milk": ["oat milk", "almond milk", "soy milk|soya milk", "rice milk", "coconut milk"],
    "spread": ["butter", "margarine"],
    "sweetener": ["sugar", "honey", "stevia", "maple syrup", "agave syrup"],
    "side_grain": ["rice", "couscous", "quinoa", "bulgur"],
    "city": ["Vienna|Wien", "Prague|Praha", "Madrid", "Amsterdam", "Copenhagen", "Stockholm", "Oslo", "Helsinki", "Warsaw", "Budapest", "Athens",
             "Toronto", "Montreal", "Vancouver", "Austin", "Denver", "Boston", "Melbourne", "Sydney", "Auckland", "Nairobi", "Lagos", "Cairo",
             "Mumbai|Bombay", "Delhi", "Bangkok", "Jakarta", "Manila", "Mexico City", "Bogotá|Bogota", "Lima", "Buenos Aires", "Cape Town",
             "Zurich|Zürich", "Munich|München", "Hamburg", "Milan|Milano", "New York City|NYC|New York", "Los Angeles|LA", "San Francisco|SF",
             "Ho Chi Minh City|Saigon", "Saint Petersburg|St. Petersburg", "Berlin"],
}
ROLE_TEST = {"password", "ride", "mq", "sweetener", "search_engine", "web_rb"}          # held out of training
TECH = {"pkg_js", "pkg_py", "lang", "rdbms", "docdb", "cache", "graphdb", "search", "warehouse", "ci", "cloud", "editor", "web_py", "web_js",
        "web_rb", "web_jvm", "frontend", "orch", "container", "mq", "apm", "vcs", "wiki"}
COMP_ROLES = [("lang", "rdbms"), ("lang", "docdb"), ("lang", "cache"), ("lang", "ci"), ("lang", "cloud"), ("lang", "editor"), ("lang", "mq"),
              ("rdbms", "cloud"), ("rdbms", "cache"), ("rdbms", "search"), ("ci", "cloud"), ("container", "orch"), ("frontend", "web_py"),
              ("frontend", "web_js"), ("frontend", "web_jvm"), ("frontend", "web_rb"), ("web_py", "rdbms"), ("web_js", "rdbms"), ("web_jvm", "rdbms"),
              ("web_rb", "rdbms"), ("apm", "cloud"), ("chat", "video"), ("password", "browser"), ("orch", "cloud"), ("warehouse", "lang"),
              ("graphdb", "lang"), ("vcs", "editor"), ("wiki", "chat"), ("mq", "rdbms")]
OWN_LANG = {"web_py": "Python", "web_js": "JavaScript|JS", "web_rb": "Ruby", "web_jvm": "Java", "pkg_js": "JavaScript|JS", "pkg_py": "Python"}
FOOD_PAIRS = [("bread", "butter"), ("fish", "chips"), ("peanut butter", "jelly"), ("burger", "fries"), ("cereal", "milk"), ("coffee", "croissant"),
              ("tea", "biscuits"), ("salsa", "tortilla chips"), ("pasta", "parmesan"), ("rice", "curry"), ("wine", "cheese"), ("hummus", "pita bread"),
              ("strawberries", "cream"), ("bacon", "eggs"), ("hot dog", "mustard"), ("pancakes", "maple syrup"), ("sushi", "soy sauce")]
FOOD_TEST = {("pancakes", "maple syrup"), ("sushi", "soy sauce"), ("hot dog", "mustard")}
UNREL_ROLES = [("airline", "lang"), ("airline", "rdbms"), ("airline", "editor"), ("airline", "ci"), ("hot_drink", "orch"), ("plant_milk", "ci"),
               ("sweetener", "cloud"), ("streaming", "rdbms"), ("ride", "editor"), ("password", "side_grain"), ("transport", "docdb"),
               ("browser", "spread"), ("phone_os", "sweetener"), ("maps", "plant_milk"), ("side_grain", "frontend"), ("spread", "ci"),
               ("airline", "frontend"), ("streaming", "pkg_py"), ("hot_drink", "warehouse"), ("plant_milk", "graphdb"), ("airline", "search"),
               ("transport", "web_js"), ("side_grain", "apm"), ("payments", "side_grain"), ("streaming", "cache")]
# The KG notebooks' own corpus items: any pair of two of these stays out of training.
CORPUS = {"npm", "pnpm", "python", "go", "jenkins", "neo4j", "postgresql", "confluence", "berlin", "cypher"}
CORPUS_EXTRA = {frozenset({"jenkins", "github actions"})}


AMBIG = {"Neptune", "Ember", "Edge", "Delta", "Southwest", "Travis", "Kafka", "Pulsar", "Nomad", "Teams", "Express", "Rails", "Chrome", "Mesos"}


def names(item):
    return item.split("|")


def nm(rng, item):
    """A name for a cross-kind pair: never a one-word alias that is also an ordinary word (those only appear in 'same' pairs)."""
    return rng.choice([n for n in names(item) if n not in AMBIG] or names(item)[:1])


def canon(item):
    return names(item)[0].lower()


def corpus_pair(x, y) -> bool:
    a, b = canon(x), canon(y)
    return (a in CORPUS and b in CORPUS) or frozenset({a, b}) in CORPUS_EXTRA


REL_ALT_Q = [
    "Are `a` and `b` alternatives -- two options filling the same role, so that choosing one usually means dropping the other? Answer no if they are different kinds of thing, or things commonly used together.",  # verbatim
    "Do `a` and `b` compete for the same role, so that someone usually picks one or the other?",
    "Are `a` and `b` substitutes for each other?",
    "Would adopting `a` usually mean giving up `b`?",
    "Is `b` an alternative to `a`: same job, and people normally settle on one?",
    "If a user switched from `a` to `b`, would that be a switch between two options for the same role?",
    "Are these two interchangeable choices for one purpose (not the same thing under two names, and not things used together)?",
    "Do `a` and `b` fill the same slot, such that one replaces the other?",
    "Is one of `a` and `b` a drop-in rival to the other?",                                                               # held out
    "Should preferring `b` supersede an earlier preference for `a`?",                                                     # held out
]
REL_ALT_CRIT = [{"true": "They compete for the same role; a person typically settles on one.", "false": "They do not compete: different kinds of thing, or complementary, or the same thing."},
                {"true": "Same role, competing options.", "false": "Different roles, used together, unrelated, or two names for one thing."}]
REL_KIND_Q = ["How do `a` and `b` relate to each other?", "What is the relationship between `a` and `b`?", "Classify the pair (`a`, `b`).",
              "How are these two things related?", "Which of these describes `a` and `b`?", "Pick the relation that holds between `a` and `b`.",
              "In a user profile, how would `a` and `b` relate?", "Relation between the two items?",
              "Which relation links `a` to `b`?", "Characterise how `a` stands to `b`."]                                         # last two held out
REL_KIND_CRIT = [
    {"alternatives": "Two options for the same role, such that adopting one usually means giving the other up: two package managers, two programming languages for the same job, two CI systems.",
     "complementary": "Different kinds of thing that are commonly held together and do not compete: a language and a database, a tool and a city.",
     "same": "Two names for one and the same thing.", "unrelated": "Nothing to do with each other."},                       # verbatim
    {"alternatives": "Competing options for one role.", "complementary": "Different roles, often used together.", "same": "One thing, two names.", "unrelated": "No connection."},
    {"alternatives": "Rivals: you would normally pick one of them.", "complementary": "They go together and do not compete.", "same": "Aliases or spellings of a single thing.",
     "unrelated": "They have nothing to do with each other."},
    {"alternatives": "Substitutes for the same job.", "complementary": "Parts of one setup or routine, used side by side.", "same": "Identical referent.", "unrelated": "Different worlds entirely."},  # held out
]


def relpair_case(rng, i, split):
    test = split == "test"
    hold = test and rng.random() < 0.5

    def allowed(*rs):                                                 # held-out roles: at least one in test-held mode, none otherwise
        h = any(r in ROLE_TEST for r in rs)
        return h if hold else not h

    noul = rng.random() < 0.45
    k = rng.random()
    if noul:                                                           # the alternatives noul: about 45 % positives
        k = 0.0 if rng.random() < 0.45 else 0.3 + 0.7 * rng.random()
    for _ in range(50):
        if k < 0.3:
            kind = "alternatives"
            r = rng.choice([x for x in ROLES if x != "city" and allowed(x)])
            x, y = rng.sample(ROLES[r], 2)
        elif k < 0.55:
            kind = "complementary"
            sub = rng.random()
            own = [r for r in OWN_LANG if allowed(r)]
            tech = [r for r in TECH if allowed(r)]
            if sub < 0.55:
                r1, r2 = rng.choice([p for p in COMP_ROLES if allowed(*p)])
                x, y = rng.choice(ROLES[r1]), rng.choice(ROLES[r2])
            elif sub < 0.7 and own:
                r1 = rng.choice(own)
                x, y = rng.choice(ROLES[r1]), OWN_LANG[r1]
            elif sub < 0.85 or not tech:
                fp = [p for p in FOOD_PAIRS if (p in FOOD_TEST) == hold]
                x, y = rng.choice(fp)
            else:                                                     # a tool and a city
                x, y = rng.choice(ROLES[rng.choice(tech)]), rng.choice(ROLES["city"])
        elif k < 0.75:
            kind = "same"
            x = y = rng.choice([it for r in ROLES if allowed(r) for it in ROLES[r] if "|" in it])
        else:
            kind = "unrelated"
            r1, r2 = rng.choice([p for p in UNREL_ROLES if allowed(*p)])
            x, y = rng.choice(ROLES[r1]), rng.choice(ROLES[r2])
        if kind == "same" or test or not corpus_pair(x, y):
            break
    else:
        return []
    if kind == "same":
        ns = names(x)
        if rng.random() < 0.85:
            a, b = rng.sample(ns, 2)
        else:                                                         # one name, written differently
            a = rng.choice(ns)
            b = a.lower() if a.lower() != a else a.title() if a.title() != a else a.upper()
            if b == a:
                a, b = rng.sample(ns, 2)
    else:
        a, b = nm(rng, x), nm(rng, y)
    if not test and kind != "same" and corpus_pair(x, y):
        return []
    if rng.random() < 0.5:
        a, b = b, a
    state = {"a": a, "b": b}
    task = "relpair"
    if noul:
        crit = rng.choice(REL_ALT_CRIT) if rng.random() < 0.7 else None
        ins = pool(rng, REL_ALT_Q, test)
        return [verify(rng, state, ins, kind == "alternatives", task, i, split, crit=crit)]
    cs = REL_KIND_CRIT[-1] if test and rng.random() < 0.5 else rng.choice(REL_KIND_CRIT[:-1])
    if rng.random() < 0.3:
        it = list(cs.items())
        rng.shuffle(it)
        cs = dict(it)
    return [choose(rng, state, pool(rng, REL_KIND_Q, test), cs, kind, task, i, split)]


# ================================================================== D. modality of a sentence / claim

# relation → (base, past/pp, gerund, KG relation name, description). {O} object, {P} person, {X} product, {L} place.
RELS = {
    "acquire": ("acquire {O}", "acquired {O}", "acquiring {O}", "acquires", "The head company is buying or has bought the tail."),
    "partner": ("partner with {O}", "partnered with {O}", "partnering with {O}", "partners_with", "A stated partnership, alliance, or joint venture."),
    "supply": ("supply {G} to {O}", "supplied {G} to {O}", "supplying {G} to {O}", "supplies", "The head provides goods or inputs to the tail company."),
    "sue": ("sue {O}", "sued {O}", "suing {O}", "sues", "The head has brought or is bringing a lawsuit against the tail."),
    "hire": ("hire {P} as {R}", "hired {P} as {R}", "hiring {P} as {R}", "employs", "The head company employs or appoints the tail person."),
    "launch": ("launch {X}", "launched {X}", "launching {X}", "launches", "The head company releases the tail product to the market."),
    "invest": ("invest ${n} million in {O}", "invested ${n} million in {O}", "investing ${n} million in {O}", "invests_in", "The head puts money into the tail."),
    "open": ("open a {F} in {L}", "opened a {F} in {L}", "opening a {F} in {L}", "operates_in", "The head runs a facility or business in the tail place."),
    "merge": ("merge with {O}", "merged with {O}", "merging with {O}", "merges_with", "The two companies combine into one."),
    "divest": ("sell its {S} unit to {O}", "sold its {S} unit to {O}", "selling its {S} unit to {O}", "sells_unit_to", "The head sells a business unit to the tail."),
    "license": ("license its {T} technology to {O}", "licensed its {T} technology to {O}", "licensing its {T} technology to {O}", "licenses_to", "The head grants the tail rights to use its technology."),
}
GOODS = ["battery cells", "steel coils", "packaging film", "control boards", "frozen vegetables", "jet fuel", "optical sensors", "cardboard cartons", "solar glass", "brake pads"]
FAC = ["plant", "distribution centre", "research lab", "data centre", "final-assembly line", "flagship store"]
SEG = ["consumer", "industrial coatings", "freight brokerage", "payments", "animal health", "spare parts", "insurance", "media"]
TECHW = ["battery", "imaging", "compression", "speech-recognition", "water-filtration", "encryption"]
PLACES = ["Ohio", "Bavaria", "Monterrey", "Gdańsk", "Lyon", "Nairobi", "Ontario", "Kraków", "Rotterdam", "Porto Alegre", "Hanoi", "Queensland"]
EXEC_ROLES = ["chief financial officer", "head of research", "chief executive", "general counsel", "chief operating officer", "head of sales"]
WHEN = [" in {M} {Y}", " last year", " on {D}", " in {Y}", " earlier this month", "", "", " last quarter"]
FWHEN = [" next year", " by {Y2}", " in {M} {Y2}", " later this year", " within {k} months", " in the second half"]
MOD = {  # modality → list of (template, held_out?) ; {S} subject, {base}/{past}/{ing}, {when}/{fwhen}
    "asserted": [("{S} {past}{when}.", 0), ("{S} has {past}.", 0), ("{S} said on {D} that it had {past}.", 0), ("{S} confirmed that it {past}{when}.", 0),
                 ("In a filing, {S} disclosed that it {past}{when}.", 0), ("Although analysts had not expected the move, {S} {past}{when}.", 0),
                 ("{S}, which does not disclose deal terms, {past}{when}.", 0), ("{S} {past}{when}, a move that could strengthen its position in {L2}.", 0),
                 ("{S}, which will report earnings on {D}, {past}{when}.", 0), ("Contrary to earlier reports, {S} did {base}{when}.", 0),
                 ("{S} announced that it had {past}.", 0), ("{S} {past}{when}, the company's spokesperson said.", 0),
                 ("Records show that {S} {past} in {Y}.", 1), ("{S} went ahead and {past}{when}.", 1)],
    "hypothetical": [("{S} may {base}.", 0), ("{S} might {base}.", 0), ("{S} could {base} if {cond}.", 0), ("{S} is considering {ing}.", 0),
                     ("{S} is in talks to {base}.", 0), ("If approved by regulators, {S} would {base}.", 0), ("{S} would {base} if {cond}.", 0),
                     ("Pending regulatory approval, {S} would {base}.", 0), ("{S} is rumoured to be preparing to {base}.", 0),
                     ("People familiar with the matter said {S} could {base}.", 0), ("{S} has not ruled out {ing}.", 0), ("{S} has floated the idea of {ing}.", 0),
                     ("Should the board approve it, {S} would {base}.", 0), ("Sources said {S} {past}, though the company has not confirmed it.", 0),
                     ("{S} is mulling {ing}.", 1), ("{S} is exploring whether to {base}.", 1), ("{S} is reportedly weighing whether to {base}.", 1)],
    "negated": [("{S} denied that it planned to {base}.", 0), ("{S} has ruled out {ing}.", 0), ("{S} will not {base}.", 0), ("{S} has no plans to {base}.", 0),
                ("{S} does not expect to {base}.", 0), ("{S} did not {base}{when}.", 0), ("{S} denied reports that it had {past}.", 0),
                ("It is not true that {S} {past}, a spokesperson said.", 0), ("{S} said it is not in talks to {base}.", 0), ("{S} has never {past}.", 0),
                ("{S} walked away from plans to {base}.", 0), ("{S} no longer intends to {base}.", 0), ("{S} confirmed it will not {base}.", 0),
                ("{S} scrapped plans to {base}.", 1), ("{S} abandoned its effort to {base}.", 1), ("{S} rejected the idea of {ing}.", 1)],
    "forward_looking": [("{S} expects to {base}{fwhen}.", 0), ("{S} plans to {base}{fwhen}.", 0), ("{S} will {base}{fwhen}.", 0), ("{S} intends to {base}{fwhen}.", 0),
                        ("{S} forecast that it would {base}{fwhen}.", 0), ("{S} is set to {base}{fwhen}.", 0), ("{S} aims to {base}{fwhen}.", 0),
                        ("Management expects {S} to {base}{fwhen}.", 0), ("{S} guided investors to expect it to {base}{fwhen}.", 0),
                        ("{S} is poised to {base}{fwhen}.", 1), ("{S} is scheduled to {base}{fwhen}.", 1), ("{S} anticipates that it will {base}{fwhen}.", 1)],
}
COND = ["shareholders approve the deal", "financing is secured", "the regulator signs off", "prices recover", "talks succeed", "the board agrees"]
ST_CRIT = [
    {"asserted": "The text states this relationship as a present or past fact. It holds, or it held.",
     "hypothetical": "The text raises this relationship only as a possibility, a condition, a proposal awaiting approval, or something that would happen if something else did. It is not claimed to have happened.",
     "negated": "The text denies this relationship, or states that it will not happen, is not planned, or is not expected.",
     "forward_looking": "The text projects, forecasts or guides that this relationship will hold in the future. It has not happened yet, but it is not merely conditional."},  # verbatim
    {"asserted": "Stated as fact, now or in the past.", "hypothetical": "Only possible, conditional, rumoured or proposed.", "negated": "Denied, ruled out, or said not to be planned.",
     "forward_looking": "Projected or planned for the future, not conditional."},
    {"asserted": "It happened or holds: a plain statement of fact.", "hypothetical": "It might happen, could happen, would happen if…, or is only reported by unnamed sources.",
     "negated": "The text says it did not, will not, or is not expected to happen.", "forward_looking": "The text says it is expected, planned or scheduled to happen."},
    {"asserted": "Factual claim.", "hypothetical": "Speculative or conditional claim.", "negated": "Denied claim.", "forward_looking": "Future-tense projection."},  # held out
]
ST_Q_SENT = [
    "Does `sentence` assert its main claim as a fact, or does it hedge, deny, project or merely propose it? Judge the modality of the sentence, not whether the claim is plausible.",  # verbatim
    "How does `sentence` present its main claim?",
    "What is the modality of the main claim in `sentence`?",
    "Is the main claim of the sentence stated as fact, hedged, denied or projected?",
    "Classify how certain the sentence is about what it reports.",
    "Judge how `sentence` presents its claim, not whether it is true.",
    "Is the event in `sentence` asserted, hypothetical, negated or forward-looking?",
    "Read `sentence`. Does it state, speculate about, deny or forecast its main event?",
    "How committed is the writer of `sentence` to its main claim?",                                                      # held out
    "Which modality best fits the sentence's central claim?",                                                             # held out
]
ST_Q_EDGE = [
    "How does the {doc} present this claim: “{h}” — {rel} → “{t}”? Here, “{rel}” means: {desc} Judge only how it is presented, not whether it is plausible.",  # verbatim
    "In `{key}`, how is the relationship “{h}” — {rel} → “{t}” presented? (“{rel}”: {desc})",
    "Claim: “{h}” {rel} “{t}”. How does the {doc} present it?",
    "What is the modality of the claim that “{h}” {rel} “{t}” in the {doc}? Here, “{rel}” means: {desc}",
    "Judge how the {doc} treats this edge: “{h}” — {rel} → “{t}”.",
    "Does the {doc} assert, hedge, deny or forecast that “{h}” {rel} “{t}”?",
    "Edge “{h}” — {rel} → “{t}”: how does the text present it? Ignore other claims in the text.",
    "For the relation {rel} from “{h}” to “{t}”, how does `{key}` present it?",
    "How is the link between “{h}” and “{t}” ({rel}) worded in the {doc}?",                                               # held out
    "Status of “{h}” — {rel} → “{t}” in the {doc}?",                                                                     # held out
]
FACT_Q = [
    ("If a knowledge graph recorded the relationship this sentence is about as a plain fact, would the graph be correct? Be careful with denials: a sentence saying something will NOT happen or is NOT planned does not make that thing a fact.",
     {"true": "Yes: the sentence asserts the relationship as true now or in the past.", "false": "No: the sentence denies it, conditions it, proposes it, forecasts it, or only reports that someone said it might happen."}),  # verbatim
    ("Can the main claim of `sentence` be recorded as a fact that currently holds?",
     {"true": "The sentence asserts it as true now or in the past.", "false": "The sentence denies it, conditions it, proposes it, or places it in the future."}),  # first wording
    ("Is the main claim of `sentence` a fact, as written?", None),
    ("Does `sentence` state that its main event actually happened or holds?", None),
    ("Would it be correct to store the sentence's main claim as an established fact?", None),
    ("Is the sentence a factual statement of its main claim, rather than a denial, a condition, a rumour or a forecast?", None),
    ("Taken at face value, does `sentence` report something that has happened or is true now?", None),
    ("Should a fact extractor record the main relation in `sentence` as true?", None),
    ("Does the sentence commit to its main claim as true?", None),                                                        # held out
    ("Is the central claim of `sentence` asserted as fact?", None),                                                       # held out
]
FACT_Q_EDGE = [
    "If a knowledge graph recorded “{h}” — {rel} → “{t}” as a plain fact based on the {doc}, would the graph be correct?",
    "Does the {doc} state as fact that “{h}” {rel} “{t}”?",
    "Is “{h}” — {rel} → “{t}” asserted as true (now or in the past) by the {doc}?",
    "Can the edge “{h}” — {rel} → “{t}” be recorded as a fact from this {doc}?",
    "According to the {doc}, is it a fact that “{h}” {rel} “{t}”? Denials, conditions, rumours and forecasts do not count.",
    "Would recording “{h}” {rel} “{t}” as fact be supported by the {doc}?",
    "Does the text assert the claim “{h}” — {rel} → “{t}”?",
    "Is “{h}” {rel} “{t}” stated as a fact in `{key}`?",
    "Is the edge “{h}” — {rel} → “{t}” a fact according to the text?",                                                    # held out
    "Does the {doc} make “{h}” {rel} “{t}” a fact?",                                                                     # held out
]
SUP_Q = [
    ("Does the {doc} say anything that bears on this claim: “{h}” — {rel} → “{t}”? Here, “{rel}” means: {desc} Answer yes if the {doc} discusses this relationship at all, in any modality — asserted, denied, proposed or merely possible. Answer no if the {doc} does not connect these two things.",
     {"true": "The {doc} addresses this relationship between these two specific things.", "false": "The {doc} does not connect these two things, or one of them does not appear in it."}),  # verbatim (document → doc)
    ("Does the {doc} discuss “{h}” {rel} “{t}” in any way, whether as fact, denial, plan or possibility?", None),
    ("Is the relationship “{h}” — {rel} → “{t}” mentioned at all in `{key}`, in any modality?", None),
    ("Does the text connect “{h}” and “{t}” through {rel}, even if only as a rumour or a denial?", None),
    ("Is there anything in the {doc} about “{h}” {rel} “{t}”?", None),
    ("Does the {doc} address the claim “{h}” — {rel} → “{t}” (asserted, denied, planned or hypothetical)?", None),
    ("Relevance check: does the {doc} talk about “{h}” and “{t}” being linked by {rel}?", None),
    ("Is the edge “{h}” — {rel} → “{t}” discussed by the text in any modality?", None),
    ("Does the {doc} bear on “{h}” — {rel} → “{t}” at all?", None),                                                        # held out
    ("Is “{h}” {rel} “{t}” raised anywhere in the {doc}?", None),                                                          # held out
]
FILL_SENT = ["The company's shares were little changed.", "Revenue for the quarter was ${n} million.", "The statement was released before markets opened.",
             "Analysts had expected the announcement.", "Weather disrupted shipments in the north.", "The group employs about {k} people.",
             "A conference call is scheduled for {D}.", "Margins improved on lower freight costs.", "No financial terms were disclosed for the earlier deal."]


def mk_subj(rng, test):
    c = mk_company(rng, test)
    return rng.choice([c["base"], c["base"], c["full"], c["stem"] if len(c["stem"]) > 5 else c["base"]]), c


def clause(rng, test, modality, rel, S, O):
    """One sentence about (S, rel, O) in the given modality; returns (sentence, tail text)."""
    held = int(test and rng.random() < 0.5)                            # held-out cue words: test only
    t = rng.choice([t for t, h in MOD[modality] if h == held])
    base, past, ing, _, _ = RELS[rel]
    P = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    X = f"{coin(rng)} {rng.choice(['One', 'Pro', 'Cloud', 'Go', str(rng.randint(2, 9))])}"
    L = rng.choice(PLACES)
    kv = dict(O=O, G=rng.choice(GOODS), P=P, R=rng.choice(EXEC_ROLES), X=X, n=rng.choice([20, 45, 120, 300, 750]), F=rng.choice(FAC), L=L,
              S=rng.choice(SEG), T=rng.choice(TECHW))
    tail = {"hire": P, "launch": X, "open": L}.get(rel, O)
    b, pa, g = (s.format(**kv) for s in (base, past, ing))
    y = rng.randint(2019, 2025)
    w = rng.choice(WHEN).format(M=rng.choice(MONTHS), Y=y, D=rng.choice(DAYS))
    fw = rng.choice(FWHEN).format(M=rng.choice(MONTHS), Y2=rng.randint(2026, 2030), k=rng.choice([6, 12, 18]))
    if modality == "asserted" and rel == "acquire" and rng.random() < 0.15:
        sent = rng.choice([f"{O} is a wholly owned subsidiary of {S}.", f"{S} completed its acquisition of {O}{w}.", f"{S} closed its purchase of {O}{w}."])
    elif modality == "asserted" and rel == "supply" and rng.random() < 0.3:
        sent = f"{S} supplies {kv['G']} to {O}" + rng.choice([".", ", according to its annual report.", " under a long-term contract."])
    else:
        sent = t.format(S=S, base=b, past=pa, ing=g, when=w, fwhen=fw, cond=rng.choice(COND), D=rng.choice(DAYS), Y=y, L2=rng.choice(PLACES))
    sent = re.sub(r"\.\.(?=\s|$|;)", ".", fixposs(sent))
    return sent[0].upper() + sent[1:], tail


MODS = ["asserted", "hypothetical", "negated", "forward_looking"]


def modality_case(rng, i, split):
    test = split == "test"
    task = "modality"
    k, q = rng.random(), rng.random()
    fact_q = (k < 0.5 and q >= 0.6) or (k >= 0.5 and 0.55 <= q < 0.8)
    if fact_q:                                                         # factual nouls: about 45 % asserted
        mod = "asserted" if rng.random() < 0.45 else rng.choice(MODS[1:])
    else:
        mod = rng.choice(MODS)
    rels = [r for r in RELS if r != "license" or test]                 # 'license' is a held-out relation
    rel = rng.choice(rels)
    S, _ = mk_subj(rng, test)
    O, _ = mk_subj(rng, test)
    if O == S:
        return []
    sent, tail = clause(rng, test, mod, rel, S, O)
    soft = None
    p = None
    if sent.startswith("Sources said"):
        soft, p = {"asserted": 0.06}, pg(rng, 0.9, 0.93)
    rows = []
    if k < 0.5:                                                        # sentence-level questions (as in the modality-trap notebook)
        state = {"sentence": sent}
        if q < 0.6:
            crit = ST_CRIT[-1] if test and rng.random() < 0.5 else rng.choice(ST_CRIT[:-1])
            if rng.random() < 0.3:
                it = list(crit.items())
                rng.shuffle(it)
                crit = dict(it)
            rows.append(choose(rng, state, pool(rng, ST_Q_SENT, test), crit, mod, task, i, split, p=p, soft=soft))
        else:
            ins, crit = pool(rng, FACT_Q, test)
            rows.append(verify(rng, state, ins, mod == "asserted", task, i, split, crit=crit))
        return rows
    # edge-level questions over a sentence (possibly two clauses) or a short document
    _, _, _, rname, desc = RELS[rel]
    clauses = [(sent, S, rname, tail, mod, desc)]
    for _ in range(rng.choice([0, 1, 1, 2, 3])):
        m2, r2 = rng.choice(MODS), rng.choice(rels)
        S2, _ = mk_subj(rng, test)
        O2, _ = mk_subj(rng, test)
        if len({S, O, S2, O2}) < 4:
            continue
        s2, t2 = clause(rng, test, m2, r2, S2, O2)
        clauses.append((s2, S2, RELS[r2][3], t2, m2, RELS[r2][4]))
    order = clauses[:]
    rng.shuffle(order)
    if len(order) == 2 and rng.random() < 0.5:                         # two clauses in one sentence
        a, b = order[0][0], order[1][0]
        if not b.startswith(order[1][1]):
            b = b[0].lower() + b[1:]
        text = a[:-1] + rng.choice(["; separately, ", "; meanwhile, ", "; in other news, "]) + b
        key, doc = "sentence", "sentence"
    else:
        parts = [c[0] for c in order]
        for f_ in rng.sample(FILL_SENT, rng.randint(0, 3)):
            parts.insert(rng.randint(0, len(parts)), fill(rng, f_, D=rng.choice(DAYS)))
        text = " ".join(parts)
        key, doc = ("document", "document") if len(parts) > 1 else ("sentence", "sentence")
    state = {key: text}
    s_, h, rn, t, m, ds = clauses[0]                                   # the asked-about clause; its place in the text is shuffled
    fmt = dict(h=h, t=t, rel=rn, desc=ds, doc=doc, key=key)
    soft = {"asserted": 0.06} if s_.startswith("Sources said") else None
    if q < 0.55:
        crit = ST_CRIT[0] if rng.random() < 0.6 else (ST_CRIT[-1] if test and rng.random() < 0.5 else rng.choice(ST_CRIT[:-1]))
        rows.append(choose(rng, state, pool(rng, ST_Q_EDGE, test).format(**fmt), crit, m, task, i, split, p=pg(rng, 0.9, 0.93) if soft else None, soft=soft))
    elif q < 0.8:
        rows.append(verify(rng, state, pool(rng, FACT_Q_EDGE, test).format(**fmt), m == "asserted", task, i, split))
    else:
        # supported: the true edge, or a head and tail from two different clauses (not connected)
        if len(clauses) >= 2 and rng.random() < 0.5:
            c1, c2 = rng.sample(clauses, 2)
            fmt.update(h=c1[1], t=c2[3], rel=c1[2], desc=c1[5])
        elif rng.random() < 0.25:
            fmt.update(t=mk_subj(rng, test)[0])                          # a tail the text never mentions
        truth = any(c[1] == fmt["h"] and c[3] == fmt["t"] for c in clauses)
        ins, crit = pool(rng, SUP_Q, test)
        ins = ins.format(**fmt)
        crit = {k2: v.format(**fmt) for k2, v in crit.items()} if crit else None
        rows.append(verify(rng, state, ins, truth, task, i, split, crit=crit))
    return rows


# ================================================================== driver

GENS = [(em_case, 0.40), (mention_case, 0.20), (relpair_case, 0.20), (modality_case, 0.20)]
ONLY = {"em": [(em_case, 1.0)], "mention": [(mention_case, 1.0)], "relpair": [(relpair_case, 1.0)], "modality": [(modality_case, 1.0)]}


def label_of(r):
    if r["kind"] == "verify":
        return "yes" if r["gold"] == 0 else "no"
    if r["kind"] == "score":
        return f"L{r['gold']}"
    return r["choices"][r["gold"]].split(":")[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120000)
    ap.add_argument("--split", choices=("train", "test"), default="train")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", choices=list(ONLY), default=None, help="generate a single task")
    ap.add_argument("--exclude", default=None, help="a jsonl whose (state, question) pairs must not be repeated, e.g. the train file")
    a = ap.parse_args()
    gens = ONLY[a.only] if a.only else GENS
    rng = random.Random(f"kg:{a.split}:{a.seed}:{a.only or ''}")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    seen, n = set(), 0
    if a.exclude:
        with open(a.exclude) as fx:
            for line in fx:
                r = json.loads(line)
                seen.add((r["state"], r["question"]))
    stats: dict[str, Counter] = {}
    with out.open("w") as fo:
        i = 0
        while n < a.n:
            gen = rng.choices([g for g, _ in gens], [w for _, w in gens])[0]
            for r in gen(rng, i, a.split):
                key = (r["state"], r["question"])
                if key in seen:
                    continue
                seen.add(key)
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
                stats.setdefault(r["task"], Counter())[f"{r['kind']}:{label_of(r)}"] += 1
                n += 1
            i += 1
    print(f"{n} rows → {out}")
    for t, c in sorted(stats.items()):
        tot = sum(c.values())
        print(f"  {t:9s} {tot:7d} ({tot / n:.0%})  " + "  ".join(f"{k}={v}" for k, v in sorted(c.items())))


if __name__ == "__main__":
    main()
