"""Knowledge-graph judgments over LLM-written documents and support threads, with labels fixed by code.

    uv run python -m teacher.kg_llm --job docs    --calls 2500 --out data/kg/llm-docs-train.jsonl
    uv run python -m teacher.kg_llm --job docs    --calls 150  --split test --out data/kg/llm-docs-test.jsonl
    uv run python -m teacher.kg_llm --job threads --calls 800  --out data/kg/llm-threads-train.jsonl
    uv run python -m teacher.kg_llm --job threads --calls 60   --split test --out data/kg/llm-threads-test.jsonl
    uv run python -m teacher.kg_llm --report data/kg/llm-docs-train.jsonl        (acceptance, agreement, label balance)

Why: decima-agent 2.1 failed as a drop-in for Jev in Lyon's KG notebooks (~/Rande/decima-work/kgx/DECIMA-RESULTS.md).
The gaps are judgments over whole documents: how an edge is presented (asserted / hypothetical / negated /
forward_looking), whether the document bears on it at all ("supported"), which relation joins a pair (with `none`
and direction), what a mention stands for (ontology descriptions as long options), and intent / priority /
resolved over 1-1.5k-token support threads.

How labels stay honest:
- docs    : code invents a small world (fictional companies, people, regulators, products, places, tickers,
            metrics) and 4-8 facts (head, relation, tail, modality). Gemma (rande-fast-local) only writes an
            article around them; every name must appear verbatim or the call is dropped. Then a separate Gemma call
            that sees only the document and unlabeled questions answers status (plus an `absent` option),
            relation and type; a row is kept only when that blind answer equals code's label.
- threads : code picks intent, priority, resolved, channel and subject; Gemma writes the thread; a blind call
            answers intent, resolved and priority; intent / resolved rows are kept on agreement, priority within
            one level (soft label).

Questions are rendered exactly as decima.systemone turns TypeSafe questions into Decima ones (choice options
"key: description", noul "instructions\\nTrue if: …\\nFalse if: …" with yes / no, score levels in order), using
the report's wording plus paraphrases. The kgx corpora are the held-out test: nothing here is taken from them
and their names are blocked. `--split test` uses held-out article styles, channels and the `legal` ontology.
Resumable by call index; a per-call stats sidecar (<out>.stats.jsonl) feeds --report.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from teacher.agent_proc import choice_opts, row, verify  # verify renders nouls with noul_text
from teacher.client import Teacher, TeacherError
from teacher.prompts import loads_lenient

WRITER = CHECKER = "rande-fast-local"          # rande-strong-local is stopped at night; never depend on it

# ------------------------------------------------------------------ names (all invented; kgx corpus names blocked)

BLOCK = {"northwind", "halcyon", "cascade", "meridian", "blackmere", "cobalt", "kestrel", "ironbridge", "redwood", "solent",
         "torrent", "vantage", "aurora", "hlcn", "nwl", "raman", "duarte", "vasquez", "webb", "iyer", "ingersoll", "ferreira",
         "shah", "okonkwo", "priya", "marcus", "elena", "tomas", "penang", "porto", "dresden", "telematics", "microsystems"}
SA = ["Bel", "Cor", "Dar", "El", "Fen", "Gal", "Hal", "Ith", "Jor", "Kel", "Lum", "Mar", "Nor", "Or", "Pel", "Quin", "Ros", "Sol",
      "Tal", "Ul", "Val", "Wex", "Zan", "Ash", "Bry", "Cal", "Dov", "Ev", "Fal", "Gren", "Har", "Lan", "Mer", "Nev", "Os", "Pry", "Rav",
      "Sev", "Tor", "Vey", "Wyn", "Yel", "Brin", "Cael", "Dra", "Est", "Fin", "Gol", "Hel", "Isk", "Kor", "Liv", "Mon", "Ner", "Pax", "Tre"]
SB = ["a", "e", "i", "o", "", "", "u"]
SC = ["ra", "ris", "tex", "vis", "dor", "lan", "mont", "vara", "nix", "ton", "ven", "lis", "tek", "ria", "mar", "dyn", "rex", "lon",
      "via", "gen", "nova", "sys", "thea", "wick", "brook", "dale", "mere", "quist", "ford", "ova", "tric", "lux", "vant", "rion"]
REAL = {"valaris", "solera", "corteva", "veritas", "telus", "zenith", "cortex", "solaris", "halo", "nova", "elan", "orion", "valero", "corvus", "telos", "helix", "avalon", "kellogg", "marlon"}
FIRST = ["Amara", "Bjorn", "Chiara", "Darius", "Esther", "Farid", "Greta", "Hiroshi", "Ingrid", "Jamal", "Keiko", "Lorenzo", "Mireille",
         "Nikolai", "Oluwaseun", "Paloma", "Quentin", "Rashida", "Soren", "Tanvir", "Ursula", "Viktor", "Wanjiru", "Xavier", "Yusuf",
         "Zofia", "Anders", "Beatriz", "Cyrus", "Delphine", "Emeka", "Fatima", "Gideon", "Hana", "Ilya", "Josefina", "Kofi", "Leila",
         "Mateo", "Noor", "Orla", "Pavel", "Rosalind", "Saanvi", "Teodor", "Valentina", "Wen", "Yara", "Bogdan", "Camille", "Dmitri",
         "Elif", "Henrik", "Imani", "Kasimir", "Linnea", "Mehmet", "Nadia", "Rafael", "Sigrid", "Thandiwe", "Arjun", "Margot", "Lucan"]
LAST = ["Achterberg", "Bellweather", "Castellanos", "Dragomir", "Eskildsen", "Fairbanks", "Galloway", "Haverkamp", "Ishikawa",
        "Jablonski", "Kowalczyk", "Lindqvist", "Mbeki", "Nakashima", "Oyelaran", "Pellegrini", "Quarshie", "Rautenbach", "Szabo",
        "Thorvaldsen", "Uchenna", "Valtonen", "Whitcombe", "Yilmaz", "Zielinski", "Abernathy", "Brennecke", "Cavanagh", "Delacroix",
        "Engstrom", "Fontaine", "Grimaldi", "Hollis", "Ibarra", "Kerrigan", "Lachance", "Montoya", "Nwosu", "Okafor", "Petrakis",
        "Rasmussen", "Sorensen", "Tremblay", "Vasiliev", "Wexford", "Adebayo", "Bergstrom", "Chukwu", "Demir", "Halvorsen", "Moreau"]


def pseudo(rng, used: set) -> str:
    for _ in range(300):
        st, en = rng.choice(SA), rng.choice(SC)
        mid = rng.choice(SB) if st[-1] not in "aeiouy" and en[0] not in "aeiou" else ""
        w = (st + mid + en).capitalize()
        if 5 <= len(w) <= 10 and w.lower() not in BLOCK | REAL and not any(w in u or u in w or w[:4] == u[:4] or w[-4:] == u[-4:] for u in used):
            used.add(w)
            return w
    raise ValueError("name space exhausted")


def ticker_of(w: str) -> str:
    cons = [c for c in w[1:].upper() if c not in "AEIOU"]
    t = (w[0].upper() + "".join(cons))[: 3 + (len(cons) > 3)]
    return t if len(t) >= 3 else (w.upper() + "X")[:3]


def initials(name: str) -> str:
    return "".join(p[0] for p in name.split() if p[0].isupper())


# ------------------------------------------------------------------ domains

DOMAINS = {
    "pharma": dict(suffix=["Therapeutics", "Biosciences", "Pharma", "Medical", "Bio"], word=["Health", "Life Sciences", "Diagnostics"],
                   reg=["Medicines Agency", "Health Products Authority", "Drug Review Board"],
                   metric=["product sales", "R&D spending", "net loss", "gross margin", "cash runway"],
                   segment=["Consumer Health", "Specialty Care", "Diagnostics", "Vaccines"],
                   commodity=["active pharmaceutical ingredients", "medical-grade plastics", "cobalt"],
                   sector=["biotechnology", "medical devices", "generic drugs"], risk=["patent expiry", "clinical trial failure", "pricing pressure"],
                   condition=["type 2 diabetes", "psoriasis", "chronic migraine", "atrial fibrillation", "rheumatoid arthritis", "asthma"]),
    "tech": dict(suffix=["Systems", "Software", "Labs", "Technologies", "Networks"], word=["Cloud", "Data", "Digital", "Semiconductor"],
                 reg=["Digital Markets Authority", "Data Protection Office", "Telecom Regulatory Commission"],
                 metric=["annual recurring revenue", "operating margin", "free cash flow", "net revenue retention", "subscription revenue"],
                 segment=["Cloud Services", "Enterprise Software", "Devices", "Payments"],
                 commodity=["polysilicon", "rare-earth magnets", "memory chips"],
                 sector=["cloud software", "consumer electronics", "cybersecurity"], risk=["cyber incidents", "export controls", "customer concentration"]),
    "retail": dict(suffix=["Brands", "Foods", "Retail", "Outfitters", "Home"], word=["Grocers", "Apparel", "Market"],
                   reg=["Consumer Safety Commission", "Food Standards Agency", "Fair Trading Office"],
                   metric=["same-store sales", "gross margin", "e-commerce sales", "inventory turnover", "comparable sales"],
                   segment=["Online", "Wholesale", "Private Label", "Home Goods"],
                   commodity=["cotton", "cocoa", "coffee beans", "palm oil"],
                   sector=["grocery retail", "apparel", "home furnishings"], risk=["supply disruption", "currency volatility", "labour shortages"]),
    "energy": dict(suffix=["Energy", "Resources", "Power", "Industries", "Materials"], word=["Renewables", "Mining", "Grid"],
                   reg=["Energy Regulatory Board", "Environmental Protection Office", "Mining Safety Authority"],
                   metric=["adjusted EBITDA", "production volume", "capital expenditure", "net debt", "realized price per tonne"],
                   segment=["Upstream", "Battery Materials", "Utility Services", "Specialty Metals"],
                   commodity=["lithium", "copper", "natural gas", "nickel", "aluminium"],
                   sector=["battery materials", "renewable power", "industrial metals"], risk=["commodity price swings", "permitting delays", "tariff exposure"]),
    "logistics": dict(suffix=["Logistics", "Shipping", "Rail", "Freight", "Air Cargo"], word=["Express", "Transport", "Ports"],
                      reg=["Transport Safety Board", "Maritime Authority", "Competition Authority"],
                      metric=["revenue per shipment", "operating ratio", "fuel costs", "on-time delivery rate", "volume growth"],
                      segment=["Contract Logistics", "Ocean Forwarding", "Last-Mile Delivery", "Cold Chain"],
                      commodity=["diesel", "jet fuel", "marine fuel"],
                      sector=["contract logistics", "container shipping", "parcel delivery"], risk=["fuel price volatility", "port congestion", "driver shortages"]),
    "finance": dict(suffix=["Bank", "Capital", "Financial", "Insurance", "Asset Management"], word=["Savings", "Trust", "Credit"],
                    reg=["Banking Supervisory Authority", "Securities Commission", "Financial Conduct Office"],
                    metric=["net interest margin", "deposit growth", "CET1 ratio", "loan loss provisions", "assets under management"],
                    segment=["Wealth Management", "Retail Banking", "Commercial Lending", "Insurance"],
                    commodity=["gold"], sector=["retail banking", "asset management", "insurance"],
                    risk=["interest-rate risk", "credit losses", "deposit outflows"]),
}


DOMLABEL = {"pharma": "a pharmaceutical company", "tech": "a technology company", "retail": "a consumer-goods or retail company",
            "energy": "an energy or materials company", "logistics": "a logistics company", "finance": "a financial-services company"}


def product_name(rng, dom: str, used: set) -> tuple[str, str, str]:
    """(name, short gloss for the writer, product subtype)"""
    w = pseudo(rng, used)
    if dom == "pharma":
        if rng.random() < 0.6:
            return w.lower().rstrip("aeiou").capitalize() + rng.choice(["mab", "tinib", "vir", "zole", "pril", "stat", "dronate"]), "a drug", "drug"
        return f"{w} {rng.choice(['Pulse', 'Flow', 'Sense', 'Guard'])} {rng.randint(2, 9)}", rng.choice(["an insulin pump", "a cardiac monitor", "a glucose sensor", "a surgical stapler"]), "device"
    gloss = {"tech": ["a cloud platform", "a chip", "a payments app", "a security suite", "a laptop line"],
             "retail": ["a sneaker line", "a kitchen blender", "a snack brand", "a skincare range", "an e-bike"],
             "energy": ["a battery cell", "a wind turbine model", "a home solar kit", "a hydrogen electrolyser"],
             "logistics": ["a freight-tracking platform", "a delivery van", "a same-day delivery service", "a cargo drone"],
             "finance": ["a credit card", "a robo-advisor app", "a mortgage product", "a small-business loan product"]}[dom]
    tail = rng.choice(["", " Pro", " X", f" {rng.randint(2, 900)}", " One", " Max", " Edge", " Go"])
    return w + tail, rng.choice(gloss), "product"


def place_name(rng, used: set) -> str:
    w = pseudo(rng, used)
    return rng.choice([w, w, f"Port {w}", f"{w} Bay", f"{w} Province", f"{w} Valley", f"North {w}", f"{w}shire"])


# ------------------------------------------------------------------ relations (head kind, tail kind, description, symmetric)

REL = {
    "subsidiary_of": ("company", "company", "The head company is owned or controlled by the tail company.", False),
    "acquires": ("company", "company", "The head company is buying or has bought the tail.", False),
    "has_stake_in": ("company", "company", "The head holds an ownership interest in the tail.", False),
    "supplies": ("company", "company", "The head provides goods or inputs to the tail company.", False),
    "partners_with": ("company", "company", "A stated partnership, alliance, or joint venture.", True),
    "competes_with": ("company", "company", "The two companies are stated rivals in a market.", True),
    "ceo_of": ("person", "company", "The person is the chief executive of the company.", False),
    "works_for": ("person", "company", "The person is employed by the company or holds a role there other than chief executive.", False),
    "founded": ("person", "company", "The person started or co-founded the company.", False),
    "investigated_by": ("company", "regulator", "The regulator is examining, probing or reviewing the company's conduct.", False),
    "fined_by": ("company", "regulator", "The regulator imposes or has imposed a monetary penalty on the company.", False),
    "launches": ("company", "product", "The company brings or has brought the product to market.", False),
    "recalls": ("company", "product", "The company withdraws or has withdrawn the product from sale over a defect or safety problem.", False),
    "operates_in": ("company", "place", "The company does business in this place.", False),
    "exits": ("company", "place", "The company withdraws or has withdrawn from doing business in this place.", False),
    "reports": ("company", "metric", "The company has disclosed a figure for this measure.", False),
}
ALIAS = {"ceo_of": "chief_executive_of", "works_for": "employed_by", "investigated_by": "under_review_by", "fined_by": "penalized_by",
         "launches": "introduces", "operates_in": "active_in", "reports": "discloses", "acquires": "buys", "has_stake_in": "owns_stake_in",
         "supplies": "supplier_to", "partners_with": "allied_with", "competes_with": "rival_of", "subsidiary_of": "owned_by",
         "founded": "founder_of", "recalls": "withdraws", "exits": "leaves"}
PAIR_RELS = defaultdict(list)
for _r, (_h, _t, _d, _s) in REL.items():
    PAIR_RELS[(_h, _t)].append(_r)
NONE_DESC = "The document does not relate these two things in any of the ways listed, or relates them only in the opposite direction."
NONE_ALT = "None of these relationships is stated or discussed from the first to the second (including when it only runs the other way)."

MODS = ["asserted", "hypothetical", "negated", "forward_looking"]
STATUS_CRIT = {
    "asserted": "The text states this relationship as a present or past fact. It holds, or it held.",
    "hypothetical": "The text raises this relationship only as a possibility, a condition, a proposal awaiting approval, or something that would happen if something else did. It is not claimed to have happened.",
    "negated": "The text denies this relationship, or states that it will not happen, is not planned, or is not expected.",
    "forward_looking": "The text projects, forecasts or guides that this relationship will hold in the future. It has not happened yet, but it is not merely conditional.",
}
STATUS_ALT = {
    "fact": "Stated as true now or in the past.",
    "possible": "Only raised as a possibility, a condition, a rumour or a proposal that still needs approval; not said to have happened.",
    "denied": "Denied, ruled out, or said not to be planned or not expected.",
    "future": "Presented as a plan, forecast or guidance for the future, with no condition attached.",
}
REL_PHRASE = {
    "subsidiary_of": "{h} being owned or controlled by {t} (a subsidiary or unit of {t})",
    "acquires": "{h} buying {t}",
    "has_stake_in": "{h} holding an ownership stake in {t}",
    "supplies": "{h} supplying goods, parts or inputs to {t}",
    "partners_with": "{h} and {t} working together in a partnership, alliance or joint venture",
    "competes_with": "{h} and {t} being rivals in the same market",
    "ceo_of": "{h} being the chief executive of {t}",
    "works_for": "{h} working for {t} in a role other than chief executive (name the role)",
    "founded": "{h} having founded or co-founded {t}",
    "investigated_by": "{t} investigating or reviewing the conduct of {h}",
    "fined_by": "{t} fining {h}",
    "launches": "{h} launching {t}",
    "recalls": "{h} recalling {t} over a defect or safety problem",
    "operates_in": "{h} doing business in {t}",
    "exits": "{h} pulling out of {t}",
    "reports": "{h} disclosing a figure for its {t}",
}
WRITE_MOD = {
    "asserted": ("AS A FACT: it holds now or it happened (it may be attributed, e.g. the company said it had …)",
                 []),
    "hypothetical": ("ONLY AS A POSSIBILITY: a rumour being weighed, a condition, or a proposal still awaiting approval; never say it happened or definitely will",
                     ["could", "might", "is weighing", "is exploring", "would … if …", "has proposed, subject to approval", "people familiar with the talks said a deal may", "is considering", "should regulators sign off"]),
    "negated": ("AS DENIED: it is not the case, did not happen, will not happen, is not planned or is not expected",
                ["denied", "has no plans to", "ruled out", "does not", "never", "is not expected to", "rejected reports that", "dismissed speculation that", "declined to"]),
    "forward_looking": ("AS AN UNCONDITIONAL PLAN OR FORECAST: it will happen or hold in the future; it has not happened yet; attach no condition or approval",
                        ["expects to … by next year", "will begin … in 2027", "is scheduled to", "guided that … next quarter", "plans to … starting in the spring", "is set to", "intends to … later this year", "forecast that"]),
}
ABSENT = "The document does not discuss this relationship between these two things at all."

STATUS_Q = [
    "How does the document present this claim: “{h}” — {r} → “{t}”? Here, “{r}” means: {d} Judge only how it is presented, not whether it is plausible.",
    "How does the document present this claim: “{h}” — {r} → “{t}”? Here, “{r}” means: {d} Judge only how it is presented, not whether it is plausible.",
    "In `document`, is the relationship “{h}” — {r} → “{t}” stated as a fact, raised as a possibility, denied, or projected into the future? “{r}” means: {d}",
    "Consider the edge “{h}” —{r}→ “{t}” ({r}: {d}) How does `document` present it? Judge the wording, not whether it is likely.",
    "What assertion status does `document` give to the claim that “{h}” {rs} “{t}”? (“{r}”: {d})",
]
SUPPORTED_Q = [
    "Does the document say anything that bears on this claim: “{h}” — {r} → “{t}”? Here, “{r}” means: {d} Answer yes if the document discusses this relationship at all, in any modality — asserted, denied, proposed or merely possible. Answer no if the document does not connect these two things.",
    "Does the document say anything that bears on this claim: “{h}” — {r} → “{t}”? Here, “{r}” means: {d} Answer yes if the document discusses this relationship at all, in any modality — asserted, denied, proposed or merely possible. Answer no if the document does not connect these two things.",
    "Is the edge “{h}” —{r}→ “{t}” discussed anywhere in `document` — stated, denied, forecast or only floated? (“{r}”: {d}) Answer no if the document never connects these two.",
    "Does `document` connect “{h}” and “{t}” through “{r}” ({d}) in any way, including as a denial, a plan or a possibility?",
]
SUPPORTED_CRIT = [{"true": "The document addresses this relationship between these two specific things.",
                   "false": "The document does not connect these two things, or one of them does not appear in it."},
                  {"true": "The text discusses this relationship between these two, in any modality.",
                   "false": "The text never relates these two in this way, or one of them is missing."}]
RELATION_Q = [
    "In `document`, which of these relationships does the text state or discuss from “{h}” to “{t}”, in that direction? Pick the one relationship the text supports, or none.",
    "In `document`, which of these relationships does the text state or discuss from “{h}” to “{t}”, in that direction? Pick the one relationship the text supports, or none.",
    "Which relationship from “{h}” to “{t}” (reading in that direction) does `document` state or discuss — as a fact, a denial, a plan or a possibility? Choose none if it relates them in none of these ways or only the other way round.",
    "Read `document`. How is “{h}” related to “{t}”, with “{h}” as the head? Pick one relationship, or none.",
]
TYPE_Q = [
    "In `document`, what kind of real-world thing does the mention “{m}” stand for? Judge the thing it refers to in this document, not the form of the expression: a name, nickname, abbreviation or ticker symbol used to talk about something stands for that thing.",
    "In `document`, what kind of real-world thing does the mention “{m}” stand for? Judge the thing it refers to in this document, not the form of the expression: a name, nickname, abbreviation or ticker symbol used to talk about something stands for that thing.",
    "What type of entity is “{m}” in `document`? Classify it by what it refers to, not by its surface form: a ticker, short name or abbreviation takes the type of the thing it names.",
    "Classify the mention “{m}” in `document` with the ontology below. Judge what it stands for in this document.",
]

# ------------------------------------------------------------------ ontologies for typing (kind → type)

BUSINESS = {
    "company": "A commercial issuer or private firm, referred to by name.",
    "person": "An executive, director, founder, analyst, or named official.",
    "regulator": "A government body, agency, central bank, or court.",
    "product": "A named product, service, or platform.",
    "business_segment": "A reportable segment or division of a company.",
    "geography": "A country, region, state, city, or named market.",
    "sector": "An industry classification, e.g. semiconductors, freight, retail banking.",
    "security": "An issued instrument: shares, notes, bonds, or a ticker symbol.",
    "financial_metric": "A named measure: revenue, operating margin, EPS, free cash flow.",
    "risk_factor": "A disclosed risk or exposure, e.g. tariff exposure, supply disruption.",
    "litigation": "A named case, investigation, or enforcement action.",
    "business_event": "The newsworthy corporate event itself: acquisition, layoff, recall, guidance cut.",
    "commodity": "A raw material or physical input: diesel, silicon, copper.",
}
ONTOLOGIES = {   # name → (domains it fits, {type: description}, {kind: type})
    "business": (None, BUSINESS, {"company": "company", "ticker": "company", "short": "company", "person": "person", "regulator": "regulator",
                                  "reg_abbr": "regulator", "product": "product", "drug": "product", "device": "product", "place": "geography",
                                  "metric": "financial_metric", "segment": "business_segment", "commodity": "commodity", "security": "security",
                                  "case": "litigation", "sector": "sector", "risk": "risk_factor"}),
    "medical": (("pharma",), {
        "organization": "A company, trial sponsor, hospital group or other institution, referred to by name.",
        "person": "A named individual: an executive, investigator, clinician or official.",
        "health_authority": "A government agency that approves, inspects or regulates medicines and medical devices.",
        "drug": "A named medicine, therapy or drug candidate.",
        "medical_device": "A named instrument, implant, monitor or other piece of medical equipment.",
        "condition": "A disease, disorder or symptom that a therapy is meant to treat.",
        "location": "A country, region or city.",
        "financial_measure": "A named financial or operating figure, such as product sales or R&D spending.",
        "risk": "A threat to the business, such as patent expiry or a failed trial.",
    }, {"company": "organization", "ticker": "organization", "short": "organization", "person": "person", "regulator": "health_authority",
        "reg_abbr": "health_authority", "drug": "drug", "device": "medical_device", "condition": "condition", "place": "location",
        "metric": "financial_measure", "risk": "risk"}),
    "software": (("tech",), {
        "organization": "A company or other organization, referred to by name or by a short form of its name.",
        "person": "A named individual, such as an engineer, executive or analyst.",
        "software_or_device": "A named application, platform, cloud service, chip or device.",
        "authority": "A government agency, regulator or court that oversees technology companies.",
        "region": "A country, region or city where something is based or sold.",
        "kpi": "A business measure such as annual recurring revenue, churn or operating margin.",
        "business_unit": "A division or segment inside a company.",
        "component": "A physical input or part bought in bulk, such as memory chips or polysilicon.",
    }, {"company": "organization", "ticker": "organization", "short": "organization", "person": "person", "product": "software_or_device",
        "regulator": "authority", "reg_abbr": "authority", "place": "region", "metric": "kpi", "segment": "business_unit", "commodity": "component"}),
    "retail": (("retail",), {
        "brand_owner": "A company that makes, sells or distributes consumer goods, referred to by name or a short form of it.",
        "person": "A named individual: executive, founder, buyer or analyst.",
        "consumer_product": "A named product, product line or private-label range sold to shoppers.",
        "raw_material": "An ingredient or material goods are made from: cotton, cocoa, palm oil.",
        "market": "A country, region or city where goods are sold.",
        "watchdog": "A public body that sets or enforces consumer, food or product-safety rules.",
        "trading_metric": "A measure of trading performance: same-store sales, revenue, gross margin.",
        "channel_or_division": "A sales channel or division of the business, such as Online or Wholesale.",
    }, {"company": "brand_owner", "ticker": "brand_owner", "short": "brand_owner", "person": "person", "product": "consumer_product",
        "commodity": "raw_material", "place": "market", "regulator": "watchdog", "reg_abbr": "watchdog", "metric": "trading_metric",
        "segment": "channel_or_division"}),
    "industrial": (("energy", "logistics"), {
        "operator": "A company that owns, runs or builds energy, industrial or transport assets, referred to by name or a short form of it.",
        "person": "A named individual: executive, engineer, analyst or official.",
        "government_body": "A ministry, regulator, safety board or other public authority.",
        "equipment_or_service": "A named product, machine, vehicle, platform or service offering.",
        "input_material": "A fuel, mineral or raw material: lithium, diesel, copper.",
        "site_or_region": "A country, region, city, port or other named place.",
        "performance_figure": "A reported operating or financial number: EBITDA, volumes, capital expenditure.",
        "hazard": "A risk the business is exposed to, such as fuel price swings or permitting delays.",
    }, {"company": "operator", "ticker": "operator", "short": "operator", "person": "person", "regulator": "government_body",
        "reg_abbr": "government_body", "product": "equipment_or_service", "commodity": "input_material", "place": "site_or_region",
        "metric": "performance_figure", "risk": "hazard"}),
    "legal": (None, {     # held out: test split only
        "party": "A company or organization involved in the matter, referred to by name or a short form of its name.",
        "individual": "A named person: an executive, lawyer, judge, official or witness.",
        "authority": "A court, regulator, prosecutor or government agency.",
        "proceeding": "A named case, investigation, inquiry or enforcement action.",
        "subject_matter": "The product, service or asset that the matter is about.",
        "jurisdiction": "A country, state, region or city.",
        "instrument": "A security or contract: shares, notes, bonds, a licence.",
    }, {"company": "party", "ticker": "party", "short": "party", "person": "individual", "regulator": "authority", "reg_abbr": "authority",
        "case": "proceeding", "product": "subject_matter", "drug": "subject_matter", "device": "subject_matter", "place": "jurisdiction",
        "security": "instrument"}),
}
STYLES = ["wire-service news story", "sell-side analyst note", "company press release", "quarterly earnings recap",
          "industry blog post", "trade-journal feature", "investor newsletter item", "regional business-section article",
          "morning markets briefing paragraph set", "summary of a conference call"]
STYLES_TEST = ["market-close roundup", "investigative feature excerpt", "question-and-answer interview with an analyst"]


# ------------------------------------------------------------------ docs: world, prompt, checks

def make_world(rng, split: str) -> dict:
    used: set = set()
    dom = rng.choice(list(DOMAINS))
    D = DOMAINS[dom]
    ents = []                          # dicts: id, kind, name, gloss, (sub)

    def add(kind, name, gloss="", **kw):
        e = {"id": len(ents), "kind": kind, "name": name, "gloss": gloss, **kw}
        ents.append(e)
        return e

    n_core = rng.randint(4, 6)
    n_comp = rng.randint(2, min(4, n_core - 1))
    for _ in range(n_comp):
        w = pseudo(rng, used)
        u = rng.random()
        name = f"{w} {rng.choice(D['word'])} {rng.choice(D['suffix'])}" if u < 0.3 else f"{w} {rng.choice(D['suffix'])}" if u < 0.85 else w
        add("company", name, DOMLABEL[dom], short=w if name != w else None)
    others = []
    for _ in range(n_core - n_comp):
        others.append(rng.choices(["person", "regulator", "product", "place", "metric"], [3, 2, 2, 1.5, 1.2])[0])
    if others.count("regulator") > 1:
        others = [o for o in others if o != "regulator"] + ["regulator"]
    if others.count("metric") > 1:
        others = [o for o in others if o != "metric"] + ["metric"]
    for k in others:
        if k == "person":
            while True:
                nm = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
                if nm not in {e['name'] for e in ents}:
                    break
            add("person", nm, "a person; give them no job, title or tie to any listed company beyond what the facts say — e.g. call them an industry veteran or an independent analyst if needed")
        elif k == "regulator":
            add("regulator", f"{pseudo(rng, used)} {rng.choice(D['reg'])}", "a government regulator")
        elif k == "product":
            nm, gloss, sub = product_name(rng, dom, used)
            add("product", nm, gloss, sub=sub)
        elif k == "place":
            add("place", place_name(rng, used), "a place (country, region or city)")
        else:
            add("metric", rng.choice(D["metric"]), "a financial measure")
    # facts: one per unordered pair, legal type pairs only
    pairs = []
    for a in ents:
        for b in ents:
            if a is not b and PAIR_RELS.get((a["kind"], b["kind"])):
                if a["kind"] == b["kind"] and a["id"] > b["id"]:
                    continue
                pairs.append((a, b))
    rng.shuffle(pairs)
    want = rng.randint(4, 8)
    facts, used_pairs, ceo, sub_head, prod_used = [], set(), set(), set(), set()
    for a, b in pairs:
        if len(facts) >= want:
            break
        if a["kind"] == b["kind"] == "company" and rng.random() < 0.5:
            a, b = b, a
        rels = list(PAIR_RELS[(a["kind"], b["kind"])])
        if a["kind"] == "person":
            if a["id"] in ceo or any(f["h"] == a["id"] for f in facts):
                continue                                    # one role per person keeps the world consistent
        if b["kind"] == "product" and b["id"] in prod_used:
            continue
        rel = rng.choice(rels)
        if rel == "ceo_of" and b["id"] in ceo:
            rel = rng.choice(["works_for", "founded"])
        if rel == "subsidiary_of" and (a["id"] in sub_head or any(f["r"] == "subsidiary_of" and f["h"] == b["id"] for f in facts)):
            rel = rng.choice(["supplies", "partners_with", "competes_with"])
        mod = rng.choices(MODS, [0.34, 0.22, 0.22, 0.22])[0]
        facts.append({"h": a["id"], "r": rel, "t": b["id"], "m": mod})
        used_pairs.add(frozenset((a["id"], b["id"])))
        if rel == "ceo_of":
            ceo |= {a["id"], b["id"]}
        if rel == "subsidiary_of":
            sub_head.add(a["id"])
        if b["kind"] == "product":
            prod_used.add(b["id"])
    for a, b in pairs:                                      # every entity takes part in at least one fact
        if len(facts) >= 8:
            break
        if frozenset((a["id"], b["id"])) in used_pairs or all(x["id"] in {f["h"] for f in facts} | {f["t"] for f in facts} for x in (a, b)):
            continue
        if a["kind"] == "person" and any(f["h"] == a["id"] for f in facts) or b["kind"] == "product" and b["id"] in prod_used:
            continue
        rel = rng.choice([r_ for r_ in PAIR_RELS[(a["kind"], b["kind"])] if r_ not in ("ceo_of", "subsidiary_of")])
        facts.append({"h": a["id"], "r": rel, "t": b["id"], "m": rng.choices(MODS, [0.34, 0.22, 0.22, 0.22])[0]})
        used_pairs.add(frozenset((a["id"], b["id"])))
        if b["kind"] == "product":
            prod_used.add(b["id"])
    inv = {f["h"] for f in facts} | {f["t"] for f in facts}
    remap = {e["id"]: j for j, e in enumerate(x for x in ents if x["id"] in inv)}
    ents = [e | {"id": remap[e["id"]]} for e in ents if e["id"] in inv]
    facts = [f | {"h": remap[f["h"]], "t": remap[f["t"]]} for f in facts]
    if len(facts) < 3 or len(ents) < 3:
        raise ValueError("too few facts")
    ms = [f["m"] for f in facts]
    if "asserted" not in ms:
        rng.choice(facts)["m"] = "asserted"                # every trap has a factual control in the same document
    if all(f["m"] == "asserted" for f in facts):
        rng.choice(facts)["m"] = rng.choice(MODS[1:])
    # aliases and decorations (typing targets)
    aliases = []
    comps = [e for e in ents if e["kind"] == "company"]
    if rng.random() < 0.55:
        c = rng.choice(comps)
        aliases.append({"kind": "ticker", "name": ticker_of(c["short"] or c["name"].split()[0]), "of": c["id"]})
    shorts = [c for c in comps if c.get("short")]
    if shorts and rng.random() < 0.5:
        c = rng.choice(shorts)
        aliases.append({"kind": "short", "name": c["short"], "of": c["id"]})
    regs = [e for e in ents if e["kind"] == "regulator"]
    if regs and rng.random() < 0.6:
        aliases.append({"kind": "reg_abbr", "name": initials(regs[0]["name"]), "of": regs[0]["id"]})
    decos = []
    for k in rng.sample(["segment", "commodity", "sector", "risk", "security", "case", "condition"], rng.randint(0, 2)):
        c = rng.choice(comps)
        if k == "segment":
            decos.append({"kind": k, "name": rng.choice(D["segment"]), "note": f"a business segment of {c['name']}"})
        elif k in ("commodity", "sector", "risk"):
            decos.append({"kind": k, "name": rng.choice(D[k]), "note": {"commodity": f"a raw material or input that matters to {c['name']}",
                                                                          "sector": f"the industry {c['name']} belongs to",
                                                                          "risk": f"a risk {c['name']} discloses"}[k]})
        elif k == "security":
            nm = rng.choice([f"{rng.randint(2028, 2036)} senior notes", "Class B shares", f"convertible bonds due {rng.randint(2028, 2035)}",
                             f"{rng.choice(['4.25', '5.5', '6.125', '3.75'])}% notes due {rng.randint(2028, 2036)}"])
            decos.append({"kind": k, "name": nm, "note": f"securities issued by {c['name']}; e.g. mention their price or yield"})
        elif k == "case":
            decos.append({"kind": k, "name": f"{rng.choice(LAST)} v. {c['short'] or c['name']}", "note": f"a lawsuit against {c['name']}, mentioned in passing"})
        elif k == "condition" and dom == "pharma":
            decos.append({"kind": k, "name": rng.choice(D["condition"]), "note": "a disease one of the products is meant to treat"})
    for e in ents:
        if e["kind"] == "product":
            e["tkind"] = e.get("sub", "product")
    cues = [rng.sample(WRITE_MOD[f["m"]][1], min(2, len(WRITE_MOD[f["m"]][1]))) for f in facts]
    return {"domain": dom, "ents": ents, "facts": facts, "aliases": aliases, "decos": decos, "cues": cues,
            "style": rng.choice(STYLES_TEST if split == "test" else STYLES), "words": rng.choice([(150, 250), (220, 330), (300, 450)])}


def writer_prompt(w: dict) -> str:
    E = w["ents"]
    lines = [f"- {e['name']} ({e['gloss']})" for e in E]
    for a in w["aliases"]:
        of = E[a["of"]]["name"]
        lines.append({"ticker": f"- {a['name']} — the stock ticker of {of}. Use \"{a['name']}\" at least once as a stand-in name for the company itself, as the subject of a sentence about something the company did, decided or said (e.g. \"{a['name']} said on Monday…\", \"{a['name']}'s board approved…\"). Do not use the ticker for share-price moves.",
                      "short": f"- {a['name']} — a short form of {of}. After first mentioning the full name, also call the company \"{a['name']}\" at least once.",
                      "reg_abbr": f"- {a['name']} — the abbreviation of {of}. Introduce it (\"{of} ({a['name']})\") and use \"{a['name']}\" later at least once."}[a["kind"]])
    for d in w["decos"]:
        lines.append(f"- {d['name']} [for you only, do not copy this note: {d['note']}]. Mention it in passing, exactly as written, in your own words.")
    facts = []
    for k, f in enumerate(w["facts"], 1):
        h, t = E[f["h"]]["name"], E[f["t"]]["name"]
        how, cues = WRITE_MOD[f["m"]]
        cue = ", ".join(f"'{c}'" for c in w["cues"][k - 1])
        facts.append(f"{k}. {REL_PHRASE[f['r']].format(h=h, t=t)} — write this {how}." + (f" Possible wording, only if it reads naturally: {cue}." if cue else ""))
    lo, hi = w["words"]
    return (f"Write a realistic {w['style']} ({lo}-{hi} words) about a fictional {w['domain']} business story. Everything is invented.\n\n"
            "Names (use each one EXACTLY as written, same spelling and capitalisation, at least once):\n" + "\n".join(lines) +
            "\n\nThe story must contain these facts, each expressed exactly once, in the stated way, and never also in any other way "
            "(for example, a fact to be denied must not also be stated as true elsewhere):\n" + "\n".join(facts) +
            "\n\nRules:\n- Do not state, deny or speculate about any other relationship between the listed names beyond the facts above "
            "(e.g. do not call two listed companies rivals or partners unless a fact says so). You may add generic context, numbers, dates, "
            "quotes and unnamed sources, but no other named companies.\n- Weave the facts into natural prose in any order; do not list them "
            "and do not label them (never write words like 'hypothetical', 'asserted', 'negated' or 'forward-looking').\n"
            "- Each fact appears once: do not restate it elsewhere in another form (e.g. no extra forecast of a figure already reported).\n"
            "- Write like a real journalist or analyst: plain, idiomatic English, varied sentence structure, concrete numbers, dates and "
            "quotes. Avoid filler adverbs such as 'currently' or 'already' and the stiff phrase 'it is not the case that'.\n\n"
            'Output JSON only: {"document": "<the full text, with a headline line first if the format has one>"}')


def appears(doc: str, name: str) -> bool:
    return re.search(r"(?<![\w-])" + re.escape(name) + r"(?![\w-])", doc) is not None


def alias_used(doc: str, alias: str, full: str) -> bool:
    """The short form occurs on its own, not only inside the full name."""
    stripped = doc.replace(full, " ")
    return appears(stripped, alias)


def claim_text(rng, w, f, use_alias: bool) -> tuple[str, str]:
    E = w["ents"]

    def nm(i):
        if use_alias:
            al = [a for a in w["aliases"] if a["of"] == i]
            if al and rng.random() < 0.5:
                return rng.choice(al)["name"]
        return E[i]["name"]
    return nm(f["h"]), nm(f["t"])


def opts_block(crit: dict) -> str:
    return "\n".join(f"- {k}: {v}" for k, v in crit.items())


async def blind(tc, doc_or_state: str, task: str, items: list[str], options: str | None, keys_hint: str) -> list[str | None]:
    """One unlabeled call: the checker sees the document and the questions, never code's labels."""
    prompt = (f"Read this document carefully.\n\nDocument:\n\"\"\"\n{doc_or_state}\n\"\"\"\n\n{task}\n"
              + (f"\nOptions:\n{options}\n" if options else "") + "\nItems:\n" + "\n".join(f"{k + 1}. {s}" for k, s in enumerate(items))
              + f'\n\nAnswer every item, in order. Output JSON only: {{"answers": [{keys_hint}, ...]}} with exactly {len(items)} answers.')
    r = await tc.chat([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=60 + 25 * len(items), thinking=False, json_mode=True, tag="kg-check")
    ans = loads_lenient(r.text).get("answers", [])
    if not isinstance(ans, list) or len(ans) != len(items):
        return [None] * len(items)
    return [str(a).strip().strip("\"'`").split(":")[0].strip() if a is not None else None for a in ans]


def soft(n, gold, p=0.9):
    return [p if j == gold else (1 - p) / (n - 1) for j in range(n)]


async def job_docs(tw, tc, rng, i, split):
    w = make_world(rng, split)
    E = w["ents"]
    r = await tw.chat([{"role": "user", "content": writer_prompt(w)}], temperature=0.9, max_tokens=1400, thinking=False, json_mode=True, tag="kg-docs")
    doc = loads_lenient(r.text).get("document", "")
    st = {"names_ok": False, "words": 0}
    if not isinstance(doc, str) or not doc.strip():
        return [], st
    doc = doc.strip()
    st["words"] = len(doc.split())
    names = [e["name"] for e in E] + [d["name"] for d in w["decos"]]
    ok = all(appears(doc, n) for n in names)
    w["aliases"] = [a for a in w["aliases"] if alias_used(doc, a["name"], E[a["of"]]["name"])]   # an unused alias is dropped, not the doc
    bad = re.search(r"\b(hypothetical|asserted|negated|forward[-_ ]looking|modality)\b|→|\b[a-z]+_[a-z_]+\b", doc, re.I) or any(
        re.search(rf"\b{b}\b", doc, re.I) for b in BLOCK)
    st["names_ok"] = bool(ok) and not bad and 100 <= st["words"] <= 600
    if not st["names_ok"]:
        st["missing"] = [n for n in names if not appears(doc, n)]
        st["bad"] = bad.group(0) if hasattr(bad, "group") else bool(bad)
    if not st["names_ok"]:
        return [], st
    state = {"document": doc}
    alias = rng.random() < 0.3
    rname = (lambda x: ALIAS[x]) if alias else (lambda x: x)
    fact_pair = {frozenset((f["h"], f["t"])) for f in w["facts"]}

    # -- candidate pairs: facts, reversed company pairs, unconnected legal pairs
    unconnected = []
    for a in E:
        for b in E:
            if a is not b and PAIR_RELS.get((a["kind"], b["kind"])) and frozenset((a["id"], b["id"])) not in fact_pair:
                if a["kind"] == b["kind"] and a["id"] > b["id"]:
                    continue
                unconnected.append((a["id"], b["id"]))
    rng.shuffle(unconnected)
    unconnected = unconnected[:5]
    negs = []                                               # (h, rel, t) claims for supported=no
    for h, t in unconnected[: rng.randint(2, 4)]:
        negs.append({"h": h, "r": rng.choice(PAIR_RELS[(E[h]["kind"], E[t]["kind"])]), "t": t, "m": None})

    # -- blind A: status (+ absent) for facts and negatives, shuffled together
    claims = [(f, *claim_text(rng, w, f, rng.random() < 0.25)) for f in w["facts"]] + [(f, E[f["h"]]["name"], E[f["t"]]["name"]) for f in negs]
    order = list(range(len(claims)))
    rng.shuffle(order)
    crit_a = {**STATUS_CRIT, "absent": ABSENT}
    items = [f"“{claims[k][1]}” — {rname(claims[k][0]['r'])} → “{claims[k][2]}”  ({rname(claims[k][0]['r'])}: {REL[claims[k][0]['r']][2]})" for k in order]
    ans = await blind(tc, doc, "For each claim below, how does the document present the relationship between these two specific things? "
                      "Judge only how it is presented, not whether it is plausible.", items, opts_block(crit_a), '"<option>"')
    blind_status = {order[j]: ans[j] for j in range(len(order))}

    # -- blind B: relation for fact pairs, reversed company pairs and unconnected pairs
    rel_items = []                                          # (h, t, label, kind)
    for f in w["facts"]:
        rel_items.append((f["h"], f["t"], f["r"], "fact:" + f["m"]))
        if E[f["h"]]["kind"] == E[f["t"]]["kind"] == "company":
            rev = next((g["r"] for g in w["facts"] if g["h"] == f["t"] and g["t"] == f["h"]), None)
            rel_items.append((f["t"], f["h"], f["r"] if REL[f["r"]][3] else rev or "none", "reverse"))
    for h, t in unconnected:
        rel_items.append((h, t, "none", "unconnected"))
    rng.shuffle(rel_items)
    descs = [f"from “{E[h]['name']}” to “{E[t]['name']}”, in that direction. Options: " +
             "; ".join(f"{rname(x)} ({REL[x][2]})" for x in PAIR_RELS[(E[h]['kind'], E[t]['kind'])]) + f"; none ({NONE_DESC})"
             for h, t, _, _ in rel_items]
    rel_ans = await blind(tc, doc, "For each pair below, which relationship does the text state or discuss from the first thing to the second, "
                          "in that direction (as a fact, a denial, a plan or a possibility)? Pick one of that item's options, or none.",
                          descs, None, '"<relation or none>"')
    inv = {rname(x): x for x in REL} | {"none": "none"}
    rel_blind = [inv.get(a) if a else None for a in rel_ans]

    # -- blind C: types for 3-6 mentions under one ontology
    if split == "test" and rng.random() < 0.5:
        oname = "legal"
    else:
        fits = [o for o, (doms, _, _) in ONTOLOGIES.items() if o != "legal" and doms and w["domain"] in doms]
        oname = "business" if (not fits or rng.random() < 0.5) else fits[0]
    _, ocrit, omap = ONTOLOGIES[oname]
    pool = []
    for a in w["aliases"]:
        pool.append((a["name"], a["kind"], 2))
    for d in w["decos"]:
        pool.append((d["name"], d["kind"], 2))
    for e in E:
        pool.append((e["name"], e.get("tkind", e["kind"]), 1))
    pool = [p for p in pool if p[1] in omap]
    rng.shuffle(pool)
    pool.sort(key=lambda p: -p[2])
    mentions = pool[: rng.randint(3, 6)]
    rng.shuffle(mentions)
    okeys = list(ocrit)
    if rng.random() < 0.5 and oname != "business" or rng.random() < 0.3:
        rng.shuffle(okeys)
    ocrit = {k: ocrit[k] for k in okeys}
    type_ans = await blind(tc, doc, "For each mention below, what kind of real-world thing does it stand for in this document? Judge the thing it "
                           "refers to, not the form of the expression: a name, nickname, abbreviation or ticker symbol used to talk about something "
                           "stands for that thing.", [f"“{m[0]}”" for m in mentions], opts_block(ocrit), '"<type>"') if mentions else []

    # -- rows
    rows, k = [], 0
    st.update(status=[], neg=[], rel=[], type=[], onto=oname)

    def add(r_):
        nonlocal k
        rows.append(r_ | {"source": "kg-llm"})
        k += 1
    for idx, (f, h, t) in enumerate(claims):
        b = blind_status.get(idx)
        rel = rname(f["r"])
        fmt = dict(h=h, t=t, r=rel, d=REL[f["r"]][2], rs=rel.replace("_", " "))
        if f["m"] is None:                                     # negative claim
            agree_rel = all(rb == "none" or frozenset((hh, tt)) != frozenset((f["h"], f["t"]))
                            for (hh, tt, _, _), rb in zip(rel_items, rel_blind))
            st["neg"].append([b, agree_rel])
            if b == "absent" and agree_rel:
                q = rng.choice(SUPPORTED_Q).format(**fmt)
                u = rng.random()
                add(verify(state, q, False, "kg-supported", i * 100 + k, split, crit=None if u < 0.1 else SUPPORTED_CRIT[u < 0.3], p=0.95))
            continue
        st["status"].append([f["m"], b])
        if b != f["m"]:
            continue
        if rng.random() < 0.75:
            crit, gold = STATUS_CRIT, MODS.index(f["m"])
        else:
            crit, gold = STATUS_ALT, MODS.index(f["m"])
        add(row("choose", state, rng.choice(STATUS_Q).format(**fmt), choice_opts(crit), soft(4, gold, 0.91), "kg-status", i * 100 + k, split))
        u = rng.random()
        add(verify(state, rng.choice(SUPPORTED_Q).format(**fmt), True, "kg-supported", i * 100 + k, split,
                   crit=None if u < 0.1 else SUPPORTED_CRIT[u < 0.3], p=0.95))
    status_ok = {(f["h"], f["t"]): blind_status.get(j) == f["m"] for j, f in enumerate(w["facts"])}
    for (h, t, lab, kind), b in zip(rel_items, rel_blind):
        st["rel"].append([lab, b, kind])
        # A denied or merely possible relation is still "discussed" (the report's convention). The checker often answers
        # none there; when its blind status answer already confirmed the denial / possibility, the label stands.
        conv = kind in ("fact:negated", "fact:hypothetical") and b == "none" and status_ok.get((h, t))
        if b != lab and not conv:
            continue
        cands = PAIR_RELS[(E[h]["kind"], E[t]["kind"])]
        crit = {rname(x): REL[x][2] for x in cands} | {"none": NONE_DESC if rng.random() < 0.8 else NONE_ALT}
        keys = list(crit)
        hn, tn = E[h]["name"], E[t]["name"]
        add(row("choose", state, rng.choice(RELATION_Q).format(h=hn, t=tn), choice_opts(crit),
                soft(len(keys), keys.index(rname(lab) if lab != "none" else "none"), 0.9), "kg-relation", i * 100 + k, split))
    for (m, kind, _), b in zip(mentions, type_ans):
        lab = omap[kind]
        st["type"].append([lab, b, kind])
        if b != lab:
            continue
        keys = list(ocrit)
        p = soft(len(keys), keys.index(lab), 0.9)
        if kind == "ticker" and "security" in keys:            # a ticker is a defensible "security" too; keep that mass
            p[keys.index(lab)], p[keys.index("security")] = 0.8, 0.12
        add(row("choose", state, rng.choice(TYPE_Q).format(m=m), choice_opts(ocrit), p, "kg-type", i * 100 + k, split))
    return rows, st


# ------------------------------------------------------------------ threads

INTENT_SETS = {
    "support": {
        "refund request": "The customer wants money returned for something they paid for.",
        "technical support": "Something the customer owns is not working and they want it fixed or diagnosed.",
        "order status": "The customer wants to know where an order is or when it will arrive.",
        "billing dispute": "The customer disagrees with a charge, an amount, or an invoice.",
        "complaint": "The customer is expressing dissatisfaction and wants it acknowledged, beyond any single fix.",
        "cancellation": "The customer wants to cancel an order, a plan, or a service.",
    },
    "saas": {
        "access problem": "The customer cannot log in or reach their account and wants access back.",
        "bug report": "A feature behaves wrongly and the customer reports it so that it gets fixed.",
        "feature request": "The customer asks for a capability the product does not have yet.",
        "plan change": "The customer wants to upgrade, downgrade or change seats on their subscription.",
        "data request": "The customer wants their data exported, deleted or corrected.",
        "account closure": "The customer wants to close the account and stop the subscription.",
    },
    "travel": {
        "booking change": "The customer wants to change dates, names or seats on an existing booking.",
        "lost baggage": "The customer's luggage is missing, delayed or damaged.",
        "refund claim": "The customer wants money back for a cancelled, delayed or unused service.",
        "loyalty points": "The customer asks about missing or expiring loyalty points or status.",
        "special assistance": "The customer needs help arranged: wheelchair, medical needs, travelling with a child or a pet.",
        "service complaint": "The customer is unhappy with how they were treated and wants it acknowledged.",
    },
}
BUSINESSES = {"support": ["an online electronics store", "a furniture retailer", "a meal-kit subscription", "a mobile phone carrier",
                          "a home-internet provider", "a smart-home device maker", "a fitness-club chain", "an online pharmacy",
                          "a streaming service", "a bike-share app", "an appliance maker", "a print-on-demand shop"],
              "saas": ["a project-management SaaS", "an accounting platform for small businesses", "a CRM tool", "a website builder",
                       "a cloud file-storage service", "an HR and payroll platform", "an email-marketing tool", "a password manager"],
              "travel": ["an airline", "a hotel chain", "a rail operator", "a ferry company", "an online travel agency", "a car-rental firm"]}
BUSINESSES_TEST = {"support": ["a pet-supplies shop", "a solar-panel installer"], "saas": ["a video-conferencing service"], "travel": ["a cruise line"]}
CHANNELS = ["email", "live chat", "phone call transcript", "web contact form", "social media direct message", "in-app messaging"]
CHANNELS_TEST = ["SMS", "community forum thread"]
PRIORITY = [
    "Low: no deadline, nothing is broken, the customer is content to wait.",
    "Medium: a real problem with a workaround, or a question that should be answered within normal service levels.",
    "High: the customer is blocked or out of pocket, is asking repeatedly, or a service level is at risk.",
    "Urgent: an unresolved failure with an explicit deadline, a threat to leave or escalate publicly, or a breached commitment.",
]
PRIO_WRITE = ["calm and patient; no deadline; nothing is broken or costing them money; they are happy to wait",
              "a real but bearable problem: there is a workaround, or it is an ordinary question; normal tone; no deadline, no threats",
              "the customer is blocked or out of pocket, and has to ask again or follow up because answers are slow; but there is NO hard "
              "deadline, NO threat to leave, post publicly or escalate, and NO promise the company already broke",
              "an explicit hard deadline (e.g. tomorrow morning, before a flight, before payroll runs) and/or a threat to leave, post publicly or "
              "escalate, or a promise the company already broke; the failure is still not fixed when the urgency is raised"]
RESOLVED_WRITE = {True: ["the agent fixes the problem or grants the request, and the customer confirms it worked and thanks them in the last turns"],
                  False: ["the agent escalates to another team and the customer is left waiting, still without a fix",
                          "the agent offers something the customer rejects; the customer remains unhappy at the end",
                          "the agent asks the customer for information or to wait several days; nothing is fixed yet at the last turn",
                          "the agent promises a callback or a fix later; the last customer turn makes clear they are still waiting"]}
INTENT_Q = ["What does the customer want from this thread, taken as a whole? Judge the customer's turns in `turns`, not the agent's.",
            "What does the customer want from this thread, taken as a whole? Judge the customer's turns in `turns`, not the agent's.",
            "Read the whole conversation in `turns`. What is the customer's main request? Go by what the customer asks for, not by what the agent offers.",
            "Classify this support thread by the customer's underlying goal."]
PRIORITY_Q = ["How urgently does this thread need attention, as of its last turn?",
              "How urgently does this thread need attention, as of its last turn?",
              "As of the final turn, how urgent is this support thread?",
              "Triage: what priority should this conversation get right now?"]
PRIORITY_Q_RESOLVED = ["How urgent was the customer's issue when they raised it, before it was handled?",
                       "Judging from the thread, how urgent was this customer's problem at its worst point?",
                       "Before the agent dealt with it, how urgently did this customer's issue need attention?"]
RESOLVED_Q = ["By the last turn in `turns`, has the customer's issue been resolved to the customer's evident satisfaction?",
              "By the last turn in `turns`, has the customer's issue been resolved to the customer's evident satisfaction?",
              "Is the customer's problem solved, and do they accept the outcome, by the end of the thread?",
              "Does this conversation end with the customer's issue settled and the customer satisfied?"]
RESOLVED_CRIT = {"true": "The customer's problem is fixed or their request granted, and they accept it.",
                 "false": "The issue is open, deferred, escalated, or the customer is still dissatisfied."}

_TOK = None


def ntok(s: str) -> int:
    global _TOK
    if _TOK is None:
        try:
            import os
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            from transformers import AutoTokenizer
            _TOK = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-base")
        except Exception:  # noqa: BLE001
            _TOK = False
    return len(_TOK(s)["input_ids"]) if _TOK else len(s) // 4


async def job_threads(tw, tc, rng, i, split):
    iset = rng.choices(list(INTENT_SETS), [0.5, 0.25, 0.25])[0]
    crit = INTENT_SETS[iset]
    intent = rng.choice(list(crit))
    prio = rng.randrange(4)
    resolved = rng.random() < 0.5
    how = rng.choice(RESOLVED_WRITE[resolved])
    biz = rng.choice((BUSINESSES_TEST if split == "test" else BUSINESSES)[iset])
    channel = rng.choice(CHANNELS_TEST if split == "test" else CHANNELS)
    used: set = set()
    company = f"{pseudo(rng, used)}{rng.choice(['', ' Direct', ' Home', ' Cloud', ' Air', ' Go', ' Plus'])}"
    n_turns = rng.randint(10, 16)
    yr = rng.choice([2025, 2026])
    opened = f"{yr}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}T{rng.randint(6, 22):02d}:{rng.randint(0, 59):02d}:00Z"
    generic = rng.random() < 0.5
    L = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    ref = rng.choice([f"{rng.choice(L)}{rng.choice(L)}-{rng.randint(10000, 9999999)}", f"#{rng.randint(100000, 99999999)}",
                      f"{rng.choice(L)}{rng.randint(1000, 999999)}{rng.choice(L)}", f"{rng.choice(L)}{rng.choice(L)}{rng.choice(L)}{rng.randint(100, 99999)}"])
    prompt = (f"Write a realistic customer-support thread between a customer and an agent of {company}, {biz}, over {channel}. "
              f"Exactly {n_turns} turns, starting with the customer; mostly alternating, though a customer may send two messages in a row. "
              f"Total length 600-1300 words; vary turn lengths (one line to a long paragraph). Use concrete invented details: order or "
              f"account numbers, dates, amounts, product names, error messages. The thread was opened on {opened[:10]}; keep dates consistent with that. "
              f"The main order, booking or account reference is {ref}; any other reference numbers must look different from it.\n\n"
              f"The customer's underlying goal: {crit[intent]} They may mention side issues, but this is what they want from the thread.\n"
              f"Urgency{' of the issue while it was open' if resolved else ', as of the last turn'}: {PRIO_WRITE[prio]}.\n"
              f"How it ends: {how}.\n\n"
              "Show all of this only through what people say. Never name a category, priority level or status (no 'priority', 'urgent ticket', "
              "'resolved', 'intent'). Do not let the agent summarise the case at the end.\n"
              + ("Also give the thread a short, generic email-style subject that does not reveal the customer's goal (e.g. an order or ticket number).\n"
                 if generic else "Also give the thread a short subject line the customer might write, without words like urgent, ASAP or priority.\n")
              + 'Output JSON only: {"subject": "...", "turns": [{"speaker": "customer" or "agent", "text": "..."}, ...]}')
    r = await tw.chat([{"role": "user", "content": prompt}], temperature=0.9, max_tokens=3200, thinking=False, json_mode=True, tag="kg-threads")
    d = loads_lenient(r.text)
    turns = d.get("turns") if isinstance(d, dict) else None
    st = {"names_ok": False}
    if not isinstance(turns, list):
        return [], st
    turns = [{"speaker": t.get("speaker", "").strip().lower(), "text": str(t.get("text", "")).strip()} for t in turns if isinstance(t, dict)]
    subject = str(d.get("subject", "")).strip()[:120]
    if not subject or re.search(r"urgent|asap|priority|immediate", subject, re.I):
        subject = rng.choice([f"Ticket #{rng.randint(10000, 99999)}", f"Case {rng.randint(100000, 999999)}", "Re: your recent message", "Help please"])
    state = {"subject": subject, "channel": channel, "opened": opened, "turns": turns}
    s = json.dumps(state, ensure_ascii=False)
    st["tokens"] = ntok(s)
    st["turns"] = len(turns)
    ok = (9 <= len(turns) <= 17 and all(t["speaker"] in ("customer", "agent") and t["text"] for t in turns)
          and turns[0]["speaker"] == "customer" and 500 <= st["tokens"] <= 1750
          and not re.search(r"\b(intent|priority level)\b", s, re.I))
    st["names_ok"] = ok
    if not ok:
        return [], st
    prompt_c = (f"Read this customer-support thread.\n\nThread (JSON):\n{s}\n\nAnswer three questions.\n"
                f"1. intent — What does the customer want from this thread, taken as a whole? Judge the customer's turns, not the agent's. Options:\n{opts_block(crit)}\n"
                f"2. resolved — By the last turn, has the customer's issue been resolved to the customer's evident satisfaction? true if: {RESOLVED_CRIT['true']} false if: {RESOLVED_CRIT['false']}\n"
                f"3. priority — {PRIORITY_Q_RESOLVED[0] if resolved else PRIORITY_Q[0]} Levels:\n" + "\n".join(f"{j}. {p}" for j, p in enumerate(PRIORITY)) +
                '\n\nOutput JSON only: {"intent": "<option>", "resolved": true or false, "priority": <0-3>}')
    rc = await tc.chat([{"role": "user", "content": prompt_c}], temperature=0.0, max_tokens=80, thinking=False, json_mode=True, tag="kg-check")
    b = loads_lenient(rc.text)
    bi = str(b.get("intent", "")).split(":")[0].strip()
    br = b.get("resolved")
    br = br if isinstance(br, bool) else {"true": True, "false": False}.get(str(br).lower())
    try:
        bp = int(b.get("priority"))
    except (TypeError, ValueError):
        bp = None
    st.update(intent=[intent, bi, iset], resolved=[resolved, br], priority=[prio, bp])
    rows, k = [], 0
    if bi == intent:
        keys = list(crit)
        c2 = {kk: crit[kk] for kk in keys}
        if rng.random() < 0.3:
            rng.shuffle(keys)
            c2 = {kk: crit[kk] for kk in keys}
        rows.append(row("choose", state, rng.choice(INTENT_Q), choice_opts(c2), soft(len(keys), keys.index(intent), 0.9), "kg-intent", i * 10 + k, split)); k += 1
    if br is resolved:
        rows.append(verify(state, rng.choice(RESOLVED_Q), resolved, "kg-resolved", i * 10 + k, split,
                           crit=RESOLVED_CRIT if rng.random() < 0.85 else None, p=0.94)); k += 1
    if bp is not None and abs(bp - prio) <= 1:
        p = [0.02] * 4
        if bp == prio:
            p[prio] = 0.88
            for j in (prio - 1, prio + 1):
                if 0 <= j < 4:
                    p[j] = 0.05
        else:
            p[prio], p[bp] = 0.68, 0.26
        rows.append(row("score", state, rng.choice(PRIORITY_Q_RESOLVED if resolved else PRIORITY_Q), list(PRIORITY), p, "kg-priority", i * 10 + k, split)); k += 1
    return [r_ | {"source": "kg-llm"} for r_ in rows], st


JOBS = {"docs": job_docs, "threads": job_threads}


# ------------------------------------------------------------------ driver and report

async def run(a) -> None:
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    sidecar = Path(str(out) + ".stats.jsonl")
    done = set()
    if out.exists():
        done = {json.loads(l)["call"] for l in out.open() if l.strip()}
    todo = [i for i in range(a.calls) if i not in done]
    print(f"{len(done)} calls done, {len(todo)} to go", flush=True)
    st = {"calls": 0, "rows": 0, "failed": 0, "kept_docs": 0}
    fo, fs = out.open("a"), sidecar.open("a")
    async with Teacher(model=WRITER, concurrency=a.writer_streams) as tw, Teacher(model=CHECKER, concurrency=a.checker_streams) as tc:
        sem = asyncio.Semaphore(a.writer_streams + a.checker_streams)

        async def one(i):
            async with sem:
                rng = random.Random(f"kg-llm:{a.job}:{a.split}:{a.seed}:{i}")
                try:
                    rows, s = await JOBS[a.job](tw, tc, rng, i, a.split)
                except (TeacherError, ValueError, AttributeError, KeyError, TypeError, IndexError) as e:
                    st["failed"] += 1
                    rows, s = [], {"error": repr(e)[:200]}
                for r in rows:
                    fo.write(json.dumps({**r, "call": i}, ensure_ascii=False) + "\n")
                if not rows:
                    fo.write(json.dumps({"empty": True, "call": i}) + "\n")
                fs.write(json.dumps({"call": i, **s}, ensure_ascii=False) + "\n")
                fo.flush(); fs.flush()
                st["calls"] += 1; st["rows"] += len(rows); st["kept_docs"] += bool(rows)
                if st["calls"] % 25 == 0:
                    print(st, flush=True)
        await asyncio.gather(*(one(i) for i in todo))
    fo.close(); fs.close()
    print("done:", st, flush=True)


def report(path: str) -> None:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    stats = {}
    sp = Path(path + ".stats.jsonl")
    if sp.exists():
        for l in sp.open():
            if l.strip():
                s = json.loads(l); stats[s["call"]] = s
    calls = {r["call"] for r in rows}
    real = [r for r in rows if not r.get("empty")]
    print(f"{path}: {len(calls)} calls, {len(real)} rows, {sum(1 for r in rows if r.get('empty'))} empty calls")
    S = list(stats.values())
    if S:
        err = sum(1 for s in S if "error" in s)
        ok = sum(1 for s in S if s.get("names_ok"))
        print(f"  writer/format ok (names present, length, no label words): {ok}/{len(S)} = {ok / len(S):.1%}   errors: {err}")

    def agree(key, f=lambda x: x[0] == x[1], by=lambda x: x[0]):
        pairs = [p for s in S for p in ([s[key]] if s.get(key) and not isinstance(s[key][0], list) else s.get(key, []))]
        if not pairs:
            return
        n = sum(f(p) for p in pairs)
        print(f"  blind agreement {key}: {n}/{len(pairs)} = {n / len(pairs):.1%}")
        grp = defaultdict(lambda: [0, 0])
        for p in pairs:
            g = by(p); grp[g][0] += f(p); grp[g][1] += 1
        print("     " + "  ".join(f"{g}: {a}/{b}" for g, (a, b) in sorted(grp.items(), key=lambda x: -x[1][1])))
    if any("status" in s for s in S):
        agree("status")
        agree("neg", f=lambda p: p[0] == "absent" and p[1], by=lambda p: f"blind={p[0]}{'' if p[1] else '/rel-connected'}")
        agree("rel", by=lambda p: f"{p[2]}")
        conv = sum(1 for s in S for p in s.get("rel", []) if p[2] in ("fact:negated", "fact:hypothetical") and p[1] == "none")
        print(f"     (kept by convention: denied / possible fact relations the checker called none, status confirmed: ≤{conv})")
        agree("rel", by=lambda p: f"{p[0]}")
        agree("type", by=lambda p: p[2])
        conf = Counter((p[0], p[1]) for s in S for p in s.get("status", []) if p[0] != p[1])
        print("  status disagreements (code→blind):", dict(conf.most_common(10)))
        conf = Counter((p[0], p[1]) for s in S for p in s.get("type", []) if p[0] != p[1])
        print("  type disagreements (code→blind):", dict(conf.most_common(10)))
        conf = Counter((p[0], p[1]) for s in S for p in s.get("rel", []) if p[0] != p[1])
        print("  relation disagreements (code→blind):", dict(conf.most_common(10)))
    if any("intent" in s for s in S):
        agree("intent", by=lambda p: p[2])
        agree("resolved", by=lambda p: str(p[0]))
        agree("priority", f=lambda p: p[1] is not None and abs(p[0] - p[1]) <= 1, by=lambda p: f"code={p[0]}")
        agree("priority", by=lambda p: f"exact,code={p[0]}")
        toks = [s["tokens"] for s in S if "tokens" in s]
        if toks:
            toks.sort(); print(f"  state tokens: min {toks[0]} median {toks[len(toks) // 2]} max {toks[-1]}")
    by_task = defaultdict(Counter)
    for r in real:
        lab = r["choices"][r["gold"]].split(":")[0]
        by_task[r["task"]][lab] += 1
    for t, c in sorted(by_task.items()):
        print(f"  {t}: {sum(c.values())} rows  " + "  ".join(f"{k}={v}" for k, v in c.most_common()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", choices=list(JOBS))
    ap.add_argument("--calls", type=int, default=100)
    ap.add_argument("--split", choices=("train", "test"), default="train")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--writer-streams", type=int, default=4)
    ap.add_argument("--checker-streams", type=int, default=4)
    ap.add_argument("--out")
    ap.add_argument("--report", nargs="*")
    a = ap.parse_args()
    if a.report:
        for p in a.report:
            report(p)
        return
    if not (a.job and a.out):
        ap.error("--job and --out are required unless --report")
    if a.writer_streams + a.checker_streams > 12:
        ap.error("keep total streams ≤ 12 (the GX10 is shared)")
    asyncio.run(run(a))


if __name__ == "__main__":
    main()
