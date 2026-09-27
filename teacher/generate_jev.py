"""Generator 3: long, realistic "Jev-style" business decisions — one long state, several questions.

    uv run python -m teacher.generate_jev --n 20 --streams 2 --out /tmp/x.jsonl       # smoke test
    setsid nohup uv run python -m teacher.generate_jev --n 12000 --streams 4 \
        --out data/teacher/jevgen-rande-fast-local.jsonl > runs/logs/jevgen.log 2>&1 &

Why a separate generator: generate.py asks for 10 short states per call (1–6 sentences).
The decisions Decima is weakest on look different — a 300–1500-word artefact (JSON logs,
an alert payload, an email thread, an invoice, an agent's tool trace, a support chat, a
policy plus a request, retrieved passages plus a query, a proposed tool call) and several
independent questions about it, each with its own kind and choice set. Here one call =
one state + 3–5 questions; we store ONE ROW PER QUESTION (the state repeated, a distinct
id per row, a shared `group` so splits/analysis can keep a state's questions together).

Reply format is deliberately NOT a single JSON object: long states full of quotes, JSON
and newlines are where JSON-mode escaping breaks. The teacher writes the state as raw
text between markers, then a small JSON block for the questions.

Grid (seeded; call i always asks for the same cell, so runs resume by cell id):
  state language EN 70 / FA 10 / AR 10 / RU 10; for non-EN states ~30 % English choices
  question kinds choose 45 / verify 30 / score 25; choose 3–12 options, score 3–7 levels
  ~15 % of choose questions end in a none-of-the-above / insufficient-information option
  ~30 % of questions are asked to be genuinely uncertain (top prob ≤ 0.65)
Validation: probs length = choices, in [0,1], sum within ±10 % (renormalised); verify has
2 choices; gold must be the argmax or within 0.05 of it (near-tie: the teacher's gold is
kept and `gold_tie` is set); duplicate choices / states / ids are dropped. Rejects go to
<out>.rejects.jsonl.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from decima.normalize import normalize
from teacher.client import Teacher, TeacherError
from teacher.prompts import LANG_NAME, NONE_TEXT, loads_lenient

LANG_W = {"en": 0.70, "fa": 0.10, "ar": 0.10, "ru": 0.10}
ENGLISH_CHOICES = 0.30            # share of non-English states whose choices are in English
KIND_W = {"choose": 0.45, "verify": 0.30, "score": 0.25}
CHOOSE_N = ([3, 4, 5, 6, 7, 8, 10, 12], [0.14, 0.18, 0.18, 0.14, 0.10, 0.10, 0.08, 0.08])
SCORE_N = ([3, 4, 5, 6, 7], [0.22, 0.22, 0.34, 0.07, 0.15])
N_Q = ([3, 4, 5], [0.3, 0.4, 0.3])
NONE_RATE = 0.15                  # choose questions that carry a none / insufficient-information option
NONE_GOLD = 0.35                  # …of which the state really does not support any other option
UNCERTAIN = 0.30                  # questions asked to be genuinely uncertain
LENGTHS = ([(300, 500), (500, 800), (800, 1100), (1100, 1500)], [0.25, 0.35, 0.25, 0.15])
MIN_WORDS, MAX_WORDS = 200, 2400  # accepted range (the target band is advisory; models undershoot)

INSUFF_TEXT = {
    "en": "insufficient information to decide",
    "fa": "اطلاعات کافی برای تصمیم‌گیری وجود ندارد",
    "ar": "المعلومات غير كافية لاتخاذ القرار",
    "ru": "недостаточно информации для решения",
}

# domain → (description, sub-scenarios, typical artefact formats). Written from the public
# descriptions of the use cases only; nothing here is taken from an evaluation set.
DOMAINS: dict[str, tuple[str, list[str], list[str]]] = {
    "support_triage": ("customer-support triage and routing for a software or consumer company",
        ["billing dispute after a plan change", "outage report mixed with a feature request", "account lockout after SSO migration",
         "angry repeat contact about an unresolved refund", "enterprise customer asking about data export before churning",
         "bug report with reproduction steps and screenshots described", "request that belongs to a partner, not the vendor"],
        ["support chat transcript with timestamps", "email thread with quoted replies", "ticket with metadata fields and customer history"]),
    "model_routing": ("routing an incoming request to a cheap small model, a mid model, or a large reasoning model by difficulty and risk",
        ["multi-step math word problem", "short factual lookup", "long legal document summarisation", "code refactor across files",
         "ambiguous creative brief", "customer message needing tone care", "data extraction from a messy table"],
        ["the user request with conversation history and a router metadata block (tokens, tenant tier, latency budget)", "API request JSON with the prompt and prior turns"]),
    "rag_relevance": ("judging whether retrieved passages are relevant to a user query in a retrieval-augmented system",
        ["internal HR policy search", "product documentation lookup", "medical-information FAQ (non-diagnostic)", "financial filings search",
         "engineering wiki search", "legal knowledge base", "travel policy search"],
        ["a user query followed by 4–8 numbered retrieved passages with source titles and scores", "search log with query, filters and top-k chunks"]),
    "citation_support": ("checking whether a generated answer's claims and citations are supported by the cited sources",
        ["answer citing two of five sources, one claim unsupported", "numbers misquoted from a report", "correct answer with a wrong citation index",
         "answer that over-generalises a source", "sources that contradict each other"],
        ["question, generated answer with bracketed citations, and the cited source excerpts", "grounding-check payload with claims list and evidence snippets"]),
    "tool_risk": ("gating an AI agent's proposed tool/action call by risk: read-only vs reversible vs destructive, and whether it touches production",
        ["database migration proposal", "bulk email send", "cloud resource deletion", "feature-flag change", "file cleanup script",
         "payment refund via API", "permission change in an identity provider", "read-only analytics query"],
        ["agent plan followed by a proposed tool call in JSON with arguments and target environment", "change request with a shell command and context from the conversation"]),
    "agent_verification": ("verifying an AI agent's work: is the task actually done, did it follow policy, is it looping or repeating itself",
        ["coding agent claims tests pass", "research agent returns a report", "customer-service agent issued a credit", "browser agent booking travel",
         "data agent built a dashboard", "agent stuck retrying the same failing call", "agent that skipped an approval step"],
        ["the user's task, then a step-by-step agent trace of tool calls and observations, then the agent's final message", "agent run log in JSON lines"]),
    "security_incident": ("security operations: triaging alerts and incidents",
        ["impossible-travel login", "EDR alert on a developer laptop", "suspicious OAuth app consent", "data exfiltration via cloud storage",
         "brute force against VPN", "malicious package in CI dependency", "insider accessing unusual records", "false positive from a scanner"],
        ["SIEM alert JSON payload plus enrichment and analyst notes", "raw auth / proxy / EDR log lines", "incident ticket with timeline"]),
    "invoice_ap": ("accounts-payable invoice processing",
        ["duplicate invoice with a changed number", "PO mismatch on quantity", "vendor bank-detail change request", "missing tax ID",
         "invoice in a foreign currency with rounding", "early-payment discount window", "services invoice without receipt of goods"],
        ["OCR text of an invoice with line items, totals, tax and remittance details", "AP workflow record with PO, receipt and invoice fields", "vendor email with an attached invoice described"]),
    "fraud": ("payments and account fraud review",
        ["card-not-present order with mismatched addresses", "account takeover signals", "refund abuse pattern", "promo abuse with linked accounts",
         "merchant chargeback spike", "wire request from a new beneficiary", "synthetic identity at onboarding"],
        ["transaction record with risk signals and device fingerprint", "case notes with a timeline of account events", "rules-engine output JSON"]),
    "moderation": ("trust & safety content moderation on a platform",
        ["heated political comment thread", "marketplace listing that may be counterfeit", "harassment across several DMs", "self-promotion spam",
         "satire mistaken for misinformation", "graphic news footage description", "coordinated review brigading"],
        ["reported content with surrounding context and reporter reason", "moderation queue item with user history and prior strikes"]),
    "hr": ("HR and people operations",
        ["leave request overlapping a blackout period", "complaint about a manager", "relocation and payroll change", "performance-improvement follow-up",
         "candidate background-check discrepancy", "expense for a team event", "accommodation request"],
        ["employee email thread", "HR case record with policy excerpt", "HRIS form submission with free-text justification"]),
    "legal_contract": ("legal review of contract clauses and requests",
        ["limitation-of-liability clause redline", "auto-renewal and termination notice", "data-processing addendum gaps", "IP assignment in a contractor agreement",
         "non-compete scope", "indemnity carve-outs", "governing-law change request"],
        ["contract excerpt with numbered clauses plus the counterparty's redline comments", "legal intake request with the relevant clause pasted"]),
    "healthcare_triage": ("non-diagnostic healthcare administration and message triage (routing and urgency only, never diagnosis)",
        ["patient portal message about medication refill", "appointment reschedule with worsening symptoms mentioned", "billing question about a claim",
         "prior-authorisation paperwork", "lab result inquiry", "referral routing", "caregiver asking for records"],
        ["patient portal message thread with clinic metadata", "intake form with free-text reason for visit", "call-centre note"]),
    "ecommerce": ("e-commerce operations",
        ["return outside the window with a defect claim", "wrong item shipped", "marketplace seller dispute", "price-match request",
         "subscription box cancellation", "bulk B2B order change", "product review flagged as incentivised"],
        ["order record JSON plus customer chat", "seller-support ticket with order history", "returns-portal submission"]),
    "logistics": ("logistics, shipping and warehouse exceptions",
        ["customs hold on an international parcel", "temperature excursion on a cold-chain shipment", "missed pickup window", "damaged pallet at receiving",
         "address correction after dispatch", "carrier capacity shortfall", "inventory count mismatch"],
        ["tracking event log", "driver / dispatcher message thread", "WMS exception record with SKU lines"]),
    "it_ops": ("IT operations and SRE incident handling",
        ["disk filling on a database host", "latency spike after a deploy", "certificate expiry warning", "flapping health check",
         "DNS misconfiguration", "queue backlog growing", "on-call handoff with an unresolved alert"],
        ["monitoring alert payload plus recent deploy log", "incident chat channel excerpt", "kubectl / system log output"]),
    "code_review": ("code review risk assessment",
        ["auth middleware change", "database query change without an index", "dependency bump with a breaking change", "config change for feature rollout",
         "refactor touching payment rounding", "test-only change", "logging change that may leak PII"],
        ["pull-request description plus a unified diff and CI summary", "review thread with inline comments"]),
    "it_helpdesk": ("internal IT helpdesk",
        ["new-hire access bundle", "lost laptop", "MFA device replacement", "software license request", "shared-drive permission request",
         "VPN not connecting from a hotel", "phishing email forwarded by an employee"],
        ["helpdesk ticket with employee profile fields", "chat with the IT bot handing off to a human"]),
    "insurance": ("insurance claims and policy servicing",
        ["water-damage claim with late notice", "auto claim with conflicting statements", "coverage question about a rider", "claim with possible pre-existing damage",
         "policy lapse and reinstatement", "subrogation opportunity"],
        ["first-notice-of-loss form plus adjuster notes", "policyholder email thread with policy excerpt"]),
    "banking": ("retail and business banking operations",
        ["dispute of a recurring charge", "KYC refresh with missing document", "large cash deposit flagged", "loan hardship request",
         "business account signatory change", "failed international transfer"],
        ["core-banking case record with customer messages", "compliance alert with transaction list"]),
    "procurement": ("procurement and vendor management",
        ["new SaaS vendor security questionnaire", "sole-source justification", "contract renewal with a price increase", "vendor with a sanctions-list near match",
         "purchase request over approval threshold"],
        ["purchase request form with justification and quotes", "vendor-risk assessment record"]),
    "compliance_privacy": ("privacy and regulatory compliance",
        ["data-subject access request", "deletion request conflicting with retention", "marketing consent question", "cross-border transfer",
         "breach notification assessment", "vendor sharing personal data"],
        ["privacy request intake with identity-verification notes", "policy excerpt plus an internal request email"]),
    "sales_crm": ("sales and CRM operations",
        ["inbound lead qualification", "renewal risk signals", "discount approval request", "territory conflict between reps", "deal stage update from call notes"],
        ["CRM record with activity timeline", "call transcript summary plus email"]),
    "education": ("education administration",
        ["grade appeal", "academic-integrity flag from a similarity report", "accommodation for exams", "late enrolment", "scholarship eligibility"],
        ["student email thread with course record", "LMS submission metadata plus instructor notes"]),
    "travel_expense": ("corporate travel and expense",
        ["expense report with an out-of-policy hotel", "flight change due to weather", "per-diem dispute", "missing receipt affidavit", "duplicate expense lines"],
        ["expense report lines with policy excerpt", "travel booking itinerary plus traveller message"]),
    "real_estate": ("property management",
        ["maintenance request with safety implications", "lease renewal negotiation", "noise complaint", "security-deposit dispute", "rent payment plan request"],
        ["tenant portal message thread", "work-order record with inspection notes"]),
    "public_services": ("government and public-service case handling",
        ["benefits application with a missing document", "permit application with a zoning question", "tax notice dispute", "records request"],
        ["application form fields plus case-worker notes", "citizen email with an agency policy excerpt"]),
    "data_quality": ("data engineering and data-quality decisions",
        ["pipeline job failed with schema drift", "duplicate customer records to merge", "anomaly in daily revenue metric", "PII found in an analytics table"],
        ["job run log plus schema diff", "data-quality check report JSON"]),
    "email_ops": ("executive / shared-inbox email triage",
        ["meeting request conflict", "vendor escalation", "press inquiry", "invoice reminder", "internal announcement needing action", "possible phishing"],
        ["a thread of several emails with headers", "inbox digest of related messages"]),
    "product_feedback": ("product-feedback and bug-report analysis",
        ["app-store reviews after a release", "NPS verbatims", "feature request from a key account", "regression reports clustering on one device"],
        ["batch of user feedback items with metadata", "bug tracker issue with comments"]),
}

KIND_RULE = {
    "choose": "choose — exactly {n} mutually exclusive options (category, destination, action, root cause…); exactly one is correct.",
    "score": "score — exactly {n} ORDERED levels of one quantity (urgency, risk, severity, confidence, quality, relevance…), listed lowest → highest; the option texts name the levels. Mass must sit on adjacent levels, never split between far-apart levels.",
    "verify": "verify — a yes/no question about one specific proposition; exactly 2 options, the first affirming (yes…), the second denying (no…).",
}


@dataclass(frozen=True)
class QSpec:
    kind: str
    n: int
    none: str | None      # None | "none" | "insufficient"
    none_gold: bool
    uncertain: bool


@dataclass
class JevCell:
    i: int
    domain: str
    scenario: str
    fmt: str
    state_lang: str
    choice_lang: str
    words: tuple[int, int]
    date: str = ""
    qs: list[QSpec] = field(default_factory=list)


def _pick(rng: random.Random, w: dict):
    return rng.choices(list(w), weights=list(w.values()))[0]


def cell(i: int, seed: int = 0) -> JevCell:
    rng = random.Random(f"jev:{seed}:{i}")
    dom = rng.choice(list(DOMAINS))
    _, scen, fmts = DOMAINS[dom]
    sl = _pick(rng, LANG_W)
    cl = "en" if sl != "en" and rng.random() < ENGLISH_CHOICES else sl
    c = JevCell(i=i, domain=dom, scenario=rng.choice(scen), fmt=rng.choice(fmts), state_lang=sl, choice_lang=cl,
                words=rng.choices(*LENGTHS)[0],
                date=f"{rng.randint(2022, 2026)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}")
    for _ in range(rng.choices(*N_Q)[0]):
        k = _pick(rng, KIND_W)
        n = 2 if k == "verify" else rng.choices(*(CHOOSE_N if k == "choose" else SCORE_N))[0]
        none = None
        if k == "choose" and rng.random() < NONE_RATE:
            none = rng.choice(["none", "insufficient"])
            n = max(n, 3)
        c.qs.append(QSpec(k, n, none, none is not None and rng.random() < NONE_GOLD, rng.random() < UNCERTAIN))
    if not any(q.uncertain for q in c.qs) and rng.random() < 0.5:   # most states get at least one hard question
        j = rng.randrange(len(c.qs)); q = c.qs[j]
        c.qs[j] = QSpec(q.kind, q.n, q.none, q.none_gold, True)
    return c


SYSTEM = (
    "You produce training data for Decima, a small decision model used inside business software. Decima receives a "
    "STATE (the full text a system has in front of it), a QUESTION and a list of OPTIONS, and outputs a calibrated "
    "probability distribution over the options. You are the expert: you write realistic states and label them the "
    "way a careful senior practitioner would, with honest uncertainty."
)


def jev_prompt(c: JevCell) -> list[dict]:
    desc = DOMAINS[c.domain][0]
    lang = LANG_NAME[c.state_lang]
    opt_lang = LANG_NAME[c.choice_lang]
    q_lines = []
    for k, q in enumerate(c.qs, 1):
        line = f"Q{k}: " + KIND_RULE[q.kind].format(n=q.n)
        if q.none:
            txt = (NONE_TEXT if q.none == "none" else INSUFF_TEXT)[c.choice_lang]
            line += f' The LAST option must be exactly "{txt}".'
            line += (" In this question it IS the correct answer: the state genuinely does not support any other option."
                     if q.none_gold else " Here it should get only a small probability.")
        line += (" Make this question genuinely uncertain from the state: top probability between 0.35 and 0.65, mass shared by 2–3 options."
                 if q.uncertain else " Label it with the confidence the evidence warrants (can be high).")
        q_lines.append(line)
    lo, hi = c.words
    body = f"""Write ONE realistic state and {len(c.qs)} independent questions about it.

Domain: {desc}.
Scenario: {c.scenario}.
Artefact form: {c.fmt}. Start the state the way this form really starts (no generic "[SYSTEM LOG]" banner unless the form is a log).
Events happen around {c.date}; invent fresh organisation and person names that fit the language.
State length: {lo}–{hi} words — at least {lo} words, aim for about {(lo + hi) // 2}; a shorter state is unusable, so write the full artefact with all its parts. Language of the state and the questions: {lang}. Language of the options: {opt_lang}.

The state is the raw material a system would actually see — not a summary. Use concrete, invented but plausible details:
company and person names, IDs, timestamps, amounts, versions, hostnames, error strings, quoted policy text. Include the
noise real artefacts have (signatures, irrelevant log lines, boilerplate, side topics) and at least one detail that
matters for a question but is easy to miss. Do NOT state the answers or reuse option wording inside the state.
Machine-readable parts (JSON, logs, tables) stay as they would be in reality; free text is in {lang}.

Questions (each answerable from the state alone, each about a DIFFERENT aspect; vary the phrasing, no templates;
variation seed {c.i}):
{chr(10).join(q_lines)}

For every question give "probs": one number per option, summing to 1, the probability a careful expert assigns.
Never assign exactly 0 to an option a reasonable reader could pick. "gold" = 0-based index of the best option (the argmax).

Output format — exactly this, nothing before or after:
<<<STATE
(the state text)
STATE>>>
{{"questions": [{{"kind": "choose|score|verify", "question": "...", "options": ["...", ...], "probs": [...], "gold": <int>}}, ...]}}"""
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": body}]


# ---------------------------------------------------------------- parsing / validation ----

class Reject(ValueError):
    pass


_STATE_RE = re.compile(r"<<<\s*STATE\s*(?:>>>)?\s*\n(.*?)\n\s*(?:<<<\s*)?(?:END[ _]?)?STATE\s*>>>", re.S)


def split_reply(text: str) -> tuple[str, dict]:
    m = _STATE_RE.search(text)
    if m:
        return m.group(1).strip(), loads_lenient(text[m.end():])
    # fallback: opening marker present, closing one forgotten → the state runs up to the questions JSON
    a = re.search(r"<<<\s*STATE\s*(?:>>>)?\s*\n", text)
    b = list(re.finditer(r'\{\s*"questions"\s*:', text))
    if not (a and b) or b[-1].start() <= a.end():
        raise Reject("no state markers")
    state = re.sub(r"\n\s*(?:<<<\s*)?(?:END[ _]?)?STATE\s*(?:>>>)?\s*$", "", text[a.end():b[-1].start()].rstrip())
    return state.strip(), loads_lenient(text[b[-1].start():])


def n_words(s: str) -> int:
    return len(s.split())


def check_question(q: dict, spec: QSpec, c: JevCell) -> dict:
    if not isinstance(q, dict):
        raise Reject("question not an object")
    opts = q.get("options") or q.get("choices")
    if not isinstance(opts, list) or not all(isinstance(o, str) and o.strip() for o in opts):
        raise Reject("options")
    opts = [normalize(o, c.choice_lang) for o in opts]
    n = len(opts)
    if len(set(o.casefold() for o in opts)) != n:
        raise Reject("duplicate options")
    kind = spec.kind
    if kind == "verify" and n != 2:
        raise Reject("verify needs 2 options")
    if kind == "choose" and not (3 <= n <= 12) or kind == "score" and not (3 <= n <= 7):
        raise Reject(f"{kind} option count {n}")
    if n != spec.n and abs(n - spec.n) > 1:
        raise Reject(f"option count {n} != {spec.n}")
    probs = q.get("probs")
    if not isinstance(probs, list) or len(probs) != n:
        raise Reject("probs length")
    try:
        p = [float(v) for v in probs]
    except (TypeError, ValueError):
        raise Reject("probs type")
    if any(not (0.0 <= v <= 1.0) for v in p):
        raise Reject("probs range")
    s = sum(p)
    if not (0.9 <= s <= 1.1):
        raise Reject(f"mass {s:.3f}")
    p = [max(v, 1e-4) for v in p]
    s = sum(p)
    p = [round(v / s, 5) for v in p]
    gold = q.get("gold")
    if isinstance(gold, float) and gold.is_integer():
        gold = int(gold)
    if not isinstance(gold, int) or not (0 <= gold < n):
        raise Reject("gold")
    top = max(range(n), key=p.__getitem__)
    tie = False
    if gold != top:
        if p[top] - p[gold] > 0.05:
            raise Reject(f"gold {gold} is not argmax {top} ({p[gold]:.2f} vs {p[top]:.2f})")
        tie = True                                  # near-tie: keep the teacher's gold, flag it
    none_idx = None
    if spec.none:
        want = (NONE_TEXT if spec.none == "none" else INSUFF_TEXT)[c.choice_lang]
        if normalize(want, c.choice_lang).casefold() in opts[-1].casefold():
            none_idx = n - 1
    question = q.get("question")
    if not isinstance(question, str) or not question.strip():
        raise Reject("empty question")
    rec = {"kind": kind, "question": normalize(question, c.state_lang), "choices": opts, "gold": gold, "probs": p,
           "none_idx": none_idx, "uncertain_asked": spec.uncertain}
    if tie:
        rec["gold_tie"] = True
    return rec


def parse_jev(text: str, c: JevCell) -> tuple[str, list[dict], list[str]]:
    state, js = split_reply(text)
    state = normalize(state, c.state_lang)
    w = n_words(state)
    if not (MIN_WORDS <= w <= MAX_WORDS):
        raise Reject(f"state length {w} words")
    qs = js.get("questions") if isinstance(js, dict) else None
    if not isinstance(qs, list) or not qs:
        raise Reject("no questions list")
    out, rejects, seen_q = [], [], set()
    for k, q in enumerate(qs[: len(c.qs)]):
        try:
            r = check_question(q, c.qs[k], c)
            if r["question"].casefold() in seen_q:
                raise Reject("repeated question")
            seen_q.add(r["question"].casefold())
            out.append(r)
        except (Reject, TypeError, ValueError, AttributeError) as e:
            rejects.append(f"q{k}: {e}")
    if len(qs) < len(c.qs):
        rejects.append(f"only {len(qs)}/{len(c.qs)} questions")
    return state, out, rejects


def example_id(state: str, question: str, choices: list[str]) -> str:
    key = state + "\x1f" + question + "\x1f" + "\x1f".join(choices)
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def load_done(path: Path) -> tuple[set[int], set[str], set[str], int]:
    cells, ids, states, rows = set(), set(), set(), 0
    if path.exists():
        for line in path.open():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            cells.add(r["cell"]); ids.add(r["id"])
            if r.get("source") == "jevgen":
                rows += 1
                states.add(hashlib.sha1(r["state"].encode()).hexdigest())
    return cells, ids, states, rows


# ---------------------------------------------------------------- run ----

async def run(args) -> None:
    out = Path(args.out or f"data/teacher/jevgen-{args.model or Teacher().model}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    rej_log = out.with_suffix(".rejects.jsonl")
    done_cells, seen, seen_states, rows0 = load_done(out)
    mean_q = sum(a * b for a, b in zip(*N_Q))
    n_calls = args.calls or math.ceil(args.n / (mean_q * 0.92))    # ~8 % of questions rejected in pilots
    todo = [i for i in range(args.offset, args.offset + n_calls) if i not in done_cells]
    print(f"{len(done_cells)} cells done, {rows0} rows on disk; {len(todo)} calls to go (target ~{args.n} rows) → {out}", flush=True)
    if not todo:
        return
    t0 = time.time()
    st = {"calls": 0, "failed": 0, "rows": 0, "rejects": 0, "dupes": 0, "tok": 0}
    fo, fr = out.open("a"), rej_log.open("a")

    async with Teacher(model=args.model, concurrency=args.streams) as t:
        async def one(i: int):
            c = cell(i, args.seed)
            r = None
            try:
                r = await t.chat(jev_prompt(c), max_tokens=args.max_tokens, temperature=args.temperature,
                                 thinking=args.thinking, tag=f"jev:{c.domain}")
                if r.finish_reason == "length":
                    raise Reject("truncated (length)")
                state, recs, rejects = parse_jev(r.text, c)
            except (TeacherError, Reject, ValueError) as e:
                st["failed"] += 1
                rec = {"cell": i, "error": str(e)[:300]}
                if r is not None and isinstance(e, Reject):
                    rec["head"], rec["tail"] = r.text[:200], r.text[-200:]
                fr.write(json.dumps(rec, ensure_ascii=False) + "\n"); fr.flush()
                if isinstance(e, Reject):   # a bad reply is final for this cell; a transport failure is retried on resume
                    fo.write(json.dumps({"id": f"empty-{i}", "cell": i, "source": "empty"}) + "\n"); fo.flush()
                return
            st["tok"] += int(r.usage.get("completion_tokens", 0))
            for why in rejects:
                fr.write(json.dumps({"cell": i, "reject": why}) + "\n")
            st["rejects"] += len(rejects)
            sh = hashlib.sha1(state.encode()).hexdigest()
            wrote = 0
            if sh in seen_states:
                st["dupes"] += len(recs); recs = []
            seen_states.add(sh)
            group = f"jev-{args.seed}-{i}"
            for k, rec in enumerate(recs):
                rid = example_id(state, rec["question"], rec["choices"])
                if rid in seen:
                    st["dupes"] += 1
                    continue
                seen.add(rid)
                row = {"kind": rec.pop("kind"), "domain": c.domain, "state_lang": c.state_lang, "choice_lang": c.choice_lang,
                       "state": state, **rec, "id": rid, "group": group, "q_index": k,
                       "source": "jevgen", "model": t.model, "cell": i, "scenario": c.scenario, "format": c.fmt}
                fo.write(json.dumps(row, ensure_ascii=False) + "\n")
                wrote += 1
            st["rows"] += wrote
            if not wrote:
                fo.write(json.dumps({"id": f"empty-{i}", "cell": i, "source": "empty"}) + "\n")
            fo.flush(); fr.flush()
            st["calls"] += 1
            if st["calls"] % args.log_every == 0:
                el = time.time() - t0
                rate = st["rows"] / el
                left = (len(todo) - st["calls"] - st["failed"]) * st["rows"] / max(st["calls"], 1)
                print(f"  {st['calls']}/{len(todo)} calls  {st['rows']} rows ({rows0 + st['rows']} total)  {rate*3600:.0f} rows/h  "
                      f"{st['tok']/el:.0f} tok/s  rej={st['rejects']} dup={st['dupes']} fail={st['failed']}  "
                      f"eta={left/max(rate,1e-9)/3600:.1f}h", flush=True)

        await asyncio.gather(*(one(i) for i in todo))
    fo.close(); fr.close()
    print(f"done: {st}  {time.time()-t0:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="target number of question rows (calls = n / ~3.7)")
    ap.add_argument("--calls", type=int, default=0, help="override: exact number of calls (states)")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--streams", type=int, default=4, help="concurrent teacher requests (keep ≤ 4: heat)")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--thinking", choices=["on", "off", "default"], default="off")
    ap.add_argument("--log-every", type=int, default=10)
    args = ap.parse_args()
    args.thinking = {"on": True, "off": False, "default": None}[args.thinking]
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
