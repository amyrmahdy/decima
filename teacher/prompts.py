"""Prompt templates and reply validation shared by the generators and the throughput bench.

Two call shapes, because choice count changes what is cheap to ask for:

  inline   (≤ 12 choices)  every example carries its own choice list and a full
                           probability vector
  catalog  (≥ 20 choices)  one shared choice set per call ("the intent catalogue"),
                           then N states labeled against it with a sparse top-k
                           distribution — the remaining mass is spread over the rest.
                           This is also the realistic shape: fixed choice set, many states.

Everything the teacher returns is validated here; a distribution that is not a proper
probability vector is rejected rather than repaired beyond renormalising a ±2 % drift.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from decima.normalize import normalize

LANG_NAME = {"en": "English", "fa": "Persian (Farsi)", "ar": "Arabic", "ru": "Russian"}

NONE_TEXT = {
    "en": "none of the above",
    "fa": "هیچ‌کدام از موارد بالا",
    "ar": "لا شيء مما سبق",
    "ru": "ничего из перечисленного",
}

DOMAINS = {
    "support": "customer support tickets and chat transcripts (SaaS, telecom, utilities)",
    "ops": "operations and request routing: which team, queue, runbook or service owner handles an incident, alert or ticket",
    "security": "security and content moderation: abuse reports, phishing, policy violations, suspicious activity",
    "legal": "legal and compliance: contract clauses, regulatory questions, consent, data-protection requests",
    "finance": "personal and business finance: banking requests, transactions, invoices, expense claims, fraud",
    "hr": "HR and people operations: leave, payroll, onboarding, complaints, performance",
    "healthcare": "healthcare intake and triage: patient messages, symptoms, appointment requests, prescriptions",
    "ecommerce": "e-commerce: orders, returns, shipping, product questions, reviews, marketplace disputes",
    "agent_tools": "an AI agent choosing which tool / function / API to call next given the conversation and available tools",
    "email": "email triage: inbox messages to be categorised, prioritised or routed",
    "docqa": "document QA: a paragraph from a manual, policy, report or article and a question about what it states",
    "travel": "travel and booking: flights, hotels, changes, cancellations, loyalty programmes",
    "it_helpdesk": "IT helpdesk: access requests, device problems, software issues, outages",
    "government": "government and public services: permits, taxes, civil registry, benefits",
    "education": "education: student requests, course administration, assessments, academic integrity",
    "logistics": "logistics and delivery: shipments, warehouse events, driver messages, exceptions",
    "insurance": "insurance: claims, policy questions, underwriting, coverage disputes",
    "social": "social media and community: posts, comments, reports, engagement decisions",
    "realestate": "real estate and property management: tenants, listings, maintenance, leases",
    "dev": "software engineering: bug reports, log excerpts, code review comments, CI failures",
}

KIND_DESC = {
    "choose": "CHOOSE — exactly one choice is the correct answer; the choices are mutually exclusive categories, intents, destinations or actions.",
    "score": "SCORE — the choices are ORDERED LEVELS of one quantity, listed from lowest to highest (e.g. urgency, risk, sentiment, severity, quality). The choice texts must name the levels. Probability mass should be concentrated on adjacent levels: an expert unsure between 'high' and 'very high' does not put mass on 'very low'.",
    "verify": "VERIFY — the question states a PROPOSITION about the state and asks whether it holds. Exactly two choices, phrased in the choice language: the first affirms (e.g. 'yes, …'), the second denies (e.g. 'no, …'). Mix cases where it is clearly true, clearly false, and genuinely uncertain from the state.",
    "rank": "RANK — every choice is judged INDEPENDENTLY against the state: for each choice, the probability that it applies / is relevant / should be selected. Several may apply, none may apply; the probabilities do NOT sum to 1.",
}

_STYLE = """States must be realistic and concrete: real-sounding names, amounts, dates, product names, error text. Vary length (1 to 6 sentences) and form: a ticket, a chat excerpt, log lines, an email, a document paragraph, an agent's tool-call context, a form submission. Vary difficulty: some obvious, many that require reading carefully, some genuinely ambiguous. Never restate the answer or the choice text inside the state. Distributions must be CALIBRATED: concentrate mass when the state is clear, spread it across the plausible choices when it is ambiguous; never assign exactly 0 to a choice that a reasonable reader could pick."""

SYSTEM = (
    "You produce training data for Decima, a small decision model. Decima receives a STATE (text describing a "
    "situation), a QUESTION and a list of CHOICES written in natural language, and must output a calibrated "
    "probability distribution over the choices. You are the expert labeler: you invent realistic examples and "
    "assign the probabilities a careful domain expert would. Answer with JSON only."
)


@dataclass(frozen=True)
class Cell:
    """One point of the generation grid — everything that decides what a call asks for."""

    domain: str
    kind: str
    n_choices: int
    state_lang: str
    choice_lang: str
    none: bool          # append a "none of the above" choice
    n_examples: int
    seed: int

    @property
    def catalog(self) -> bool:
        return self.n_choices >= 20


def _none_slots(c: Cell) -> str:
    """Which examples (1-based) have 'none of the above' as gold: ~a third, seeded, stated explicitly
    because 'roughly a third' produced 1 in 40 in the samples."""
    import random

    rng = random.Random(c.seed)
    k = max(1, round(c.n_examples / 3))
    return ", ".join(str(i) for i in sorted(rng.sample(range(1, c.n_examples + 1), k)))


def generate_prompt(c: Cell) -> list[dict]:
    lang_line = (
        f"State language: {LANG_NAME[c.state_lang]}. Question and choices language: {LANG_NAME[c.choice_lang]}."
        if c.state_lang != c.choice_lang
        else f"Language of state, question and choices: {LANG_NAME[c.state_lang]}."
    )
    none_line = (
        f"The LAST choice must be \"{NONE_TEXT[c.choice_lang]}\". In examples {_none_slots(c)} (1-based) it is the "
        "correct answer: the state is a realistic message from the same domain that matches NONE of the other choices "
        "(a different request, off-topic, or too vague to place). In the other examples it gets a small probability."
        if c.none else ""
    )
    ambiguity = (
        f"At least {max(1, c.n_examples // 3)} of the examples must be genuinely ambiguous: the top probability at most 0.6, "
        "with the mass shared between two or three plausible choices."
        if c.kind != "rank" else "Several examples must have two or more choices that apply, and at least one where none applies."
    )
    kind_line = KIND_DESC[c.kind]
    if not c.catalog:
        body = f"""Create {c.n_examples} examples.
Domain: {DOMAINS[c.domain]}.
Kind: {kind_line}
Each example has exactly {c.n_choices} choices. {none_line}
{lang_line}
{_STYLE}
{ambiguity}
Vary the questions and choice sets across examples (variation seed {c.seed}).
Output: {{"examples": [{{"state": "...", "question": "...", "choices": ["...", ...], "gold": <0-based index of the correct choice>, "probs": [<{c.n_choices} numbers>]}}, ...]}}
For choose/score/verify the probs sum to 1; for rank each prob is independent in [0, 1]."""
    else:
        body = f"""First create ONE shared catalogue of exactly {c.n_choices} distinct choices for this domain and kind, then {c.n_examples} states labeled against that catalogue.
Domain: {DOMAINS[c.domain]}.
Kind: {kind_line}
{none_line}
{lang_line}
{_STYLE}
{ambiguity}
The catalogue entries are short natural-language descriptions (2–8 words), specific enough to be told apart, and include near-neighbours that make the decision hard. Use variation seed {c.seed} to pick a fresh sub-area of the domain.
For each state give "gold" (0-based index of the correct catalogue entry) and "top": the 3 to 6 most plausible entries as [index, probability] pairs whose probabilities sum to at most 1 (remaining mass, if any, is spread over the other entries). For rank kind, "top" lists every entry with probability ≥ 0.05 and each probability is independent.
Output: {{"question": "...", "choices": [<{c.n_choices} strings>], "examples": [{{"state": "...", "gold": <int>, "top": [[<int>, <float>], ...]}}, ...]}}"""
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": body}]


def label_prompt(items: list[dict], kind: str, choice_lang: str) -> list[dict]:
    """Ask for distributions over EXISTING states with a given question + choice set.

    `items` = [{"id": ..., "state": ...}]; one shared question/choices per call keeps the
    prompt short and mirrors inference (one catalogue, many states)."""
    q, choices = items[0]["question"], items[0]["choices"]
    listing = "\n".join(f"{i}: {c}" for i, c in enumerate(choices))
    states = "\n\n".join(f"[{i}] {it['state']}" for i, it in enumerate(items))
    rule = (
        "each probability is independent in [0, 1] (several choices may apply)" if kind == "rank"
        else "probabilities over the choices sum to 1"
    )
    body = f"""Assign a calibrated probability distribution to each state below.
Kind: {KIND_DESC[kind]}
Question: {q}
Choices ({LANG_NAME[choice_lang]}):
{listing}

Rules: {rule}; concentrate mass when the state is clear, spread it over the plausible choices when it is ambiguous; give "gold" as the single best choice index. If a state matches none of the choices well and there is a "none of the above" entry, use it.
States:
{states}

Output: {{"labels": [{{"i": <state number>, "gold": <int>, "top": [[<choice index>, <prob>], ...]}}, ...]}} — "top" holds the 3 to 6 most plausible choices (all of them if fewer); list every state exactly once."""
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": body}]


# ---------------------------------------------------------------- validation ----

class Reject(ValueError):
    pass


def densify(top: list, n: int, kind: str) -> list[float]:
    """Sparse [index, prob] pairs → full vector. Leftover mass (for non-rank) goes uniformly to
    the unlisted choices, floored at 1e-4 each so no choice is ever exactly impossible."""
    p = [0.0] * n
    seen = set()
    for pair in top:
        if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
            raise Reject("bad top pair")
        i, v = int(pair[0]), float(pair[1])
        if not (0 <= i < n) or i in seen or not (0.0 <= v <= 1.0):
            raise Reject("bad top index/prob")
        seen.add(i)
        p[i] = v
    if kind == "rank":
        return p
    s = sum(p)
    if s > 1.02 or s < 0.5:
        raise Reject(f"top mass {s:.2f}")
    rest = [i for i in range(n) if i not in seen]
    left = max(0.0, 1.0 - s)
    floor = 1e-4
    if rest:
        share = max(floor, left / len(rest))
        for i in rest:
            p[i] = share
    return _renorm(p)


def _renorm(p: list[float]) -> list[float]:
    s = sum(p)
    if not (0.9 <= s <= 1.1):
        raise Reject(f"mass {s:.3f}")
    return [v / s for v in p]


def check_probs(probs, n: int, kind: str) -> list[float]:
    if not isinstance(probs, list) or len(probs) != n:
        raise Reject("probs length")
    try:
        p = [float(v) for v in probs]
    except (TypeError, ValueError):
        raise Reject("probs type")
    if any(v < 0 or v > 1 or v != v for v in p):
        raise Reject("probs range")
    if kind == "rank":
        return p
    return _renorm([max(v, 1e-4) for v in p])   # no choice is ever exactly impossible


def _clean(s, lang: str) -> str:
    if not isinstance(s, str) or not s.strip():
        raise Reject("empty text")
    return normalize(s, lang)


def finish_example(state, question, choices, gold, probs, c: Cell, extra: dict | None = None) -> dict:
    """Normalise text per language, validate, and shape the stored record."""
    n = c.n_choices
    choices = [_clean(x, c.choice_lang) for x in choices]
    if len(choices) != n or len(set(choices)) != n:
        raise Reject("choice count / duplicates")
    if not isinstance(gold, int) or not (0 <= gold < n):
        raise Reject("gold")
    rec = {
        "kind": c.kind, "domain": c.domain,
        "state_lang": c.state_lang, "choice_lang": c.choice_lang,
        "state": _clean(state, c.state_lang), "question": _clean(question, c.choice_lang),
        "choices": choices, "gold": gold, "probs": [round(v, 5) for v in probs],
        "none_idx": n - 1 if c.none else None,
    }
    if extra:
        rec.update(extra)
    return rec


def parse_generate(reply_json: dict, c: Cell) -> tuple[list[dict], list[str]]:
    """Reply → (valid records, reject reasons). Partial credit: one bad example does not sink the call."""
    out, rejects = [], []
    exs = reply_json.get("examples")
    if not isinstance(exs, list):
        return [], ["no examples list"]
    if c.catalog:
        q, ch = reply_json.get("question"), reply_json.get("choices")
        if not isinstance(ch, list):
            return [], ["no catalogue"]
        if not (0.8 * c.n_choices <= len(ch) <= 1.25 * c.n_choices):
            return [], [f"catalogue size {len(ch)} != {c.n_choices}"]
        if len(ch) != c.n_choices:   # a 97- or 104-entry catalogue is still a fine decision; keep the real size
            c = Cell(**{**c.__dict__, "n_choices": len(ch)})
        for e in exs:
            try:
                p = densify(e.get("top", []), c.n_choices, c.kind)
                out.append(finish_example(e.get("state"), q, ch, e.get("gold"), p, c))
            except (Reject, TypeError, ValueError, AttributeError) as err:
                rejects.append(str(err))
    else:
        for e in exs:
            try:
                p = check_probs(e.get("probs"), c.n_choices, c.kind)
                out.append(finish_example(e.get("state"), e.get("question"), e.get("choices"), e.get("gold"), p, c))
            except (Reject, TypeError, ValueError, AttributeError) as err:
                rejects.append(str(err))
    return out, rejects


def parse_label(reply_json: dict, items: list[dict], kind: str) -> tuple[list[dict], list[str]]:
    out, rejects = [], []
    labels = reply_json.get("labels")
    if not isinstance(labels, list):
        return [], ["no labels list"]
    n = len(items[0]["choices"])
    seen = set()
    for lab in labels:
        try:
            i = int(lab["i"])
            if not (0 <= i < len(items)) or i in seen:
                raise Reject("state index")
            seen.add(i)
            p = densify(lab.get("top", []), n, kind)
            gold = lab.get("gold")
            if not isinstance(gold, int) or not (0 <= gold < n):
                raise Reject("gold")
            out.append({**items[i], "gold": gold, "probs": [round(v, 5) for v in p]})
        except (Reject, TypeError, ValueError, KeyError) as err:
            rejects.append(str(err))
    return out, rejects


def loads_lenient(text: str) -> dict:
    """JSON mode is not always honoured by every model; strip a fence and trailing prose."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t[3:]
        t = t.rsplit("```", 1)[0]
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        a, b = t.find("{"), t.rfind("}")
        if a < 0 or b < 0:
            raise
        return json.loads(t[a : b + 1])


def questions_prompt(desc: str, sample_states: list[str], state_lang: str, choice_lang: str, n: int, catalogue: list[str] | None) -> list[dict]:
    """Ask the teacher to invent `n` question + choice sets a system might ask about these
    states — different from the source's own label set, so the same state is seen under
    several decisions. A subset of the real catalogue (plus none-of-the-above) is added by
    the caller, not here."""
    shown = "\n".join(f"- {s[:300]}" for s in sample_states[:8])
    cat = f"\nThe source's own label set (do NOT reuse it as-is): {', '.join(catalogue[:40])}" if catalogue else ""
    body = f"""Below are sample states from a source described as: {desc}. State language: {LANG_NAME[state_lang]}.{cat}

{shown}

Invent {n} DIFFERENT decisions a software system might need to make about states like these, written in {LANG_NAME[choice_lang]}. Cover all four kinds:
- choose: 3–12 mutually exclusive options (department, action, category, priority queue…); include a "{NONE_TEXT[choice_lang]}" option in some of them as the last choice
- score: 3–7 ORDERED levels of one quantity (urgency, risk, sentiment, complexity, confidence…), lowest to highest
- verify: a proposition about the state with exactly two choices, the first affirming, the second denying
- rank: 4–10 options judged independently (which tags apply, which follow-up actions are appropriate, which teams should be notified…)
Make the questions genuinely useful and answerable from the state alone.
Output: {{"questions": [{{"kind": "choose|score|verify|rank", "question": "...", "choices": ["...", ...]}}, ...]}}"""
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": body}]
