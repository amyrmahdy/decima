"""Generator 4: System One (Jev-schema) tasks — one fixed question, many short states.

    uv run python -m teacher.generate_s1 --calls 4 --streams 2 --out /tmp/s1.jsonl          # smoke test
    setsid nohup uv run python -m teacher.generate_s1 --calls 5000 --streams 4 \
        --out data/teacher/s1-rande-fast-local.jsonl > runs/logs/s1gen.log 2>&1 &

Why: TypeSafe's wire format (shared by Kev, Ollaya and the community benchmarks) asks three kinds of
question, and Decima v1.0 was trained on only one of them in that form:
  choice — `criteria` maps a short key to a one-line description of when it applies;
  noul   — a *statement* (not a question) to judge true or false for the state, often a policy rule
           with its own thresholds; optional criteria describe what true / false mean;
  score  — 2–10 ordered levels, each a short description, lowest → highest.
Deployment looks like one fixed question and a stream of states, so one call = one task + 10 states.

Rows are stored in the raw Jev schema (one row per state, a shared `group` per task) so that the
training renderings can be varied later (teacher/render_s1.py) without regenerating anything.

Grid (seeded; call i always asks for the same cell, so runs resume by cell id):
  type choice 40 / noul 35 / score 25; choice 2–10 options (30 % end in a catch-all key),
  score 2–7 levels; state language EN 75 / FA 10 / AR 8 / RU 7, and for non-EN states the
  question is in English half the time; domains from DOMAINS below (written for this generator,
  nothing taken from an evaluation set — the mix step decontaminates against every eval item).
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
from dataclasses import dataclass
from pathlib import Path

from decima.normalize import normalize
from teacher.client import Teacher, TeacherError
from teacher.prompts import LANG_NAME, loads_lenient

TYPE_W = {"choice": 0.40, "noul": 0.35, "score": 0.25}
CHOICE_N = ([2, 3, 4, 5, 6, 7, 8, 10], [0.06, 0.14, 0.20, 0.20, 0.15, 0.11, 0.08, 0.06])
SCORE_N = ([2, 3, 4, 5, 6, 7], [0.06, 0.34, 0.24, 0.24, 0.04, 0.08])
CATCH_ALL = 0.30                 # choice tasks whose last key is a catch-all ("other", "none", …)
LANG_W = {"en": 0.75, "fa": 0.10, "ar": 0.08, "ru": 0.07}
ENGLISH_QUESTION = 0.50          # non-EN states whose question / criteria are in English
N_STATES = 10

# Business and everyday decision settings, written for this generator from general knowledge.
DOMAINS = [
    "SaaS customer support inbox", "e-commerce order and returns desk", "retail bank customer messages",
    "credit-card transaction monitoring", "insurance claims intake", "health-clinic front desk (routing and urgency only, never diagnosis)",
    "veterinary clinic reception (urgency only)", "pharmacy customer service (no medical advice)", "HR helpdesk for employees",
    "recruiting pipeline (screening notes and candidate emails)", "legal intake for a small law firm", "contract review for procurement",
    "real-estate agency inquiries", "property-management maintenance requests", "hotel guest messages", "airline passenger support",
    "food-delivery app support", "restaurant reservations and reviews", "logistics and parcel tracking exceptions",
    "ride-hailing driver and rider reports", "telecom and internet provider support", "utility company (power, water) customer service",
    "city 311 service requests", "school and university administration emails", "online course learner questions",
    "game studio player reports and moderation", "social-media content moderation queue", "marketplace listing review (policy compliance)",
    "advertising policy review", "app-store review triage", "developer platform (API errors and quota questions)",
    "DevOps on-call alerts and incident chat", "security operations (phishing reports, suspicious logins)", "IT helpdesk tickets",
    "privacy and data-subject requests", "accounts payable (invoices and vendor emails)", "expense-report review", "tax-preparation service clients",
    "nonprofit donor and volunteer messages", "event ticketing support", "automotive dealership and repair shop", "manufacturing quality reports",
    "agriculture co-op member requests", "energy-efficiency home audits", "fitness club memberships", "pet-sitting and grooming bookings",
    "news-site reader comments", "customer reviews for a consumer electronics brand", "B2B sales inbound leads", "subscription box service",
    "public library patron questions", "coworking space members", "moving company quotes and complaints", "home-appliance warranty desk",
    "travel agency bookings", "wedding and event planners", "freelance marketplace disputes", "crypto exchange support (no trading advice)",
    "email triage for a busy executive", "meeting notes and action items in a product team", "code review comments", "documentation feedback widget",
]

TYPE_RULE = {
    "choice": (
        "a CHOICE task: route or classify each state into exactly {n} mutually exclusive options. Give each option a short "
        "snake_case key (1–3 words) and a one-line description of when it applies, like a config file a developer wrote."
        "{catch}"
    ),
    "noul": (
        "a NOUL task: ONE statement (a declarative proposition, NOT a question) that software must judge true or false for "
        "each state. Make it specific: a property of the message, a policy condition or an eligibility rule; if it needs a "
        "rule or threshold (days, amounts, counts, plan names), write that rule into the statement itself so it can be "
        "decided from the state plus the statement alone. Optionally add \"criteria\" with one-line descriptions of what makes "
        "it true and false."
    ),
    "score": (
        "a SCORE task: one quantity rated on exactly {n} ORDERED levels, listed lowest → highest; each level is a short "
        "description (not just a number), e.g. how urgent, how frustrated, how risky, how complete, how likely to churn."
    ),
}


@dataclass
class S1Cell:
    i: int
    type: str
    n: int
    catch_all: bool
    domain: str
    state_lang: str
    q_lang: str


def _pick(rng: random.Random, w: dict):
    return rng.choices(list(w), weights=list(w.values()))[0]


def cell(i: int, seed: int = 0) -> S1Cell:
    rng = random.Random(f"s1:{seed}:{i}")
    t = _pick(rng, TYPE_W)
    n = rng.choices(*CHOICE_N)[0] if t == "choice" else rng.choices(*SCORE_N)[0] if t == "score" else 2
    sl = _pick(rng, LANG_W)
    ql = "en" if sl != "en" and rng.random() < ENGLISH_QUESTION else sl
    return S1Cell(i=i, type=t, n=n, catch_all=t == "choice" and n >= 3 and rng.random() < CATCH_ALL,
                  domain=rng.choice(DOMAINS), state_lang=sl, q_lang=ql)


SYSTEM = (
    "You produce training data for a small decision model used inside business software. It receives a STATE (the text a "
    "system has in front of it) and a typed QUESTION, and outputs calibrated probabilities. You are the expert: you design "
    "realistic decision tasks and label real-looking inputs the way a careful senior practitioner would, with honest uncertainty."
)

LABEL_RULE = {
    "choice": '"probs": an object mapping EVERY option key to its probability (sum 1)',
    "noul": '"p_true": the probability that the statement is true for this state',
    "score": '"probs": a list with one probability per level, lowest → highest (sum 1); mass on adjacent levels only',
}

SHAPE = {
    "choice": '{"instructions": "...", "criteria": {"key_one": "when it applies", ...}}',
    "noul": '{"instructions": "<the statement>", "criteria": {"true": "...", "false": "..."}}   (criteria optional)',
    "score": '{"instructions": "...", "criteria": ["lowest level description", ..., "highest level description"]}',
}


def s1_prompt(c: S1Cell) -> list[dict]:
    catch = (" The LAST option is a catch-all (key like \"other\", \"general\" or \"none\") for states that fit none of the rest."
             if c.catch_all else "")
    rule = TYPE_RULE[c.type].format(n=c.n, catch=catch)
    lang, qlang = LANG_NAME[c.state_lang], LANG_NAME[c.q_lang]
    balance = ("About half of the states make the statement true and half false; several false ones must be near-misses "
               "(almost satisfy the rule but fail one condition), and some true ones must satisfy it in a non-obvious way."
               if c.type == "noul" else
               "Cover every option or level at least once; include near-confusions between neighbouring options or levels.")
    body = f"""Design ONE decision task and label {N_STATES} states for it.

Setting: {c.domain}.
Task: {rule}
Language of the question text and criteria: {qlang}. Language of the states: {lang}.

States: {N_STATES} different, realistic inputs this software would really receive (messages, tickets, reviews, log lines,
form entries, short emails, chat turns). Vary the length from one short line to about 120 words, the tone, the writer
and the formality. Include hard cases: sarcasm, negation, mixed signals, irrelevant details, typos, a request that only
looks like another category. {balance} Do NOT copy wording of the criteria into the states. 2–3 states should be genuinely
ambiguous (top probability 0.4–0.65); label the others with the confidence the evidence warrants (can be high).
Variation seed {c.i}; invent fresh names, products and numbers.

Output JSON only:
{{"question": {SHAPE[c.type]},
  "cases": [{{"state": "...", {LABEL_RULE[c.type]}}}, ... {N_STATES} cases]}}"""
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": body}]


class Reject(ValueError):
    pass


_KEY = re.compile(r"^[a-z][a-z0-9_]{0,40}$")


def _norm_probs(p: list[float]) -> list[float]:
    if any(not (0.0 <= v <= 1.0) for v in p):
        raise Reject("probs range")
    s = sum(p)
    if not (0.9 <= s <= 1.1):
        raise Reject(f"mass {s:.3f}")
    p = [max(v, 1e-4) for v in p]
    s = sum(p)
    return [round(v / s, 5) for v in p]


def parse_s1(text: str, c: S1Cell) -> tuple[dict, list[dict], list[str]]:
    js = loads_lenient(text)
    q = js.get("question") if isinstance(js, dict) else None
    if not isinstance(q, dict):
        raise Reject("no question")
    ins = q.get("instructions")
    if not isinstance(ins, str) or not ins.strip():
        raise Reject("instructions")
    ins = normalize(ins.strip(), c.q_lang)
    crit = q.get("criteria")
    if c.type == "choice":
        if not isinstance(crit, dict) or not all(isinstance(v, str) and v.strip() for v in crit.values()):
            raise Reject("choice criteria")
        crit = {str(k).strip(): normalize(v.strip(), c.q_lang) for k, v in crit.items()}
        if not all(_KEY.match(k) for k in crit):
            raise Reject(f"keys {list(crit)[:4]}")
        if abs(len(crit) - c.n) > 1 or len(crit) < 2:
            raise Reject(f"{len(crit)} options, wanted {c.n}")
    elif c.type == "score":
        if not isinstance(crit, list) or not all(isinstance(v, str) and v.strip() for v in crit):
            raise Reject("score levels")
        crit = [normalize(v.strip(), c.q_lang) for v in crit]
        if not (2 <= len(crit) <= 10) or abs(len(crit) - c.n) > 1:
            raise Reject(f"{len(crit)} levels, wanted {c.n}")
        if len({v.casefold() for v in crit}) != len(crit):
            raise Reject("duplicate levels")
    else:
        if ins.rstrip().endswith("?"):
            raise Reject("noul is a question, not a statement")
        if crit is not None:
            if not isinstance(crit, dict) or not all(isinstance(v, str) for v in crit.values()):
                crit = None
            else:
                crit = {str(k).lower(): normalize(v.strip(), c.q_lang) for k, v in crit.items() if v.strip()}
                if set(crit) != {"true", "false"}:
                    crit = None
    task = {"type": c.type, "instructions": ins, "criteria": crit}

    cases = js.get("cases")
    if not isinstance(cases, list) or not cases:
        raise Reject("no cases")
    out, rejects, seen = [], [], set()
    for k, cs in enumerate(cases[: N_STATES + 2]):
        try:
            st = cs.get("state") if isinstance(cs, dict) else None
            if not isinstance(st, str) or len(st.strip()) < 8:
                raise Reject("state")
            st = normalize(st.strip(), c.state_lang)
            if st.casefold() in seen:
                raise Reject("duplicate state")
            if c.type == "choice":
                pr = cs.get("probs")
                if not isinstance(pr, dict) or set(pr) != set(crit):
                    raise Reject("probs keys")
                p = _norm_probs([float(pr[key]) for key in crit])
                label = {"probs": dict(zip(crit, p))}
            elif c.type == "score":
                pr = cs.get("probs")
                if not isinstance(pr, list) or len(pr) != len(crit):
                    raise Reject("probs length")
                label = {"probs": _norm_probs([float(v) for v in pr])}
            else:
                v = float(cs.get("p_true"))
                if not (0.0 <= v <= 1.0):
                    raise Reject("p_true range")
                label = {"p_true": round(min(max(v, 1e-3), 1 - 1e-3), 4)}
            seen.add(st.casefold())
            out.append({"state": st, **label})
        except (Reject, TypeError, ValueError, AttributeError) as e:
            rejects.append(f"case{k}: {e}")
    if c.type == "noul" and out:
        t = sum(r["p_true"] >= 0.5 for r in out)
        if t == 0 or t == len(out):
            raise Reject(f"noul labels all one side ({t}/{len(out)} true)")
    if len(out) < N_STATES // 2:
        raise Reject(f"only {len(out)} valid cases")
    return task, out, rejects


def row_id(task: dict, state: str) -> str:
    key = json.dumps(task, sort_keys=True, ensure_ascii=False) + "\x1f" + state
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def load_done(path: Path) -> tuple[set[int], set[str], int]:
    cells, ids, rows = set(), set(), 0
    if path.exists():
        for line in path.open():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            cells.add(r["cell"]); ids.add(r["id"])
            rows += r.get("source") == "s1gen"
    return cells, ids, rows


async def run(args) -> None:
    out = Path(args.out or f"data/teacher/s1-{args.model or Teacher().model}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    rej_log = out.with_suffix(".rejects.jsonl")
    done_cells, seen, rows0 = load_done(out)
    n_calls = args.calls or math.ceil(args.n / (N_STATES * 0.9))
    todo = [i for i in range(args.offset, args.offset + n_calls) if i not in done_cells]
    print(f"{len(done_cells)} cells done, {rows0} rows on disk; {len(todo)} calls to go → {out}", flush=True)
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
                r = await t.chat(s1_prompt(c), max_tokens=args.max_tokens, temperature=args.temperature,
                                 thinking=args.thinking, json_mode=True, tag=f"s1:{c.type}")
                if r.finish_reason == "length":
                    raise Reject("truncated (length)")
                task, cases, rejects = parse_s1(r.text, c)
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
            wrote = 0
            for k, cs in enumerate(cases):
                rid = row_id(task, cs["state"])
                if rid in seen:
                    st["dupes"] += 1
                    continue
                seen.add(rid)
                row = {**task, **cs, "id": rid, "group": f"s1-{args.seed}-{i}", "case_index": k,
                       "domain": c.domain, "state_lang": c.state_lang, "q_lang": c.q_lang,
                       "source": "s1gen", "model": t.model, "cell": i}
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
    ap.add_argument("--n", type=int, default=40, help="target number of state rows (calls = n / ~9)")
    ap.add_argument("--calls", type=int, default=0, help="override: exact number of calls (tasks)")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--streams", type=int, default=4, help="concurrent teacher requests (keep ≤ 4: heat)")
    ap.add_argument("--max-tokens", type=int, default=6144)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--thinking", choices=["on", "off", "default"], default="off")
    ap.add_argument("--log-every", type=int, default=10)
    args = ap.parse_args()
    args.thinking = {"on": True, "off": False, "default": None}[args.thinking]
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
