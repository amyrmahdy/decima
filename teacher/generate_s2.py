"""Generator 5 (Decima 2): the System One corpus v2 — Jev-schema tasks shaped like real use, with
reasoning anchors, several questions per state, long states and more languages. See release/PLAN-v2.md (L1).

    uv run python -m teacher.generate_s2 --calls 6 --streams 3 --out /tmp/s2.jsonl            # smoke test
    setsid nohup uv run python -m teacher.generate_s2 --calls 30000 --streams 8 \
        --out data/teacher/s2-rande-fast-local.jsonl > runs/logs/s2gen.log 2>&1 &

Differences from generate_s1:
  * clusters weighted like deployment (ops/enterprise, security/DevOps/compliance, safety/moderation,
    linguistic/content, triage/services) plus a REASONING cluster (~28 %): statements and scales whose
    answer needs a rule applied to facts in the state — numeric thresholds, dates relative to a date
    given in the state, two conditions, exceptions ("unless …"), negation, two-hop (a policy excerpt in
    the state + the statement), unit/number comparisons — with near-miss false cases;
  * 1–3 questions per task (mixed types) labelled on the same states → more labels per teacher token;
  * every question gets a short snake_case `id` (TypeSafe users name questions; the id stands in for
    missing instructions at inference);
  * long-state calls (~18 %): 3 states of 250–900 words (logs, threads, contracts, tickets with history);
  * languages EN 70 / FA 8 / AR 5 / RU 5 / ES 3 / DE 3 / FR 2 / TR 2 / ZH 2, question in English for
    half of the non-English states.
Rows: one per (state, question), raw Jev schema, shared `group` (task) and `sgroup` (state).
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
from teacher.generate_s1 import DOMAINS as S1_DOMAINS, Reject, _norm_probs, _KEY
from teacher.prompts import LANG_NAME as _LN, loads_lenient

LANG_NAME = {**_LN, "es": "Spanish", "de": "German", "fr": "French", "tr": "Turkish", "zh": "Chinese (Simplified)"}
LANG_W = {"en": 0.70, "fa": 0.08, "ar": 0.05, "ru": 0.05, "es": 0.03, "de": 0.03, "fr": 0.02, "tr": 0.02, "zh": 0.02}
ENGLISH_QUESTION = 0.5
TYPE_W = {"choice": 0.38, "noul": 0.37, "score": 0.25}
CHOICE_N = ([2, 3, 4, 5, 6, 7, 8, 10, 12, 16], [0.06, 0.13, 0.18, 0.18, 0.14, 0.10, 0.08, 0.06, 0.04, 0.03])
SCORE_N = ([2, 3, 4, 5, 6, 7, 10], [0.06, 0.32, 0.22, 0.24, 0.05, 0.07, 0.04])
NQ = ([1, 2, 3], [0.55, 0.28, 0.17])
LONG = 0.18

CLUSTERS: dict[str, tuple[float, list[str]]] = {
    "ops_enterprise": (0.20, [
        "procurement and purchase approvals", "invoice and accounts-payable exceptions", "IT asset and access requests",
        "HR leave and policy questions", "sales pipeline and CRM hygiene", "customer success and churn signals",
        "contract renewals and SLAs", "logistics and shipment exceptions", "inventory and supply planning",
        "facilities and workplace requests", "project status updates and blockers", "meeting notes and action items"]),
    "security_devops_compliance": (0.17, [
        "on-call alerts and incident triage", "CI/CD pipeline failures", "cloud cost anomalies", "access reviews and permissions",
        "phishing and suspicious-login reports", "vulnerability and dependency advisories", "data-retention and GDPR requests",
        "SOC 2 / ISO audit evidence", "change-management tickets", "log lines from production services", "API error reports"]),
    "safety_moderation": (0.12, [
        "user-generated comments", "marketplace listings", "ad creatives and landing pages", "dating-app profiles and messages",
        "gaming chat", "app-store reviews", "forum posts with links", "creator content reports"]),
    "linguistic_content": (0.10, [
        "tone and intent of emails", "sentiment and emotion in reviews", "summary faithfulness checks", "document type and section labelling",
        "language register and politeness", "claims vs opinions", "question type and answerability", "translation quality notes"]),
    "triage_services": (0.13, [
        "customer support tickets", "clinic front-desk routing (urgency only, never diagnosis)", "veterinary reception (urgency only)",
        "insurance claims intake", "city service requests", "utility outages", "travel disruptions", "banking customer messages",
        "telecom support", "property maintenance requests"]),
    "reasoning": (0.28, ["policy and eligibility rules", "deadlines and date arithmetic", "amount and quota thresholds",
        "multi-condition approvals", "exceptions and overrides", "two-hop rules (policy text + case facts)", "unit and number comparisons",
        "counting items or events in the text", "ordering and comparison of values", "negation and scope"]),
}
EXTRA_DOMAINS = S1_DOMAINS   # the s1 everyday settings, mixed into the non-reasoning clusters

REASONING_RULE = (
    "This is a REASONING task. The answer must require applying an explicit rule to facts in the state: numeric thresholds, "
    "dates relative to a date stated in the state, two conditions (and/or), an exception (\"unless …\"), negation, comparing "
    "numbers or units, counting, or two hops (a short policy excerpt inside the state + the facts). Put the rule in the "
    "question (statement, instructions or criteria) or as a policy excerpt inside each state. Include near-miss cases that "
    "fail by one condition, and correct cases that pass in a non-obvious way. Keep the arithmetic easy but exact."
)

TYPE_RULE = {
    "choice": "CHOICE with exactly {n} mutually exclusive options: short snake_case keys, each with a one-line description of when it applies.{catch}",
    "noul": "NOUL: ONE declarative statement (NOT a question) to judge true/false for each state; optionally \"criteria\": {{\"true\": …, \"false\": …}}.",
    "score": "SCORE with exactly {n} ORDERED levels, lowest → highest, each a short description (may include numeric ranges).",
}
SHAPE = {
    "choice": '{"id": "snake_case_name", "type": "choice", "instructions": "...", "criteria": {"key": "when it applies", ...}}',
    "noul": '{"id": "snake_case_name", "type": "noul", "instructions": "<statement>", "criteria": {"true": "...", "false": "..."}}',
    "score": '{"id": "snake_case_name", "type": "score", "instructions": "...", "criteria": ["lowest", ..., "highest"]}',
}
LABEL = {"choice": "{key: probability, …} over EVERY option key", "noul": "probability that the statement is true",
         "score": "[probability per level, lowest → highest]"}


@dataclass
class QSpec:
    type: str
    n: int
    catch_all: bool


@dataclass
class S2Cell:
    i: int
    cluster: str
    topic: str
    state_lang: str
    q_lang: str
    long: bool
    n_states: int
    qs: list[QSpec] = field(default_factory=list)


def _pick(rng: random.Random, w: dict):
    return rng.choices(list(w), weights=list(w.values()))[0]


def cell(i: int, seed: int = 0, only_en: bool = False) -> S2Cell:
    rng = random.Random(f"s2:{seed}:{i}")
    cl = rng.choices(list(CLUSTERS), weights=[v[0] for v in CLUSTERS.values()])[0]
    topics = CLUSTERS[cl][1] + ([] if cl == "reasoning" else EXTRA_DOMAINS)
    sl = _pick(rng, LANG_W)
    if only_en:                      # a second teacher whose non-English output is unreliable writes English only
        sl = "en"
    long = rng.random() < LONG
    c = S2Cell(i=i, cluster=cl, topic=rng.choice(topics), state_lang=sl,
               q_lang="en" if sl != "en" and rng.random() < ENGLISH_QUESTION else sl, long=long, n_states=3 if long else 10)
    for _ in range(rng.choices(*NQ)[0]):
        t = _pick(rng, TYPE_W)
        n = rng.choices(*CHOICE_N)[0] if t == "choice" else rng.choices(*SCORE_N)[0] if t == "score" else 2
        c.qs.append(QSpec(t, n, t == "choice" and n >= 3 and rng.random() < 0.3))
    return c


SYSTEM = (
    "You produce training data for a small decision model used inside software. It receives a STATE (the text a system has "
    "in front of it) and typed QUESTIONS, and outputs calibrated probabilities. You design realistic decision tasks and label "
    "real-looking inputs like a careful senior practitioner, with honest uncertainty."
)


def s2_prompt(c: S2Cell) -> list[dict]:
    lang, qlang = LANG_NAME[c.state_lang], LANG_NAME[c.q_lang]
    qlines = []
    for k, q in enumerate(c.qs, 1):
        catch = ' The LAST option is a catch-all ("other"/"none") for states that fit none of the rest.' if q.catch_all else ""
        qlines.append(f"Q{k}: " + TYPE_RULE[q.type].format(n=q.n, catch=catch) + f"  Shape: {SHAPE[q.type]}")
    length = ("Each state is a LONG artefact of 250–900 words (a log excerpt, an email thread, a contract clause set, a ticket "
              "with history, a policy plus a request), with realistic noise and one decisive detail that is easy to miss."
              if c.long else "Vary the length from one short line to about 120 words.")
    reasoning = REASONING_RULE if c.cluster == "reasoning" else (
        "Include hard cases: sarcasm, negation, mixed signals, irrelevant details, typos, look-alikes of another option.")
    labels = ", ".join(f'"{q_id}": {LABEL[q.type]}' for q_id, q in zip([f"<id of Q{k}>" for k in range(1, len(c.qs) + 1)], c.qs))
    body = f"""Design ONE decision setting with {len(c.qs)} question(s) and label {c.n_states} states for every question.

Setting: {c.topic} ({c.cluster.replace('_', ' ')}).
Question(s) — each about a DIFFERENT aspect of the same states:
{chr(10).join(qlines)}
Language of question texts and criteria descriptions: {qlang}. Language of the states: {lang}.
Question ids and choice keys are ALWAYS English snake_case ASCII (they are code identifiers), whatever the language.

States: {c.n_states} different, realistic inputs this software would really receive. {length} {reasoning}
Cover every option / level and both truth values across the states; 2–3 states should be genuinely ambiguous for at
least one question (top probability 0.4–0.65). Do not copy criteria wording into the states. Variation seed {c.i};
invent fresh names, products, numbers and dates.

Output JSON only:
{{"questions": [<question objects in the shapes above>],
  "cases": [{{"state": "...", "labels": {{{labels}}}}}, ... {c.n_states} cases]}}"""
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": body}]


def _norm_q(q: dict, spec: QSpec, c: S2Cell) -> dict:
    if not isinstance(q, dict) or q.get("type") != spec.type:
        raise Reject("question type")
    qid = str(q.get("id") or "").strip()
    if not _KEY.match(qid):
        raise Reject(f"id {qid!r}")
    ins = q.get("instructions")
    if not isinstance(ins, str) or not ins.strip():
        raise Reject("instructions")
    ins = normalize(ins.strip(), c.q_lang)
    crit = q.get("criteria")
    if spec.type == "choice":
        if not isinstance(crit, dict) or not all(isinstance(v, str) and v.strip() for v in crit.values()):
            raise Reject("choice criteria")
        crit = {str(k).strip(): normalize(v.strip(), c.q_lang) for k, v in crit.items()}
        if not all(_KEY.match(k) for k in crit) or len(crit) < 2 or abs(len(crit) - spec.n) > 1:
            raise Reject(f"choice keys/count {len(crit)}")
    elif spec.type == "score":
        if not isinstance(crit, list) or not all(isinstance(v, str) and v.strip() for v in crit):
            raise Reject("score levels")
        crit = [normalize(v.strip(), c.q_lang) for v in crit]
        if not (2 <= len(crit) <= 10) or abs(len(crit) - spec.n) > 1 or len({v.casefold() for v in crit}) != len(crit):
            raise Reject(f"score levels {len(crit)}")
    else:
        if ins.rstrip().endswith(("?", "؟")):
            raise Reject("noul is a question")
        if isinstance(crit, dict):
            crit = {str(k).lower(): normalize(str(v).strip(), c.q_lang) for k, v in crit.items() if str(v).strip()}
            crit = crit if set(crit) == {"true", "false"} else None
        else:
            crit = None
    return {"qid": qid, "type": spec.type, "instructions": ins, "criteria": crit}


def _label(q: dict, v) -> dict:
    if q["type"] == "choice":
        if not isinstance(v, dict) or set(v) != set(q["criteria"]):
            raise Reject("choice label keys")
        return {"probs": dict(zip(q["criteria"], _norm_probs([float(v[k]) for k in q["criteria"]])))}
    if q["type"] == "score":
        if not isinstance(v, list) or len(v) != len(q["criteria"]):
            raise Reject("score label length")
        return {"probs": _norm_probs([float(x) for x in v])}
    p = float(v["p_true"] if isinstance(v, dict) else v)
    if not 0.0 <= p <= 1.0:
        raise Reject("p range")
    return {"p_true": round(min(max(p, 1e-3), 1 - 1e-3), 4)}


def parse_s2(text: str, c: S2Cell) -> tuple[list[dict], list[dict], list[str]]:
    js = loads_lenient(text)
    qs = js.get("questions") if isinstance(js, dict) else None
    if not isinstance(qs, list) or len(qs) < len(c.qs):
        raise Reject("questions")
    questions = [_norm_q(q, spec, c) for q, spec in zip(qs, c.qs)]
    if len({q["qid"] for q in questions}) != len(questions):
        raise Reject("duplicate ids")
    cases = js.get("cases")
    if not isinstance(cases, list) or not cases:
        raise Reject("no cases")
    out, rejects, seen = [], [], set()
    lo = 120 if c.long else 1
    for k, cs in enumerate(cases[: c.n_states + 2]):
        try:
            st = cs.get("state") if isinstance(cs, dict) else None
            if not isinstance(st, str) or len(st.strip()) < 8:
                raise Reject("state")
            st = normalize(st.strip(), c.state_lang)
            if c.long and len(st.split()) < lo and c.state_lang not in ("zh",):
                raise Reject(f"long state too short ({len(st.split())} words)")
            if st.casefold() in seen:
                raise Reject("duplicate state")
            labs = cs.get("labels")
            if len(questions) == 1 and (not isinstance(labs, dict) or questions[0]["qid"] not in labs):
                labs = {questions[0]["qid"]: labs}        # a single question's label given bare
            if not isinstance(labs, dict):
                raise Reject("labels")
            row_labels = {}
            for q in questions:
                if q["qid"] in labs:
                    try:
                        row_labels[q["qid"]] = _label(q, labs[q["qid"]])
                    except (Reject, TypeError, ValueError, KeyError) as e:
                        rejects.append(f"case{k}.{q['qid']}: {e}")
            if not row_labels:
                raise Reject("no valid labels")
            seen.add(st.casefold())
            out.append({"state": st, "labels": row_labels})
        except (Reject, TypeError, ValueError, AttributeError) as e:
            rejects.append(f"case{k}: {e}")
    for q in questions:
        if q["type"] == "noul":
            vals = [r["labels"][q["qid"]]["p_true"] >= 0.5 for r in out if q["qid"] in r["labels"]]
            if vals and (all(vals) or not any(vals)) and len(vals) >= 3:
                for r in out:
                    r["labels"].pop(q["qid"], None)
                rejects.append(f"{q['qid']}: noul labels all one side")
    out = [r for r in out if r["labels"]]
    if len(out) < max(2, c.n_states // 2):
        raise Reject(f"only {len(out)} valid cases")
    return questions, out, rejects


def rid(q: dict, state: str) -> str:
    return hashlib.sha1((json.dumps(q, sort_keys=True, ensure_ascii=False) + "\x1f" + state).encode()).hexdigest()[:16]


def load_done(path: Path) -> tuple[set[int], set[str], int]:
    cells, ids, rows = set(), set(), 0
    if path.exists():
        for line in path.open():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            cells.add(r["cell"]); ids.add(r["id"])
            rows += r.get("source") == "s2gen"
    return cells, ids, rows


async def run(args) -> None:
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rej_log = out.with_suffix(".rejects.jsonl")
    done_cells, seen, rows0 = load_done(out)
    todo = [i for i in range(args.offset, args.offset + args.calls) if i not in done_cells]
    print(f"{len(done_cells)} cells done, {rows0} rows on disk; {len(todo)} calls to go → {out}", flush=True)
    if not todo:
        return
    t0 = time.time()
    st = {"calls": 0, "failed": 0, "rows": 0, "rejects": 0, "tok": 0}
    fo, fr = out.open("a"), rej_log.open("a")
    async with Teacher(model=args.model, concurrency=args.streams) as t:
        async def one(i: int):
            c = cell(i, args.seed, args.only_en)
            r = None
            try:
                r = await t.chat(s2_prompt(c), max_tokens=args.max_tokens, temperature=args.temperature,
                                 thinking=args.thinking, json_mode=True, tag=f"s2:{c.cluster}")
                if r.finish_reason == "length":
                    raise Reject("truncated (length)")
                questions, cases, rejects = parse_s2(r.text, c)
            except (TeacherError, Reject, ValueError) as e:
                st["failed"] += 1
                rec = {"cell": i, "error": str(e)[:300]}
                if r is not None and isinstance(e, Reject):
                    rec["head"], rec["tail"] = r.text[:200], r.text[-200:]
                fr.write(json.dumps(rec, ensure_ascii=False) + "\n"); fr.flush()
                if isinstance(e, Reject):
                    fo.write(json.dumps({"id": f"empty-{i}", "cell": i, "source": "empty"}) + "\n"); fo.flush()
                return
            st["tok"] += int(r.usage.get("completion_tokens", 0))
            st["rejects"] += len(rejects)
            for why in rejects:
                fr.write(json.dumps({"cell": i, "reject": why}, ensure_ascii=False) + "\n")
            wrote = 0
            for k, cs in enumerate(cases):
                for q in questions:
                    lab = cs["labels"].get(q["qid"])
                    if lab is None:
                        continue
                    task = {kk: q[kk] for kk in ("qid", "type", "instructions", "criteria")}
                    row_id = rid(task, cs["state"])
                    if row_id in seen:
                        continue
                    seen.add(row_id)
                    fo.write(json.dumps({**task, "state": cs["state"], **lab, "id": row_id,
                                         "group": f"s2-{args.seed}-{i}-{q['qid']}", "sgroup": f"s2-{args.seed}-{i}-s{k}",
                                         "cluster": c.cluster, "topic": c.topic, "long": c.long,
                                         "state_lang": c.state_lang, "q_lang": c.q_lang,
                                         "source": "s2gen", "model": t.model, "cell": i}, ensure_ascii=False) + "\n")
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
                      f"{st['tok']/el:.0f} tok/s  rej={st['rejects']} fail={st['failed']}  eta={left/max(rate,1e-9)/3600:.1f}h", flush=True)
        await asyncio.gather(*(one(i) for i in todo))
    fo.close(); fr.close()
    print(f"done: {st}  {time.time()-t0:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", type=int, default=6)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--streams", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=9000)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--thinking", choices=["on", "off", "default"], default="off")
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--only-en", action="store_true", help="English states only (for a teacher that is weak in other languages)")
    args = ap.parse_args()
    args.thinking = {"on": True, "off": False, "default": None}[args.thinking]
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
