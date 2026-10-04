"""Persian (fa) edition of jabr/classifier-benchmark v2: same tasks, same labels, Persian text.

    uv run --with httpx --with tomli-w python release/jabr-fa/translate.py --cases <cb>/cases/v2.toml --out release/jabr-fa

One call per task translates the question (instructions, criteria descriptions / levels) and every state into
Persian. Option keys, labels and the number of cases never change, so each fa case is the exact counterpart
of an English one. The translator also flags tasks whose label depends on English itself (e.g. grammar
errors), which are kept out of the main suite.

Writes:
  translations.jsonl      raw per-task translations (resumable)
  v2-fa.toml              everything Persian except option keys
  v2-fa-state.toml        Persian states, English question (a developer's English schema over Persian input)
  excluded.json           language-dependent tasks and the translator's reason
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import tomllib
from pathlib import Path

import httpx
import tomli_w

URL = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM = """You are a professional English→Persian (Farsi, Iran) translator preparing an evaluation set.
Translate naturally, the way a native Persian speaker in Iran would actually write the text (a support
message reads like a real Persian support message; an email like a Persian email). Rules:
- Keep the meaning exactly, including every fact, number, amount, date, time, quantity and hedge, so the
  correct answer stays the same. Write numbers with Western digits (0-9).
- Keep code, SQL, URLs, e-mail addresses, file names, API keys, product names and identifiers as they are.
- Localize nothing that would change the answer (keep currencies and units as given).
- Do not add, drop, explain or soften anything.
- If the correct answer to a case depends on English itself (spelling, grammar, wordplay, English reading
  level) and cannot survive translation, set "language_dependent": true for the task and say why."""


def prompt(task: dict) -> str:
    q = task["question"]
    payload = {"instructions": q["instructions"], "criteria": q.get("criteria"),
               "states": [c["state"] for c in task["cases"]]}
    return (f"Task type: {task['type']}. The criteria are "
            + {"choice": "a mapping option_key → description: translate the descriptions, keep the keys",
               "noul": "true/false descriptions (or null): translate the descriptions, keep the keys",
               "score": "an ordered list of levels: translate each level, keep the order"}[task["type"]]
            + ".\nTranslate this JSON. Return JSON only, same shape, plus \"language_dependent\" (bool) and \"note\" (str):\n"
            + json.dumps(payload, ensure_ascii=False, indent=1))


def check(task: dict, out: dict) -> str | None:
    q, n = task["question"], len(task["cases"])
    if not isinstance(out.get("states"), list) or len(out["states"]) != n:
        return "state count"
    if any(not isinstance(s, str) or not s.strip() for s in out["states"]):
        return "empty state"
    if not isinstance(out.get("instructions"), str) or not out["instructions"].strip():
        return "instructions"
    c0, c1 = q.get("criteria"), out.get("criteria")
    if isinstance(c0, dict) and (not isinstance(c1, dict) or set(c0) != set(c1)):
        return "criteria keys"
    if isinstance(c0, list) and (not isinstance(c1, list) or len(c0) != len(c1)):
        return "levels"
    out["states"] = [s.translate(DIGITS) for s in out["states"]]
    out["flags"] = [i for i, (en, fa) in enumerate(zip([c["state"] for c in task["cases"]], out["states"]))
                    if numbers(en) != numbers(fa)
                    or (not re.search(r"[؀-ۿ]", fa) and re.search(r"[A-Za-z]{4,}\s+[A-Za-z]{4,}", en))]
    return None


DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")


def numbers(s: str) -> list[str]:
    """Numbers as a multiset, ignoring thousands separators; a case whose numbers changed is flagged for review."""
    return sorted(re.findall(r"\d+", re.sub(r"(?<=\d)[,٬](?=\d{3})", "", s)))


async def translate(client: httpx.AsyncClient, model: str, task: dict) -> dict:
    last = None
    for attempt in range(4):
        key = (Path.home() / ".config/openrouter/key").read_text().strip()
        r = await client.post(URL, headers={"Authorization": f"Bearer {key}"}, json={
            "model": model, "temperature": 0, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt(task)}]}, timeout=300)
        try:
            text = r.json()["choices"][0]["message"]["content"]
            out = json.loads(text[text.find("{"): text.rfind("}") + 1])
        except (KeyError, ValueError, IndexError) as e:
            last = f"bad response: {e}: {r.text[:200]}"
            continue
        last = check(task, out)
        if last is None:
            return out
    raise RuntimeError(f"{task['id']}: {last}")


CURRENCY = [(re.compile(r"\$(\d[\d,]*(?:\.\d+)?)"), r"\1 دلار"), (re.compile(r"€(\d[\d,]*(?:\.\d+)?)"), r"\1 یورو")]


def postedit(tr: dict, fixes: dict) -> dict:
    """The review pass (fixes.json: native-style rewrites of single cases, keyed by task and case index) plus two
    house rules applied everywhere: an amount is written "85 دلار", never "$85", inside Persian text (a Latin
    currency sign breaks right-to-left rendering), and «در» (door) is never spelled «درب»."""
    out = {}
    for tid, f in tr.items():
        f = json.loads(json.dumps(f))
        fx = fixes.get(tid, {})
        for i, text in fx.get("cases", {}).items():
            f["states"][int(i)] = text
        if "instructions" in fx:
            f["instructions"] = fx["instructions"]
        for k, v in fx.get("criteria", {}).items():
            f["criteria"][int(k) if isinstance(f["criteria"], list) else k] = v
        for j, s in enumerate(f["states"]):
            if re.search(r"[\u0600-\u06FF]", s):
                for rx, rep in CURRENCY:
                    s = rx.sub(rep, s)
                f["states"][j] = re.sub(r"(?<![\u0600-\u06FF])درب(?![\u0600-\u06FF])", "در", s)
        out[tid] = f
    return out


def suite(tasks: list[dict], tr: dict, full: bool, suffix: str) -> dict:
    out = []
    for t in tasks:
        f = tr[t["id"]]
        q = {"instructions": f["instructions"] if full else t["question"]["instructions"]}
        if t["question"].get("criteria") is not None:
            q["criteria"] = f["criteria"] if full else t["question"]["criteria"]
        out.append({"id": f"{t['id']}_{suffix}", "type": t["type"], "question": q,
                    "cases": [{"state": s, "expected": c["expected"]} for s, c in zip(f["states"], t["cases"])]})
    return {"task": out}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", default="release/jabr-fa")
    ap.add_argument("--model", required=True, help="OpenRouter model id of the translator")
    ap.add_argument("--streams", type=int, default=6)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tasks = tomllib.loads(Path(a.cases).read_text())["task"]
    raw = out / "translations.jsonl"
    tr = {json.loads(l)["id"]: json.loads(l)["fa"] for l in raw.open()} if raw.exists() else {}
    sem = asyncio.Semaphore(a.streams)
    async with httpx.AsyncClient() as client:
        async def one(t):
            async with sem:
                try:
                    fa = await translate(client, a.model, t)
                except RuntimeError as e:
                    print("FAILED", e, flush=True); return
                tr[t["id"]] = fa
                with raw.open("a") as fo:
                    fo.write(json.dumps({"id": t["id"], "model": a.model, "fa": fa}, ensure_ascii=False) + "\n")
                print("ok", t["id"], len(t["cases"]), "flags", fa["flags"], "dep" if fa.get("language_dependent") else "", flush=True)
        await asyncio.gather(*(one(t) for t in tasks if t["id"] not in tr))
    missing = [t["id"] for t in tasks if t["id"] not in tr]
    if missing:
        print("missing:", missing); return
    fixes = json.loads((out / "fixes.json").read_text()) if (out / "fixes.json").exists() else {}
    tr = postedit(tr, fixes)
    keep = [t for t in tasks if not tr[t["id"]].get("language_dependent")]
    (out / "excluded.json").write_text(json.dumps({t["id"]: tr[t["id"]].get("note", "") for t in tasks
                                                   if tr[t["id"]].get("language_dependent")}, ensure_ascii=False, indent=1))
    (out / "v2-fa.toml").write_text("# jabr/classifier-benchmark v2, Persian edition (question and states in Persian).\n"
                                    + tomli_w.dumps(suite(keep, tr, True, "fa")))
    (out / "v2-fa-state.toml").write_text("# jabr/classifier-benchmark v2, Persian states with the original English questions.\n"
                                          + tomli_w.dumps(suite(keep, tr, False, "fas")))
    print(f"{len(keep)} tasks, {sum(len(t['cases']) for t in keep)} cases; excluded {len(tasks) - len(keep)}")


if __name__ == "__main__":
    asyncio.run(main())
