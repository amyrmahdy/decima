"""Render System One rows (teacher/generate_s1.py) into Decima training rows.

    uv run python -m teacher.render_s1 --src data/teacher/s1-rande-fast-local.jsonl --out data/mix-k --copies 3

Each raw row is rendered `--copies` times with different, seeded surface forms, so the model learns the
task and not one wording of it:
  choice → choose; options as "key: description" (55 %), the description alone (20 %), the key alone
           with spaces (10 %) or "key — description" (15 %); the teacher's probabilities as target.
  noul   → verify; question = the statement itself (60 %), the statement + its true/false criteria (15 %,
           when present), "Is the following true? …" (13 %) or "True or false: …" (12 %); options
           yes/no (70 %) or true/false (30 %); target [p, 1 − p].
  score  → score; options = the level descriptions (85 %) or numbered "1. …" (15 %); teacher probabilities.
Decontamination: a row is dropped if its state or question matches any evaluation item (runs/items/*.jsonl)
or any case of jabr/classifier-benchmark (data/decontam/*.jsonl) — same exact / 200-char-prefix rule as
teacher/gold.py. Hold-out: whole tasks (groups) go to --dev-items by hash (5 %), written as bench items
(suite s1dev/<type>), so dev accuracy measures new tasks, not new states of seen tasks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from teacher.gold import Contam

YES_NO = {"en": ["yes", "no"], "fa": ["بله", "خیر"], "ar": ["نعم", "لا"], "ru": ["да", "нет"]}
TRUE_FALSE = {"en": ["true", "false"], "fa": ["درست", "نادرست"], "ar": ["صحيح", "خطأ"], "ru": ["верно", "неверно"]}
ASK_TRUE = {"en": "Is the following true? {s}", "fa": "آیا این درست است؟ {s}", "ar": "هل العبارة التالية صحيحة؟ {s}",
            "ru": "Верно ли следующее? {s}"}
TRUE_OR_FALSE = {"en": "True or false: {s}", "fa": "درست یا نادرست: {s}", "ar": "صح أم خطأ: {s}", "ru": "Верно или неверно: {s}"}
CRIT = {"en": "{s}\nTrue if: {t}\nFalse if: {f}", "fa": "{s}\nدرست اگر: {t}\nنادرست اگر: {f}",
        "ar": "{s}\nصحيح إذا: {t}\nخطأ إذا: {f}", "ru": "{s}\nВерно, если: {t}\nНеверно, если: {f}"}


def _h(s: str) -> int:
    return int(hashlib.sha1(s.encode()).hexdigest()[:8], 16)


def render(r: dict, copy: int) -> dict:
    rng = random.Random(f"{r['id']}:{copy}")
    ql = r.get("q_lang", "en")
    base = {"state": r["state"], "state_lang": r["state_lang"], "choice_lang": ql}
    if r["type"] == "choice":
        keys = list(r["probs"])
        u = rng.random()
        style = "kd" if u < 0.55 else "d" if u < 0.75 else "k" if u < 0.85 else "k—d"
        crit = r["criteria"]
        fmt = {"kd": lambda k: f"{k}: {crit[k]}", "d": lambda k: crit[k], "k": lambda k: k.replace("_", " "),
               "k—d": lambda k: f"{k} — {crit[k]}"}[style]
        choices = [fmt(k) for k in keys]
        if len({c.casefold() for c in choices}) != len(choices):
            choices = [f"{k}: {crit[k]}" for k in keys]
        probs = [r["probs"][k] for k in keys]
        row = {**base, "kind": "choose", "question": r["instructions"], "choices": choices, "probs": probs}
    elif r["type"] == "noul":
        s, p, crit = r["instructions"], r["p_true"], r.get("criteria")
        lang = ql if ql in YES_NO else "en"
        u = rng.random()
        if crit and u < 0.15:
            q = CRIT[lang].format(s=s, t=crit["true"], f=crit["false"])
        elif u < 0.75:
            q = s
        elif u < 0.88:
            q = ASK_TRUE[lang].format(s=s)
        else:
            q = TRUE_OR_FALSE[lang].format(s=s)
        opts = (YES_NO if rng.random() < 0.7 else TRUE_FALSE)[lang]
        row = {**base, "kind": "verify", "question": q, "choices": list(opts), "probs": [p, round(1 - p, 5)]}
    else:
        levels = r["criteria"]
        choices = levels if rng.random() < 0.85 else [f"{i + 1}. {lv}" for i, lv in enumerate(levels)]
        row = {**base, "kind": "score", "question": r["instructions"], "choices": list(choices), "probs": r["probs"]}
    row["gold"] = max(range(len(row["probs"])), key=row["probs"].__getitem__)
    key = row["state"] + "\x1f" + row["question"] + "\x1f" + "\x1f".join(row["choices"])
    row["id"] = hashlib.sha1(key.encode()).hexdigest()[:16]
    row.update(source="s1", s1_type=r["type"], group=r["group"], copy=copy)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--copies", type=int, default=3)
    ap.add_argument("--dev", type=float, default=0.05)
    ap.add_argument("--dev-items", default="runs/items-s1/s1dev.jsonl", help="held-out tasks as bench items (outside runs/items)")
    a = ap.parse_args()
    contam = [Contam(Path("runs/items")), Contam(Path("data/decontam"))]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    Path(a.dev_items).parent.mkdir(parents=True, exist_ok=True)
    ft, fd = (out / "s1-train.jsonl").open("w"), open(a.dev_items, "w")
    st, seen = Counter(), set()
    for src in a.src:
        for line in open(src):
            r = json.loads(line)
            if r.get("source") not in ("s1gen", "s2gen"):
                continue
            st["raw"] += 1
            if any(c.hit([r["state"], r["instructions"]]) for c in contam):
                st["decontam"] += 1
                continue
            dev = (_h(r["group"]) % 10_000) / 10_000 < a.dev
            for k in range(1 if dev else a.copies):
                row = render(r, k)
                if (row["id"], k) in seen:      # a copy that renders like an earlier one stays: it is extra weight
                    st["dupe"] += 1
                    continue
                seen.add((row["id"], k))
                if dev:
                    item = {"suite": f"s1dev/{r['type']}", "lang": row["state_lang"], "kind": row["kind"], "question": row["question"],
                            "choices": row["choices"], "id": f"s1dev/{row['id']}", "state": row["state"], "gold": row["gold"],
                            "probs": row["probs"], "split": "eval"}
                    fd.write(json.dumps(item, ensure_ascii=False) + "\n")
                else:
                    ft.write(json.dumps(row, ensure_ascii=False) + "\n")
                st[("dev" if dev else "train") + ":" + row["kind"]] += 1
    ft.close(); fd.close()
    print(dict(st))
    (out / "s1-stats.json").write_text(json.dumps(dict(st), indent=1))


if __name__ == "__main__":
    main()
