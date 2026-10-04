"""Natural-language renderings of procedural scenes: an LLM rewrites the facts, code keeps the labels exact.

    uv run python -m teacher.paraphrase_proc --scenes 6000 --model rande-strong-local --streams 8 \
        --out data/proc/proc-para.jsonl

Why: the template renderings in teacher/procedural.py teach the skill but also the templates — a student
trained only on them misreads new wording ("1 tile ahead" vs "right in front of the hero"). Here a writer
model rewrites each scene's canonical facts as three differently styled sensor reports (HUD, telemetry, log,
narrative, bullets…). Every number must survive, nothing may be added, and no report may say what to do or
what is safe — checked in code; the label comes from the scene, never from the writer. Resumable by scene.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
from pathlib import Path

from teacher.client import Teacher, TeacherError
from teacher.procedural import SCREEN, COMPASS, grid_render, grid_rows, grid_scene, side_render, side_rows, side_scene
from teacher.prompts import loads_lenient

CONTEXTS = ["a retro video game HUD", "robot telemetry", "a game engine debug log", "a driving assistant", "a drone controller",
            "a text adventure narrator", "a warehouse fleet dashboard", "a mobile game's accessibility narration", "a sports-style commentary",
            "a sensor bus message"]
BANNED = re.compile(r"\b(should|must|best|recommend\w*|safe(ly|st)?|unsafe|advis\w*|ought|suggest\w*|optimal|wise)\b", re.I)

SYSTEM = "You rewrite sensor readings. You never add facts, never drop facts, and never give advice."


def prompt(facts: str, context: str) -> list[dict]:
    body = f"""Facts about a scene:
{facts}

Rewrite these facts as THREE different reports that {context} might show. Make the three reports look clearly different
from each other (for example: terse labelled lines, one compact sentence, a bullet list, key=value pairs, a short narration).
Rules: keep every fact and every number exactly (write numbers as digits); add nothing new; do not say what anyone should do,
what is best, or whether something is safe or dangerous.
Output JSON only: {{"reports": ["...", "...", "..."]}}"""
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": body}]


def numbers(s: str) -> list[str]:
    return re.findall(r"\d+", s)


def valid(report: str, facts: str) -> bool:
    if not isinstance(report, str) or len(report) < 10 or BANNED.search(report):
        return False
    have, want = numbers(report), numbers(facts)
    return all(have.count(n) >= want.count(n) for n in set(want)) and set(have) <= set(want)   # nothing lost, nothing invented


async def run(a) -> None:
    out = Path(a.out)
    done = set()
    if out.exists():
        done = {json.loads(l)["scene"] for l in out.open() if l.strip()}
    rng0 = random.Random(f"para:{a.seed}")
    jobs = []
    for i in range(a.scenes):
        rng = random.Random(f"para:{a.seed}:{i}")
        fam = "grid" if i % 2 == 0 else "side"
        if fam == "grid":
            sc, vocab = grid_scene(rng), (COMPASS if rng.random() < 0.3 else SCREEN)
            facts = grid_render(sc, random.Random(f"canon:{i}:1"), vocab)
            while facts.startswith("{") or "|" in facts:        # canonical = a prose or line rendering, not JSON / pipes
                facts = grid_render(sc, rng, vocab)
        else:
            sc, vocab = side_scene(rng), None
            facts = side_render(sc, rng)
            while facts.startswith("{") or "|" in facts:
                facts = side_render(sc, rng)
        if i not in done:
            jobs.append((i, fam, sc, vocab, facts, rng0.choice(CONTEXTS)))
    print(f"{len(done)} scenes done, {len(jobs)} to go", flush=True)
    fo = out.open("a")
    st = {"scenes": 0, "rows": 0, "bad": 0, "failed": 0}
    async with Teacher(model=a.model, concurrency=a.streams) as t:
        async def one(i, fam, sc, vocab, facts, ctx):
            try:
                r = await t.chat(prompt(facts, ctx), max_tokens=900, temperature=0.9, thinking=False, json_mode=True, tag="para")
                reports = loads_lenient(r.text).get("reports")
                if not isinstance(reports, list):
                    raise ValueError("no reports")
            except (TeacherError, ValueError, AttributeError):
                st["failed"] += 1
                return
            rng = random.Random(f"rows:{i}")
            wrote = 0
            for k, rep in enumerate(reports[:3]):
                if not valid(rep, facts):
                    st["bad"] += 1
                    continue
                rows = grid_rows(rng, i * 10 + k, sc=sc, state=rep.strip(), vocab=vocab) if fam == "grid" else \
                    side_rows(rng, i * 10 + k, sc=sc, state=rep.strip())
                for row in rows:
                    fo.write(json.dumps({**row, "source": "proc-para", "scene": i, "context": ctx}, ensure_ascii=False) + "\n")
                    wrote += 1
            if not wrote:
                fo.write(json.dumps({"scene": i, "source": "empty"}) + "\n")
            fo.flush()
            st["scenes"] += 1; st["rows"] += wrote
            if st["scenes"] % 200 == 0:
                print(st, flush=True)
        await asyncio.gather(*(one(*j) for j in jobs))
    fo.close()
    print("done:", st, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=None)
    ap.add_argument("--streams", type=int, default=8)
    ap.add_argument("--out", default="data/proc/proc-para.jsonl")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
