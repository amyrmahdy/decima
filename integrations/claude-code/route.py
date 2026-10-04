#!/usr/bin/env python3
"""Pick the cheapest model that can do a task well, before the run starts.

    python3 route.py "fix the flaky date test in utils/"                 → prints the chosen model
    python3 route.py --run "rename getUser to findUser in src/users"     → runs `claude -p --model <model> <task>`

Tiers and the models they map to come from DECIMA_TIERS (JSON) or the defaults below, which use Claude Code's model
aliases; for local backends point them at your own names (e.g. rande-fast / rande-strong). When Decima is unsure
(confidence < DECIMA_ROUTE_MIN, default 0.6) or the server is down, it goes up a tier, never down: a wrong cheap pick
costs a failed run, a wrong expensive one costs cents.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from decima_hook import DecimaDown, ask  # noqa: E402

TIERS = {
    "haiku": "Mechanical change in one or two files: rename, typo, lint fix, small config edit",
    "sonnet": "Normal feature or bug fix inside an existing pattern, a few files, tests exist",
    "opus": "Unclear cause, cross-cutting change, architecture, migration or security code",
    "other": "Not a coding task, or too vague to judge",
}
MODELS = json.loads(os.environ.get("DECIMA_TIERS", "{}")) or {"haiku": "haiku", "sonnet": "sonnet", "opus": "opus", "other": "opus"}
ORDER = ["haiku", "sonnet", "opus"]
MIN_CONF = float(os.environ.get("DECIMA_ROUTE_MIN", "0.6"))


def route(task: str) -> tuple[str, str]:
    try:
        a = ask({"task": task}, {"tier": {"type": "choice", "instructions": "Which model tier does `task` need to be done well?", "criteria": TIERS}})["tier"]
    except (DecimaDown, KeyError) as e:
        return MODELS["opus"], f"Decima unavailable ({e}) → strongest tier"
    tier, conf = a["choice"], a["confidence"]
    if tier == "other":
        return MODELS["other"], f"other ({conf:.2f}) → {MODELS['other']}"
    if conf < MIN_CONF and tier in ORDER[:-1]:
        up = ORDER[ORDER.index(tier) + 1]
        return MODELS[up], f"{tier} ({conf:.2f}) unsure → up to {up}"
    return MODELS[tier], f"{tier} ({conf:.2f})"


if __name__ == "__main__":
    run = "--run" in sys.argv
    task = " ".join(a for a in sys.argv[1:] if a != "--run")
    if not task:
        sys.exit("usage: route.py [--run] \"task description\"")
    model, why = route(task)
    print(f"router: {why} -> {model}", file=sys.stderr)
    if run:
        sys.exit(subprocess.run(["claude", "-p", "--model", model, task]).returncode)
    print(model)
