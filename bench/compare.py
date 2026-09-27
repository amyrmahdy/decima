"""Before/after table from two or more bench result files.

    uv run python -m bench.compare runs/baseline-e5-small-500-part*.json runs/pilot-decima-300.json
Several files for one system are merged (the baseline was run in two parts)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from tabulate import tabulate


def load(paths: list[str]) -> tuple[str, dict]:
    rows, name = {}, None
    for p in paths:
        d = json.loads(Path(p).read_text())
        name = name or f"{d['system']} ({d.get('limit')})"
        for r in d["rows"]:
            rows[r["task"]] = r
    return name, rows


def main() -> None:
    groups: list[list[str]] = [[]]
    for a in sys.argv[1:]:          # "--" separates systems; otherwise each file is its own system
        if a == "--":
            groups.append([])
        else:
            groups[-1].append(a)
    groups = [g for g in groups if g]
    systems = [load(g) for g in groups]
    tasks = list(systems[0][1])
    head = ["task", "kind", "n_ch"]
    for name, _ in systems:
        head += [f"acc {name[:18]}", "ece_cal"]
    body = []
    for t in tasks:
        r0 = systems[0][1][t]
        row = [t, r0["kind"], r0["choices"]]
        for _, rows in systems:
            r = rows.get(t)
            row += [r["acc"] if r else None, r["ece_cal"] if r else None]
        body.append(row)
    if len(systems) > 1:
        body.append(["mean", "", ""] + [x for _, rows in systems for x in (
            sum(rows[t]["acc"] for t in tasks if t in rows) / len(tasks), sum(rows[t]["ece_cal"] for t in tasks if t in rows) / len(tasks))])
    print(tabulate(body, headers=head, floatfmt=".3f"))


if __name__ == "__main__":
    main()
