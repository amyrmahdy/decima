"""The `decima` command.

    decima "I was charged twice and nobody answers" --preset triage
    decima "Is this a refund request?" … → see `decima --help`
    echo "rm -rf build/" | decima --preset agent-gate --json
    decima --file tickets.txt --lines --preset triage --json     # one decision per line, JSON lines
    decima presets [name]                                       # list presets / show one preset's questions
    decima serve [--model decima-base] [--port 11436]           # TypeSafe-compatible HTTP server
    decima mcp [--model …]                                      # MCP server over stdio
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .router import DEFAULT_MODEL, Router, as_state, list_presets, load_preset, resolve_model, short_name

BAR = 20
_TTY = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _TTY else s


def bar(p: float, width: int = BAR) -> str:
    eighths = round(max(0.0, min(1.0, p)) * width * 8)
    full, rest = divmod(eighths, 8)
    s = "█" * full + (" ▏▎▍▌▋▊▉"[rest] if rest else "")
    return s + " " * (width - len(s))


def _cut(s: str, n: int) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1] + "…"


LABEL = 32


def _rows(a: dict, top: int) -> tuple[list[tuple[str, float, bool]], str, int]:
    """(label, p, is the answer) rows for one answer, a note for its first row, and how many rows were left out."""
    if a["type"] == "noul":
        p = a["noul"]
        return [("yes" if p >= 0.5 else "no", p, True)], "p(yes)", 0
    if a["type"] == "score":
        best = round(a["score"])
        rows = [(f"{i} {a['legend'][i]}", p, int(i) == best) for i, p in a["probabilities"].items()]
        return rows, f"score {a['score']:.2f}", 0
    rows = sorted(((c, p, c == a["choice"]) for c, p in a["probabilities"].items()), key=lambda r: -r[1])
    return rows[:top], "", max(0, len(rows) - top)


def render(result: dict, top: int = 4) -> str:
    """Compact terminal view of one /v1/systemone response: a block per question, a probability bar per option."""
    blocks = {qid: _rows(a, top) for qid, a in result["answers"].items()}
    w = max(map(len, blocks)) + 2
    lw = min(LABEL, max(len(r[0]) for rows, _, _ in blocks.values() for r in rows))
    lines = []
    for qid, (rows, note, more) in blocks.items():
        noul = result["answers"][qid]["type"] == "noul"
        for j, (label, p, best) in enumerate(rows):
            head = _c("1", qid.ljust(w)) if j == 0 else " " * w
            name = _cut(label, lw).ljust(lw)
            if noul:
                name = _c("32;1" if p >= 0.5 else "31;1", name)
            else:
                name = _c("1", name) if best else _c("2", name) if p < 0.05 else name
            tail = f"  {_c('2', note)}" if j == 0 and note else ""
            lines.append(f"{head}{name}  {_c('36' if best else '2', bar(p))} {p:.2f}{tail}")
        if more:
            lines.append(" " * w + _c("2", f"… {more} more"))
    return "\n".join(lines)


def _split(v: str) -> list[str]:
    return [x.strip() for x in v.split("|" if "|" in v else ",") if x.strip()]


def build_questions(a) -> dict | None:
    qs = {}
    if a.questions:
        qs.update(json.loads(Path(a.questions).read_text()))
    if a.question:
        if a.choices:
            crit = dict((x.split("=", 1) + [""])[:2] for x in _split(a.choices))
            qs["answer"] = {"type": "choice", "instructions": a.question, "criteria": {k.strip(): v.strip() for k, v in crit.items()}}
        elif a.levels:
            qs["answer"] = {"type": "score", "instructions": a.question, "criteria": _split(a.levels)}
        else:
            qs["answer"] = {"type": "noul", "instructions": a.question}
    return qs or None


def cmd_presets(argv: list[str]) -> int:
    if argv:
        p = load_preset(argv[0])
        print(_c("1", argv[0]) + f"  {p['description']}\n" + _c("2", f"model {p['model']}"))
        if "state" in p:
            print(_c("2", f"text becomes {json.dumps(p['state'])}"))
        for qid, q in p["questions"].items():
            req = f"  (needs {', '.join(q['requires'])})" if q.get("requires") else ""
            print(f"\n  {_c('1', qid)} {_c('2', q['type'] + req)}\n    {q.get('instructions', qid)}")
            crit = q.get("criteria")
            if isinstance(crit, dict):
                for k, v in crit.items():
                    print(f"      {_c('36', k)}: {v}")
            elif isinstance(crit, list):
                for i, v in enumerate(crit):
                    print(f"      {_c('36', str(i))}: {v}")
        return 0
    ps = list_presets()
    w = max(map(len, ps)) + 2
    for name, desc in ps.items():
        print(f"{_c('1', name.ljust(w))}{desc}  {_c('2', short_name(load_preset(name)['model']))}")
    return 0


def cmd_serve(argv: list[str]) -> int:
    from . import serve

    argv = list(argv)
    if "--model" in argv:
        i = argv.index("--model") + 1
        given = argv[i]
        argv[i] = resolve_model(given)
        if "--name" not in argv:
            argv += ["--name", short_name(given)]
    sys.argv = ["decima serve", *argv]
    serve.main()
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")     # "PyTorch was not found": the runtime never needs it
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    if argv and argv[0] == "presets":
        return cmd_presets(argv[1:])
    if argv and argv[0] == "serve":
        return cmd_serve(argv[1:])
    if argv and argv[0] == "mcp":
        from .mcp_server import main as mcp_main

        return mcp_main(argv[1:])
    if argv and argv[0] == "--":
        argv = argv[1:]

    ap = argparse.ArgumentParser(prog="decima", description="Calibrated decisions on CPU: text + questions → probabilities.",
                                 epilog="Also: `decima presets [name]`, `decima serve [--model M] [--port P]`, `decima mcp`.")
    ap.add_argument("text", nargs="*", help="the state (text or a JSON object); default: --file or stdin")
    ap.add_argument("-p", "--preset", help=f"one of: {', '.join(list_presets())} (or a preset .json file)")
    ap.add_argument("-q", "--question", help="your own question: a yes/no statement (noul) unless --choices or --levels is given")
    ap.add_argument("-c", "--choices", help="options for --question: 'a,b,c' or 'a=description|b=description'")
    ap.add_argument("--levels", help="ordered levels for --question (score): 'low,medium,high'")
    ap.add_argument("--questions", help="a JSON file of TypeSafe questions {name: {type, instructions, criteria}}")
    ap.add_argument("-m", "--model", help=f"Hub repo, short name or local export (default: the preset's, else {short_name(DEFAULT_MODEL)})")
    ap.add_argument("--precision", default="int8", choices=["int8", "fp32"])
    ap.add_argument("-t", "--threads", type=int, default=1)
    ap.add_argument("-f", "--file", help="read the state from a file ('-' = stdin)")
    ap.add_argument("--lines", action="store_true", help="one decision per non-empty input line")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--json", action="store_true", help="print the /v1/systemone response (JSON lines with --lines)")
    a = ap.parse_args(argv)

    if a.text:
        text = " ".join(a.text)
    elif a.file and a.file != "-":
        text = Path(a.file).read_text()
    elif a.file == "-" or not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        ap.print_usage(sys.stderr)
        print("decima: give a text, --file or stdin", file=sys.stderr)
        return 2
    states = [x for x in text.splitlines() if x.strip()] if a.lines else [text.strip()]
    if not any(states):
        print("decima: the input is empty", file=sys.stderr)
        return 2
    router = Router(model=a.model, precision=a.precision, threads=a.threads)
    try:
        questions = build_questions(a)
        if not questions and not a.preset:
            a.preset = "triage"
            print(_c("2", "no --preset or --question given: using --preset triage"), file=sys.stderr)
        preset = load_preset(a.preset) if a.preset else None
        results = router.predict_many([as_state(s, preset) for s in states], questions, preset=preset, lang=a.lang)
    except (ValueError, OSError) as e:
        print(f"decima: {e}", file=sys.stderr)
        return 2

    for s, r in zip(states, results):
        for warn in r.get("warnings", []):
            print(_c("33", f"warning: {warn}"), file=sys.stderr)
        if a.json:
            print(json.dumps(r, ensure_ascii=False, indent=None if a.lines else 2))
            continue
        if a.lines:
            print(_c("1", _cut(s, 100)))
        else:
            print(_c("2", " · ".join(x for x in (r["model"], a.preset and Path(a.preset).stem, f"{r['latency_ms']:.0f} ms") if x)))
        print(render(r))
        if a.lines:
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
