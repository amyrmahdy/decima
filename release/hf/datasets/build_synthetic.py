"""Assemble the public `decima-synthetic-decisions` dataset from the teacher's raw output.

    uv run python release/hf/datasets/build_synthetic.py --out /path/to/scratch/ds
    uv run python release/hf/datasets/build_synthetic.py --out ... --with-relabels --format parquet jsonl

Inputs (git-ignored, produced by the generators in teacher/):

    data/teacher/generate-rande-fast-local.jsonl   teacher.generate      → config "short"
    data/teacher/jevgen-rande-fast-local.jsonl     teacher.generate_jev  → config "long"
    data/teacher/label-rande-fast-local.jsonl      teacher.label         → config "relabel" (opt-in,
        ONLY the rows whose `batch` starts with "gen:" — our own generated states re-asked under new
        questions. Rows labelled on public train splits (banking77, clinc150, massive, ag_news, sst5,
        xnli) carry third-party text and are NEVER exported by this script.)
    data/jevgen-snap/jevgen.jsonl                  ids of the long rows Decima-small v1i trained on

Every row that comes out is fully synthetic: state, question, choices and probabilities were all
written by the teacher (Gemma-4-26B-A4B-it, NVFP4 build nvidia/Gemma-4-26B-A4B-NVFP4, served by vLLM).

Released schema (all configs; config-specific columns are null elsewhere):

    id               str   16-hex sha1 of state␟question␟choices (the generators' own id)
    config           str   short | long | relabel
    group            str   rows that must stay in the same split (same generation call / same state)
    kind             str   choose | score | verify | rank
    domain           str   generation-grid domain
    state_lang       str   en | fa | ar | ru
    choice_lang      str   en | fa | ar | ru (question language = choice_lang for short/relabel,
                           = state_lang for long)
    cross_lingual    bool  state_lang != choice_lang
    state            str
    question         str
    choices          list[str]
    n_choices        int
    gold             int   teacher's single best choice (0-based)
    probs            list[float]  teacher distribution; sums to 1 except for kind=rank (independent
                           per-choice probabilities in [0,1])
    label_format     str   full (teacher wrote every probability) | topk (teacher wrote 3-6
                           [index, prob] pairs; leftover mass spread uniformly, floor 1e-4)
    none_idx         int?  index of the "none of the above" / "insufficient information" choice
    scenario         str?  long only: sub-scenario the state was asked for
    artefact_format  str?  long only: artefact form the state imitates (email thread, log, …)
    q_index          int?  long only: position of the question among the state's 3-5 questions
    uncertain_requested bool? long only: the prompt asked for a genuinely uncertain label
    gold_tie         bool? long only: gold is within 0.05 of the argmax but not the argmax
    in_decima_small_training bool  row was in Decima-small (v1i) training data

Internal fields dropped: `model` (local gateway alias "rande-fast-local"; the teacher is named in
the card instead), `cell` (generation-grid call index; folded into `group`), `source` (becomes
`config`), `batch` (label-job batch key), rows with source="empty" (resume markers for calls that
produced nothing). Renamed: `format` → `artefact_format`, `uncertain_asked` → `uncertain_requested`.

Splits: `train` / `test`, assigned by sha1(group) so every question about one state (and every
example from one catalogue call) lands in the same split. Output is sorted by id → byte-identical
JSONL across runs; Parquet is identical given the same pyarrow version.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
GEN = ROOT / "data/teacher/generate-rande-fast-local.jsonl"
JEV = ROOT / "data/teacher/jevgen-rande-fast-local.jsonl"
LAB = ROOT / "data/teacher/label-rande-fast-local.jsonl"
SNAP = ROOT / "data/jevgen-snap/jevgen.jsonl"

TEACHER = "google/gemma-4-26B-A4B-it (NVFP4 build nvidia/Gemma-4-26B-A4B-NVFP4, vLLM, thinking off, T=0.9)"
LANGS = {"en", "fa", "ar", "ru"}
KINDS = {"choose", "score", "verify", "rank"}
TOPK_MIN_CHOICES = 16   # generate.py: catalogue calls (>= 20 asked, >= 0.8x accepted) use sparse top-k
DROPPED = ["model", "cell", "source", "batch", "format→artefact_format", "uncertain_asked→uncertain_requested"]

COLUMNS = ["id", "config", "group", "kind", "domain", "state_lang", "choice_lang", "cross_lingual", "state",
           "question", "choices", "n_choices", "gold", "probs", "label_format", "none_idx", "scenario",
           "artefact_format", "q_index", "uncertain_requested", "gold_tie", "in_decima_small_training"]

# Optional: free-mail domains → RFC 2606 reserved ".example" names, so no invented address can be
# a real mailbox. Off by default (keeps rows identical to what the model was trained on).
FREE_MAIL = ["gmail.com", "googlemail.com", "mail.ru", "yandex.ru", "yahoo.com", "hotmail.com", "outlook.com",
             "protonmail.com", "icloud.com", "live.com", "aol.com", "gmx.de", "web.de", "yahoo.co.uk", "inbox.ru",
             "bk.ru", "list.ru", "rambler.ru", "mail.com", "proton.me", "gmx.com", "gmx.net", "msn.com", "ymail.com",
             "me.com", "yandex.com", "zoho.com", "yahoo.fr", "hotmail.fr", "orange.fr", "libero.it", "qq.com", "163.com"]
_MAIL_RE = re.compile(r"(@)(" + "|".join(re.escape(d) for d in FREE_MAIL) + r")\b", re.I)


_ANY_MAIL = re.compile(r"(?<![\w.+-])([\w.+-]+)@([A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*)\.([A-Za-z]{2,})\b")
_RESERVED = {"example", "test", "invalid", "localhost"}


def scrub(s: str) -> str:
    """Every e-mail address gets a reserved TLD (RFC 2606): name@host.tld → name@host.example, so no address in
    the release points at a real domain — free-mail providers first (gmail.com → gmail.example), then the rest."""
    s = _MAIL_RE.sub(lambda m: m.group(1) + m.group(2).split(".")[0].lower() + ".example", s)
    return _ANY_MAIL.sub(lambda m: m.group(0) if m.group(3).lower() in _RESERVED
                         else f"{m.group(1)}@{m.group(2)}.example", s)


def split_of(group: str, test_pct: int) -> str:
    return "test" if int(hashlib.sha1(group.encode()).hexdigest()[:8], 16) % 100 < test_pct else "train"


def validate(r: dict) -> None:
    n = len(r["choices"])
    assert r["kind"] in KINDS and r["state_lang"] in LANGS and r["choice_lang"] in LANGS, r["id"]
    assert n >= 2 and len(r["probs"]) == n and 0 <= r["gold"] < n, r["id"]
    assert all(isinstance(c, str) and c.strip() for c in r["choices"]) and r["state"].strip() and r["question"].strip(), r["id"]
    assert all(0.0 <= p <= 1.0 for p in r["probs"]), r["id"]
    if r["kind"] != "rank":
        assert abs(sum(r["probs"]) - 1) < 1e-3, (r["id"], sum(r["probs"]))
    if r["kind"] == "verify":
        assert n == 2, r["id"]
    if r["none_idx"] is not None:
        assert 0 <= r["none_idx"] < n, r["id"]


def base(r: dict, config: str, group: str, label_format: str, in_train: bool) -> dict:
    out = {c: None for c in COLUMNS}
    out.update(id=r["id"], config=config, group=group, kind=r["kind"], domain=r["domain"],
               state_lang=r["state_lang"], choice_lang=r["choice_lang"],
               cross_lingual=r["state_lang"] != r["choice_lang"], state=r["state"], question=r["question"],
               choices=list(r["choices"]), n_choices=len(r["choices"]), gold=int(r["gold"]),
               probs=[float(p) for p in r["probs"]], label_format=label_format, none_idx=r.get("none_idx"),
               in_decima_small_training=in_train)
    return out


def read_jsonl(p: Path):
    with p.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load(with_relabels: bool) -> tuple[list[dict], Counter]:
    rows, skipped = [], Counter()
    for r in read_jsonl(GEN):
        if r.get("source") != "generate":
            skipped["generate:" + str(r.get("source"))] += 1
            continue
        fmt = "topk" if len(r["choices"]) >= TOPK_MIN_CHOICES else "full"
        rows.append(base(r, "short", f"short-{r['cell']}", fmt, True))

    snap = {r["id"] for r in read_jsonl(SNAP)} if SNAP.exists() else set()
    for r in read_jsonl(JEV):
        if r.get("source") != "jevgen":
            skipped["jevgen:" + str(r.get("source"))] += 1
            continue
        o = base(r, "long", r["group"], "full", r["id"] in snap)
        o.update(scenario=r["scenario"], artefact_format=r["format"], q_index=r["q_index"],
                 uncertain_requested=bool(r["uncertain_asked"]), gold_tie=bool(r.get("gold_tie", False)))
        rows.append(o)

    if with_relabels:
        for r in read_jsonl(LAB):
            b = r.get("batch", "")
            if r.get("source") != "label" or not b.startswith("gen:"):
                skipped["label:public-or-empty (never exported)"] += 1
                continue
            # group = the state: the same generated state re-asked under several questions stays together
            g = "relabel-" + hashlib.sha1(r["state"].encode()).hexdigest()[:12]
            rows.append(base(r, "relabel", g, "topk", True))
    return rows, skipped


def dedup(rows: list[dict]) -> tuple[list[dict], int]:
    seen, out = set(), []
    for r in sorted(rows, key=lambda r: ({"short": 0, "long": 1, "relabel": 2}[r["config"]], r["id"])):
        if r["id"] in seen:
            continue
        seen.add(r["id"])
        out.append(r)
    return out, len(rows) - len(out)


def stats(rows: list[dict]) -> dict:
    s: dict = {}
    by = defaultdict(list)
    for r in rows:
        by[(r["config"], r["split"])].append(r)
    for (cfg, sp), rs in sorted(by.items()):
        top = [max(r["probs"]) / (sum(r["probs"]) if r["kind"] == "rank" else 1) for r in rs if r["kind"] != "rank"]
        words = sorted(len(r["state"].split()) for r in rs)
        bucket = Counter("2" if r["n_choices"] == 2 else "3-5" if r["n_choices"] <= 5 else "6-12" if r["n_choices"] <= 12
                         else "13-50" if r["n_choices"] <= 50 else "51+" for r in rs)
        s[f"{cfg}/{sp}"] = {
            "rows": len(rs), "states": len({r["state"] for r in rs}), "groups": len({r["group"] for r in rs}),
            "kind": dict(Counter(r["kind"] for r in rs)), "state_lang": dict(Counter(r["state_lang"] for r in rs)),
            "choice_lang": dict(Counter(r["choice_lang"] for r in rs)),
            "cross_lingual": sum(r["cross_lingual"] for r in rs), "with_none_option": sum(r["none_idx"] is not None for r in rs),
            "none_is_gold": sum(r["none_idx"] is not None and r["gold"] == r["none_idx"] for r in rs),
            "n_choices": dict(bucket), "max_choices": max(r["n_choices"] for r in rs),
            "label_format": dict(Counter(r["label_format"] for r in rs)),
            "domains": len({r["domain"] for r in rs}),
            "state_words": {"p10": words[len(words) // 10], "median": words[len(words) // 2],
                            "p90": words[9 * len(words) // 10], "max": words[-1]},
            "non_rank_top_prob": {"mean": round(sum(top) / max(len(top), 1), 4),
                                  "share_ge_0.9": round(sum(t >= 0.9 for t in top) / max(len(top), 1), 4),
                                  "share_le_0.6": round(sum(t <= 0.6 for t in top) / max(len(top), 1), 4)},
            "gold_is_argmax": round(sum(r["gold"] == max(range(r["n_choices"]), key=r["probs"].__getitem__) for r in rs) / len(rs), 4),
            "in_decima_small_training": sum(r["in_decima_small_training"] for r in rs),
        }
    return s


def write_jsonl(rows: list[dict], path: Path) -> None:
    # fixed gzip mtime → byte-identical output across runs
    with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as f:
        for r in rows:
            f.write((json.dumps({c: r[c] for c in COLUMNS}, ensure_ascii=False) + "\n").encode())


def write_parquet(rows: list[dict], path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    pa.set_cpu_count(2); pa.set_io_thread_count(2)
    schema = pa.schema([
        ("id", pa.string()), ("config", pa.string()), ("group", pa.string()), ("kind", pa.string()),
        ("domain", pa.string()), ("state_lang", pa.string()), ("choice_lang", pa.string()),
        ("cross_lingual", pa.bool_()), ("state", pa.string()), ("question", pa.string()),
        ("choices", pa.list_(pa.string())), ("n_choices", pa.int32()), ("gold", pa.int32()),
        ("probs", pa.list_(pa.float64())), ("label_format", pa.string()), ("none_idx", pa.int32()),
        ("scenario", pa.string()), ("artefact_format", pa.string()), ("q_index", pa.int32()),
        ("uncertain_requested", pa.bool_()), ("gold_tie", pa.bool_()), ("in_decima_small_training", pa.bool_()),
    ])
    table = pa.Table.from_pylist([{c: r[c] for c in COLUMNS} for r in rows], schema=schema)
    pq.write_table(table, path, compression="zstd", row_group_size=20_000)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="output directory (NOT inside the repo)")
    ap.add_argument("--with-relabels", action="store_true", help="add config 'relabel' (generated states re-asked; label file gen:* rows only)")
    ap.add_argument("--scrub-free-mail", action="store_true", help="rewrite @gmail.com etc. to @gmail.example in states")
    ap.add_argument("--test-pct", type=int, default=2)
    ap.add_argument("--format", nargs="+", default=["parquet", "jsonl"], choices=["parquet", "jsonl"])
    args = ap.parse_args()

    out = Path(args.out).resolve()
    if ROOT in out.parents or out == ROOT:
        raise SystemExit(f"refusing to write data inside the repo: {out}")
    rows, skipped = load(args.with_relabels)
    rows, dupes = dedup(rows)
    for r in rows:
        if args.scrub_free_mail:   # every text field: states, questions and options can all quote an address
            r["state"] = scrub(r["state"])
            r["question"] = scrub(r["question"])
            r["choices"] = [scrub(c) for c in r["choices"]]
        validate(r)
        r["split"] = split_of(r["group"], args.test_pct)

    configs = sorted({r["config"] for r in rows})
    files = {}
    for cfg in configs:
        for sp in ("train", "test"):
            part = sorted((r for r in rows if r["config"] == cfg and r["split"] == sp), key=lambda r: r["id"])
            d = out / cfg
            d.mkdir(parents=True, exist_ok=True)
            if "parquet" in args.format:
                p = d / f"{sp}.parquet"; write_parquet(part, p); files[str(p.relative_to(out))] = sha256(p)
            if "jsonl" in args.format:
                p = d / f"{sp}.jsonl.gz"; write_jsonl(part, p); files[str(p.relative_to(out))] = sha256(p)

    manifest = {
        "dataset": "decima-synthetic-decisions", "teacher": TEACHER, "columns": COLUMNS,
        "dropped_internal_fields": DROPPED, "scrub_free_mail": args.scrub_free_mail, "test_pct": args.test_pct,
        "inputs": {str(p.relative_to(ROOT)): sha256(p) for p in (GEN, JEV, LAB, SNAP) if p.exists()
                   and (p != LAB or args.with_relabels)},
        "skipped_input_rows": dict(skipped), "duplicate_ids_dropped": dupes,
        "stats": stats(rows), "files": files,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: manifest[k] for k in ("skipped_input_rows", "duplicate_ids_dropped", "files")}, indent=1))
    for k, v in manifest["stats"].items():
        print(f"{k:16s} rows={v['rows']:6d} states={v['states']:6d} kind={v['kind']} lang={v['state_lang']}")


if __name__ == "__main__":
    main()
