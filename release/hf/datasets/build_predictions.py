"""Assemble the public `decima-bench-predictions` dataset from the evaluation outputs in runs/.

    uv run python release/hf/datasets/build_predictions.py --out /path/outside/repo/decima-bench-predictions

Inputs (git-ignored, produced by bench.items / bench.predict / bench.score / scripts/audit/*.py):

    runs/items/<set>.jsonl              the evaluation items (third-party text — NEVER copied; only a
                                        text-free manifest is derived from them)
    runs/preds/<set>--<system>.jsonl    every system's probability vector per item (GPU, PyTorch)
    runs/x86/preds/*.jsonl              the ONNX export on x86 CPU (fp32 / int8), if present
    runs/<system>-<set>.json            bench.score / bench.jdi_index outputs (scores + the exact inputs)
    runs/final/                         the clean release re-run of Decima-small (fresh processes, frozen
                                        items). Used instead of the development files it replaces once
                                        runs/final/README.txt exists (written when the re-run completes).

Output layout (the dataset card describes it):

    preds/<set>--<system>.parquet       id: str, probs: list<float64>  (aligned with the item's choices)
    preds-x86/<set>--v1i-x86-<fp32|int8>.parquet
    items/<set>.manifest.parquet        id, suite, split, lang, kind, n_choices, group, perm, sha256
                                        (sha256 of [state, question, choices, gold]; see bench/verify_items.py)
    results/<system>-<set>.json         scores, per suite, with the items file and preds map used
    results/phase1*-summary.md          ablation summaries; results/latency-*-gx10-indicative.json
    results/x86/*.json                  int8-vs-fp32 comparison and x86 speed
    audit/*.json                        overlap, bootstrap CIs, calibration, parameter counts
    determinism.txt                     the release re-run vs the development run (when present)
    code-ref.json                       repository, commit, item-builder commands, resolved dataset revisions
    files.json                          sha256 and size of every file above
    README.md                           the dataset card

Probabilities are stored as float64, so re-scoring reproduces the recorded scores exactly. Parquet files
are zstd-compressed; rows keep the prediction file's order. `--format jsonl` writes `.jsonl.gz` instead
(no pyarrow needed). The script refuses to write inside the repository and checks that every predicted id
exists in its item set.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from bench.verify_items import manifest_row  # noqa: E402

REPO_URL, TAG = "https://github.com/amyrmahdy/decima", "v1.0.0"
CARD = ROOT / "release/hf/datasets/decima-bench-predictions/README.md"
ITEM_SETS = ("kev", "laya", "decima", "jevtyped", "btzsc", "jdi")
PRED_SET_ITEMS = {"jdi-all": "jdi", "jdi-rest": "jdi"}   # prediction set → item set (default: same name)
BUILD = {  # how each item set is rebuilt (dataset card, "How to reproduce every table")
    "kev": "uv run python -m bench.items --suites 'kev/*' --calib 300 --out runs/items/kev.jsonl",
    "laya": "uv run python -m bench.items --suites 'laya/*' --calib 300 --out runs/items/laya.jsonl",
    "decima": "uv run python -m bench.items --suites 'decima/*' --calib 500 --out runs/items/decima.jsonl",
    "jevtyped": "uv run python -m bench.items --suites 'jevbench/public' 'typed/test' --calib 1000 "
                "--out runs/items/jevtyped.jsonl",
    "btzsc": "uv run python -m bench.items --suites 'btzsc/*' --out runs/items/btzsc.jsonl",
    "jdi": "uv run python -m bench.items --suites 'jdi/*' --no-flips --calib 0 --limit 1000000000 "
           "--out runs/items/jdi.jsonl",
}
RESULT = re.compile(r"^(?P<sys>.+)-(?P<set>kev|laya|decima|jevtyped|btzsc|jdi)\.json$")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_rows(rows: list[dict], path: Path, fmt: str) -> Path:
    """Write rows as zstd Parquet (column types inferred) or as gzip JSONL with a fixed mtime."""
    if fmt == "parquet":
        import pyarrow as pa
        import pyarrow.parquet as pq

        out = path.parent / f"{path.name}.parquet"   # names contain dots (kev-0.5b)
        pq.write_table(pa.Table.from_pylist(rows), out, compression="zstd", row_group_size=50_000)
        return out
    out = path.parent / f"{path.name}.jsonl.gz"
    with open(out, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
        for r in rows:
            gz.write((json.dumps(r, ensure_ascii=False) + "\n").encode())
    return out


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def dataset_revisions() -> dict[str, str]:
    """Revisions of the Hub datasets the loaders resolved, read from the local HF cache (refs/main).
    Only datasets named in bench/*.py are listed — those are the ones the item builders load."""
    hub = Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub")
    src = "\n".join(p.read_text() for p in (ROOT / "bench").glob("*.py"))
    out = {}
    for d in sorted(hub.glob("datasets--*")):
        repo = d.name.removeprefix("datasets--").replace("--", "/", 1)
        ref = d / "refs" / "main"
        if f'"{repo}"' in src and ref.exists():
            out[repo] = ref.read_text().strip()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--format", choices=("parquet", "jsonl"), default="parquet")
    ap.add_argument("--dev-only", action="store_true", help="ignore runs/final even if the re-run is complete")
    ap.add_argument("--commit", default=None, help="commit to record in code-ref.json (default: this checkout's HEAD)")
    args = ap.parse_args()

    out = Path(args.out).expanduser().resolve()
    if out == ROOT or ROOT in out.parents:
        raise SystemExit("refusing to write inside the repository")
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")
    if args.format == "parquet":
        import pyarrow as pa

        pa.set_cpu_count(2)
        pa.set_io_thread_count(2)
    runs = ROOT / "runs"
    final = runs / "final"
    use_final = not args.dev_only and (final / "README.txt").exists()
    if not use_final:
        print("note: runs/final is not complete (no README.txt) — packaging the development predictions")
    for sub in ("preds", "preds-x86", "items", "results/x86", "audit"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    # 1. text-free manifests of every item set
    ids: dict[str, set[str]] = {}
    for s in ITEM_SETS:
        items = read_jsonl(runs / "items" / f"{s}.jsonl")
        ids[s] = {i["id"] for i in items}
        p = write_rows([manifest_row(i) for i in items], out / "items" / f"{s}.manifest", args.format)
        print(f"items/{p.name}: {len(items):,}")

    # 2. predictions (release re-run of Decima-small replaces the development file of the same name)
    sources = {Path(f).name: Path(f) for f in glob.glob(str(runs / "preds" / "*.jsonl"))}
    replaced = []
    if use_final:
        for f in glob.glob(str(final / "preds" / "*.jsonl")):
            replaced.append(Path(f).name)
            sources[Path(f).name] = Path(f)
    x86 = {Path(f).name: Path(f) for f in glob.glob(str(runs / "x86" / "preds" / "*.jsonl"))}
    n_rows = 0
    for folder, group in (("preds", sources), ("preds-x86", x86)):
        for name, path in sorted(group.items()):
            pset = name.split("--", 1)[0]
            iset = PRED_SET_ITEMS.get(pset, pset)
            rows = [{"id": r["id"], "probs": r["probs"]} for r in read_jsonl(path)]
            unknown = [r["id"] for r in rows if r["id"] not in ids[iset]]
            if unknown:
                raise SystemExit(f"{path}: {len(unknown)} ids not in runs/items/{iset}.jsonl, e.g. {unknown[0]}")
            write_rows(rows, out / folder / name.removesuffix(".jsonl"), args.format)
            n_rows += len(rows)
    print(f"preds: {len(sources)} GPU files ({len(replaced)} from the release re-run) + {len(x86)} x86 files, "
          f"{n_rows:,} rows")

    # 3. scores, summaries, audits
    results = {p.name: p for p in runs.glob("*.json") if RESULT.match(p.name)}
    if use_final:
        results.update({p.name: p for p in final.glob("*.json") if RESULT.match(p.name)})
        if (final / "determinism.txt").exists():
            shutil.copy2(final / "determinism.txt", out / "determinism.txt")
    for name, p in sorted(results.items()):
        shutil.copy2(p, out / "results" / name)
    for p in [*runs.glob("phase1*-summary.md"), *runs.glob("latency-*-gx10-indicative.json")]:
        shutil.copy2(p, out / "results" / p.name)
    for p in (runs / "x86").glob("*.json"):
        shutil.copy2(p, out / "results" / "x86" / p.name)
    for p in (runs / "audit").glob("*.json"):
        shutil.copy2(p, out / "audit" / p.name)

    # 4. provenance
    from bench.suites_public import JDI_KIT_COMMIT, JDI_KIT_URL, JEVBENCH_COMMIT

    commit = args.commit or git("rev-parse", "HEAD")
    ref = {
        "repository": REPO_URL, "tag": TAG, "commit": commit,
        "commit_note": None if args.commit else "commit of the checkout this package was built from",
        "dirty_worktree": bool(git("status", "--porcelain", "--untracked-files=no")) if not args.commit else None,
        "decima_small_predictions": "release re-run (fresh processes, frozen items)" if use_final else "development run",
        "replaced_by_release_rerun": sorted(replaced),
        "item_builders": BUILD,
        "jdi_panel_note": "jdi-all = the 19-benchmark panel subset of runs/items/jdi.jsonl (ids in preds/jdi-all--*)",
        "external_sources": {
            "jevbench": {"repo": "https://github.com/fstandhartinger/jevbench", "commit": JEVBENCH_COMMIT},
            "jdi_kit": {"repo": JDI_KIT_URL, "commit": JDI_KIT_COMMIT},
        },
        "dataset_revisions": dataset_revisions(),
        "dataset_revisions_note": "Hub revisions resolved when the items were built (local HF cache refs/main). "
                                  "If the Hub copy has moved, load these revisions; bench.verify_items detects drift.",
    }
    (out / "code-ref.json").write_text(json.dumps(ref, indent=2) + "\n")
    if CARD.exists():
        shutil.copy2(CARD, out / "README.md")

    files = {}
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name != "files.json":
            rel = str(p.relative_to(out))
            files[rel] = {"sha256": sha256(p), "bytes": p.stat().st_size}
    (out / "files.json").write_text(json.dumps(files, indent=1) + "\n")
    total = sum(f["bytes"] for f in files.values())
    print(f"{len(files)} files, {total / 1e6:,.1f} MB → {out}")


if __name__ == "__main__":
    main()
