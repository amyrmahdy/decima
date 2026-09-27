"""Check locally rebuilt evaluation items against a published, text-free manifest.

    uv run python -m bench.verify_items --items runs/items/kev.jsonl --manifest items/kev.manifest.parquet

The `decima-bench-predictions` dataset ships no evaluation text. For every item it ships a manifest row
(`id`, `suite`, `split`, `lang`, `kind`, `n_choices`, `group`/`perm` for flip copies, and `sha256`, a hash
of the item's state, question, choices and gold label — see `item_sha256`). Rebuild the items from the
original datasets with `bench.items`, then run this to confirm they are the items the predictions were
made on: same ids, same metadata, same hash. A mismatch means the upstream dataset (or a loader) changed;
the predictions for mismatching ids are then not comparable. Exit status: 0 = identical, 1 = differences.

Manifests are read from `.parquet` (needs pyarrow), `.jsonl` or `.jsonl.gz`.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

FIELDS = ("suite", "split", "lang", "kind", "n_choices", "group", "perm")


def item_sha256(item: dict) -> str:
    """sha256 of the canonical JSON of [state, question, choices, gold] — the text-bearing fields plus the
    label, so a changed text, choice order or label all show up, while the manifest carries no text."""
    payload = [item.get("state"), item.get("question"), item.get("choices"), item.get("gold")]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def manifest_row(item: dict) -> dict:
    """The manifest entry for one item: identifiers and shape only, no text, no gold."""
    row = {"id": item["id"], "suite": item["suite"], "split": item["split"], "lang": item.get("lang"),
           "kind": item["kind"], "n_choices": len(item["choices"]), "group": item.get("group"),
           "perm": item.get("perm")}
    row["sha256"] = item_sha256(item)
    return row


def read_rows(path: str | Path) -> list[dict]:
    p = Path(path)
    if p.suffix == ".parquet":
        import pyarrow.parquet as pq

        return pq.read_table(p).to_pylist()
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def verify(items: list[dict], manifest: list[dict], show: int = 10) -> int:
    want = {r["id"]: r for r in manifest}
    have = {i["id"]: manifest_row(i) for i in items}
    missing, extra = sorted(want.keys() - have.keys()), sorted(have.keys() - want.keys())
    bad_hash, bad_meta = [], []
    for k in want.keys() & have.keys():
        w, h = want[k], have[k]
        if w["sha256"] != h["sha256"]:
            bad_hash.append(k)
        elif any(w.get(f) != h.get(f) for f in FIELDS):
            bad_meta.append(k)
    ok = len(want) - len(missing) - len(bad_hash) - len(bad_meta)
    print(f"manifest {len(want):,} items · rebuilt {len(have):,} · identical {ok:,} · missing {len(missing):,} · "
          f"extra {len(extra):,} · text/label differs {len(bad_hash):,} · metadata differs {len(bad_meta):,}")
    for name, ids in (("missing", missing), ("extra", extra), ("text/label differs", bad_hash),
                      ("metadata differs", bad_meta)):
        for k in sorted(ids)[:show]:
            print(f"  {name}: {k}")
    return int(bool(missing or extra or bad_hash or bad_meta))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--items", required=True, help="rebuilt items (bench.items output, .jsonl)")
    ap.add_argument("--manifest", required=True, help="items/<set>.manifest.parquet from the predictions dataset")
    ap.add_argument("--show", type=int, default=10, help="differing ids to print per category")
    args = ap.parse_args()
    sys.exit(verify(read_rows(args.items), read_rows(args.manifest), args.show))


if __name__ == "__main__":
    main()
