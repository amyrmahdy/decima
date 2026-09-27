"""Pilot / dataset statistics: histograms, entropy, gold-vs-argmax agreement, random samples.

    uv run python -m teacher.stats data/teacher/*.jsonl --show 10
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import random


def entropy(p: list[float]) -> float:
    return -sum(x * math.log(x) for x in p if x > 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--show", type=int, default=0, help="random examples to print per language")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    recs = []
    for pat in args.paths:
        for p in glob.glob(pat):
            for line in open(p):
                r = json.loads(line)
                if r.get("source") not in (None, "empty") and "probs" in r:
                    recs.append(r)
    print(f"{len(recs)} examples")
    for key in ("source", "kind", "state_lang", "choice_lang", "domain"):
        c = collections.Counter(r[key] for r in recs)
        print(f"{key:12s}", "  ".join(f"{k}={v} ({v/len(recs):.0%})" for k, v in sorted(c.items(), key=lambda kv: -kv[1])[:24]))
    c = collections.Counter(len(r["choices"]) for r in recs)
    print(f"{'n_choices':12s}", "  ".join(f"{k}={v}" for k, v in sorted(c.items())))
    cross = sum(r["state_lang"] != r["choice_lang"] for r in recs)
    print(f"cross-lingual {cross/len(recs):.1%}")
    ch = [r for r in recs if r["kind"] == "choose"]
    with_none = [r for r in ch if r.get("none_idx") is not None]
    none_gold = sum(r["gold"] == r["none_idx"] for r in with_none)
    print(f"choose: {len(ch)}; with none-option {len(with_none)} ({len(with_none)/max(1,len(ch)):.0%}); none is gold in {none_gold} ({none_gold/max(1,len(with_none)):.0%} of those)")
    for kind in ("choose", "score", "verify", "rank"):
        rs = [r for r in recs if r["kind"] == kind]
        if not rs:
            continue
        ent = [entropy(r["probs"]) for r in rs] if kind != "rank" else [0.0]
        agree = sum(max(range(len(r["probs"])), key=r["probs"].__getitem__) == r["gold"] for r in rs) / len(rs)
        top = [max(r["probs"]) for r in rs]
        print(f"{kind:8s} n={len(rs):6d}  mean entropy={sum(ent)/len(ent):.3f} nats  gold==argmax {agree:.3f}  mean top-p {sum(top)/len(top):.3f}  "
              f"ambiguous(top≤0.6) {sum(t <= 0.6 for t in top)/len(top):.0%}")
    lens = [len(r["state"]) for r in recs]
    print(f"state chars: mean {sum(lens)/len(lens):.0f}  p10 {sorted(lens)[len(lens)//10]}  p90 {sorted(lens)[9*len(lens)//10]}")
    if args.show:
        rng = random.Random(args.seed)
        for lang in ("en", "fa", "ar", "ru"):
            rs = [r for r in recs if r["state_lang"] == lang]
            print(f"\n================= {lang}: {args.show} of {len(rs)}")
            for r in rng.sample(rs, min(args.show, len(rs))):
                print(f"[{r['source']}/{r['kind']}/{r['domain']} → {r['choice_lang']}] {r['state'][:300]}")
                print(f"   Q: {r['question']}")
                order = sorted(range(len(r["choices"])), key=lambda i: -r["probs"][i])[:6]
                for i in order:
                    print(f"   {'*' if i == r['gold'] else ' '} {r['probs'][i]:.2f} {r['choices'][i][:90]}")
                if len(r["choices"]) > 6:
                    print(f"     … {len(r['choices'])-6} more choices")


if __name__ == "__main__":
    main()
