"""Weight-space ensemble of two Decima checkpoints (WiSE-FT style): θ = (1 − α)·θ_A + α·θ_B.

    uv run python scripts/soup.py checkpoints/v1a2/best checkpoints/v1g/best --alpha 0.5 --out checkpoints/soup-a2g-50/best

Why: later runs gained a lot on the label spaces in our gold data but lost true zero-shot ability
(BTZSC clean-18: v0 .586 → v1c .543). Interpolating a general and a specialised fine-tune of the same
backbone usually keeps most of the specialised gain and recovers the general one. Both checkpoints must
share the backbone and head shapes; the result takes B's config (layout, lengths, temperature) and tokenizer.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from decima.model import DecimaModel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--alpha", type=float, default=0.5, help="weight of B")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ma, mb = DecimaModel.load(args.a), DecimaModel.load(args.b)
    sa, sb = ma.state_dict(), mb.state_dict()
    if sa.keys() != sb.keys() or any(sa[k].shape != sb[k].shape for k in sa):
        raise SystemExit("checkpoints differ in parameter names or shapes (different backbone?)")
    mb.load_state_dict({k: ((1 - args.alpha) * sa[k].float() + args.alpha * sb[k].float()).to(sb[k].dtype)
                        if sb[k].is_floating_point() else sb[k] for k in sb})
    out = Path(args.out)
    mb.save(out)
    shutil.copytree(Path(args.b) / "encoder", out / "encoder", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("*.safetensors", "*.bin", "config.json"))   # tokenizer files only
    cfg = json.loads((out / "decima.json").read_text())
    cfg["soup"] = {"a": args.a, "b": args.b, "alpha": args.alpha}
    (out / "decima.json").write_text(json.dumps(cfg, indent=2))
    print(f"soup α={args.alpha}: {args.a} ⊕ {args.b} → {out}")


if __name__ == "__main__":
    main()
