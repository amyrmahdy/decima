"""Run a Laya checkpoint over a Decima items JSONL and write per-item probabilities.

Runs in its own venv (Laya pins nothing and pulls its own transformers); never import this
from the decima package. Setup on the GX10 (aarch64 + GB10, CUDA 12.8 wheels):

    export PATH=$HOME/.local/bin:$PATH
    V=$HOME/venvs/laya
    uv venv --python 3.12 $V
    VIRTUAL_ENV=$V uv pip install --index-url https://download.pytorch.org/whl/cu128 "torch==2.11.0"
    VIRTUAL_ENV=$V uv pip install "laya==0.3.3" "transformers==5.17.0" safetensors huggingface_hub numpy

    $V/bin/python bench/external/laya_predict.py --items items.jsonl --model laya --out preds.jsonl

Input rows: {"id", "kind", "question", "choices", "state", ...} as built by bench/suites_laya.py.
Output rows: {"id", "probs": [aligned with choices], "latency_ms", "logits", "model"}.

Mapping onto Laya's API (`Agent.system_one(state, {qid: qdef})`):
    choose -> {"type": "choice", "instructions": question, "criteria": {choice: None, ...}}
              Laya renders a None-valued criterion as the bare key, so the model reads each
              choice string verbatim ("alarm_set: alarm set" for the Laya suites).
    score  -> {"type": "score", "criteria": choices}   (Laya renders "level i: <choice>")
    verify -> {"type": "noul"}, choices ["yes", "no"] -> probs [p_true, p_false]
    rank   -> treated as choice (Laya has no independent per-option scoring)
The state string is passed through unchanged (Laya json-dumps dict states; suites_laya already
stores the dumped string, so the tokens are identical to Laya's own benchmark).

`probs` use the checkpoint's shipped temperature (what a Laya user gets; laya-multilingual ships
all 1.0). `logits` are raw, for refitting temperature. The forward pass mirrors
`Agent.system_one` exactly; we call the pieces directly only to skip its 4-decimal rounding.
Autocast is fp16 by default: Laya's published run was on a T4 (sm75 -> fp16), and fp16/fp32
reproduce it exactly; the bf16 Laya picks on sm>=80 flips about one item per 300 suite.
`latency_ms` is one item per call (batch 1), CUDA-synchronised, model already warm.

--model: laya | laya-multilingual (standalone hub repos) | router (laya.Router: script/language
detection on the state picks english vs multilingual per item; the pick is in "model").
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("USE_TF", "0")

import numpy as np
import torch

REPOS = {"laya": "convaiinnovations/laya", "laya-multilingual": "convaiinnovations/laya-multilingual"}


def to_qdef(it: dict) -> dict:
    kind, q, ch = it["kind"], it["question"], it["choices"]
    if kind in ("choose", "rank"):
        if len(set(ch)) != len(ch):
            raise ValueError("duplicate choices in %s" % it["id"])
        return {"type": "choice", "instructions": q, "criteria": {c: None for c in ch}}
    if kind == "score":
        return {"type": "score", "instructions": q, "criteria": list(ch)}
    if kind == "verify":
        return {"type": "noul", "instructions": q}
    raise ValueError("unknown kind %r" % kind)


@torch.no_grad()
def forward(agent, state, qdef) -> np.ndarray:
    """Agent.system_one's forward pass for one question, returning raw logits."""
    from laya.common import QTYPES, build_sequence, collate_items, render_options

    q = agent._to_internal(qdef)
    seq, markers = build_sequence(agent.tok, state, q, agent.cfg.get("max_len", 512), agent.cfg.get("head_max_len", 192))
    if len(markers) != len(render_options(q)):
        raise ValueError("options exceed head_max_len")
    b = collate_items([[{"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]]}]], agent.tok.pad_token_id)
    dev = agent.device
    with torch.autocast(device_type=dev.type, dtype=agent.dtype, enabled=dev.type == "cuda"):
        logits, _ = agent.model(b["input_ids"].to(dev), b["attention_mask"].to(dev), b["marker_pos"].to(dev),
                                b["marker_mask"].to(dev), b["qtype"].to(dev))
    return logits[0, : len(markers)].float().cpu().numpy()


def shipped_probs(agent, qdef, z: np.ndarray) -> np.ndarray:
    from laya.common import QTYPES, temp_bucket

    qt = QTYPES[qdef["type"]]
    t = agent.temperature_by_options.get(temp_bucket(qt, len(z)), agent.temperature[qt])
    z = z / max(1e-3, float(t))
    p = np.exp(z - z.max())
    return p / p.sum()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True, choices=["laya", "laya-multilingual", "router"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32", "shipped"],
                    help="autocast dtype. fp16 (default) matches Laya's published T4 run exactly; "
                         "'shipped' is what laya picks on this GPU (bf16 on sm>=80, flips ~1/300 items)")
    a = ap.parse_args()

    if a.device.startswith("cuda"):
        # GB10 quirk: creating the CUDA context *after* Laya has built its model on the CPU fails
        # with "CUDA error: out of memory" when the box is near full, and Laya then silently
        # falls back to CPU. Creating the context first avoids it.
        torch.zeros(1, device=a.device)

    import laya

    items = [json.loads(l) for l in open(a.items) if l.strip()]
    if a.model == "router":
        router = laya.Router(standalone_repos=True, device=a.device).preload(["english", "multilingual"])

        def pick(st):
            key = router.route(st)["model"]
            return key, router.load(key)
    else:
        agent = laya.load(REPOS[a.model], device=a.device)
        agent.model.eval()
        pick = lambda st: (a.model, agent)
    loaded = list(router._agents.values()) if a.model == "router" else [agent]
    dt = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}.get(a.dtype)
    for ag in loaded:
        if dt is not None and ag.device.type == "cuda":
            ag.dtype = dt
    if any(ag.device.type != a.device.split(":")[0] for ag in loaded):
        sys.exit("laya fell back to CPU; refusing to report %s latency" % a.device)

    for it in items[: a.warmup]:
        _, ag = pick(it["state"])
        forward(ag, it["state"], to_qdef(it))

    sync = torch.cuda.synchronize if a.device.startswith("cuda") else (lambda: None)
    n_fail = 0
    with open(a.out, "w") as f:
        for it in items:
            qdef = to_qdef(it)
            sync()
            t0 = time.perf_counter()
            name, ag = pick(it["state"])
            try:
                z = forward(ag, it["state"], qdef)
            except ValueError as e:  # options do not fit Laya's head budget: uniform, flagged
                n_fail += 1
                row = {"id": it["id"], "probs": [1.0 / len(it["choices"])] * len(it["choices"]),
                       "latency_ms": 0.0, "error": str(e), "model": name}
                f.write(json.dumps(row) + "\n")
                continue
            p = shipped_probs(ag, qdef, z)
            sync()
            ms = (time.perf_counter() - t0) * 1000
            if qdef["type"] == "noul":  # Laya order is [false, true]; ours follows the item's choices
                yes_first = it["choices"][0].strip().lower().split(":")[0].split()[0] in ("yes", "true")
                if yes_first:
                    p, z = p[::-1], z[::-1]
            row = {"id": it["id"], "probs": [float(x) for x in p], "latency_ms": round(ms, 3),
                   "logits": [float(x) for x in z], "model": name}
            f.write(json.dumps(row) + "\n")
    print("wrote %d preds to %s (%d did not fit)" % (len(items), a.out, n_fail), file=sys.stderr)


if __name__ == "__main__":
    main()
