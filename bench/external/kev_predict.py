"""Run a Kev checkpoint (github.com/jaredpalmer/kev) over a Decima items JSONL and write per-item probabilities.

Runs in its own venv with Kev's own package; never import this from the decima package. Setup on the GX10
(aarch64 + GB10). Kev pins torch 2.8.0 / transformers 5.17.0 / peft 0.21.0 (its uv.lock); torch 2.8.0 has
no aarch64 cu128 wheel, the cu129 one is used:

    export PATH=$HOME/.local/bin:$PATH
    K=$HOME/venvs/kev
    git clone https://github.com/jaredpalmer/kev $K/repo && git -C $K/repo checkout 557598f
    uv venv --python 3.12 $K/venv
    VIRTUAL_ENV=$K/venv uv pip install --index-url https://download.pytorch.org/whl/cu129 "torch==2.8.0"
    VIRTUAL_ENV=$K/venv uv pip install "transformers==5.17.0" "peft==0.21.0" "accelerate==1.15.0" \
        "datasets==5.0.1" "pydantic==2.13.5" "numpy==2.5.3" "scikit-learn==1.9.1" "safetensors==0.8.0" \
        "huggingface-hub==1.32.0" "tokenizers==0.23.2"
    VIRTUAL_ENV=$K/venv uv pip install --no-deps -e $K/repo

    $K/venv/bin/python bench/external/kev_predict.py --items items.jsonl --model kev-0.5b --out preds.jsonl --device cuda

Input rows: {"id", "kind", "question", "choices", "state", ...} (bench/items.py).
Output rows: {"id", "probs": [aligned with choices], "latency_ms", "logits", "model", "state_truncated"}.

Loading goes through Kev's own `kev.checkpoint.Checkpoint(...).load()` (the path every Kev number uses: fp32,
LoRA merged in fp32, TF32 and fused SDPA off on CUDA as in `kev.predictors.LocalPredictor`), so newer
checkpoints (option isolation, hybrid Qwen3.5 row form) run exactly as Kev runs them. `probs` are RAW
(temperature 1.0): kev-0.5b's published numbers are raw, and newer checkpoints' carried temperature is
ignored so the scoreboard fits every system's temperature the same way; `logits` are raw too.

Mapping onto Kev's internal record (what `kev.api.to_record` produces from a /v1/systemone request):
    state  -> rendered state string, passed unchanged (render(str) is the identity)
    question -> "instr", unchanged
    choose / rank -> Choice: options = choices verbatim (a criterion with a null description renders as the
              bare key, so {choice: None} gives exactly the choice string; bench/suites_kev.py already stores
              Kev's rendered "key: description" strings)
    score  -> Score: options = the level texts in order (Kev renders levels verbatim)
    verify -> Noul: Kev's option order is ["no[: desc]", "yes[: desc]"]; we find the yes/no choice by its
              leading word (yes/true vs no/false), so ["yes", "no"] and flipped ["no", "yes"] both map
              correctly; choices that name neither are read as [yes, no] and rendered "yes: <c0>", "no: <c1>".
    The model itself has no question-type input: type only decides how options are rendered.
Context: Kev's training context (state <= 384 tokens, truncated like the published eval; branch <= 1024).
A record whose branch does not fit is retried with Kev's serving limits (8192/8192) and flagged "context":
"serve". `latency_ms` is one item per forward (batch 1), CUDA-synchronised, model warm.

--model: kev-0.5b | kev-0.6b | kev-0.8b | kev-4b | kev-8b | kev-9b (jaredpalmer/<name>), any Hub id
(optionally @revision) or a local run directory. Qwen3.5 checkpoints (0.8b/4b/9b) run the row form; without
flash-linear-attention / causal-conv1d installed transformers uses its (exact, slower) PyTorch reference kernels.
Memory on the GX10 (~18 GB free): kev-0.5b / kev-0.8b fp32 fit; kev-4b needs --dtype bf16 (~9 GB, untested
here); kev-8b / kev-9b do not fit alongside the vLLM workers.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("USE_TF", "0")

import torch

from kev.checkpoint import Checkpoint, LoadOptions
from kev.model import MAX_BRANCH, MAX_STATE, SERVE_MAX_BRANCH, SERVE_MAX_STATE

YES, NO = ("yes", "true"), ("no", "false")


def _lead(text: str) -> str:
    return text.strip().split(":")[0].split()[0].lower() if text.strip() else ""


def to_question(item: dict) -> tuple[dict, list[int]]:
    """-> (Kev internal question {"instr", "options", "label"}, perm) with Kev option j = item choice perm[j]."""
    kind, ch = item["kind"], list(item["choices"])
    if kind == "verify":
        if len(ch) != 2:
            raise ValueError(f"{item['id']}: verify needs exactly 2 choices")
        lead = [_lead(c) for c in ch]
        if lead[0] in NO and lead[1] in YES:
            yes, no = 1, 0
        else:
            yes, no = 0, 1
        opts = []
        for j, word in ((no, "no"), (yes, "yes")):
            opts.append(ch[j] if _lead(ch[j]) in (YES if word == "yes" else NO) else f"{word}: {ch[j]}")
        return {"instr": item["question"], "options": opts, "label": 0}, [no, yes]
    if kind not in ("choose", "rank", "score"):
        raise ValueError(f"{item['id']}: unknown kind {kind!r}")
    return {"instr": item["question"], "options": ch, "label": 0}, list(range(len(ch)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="kev-0.5b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"], help="fp32 = Kev's reported-numbers path")
    ap.add_argument("--limit", type=int, default=0, help="first N items only (smoke tests)")
    args = ap.parse_args()

    run = args.model if ("/" in args.model or os.path.isdir(args.model)) else f"jaredpalmer/{args.model}"
    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
    # bf16: keep the adapter unmerged, otherwise Kev loads the backbone in fp32 first to merge (2x the memory)
    opts = LoadOptions(dtype=torch.bfloat16 if args.dtype == "bf16" else torch.float32, merge=args.dtype == "fp32", temperature=1.0)
    ck = Checkpoint(run)
    tok, model = ck.load(args.device, opts)
    print(f"loaded {run} base={ck.meta.base} option_isolation={ck.meta.option_isolation} "
          f"hybrid={getattr(model, 'hybrid', False)} carried_T={ck.meta.temperature} (ignored)", file=sys.stderr, flush=True)

    items = [json.loads(l) for l in open(args.items) if l.strip()]
    if args.limit:
        items = items[: args.limit]
    sync = torch.cuda.synchronize if args.device == "cuda" else (lambda: None)

    with torch.no_grad():   # warm-up
        q, _ = to_question(items[0])
        model.forward(model.encode(tok, {"state": items[0]["state"], "questions": [q]}, max_state=MAX_STATE, max_branch=MAX_BRANCH))

    n_serve = 0
    t0 = time.time()
    with open(args.out, "w") as f, torch.no_grad():
        for n, it in enumerate(items, 1):
            q, perm = to_question(it)
            rec = {"state": it["state"], "questions": [q]}
            context = "train"
            try:
                enc = model.encode(tok, rec, max_state=MAX_STATE, max_branch=MAX_BRANCH)
            except ValueError:
                enc = model.encode(tok, rec, max_state=SERVE_MAX_STATE, max_branch=SERVE_MAX_BRANCH)
                context = "serve"; n_serve += 1
            sync(); start = time.perf_counter()
            z = model.forward(enc)[0].float()
            sync(); ms = 1000 * (time.perf_counter() - start)
            z = z.cpu().tolist()
            logits = [0.0] * len(z)
            for j, src in enumerate(perm):
                logits[src] = z[j]
            probs = torch.softmax(torch.tensor(logits), -1).tolist()
            f.write(json.dumps({"id": it["id"], "probs": probs, "latency_ms": ms, "logits": logits, "model": args.model,
                                "state_truncated": bool(enc.get("state_truncated")), "context": context}) + "\n")
            if n % 200 == 0:
                print(f"  {n}/{len(items)}  {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    print(f"wrote {len(items)} preds -> {args.out} ({n_serve} needed serving context)", file=sys.stderr)


if __name__ == "__main__":
    main()
