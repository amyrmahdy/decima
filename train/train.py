"""Train the student.

    uv run python -m train.train --data data/teacher/*.jsonl --out checkpoints/pilot --epochs 3

Loss = KL(teacher ‖ student) + λ·NLL(gold), per kind:
  choose / verify / score   soft-target cross-entropy against the teacher vector (the
                            KL up to the teacher's constant entropy) over the kind's own
                            log-probs — the ordinal head's for score, so the thresholds
                            learn from the teacher's mass on adjacent levels
  rank                      per-choice binary cross-entropy against the teacher's
                            independent probabilities (no NLL term: gold is just argmax)
Then temperature scaling on the hold-out slice, written into the checkpoint config.

Crash safety: every --ckpt-minutes a single file <out>/resume.pt (model + optimizer + schedule +
position in the epoch's batch order + RNG) is written atomically; --resume (the default in
scripts/train_v1.sh) continues from it, skipping the batches already done. Before this, a run only
saved at epoch end — a 12.5 h Decima-base run was lost at 96 % to a host-RAM OOM kill.
"""

from __future__ import annotations

import argparse
import glob
import os
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from decima.calibration import fit_temperature
from decima.model import DecimaConfig, DecimaModel
from train.data import Collator, load_examples, make_batches, split


def loss_fn(logp: torch.Tensor, batch: dict, nll_weight: float) -> tuple[torch.Tensor, dict]:
    """logp [B, max_n] per-kind log-probs (−inf on padding)."""
    q, gold = batch["targets"], batch["gold"]
    valid = logp > float("-inf")
    lp = logp.masked_fill(~valid, 0.0)
    is_rank = torch.tensor([k == "rank" for k in batch["kinds"]], device=logp.device)
    # soft cross-entropy for the normalised kinds
    ce = -(q * lp).sum(1)
    nll = -lp.gather(1, gold.unsqueeze(1)).squeeze(1)
    # BCE for rank: lp = log σ(s); log(1−σ) = log1p(−exp(lp))
    log1m = torch.log1p(-torch.exp(lp).clamp(max=1 - 1e-6))
    bce = -(q * lp + (1 - q) * log1m).masked_fill(~valid, 0.0).sum(1) / valid.sum(1)
    per_ex = torch.where(is_rank, bce, ce + nll_weight * nll)
    loss = per_ex.mean()
    with torch.no_grad():
        n_soft = int((~is_rank).sum())
        parts = {"ce": float(ce[~is_rank].mean()) if n_soft else 0.0, "nll": float(nll[~is_rank].mean()) if n_soft else 0.0,
                 "bce": float(bce[is_rank].mean()) if n_soft < len(is_rank) else 0.0}
    return loss, parts


@torch.no_grad()
def evaluate(model, collate, batches, device) -> tuple[dict, dict]:
    """Hold-out metrics per kind + raw scores for temperature fitting (choose/verify)."""
    model.eval()
    acc, agree, kl, n = {}, {}, {}, {}
    cal_logits, cal_labels = [], []
    for b in batches:
        bt = collate(b, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            Hs = model.encode(bt["s_ids"], bt["s_mask"]); Hc = model.encode(bt["c_ids"], bt["c_mask"])
            s, z = model.choice_scores(Hs, bt["s_mask"], Hc, bt["c_mask"], bt["owner"])
        s, z = s.float(), z.float()
        logp = model.log_probs(s, z, bt["owner"], Hs.shape[0], bt["kinds"])
        pred = logp.argmax(1)
        targ_arg = bt["targets"].argmax(1)
        for i, k in enumerate(bt["kinds"]):
            n[k] = n.get(k, 0) + 1
            acc[k] = acc.get(k, 0) + int(pred[i] == bt["gold"][i])
            agree[k] = agree.get(k, 0) + int(pred[i] == targ_arg[i])
            if k != "rank":
                q = bt["targets"][i]; lp = logp[i]
                m = q > 0
                kl[k] = kl.get(k, 0.0) + float((q[m] * (torch.log(q[m]) - lp[m])).sum())
        s_pad = model.group(s, bt["owner"], Hs.shape[0], float("-inf"))
        for i, k in enumerate(bt["kinds"]):
            if k in ("choose", "verify"):
                cal_logits.append(s_pad[i].cpu().numpy()); cal_labels.append(int(bt["gold"][i]))
    model.train()
    metrics = {k: {"n": n[k], "acc": acc[k] / n[k], "teacher_agree": agree[k] / n[k],
                   **({"kl": kl[k] / n[k]} if k in kl else {})} for k in n}
    tot = sum(n.values())
    metrics["all"] = {"n": tot, "acc": sum(acc.values()) / tot, "teacher_agree": sum(agree.values()) / tot}
    # pad calibration logits to a common width
    if cal_logits:
        w = max(len(x) for x in cal_logits)
        L = np.full((len(cal_logits), w), -1e9, dtype=np.float32)
        for i, x in enumerate(cal_logits):
            L[i, : len(x)] = x
        cal = {"logits": L, "labels": np.array(cal_labels)}
    else:
        cal = {}
    return metrics, cal


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--head-lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--choice-budget", type=int, default=384)
    ap.add_argument("--max-states", type=int, default=48)
    ap.add_argument("--state-token-budget", type=int, default=0, help="cap a batch's padded state tokens (0 = off); needed for long states")
    ap.add_argument("--nll-weight", type=float, default=0.3)
    ap.add_argument("--holdout", type=float, default=0.05)
    ap.add_argument("--scorer-layers", type=int, default=2)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--eval-batches", type=int, default=400)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    # model shape — defaults reproduce V0
    ap.add_argument("--backbone", default=DecimaConfig.backbone)
    ap.add_argument("--max-state-tokens", type=int, default=DecimaConfig.max_state_tokens)
    ap.add_argument("--max-choice-tokens", type=int, default=DecimaConfig.max_choice_tokens)
    ap.add_argument("--question-in-state", action="store_true", help="encode 'question\\n state' together (V1)")
    ap.add_argument("--no-question-in-choices", action="store_true", help="choice texts without the question (V1)")
    ap.add_argument("--state-prefix", default=DecimaConfig.state_prefix)
    ap.add_argument("--choice-prefix", default=DecimaConfig.choice_prefix)
    ap.add_argument("--init", default=None, help="checkpoint dir to start from (same backbone); its shape flags are overridden by these")
    ap.add_argument("--ckpt-minutes", type=float, default=30, help="write <out>/resume.pt this often (0 = never)")
    ap.add_argument("--resume", action="store_true", help="continue from <out>/resume.pt if it exists")
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    paths = [p for pat in args.data for p in sorted(glob.glob(pat))]

    from transformers import AutoTokenizer

    shape = dict(scorer_layers=args.scorer_layers, max_state_tokens=args.max_state_tokens, max_choice_tokens=args.max_choice_tokens,
                 question_in_state=args.question_in_state, question_in_choices=not args.no_question_in_choices,
                 state_prefix=args.state_prefix, choice_prefix=args.choice_prefix)
    if args.init:
        model = DecimaModel.load(args.init, args.device)
        for k, v in shape.items():
            if k != "scorer_layers":
                setattr(model.cfg, k, v)
        model.cfg.temperature = 1.0
        cfg, tok = model.cfg, AutoTokenizer.from_pretrained(Path(args.init) / "encoder")
        model.train()
    else:
        cfg = DecimaConfig(backbone=args.backbone, **shape)
        model = DecimaModel(cfg).to(args.device)
        tok = AutoTokenizer.from_pretrained(cfg.backbone)
    collate = Collator(tok, cfg)
    # data after the model is on the device: on the GB10 (unified memory) a CUDA context created after
    # ~1M examples are in RAM has failed to allocate even the weights
    examples = load_examples(paths)
    train_ex, hold_ex = split(examples, args.holdout)
    del examples
    print(f"{len(train_ex) + len(hold_ex)} examples from {len(paths)} files → train {len(train_ex)} / holdout {len(hold_ex)}", flush=True)
    counts = model.param_counts()
    print("params:", counts, flush=True)

    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    body_params = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    opt = torch.optim.AdamW([{"params": body_params, "lr": args.lr}, {"params": head_params, "lr": args.head_lr}],
                            weight_decay=args.weight_decay, betas=(0.9, 0.98))
    steps_per_epoch = len(make_batches(train_ex, args.choice_budget, args.max_states, 0, args.state_token_budget or None, args.max_state_tokens))
    total = int(steps_per_epoch * args.epochs)
    warm = int(total * args.warmup)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / max(1, warm) if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    name = out.name
    log = Path(f"runs/train-{name}.jsonl").open("a")
    def emit(rec):
        rec["t"] = datetime.now(timezone.utc).isoformat(timespec="seconds"); log.write(json.dumps(rec) + "\n"); log.flush()
    emit({"event": "start", "args": vars(args), "params": counts, "n_train": len(train_ex), "n_holdout": len(hold_ex), "steps": total})

    hold_batches = make_batches(hold_ex, args.choice_budget, args.max_states, 0, args.state_token_budget or None, args.max_state_tokens)[: args.eval_batches]
    step, t0, best = 0, time.time(), -1.0
    epoch, skip_to, spent = 0, 0, 0.0
    resume_p = out / "resume.pt"

    def save_resume(bi: int) -> None:
        tmp = out / "resume.pt.tmp"
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "step": step, "epoch": epoch, "bi": bi, "best": best, "spent": spent + time.time() - t0,
                    "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    "total": total}, tmp)
        os.replace(tmp, resume_p)                     # atomic: a crash mid-write leaves the previous file intact

    if args.resume and resume_p.exists():
        ck = torch.load(resume_p, map_location=args.device, weights_only=False)
        if ck["total"] != total:
            raise SystemExit(f"resume.pt was written for {ck['total']} steps, this run has {total}: different data or flags")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        step, epoch, skip_to, best, spent = ck["step"], ck["epoch"], ck["bi"], ck["best"], ck["spent"]
        torch.set_rng_state(ck["rng"].cpu())
        if ck["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([r.cpu() for r in ck["cuda_rng"]])
        del ck
        emit({"event": "resume", "step": step, "epoch": epoch, "batch": skip_to})
        print(f"resumed from {resume_p}: step {step}/{total}, epoch {epoch}, batch {skip_to}", flush=True)
    last_ckpt = time.time()
    model.train()
    while step < total:
        for bi, b in enumerate(make_batches(train_ex, args.choice_budget, args.max_states, epoch + 1, args.state_token_budget or None, args.max_state_tokens)):
            if step >= total:
                break
            if bi < skip_to:                          # already trained on before the resume
                continue
            bt = collate(b, args.device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
                Hs = model.encode(bt["s_ids"], bt["s_mask"]); Hc = model.encode(bt["c_ids"], bt["c_mask"])
                s, z = model.choice_scores(Hs, bt["s_mask"], Hc, bt["c_mask"], bt["owner"])
            logp = model.log_probs(s.float(), z.float(), bt["owner"], Hs.shape[0], bt["kinds"])
            loss, parts = loss_fn(logp, bt, args.nll_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            if step % args.log_every == 0:
                rec = {"event": "step", "step": step, "epoch": round(step / steps_per_epoch, 3), "loss": round(float(loss.detach()), 4),
                       **{k: round(v, 4) for k, v in parts.items()}, "gnorm": round(float(gn.detach()), 3),
                       "lr": sched.get_last_lr()[0], "elapsed_s": round(time.time() - t0)}
                emit(rec)
                print(f"  step {step}/{total} ep {rec['epoch']:.2f} loss {rec['loss']:.4f} ce {parts['ce']:.3f} nll {parts['nll']:.3f} "
                      f"bce {parts['bce']:.3f} gn {rec['gnorm']:.2f} lr {rec['lr']:.2e} {rec['elapsed_s']}s", flush=True)
            if args.ckpt_minutes and time.time() - last_ckpt >= args.ckpt_minutes * 60:
                save_resume(bi + 1); last_ckpt = time.time()
                emit({"event": "checkpoint", "step": step})
        skip_to = 0
        epoch += 1
        metrics, cal = evaluate(model, collate, hold_batches, args.device)
        emit({"event": "eval", "epoch": epoch, "step": step, "metrics": metrics})
        print(f"epoch {epoch}: " + "  ".join(f"{k} acc={v['acc']:.3f} agree={v['teacher_agree']:.3f}" + (f" kl={v['kl']:.3f}" if "kl" in v else "")
                                            for k, v in metrics.items()), flush=True)
        score = metrics["all"]["acc"]
        model.save(out / "last"); tok.save_pretrained(out / "last" / "encoder")
        if score > best:
            best = score
            model.save(out / "best"); tok.save_pretrained(out / "best" / "encoder")

    # ---- temperature on the hold-out slice (choose/verify scores), written into both checkpoints
    T = 1.0
    if cal:
        T = fit_temperature(cal["logits"], cal["labels"])
    for sub in ("last", "best"):
        cfgp = out / sub / "decima.json"
        if cfgp.exists():
            c = json.loads(cfgp.read_text()); c["temperature"] = T; cfgp.write_text(json.dumps(c, indent=2))
    emit({"event": "done", "temperature": T, "best_holdout_acc": best, "elapsed_s": round(spent + time.time() - t0)})
    resume_p.unlink(missing_ok=True)                  # finished: best/ and last/ are the artefacts
    (out / "resume.pt.tmp").unlink(missing_ok=True)
    print(f"done. T={T:.3f}  best holdout acc={best:.3f}  → {out}/best", flush=True)


if __name__ == "__main__":
    main()
