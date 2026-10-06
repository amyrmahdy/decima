"""Teacher JSONL → training batches.

A batch is B states plus every choice of every one of those states, flattened: the
scorer takes (state tokens, choice tokens, owner index). Choice counts range from 2 to
100, so batches are packed by a *choice budget*, not a fixed B — a batch of 100-choice
decisions holds a handful of states, a batch of verify questions holds many.

The 5 % hold-out is chosen by example id hash, so it is the same slice on every run and
survives regeneration; it is used for the epoch metrics and for fitting temperature,
never for weights.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import torch

from decima.model import DecimaConfig

KINDS = ("choose", "score", "verify", "rank")
KEEP = ("id", "kind", "state", "question", "choices", "gold", "probs", "state_lang", "choice_lang")


def load_examples(paths: list[str | Path]) -> list[dict]:
    out = []
    for p in paths:
        for line in Path(p).open():
            r = json.loads(line)
            if r.get("source") in (None, "empty") or "probs" not in r:
                continue
            if r["kind"] not in KINDS or len(r["probs"]) != len(r["choices"]) or not (0 <= r["gold"] < len(r["choices"])):
                continue
            out.append({k: r[k] for k in KEEP})   # drop provenance fields: 1M rows share unified memory with the GPU
    return out


def is_holdout(ex: dict, frac: float = 0.05) -> bool:
    h = int(hashlib.sha1(ex["id"].encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return h < frac


def split(examples: list[dict], frac: float = 0.05) -> tuple[list[dict], list[dict]]:
    tr, ho = [], []
    for e in examples:
        (ho if is_holdout(e, frac) else tr).append(e)
    return tr, ho


def make_batches(examples: list[dict], choice_budget: int = 384, max_states: int = 48, seed: int = 0,
                 token_budget: int | None = None, max_state_tokens: int = 512) -> list[list[dict]]:
    """Greedy packing by choice count; order shuffled per call (pass epoch as seed).

    With `token_budget`, a batch also stops when its states would exceed that many (estimated) tokens, counted at the
    longest state in the batch since the collator pads to it; states are then grouped by length within each chunk, so
    long agent contexts batch with each other instead of padding short ones. Without it, the packing is unchanged."""
    rng = random.Random(seed)
    idx = list(range(len(examples)))
    rng.shuffle(idx)
    est = (lambda i: min(max_state_tokens, 8 + len(examples[i]["state"]) // 3 + len(examples[i]["question"]) // 4)) if token_budget else None
    # sort within coarse chunks so batches are homogeneous in choice count (less padding), yet still random
    chunk = 2048
    batches, cur, used, longest = [], [], 0, 0
    for c0 in range(0, len(idx), chunk):
        key = (lambda i: (est(i) // 128, len(examples[i]["choices"]))) if token_budget else (lambda i: len(examples[i]["choices"]))
        block = sorted(idx[c0 : c0 + chunk], key=key)
        for i in block:
            n = len(examples[i]["choices"])
            t = est(i) if token_budget else 0
            if cur and (used + n > choice_budget or len(cur) >= max_states
                        or (token_budget and max(longest, t) * (len(cur) + 1) > token_budget)):
                batches.append(cur); cur, used, longest = [], 0, 0
            cur.append(examples[i]); used += n; longest = max(longest, t)
    if cur:
        batches.append(cur)
    rng.shuffle(batches)
    return batches


class Collator:
    def __init__(self, tokenizer, cfg: DecimaConfig):
        self.tok, self.cfg = tokenizer, cfg
        self.ls, self.lc = cfg.max_state_tokens, cfg.max_choice_tokens

    def __call__(self, batch: list[dict], device: str = "cpu") -> dict:
        states = [self.cfg.state_of(e["state"], e["question"], e["state_lang"]) for e in batch]
        choices, owner, targets, gold, kinds = [], [], [], [], []
        max_n = max(len(e["choices"]) for e in batch)
        for b, e in enumerate(batch):
            for c in self.cfg.choices_of(self.tok, e["question"], e["choices"], e["choice_lang"]):
                choices.append(c)
                owner.append(b)
            p = list(e["probs"]) + [0.0] * (max_n - len(e["choices"]))
            targets.append(p); gold.append(e["gold"]); kinds.append(e["kind"])
        s = self.tok(states, padding=True, truncation=True, max_length=self.ls, return_tensors="pt")
        c = self.tok(choices, padding=True, truncation=True, max_length=self.lc, return_tensors="pt")
        return {
            "s_ids": s["input_ids"].to(device), "s_mask": s["attention_mask"].to(device),
            "c_ids": c["input_ids"].to(device), "c_mask": c["attention_mask"].to(device),
            "owner": torch.tensor(owner, device=device),
            "targets": torch.tensor(targets, device=device), "gold": torch.tensor(gold, device=device),
            "kinds": kinds,
        }
