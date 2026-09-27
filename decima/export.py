"""ONNX export — two graphs, because the runtime caches them differently.

    encoder.onnx   (input_ids, attention_mask) → token states        run once per state,
                                                                      once per choice set
    scorer.onnx    (state_h, state_mask, choice_h, choice_mask) → (scores, z)   per decision

int8 quantisation and the CPU latency measurement happen on x86, not here — the Grace
CPU is not the deployment target. The ordinal head is tiny and kept in numpy on the
runtime side (`ordinal_probs`), so the scorer graph stays one shape for every kind.

    uv run python -m decima.export checkpoints/v0/best --out export/v0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from .model import DecimaModel


class _Encoder(nn.Module):
    """Two positional inputs, keyword call inside — the legacy exporter maps positional args
    onto the transformers signature and collides with `use_cache` otherwise."""

    def __init__(self, enc):
        super().__init__()
        self.enc = enc

    def forward(self, input_ids, attention_mask):
        return self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


class _Scorer(nn.Module):
    def __init__(self, m: DecimaModel):
        super().__init__()
        self.m = m

    def forward(self, state_h, state_mask, choice_h, choice_mask):
        n = choice_h.shape[0]
        owner = torch.zeros(n, dtype=torch.long)          # one state per call at inference
        s, z = self.m.choice_scores(state_h, state_mask, choice_h, choice_mask, owner)
        return s, z


def export(checkpoint: str, out: str, opset: int = 17) -> None:
    out_p = Path(out); out_p.mkdir(parents=True, exist_ok=True)
    m = DecimaModel.load(checkpoint).float().eval()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(Path(checkpoint) / "encoder")
    tok.save_pretrained(out_p / "tokenizer")

    enc = tok(["query: hello world"], return_tensors="pt")
    torch.onnx.export(
        _Encoder(m.encoder), (enc["input_ids"], enc["attention_mask"]), out_p / "encoder.onnx",
        input_names=["input_ids", "attention_mask"], output_names=["token_states"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"}, "token_states": {0: "batch", 1: "seq"}},
        opset_version=opset, dynamo=False,
    )
    d = m.encoder.config.hidden_size
    sh, sm = torch.randn(1, 12, d), torch.ones(1, 12, dtype=torch.long)
    ch, cm = torch.randn(3, 7, d), torch.ones(3, 7, dtype=torch.long)
    torch.onnx.export(
        _Scorer(m), (sh, sm, ch, cm), out_p / "scorer.onnx",
        input_names=["state_h", "state_mask", "choice_h", "choice_mask"], output_names=["scores", "z"],
        dynamic_axes={"state_h": {1: "state_seq"}, "state_mask": {1: "state_seq"},
                      "choice_h": {0: "n_choices", 1: "choice_seq"}, "choice_mask": {0: "n_choices", 1: "choice_seq"},
                      "scores": {0: "n_choices"}, "z": {0: "n_choices"}},
        opset_version=opset, dynamo=False,
    )
    # what the runtime needs besides the graphs: temperature, the ordinal head weights, prefixes
    head = {"ord_g_w": m.ord_g.weight.squeeze(0).tolist(), "ord_g_b": float(m.ord_g.bias),
            "ord_gap_w": m.ord_gap.weight.squeeze(0).tolist(), "ord_gap_b": float(m.ord_gap.bias)}
    (out_p / "decima.json").write_text(json.dumps({**json.loads((Path(checkpoint) / "decima.json").read_text()),
                                                   "ordinal": head}, indent=2))
    print(f"exported → {out_p}: " + ", ".join(f"{p.name} {p.stat().st_size/1e6:.1f} MB" for p in sorted(out_p.glob('*.onnx'))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    export(args.checkpoint, args.out)


if __name__ == "__main__":
    main()
