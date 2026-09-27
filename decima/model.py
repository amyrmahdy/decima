"""The Decima student: e5-small body + late-interaction option scorer.

    state ──encoder──▶ token states H_s              (once per decision, cacheable)
    "question · choice_k" ──encoder──▶ token states H_k    (once per choice set, cacheable)

Where the question goes is a config choice. V0 put it only on the choice side, so a question
that carries content ("Hypothesis: …", a BoolQ question) never met the state inside the
encoder — NLI and yes/no reading sat near chance. `question_in_state` encodes
"question \n state" together (the state cache is then per (state, question));
`question_in_choices=False` leaves each choice text bare, which also frees the choice token
budget and lets one choice-set cache serve every question.
    H_k ──[self-attn · cross-attn over H_s · FFN] × 2──▶ pooled z_k ──▶ score s_k

Why late interaction rather than a cross-encoder: choices never share a context
window, so a 1,000-choice decision costs 1,000 cheap scorer passes over cached choice
states, not a 1,000-way prompt. Why not a bi-encoder: a dot product cannot read the
state *conditioned on the choice* — "refund" vs "refund status" needs attention over
the state tokens. The bi-encoder similarity survives as a skip term so training starts
from the zero-shot baseline instead of from noise.

Output heads, by kind (the choice texts are the same kind of input in every case):
  choose / verify   softmax over s_k
  score             cumulative-link ordinal head: P(y > k) = σ(g − θ_k) with a shared
                    latent g and thresholds θ_k that are increasing by construction and
                    derived from the level texts, so "low/medium/high" and a 7-point
                    scale get their own spacing
  rank              σ(s_k) independently per choice
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .normalize import normalize
from .types import Decision, Question

BACKBONE = "intfloat/multilingual-e5-small"


@dataclass
class DecimaConfig:
    backbone: str = BACKBONE
    scorer_layers: int = 2
    heads: int = 6
    ffn: int = 1536
    dropout: float = 0.1
    max_state_tokens: int = 256
    max_choice_tokens: int = 48
    temperature: float = 1.0          # post-hoc calibration, set by train/calibrate
    question_in_state: bool = False   # V0: False (checkpoints without the key load as V0)
    question_in_choices: bool = True
    state_prefix: str = "query: "     # e5's training prefixes; other backbones may want ""
    choice_prefix: str = "passage: "

    def state_of(self, state: str, question: str, lang: str) -> str:
        return state_text(state, lang, question if self.question_in_state else None, self.state_prefix)

    def choice_of(self, question: str, choice: str, lang: str) -> str:
        return choice_text(question if self.question_in_choices else "", choice, lang, self.choice_prefix)


class Attention(nn.Module):
    """Plain multi-head attention with explicit reshapes. nn.MultiheadAttention works for
    training but the ONNX exporter bakes the trace's shapes into it; this traces dynamically."""

    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.q, self.k, self.v, self.o = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mem, mem_pad):
        """x [N, Lq, d] queries · mem [N, Lk, d] keys/values · mem_pad [N, Lk] True where padding."""
        N, Lq, _ = x.shape
        Lk = mem.shape[1]
        q = self.q(x).view(N, Lq, self.h, self.dk).transpose(1, 2)          # [N, h, Lq, dk]
        k = self.k(mem).view(N, Lk, self.h, self.dk).transpose(1, 2)
        v = self.v(mem).view(N, Lk, self.h, self.dk).transpose(1, 2)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.dk)                 # [N, h, Lq, Lk]
        att = att.masked_fill(mem_pad[:, None, None, :], -1e4)
        att = self.drop(att.softmax(-1))
        return self.o((att @ v).transpose(1, 2).reshape(N, Lq, -1))


class ScorerLayer(nn.Module):
    """One decoder-style block: choice tokens attend among themselves, then over the state."""

    def __init__(self, d: int, heads: int, ffn: int, dropout: float):
        super().__init__()
        self.self_attn = Attention(d, heads, dropout)
        self.cross_attn = Attention(d, heads, dropout)
        self.ff = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn, d))
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, x_pad, mem, mem_pad):
        h = self.n1(x)
        x = x + self.drop(self.self_attn(h, h, x_pad))
        h = self.n2(x)
        x = x + self.drop(self.cross_attn(h, mem, mem_pad))
        return x + self.drop(self.ff(self.n3(x)))


def masked_mean(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).to(h.dtype)
    return (h * m).sum(1) / m.sum(1).clamp(min=1.0)


class DecimaModel(nn.Module):
    def __init__(self, cfg: DecimaConfig):
        super().__init__()
        from transformers import AutoModel

        self.cfg = cfg
        # fp32 master weights: some backbones ship bf16 (granite r2) and transformers 5 keeps the stored dtype
        self.encoder = AutoModel.from_pretrained(cfg.backbone, dtype=torch.float32)
        d = self.encoder.config.hidden_size
        self.layers = nn.ModuleList(ScorerLayer(d, cfg.heads, cfg.ffn, cfg.dropout) for _ in range(cfg.scorer_layers))
        self.out_norm = nn.LayerNorm(d)
        self.score = nn.Linear(d, 1)
        self.sim_scale = nn.Parameter(torch.tensor(20.0))   # bi-encoder skip: cos / 0.05, as in the baseline
        self.sim_center = nn.Parameter(torch.tensor(0.85))  # e5 cosines sit near 0.85; centring keeps σ(s) and the ordinal latent sane at init
        # ordinal head: latent g from the mean attended level vector, threshold gaps from each level's own vector
        self.ord_g = nn.Linear(d, 1)
        self.ord_gap = nn.Linear(d, 1)
        nn.init.zeros_(self.score.weight); nn.init.zeros_(self.score.bias)   # start exactly at the bi-encoder baseline

    # ------------------------------------------------------------------ encoding
    def encode(self, input_ids, attention_mask):
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    def choice_scores(self, Hs, ms, Hc, mc, owner):
        """Hs [B, Ls, d] state tokens · ms [B, Ls] mask · Hc [N, Lc, d] choice tokens · mc [N, Lc] ·
        owner [N] index of the state each choice belongs to → (scores [N], z [N, d])."""
        mem, mem_pad = Hs[owner], ~ms[owner].bool()
        x, x_pad = Hc, ~mc.bool()
        for layer in self.layers:
            x = layer(x, x_pad, mem, mem_pad)
        z = masked_mean(self.out_norm(x), mc)
        s = self.score(z).squeeze(-1)
        sim = F.cosine_similarity(masked_mean(Hs, ms)[owner], masked_mean(Hc, mc), dim=-1)
        return s + self.sim_scale * (sim - self.sim_center), z

    # ------------------------------------------------------------------ heads
    @staticmethod
    def group(values: torch.Tensor, owner: torch.Tensor, B: int, fill: float) -> torch.Tensor:
        """Flat per-choice values → padded [B, max_n]."""
        counts = torch.bincount(owner, minlength=B)
        max_n = int(counts.max())
        pos = torch.arange(len(owner), device=owner.device) - torch.cumsum(counts, 0)[owner] + counts[owner]
        out = values.new_full((B, max_n), fill)
        out[owner, pos] = values
        return out

    def ordinal_log_probs(self, s_pad: torch.Tensor, z_pad: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Cumulative-link over ordered levels. z_pad [B, K, d], valid [B, K].
        g = shared latent (mean level vector + mean score), θ_k = cumsum of softplus gaps → increasing.
        Returns log P(level = k), padded with -inf."""
        K = z_pad.shape[1]
        zm = (z_pad * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True).clamp(min=1)
        # latent = expected level index under the per-level scores (centred), plus a learned state term;
        # bounded in ±(K−1)/2 so the init lands mid-scale instead of on an end level
        idx = torch.arange(K, device=s_pad.device, dtype=s_pad.dtype)
        n_valid = valid.sum(1, keepdim=True).to(s_pad.dtype)
        expected = (F.softmax(s_pad, 1) * idx).sum(1, keepdim=True) - (n_valid - 1) / 2
        g = self.ord_g(zm).squeeze(-1) + expected.squeeze(1)
        gaps = F.softplus(self.ord_gap(z_pad).squeeze(-1)) + 1e-3                     # [B, K] strictly positive
        theta = torch.cumsum(gaps, 1)[:, :-1] - gaps.sum(1, keepdim=True) / 2       # centred thresholds θ_0 < … < θ_{K-2}
        c = g.unsqueeze(1) - theta                                                    # [B, K-1] logits of P(y > k)
        gt = F.logsigmoid(c)                                                          # log P(y > k)
        le = F.logsigmoid(-c)                                                         # log P(y ≤ k)
        # P(y = k) = P(y > k-1) − P(y > k); as a log via P(y>k-1)·(1 − σ(c_k)/σ(c_{k-1}))
        # numerically: p_k = σ(c_{k-1}) − σ(c_k), with σ(c_{-1}) = 1 and σ(c_{K-1}) = 0
        sig = torch.sigmoid(c)
        one = torch.ones_like(sig[:, :1]); zero = torch.zeros_like(one)
        p = torch.cat([one, sig], 1) - torch.cat([sig, zero], 1)                     # [B, K], monotone thresholds ⇒ ≥ 0
        logp = torch.log(p.clamp(min=1e-7))
        # levels beyond a shorter example's K are padding: renormalise over valid ones
        logp = logp.masked_fill(~valid, float("-inf"))
        return logp - torch.logsumexp(logp, 1, keepdim=True)

    def log_probs(self, s: torch.Tensor, z: torch.Tensor, owner: torch.Tensor, B: int, kinds: list[str],
                  temperature: float = 1.0) -> torch.Tensor:
        """→ [B, max_n] log-probabilities per kind (rank: log σ(s), not normalised).
        `temperature` divides the scores before any head, so one scalar calibrates all three."""
        s_pad = self.group(s / temperature, owner, B, float("-inf"))
        valid = s_pad > float("-inf")
        soft = F.log_softmax(s_pad, 1)
        kinds_t = torch.tensor([{"choose": 0, "verify": 0, "score": 1, "rank": 2}[k] for k in kinds], device=s.device)
        out = soft
        if (kinds_t == 1).any():
            z_pad = self._group_vec(z, owner, B)
            ordi = self.ordinal_log_probs(s_pad, z_pad, valid)
            out = torch.where((kinds_t == 1).unsqueeze(1), ordi, out)
        if (kinds_t == 2).any():
            out = torch.where((kinds_t == 2).unsqueeze(1), F.logsigmoid(s_pad).masked_fill(~valid, float("-inf")), out)
        return out

    def _group_vec(self, z: torch.Tensor, owner: torch.Tensor, B: int) -> torch.Tensor:
        counts = torch.bincount(owner, minlength=B)
        max_n = int(counts.max())
        pos = torch.arange(len(owner), device=owner.device) - torch.cumsum(counts, 0)[owner] + counts[owner]
        out = z.new_zeros((B, max_n, z.shape[-1]))
        out[owner, pos] = z
        return out

    def forward(self, batch: dict) -> torch.Tensor:
        Hs = self.encode(batch["s_ids"], batch["s_mask"])
        Hc = self.encode(batch["c_ids"], batch["c_mask"])
        s, z = self.choice_scores(Hs, batch["s_mask"], Hc, batch["c_mask"], batch["owner"])
        return self.log_probs(s, z, batch["owner"], Hs.shape[0], batch["kinds"])

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> None:
        p = Path(path); p.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(p / "encoder")
        head = {k: v for k, v in self.state_dict().items() if not k.startswith("encoder.")}
        torch.save(head, p / "head.pt")
        (p / "decima.json").write_text(json.dumps(asdict(self.cfg), indent=2))

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "DecimaModel":
        p = Path(path)
        raw = json.loads((p / "decima.json").read_text())
        known = DecimaConfig.__dataclass_fields__
        cfg = DecimaConfig(**{k: v for k, v in raw.items() if k in known})   # extra keys (e.g. "soup" provenance) are metadata
        cfg.backbone = str(p / "encoder")
        m = cls(cfg)
        m.load_state_dict(torch.load(p / "head.pt", map_location="cpu"), strict=False)
        return m.to(device).eval()

    def param_counts(self) -> dict[str, int]:
        emb = self.encoder.get_input_embeddings().weight.numel()   # e5: word_embeddings · ModernBERT: tok_embeddings
        enc = sum(x.numel() for x in self.encoder.parameters()) - emb
        head = sum(x.numel() for n, x in self.named_parameters() if not n.startswith("encoder."))
        return {"vocab_table": emb, "transformer_body": enc, "head": head, "body+head": enc + head}


# ---------------------------------------------------------------------- inference

def state_text(s: str, lang: str, question: str | None = None, prefix: str = "query: ") -> str:
    """Question first, so truncation of a long state cuts its tail, never the question."""
    return prefix + normalize(f"{question}\n{s}" if question else s, lang)


def choice_text(question: str, c: str, lang: str, prefix: str = "passage: ") -> str:
    return prefix + normalize(f"{question} {c}".strip(), lang)


class Decima:
    """Inference wrapper with the same surface as the baseline: logits(states, q) / decide(state, q).
    Choice token states are cached per (question, choices, lang) — the steady-state case."""

    def __init__(self, path: str | Path, device: str = "cpu"):
        from transformers import AutoTokenizer

        self.model = DecimaModel.load(path, device)
        self.tok = AutoTokenizer.from_pretrained(Path(path) / "encoder")
        self.device = device
        self.cfg = self.model.cfg
        self._cache: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
        self.model_name = str(path)

    @torch.no_grad()
    def _encode(self, texts: list[str], max_len: int, batch_size: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
        Hs, Ms = [], []
        for i in range(0, len(texts), batch_size):
            b = self.tok(texts[i : i + batch_size], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(self.device)
            Hs.append(self.model.encode(b["input_ids"], b["attention_mask"])); Ms.append(b["attention_mask"])
        L = max(h.shape[1] for h in Hs)
        Hs = [F.pad(h, (0, 0, 0, L - h.shape[1])) for h in Hs]
        Ms = [F.pad(m, (0, L - m.shape[1])) for m in Ms]
        return torch.cat(Hs), torch.cat(Ms)

    def _choices(self, q: Question):
        key = (q.text if self.cfg.question_in_choices else "", tuple(q.choices), q.lang)
        if key not in self._cache:
            if len(self._cache) > 256:
                self._cache.clear()
            self._cache[key] = self._encode([self.cfg.choice_of(q.text, c, q.lang) for c in q.choices], self.cfg.max_choice_tokens)
        return self._cache[key]

    @torch.no_grad()
    def logits(self, states: list[str], q: Question, batch_size: int = 32, max_pairs: int = 256) -> "np.ndarray":
        """Calibrated log-probabilities [n_states, n_choices] (softmax of these = probabilities;
        for rank they are log σ, not normalised). Named `logits` to match the baseline.
        At most `max_pairs` (state, choice) pairs go through the scorer at once — retrieval-sized
        choice sets (1,000+ candidates) × long states would not fit otherwise. Choice scores are
        independent, so chunking changes nothing but memory."""
        import numpy as np

        Hc, mc = self._choices(q)
        n = len(q.choices)
        B_max = max(1, min(batch_size, max_pairs // n))
        out = []
        for i in range(0, len(states), B_max):
            chunk = states[i : i + B_max]
            Hs, ms = self._encode([self.cfg.state_of(s, q.text, q.lang) for s in chunk], self.cfg.max_state_tokens)
            B = len(chunk)
            owner = torch.arange(B, device=self.device).repeat_interleave(n)
            s_parts, z_parts = [], []
            for c0 in range(0, B * n, max_pairs):          # flat (state, choice) pairs, owner-major
                sl = slice(c0, min(c0 + max_pairs, B * n))
                cidx = torch.arange(sl.start, sl.stop, device=self.device) % n
                sp, zp = self.model.choice_scores(Hs, ms, Hc[cidx], mc[cidx], owner[sl])
                s_parts.append(sp); z_parts.append(zp)
            lp = self.model.log_probs(torch.cat(s_parts), torch.cat(z_parts), owner, B, [q.kind] * B, self.cfg.temperature)
            out.append(lp.float().cpu().numpy())
        return np.concatenate(out)

    def decide(self, state: str, q: Question) -> Decision:
        import numpy as np

        t0 = time.perf_counter()
        lp = self.logits([state], q)[0]
        if q.kind == "rank":
            p = 1 / (1 + np.exp(-lp))
        else:
            p = np.exp(lp - lp.max()); p /= p.sum()
        return Decision(probs=p.tolist(), choices=q.choices, latency_ms=(time.perf_counter() - t0) * 1e3)
