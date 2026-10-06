"""ONNX runtime for an exported Decima (fp32 or int8): no torch, same surface as `Decima`.

    rt = DecimaOnnx("export/v0-int8", threads=1)
    rt.logits(["My card was stolen"], Question("What does the customer want?", ["block card", "check balance"]))

Choice encodings are cached per (question, choices, lang) — per (choices, lang) when the
question is encoded with the state (`question_in_state`, V1). The ordinal head for `score`
questions lives here in numpy (its weights ship in decima.json), mirroring
`DecimaModel.ordinal_log_probs` line for line.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .normalize import normalize
from .render import choice_texts
from .types import Decision, Question


def _log_softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    return x - np.log(np.exp(x).sum())


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


class DecimaOnnx:
    def __init__(self, path: str | Path, threads: int = 0):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        d = Path(path)
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self.enc = ort.InferenceSession(str(d / "encoder.onnx"), so, providers=["CPUExecutionProvider"])
        self.sc = ort.InferenceSession(str(d / "scorer.onnx"), so, providers=["CPUExecutionProvider"])
        self.tok = AutoTokenizer.from_pretrained(d / "tokenizer")
        self.cfg = json.loads((d / "decima.json").read_text())
        o = self.cfg["ordinal"]
        self.ord = (np.array(o["ord_g_w"]), o["ord_g_b"], np.array(o["ord_gap_w"]), o["ord_gap_b"])
        self._cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
        self.model_name = str(d)

    def encode(self, texts: list[str], max_len: int) -> tuple[np.ndarray, np.ndarray]:
        b = self.tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="np")
        ids, mask = b["input_ids"].astype(np.int64), b["attention_mask"].astype(np.int64)
        return self.enc.run(None, {"input_ids": ids, "attention_mask": mask})[0], mask

    def _choices(self, q: Question) -> tuple[np.ndarray, np.ndarray]:
        qt = q.text if self.cfg.get("question_in_choices", True) else ""
        key = (qt, tuple(q.choices), q.lang)
        if key not in self._cache:
            if len(self._cache) > 256:
                self._cache.clear()
            texts = choice_texts(self.tok, qt, q.choices, q.lang, self.cfg.get("choice_prefix", "passage: "), self.cfg["max_choice_tokens"],
                                 fit=self.cfg.get("choice_fit", False))
            self._cache[key] = self.encode(texts, self.cfg["max_choice_tokens"])
        return self._cache[key]

    def _state(self, state: str, q: Question) -> str:
        body = f"{q.text}\n{state}" if self.cfg.get("question_in_state", False) else state
        return self.cfg.get("state_prefix", "query: ") + normalize(body, q.lang)

    def ordinal(self, s: np.ndarray, z: np.ndarray) -> np.ndarray:
        gw, gb, pw, pb = self.ord
        K = len(s)
        expected = float((np.exp(_log_softmax(s)) * np.arange(K)).sum()) - (K - 1) / 2
        g = float(z.mean(0) @ gw + gb) + expected
        gaps = np.logaddexp(0, z @ pw + pb) + 1e-3
        theta = np.cumsum(gaps)[:-1] - gaps.sum() / 2
        sig = _sigmoid(g - theta)
        p = np.concatenate([[1.0], sig]) - np.concatenate([sig, [0.0]])
        return _log_softmax(np.log(np.clip(p, 1e-7, None)))

    def decide_logits(self, state: str, q: Question) -> np.ndarray:
        ch, cm = self._choices(q)
        h, m = self.encode([self._state(state, q)], self.cfg["max_state_tokens"])
        s, z = self.sc.run(None, {"state_h": h, "state_mask": m, "choice_h": ch, "choice_mask": cm})
        s = s / self.cfg["temperature"]
        if q.kind == "score":
            return self.ordinal(s, z)
        if q.kind == "rank":
            return -np.logaddexp(0, -s)
        return _log_softmax(s)

    @classmethod
    def from_pretrained(cls, name_or_path: str | Path, precision: str = "int8", threads: int = 1) -> "DecimaOnnx":
        """Local export dir, or a Hugging Face repo laid out as onnx/<precision>/{encoder.onnx, scorer.onnx,
        decima.json, tokenizer/}. One thread is the fastest setting at batch 1 (docs/BENCH-x86.md)."""
        p = Path(name_or_path)
        if not p.exists():
            from huggingface_hub import snapshot_download

            root = Path(snapshot_download(str(name_or_path), allow_patterns=["config.json", f"onnx/{precision}/*", f"onnx/{precision}/tokenizer/*"]))
            p = root / "onnx" / precision
        return cls(p, threads=threads)

    def decide(self, state: str, question: Question) -> Decision:
        """Probabilities over the question's choices (independent sigmoids for `rank`)."""
        t0 = time.perf_counter()
        lp = self.decide_logits(state, question)
        p = np.exp(lp) if question.kind == "rank" else np.exp(lp - lp.max()) / np.exp(lp - lp.max()).sum()
        return Decision(probs=[float(x) for x in p], choices=list(question.choices), latency_ms=(time.perf_counter() - t0) * 1e3)

    def logits(self, states: list[str], q: Question) -> np.ndarray:
        """Log-probabilities [n_states, n_choices], like `Decima.logits`."""
        return np.stack([self.decide_logits(s, q) for s in states])
