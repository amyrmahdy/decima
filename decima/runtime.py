"""ONNX runtime for an exported Decima (fp32 or int8): no torch, same surface as `Decima`.

    rt = DecimaOnnx("export/v0-int8", threads=1)
    rt.logits(["My card was stolen"], Question("What does the customer want?", ["block card", "check balance"]))

Choice encodings are cached per (question, choices, lang) — per (choices, lang) when the
question is encoded with the state (`question_in_state`, V1). The ordinal head for `score`
questions lives here in numpy (its weights ship in decima.json), mirroring
`DecimaModel.ordinal_log_probs` line for line.

Many decisions at once (`decide_many`: one state × many questions, `decide_batch`: many states × one
question) encode every (question + state) input in length-sorted, padded batches and give the same
numbers as `decide` one by one: each row is sliced back to its own length before the scorer, which
therefore sees exactly the inputs of a single call. `long="chunk"` opts into overlapping windows for
states longer than the model's window (decima/longdoc.py); `min_confidence` gates unsure answers.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .longdoc import AGGREGATES, collect, plan, probs
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

    def _encode_rows(self, rows: list[list[int]], batch_size: int, max_batch_tokens: int, widths: list[int] | None = None):
        """Yields (row index, token states [width, d]) per token-id row (width: its own length unless given),
        batch by batch: rows sorted by width, padded to the widest of their batch, at most `batch_size` rows and
        `max_batch_tokens` padded tokens per batch. A row's states do not depend on the rows or padding beside
        it (attention is masked by key, activations are quantised per token row); with one row per batch this
        is exactly `encode`."""
        widths = widths or [len(r) for r in rows]
        order = sorted(range(len(rows)), key=widths.__getitem__)
        pad = self.tok.pad_token_id or 0
        i = 0
        while i < len(order):
            j = i + 1
            while j < len(order) and j - i < batch_size and (j - i + 1) * widths[order[j]] <= max_batch_tokens:
                j += 1
            idx = order[i:j]
            L = widths[idx[-1]]
            ids, mask = np.full((len(idx), L), pad, np.int64), np.zeros((len(idx), L), np.int64)
            for k, r in enumerate(idx):
                ids[k, : len(rows[r])], mask[k, : len(rows[r])] = rows[r], 1
            h = self.enc.run(None, {"input_ids": ids, "attention_mask": mask})[0]
            i = j
            for k, r in enumerate(idx):
                yield r, h[k, : widths[r]]

    def _choice_key(self, q: Question) -> tuple[tuple, str]:
        qt = q.text if self.cfg.get("question_in_choices", True) else ""
        return (qt, tuple(q.choices), q.lang), qt

    def _choice_texts(self, q: Question, qt: str) -> list[str]:
        return choice_texts(self.tok, qt, q.choices, q.lang, self.cfg.get("choice_prefix", "passage: "), self.cfg["max_choice_tokens"],
                            fit=self.cfg.get("choice_fit", False))

    def _choices(self, q: Question) -> tuple[np.ndarray, np.ndarray]:
        key, qt = self._choice_key(q)
        if key not in self._cache:
            if len(self._cache) > 256:
                self._cache.clear()
            self._cache[key] = self.encode(self._choice_texts(q, qt), self.cfg["max_choice_tokens"])
        return self._cache[key]

    def _prefetch_choices(self, qs: list[Question], batch_size: int = 64) -> None:
        """Encode the uncached choice sets of many questions in shared batches. Each set is cut back to its own
        padded width, so the scorer gets the same [n, L, d] tensor (pad positions included) as `_choices`."""
        todo: dict[tuple, list[list[int]]] = {}
        for q in qs:
            key, qt = self._choice_key(q)
            if key not in self._cache and key not in todo:
                todo[key] = [self.tok(t, truncation=True, max_length=self.cfg["max_choice_tokens"])["input_ids"] for t in self._choice_texts(q, qt)]
        if len(todo) < 2:
            return
        if len(self._cache) + len(todo) > 256:
            self._cache.clear()
        flat = [r for rows in todo.values() for r in rows]
        widths = [max(map(len, rows)) for rows in todo.values() for _ in rows]
        hs = [h for _, h in sorted(self._encode_rows(flat, batch_size, batch_size * self.cfg["max_choice_tokens"], widths), key=lambda x: x[0])]
        k = 0
        for key, rows in todo.items():
            L = max(map(len, rows))
            mask = np.array([[1] * len(r) + [0] * (L - len(r)) for r in rows], np.int64)
            self._cache[key] = (np.stack(hs[k : k + len(rows)]), mask)
            k += len(rows)

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
        return self._score(h, m, ch, cm, q)

    def _score(self, h: np.ndarray, m: np.ndarray, ch: np.ndarray, cm: np.ndarray, q: Question) -> np.ndarray:
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

    def decide(self, state: str, question: Question, long: str | None = None, aggregate: str = "max",
               min_confidence: float | None = None) -> Decision:
        """Probabilities over the question's choices (independent sigmoids for `rank`). A state longer than
        the model's window is cut at its end, unless `long="chunk"` (see `decide_many`)."""
        if long is not None:
            return self.decide_many(state, [question], long=long, aggregate=aggregate, min_confidence=min_confidence)[0]
        t0 = time.perf_counter()
        p = probs(self.decide_logits(state, question), question.kind)
        d = Decision(probs=[float(x) for x in p], choices=list(question.choices), latency_ms=(time.perf_counter() - t0) * 1e3)
        if min_confidence is not None:
            d.abstain(min_confidence)
        return d

    def decide_many(self, state: str, questions: list[Question], **kw) -> list[Decision]:
        """Many questions about one state, batched; the same probabilities as `decide` for each question.

        batch_size (16) / max_batch_tokens (512): rows and padded tokens per encoder call. Sharing a call
          only pays for short rows (docs/BENCH-batching.md); from ~300 tokens on the budget keeps one per call.
        long: None cuts an over-long state at its end, as `decide` does; "chunk" splits it into overlapping
          windows (`overlap` tokens, default 64) and combines the per-window answers by `aggregate`
          ("max" | "mean" | "sum", decima/longdoc.py). Chunked decisions carry meta["window"] (index of the
          deciding window), meta["windows"] (character spans of the state) and meta["window_probs"].
        min_confidence: mark decisions whose top probability is below it as `abstained`.
        latency_ms is the call's wall time divided by the number of decisions."""
        return self._decide_pairs([(state, q) for q in questions], **kw)

    def decide_batch(self, states: list[str], question: Question, **kw) -> list[Decision]:
        """One question about many states, batched; options as in `decide_many`."""
        return self._decide_pairs([(s, question) for s in states], **kw)

    def _decide_pairs(self, pairs: list[tuple[str, Question]], batch_size: int = 16, max_batch_tokens: int = 512,
                      long: str | None = None, aggregate: str = "max", overlap: int = 64,
                      min_confidence: float | None = None) -> list[Decision]:
        if aggregate not in AGGREGATES:
            raise ValueError(f"aggregate must be one of {AGGREGATES}, not {aggregate!r}")
        t0 = time.perf_counter()
        jobs = plan(self.tok, pairs, self._state, self.cfg["max_state_tokens"], long, overlap)
        self._prefetch_choices([q for _, q in pairs])
        lps: list[np.ndarray] = [None] * len(jobs)
        # identical encoder inputs are encoded once: with question_in_state off (read-once models) every question
        # about a state shares one pass over it, however many questions there are
        uniq: dict[tuple, int] = {}
        rows, users = [], []
        for j, (_, _, ids) in enumerate(jobs):
            k = uniq.setdefault(tuple(ids), len(rows))
            if k == len(rows):
                rows.append(ids); users.append([])
            users[k].append(j)
        for r, h in self._encode_rows(rows, batch_size, max_batch_tokens):
            for j in users[r]:
                q = pairs[jobs[j][0]][1]
                lps[j] = self._score(h[None], np.ones((1, len(h)), np.int64), *self._choices(q), q)
        return collect(pairs, jobs, lps, aggregate, min_confidence, (time.perf_counter() - t0) * 1e3 / max(1, len(pairs)))

    def logits(self, states: list[str], q: Question) -> np.ndarray:
        """Log-probabilities [n_states, n_choices], like `Decima.logits`."""
        return np.stack([self.decide_logits(s, q) for s in states])
