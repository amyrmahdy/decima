"""Zero-shot bi-encoder baseline.

No training. State and every choice are embedded with the same small multilingual
encoder; cosine similarity / T is softmaxed over the choices. This is the honest floor
the trained Decima model has to beat, and it already gives us CPU latency numbers for
the backbone we intend to keep.
"""

from __future__ import annotations

import time
from functools import lru_cache

import numpy as np

from .calibration import softmax
from .normalize import normalize
from .types import Decision, Question

DEFAULT_MODEL = "intfloat/multilingual-e5-small"   # ~21M transformer body + 96M vocab table


class ZeroShotBiEncoder:
    def __init__(self, model_name: str = DEFAULT_MODEL, temperature: float = 0.05, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name
        self.temperature = temperature
        self._is_e5 = "e5" in model_name.lower()

    # e5 models are trained with these prefixes; other encoders ignore them.
    def _q(self, s: str) -> str:
        return f"query: {s}" if self._is_e5 else s

    def _p(self, s: str) -> str:
        return f"passage: {s}" if self._is_e5 else s

    def encode_states(self, states: list[str], lang: str = "en", batch_size: int = 64) -> np.ndarray:
        texts = [self._q(normalize(s, lang)) for s in states]
        return self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True)

    @lru_cache(maxsize=256)
    def _choice_matrix(self, question: str, choices: tuple[str, ...], lang: str) -> np.ndarray:
        texts = [self._p(normalize(f"{question} {c}".strip(), lang)) for c in choices]
        return self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

    def logits(self, states: list[str], q: Question, choice_prefix: str = "") -> np.ndarray:
        """Raw (uncalibrated) logits, shape [n_states, n_choices]."""
        S = self.encode_states(states, q.lang)
        C = self._choice_matrix(choice_prefix, tuple(q.choices), q.lang)
        return (S @ C.T) / self.temperature

    def decide(self, state: str, q: Question) -> Decision:
        t0 = time.perf_counter()
        z = self.logits([state], q)[0]
        p = softmax(z)
        return Decision(probs=p.tolist(), choices=q.choices, latency_ms=(time.perf_counter() - t0) * 1e3)
