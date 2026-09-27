"""Temperature scaling — the one calibration step every checkpoint gets before it ships."""

from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    z = logits - logits.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def nll(logits: np.ndarray, labels: np.ndarray) -> float:
    p = softmax(logits)
    return float(-np.log(p[np.arange(len(labels)), labels] + 1e-12).mean())


def fit_temperature(logits: np.ndarray, labels: np.ndarray, lo: float = 0.01, hi: float = 100.0) -> float:
    """Return T minimizing NLL of softmax(logits / T). Golden-section search on log T —
    NLL is convex in 1/T so this is exact enough and dependency-free."""
    a, b = np.log(lo), np.log(hi)
    phi = (np.sqrt(5) - 1) / 2
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = nll(logits / np.exp(c), labels), nll(logits / np.exp(d), labels)
    for _ in range(60):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = nll(logits / np.exp(c), labels)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = nll(logits / np.exp(d), labels)
    return float(np.exp((a + b) / 2))
