from __future__ import annotations

import numpy as np


def accuracy(p: np.ndarray, y: np.ndarray) -> float:
    return float((p.argmax(1) == y).mean())


def brier(p: np.ndarray, y: np.ndarray) -> float:
    """Multiclass Brier: mean over examples of Σ_k (p_k − 1[y=k])². 0 is perfect, 2 is worst."""
    onehot = np.zeros_like(p)
    onehot[np.arange(len(y)), y] = 1
    return float(((p - onehot) ** 2).sum(1).mean())


def nll(p: np.ndarray, y: np.ndarray) -> float:
    return float(-np.log(p[np.arange(len(y)), y] + 1e-12).mean())


def ece(p: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error on the top-1 confidence, equal-width bins."""
    conf, pred = p.max(1), p.argmax(1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            out += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(out)


def summarize(p: np.ndarray, y: np.ndarray) -> dict[str, float]:
    return {"acc": accuracy(p, y), "brier": brier(p, y), "nll": nll(p, y), "ece": ece(p, y)}
