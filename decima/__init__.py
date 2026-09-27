"""Decima — a tiny calibrated decision model.

state + question + arbitrary choices → calibrated probabilities. CPU-first.
"""

from .types import Decision, Question
from .normalize import normalize
from .runtime import DecimaOnnx as Decima     # the public, torch-free runtime (ONNX Runtime + tokenizer)

__all__ = ["Decima", "Decision", "Question", "normalize"]
