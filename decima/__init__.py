"""Decima — a tiny calibrated decision model.

state + question + arbitrary choices → calibrated probabilities. CPU-first.
"""

import os

os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")   # the runtime needs only the tokenizer, not PyTorch

from .types import Decision, Question
from .normalize import normalize
from .runtime import DecimaOnnx as Decima     # the public, torch-free runtime (ONNX Runtime + tokenizer)
from .router import Router, list_presets, load_preset
from .pii import DecimaPII, PII                # decima-pii: find and redact personal-data spans

__all__ = ["Decima", "DecimaPII", "Decision", "PII", "Question", "Router", "list_presets", "load_preset", "normalize"]
