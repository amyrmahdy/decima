"""Tests run on CPU against local int8 exports (not in git): DECIMA_TEST_EXPORTS or DECIMA_EXPORTS (default
<repo>/export) holding base2-int8 (512-token states), agent3-int8 (2,048) and v1i-int8. Tests that need a missing model
are skipped; nothing is downloaded."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ["HF_HUB_OFFLINE"] = "1"

ROOT = Path(__file__).resolve().parents[1]
EXPORTS = Path(os.environ.get("DECIMA_TEST_EXPORTS", os.environ.get("DECIMA_EXPORTS", ROOT / "export")))
CHECKPOINTS = Path(os.environ.get("DECIMA_CHECKPOINTS", ROOT / "checkpoints"))
LOCAL = {"decima-base": EXPORTS / "base2-int8", "decima-agent": EXPORTS / "agent3-int8", "decima-small": EXPORTS / "v1i-int8"}
HAVE = all((p / "encoder.onnx").exists() for p in LOCAL.values())
os.environ["DECIMA_MODELS"] = ",".join(f"{k}={v}" for k, v in LOCAL.items())
THREADS = 2

needs_models = pytest.mark.skipif(not HAVE, reason=f"local exports not found in {EXPORTS} (set DECIMA_TEST_EXPORTS)")


@pytest.fixture(scope="session")
def router():
    from decima import Router

    return Router(threads=THREADS)


def _load(name: str):
    p = EXPORTS / name
    if not (p / "encoder.onnx").exists():
        pytest.skip(f"{p} not available")
    from decima.runtime import DecimaOnnx

    return DecimaOnnx(p, threads=THREADS)


@pytest.fixture(scope="session")
def base2():
    return _load("base2-int8")


@pytest.fixture(scope="session")
def agent3():
    return _load("agent3-int8")


@pytest.fixture(scope="session")
def base2_torch():
    p = CHECKPOINTS / "base2" / "best"
    if not (p / "decima.json").exists():
        pytest.skip(f"{p} not available")
    import torch

    torch.set_num_threads(THREADS)
    from decima.model import Decima

    return Decima(p)
