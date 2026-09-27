"""Decima-small as a BTZSC model adapter (`btzsc.models.base.BaseModel`) + a runner that writes the
leaderboard JSON with the official harness.

    pip install btzsc "git+https://github.com/amyrmahdy/decima"

    # smoke test (2 datasets x 50 texts, CPU, ~1 min)
    python decima_btzsc.py --backend onnx --tasks agnews,imdb --max-samples 50 --out /tmp/btzsc-smoke.json

    # official run, all 22 datasets, full test sets (no --max-samples): see RUN.md for timing
    python decima_btzsc.py --backend torch --device cuda --out results/reranker/decima-small.json   # GPU box
    python decima_btzsc.py --backend onnx --precision int8 --threads 1 --out results/reranker/decima-small.json  # CPU

Protocol, fixed before the run and identical for all 22 datasets:
  * text  -> Decima state (as given; the model truncates at 512 tokens, like every harness baseline truncates)
  * the dataset's hypotheses, verbatim and in the harness's order -> Decima choices (64 tokens each)
  * one neutral question for every dataset: "Which statement best describes this example?" It carries no
    label information; Decima needs a question field, the harness baselines get none.
  * kind "choose" (softmax over the hypotheses) for every dataset. No per-dataset branching, prompt,
    threshold, temperature or calibration; nothing is fitted on BTZSC.
  * score matrix = Decima's log-probabilities; prediction = argmax.
The harness itself loads the data (including its unlabelled banking77/massive groups, gold 0), computes the
metrics and serialises the JSON; this file only supplies predictions and fills the model metadata fields
(`params`, `revision`, `precision`) that the harness cannot resolve for a custom adapter.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from btzsc.models.base import BaseModel

QUESTION = "Which statement best describes this example?"
HF_REPO = "amyrmahdy/decima-small"
PARAMS = 122_000_000
CLEAN_EXCLUDED = {"rottentomatoes": "contaminated (33 % of its texts are in the teacher-labelled SST-5 train states)",
                  "banking77": "in-distribution (train split and label space in training)",
                  "massive": "in-distribution (train split and label space in training)",
                  "agnews": "in-distribution (train split and label space in training)"}


class DecimaBTZSC(BaseModel):
    """BTZSC adapter. `reranker` is the closest leaderboard family: Decima scores every (text, label) pair
    jointly — label tokens cross-attend to the text — rather than comparing two independent embeddings."""

    model_type = "reranker"

    def __init__(self, model: str = HF_REPO, backend: str = "onnx", precision: str = "int8",
                 device: str = "cpu", threads: int = 1, batch_size: int = 32, max_pairs: int = 4096):
        self.model_name = HF_REPO  # the leaderboard row and its URL name the public repo, whatever path we load from
        self.source, self.backend, self.device = model, backend, device
        self.batch, self.max_pairs = batch_size, max_pairs
        self.revision = "unknown"
        if backend == "onnx":
            from decima import Decima

            self.d = Decima.from_pretrained(model, precision=precision, threads=threads)
            self.precision = precision
        elif backend == "torch":
            import torch
            from decima.model import Decima as TorchDecima

            path = Path(model)
            if not path.exists():  # HF repo: the PyTorch checkpoint lives under pytorch/
                from huggingface_hub import snapshot_download

                path = Path(snapshot_download(model, allow_patterns=["pytorch/*", "pytorch/**"])) / "pytorch"
            torch.set_num_threads(max(1, threads))
            self.d = TorchDecima(path, device=device)
            self.precision = "fp32"
        else:
            raise ValueError(f"backend must be onnx or torch, not {backend!r}")
        if not Path(model).exists():
            try:
                from huggingface_hub import HfApi

                self.revision = HfApi().model_info(model).sha or "unknown"
            except Exception:  # noqa: BLE001 - metadata only
                pass

    def predict_scores(self, texts: list[str], labels: list[str], batch_size: int = 32) -> np.ndarray:
        from decima import Question

        q = Question(QUESTION, list(labels), kind="choose", lang="en")
        if self.backend == "torch":
            # max_pairs bounds (text, label) pairs per scorer call; the library default (256) leaves a GPU idle on
            # 72-label banking77 (3 texts per batch). Chunking changes memory use only, never the scores.
            return self.d.logits(list(texts), q, batch_size=batch_size or self.batch, max_pairs=self.max_pairs)
        return self.d.logits(list(texts), q)

    def predict(self, texts: list[str], labels: list[str], batch_size: int = 32) -> np.ndarray:
        return self.predict_scores(texts, labels, batch_size).argmax(axis=1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run BTZSC on Decima with the official harness")
    ap.add_argument("--model", default=HF_REPO, help="HF repo id or a local export/checkpoint dir")
    ap.add_argument("--backend", default="onnx", choices=["onnx", "torch"])
    ap.add_argument("--precision", default="int8", choices=["int8", "fp32"], help="onnx backend only")
    ap.add_argument("--device", default="cpu", help="torch backend only (cuda / cpu)")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-pairs", type=int, default=4096, help="torch backend: (text, label) pairs per scorer call")
    ap.add_argument("--tasks", default="", help="comma list of task groups or dataset names (empty = all 22)")
    ap.add_argument("--max-samples", type=int, default=None, help="smoke tests only; the leaderboard needs full sets")
    ap.add_argument("--out", default="results/reranker/decima-small.json")
    ap.add_argument("--shard-out", default=None,
                    help="CPU sharding: save this run's BTZSCResults.to_dict() here (use with --tasks <dataset list>)")
    ap.add_argument("--merge", nargs="+", default=None,
                    help="merge --shard-out files (disjoint datasets) into one leaderboard JSON at --out")
    a = ap.parse_args(argv)

    from btzsc import BTZSCBenchmark, BTZSCResults  # noqa: F401
    from btzsc.leaderboard.validate import validate_result_file

    t0 = time.time()
    tasks = [t for t in a.tasks.split(",") if t] or None
    if a.merge:
        # Per-dataset metrics come from the harness in each shard; the summary is recomputed by the harness.
        from btzsc.benchmark import BTZSCResults
        from btzsc.data import TASK_GROUPS
        from btzsc.metrics import compute_task_summary

        shards = [json.loads(Path(p).read_text()) for p in a.merge]
        per: dict = {}
        for s in shards:
            dup = set(per) & set(s["per_dataset_results"])
            if dup:
                raise SystemExit(f"datasets in more than one shard: {sorted(dup)}")
            per.update(s["per_dataset_results"])
        meta = {k: shards[0][k] for k in ("model_name", "model_type", "model_params", "model_revision", "precision",
                                            "batch_size", "max_samples", "device")}
        res = BTZSCResults(per_dataset_results=per, task_summary=compute_task_summary(per, TASK_GROUPS), **meta)
        tasks = None if len(per) == 22 else list(per)
    else:
        model = DecimaBTZSC(a.model, a.backend, a.precision, a.device, a.threads, a.batch_size, a.max_pairs)
        res = BTZSCBenchmark(tasks=tasks).evaluate(model=model, batch_size=a.batch_size, max_samples=a.max_samples)
        res.model_params, res.model_revision, res.precision = PARAMS, model.revision, model.precision
        res.device = a.device if a.backend == "torch" else f"cpu ({a.threads} thread{'s' if a.threads > 1 else ''}, onnxruntime)"
        if a.shard_out:
            Path(a.shard_out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.shard_out).write_text(json.dumps(res.to_dict(), indent=1))
    wall = time.time() - t0
    res.to_json(a.out)

    per = res.per_dataset_results
    f1 = {k: v["macro_f1"] for k, v in per.items()}
    clean = [v for k, v in f1.items() if k not in CLEAN_EXCLUDED]
    print(json.dumps({"datasets": len(f1), "macro_f1_all": round(float(np.mean(list(f1.values()))), 4),
                      "macro_f1_clean": round(float(np.mean(clean)), 4) if clean else None,
                      "n_clean": len(clean), "wall_s": round(wall, 1), "out": a.out}, indent=1))
    if a.max_samples is None and tasks is None:
        errors = validate_result_file(a.out)
        print("validate-result:", "OK" if not errors else errors)
        return 1 if errors else 0
    print("partial run (tasks or max-samples set): not a leaderboard artifact", file=sys.stderr)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())
