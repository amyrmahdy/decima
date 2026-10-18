"""Baseline for the decima-guard card: protectai/deberta-v3-base-prompt-injection-v2 on the frozen `inject` rows (CPU).
    uv run python bench/baselines/protectai_injection.py   → runs/guard-baselines/protectai-inject.jsonl
"""
# Baseline for the card: protectai/deberta-v3-base-prompt-injection-v2 (Apache-2.0) on decima-guard's frozen `inject` test rows.
import json, sys, torch
torch.set_num_threads(3)
from transformers import AutoTokenizer, AutoModelForSequenceClassification
M = "protectai/deberta-v3-base-prompt-injection-v2"
tok = AutoTokenizer.from_pretrained(M); model = AutoModelForSequenceClassification.from_pretrained(M).eval()
rows = [json.loads(l) for l in open("data/guard/test-llm.jsonl")]
rows = [r for r in rows if r["task"] == "inject"]
out = open("runs/guard-baselines/protectai-inject.jsonl", "w")
inj = [i for i, l in model.config.id2label.items() if "INJ" in l.upper()][0]
for i in range(0, len(rows), 32):
    b = rows[i:i + 32]
    enc = tok([r["state"] for r in b], truncation=True, max_length=512, padding=True, return_tensors="pt")
    with torch.no_grad():
        p = model(**enc).logits.softmax(-1)[:, inj].tolist()
    for r, x in zip(b, p):
        out.write(json.dumps({"id": r["id"], "p_inject": x, "gold_yes": r["choices"][r["gold"]] == "yes", "lang": r["state_lang"], "source": r["source"]}) + "\n")
