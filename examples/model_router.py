"""Send each request to the cheapest LLM that can answer it well; when Decima is unsure, go up a tier, never down.

    python examples/model_router.py "What is the capital of Peru?"
"""

import sys

from decima import Router

MODELS = {"small": "a 3B local model", "medium": "a mid-size hosted model", "frontier": "the frontier model"}
ORDER = ["small", "medium", "frontier"]
router = Router()                                        # routing's model: decima-base


def pick(request: str, min_confidence: float = 0.4) -> tuple[str, str]:
    a = router.decide_text(request, preset="routing")["answers"]["tier"]
    tier = a["choice"]
    if a["confidence"] < min_confidence and tier != "frontier":
        up = ORDER[ORDER.index(tier) + 1]
        return up, f"{tier} unsure ({a['confidence']:.2f}) → {up}"
    return tier, f"{tier} ({a['confidence']:.2f})"


for request in sys.argv[1:] or ["hi!", "Translate 'thank you' into Japanese.", "Write a cover letter for a junior data analyst role.",
                                "Prove that the square root of 2 is irrational, then generalise to any non-square integer.",
                                "Our Kafka consumers lag after every deploy; find the cause across these services and fix it."]:
    tier, why = pick(request)
    print(f"{MODELS[tier]:24s} {why:28s} {request[:60]}")
