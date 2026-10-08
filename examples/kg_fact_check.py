"""Knowledge-graph hygiene: store only facts that are asserted, and merge only mentions of the same entity.

An extractor (an LLM, rules, …) proposes edges with the sentence they came from; Decima judges each one on CPU.

    python examples/kg_fact_check.py
"""

from decima import Router

router = Router()                                        # kg-judge's model: decima-agent

EDGES = [
    ("Acme Corp", "acquired", "Brightwave Labs", "Acme Corp acquired Brightwave Labs in March 2024."),
    ("Acme Corp", "acquired", "Brightwave Labs", "Acme Corp may acquire Brightwave Labs if regulators approve the deal."),
    ("Globex", "partners with", "Initech", "Globex denied reports that it will partner with Initech."),
    ("Umbrella", "opens plant in", "Ohio", "Umbrella expects to open its Ohio plant in 2027."),
]
for head, rel, tail, sentence in EDGES:
    a = router.decide_text(sentence, preset="kg-judge")["answers"]
    keep = a["factual"]["noul"] >= 0.5
    print(f"{'STORE' if keep else 'skip '}  {head} —{rel}→ {tail:16s} {a['status']['choice']:15s} {sentence[:50]}")

PAIRS = [
    ({"text": "Apple Inc.", "type": "company", "context": "Apple Inc. reported record iPhone revenue."},
     {"text": "Apple", "type": "company", "context": "Apple unveiled a new MacBook in Cupertino."}),
    ({"text": "Jordan", "type": "person", "context": "Michael Jordan won six NBA titles."},
     {"text": "Jordan", "type": "geography", "context": "Jordan borders Saudi Arabia."}),
]
print()
for ma, mb in PAIRS:
    s = router.predict({"mention_a": ma, "mention_b": mb}, preset="kg-judge")["answers"]["mention_link"]
    action = "merge" if s["probabilities"]["2"] >= 0.8 else "keep apart" if s["probabilities"]["0"] >= 0.5 else "review"
    print(f"{action:10s}  {ma['text']!r} ~ {mb['text']!r}  (P(same) {s['probabilities']['2']:.2f})")
