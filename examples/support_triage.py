"""Support triage: queue, priority and tone for every incoming ticket, on CPU.

    python examples/support_triage.py
"""

from decima import Router

TICKETS = [
    "Your API has returned 500 errors for an hour and our checkout is down. We are losing orders!",
    "I was charged twice for my March invoice, please refund one of them.",
    "Could you add dark mode to the mobile app? Would love that.",
    "My parcel says delivered but it's not here. Third time this month, I'm done with you.",
    "Is there a discount for 50 seats on the annual plan?",
]
QUEUES = {"billing": "Billing", "technical": "Engineering on-call", "account": "Identity", "shipping": "Logistics",
          "sales": "Sales", "other": "Support"}

router = Router()                                       # triage's model: decima-base, int8, one CPU thread
for r, ticket in zip(router.predict_many(TICKETS, preset="triage"), TICKETS):
    a = r["answers"]
    cat, urgency, mood = a["category"], a["urgency"]["score"], a["sentiment"]["choice"]
    queue = QUEUES[cat["choice"]] if cat["confidence"] >= 0.5 else "Support (unsure)"
    priority = "P1" if urgency >= 1.5 else "P2" if urgency >= 0.7 else "P3"
    flag = "  ⚑ upset customer" if mood == "negative" and urgency >= 1 else ""
    print(f"{priority}  {queue:20s} {ticket[:60]}{flag}")
