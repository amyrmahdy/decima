"""Support-ticket router: Decima picks the branch, each branch is a stub where an LLM call or a queue would go.

    pip install langgraph        # optional: without it the same `route` function is called directly
    python integrations/langchain/example.py
"""

from typing import TypedDict

from decima.langchain import route

TEAMS = {"billing": "billing", "technical": "engineering", "account": "engineering", "shipping": "operations",
         "sales": "sales", "other": "human"}
pick = route("category", preset="triage", key="ticket", min_confidence=0.5, fallback="other")
TICKETS = ["I was charged twice for my March invoice.", "The dashboard shows a blank page since this morning.",
           "Do you ship to Norway?", "hmm"]


class State(TypedDict, total=False):
    ticket: str
    queue: str


def handler(team: str):
    return lambda s: {"queue": team}


try:
    from langgraph.graph import END, START, StateGraph
except ImportError:
    for t in TICKETS:
        print(f"{TEAMS[pick({'ticket': t})]:12s} ← {t}")
else:
    g = StateGraph(State)
    for label, team in TEAMS.items():
        g.add_node(label, handler(team))
        g.add_edge(label, END)
    g.add_conditional_edges(START, pick, {k: k for k in TEAMS})
    app = g.compile()
    for t in TICKETS:
        print(f"{app.invoke({'ticket': t})['queue']:12s} ← {t}")
