# Decima in LangChain and LangGraph

`decima.langchain` wraps `decima.Router` for LangChain. Nothing here is a hard dependency: `route` works without
LangChain, and `decima_tool` / `decima_runnable` import `langchain-core` only when called.

```bash
pip install "git+https://github.com/amyrmahdy/decima" langchain-core
```

```python
from decima.langchain import decima_runnable, decima_tool, route

# A Runnable: text → /v1/systemone response. Composes with | like any other runnable.
triage = decima_runnable(preset="triage")
triage.invoke("Your API has been down for an hour and checkout is failing")["answers"]["category"]["choice"]   # 'technical'

# A tool an agent can call (fixed preset → takes `text`; no preset → takes `state` + TypeSafe `questions`)
guard = decima_tool(preset="guardrails")
decide = decima_tool()

# A LangGraph conditional edge: the branch is the answer's label (choice, "yes"/"no", or the score level)
graph.add_conditional_edges("intake", route("category", preset="triage", key="ticket"),
                            {"billing": "billing_agent", "technical": "tech_agent", "account": "tech_agent",
                             "shipping": "ops_agent", "sales": "sales_agent", "other": "human"})

# Unsure → a safe branch
graph.add_conditional_edges("check", route("injection", preset="guardrails", key="input"), {"yes": "refuse", "no": "answer"})
graph.add_conditional_edges("plan", route("tier", preset="routing", key="input", min_confidence=0.4, fallback="frontier"),
                            {"small": "small_llm", "medium": "medium_llm", "frontier": "frontier_llm"})
```

All helpers take `router=Router(model=..., threads=...)` to pin a model; by default they share one `Router` and each
preset uses its recommended model. See [`example.py`](example.py) for a runnable LangGraph router.
