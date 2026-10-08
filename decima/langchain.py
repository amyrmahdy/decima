"""LangChain and LangGraph helpers. `route` needs nothing extra; the tool and runnable need `pip install langchain-core`.

    from decima.langchain import decima_tool, decima_runnable, route

    tool = decima_tool(preset="guardrails")                # a StructuredTool an agent can call
    triage = decima_runnable(preset="triage")              # Runnable: text (or {"state", "questions"}) → answers
    graph.add_conditional_edges("intake", route("category", preset="triage", key="ticket"))   # LangGraph branch
"""

from __future__ import annotations

from typing import Any, Callable

from .router import Router, as_state, load_preset

_shared: Router | None = None


def _router(router: Router | None) -> Router:
    global _shared
    if router is not None:
        return router
    if _shared is None:
        _shared = Router()
    return _shared


def _ask(router: Router, state: Any, questions: dict | None, preset: str | None) -> dict:
    if isinstance(state, str) and preset:
        state = as_state(state, preset)
    return router.predict(state, questions, preset=preset)


def label(answer: dict) -> str:
    """One answer → a branch name: the choice, "yes"/"no" for a noul, the nearest level index for a score."""
    if answer["type"] == "choice":
        return answer["choice"]
    if answer["type"] == "noul":
        return "yes" if answer["noul"] >= 0.5 else "no"
    return str(round(answer["score"]))


def route(question: str, *, preset: str | None = None, questions: dict | None = None, key: str = "input",
          min_confidence: float = 0.0, fallback: str | None = None, router: Router | None = None) -> Callable[[dict], str]:
    """A LangGraph conditional-edge function: reads `state[key]`, answers `question` and returns its label.
    Below `min_confidence` (choice/score confidence, or |2p − 1| for a noul) it returns `fallback` instead."""

    def pick(state: dict) -> str:
        a = _ask(_router(router), state[key], questions, preset)["answers"][question]
        conf = a.get("confidence", abs(2 * a.get("noul", 0.5) - 1))
        return fallback if fallback is not None and conf < min_confidence else label(a)

    return pick


def decima_runnable(*, preset: str | None = None, questions: dict | None = None, router: Router | None = None):
    """Runnable: a state (str / dict), or {"state": …, "questions": …}, → the /v1/systemone response."""
    from langchain_core.runnables import RunnableLambda

    def run(x: Any) -> dict:
        if isinstance(x, dict) and "state" in x:
            return _ask(_router(router), x["state"], {**(questions or {}), **(x.get("questions") or {})}, preset)
        return _ask(_router(router), x, questions, preset)

    return RunnableLambda(run, name=f"decima_{preset or 'decide'}")


def decima_tool(*, preset: str | None = None, questions: dict | None = None, name: str | None = None,
                description: str | None = None, router: Router | None = None):
    """A StructuredTool. With a preset or fixed questions it takes `text`; otherwise `state` and TypeSafe `questions`."""
    from langchain_core.tools import StructuredTool

    if preset or questions:
        desc = description or (load_preset(preset)["description"] if preset else "Calibrated decision: " + ", ".join(questions))

        def fixed(text: str) -> dict:
            """Decide about `text`."""
            return _ask(_router(router), text, questions, preset)

        return StructuredTool.from_function(fixed, name=name or f"decima_{(preset or 'decide').replace('-', '_')}", description=desc)

    def decide(state: str, questions: dict) -> dict:
        """Decide about `state`. `questions`: {name: {"type": "choice"|"noul"|"score", "instructions": str, "criteria": ...}}."""
        return _ask(_router(router), state, questions, None)

    return StructuredTool.from_function(decide, name=name or "decima_decide", description=description or
                                        "Calibrated decision on CPU. questions use TypeSafe's System One schema: choice criteria "
                                        "{label: description}, score criteria [level 0, level 1, …], noul = a statement → P(true).")
