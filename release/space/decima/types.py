from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Kind = Literal["choose", "score", "verify", "rank"]


@dataclass
class Question:
    """A typed decision question. `choices` are natural-language strings supplied at
    inference time; the model never sees them during training, so any choice set works.

    choose  : pick one of N mutually exclusive options       → softmax over choices
    score   : pick one of N *ordered* levels (0..N-1)        → softmax, ordinal-aware later
    verify  : yes/no on a proposition                        → choices = ["yes", "no"]
    rank    : score every option independently against state → sigmoid per choice
    """

    text: str
    choices: list[str]
    kind: Kind = "choose"
    lang: str = "en"

    def __post_init__(self) -> None:
        if self.kind == "verify" and len(self.choices) != 2:
            raise ValueError("verify questions take exactly two choices")
        if len(self.choices) < 1:
            raise ValueError("a question needs at least one choice")


@dataclass
class Decision:
    probs: list[float]                       # aligned with question.choices
    choices: list[str]
    latency_ms: float = 0.0
    meta: dict = field(default_factory=dict)

    @property
    def top(self) -> str:
        return self.choices[max(range(len(self.probs)), key=self.probs.__getitem__)]

    @property
    def confidence(self) -> float:
        return max(self.probs)

    def as_dict(self) -> dict[str, float]:
        return dict(zip(self.choices, self.probs))

    def __repr__(self) -> str:
        return repr({c: round(p, 3) for c, p in zip(self.choices, self.probs)})
