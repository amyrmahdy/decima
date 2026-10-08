"""Decima (amyrmahdy/decima-small) backend via the decima package (int8 ONNX, CPU).

Rendering (fixed, not tuned on this benchmark):
  choice: question = instructions, options = "<key>: <description>"
  noul:   verify question = instructions, options = ["yes", "no"]
  score:  score question = instructions, options = the levels in order (ordinal head)
"""

from typing import Optional

from bench.cases import Choice, Noul, Score

from .base import Backend, Prediction


class DecimaBackend(Backend):
  name = "decima-ai"

  def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None, precision: str = "int8"):
    self.model = model_path or "amyrmahdy/decima-small"
    self.precision = precision
    self.description = f"{self.model} ({precision} ONNX, 1 CPU thread)"

  def warmup(self) -> float:
    import time

    from decima import Decima, Question

    t0 = time.perf_counter()
    self.Question = Question
    self.decima = Decima.from_pretrained(self.model, precision=self.precision)
    self.decima.decide("warmup", Question("Warm up?", ["yes", "no"], kind="verify"))
    return time.perf_counter() - t0

  def predict_choice(self, state: str, question: Choice) -> Prediction:
    keys = list(question.criteria)
    options = [f"{k}: {v}" for k, v in question.criteria.items()]
    d = self.decima.decide(state, self.Question(question.instructions, options, kind="choose"))
    probabilities = dict(zip(keys, d.probs))
    label = max(probabilities, key=probabilities.get)
    return Prediction(label=label, probabilities=probabilities, confidence=probabilities[label])

  def predict_noul(self, state: str, question: Noul) -> Prediction:
    d = self.decima.decide(state, self.Question(question.instructions, ["yes", "no"], kind="verify"))
    probability = d.probs[0]
    return Prediction(
      label="yes" if probability >= 0.5 else "no",
      probability=probability,
      confidence=abs(probability - 0.5) * 2,
      raw=probability,
    )

  def predict_score(self, state: str, question: Score) -> Prediction:
    d = self.decima.decide(state, self.Question(question.instructions, list(question.criteria), kind="score"))
    probabilities = {str(i): p for i, p in enumerate(d.probs)}
    label = max(probabilities, key=probabilities.get)
    expected = sum(i * p for i, p in enumerate(d.probs))
    return Prediction(label=label, probabilities=probabilities, confidence=probabilities[label], raw=expected)
