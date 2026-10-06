"""Choice rendering shared by training and both runtimes (torch-free).

Each option is encoded as "question option", cut to max_choice_tokens. A long question used to push the option past the
window, so "yes" and "no" became the same token sequence and every answer came out at exactly 0.5.

- `fit=True` (models trained from 2026-10-05, `"choice_fit": true` in decima.json): the option is always kept whole and only
  this copy of the question is shortened, at a word boundary, with "…" where it was cut. The full question still reaches
  the model on the state side (`question_in_state`).
- `fit=False` (earlier models): the rendering they were trained with — the right-hand cut — except when that cut makes two
  options identical, which can only ever give a tie; then the fitted rendering is used for that question.
"""

from __future__ import annotations

from .normalize import normalize


def choice_texts(tok, question: str, choices: list[str], lang: str, prefix: str, max_len: int, fit: bool = True) -> list[str]:
    if not fit:
        legacy = [prefix + normalize(f"{question} {c}".strip(), lang) for c in choices]
        rows = {tuple(tok(t, truncation=True, max_length=max_len)["input_ids"]) for t in legacy}
        if len(rows) == len(legacy):
            return legacy
    out = []
    for c in choices:
        full = prefix + normalize(f"{question} {c}".strip(), lang)
        if len(full) <= max_len - 2 or len(tok(full)["input_ids"]) <= max_len:   # a token is at least one character
            out.append(full)
            continue
        opt = normalize(c, lang)
        words = normalize(question, lang).split()
        lo, hi = 0, len(words)
        while lo < hi:                                                           # the longest question head that fits
            mid = (lo + hi + 1) // 2
            if len(tok(f"{prefix}{' '.join(words[:mid])} … {opt}")["input_ids"]) <= max_len:
                lo = mid
            else:
                hi = mid - 1
        out.append(f"{prefix}{' '.join(words[:lo])} … {opt}" if lo else prefix + opt)
    return out
