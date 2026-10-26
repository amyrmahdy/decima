"""decima-pii as a LiteLLM proxy guardrail: personal data is masked before a request reaches the model.

LiteLLM's built-in PII masking runs on Presidio, whose default recognizers are English-first; decima-pii finds names,
phones, emails, national IDs, IBANs, cards, passports, addresses, dates of birth and IP addresses in Arabic, Persian,
Urdu, Turkish and English (Persian and Arabic-Indic digits too), on CPU, inside the proxy process.

Put this file next to config.yaml and reference it as `guardrail: decima_pii_guardrail.DecimaPIIGuardrail`.
Extra keys under the guardrail's litellm_params are passed to __init__ (see config.yaml).

It implements LiteLLM's unified `apply_guardrail` hook: LiteLLM hands over the texts of the request (user, system and
assistant messages on /v1/chat/completions, /v1/responses and /v1/messages alike) and writes the masked texts back.
With `mode: post_call` the model's reply is masked the same way.
"""

from __future__ import annotations

import asyncio
from typing import Any, Literal

from fastapi import HTTPException
from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.types.guardrails import GuardrailEventHooks

from decima.pii import DEFAULT_ONNX, DEFAULT_REPO, DecimaPII, redact_spans


class DecimaPIIGuardrail(CustomGuardrail):
    """Mask (or block on) personal data with decima-pii.

    model:        Hub repo or local directory (default amyrmahdy/decima-pii)
    onnx:         ONNX file inside onnx/ (default model_nbits8.onnx, int8)
    threads:      ONNX Runtime threads per call (default 4)
    entities:     only these labels (default: all ten); e.g. [PHONE, EMAIL, NATIONAL_ID, IBAN, CARD]
    mask_format:  replacement text, "{label}" and "{text}" are filled in (default "[{label}]")
    pii_action:   "mask" (default) rewrites the text; "block" refuses the request with HTTP 400 and the labels found
    """

    def __init__(self, model: str | None = None, onnx: str | None = None, threads: int = 4, entities: list[str] | None = None,
                 mask_format: str = "[{label}]", pii_action: Literal["mask", "block"] = "mask", **kwargs: Any) -> None:
        self.pii = DecimaPII.from_pretrained(model or DEFAULT_REPO, onnx=onnx or DEFAULT_ONNX, threads=int(threads))
        self.entities = set(entities) if entities else None
        self.mask_format = mask_format
        self.pii_action = pii_action
        super().__init__(supported_event_hooks=[GuardrailEventHooks.pre_call, GuardrailEventHooks.post_call,
                                                GuardrailEventHooks.logging_only], **kwargs)

    def _scan(self, texts: list[str]) -> list[tuple[str, list[dict]]]:
        out = []
        for t in texts:
            spans = [s for s in self.pii.find(t) if self.entities is None or s["label"] in self.entities] if t and t.strip() else []
            out.append((redact_spans(t, spans, self.mask_format) if spans else t, spans))
        return out

    async def apply_guardrail(self, inputs: dict, request_data: dict, input_type: Literal["request", "response"],
                              logging_obj: Any = None) -> dict:
        texts = list(inputs.get("texts") or [])
        if not texts:
            return inputs
        scanned = await asyncio.to_thread(self._scan, texts)             # CPU work off the event loop
        found: dict[str, int] = {}
        for _, spans in scanned:
            for s in spans:
                found[s["label"]] = found.get(s["label"], 0) + 1
        if found:
            verbose_proxy_logger.info("decima-pii (%s, %s): %s", self.guardrail_name, input_type, found)
            if self.pii_action == "block":
                raise HTTPException(status_code=400, detail={"error": "Personal data is not allowed in this request.",
                                                             "guardrail": self.guardrail_name, "entities": found})
        inputs["texts"] = [t for t, _ in scanned]
        return inputs
