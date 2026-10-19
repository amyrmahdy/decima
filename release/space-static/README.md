---
title: Decima Playground
emoji: ⚖️
colorFrom: blue
colorTo: gray
sdk: static
app_file: index.html
pinned: true
license: apache-2.0
short_description: Open decision and guard models, in your browser
models:
  - amyrmahdy/decima-guard
  - amyrmahdy/decima-guard-small
  - amyrmahdy/decima-agent
  - amyrmahdy/decima-base
  - amyrmahdy/decima-small
tags: [decision-model, system-one, jev, jev-alternative, agents, guardrails, prompt-injection, llm-security, owasp, arabic, persian, knowledge-graph, calibration, multilingual, onnx, onnxruntime-web, webassembly, in-browser, zero-shot-classification]
---

# Decima Playground

**[Check a message with decima-guard](https://amyrmahdy-decima-playground.static.hf.space/guard.html):** paste a user message, a document, a model reply or a web
request, and see whether [decima-guard](https://huggingface.co/amyrmahdy/decima-guard) would block it. Prompt
injection, leaks, unsafe output and web attacks, in English, Arabic and Persian.

The main page has every open Decima model:
**[decima-guard](https://huggingface.co/amyrmahdy/decima-guard) 1.0 preview**,
**[decima-guard-small](https://huggingface.co/amyrmahdy/decima-guard-small) 1.0 preview**,
**[decima-agent](https://huggingface.co/amyrmahdy/decima-agent) 2.3**,
**[decima-base](https://huggingface.co/amyrmahdy/decima-base) 2.0** and
**[decima-small](https://huggingface.co/amyrmahdy/decima-small) 1.1**. Pick one of the presets of the
[`decima` package](https://pypi.org/project/decima-ai/) (triage, routing, moderation, pii, guard, guard-rag,
guard-output, guard-web, guardrails, agent-gate, kg-judge) or write your own question and options, paste some text,
and see the calibrated probabilities with the `/v1/systemone` response that `decima.Router` returns.

**The models run in your browser.** Nothing loads until the first decision; then the page downloads the chosen int8
ONNX export once (about 370 MB for decima-guard, decima-agent or decima-base, 187 MB for decima-guard-small, 144 MB for decima-small, then cached on your device) and
runs it with [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) on WebAssembly. There is no server behind
this Space: nothing you type leaves your device.

- **[Guard page](https://amyrmahdy-decima-playground.static.hf.space/guard.html)**: what you are checking (user message, document or tool result, model reply, web
  request), the text, and a verdict with the probability of each check. The examples there are written to show each
  check and run live; the model card has the benchmark numbers.
- **Main page** (`index.html`): model, preset or your own question, text, probability bars, the JSON answer, and the
  same call in Python and on the command line. Eight examples (triage in English and Persian, personal data in Arabic,
  model routing, an agent command gate, a knowledge-graph judgment, a prompt-injection check, decima-small on its own
  options), each recorded with `decima-ai` 2.2.0 before it was added (`examples/models.json`).
- **[Decima-small playground](small.html)**: business cases with the teacher's answers, the shuffle test, auto-routing
  with a confidence threshold, and a speed sweep on your device.
- **[City dispatch](dispatch.html)**: a live demo on decima-small.

How it is built: `decima.js` ports the Python runtime of `decima-ai` 2.3.0 (`DecimaOnnx`, script normalisation, choice
rendering, encoder → scorer, temperature, softmax / ordinal / sigmoid heads) and `router.js` ports `decima.Router`
(presets, state templates, `requires`, TypeSafe question mapping and answers). `presets.json` is the package's
`decima/presets/*.json`, unchanged. `worker.js` runs the model in a Web Worker, and the tokenizer is
[tokenizers.js](https://github.com/huggingface/tokenizers.js) reading each export's `tokenizer.json`. ONNX Runtime Web
computes the int8 layers with float activations; the Python runtime doing the same on the same weights gives the
probabilities of this page, while native ONNX Runtime on a server CPU uses int8 activations and can differ by a few
hundredths on close calls.

Examples are curated; see the model cards for benchmark accuracy and limitations. Business cases in
`examples/business_cases.json` are synthetic (generated and labelled by Gemma-4-26B-A4B for the Decima project) and
released under **CC BY 4.0**.
