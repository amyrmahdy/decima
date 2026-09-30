---
title: Decima Playground
emoji: ⚖️
colorFrom: blue
colorTo: gray
sdk: static
app_file: index.html
pinned: true
license: apache-2.0
short_description: Open Jev-style decision model, in your browser
models:
  - amyrmahdy/decima-small
tags: [decision-model, system-one, jev, jev-alternative, calibration, multilingual, onnx, onnxruntime-web, webassembly, in-browser, zero-shot-classification]
---

# Decima Playground

Interactive demo of **[Decima-small](https://huggingface.co/amyrmahdy/decima-small)**: give it a situation, a
question and your own options (free text, any number, any order) and it returns calibrated probabilities.

**The model runs in your browser.** The page downloads the int8 ONNX export once (about 140 MB, then cached on your
device) and runs it with [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) on WebAssembly. There is no
server behind this Space: nothing you type leaves your device.

- **Try it** — your own inputs, plus curated examples (multilingual routing, 150 intents, urgency levels,
  help-article relevance, yes/no, news topic). Every example was run before it was added; its recorded
  output is in `examples/gallery.json`.
- **Business cases** — long synthetic cases the model never trained on. Decima answers every question live; the
  26B teacher's recorded answer sits next to it, disagreements included.
- **Shuffle test** — reorder the options and watch the probabilities stay put.
- **Auto-route** — your browser scores 20 tickets once; then move a confidence threshold and see what gets automated.
- **Speed** — a latency sweep on your own device, next to our x86 reference numbers.

How it is built: `decima.js` is a line-for-line port of the Python runtime (`DecimaOnnx`: script normalisation,
prompt construction, encoder → scorer, temperature, softmax / ordinal / sigmoid heads), `worker.js` runs it in a Web
Worker, and the tokenizer is [tokenizers.js](https://github.com/huggingface/tokenizers.js) — the tokenizer inside
Transformers.js — reading the exported `tokenizer.json`. Token ids match the Python tokenizer on 200 of 200 test strings
in 20 languages. ONNX Runtime Web computes the int8 layers with float activations; the Python runtime doing the
same on the same weights gives probabilities within 3 × 10⁻⁴ of this page. (Native ONNX Runtime on a server CPU
uses int8 activations and can differ by up to a few hundredths on close calls.)

Examples are curated; see the model card for benchmark accuracy and limitations. Business cases in
`examples/business_cases.json` are synthetic (generated and labelled by Gemma-4-26B-A4B for the Decima project) and
released under **CC BY 4.0**.
