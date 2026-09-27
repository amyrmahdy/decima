---
title: Decima Playground
emoji: ⚖️
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.28.0
python_version: "3.11"
app_file: app.py
pinned: true
license: apache-2.0
short_description: Calibrated decisions from a 122M model on one CPU core
models:
  - "amyrmahdy/decima-small"
tags: [decision-model, calibration, multilingual, onnx, cpu, zero-shot-classification]
---

# Decima Playground

Interactive demo of **Decima-small**: give it a situation, a question and your own options (free text, any
number, any order) and it returns calibrated probabilities — about 20 ms per decision on one CPU core.

- **Try it** — your own inputs, plus curated examples (multilingual routing, 150 intents, urgency levels,
  help-article relevance, yes/no, news topic). Every example was run before it was added; its recorded
  output is in `examples/gallery.json`.
- **Business cases** — long synthetic cases the model never trained on, with the 26B teacher's answer next
  to Decima's for every question, disagreements included.
- **Shuffle test** — reorder the options and watch the probabilities stay put.
- **Auto-route with confidence** — move a threshold over 20 tickets and see what gets automated.
- **Speed** — a live latency sweep on this Space's CPU, next to our x86 reference numbers.

Model: [amyrmahdy/decima-small](https://huggingface.co/amyrmahdy/decima-small) · int8 ONNX on ONNX Runtime, no PyTorch.
Examples are curated; see the model card for benchmark accuracy and limitations.

Business cases in `examples/business_cases.json` are synthetic (generated and labelled by
Gemma-4-26B-A4B for the Decima project) and released under **CC BY 4.0**.
