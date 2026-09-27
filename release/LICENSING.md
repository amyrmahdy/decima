# Decima — licensing review for the model and dataset release

> **This is not legal advice.** It is an engineering summary of public licence texts, checked on
> 2026-09-26, so the owner can make the decisions in §5 and, if needed, take them to a lawyer. Licence
> pages change, so re-check every link on the day you publish. Where this file says "unsettled", that
> means there is no clear law or case law yet, not that the risk is zero.

## Decisions taken (2026-09-26)

- **Model and code: Apache-2.0**, released as trained (v1i), with full training-data disclosure in the model card.
- **Synthetic dataset: CC BY 4.0.**

## 1. The teacher: Gemma 4 is Apache-2.0, not the Gemma Terms of Use

The teacher is **Gemma-4-26B-A4B-it**. We served the NVFP4 build `nvidia/Gemma-4-26B-A4B-NVFP4` through
vLLM, behind a local gateway (model alias `rande-fast-local` in our data files).

| source | what it says (verbatim) |
|---|---|
| [HF card google/gemma-4-26B-A4B](https://huggingface.co/google/gemma-4-26B-A4B/blob/main/README.md) (also [-it](https://huggingface.co/google/gemma-4-26B-A4B-it)) | front matter `license: apache-2.0`, `license_link: https://ai.google.dev/gemma/docs/gemma_4_license`; body "License: Apache 2.0". Not gated (HF API `gated=False`). |
| [ai.google.dev/gemma/docs/gemma_4_license](https://ai.google.dev/gemma/docs/gemma_4_license) | Standard "Apache License, Version 2.0, January 2004" text, with no Google-specific additions. |
| [Gemma Terms of Use](https://ai.google.dev/gemma/terms) (last modified 2026-04-01) | "For Gemma 4 terms, see the Gemma 4 license". The ToU Appendix lists Gemma 1, 1.1, 2, 3, 3n, EmbeddingGemma, PaliGemma, ShieldGemma, CodeGemma, … **Gemma 4 is not listed.** |
| same page, for contrast (applies to Gemma ≤ 3n **only**) | "'Model Derivatives' means … (iii) any other machine learning model which is created by transfer of patterns of the weights, parameters, operations, or Output of Gemma, to that model in order to cause that model to perform similarly to Gemma, including distillation methods that use intermediate data representations or methods based on the generation of synthetic data Outputs by Gemma for training that model. For clarity, Outputs are not deemed Model Derivatives." / "Google claims no rights in Outputs you generate using Gemma. You and your users are solely responsible for Outputs and their subsequent uses." |
| [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy) | "You may not use nor allow others to use Gemma or Model Derivatives to: …". It binds through the ToU, which does not cover Gemma 4. The Apache text does not include it. |
| [HF card nvidia/Gemma-4-26B-A4B-NVFP4](https://huggingface.co/nvidia/Gemma-4-26B-A4B-NVFP4) | License metadata Apache-2.0; "License and Terms of Use: [Apache License 2.0 \| Gemma](https://ai.google.dev/gemma/apache_2)"; "This model is not owned or developed by NVIDIA". |
| [Google Open Source Blog, 2026-04-02](https://opensource.googleblog.com/2026/03/gemma-4-expanding-the-gemmaverse-with-apache-20.html) | "The release of Gemma 4 under the Apache 2.0 license …" / "well-understood terms for modification, reuse, and further development". |

What this means for us:

- **(i) Distributing a model trained on Gemma 4 outputs.** Apache-2.0 governs the *Work*, meaning the
  Gemma weights and code. It has no clause about outputs and no "Model Derivatives" or distillation
  clause. We do not redistribute Gemma weights, so the §4 redistribution duties (licence copy, NOTICE)
  are not triggered. Even on the most cautious reading, where our weights count as a derivative work,
  Apache-2.0 allows derivative works under any licence as long as attribution is kept. **Gemma 4 places
  no constraint on Decima's licence.** Under Gemma 1–3n it would have: our model would have been a
  "Model Derivative" and would have had to carry the Gemma ToU and the use restrictions.
- **(ii) Publishing Gemma-generated synthetic data.** Nothing in Apache-2.0 restricts it. We should
  still (a) name the teacher and its licence in the card, (b) say the data is machine-generated, and
  (c) optionally ask users to follow Google's Prohibited Use Policy as good practice. That request is
  voluntary, not a licence term.
- **Copyright in the outputs.** Text generated purely by a model may not be protected by copyright in
  some jurisdictions (for example, the US Copyright Office's human-authorship requirement). A licence
  on the synthetic dataset therefore mostly sets expectations (attribution, no warranty). It may not
  be enforceable against someone who ignores it. In the EU, the sui generis database right may still
  cover the compiled dataset.

## 2. Licence table

"Redistribute data?" asks whether we may publish the rows (text) ourselves. "Model constraint" asks
what the asset implies for Decima-small's weights licence. Mirrors we loaded from (`mteb/*`) relabel
licences, but a mirror cannot change the original terms, so the original is what counts.

| asset (how Decima used it) | licence (source) | redistribute data? | model constraint |
|---|---|---|---|
| **Gemma-4-26B-A4B-it** teacher (NVFP4 build) | Apache-2.0 ([HF](https://huggingface.co/google/gemma-4-26B-A4B-it), [Google](https://ai.google.dev/gemma/docs/gemma_4_license)) | outputs: **yes**, no licence restriction | none; attribution in the card is good practice |
| **intfloat/multilingual-e5-small** (base weights, redistributed as part of Decima) | MIT ([HF](https://huggingface.co/intfloat/multilingual-e5-small)) | n/a | any licence; **keep the MIT copyright + permission notice** in the model repo |
| **Our synthetic data**: `generate` (86.9k), `jevgen` (10.2k), label-file `gen:*` re-asks (92.4k) | ours (Gemma 4 output) | **yes** | none |
| MultiNLI `nyu-mll/multi_nli` (gold 66k) | mixed: OANC (permissive), fiction *Seven Swords* CC BY-SA 3.0, others CC BY 3.0 / US public domain ([HF card](https://huggingface.co/datasets/nyu-mll/multi_nli), [site](https://cims.nyu.edu/~sbowman/multinli/)) | yes, with attribution; SA applies to part | SA on data; whether SA reaches trained weights is unsettled (generally assumed not) |
| SNLI `stanfordnlp/snli` (gold 20k) | CC BY-SA 4.0 ([site](https://nlp.stanford.edu/projects/snli/)): "licensed under a Creative Commons Attribution-ShareAlike 4.0 International License" | yes, BY-SA | as above |
| **ANLI** `facebook/anli` (gold 30k) | **CC BY-NC 4.0** ([GitHub](https://github.com/facebookresearch/anli)): "ANLI is licensed under Creative Commons-Non Commercial 4.0" | non-commercial only | **NC risk** (see §3) |
| WANLI `alisawuffles/WANLI` (gold 25k) | CC BY 4.0 on [HF card](https://huggingface.co/datasets/alisawuffles/WANLI); the GitHub repo has no LICENSE. GPT-3-generated, crowd-revised; premises from MNLI | yes, attribution | none known. The OpenAI output terms bound the WANLI authors, not us (low risk) |
| **XNLI** `facebook/xnli` (gold 88k + **teacher-labelled 33.6k**) | **CC BY-NC 4.0** ([LICENSE](https://github.com/facebookresearch/XNLI/blob/main/LICENSE): "Creative Commons Attribution-NonCommercial 4.0 International Public License"). The HF card omits it | non-commercial only | **NC risk** |
| FarsTail, loaded from `azarijafari/FarsTail` (gold 14.5k) | Apache-2.0 ([dml-qom/FarsTail](https://github.com/dml-qom/FarsTail)); the HF mirrors have no or "unknown" licence | yes, keep notice | none |
| **multilingual-NLI-26lang-2mil7** `MoritzLaurer/...` (gold 138k) | no licence on the [HF card](https://huggingface.co/datasets/MoritzLaurer/multilingual-NLI-26lang-2mil7). It is a machine translation of MNLI, **ANLI**, Fever-NLI, LingNLI and WANLI and inherits their terms (strictest: **NC** from ANLI; Fever/LingNLI not verified) | **no** | **NC risk** (ANLI part) |
| BoolQ `google/boolq` (gold 17.8k) | CC BY-SA 3.0 ([GitHub](https://github.com/google-research-datasets/boolean-questions)): "released under the Creative Commons Share-Alike 3.0 license". Passages from Wikipedia (BY-SA) | yes, BY-SA | SA question as for SNLI |
| banking77 (`mteb/banking77`; gold-cls 42k + teacher-labelled 11.3k) | CC BY 4.0 ([PolyAI](https://github.com/PolyAI-LDN/task-specific-datasets)); the mteb mirror says MIT, but that relabel has no effect | yes, attribution | none |
| CLINC150 `clinc/clinc_oos` plus (gold-cls 42k + teacher-labelled 11.3k) | CC BY 3.0 ([LICENSE](https://github.com/clinc/oos-eval/blob/master/LICENSE)) | yes, attribution | none |
| MASSIVE (`mteb/amazon_massive_intent`; gold-cls 137k + teacher-labelled 44.8k) | CC BY 4.0 ([NOTICE](https://github.com/alexa/massive/blob/main/NOTICE.md): "licensed under CC BY 4.0, Copyright Amazon.com Inc. or its affiliates"); the mteb mirror says Apache-2.0 | yes, attribution + Amazon copyright line | none |
| **AG News** `fancyzhx/ag_news` + `sh0416/ag_news` titles (gold-cls 45k + teacher-labelled 11.4k) | **no formal licence**; the HF card says `unknown`. The author page ([Gulli](http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html), quoted via the HF card because the page was unreachable) says the data is "provided by the academic community for research purposes … and any other non-commercial activity". The text is third-party news | **no** | **NC / unclear risk** |
| **SST-5** `SetFit/sst5` (gold-cls 32k + teacher-labelled 11.4k) | **no licence** ([SST site](https://nlp.stanford.edu/sentiment/), [SetFit/sst5](https://huggingface.co/datasets/SetFit/sst5)). The text is Rotten Tomatoes review snippets (Pang & Lee) | **no** | **unclear risk** |
| Teacher labels on public train text (label file, 123.8k rows: banking77, clinc150, massive en/fa/ar/ru, agnews, sst5, xnli en/ar/ru) | state text follows the source licence; question, choices and probabilities are our Gemma output | text: **no**; labels only + a re-hydration script: yes | follows the source row (NC for xnli/agnews; unclear for sst5) |
| Eval items (btzsc, kev, laya, decima, jevbench, typed-decisions, JDI) | btzsc: per source, no licence on the [HF card](https://huggingface.co/datasets/btzsc/btzsc); typed-decisions: Apache-2.0; [jevbench](https://github.com/fstandhartinger/jevbench): MIT; [JDI kit](https://github.com/apolinario/decision-index): MIT; the rest are test splits of the datasets above | we publish **ids + probabilities only**, and loaders rebuild the text | none (test data, not trained on) |
| Competitor models whose predictions we publish (`jaredpalmer/kev-0.5b`, `kev-0.8b`, `convaiinnovations/laya`, `laya-multilingual`, e5 baseline) | Apache-2.0 / MIT (HF API) | their outputs: yes | n/a |

The row counts are the Decima-small (v1i) training inputs from `docs/EVAL.md` §1b. `mteb/FarsTail` is
listed in `teacher/sources.py`, but that repo ships only a test split, so it contributed 0 rows.

## 3. The non-commercial question

Decima-small (v1i) was trained on ANLI, XNLI and 26lang-2mil7 (all CC BY-NC or containing it), and on
AG News and SST-5 (no licence; research or non-commercial use at best). No court has settled whether a
model's weights are bound by an NC licence on its training data. Training may also be covered by
TDM exceptions (EU DSM Art. 3/4) or by fair use. Precedent on the Hub is permissive:
`MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` (trained on 26lang-2mil7, which includes
ANLI) and `facebook/bart-large-mnli` are both MIT. That shows common practice, not legal safety.
Separately, *republishing* the NC or unlicensed **text** is clearly not allowed commercially, so we
don't do it.

## 4. Recommendation

**(a) Model licence**, with options and trade-offs:

| option | pros | cons |
|---|---|---|
| **A. Apache-2.0 (or MIT) for v1i as trained**, with a "Training data licences" section that names the NC/unlicensed sources | maximum adoption; matches the base (MIT), the teacher (Apache) and Hub precedent; Apache adds a patent grant | residual, unsettled risk that the NC sources' terms are claimed to reach the weights, which matters most if Rande sells it |
| **B. CC BY-NC 4.0 for v1i** | matches the strictest data | blocks commercial users and Rande's own product use by others; CC licences are a poor fit for weights; costs adoption and reach |
| **C. "Commercial-clean" retrain, then Apache-2.0** — drop ANLI, XNLI (gold *and* teacher-labelled xnli rows), 26lang-2mil7, AG News and SST-5 (gold *and* teacher-labelled); keep MNLI, SNLI, WANLI, BoolQ, FarsTail, banking77, CLINC, MASSIVE and all synthetic data | removes the NC question; permissive licence with a clean story | one more training run plus re-eval; likely loss on AR/RU/multilingual NLI (XNLI/26lang were the main non-English NLI signal) and on AG News/SST-style suites. Needs measuring |

**My recommendation:** if Decima will be used or sold commercially, do **C** and release under
**Apache-2.0**. Replace the missing multilingual NLI with Gemma-labelled synthetic pairs, which are
clean. If v1i ships before a clean retrain, **A with full disclosure** is common practice on the Hub,
but the residual risk is the owner's to accept. Whatever the choice: keep the e5 MIT notice, credit
Gemma 4 (Apache-2.0), and list every training source with its licence in the card.

**(b) Publish as-is** (datasets we own):
- `decima-synthetic-decisions` has configs `short` (teacher.generate), `long` (Jev-style), and optionally
  `relabel` (label-file `gen:*` rows: our generated states re-asked). All rows are Gemma 4 output and
  contain no third-party text (§6). Suggested licence: **CC BY 4.0** (attribution helps reach; it also
  covers EU database rights) or Apache-2.0 to match the code. CC0 gives the widest reuse.
- `decima-bench-predictions`: every system's `{id, probs}` on every item set, scores and manifests.
  These are our measurements, with no third-party text. Suggested licence: CC BY 4.0.

**(c) Scripts only** (third-party text, mixed or NC/unclear licences):
- `data/gold` (NLI + BoolQ gold renders) → `teacher/gold.py`
- `data/gold-cls` (banking77/CLINC/MASSIVE/AG News/SST-5 renders) → `teacher/gold_cls.py`
- teacher labels on public train text (label file, non-`gen:` rows) → `teacher/label.py`, or a
  **labels-only** release (question/choices/probs + a hash of the source text, re-hydrated by a script
  that downloads the source). That is possible, but it needs a hydration script that does not exist yet.
- eval items → `bench/items.py` + suites.

Even the CC BY / BY-SA subsets (banking77, CLINC, MASSIVE, SNLI, BoolQ) are better shipped as scripts.
Republishing would mean carrying per-row attribution and share-alike, and it would copy the mteb mirrors'
licence mix-ups. Scripts should pin dataset **revisions**. `teacher/gold*.py` and the bench loaders call
`load_dataset` without `revision=`, so add that before release.

## 5. Decisions for the owner

Status 2026-09-26: items 1, 2 and 5 are **decided** — model and code Apache-2.0 with a training-data
disclosure in the card (option A), synthetic data CC BY 4.0 (the bench-predictions dataset follows it),
LICENSE file added. Items 3, 4, 6 and 8 remain open (CHECKLIST 3.1c, 3.9).

1. **Model licence**: A, B or C (§4a). If C, approve one retrain plus re-eval. → **Decided: A, Apache-2.0 + disclosure.**
2. **Synthetic dataset licence**: CC BY 4.0 / Apache-2.0 / CC0. → **Decided: CC BY 4.0.**
3. **Include the `relabel` config** (92.4k more fully synthetic rows that were in training)? Recommended: yes.
4. **Free-mail scrub**: some invented addresses use real providers (`@gmail.com` in ≈445 states,
   `@mail.ru` ≈271, `@protonmail.com` ≈99, `@outlook.com` ≈21). `build_synthetic.py --scrub-free-mail`
   rewrites them to reserved `*.example` domains, but then the rows no longer match what the model was
   trained on. Recommended: scrub. The phishing-style labels depend on "free-mail vs corporate", which
   `gmail.example` keeps.
5. **Repository licence**: → **Decided: Apache-2.0; LICENSE added.**
6. **Attribution text**: the Amazon copyright line for MASSIVE, the PolyAI/CLINC credits, and the
   citation requests (Laurer et al. 2022 for 26lang) belong in the model card's training-data section.
7. **Optional voluntary clause**: ask dataset and model users to follow Google's Gemma Prohibited Use
   Policy. It is not required for Gemma 4.
8. **Legal review** of §3 if Rande will sell Decima or services built on it.

## 6. Checks behind "no third-party text / no personal data" (2026-09-26)

- Exact-match and shared-10-word-n-gram search of every distinct `generate` (86,893) and `jevgen` (2,692)
  state against 751,601 public texts we hold locally (label-file public states + all gold/gold-cls
  segments). Result: 0 exact matches, 0 hits in jevgen, and 1 hit in generate on a generic phrase
  ("i m not sure if it s because of the"). The comparison set only covers the public data we actually
  downloaded, so it cannot rule out memorised web text in general.
- The prompts ask for invented names ("invent fresh organisation and person names"; "real-sounding
  names" in `teacher/prompts.py`). Spot checks found invented people and organisations with heavy
  repetition ("Marcus Thorne" in 814/2,692 long states, "Sarah Jenkins" in 1,180 short states). Card and
  bank numbers are mostly well-known test values (`4111 1111 1111 1111`, `DE89 3704 0044 0532 …`). There
  are 2 rows where a celebrity name is used for a fictional employee. No real person's data was found,
  but a manual review of a random sample of ≥ 200 states before publishing is still advised.
