# Decima datasets: what we publish, and what we publish only as scripts

Decima-small was trained on three kinds of data. We publish each kind differently, depending on
**who owns the text**. The per-asset licences and quotes are in [`../../LICENSING.md`](../../LICENSING.md).
Nothing here is legal advice.

| data | rows (v1i training) | whose text | how we publish |
|---|---:|---|---|
| Teacher-invented decisions (`teacher/generate.py`) | 86,900 | ours: Gemma 4 output, Apache-2.0 teacher | **dataset** `decima-synthetic-decisions`, config `short` |
| Long Jev-style business cases (`teacher/generate_jev.py`) | 10,244 (6,004 in v1i) | ours | **dataset**, config `long` |
| Our generated states re-asked under new questions (`teacher/label.py`, `gen:*` rows) | 92,427 | ours | **dataset**, config `relabel` |
| Teacher labels on **public train-split text** (`teacher/label.py`, other rows): banking77, CLINC150, MASSIVE en/fa/ar/ru, AG News, SST-5, XNLI en/ar/ru | 123,841 | **third party** (text); ours (question, choices, probabilities) | **script only** (`teacher/label.py`). Later, possibly a labels-only release plus a re-hydration script |
| Gold NLI / reading data re-rendered as decisions (`teacher/gold.py` → `data/gold`): MNLI, SNLI, ANLI, WANLI, XNLI, FarsTail, multilingual-NLI-26lang, BoolQ | 399,256 | **third party** | **script only** |
| Gold classification data re-rendered (`teacher/gold_cls.py` → `data/gold-cls`): banking77, CLINC150, MASSIVE (51 locales), AG News, SST-5 | 298,400 (1/3 used in v1i) | **third party** | **script only** |
| Evaluation items (`bench/items.py` + `bench/suites_*.py`, `bench/datasets.py`) | ≈ 203k predicted items (481,586 in the full JDI catalogue) | **third party** | **script only**. Predictions, scores and a text-free sha256 manifest go in **dataset** `decima-bench-predictions` |

## Why some data is scripts only

1. **Licences we cannot pass on.** ANLI and XNLI are CC BY-NC 4.0, and multilingual-NLI-26lang contains
   ANLI. AG News and SST-5 have no licence; AG News is "research / non-commercial" at best. Republishing
   their text would mean granting rights we do not have.
2. **Mixed attribution and share-alike duties.** SNLI and BoolQ are CC BY-SA, MNLI mixes OANC, BY-SA and
   BY, and banking77, CLINC and MASSIVE are CC BY. A single re-rendered file would have to carry every
   source's attribution and SA terms row by row. The `mteb/*` mirrors we loaded from also carry wrong
   licence tags (MIT or Apache for CC BY data), and republishing would copy that confusion.
3. **The script is the better artefact anyway.** The builders are seeded and deterministic. They record
   decontamination against every evaluation set (`data/gold*/stats.json`), and they download from the
   original source, so users accept the original terms directly.

The loaders call `load_dataset` without `revision=`, so a Hub change would silently change the data.
For the evaluation items, `decima-bench-predictions/code-ref.json` records the Hub revision of every
source dataset as resolved when our items were built, and `bench.verify_items` checks a rebuild against
the published text-free manifest, so drift is detected rather than silent. The repository is Apache-2.0.

## Why the synthetic data is safe to publish

- **Teacher terms.** Gemma 4 (Gemma-4-26B-A4B-it) is Apache-2.0, which has no restriction on outputs.
  The older Gemma Terms of Use, with their "Model Derivatives"/distillation clause, cover Gemma ≤ 3n,
  not Gemma 4.
- **No third-party text.** Each distinct generated state was checked against the 751,601 public texts
  used elsewhere in training: 0 exact matches, and one generic 10-word phrase shared.
- **Invented people.** The prompts ask for invented names. Spot checks found invented (and heavily
  repeated) names, test card numbers and example IBANs. Some addresses sit on real free-mail domains;
  `--scrub-free-mail` rewrites those to `*.example`.

## Building

```bash
# synthetic dataset → parquet + jsonl.gz splits + manifest.json (sha256 of inputs and outputs, stats)
uv run python release/hf/datasets/build_synthetic.py --out /path/outside/repo/decima-synthetic-decisions \
    --with-relabels --scrub-free-mail          # the published build uses both flags

# predictions dataset → parquet predictions, text-free item manifests, scores, code-ref.json
uv run python release/hf/datasets/build_predictions.py --out /path/outside/repo/decima-bench-predictions
```

The builder is deterministic: split membership comes from `sha1(group)`, rows are sorted by id, and
gzip mtime is fixed. It refuses to write inside the repository, never exports the rows labelled on
public text, and re-validates every row. The internal fields `model`, `cell`, `source` and `batch`
and the empty resume markers are dropped; `format` and `uncertain_asked` are renamed. See the
docstring for the schema.

| folder | card |
|---|---|
| `decima-synthetic-decisions/` | fully synthetic training data (`short`, `long`, `relabel`) |
| `decima-bench-predictions/` | every system's predictions on every eval item set, plus scores; no eval text |
