---
pretty_name: jabr classifier-benchmark v2 in Persian
license: cc0-1.0
language:
- fa
task_categories:
- text-classification
- zero-shot-classification
size_categories:
- n<1K
annotations_creators:
- expert-generated
language_creators:
- machine-generated
- expert-generated
source_datasets:
- extended|jabr/classifier-benchmark
tags:
- persian
- farsi
- benchmark
- system-one
- jev
- typesafe
- typed-decisions
- evaluation
configs:
- config_name: fa
  default: true
  data_files:
  - {split: test, path: fa/test.parquet}
- config_name: fa_state
  data_files:
  - {split: test, path: fa_state/test.parquet}
---

# jabr/classifier-benchmark v2 in Persian (فارسی)

The 48 translatable tasks of [jabr/classifier-benchmark](https://github.com/jabr/classifier-benchmark) v2, in Persian,
with the same answers: 845 cases. Every Persian case is the exact counterpart of an English one, so a decision model's
English-to-Persian gap can be measured on identical decisions. As far as we know, it is the first Persian benchmark
for System One / typed-decision models.

Made by **A. M. Madani** ([@amyrmahdy](https://github.com/amyrmahdy)).

| config | |
|---|---|
| `fa` | question, option descriptions and text all in Persian |
| `fa_state` | Persian text with the original English question: an English schema over Persian user input, the common case in production |

Columns: `id`, `task`, `type` (`noul`, `choice`, `score`), `instructions`, `criteria` (JSON), `state`, `expected`.
The benchmark's own TOML files are in `toml/`: copy them into `sources/samples/` of the benchmark and run with
`--sample v2-fa` or `--sample v2-fa-state`.

## How it was made

1. **Machine translation, one task per call**, with these rules: Persian as people in Iran actually write it; every
   fact, number, amount and hedge kept; code, URLs, identifiers and product names untouched; Western digits.
2. **Second-model check.** A different model family answered every case in English and in Persian. Cases right in
   English but wrong in Persian, and cases whose numbers changed, were flagged. On the final set that model scores
   0.948 in English and 0.941 in Persian, so very little meaning was lost.
3. **Review of all 845 cases against the English**, with 105 rewrites:
   - register (formal vs colloquial Persian, which the formality task depends on);
   - terms that misled (for example "it's fine" is «معمولی», not «خوب است»);
   - Iranian usage (for example «پیکور», «کناف», «نبش»).

   Amounts are written «85 دلار», because a `$` breaks right-to-left text. Facts that decide answers stay as in the
   English even where unusual in Iran (dollars, miles, Friday meetings).

One task, `grammar_issue`, was left out: it asks about English spelling and grammar, which cannot survive translation.

The review was done case by case, but not by a panel of native annotators. Please report results as
"machine-translated, reviewed".

## Results (macro accuracy, int8 on CPU, the benchmark's harness)

| | English v2 | `fa` | `fa_state` |
|---|---:|---:|---:|
| [decima-base](https://huggingface.co/amyrmahdy/decima-base) | 0.673 | 0.616 | 0.639 |
| [decima-agent](https://huggingface.co/amyrmahdy/decima-agent) | 0.675 | 0.640 | 0.624 |

Results from other models are welcome.

## License

CC0, like the original benchmark. The translation scripts and review notes are in
[amyrmahdy/decima/release/jabr-fa](https://github.com/amyrmahdy/decima/tree/main/release/jabr-fa).
