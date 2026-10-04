# jabr/classifier-benchmark v2 in Persian (فارسی)

The 49 tasks of [jabr/classifier-benchmark](https://github.com/jabr/classifier-benchmark) v2 (CC0), in Persian, with the
same answers. Every Persian case is the exact counterpart of an English one, so the English-vs-Persian gap of a decision
model can be measured on identical decisions.

| File | What |
|---|---|
| `v2-fa.toml` | question, option descriptions and text all in Persian (48 tasks, 845 cases) |
| `v2-fa-state.toml` | Persian text with the original English question: an English schema over Persian user input |
| `excluded.json` | the one task left out: `grammar_issue` asks about English spelling and grammar, which cannot survive translation |

Both files use the benchmark's own schema. Copy them into `sources/samples/` of the benchmark and run them with
`--sample v2-fa` or `--sample v2-fa-state`.

## How it was made

1. **Machine translation.** An LLM translated each task in one call (`translate.py`), with these rules:
   - Persian as people in Iran actually write it;
   - every fact, number, amount and hedge kept;
   - code, URLs, identifiers and product names untouched;
   - Western digits.
2. **Second-model check** (`crosscheck.py`). A different model family answered every case in English and in Persian.
   - Cases answered right in English but wrong in Persian, and cases whose numbers changed, were flagged.
   - On the final set, that model scores 0.948 in English and 0.941 in Persian: very little meaning was lost.
3. **Review of all 845 cases against the English** (`fixes.json`, 105 rewrites, applied by `translate.py`):
   - **register:** formal vs colloquial Persian, which the formality task depends on;
   - **terms that misled:** "it's fine" is «معمولی», not «خوب است»; "lamb chops" is «شیشلیک», not «کتلت»; "naan"
     stays recognisably Indian;
   - **Iranian usage:** «پیکور», «کناف», «نبش», «قوطی».
   - Two house rules apply everywhere: amounts are written «85 دلار» (a `$` breaks right-to-left text), and door is «در»,
     never «درب».
   - Facts that decide answers stay as in the English even where unusual in Iran: dollars, miles, Friday meetings,
     recipes with pork or alcohol.

The review was done case by case against the English source, but not by a panel of native annotators. Report results
on this set as "machine-translated, reviewed".

## Results (macro accuracy, int8 on CPU, the benchmark's harness)

| | English v2 | `v2-fa` | `v2-fa-state` |
|---|---:|---:|---:|
| decima-base | 0.673 | 0.616 | 0.639 |
| decima-agent | 0.675 | 0.640 | 0.624 |

License: CC0, like the original benchmark.
