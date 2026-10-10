---
pretty_name: Decima Knowledge-Graph Judgments
license: cc-by-4.0
language:
- en
task_categories:
- text-classification
- zero-shot-classification
size_categories:
- 100K<n<1M
annotations_creators:
- machine-generated
language_creators:
- machine-generated
tags:
- knowledge-graph
- entity-resolution
- entity-matching
- relation-extraction
- assertion-detection
- modality
- graphrag
- typed-decisions
- system-one
- synthetic
configs:
- config_name: entity_matching
  default: true
  data_files:
  - {split: train, path: entity_matching/train.parquet}
  - {split: test, path: entity_matching/test.parquet}
- config_name: mention_pairs
  data_files:
  - {split: train, path: mention_pairs/train.parquet}
  - {split: test, path: mention_pairs/test.parquet}
- config_name: modality
  data_files:
  - {split: train, path: modality/train.parquet}
  - {split: test, path: modality/test.parquet}
- config_name: relation_pairs
  data_files:
  - {split: train, path: relation_pairs/train.parquet}
  - {split: test, path: relation_pairs/test.parquet}
- config_name: documents
  data_files:
  - {split: train, path: documents/train.parquet}
  - {split: test, path: documents/test.parquet}
- config_name: support_threads
  data_files:
  - {split: train, path: support_threads/train.parquet}
  - {split: test, path: support_threads/test.parquet}
---

# Decima Knowledge-Graph Judgments

**About 200k judgments a knowledge-graph pipeline makes after extraction**, with exact or two-model-agreed labels: are
these two records the same entity? Is this extracted fact asserted, hypothetical, denied or only planned? Does the
document discuss this relation at all, and in which direction? What does this mention refer to? Are these two things
alternatives?

Today these checks are LLM calls, one per edge or candidate pair. This dataset trains a small local model to make them:
[decima-agent](https://huggingface.co/amyrmahdy/decima-agent) 2.2 was trained on it and, as a drop-in for Jev in
[William Lyon's public knowledge-graph notebooks](https://github.com/johnymontana/extraction-knowledge-graph-experiments),
catches 11 of 11 planted hedged facts (Jev: 8) and matches the Beer entity pairs at AUC 0.972 (Jev: 0.992). None of the
notebooks' documents or pairs are in this dataset.

Built by **A. M. Madani** ([@amyrmahdy](https://github.com/amyrmahdy)). Generators:
[`teacher/kg_proc.py`](https://github.com/amyrmahdy/decima/blob/main/teacher/kg_proc.py) and
[`teacher/kg_llm.py`](https://github.com/amyrmahdy/decima/blob/main/teacher/kg_llm.py).

## Configs

| config | train | test | state → decision |
|---|---:|---:|---|
| `entity_matching` | 50,000 | 1,638 | two records (products, companies, people, papers, places…) → same / related variant / different, plus per-field checks |
| `mention_pairs` | 24,859 | 814 | two mentions with their sentences → same entity? abbreviation or ticker? (with and without an ontology rule) |
| `modality` | 25,038 | 806 | a sentence → asserted / hypothetical / negated / forward-looking; can it be recorded as a fact? |
| `relation_pairs` | 33,392 | 742 | two things → alternatives / complementary / same / unrelated |
| `documents` | 59,896 | 3,581 | a 150–450-word business article → a fact's status, "does it discuss this?", which relation in which direction (or none), a mention's type |
| `support_threads` | 3,825 | 160 | a 10–16-turn support thread → intent, priority, resolved |

Columns: `id`, `config`, `task`, `origin` (`procedural` or `llm+blind-check`), `kind` (`choose`, `verify`, `score`),
`state`, `question`, `choices`, `gold`, `probs` (the soft target the model trained on). Questions are written the way
TypeSafe `choice` / `noul` / `score` requests render, including the wording of TypeSafe's entity-alignment cookbook.

```python
from datasets import load_dataset
ds = load_dataset("amyrmahdy/decima-kg-judgments", "modality", split="test")
```

## How the labels were made

- **`procedural`: exact, from code.** Records and mentions are generated with controlled differences (case,
  punctuation, abbreviations, legal suffixes, typos, missing fields, tickers, versions, sizes, vintages); modality
  sentences from templates with controlled cue words; relation pairs from hand-written role lists, including hard
  negatives such as two tools with different roles.
- **`llm+blind-check`.** Code invents a small fictional world and the facts with their modality; Gemma 4 (26B-A4B, run
  locally) writes the article or thread around them; a separate call that sees only the text and the question labels
  it blind, and a row is kept only when that answer matches the code's label (status 96 %, type 97 % agreement).

All companies, people and products are invented. **Test splits** hold out record domains, industries, cue words,
question phrasings, article styles, support channels and one ontology, so they measure transfer, not memory.

## Limitations

- Synthetic text: cleaner and more formulaic than real news, filings or support logs.
- `llm+blind-check` labels are only as good as two calls of the same model agreeing.
- Some labelling choices are conventions (a product's variant is "related", not "same"; a ticker without an ontology
  rule is the issuer). They are documented in the generators.

## License

CC BY 4.0. Gemma 4 is Apache-2.0.
