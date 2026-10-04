---
pretty_name: Decima Game Decisions
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
- games
- snake
- platformer
- spatial-reasoning
- text-sensors
- procedural
- soft-labels
- system-one
- synthetic
configs:
- config_name: grid
  default: true
  data_files:
  - {split: train, path: grid/train.parquet}
  - {split: test, path: grid/test.parquet}
- config_name: side_scroller
  data_files:
  - {split: train, path: side_scroller/train.parquet}
  - {split: test, path: side_scroller/test.parquet}
- config_name: snake
  data_files:
  - {split: train, path: snake/train.parquet}
  - {split: test, path: snake/test.parquet}
- config_name: reworded
  data_files:
  - {split: train, path: reworded/train.parquet}
  - {split: test, path: reworded/test.parquet}
---

# Decima Game Decisions

**About 330k "read the sensors, pick the move" decisions with exact labels from code.** A text sensor report comes in,
one decision comes out: which way to go on a grid, whether to jump in a side-scroller, and how to avoid trapping
yourself in Snake.

It is how [decima-base](https://huggingface.co/amyrmahdy/decima-base), a 321M text classifier, learned to play:

- it wins 9 of 10 levels of a side-scrolling platformer;
- it averages 22.6 apples per game of Nokia-style Snake.

The games use their own wording, which never appears here.

Built by **A. M. Madani** ([@amyrmahdy](https://github.com/amyrmahdy)).
Games and code: [github.com/amyrmahdy/decima](https://github.com/amyrmahdy/decima/tree/main/release/games).

## Configs

| config | train | test | |
|---|---:|---:|---|
| `grid` | 145,475 | 4,525 | an agent on a grid (robot, drone, rover, snake, cart, …) with a goal and per-direction free distances and blockers. Which way? Is moving X safe? Does it get closer? How good is X? |
| `side_scroller` | 97,611 | 2,389 | a hero running right: holes, enemies, low walls, coins, the flag. Run, jump, jump straight up or wait? |
| `snake` | 58,231 | 1,769 | states from real Snake games on 10–20-cell boards: per direction, the free run, what blocks it, and the size of any closed pocket a move leads into, vs the body length |
| `reworded` | 22,048 | 684 | grid and side-scroller rows reworded by Gemma 4 into other settings (robot telemetry, drone logs, …), every number kept |

Columns: `id`, `kind` (`choose`, `verify`, `score`), `state`, `question`, `choices`, `gold`, `probs` (soft target),
`context` (`reworded` only). Test rows are 3 % held out by state hash.

```python
from datasets import load_dataset
ds = load_dataset("amyrmahdy/decima-game-decisions", "snake", split="test")
r = ds[0]; print(r["state"], "\n", r["question"], r["choices"], r["probs"])
```

## How the labels were made

- **Exact utilities, soft targets.** Every option is scored by code (closer to the goal, blocked, a crash), and the
  target is a distribution over those scores. A crash never keeps more than about 1 % of the mass.
- **Snake: a flood-fill oracle.** Moving into a closed pocket smaller than your body is a slow death even when the
  next cell is free. The oracle counts the room behind every move and whether the tail will free a way out. It
  averages about 60 apples a game, against 29 for one-step lookahead. States come from games it played, with 15–30 %
  random safe moves mixed in so the snake also lands in tight spots; 5,687 of the 60,000 states contain a pocket.
- **Many phrasings:** JSON, HUD lines, terse notes, narration and compass words, so a model learns the skill and not
  one template.

## Limitations

- **One step at a time.** Each row is a single decision; there is no planning over several moves.
- **Only the situations the generators know.** Real games have mechanics these states do not describe.
- On `reworded`, labels come from the original procedural row; a rewrite that kept every number can still, rarely,
  change the meaning.

## License

CC BY 4.0. Please cite:

```bibtex
@software{madani2026decima,
  author = {Madani, Amir Mahdi},
  title  = {Decima: open, calibrated decision models},
  year   = {2026},
  url    = {https://github.com/amyrmahdy/decima}
}
```
