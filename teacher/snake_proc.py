"""Snake states with flood-fill labels: never walk into a closed pocket smaller than your body.

    uv run python -m teacher.snake_proc --n 60000 --out data/proc/snake-room.jsonl

States come from real games (the flood-fill oracle in release/games/snake.py, with 15–30 % random safe moves so the
snake also gets into tight spots), on boards of 10–20 cells. Each state lists the apple offset and, per direction, the
free run, what blocks it, and the size of any closed pocket the move leads into versus the body length.

The wording is this generator's own (JSON, HUD lines, terse notes, narration, compass words); the game's sentences in
release/games/snake.py are never used, so the game stays a held-out test. Labels come from the oracle's own scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "release" / "games"))
import snake as G  # noqa: E402

NAMES = {"up": ["up", "north", "upward"], "down": ["down", "south", "downward"], "left": ["left", "west", "leftward"], "right": ["right", "east", "rightward"]}
WALL = ["the wall", "the edge", "the border", "the boundary"]
BODY = ["its own body", "the tail", "a body segment", "itself"]
AGENT = ["the snake", "the worm", "the player", "the serpent"]
FOOD = ["the apple", "the food", "the pellet", "the fruit"]


def sid(*p):
    return hashlib.sha1("\x1f".join(map(str, p)).encode()).hexdigest()[:16]


def features(snake, apple):
    hx, hy = snake[0]
    f = {"dx": apple[0] - hx, "dy": apple[1] - hy, "len": len(snake), "moves": {}}
    for name, d in G.DIRS.items():
        k = G.free_run(snake, d)
        body = G.after_move(snake, apple, d)
        pk = G.pocket(snake, apple, d)
        f["moves"][name] = {"run": k, "blocker": "wall" if G.blocker(snake, d) == "the wall" else "body", "crash": body is None, "pocket": pk}
    return f


def utility(snake, apple):
    u = {}
    for name, d in G.DIRS.items():
        body = G.after_move(snake, apple, d)
        if body is None:
            u[name] = -1e9; continue
        n, tail_ok = G.room(body)
        safe = n >= len(body) or tail_ok
        dist = abs(body[0][0] - apple[0]) + abs(body[0][1] - apple[1])
        u[name] = (1000 if safe else n) - dist
    return u


def soft(u):
    best = max(u.values())
    ex = {k: (math.exp((v - best) / 2.0) if v > -1e8 else 0.0) for k, v in u.items()}
    s = sum(ex.values()) or 1.0                        # every move crashes: any answer is as good
    p = {k: 0.01 + 0.96 * v / s for k, v in ex.items()}
    z = sum(p.values())
    return {k: v / z for k, v in p.items()}


def render(f, rng):
    vocab = {k: rng.choice(v[:2]) for k, v in NAMES.items()} if rng.random() < 0.85 else {k: v[2] for k, v in NAMES.items()}
    food, agent = rng.choice(FOOD), rng.choice(AGENT)
    style = rng.randrange(5)
    dx, dy = f["dx"], f["dy"]
    hor = f"{abs(dx)} {vocab['right'] if dx > 0 else vocab['left']}" if dx else ""
    ver = f"{abs(dy)} {vocab['down'] if dy > 0 else vocab['up']}" if dy else ""
    off = ", ".join(x for x in (hor, ver) if x) or "here"
    if style == 0:
        st = {"length": f["len"], "food_offset": {"x": dx, "y": dy}, "moves": {}}
        for name, m in f["moves"].items():
            st["moves"][vocab[name]] = {"open_cells": m["run"], "stopped_by": m["blocker"], **({"enclosed_area": m["pocket"]} if m["pocket"] is not None else {})}
        return json.dumps(st), vocab, agent, food
    lines = []
    if style == 1:
        lines.append(f"LEN {f['len']} | FOOD {off}")
        for name, m in f["moves"].items():
            t = f"{vocab[name].upper():6s} open={m['run']} stop={m['blocker']}"
            if m["pocket"] is not None:
                t += f" trap={m['pocket']}/{f['len']}"
            lines.append(t)
    elif style == 2:
        lines.append(f"{agent.capitalize()} is {f['len']} long; {food} is {off}.")
        for name, m in f["moves"].items():
            if m["run"] == 0:
                lines.append(f"{vocab[name].capitalize()}: blocked right away by {rng.choice(WALL) if m['blocker'] == 'wall' else rng.choice(BODY)}.")
            else:
                t = f"{vocab[name].capitalize()}: {m['run']} open, then {rng.choice(WALL) if m['blocker'] == 'wall' else rng.choice(BODY)}"
                if m["pocket"] is not None:
                    t += f"; dead end, only {m['pocket']} squares enclosed there"
                lines.append(t + ".")
    elif style == 3:
        lines.append(f"Target {off}. Body length {f['len']}.")
        for name, m in f["moves"].items():
            t = f"- {vocab[name]}: " + ("instant collision" if m["run"] == 0 else f"{m['run']} clear")
            if m["pocket"] is not None:
                t += f" (enclosed region of {m['pocket']} < length {f['len']})"
            lines.append(t)
    else:
        lines.append(f"There is {food} {off} of the head, and {agent} is {f['len']} segments long.")
        for name, m in f["moves"].items():
            if m["run"] == 0:
                lines.append(f"Going {vocab[name]} would hit {rng.choice(WALL) if m['blocker'] == 'wall' else rng.choice(BODY)} at once.")
            elif m["pocket"] is not None:
                lines.append(f"Going {vocab[name]} has {m['run']} free squares but ends in a sealed-off area of {m['pocket']} squares, too small for {agent}.")
            else:
                lines.append(f"Going {vocab[name]} has {m['run']} free squares before {rng.choice(WALL) if m['blocker'] == 'wall' else rng.choice(BODY)}.")
    if rng.random() < 0.3:
        body = lines[1:]; rng.shuffle(body); lines = lines[:1] + body
    return "\n".join(lines), vocab, agent, food


def conflict(snake, apple):
    """The apple pulls one way and that way is a closed pocket: the state decima-agent 2.1 still dies in (move 326 of
    its Snake GIF: "left leads into a closed pocket of 5 cells, smaller than your 31-cell body", and it went left)."""
    hx, hy = snake[0]
    for d in G.DIRS.values():
        closer = abs(hx + d[0] - apple[0]) + abs(hy + d[1] - apple[1]) < abs(hx - apple[0]) + abs(hy - apple[1])
        if closer and G.pocket(snake, apple, d) is not None:
            return True
    return False


def rows_for(snake, apple, rng, i):
    f = features(snake, apple)
    u = utility(snake, apple)
    p = soft(u)
    state, vocab, agent, food = render(f, rng)
    out = []
    if rng.random() < 0.75:
        q = rng.choice([f"Which way should {agent} go next?", f"Best next move for {agent} to reach {food} and survive?",
                        "Pick the next direction.", f"Where should {agent} head now without trapping itself?"])
        names = list(G.DIRS)
        rng.shuffle(names)
        style = rng.randrange(3)
        opts = [vocab[n] if style == 0 else f"go {vocab[n]}" if style == 1 else f"turn {vocab[n]}" for n in names]
        out.append({"kind": "choose", "question": q, "choices": opts, "probs": [round(p[n], 5) for n in names]})
    else:
        safe = [n for n in G.DIRS if u[n] >= 1000 - 100]
        trap = [n for n in G.DIRS if -1e8 < u[n] < 1000 - 100]
        pick = rng.choice(trap) if trap and rng.random() < 0.6 else rng.choice(list(G.DIRS))
        truth = pick in safe
        q = rng.choice([f"Moving {vocab[pick]} keeps {agent} alive for the long run.", f"Is going {vocab[pick]} safe, with enough room afterwards?"])
        out.append({"kind": "verify", "question": q, "choices": ["yes", "no"], "probs": [0.95, 0.05] if truth else [0.05, 0.95]})
    for r in out:
        r.update({"state": state, "gold": max(range(len(r["probs"])), key=r["probs"].__getitem__), "state_lang": "en", "choice_lang": "en",
                  "id": sid("snake-room", i, state, r["question"]), "source": "proc", "task": "snake-room"})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conflict", action="store_true", help="only states where the apple's direction is a closed pocket")
    a = ap.parse_args()
    rng = random.Random(f"snake-room:{a.seed}")
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    n, game = 0, 0
    with out.open("w") as fo:
        while n < a.n:
            G.N = rng.choice([10, 12, 14, 16, 20])
            eps = rng.choice([0.15, 0.2, 0.3])
            snake = [(G.N // 2, G.N // 2), (G.N // 2 - 1, G.N // 2), (G.N // 2 - 2, G.N // 2)]
            g = random.Random(f"g{a.seed}:{game}")
            apple = (g.randrange(G.N), g.randrange(G.N))
            while apple in snake:
                apple = (g.randrange(G.N), g.randrange(G.N))
            for t in range(3000):
                if (conflict(snake, apple) if a.conflict else
                        rng.random() < 0.35 or any(G.pocket(snake, apple, d) is not None for d in G.DIRS.values())):
                    for r in rows_for(snake, apple, rng, n):
                        fo.write(json.dumps(r) + "\n"); n += 1
                mv = G.oracle(snake, apple)
                if rng.random() < eps:
                    ok = [k for k, d in G.DIRS.items() if G.after_move(snake, apple, d) is not None]
                    mv = rng.choice(ok) if ok else mv
                snake, apple, alive, _ = G.step_game(snake, apple, mv, g)
                if not alive or n >= a.n:
                    break
            game += 1
    G.N = 14
    print(f"{n} rows from {game} games → {out}")


if __name__ == "__main__":
    main()
