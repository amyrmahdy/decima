"""Decima plays a classic 2D platformer: each time the hero stands on the ground, the game describes what is
ahead in words and Decima picks run / jump forward / jump straight up / wait (one choice question).

    uv run python release/games/platformer.py --model export/v2-int8 --levels 10                 # score only
    uv run --with pillow --with imageio-ffmpeg python release/games/platformer.py --model export/v2-int8 \
        --record platformer.mp4 --seed 4

Original art and rules (no third-party assets). The sensor wording below is the game's own; it is not one
of the training templates in teacher/procedural.py.
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

GROUND = 9                     # ground row (rows 9–10 are ground); y grows downward
JUMP = [(1, 2.0), (1, 2.6), (1, 2.0), (1, 0)]       # forward jump: dx, height per tick; lands 4 tiles ahead
HOP = [(0, 1.6), (0, 2.2), (0, 1.2), (0, 0)]        # straight-up hop
ACTIONS = {"run": "run right", "jump": "jump forward", "jump_up": "jump straight up", "wait": "stop and wait"}
QUESTION = "What should the hero do next?"
ENEMY_NAMES = ["slime", "spiky crawler", "beetle"]


@dataclass
class Level:
    length: int
    gaps: set = field(default_factory=set)
    pipes: dict = field(default_factory=dict)        # x → height in tiles
    coins: set = field(default_factory=set)          # x of coins hovering 2 tiles above ground
    enemies: list = field(default_factory=list)      # [x(float), name]
    flag: int = 0


def make_level(seed: int, length: int = 64) -> Level:
    rng = random.Random(seed)
    lv = Level(length=length, flag=length - 2)
    x = 7
    while x < length - 6:
        kind = rng.choice(["gap", "pipe", "enemy", "coin", "flat", "gap", "pipe", "enemy"])
        if kind == "gap":
            w = rng.randint(1, 2)
            lv.gaps.update(range(x, x + w)); x += w + rng.randint(3, 5)
        elif kind == "pipe":
            lv.pipes[x] = rng.randint(1, 2); x += 1 + rng.randint(3, 5)
        elif kind == "enemy":
            lv.enemies.append([float(x + 3), rng.choice(ENEMY_NAMES)]); x += rng.randint(4, 6)
        elif kind == "coin":
            lv.coins.add(x); x += rng.randint(3, 4)
        else:
            x += rng.randint(2, 4)
    return lv


def describe(lv: Level, hx: int) -> str:
    """The game's own sensor wording (not a training template)."""
    lines = []
    for k in range(1, 7):
        x = hx + k
        if x in lv.gaps and (x - 1) not in lv.gaps:
            w = 1 + ((x + 1) in lv.gaps)
            lines.append(f"Pit: {w} tile{'s' if w > 1 else ''} wide, {k} tile{'s' if k > 1 else ''} ahead.")
        if x in lv.pipes:
            lines.append(f"Pipe: {lv.pipes[x]} tile{'s' if lv.pipes[x] > 1 else ''} tall, {k} tile{'s' if k > 1 else ''} ahead.")
    for ex, name in lv.enemies:
        k = round(ex) - hx
        if 0 < k <= 6:
            lines.append(f"Enemy ({name}): {k} tile{'s' if k > 1 else ''} ahead, moving toward you.")
    if hx in lv.coins:
        lines.append("Coin: right above you.")
    if 0 < lv.flag - hx <= 6:
        lines.append(f"Flag: {lv.flag - hx} tiles ahead.")
    return "\n".join(lines) if lines else "Nothing ahead for 6 tiles."


@dataclass
class Hero:
    x: int = 2
    h: float = 0.0
    plan: list = field(default_factory=list)
    coins: int = 0
    stomps: int = 0
    alive: bool = True
    won: bool = False


def tick(lv: Level, hero: Hero, action: str | None) -> None:
    if action and not hero.plan:
        hero.plan = {"run": [(1, 0)], "jump": list(JUMP), "jump_up": list(HOP), "wait": [(0, 0)]}[action]
    dx, h = hero.plan.pop(0) if hero.plan else (0, 0)
    nx = hero.x + dx
    if dx and h <= 0 and nx in lv.pipes:
        nx = hero.x                                  # walked into a pipe: blocked
    if dx and 0 < h < (lv.pipes.get(nx, 0)):
        nx = hero.x
    landed = hero.h > 0 and h == 0
    hero.x, hero.h = nx, h
    for e in lv.enemies:
        e[0] -= 0.5
    if landed:                                       # coming down on an enemy defeats it
        hit = [e for e in lv.enemies if abs(e[0] - hero.x) < 1.0]
        hero.stomps += len(hit)
        lv.enemies = [e for e in lv.enemies if e not in hit]
    if hero.h > 1.5 and hero.x in lv.coins:
        lv.coins.discard(hero.x); hero.coins += 1
    if hero.h == 0 and not hero.plan:
        if hero.x in lv.gaps:
            hero.alive = False
        if any(abs(e[0] - hero.x) < 0.6 for e in lv.enemies):
            hero.alive = False
    if hero.x >= lv.flag:
        hero.won = True


def play(model, seed: int, max_ticks: int = 400, on_tick=None):
    from decima.systemone import system_one
    lv = make_level(seed)
    hero = Hero()
    q = {"a": {"type": "choice", "instructions": QUESTION, "criteria": ACTIONS}}
    lat = []
    for t in range(max_ticks):
        action, text, probs = None, None, None
        if not hero.plan:
            text = describe(lv, hero.x)
            t0 = time.perf_counter()
            ans = system_one(model, text, q)["a"]
            lat.append((time.perf_counter() - t0) * 1e3)
            action, probs = ans["choice"], ans["probabilities"]
        if on_tick:
            on_tick(lv, hero, text, probs, action)
        tick(lv, hero, action)
        if not hero.alive or hero.won:
            break
    if on_tick:
        on_tick(lv, hero, None, None, None)
    return hero, t + 1, (sorted(lat)[len(lat) // 2] if lat else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="amyrmahdy/decima-small")
    ap.add_argument("--levels", type=int, default=10)
    ap.add_argument("--record", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from decima import Decima
    m = Decima.from_pretrained(a.model)
    if a.record:
        sys.path.insert(0, str(Path(__file__).parent))
        from platformer_video import record
        record(m, a.seed, a.record)
        return
    wins = 0
    for s in range(a.levels):
        hero, ticks, p50 = play(m, s)
        wins += hero.won
        print(f"level {s}: {'WON ' if hero.won else 'died'} at x={hero.x}/{make_level(s).length}  coins {hero.coins}  p50 {p50:.1f} ms")
    print(f"won {wins}/{a.levels}")


if __name__ == "__main__":
    main()
