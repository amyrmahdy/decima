"""Procedural decisions with exact labels (no teacher): spatial navigation and side-scroller scenes.

    uv run python -m teacher.procedural --n-grid 150000 --n-side 100000 --out data/proc

Why: reading a text sensor report and picking the action it implies (the goal is up AND up is free → go up;
a hole is two tiles ahead → jump) is a general skill that Decima 1.1 lacks (it crashes a Snake game on
move one and jumps on clear ground). Teacher data covers it thinly; code can generate it exactly and
without limit. Two families, many phrasings each, three question kinds:

  grid  — an agent on a grid (robot, drone, rover, snake, player, cart, …) with a goal and per-direction
          free distances / blockers (walls, crates, its own body, other agents). choose: which way;
          verify: "moving X is safe" / "… gets closer" / "… is safe and gets closer"; score: how good is X.
  side  — a hero running right in a side-scroller: holes, enemies (approaching or not), low walls/pipes,
          coins overhead, the goal flag. choose: run / jump / jump straight up / wait; verify; score.

Targets are soft distributions from exact utilities (a crash keeps ~1 % mass, never more). The demo games
(release/games/) render their states with their own templates, which are NOT used here, so the demos test
the skill on unseen phrasings. Output rows use the standard training schema (source="proc").
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

DIRS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
COMPASS = {"up": "north", "down": "south", "left": "west", "right": "east"}
SCREEN = {"up": "up", "down": "down", "left": "left", "right": "right"}
AGENTS = [("the robot", "the charging dock"), ("the drone", "the landing pad"), ("the rover", "the sample site"),
          ("the snake", "the food"), ("the player", "the exit"), ("the cart", "the loading bay"), ("the mouse", "the cheese"),
          ("the courier", "the parcel"), ("the vacuum robot", "its base"), ("the knight", "the treasure"), ("you", "the goal"),
          ("the bot", "the target"), ("the worm", "the berry"), ("the forklift", "pallet 7")]
BLOCKERS = ["a wall", "a crate", "a rock", "its own body", "another robot", "a pillar", "a locked door", "the edge of the map",
            "a shelf", "water", "a fence", "its own tail"]


def _sid(*parts) -> str:
    return hashlib.sha1("\x1f".join(map(str, parts)).encode()).hexdigest()[:16]


def _soft(util: dict[str, float], temp: float = 0.45) -> dict[str, float]:
    m = max(util.values())
    e = {k: math.exp((v - m) / temp) for k, v in util.items()}
    z = sum(e.values())
    p = {k: max(v / z, 0.004) for k, v in e.items()}
    z = sum(p.values())
    return {k: round(v / z, 5) for k, v in p.items()}


# ───────────────────────────── grid family ─────────────────────────────

def grid_scene(rng: random.Random) -> dict:
    dx, dy = 0, 0
    while dx == 0 and dy == 0:
        dx, dy = rng.randint(-9, 9), rng.randint(-9, 9)
        if rng.random() < 0.25:                      # aligned goals are common in games
            (dx, dy) = (dx, 0) if rng.random() < 0.5 else (0, dy)
    free, block = {}, {}
    for d, (ux, uy) in DIRS.items():
        toward = (ux and dx * ux > 0) or (uy and dy * uy > 0)
        dist_goal = abs(dx) if ux else abs(dy)
        if rng.random() < 0.3:
            free[d] = 0
        else:
            free[d] = rng.randint(1, 10)
        if toward and ((ux and dy == 0) or (uy and dx == 0)) and free[d] >= dist_goal:
            free[d] = max(free[d], dist_goal)       # an aligned goal can be reachable
        block[d] = rng.choice(BLOCKERS)
    if all(v == 0 for v in free.values()):
        free[rng.choice(list(DIRS))] = rng.randint(1, 5)
    return {"dx": dx, "dy": dy, "free": free, "block": block, "agent": rng.choice(AGENTS)}


def grid_util(sc: dict) -> dict[str, float]:
    u = {}
    for d, (ux, uy) in DIRS.items():
        if sc["free"][d] == 0:
            u[d] = -4.0
            continue
        closer = (ux and sc["dx"] * ux > 0) or (uy and sc["dy"] * uy > 0)
        away = (ux and sc["dx"] * ux < 0) or (uy and sc["dy"] * uy < 0)
        u[d] = (2.0 if closer else 0.0 if away else 1.0) + min(sc["free"][d], 4) * 0.08
    return u


def _rel(n: int, pos: str, neg: str, unit: str) -> str:
    return f"{abs(n)} {unit}{'s' if abs(n) != 1 else ''} {pos if n > 0 else neg}"


def grid_render(sc: dict, rng: random.Random, vocab: dict) -> str:
    agent, goal = sc["agent"]
    unit = rng.choice(["cell", "square", "tile", "step", "block"])
    parts = []
    if sc["dx"]:
        parts.append(_rel(sc["dx"], vocab["right"], vocab["left"], unit) if vocab is not COMPASS else _rel(sc["dx"], "east", "west", unit))
    if sc["dy"]:
        parts.append(_rel(sc["dy"], vocab["down"], vocab["up"], unit) if vocab is not COMPASS else _rel(sc["dy"], "south", "north", unit))
    style = rng.randrange(4)
    order = list(DIRS)
    rng.shuffle(order)
    blocked = rng.choice(["blocked by {b}", "{b} right there", "would hit {b}", "runs into {b} at once", "collides with {b} on the first step",
                          "no room: {b}", "{b} is in the way, 0 cells free"])
    opened = rng.choice(["clear for {n}", "{n} free cells", "open for {n} {u}s", "{n} {u}s of space", "free for {n}", "{n} empty {u}s"])
    if style == 0:
        lines = [f"Goal ({goal}): {', '.join(parts)}."]
        for d in order:
            f = sc["free"][d]
            lines.append(f"{vocab[d].capitalize()}: " + (blocked.format(b=sc["block"][d]) if f == 0 else opened.format(n=f, u=unit)))
        return "\n".join(lines)
    if style == 1:
        s = f"{goal[0].upper() + goal[1:]} is {' and '.join(parts)} from {agent}. "
        bits = []
        for d in order:
            f = sc["free"][d]
            where = {"up": "above", "down": "below"}.get(vocab[d], f"to the {vocab[d]}")
            bits.append(f"{where} " + (f"{sc['block'][d]} is directly adjacent" if f == 0 else f"the way is open for {f} {unit}{'s' if f > 1 else ''}"))
        return s + "; ".join(bits).capitalize() + "."
    if style == 2:
        return json.dumps({"agent": agent, "goal": goal, "goal_offset": {"dx": sc["dx"], "dy": sc["dy"], "x_right": True, "y_down": True},
                           "open_cells": {vocab[d]: sc["free"][d] for d in order},
                           "blocked_by": {vocab[d]: sc["block"][d] for d in order if sc["free"][d] == 0}})
    return f"target {goal}: {', '.join(parts)} | " + " | ".join(
        f"{vocab[d]} {sc['free'][d]}" + (f" ({sc['block'][d]})" if sc["free"][d] == 0 else "") for d in order)


def grid_rows(rng: random.Random, i: int, sc: dict | None = None, state: str | None = None, vocab: dict | None = None) -> list[dict]:
    sc = sc or grid_scene(rng)
    vocab = vocab or (COMPASS if rng.random() < 0.3 else SCREEN)
    state = state or grid_render(sc, rng, vocab)
    agent, goal = sc["agent"]
    util = grid_util(sc)
    rows = []
    k = rng.random()
    if k < 0.5:
        q = rng.choice([f"Which way should {agent} move next to reach {goal} without hitting anything?",
                        f"Next move for {agent}?", f"Where should {agent} go to get to {goal} safely?",
                        f"Choose {agent}'s next step toward {goal}."])
        style = rng.randrange(3)
        names = list(DIRS)
        opts = [vocab[d] if style == 0 else f"move {vocab[d]}" if style == 1 else f"go {vocab[d]}" for d in names]
        p = _soft(util)
        rows.append(_row("choose", state, q, opts, [p[d] for d in names], i))
    elif k < 0.85:
        d = rng.choice(list(DIRS))
        (ux, uy) = DIRS[d]
        safe = sc["free"][d] > 0
        closer = (ux and sc["dx"] * ux > 0) or (uy and sc["dy"] * uy > 0)
        kind = rng.randrange(3)
        stmt, truth = [(f"Moving {vocab[d]} is safe for {agent}.", safe),
                       (f"Moving {vocab[d]} brings {agent} closer to {goal}.", closer),
                       (f"Moving {vocab[d]} is safe and brings {agent} closer to {goal}.", safe and closer)][kind]
        p = 0.97 if truth else 0.03
        yn = ["yes", "no"] if rng.random() < 0.7 else ["true", "false"]
        rows.append(_row("verify", state, stmt, yn, [p, 1 - p], i))
    else:
        d = rng.choice(list(DIRS))
        u = util[d]
        lvl = 0 if u < 0 else 1 if u < 0.5 else 2 if u < 1.5 else 3
        levels = ["crashes", "safe but moves away from the goal", "safe, neither closer nor farther", "safe and closer to the goal"]
        p = [0.02] * 4
        p[lvl] = 0.94
        rows.append(_row("score", state, f"How good is moving {vocab[d]} for {agent} right now?", levels, p, i))
    return rows


# ───────────────────────────── side-scroller family ─────────────────────────────

ENEMIES = ["a spiky crawler", "a slime", "a beetle", "a walking mushroom", "a turtle", "a rolling barrel", "a bat flying low",
           "a crab", "a hopping frog", "a goblin"]
OBST = ["a pipe", "a brick wall", "a crate stack", "a stone block", "a tree stump", "a fence"]


def side_scene(rng: random.Random) -> dict:
    sc = {"hole": None, "enemy": None, "obst": None, "coin": None, "flag": None}
    for f in ("hole", "enemy", "obst"):
        if rng.random() < 0.38:
            sc[f] = rng.randint(1, 7)
    if rng.random() < 0.25:
        sc["coin"] = rng.choice(["overhead", "ahead_air", "ground"])
    if rng.random() < 0.12:
        sc["flag"] = rng.randint(1, 8)
    sc["hole_w"] = rng.randint(1, 3)
    sc["enemy_kind"] = rng.choice(ENEMIES)
    sc["enemy_dir"] = rng.choice(["toward", "away", "still"])
    sc["obst_kind"] = rng.choice(OBST)
    sc["obst_h"] = rng.randint(1, 3)
    return sc


def side_util(sc: dict) -> dict[str, float]:
    near = []
    if sc["hole"] is not None and sc["hole"] <= 2:
        near.append("hole")
    if sc["obst"] is not None and sc["obst"] <= 1:
        near.append("obst")
    if sc["enemy"] is not None and (sc["enemy"] <= 2 or (sc["enemy"] <= 3 and sc["enemy_dir"] == "toward")):
        near.append("enemy")
    if near:
        return {"run": -3.0, "jump": 2.0, "jump_up": -1.0, "wait": -0.5 if near == ["enemy"] and sc["enemy_dir"] == "away" else -2.0}
    if sc["coin"] == "overhead":
        return {"run": 0.5, "jump": 0.0, "jump_up": 2.0, "wait": -1.0}
    return {"run": 2.0, "jump": 0.2, "jump_up": -0.5, "wait": -1.0}


def side_render(sc: dict, rng: random.Random) -> str:
    unit = rng.choice(["tile", "block", "step"])
    near = rng.choice(["right in front of the hero", f"1 {unit} ahead", "directly ahead", f"on the very next {unit}", "immediately ahead",
                       f"one {unit} away", "just ahead"])
    far = rng.choice(["{n} {u}s ahead", "{n} {u}s away", "in {n} {u}s", "at a distance of {n} {u}s", "{n} {u}s off"])
    def at(n):
        return near if n == 1 else far.format(n=n, u=unit)
    facts = []
    if sc["hole"] is not None:
        facts.append(f"a gap {sc['hole_w']} {unit}{'s' if sc['hole_w'] > 1 else ''} wide opens {at(sc['hole'])}")
    if sc["enemy"] is not None:
        mv = {"toward": "coming toward the hero", "away": "walking away", "still": "standing still"}[sc["enemy_dir"]]
        facts.append(f"{sc['enemy_kind']} is {at(sc['enemy'])}, {mv}")
    if sc["obst"] is not None:
        facts.append(f"{sc['obst_kind']} {sc['obst_h']} {unit}{'s' if sc['obst_h'] > 1 else ''} tall stands {at(sc['obst'])}")
    if sc["coin"] == "overhead":
        facts.append("a coin floats directly above the hero")
    elif sc["coin"] == "ahead_air":
        facts.append(f"a coin floats in the air {rng.randint(2, 6)} {unit}s ahead")
    elif sc["coin"] == "ground":
        facts.append(f"a coin lies on the ground {rng.randint(2, 6)} {unit}s ahead")
    if sc["flag"] is not None:
        facts.append(f"the goal flag is {sc['flag']} {unit}s ahead")
    rng.shuffle(facts)
    style = rng.randrange(4)
    if style == 3 and facts:                         # terse HUD readout
        hud = []
        if sc["hole"] is not None:
            hud.append(f"HOLE w={sc['hole_w']} d={sc['hole']}")
        if sc["enemy"] is not None:
            hud.append(f"ENEMY {sc['enemy_kind'].split()[-1]} d={sc['enemy']} {sc['enemy_dir']}")
        if sc["obst"] is not None:
            hud.append(f"BLOCK {sc['obst_kind'].split()[-1]} h={sc['obst_h']} d={sc['obst']}")
        if sc["coin"]:
            hud.append({"overhead": "COIN above", "ahead_air": "COIN air ahead", "ground": "COIN ground ahead"}[sc["coin"]])
        if sc["flag"] is not None:
            hud.append(f"FLAG d={sc['flag']}")
        rng.shuffle(hud)
        return " | ".join(hud) + f"   (d = distance in {unit}s)"
    if not facts:
        return rng.choice(["Nothing but flat ground ahead.", "The ground ahead is flat and empty.", "Scanner: no hazards, no items ahead.",
                           '{"ahead": []}'])
    if style == 0:
        return ("Ahead: " + "; ".join(facts) + ".").replace("Ahead: a", "Ahead: A", 1)
    if style == 1:
        return " ".join(f[0].upper() + f[1:] + "." for f in facts)
    return json.dumps({"ahead": facts})


SIDE_OPTS = [
    {"run": "run right", "jump": "jump forward", "jump_up": "jump straight up", "wait": "stop and wait"},
    {"run": "keep running", "jump": "jump over it", "jump_up": "jump in place", "wait": "stand still"},
    {"run": "run: continue to the right along the ground", "jump": "jump: leap forward over what is ahead",
     "jump_up": "jump up: hop vertically on the spot", "wait": "wait: do not move this turn"},
]


def side_rows(rng: random.Random, i: int, sc: dict | None = None, state: str | None = None) -> list[dict]:
    sc = sc or side_scene(rng)
    state = state or side_render(sc, rng)
    util = side_util(sc)
    k = rng.random()
    if k < 0.55:
        names = list(util)
        o = rng.choice(SIDE_OPTS)
        q = rng.choice(["What should the hero do now?", "Choose the hero's next action.", "Next action for the player character?",
                        "How should the runner react to what is ahead?"])
        p = _soft(util)
        return [_row("choose", state, q, [o[n] for n in names], [p[n] for n in names], i)]
    if k < 0.85:
        best = max(util, key=util.get)
        stmt, truth = rng.choice([("The hero should jump forward now.", best == "jump"),
                                  ("It is safe to keep running.", best == "run"),
                                  ("Something just ahead must be jumped over.", best == "jump"),
                                  ("The hero should jump straight up now.", best == "jump_up")])
        p = 0.97 if truth else 0.03
        return [_row("verify", state, stmt, ["yes", "no"], [p, 1 - p], i)]
    near = min([v for v in (sc["hole"], sc["enemy"], sc["obst"]) if v is not None], default=9)
    lvl = 3 if util["jump"] == 2.0 and near <= 1 else 2 if util["jump"] == 2.0 else 1 if near <= 4 else 0
    p = [0.02] * 4
    p[lvl] = 0.94
    return [_row("score", state, "How urgent is it for the hero to jump?",
                 ["no need to jump", "a hazard is coming but not yet", "jump soon", "jump right now"], p, i)]


def _row(kind, state, question, choices, probs, i) -> dict:
    s = sum(probs)
    probs = [round(p / s, 5) for p in probs]
    return {"kind": kind, "state": state, "question": question, "choices": choices, "probs": probs,
            "gold": max(range(len(probs)), key=probs.__getitem__), "state_lang": "en", "choice_lang": "en",
            "id": _sid("proc", i, state, question, *choices), "source": "proc"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-grid", type=int, default=150000)
    ap.add_argument("--n-side", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/proc")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for fam, n, fn in (("grid", a.n_grid, grid_rows), ("side", a.n_side, side_rows)):
        rng = random.Random(f"{fam}:{a.seed}")
        with (out / f"proc-{fam}.jsonl").open("w") as f:
            for i in range(n):
                for r in fn(rng, i):
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(fam, n, "→", out / f"proc-{fam}.jsonl")


if __name__ == "__main__":
    main()
