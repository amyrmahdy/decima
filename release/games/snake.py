"""Decima plays Snake: every move is one decision of the shipped Decima-small weights.

The board becomes a short text "sensor reading" (where the apple is, how far each direction is free);
Decima answers one choice question over four option texts. No game-specific training, no RL, no search.

    uv run python release/games/snake.py --model export/v1k-int8 --games 5            # score only
    uv run python release/games/snake.py --model export/v1k-int8 --record out.mp4     # frames → video
"""

from __future__ import annotations

import argparse
import random
import time

from decima import Decima, Question

N = 14
DIRS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
QUESTION = "Which way should the snake move next to reach the apple without crashing?"
OPTIONS = {"up": "move up", "down": "move down", "left": "move left", "right": "move right"}


ROOM = [True]       # report closed pockets (2026-10-04); False = the original v1/v2 sensor text


def free_run(snake: list, d: tuple) -> int:
    body = set(snake[:-1])           # the tail cell moves away this step
    x, y = snake[0]
    n = 0
    while True:
        x, y = x + d[0], y + d[1]
        if not (0 <= x < N and 0 <= y < N) or (x, y) in body:
            return n
        n += 1


def blocker(snake: list, d: tuple) -> str:
    body = set(snake[:-1])
    x, y = snake[0]
    while True:
        x, y = x + d[0], y + d[1]
        if not (0 <= x < N and 0 <= y < N):
            return "the wall"
        if (x, y) in body:
            return "your own body"


def after_move(snake: list, apple: tuple, d: tuple) -> list | None:
    """The body right after moving in direction d, or None if the move crashes."""
    head = (snake[0][0] + d[0], snake[0][1] + d[1])
    if not (0 <= head[0] < N and 0 <= head[1] < N) or head in set(snake[:-1]):
        return None
    return [head] + (snake if head == apple else snake[:-1])


def room(body: list) -> tuple[int, bool]:
    """Free cells reachable from the head of `body`, and whether the tail is among the reachable cells (then the snake can
    always follow its own tail out). This is what a player sees on the board and the old sensor text left out."""
    occupied = set(body[:-1])
    tail = body[-1]
    seen, stack, n, tail_ok = {body[0]}, [body[0]], 0, False
    while stack:
        x, y = stack.pop()
        for dx, dy in DIRS.values():
            c = (x + dx, y + dy)
            if c in seen or not (0 <= c[0] < N and 0 <= c[1] < N):
                continue
            if c == tail:
                tail_ok = True
            if c in occupied:
                continue
            seen.add(c); stack.append(c); n += 1
    return n, tail_ok


def pocket(snake: list, apple: tuple, d: tuple) -> int | None:
    """Size of the closed pocket the move leads into, if it is too small to survive in; None when the move is open."""
    body = after_move(snake, apple, d)
    if body is None:
        return None
    n, tail_ok = room(body)
    return n if (n < len(body) and not tail_ok) else None


def oracle(snake: list, apple: tuple) -> str:
    """Flood-fill player: never enter a closed pocket smaller than the body, otherwise head for the apple."""
    best, bs = None, -1e9
    for name, d in DIRS.items():
        body = after_move(snake, apple, d)
        if body is None:
            continue
        n, tail_ok = room(body)
        safe = n >= len(body) or tail_ok
        dist = abs(body[0][0] - apple[0]) + abs(body[0][1] - apple[1])
        sc = (1000 if safe else n) - dist
        if sc > bs:
            best, bs = name, sc
    return best or "up"


def sense(snake: list, apple: tuple) -> str:
    hx, hy = snake[0]
    ax, ay = apple
    parts = []
    if ax != hx:
        parts.append(f"{abs(ax - hx)} cells to the {'right' if ax > hx else 'left'}")
    if ay != hy:
        parts.append(f"{abs(ay - hy)} cells {'down' if ay > hy else 'up'}")
    lines = [f"The apple is {' and '.join(parts)}."]
    for name, d in DIRS.items():
        k = free_run(snake, d)
        if k == 0:
            lines.append(f"Moving {name} crashes into {blocker(snake, d)} immediately.")
        else:
            line = f"Moving {name}: {k} free cell{'s' if k > 1 else ''}, then {blocker(snake, d)}."
            pk = pocket(snake, apple, d) if ROOM[0] else None
            if pk is not None:
                line = line[:-1] + f"; it leads into a closed pocket of {pk} cells, smaller than your {len(snake)}-cell body."
            lines.append(line)
    return "\n".join(lines)


def step_game(snake, apple, move, rng):
    d = DIRS[move]
    head = (snake[0][0] + d[0], snake[0][1] + d[1])
    if not (0 <= head[0] < N and 0 <= head[1] < N) or head in set(snake[:-1]):
        return snake, apple, False, False
    ate = head == apple
    snake = [head] + (snake if ate else snake[:-1])
    if ate:
        free = [(x, y) for x in range(N) for y in range(N) if (x, y) not in set(snake)]
        apple = rng.choice(free)
    return snake, apple, True, ate


def play(model, seed: int, max_steps: int = 600, on_step=None):
    rng = random.Random(seed)
    snake = [(N // 2, N // 2), (N // 2 - 1, N // 2), (N // 2 - 2, N // 2)]
    apple = (rng.randrange(N), rng.randrange(N))
    while apple in snake:
        apple = (rng.randrange(N), rng.randrange(N))
    q = Question(QUESTION, list(OPTIONS.values()))
    score, since_apple, lat = 0, 0, []
    for t in range(max_steps):
        text = sense(snake, apple)
        t0 = time.perf_counter()
        dec = model.decide(text, q)
        lat.append((time.perf_counter() - t0) * 1e3)
        probs = dict(zip(OPTIONS, dec.probs))
        move = max(probs, key=probs.get)
        if on_step:
            on_step(snake, apple, text, probs, move, score)
        snake, apple, alive, ate = step_game(snake, apple, move, rng)
        score += ate
        since_apple = 0 if ate else since_apple + 1
        if not alive or since_apple > 4 * N * N:
            break
    return score, t + 1, alive, sorted(lat)[len(lat) // 2]


def play_oracle(seed: int, max_steps: int = 2000):
    rng = random.Random(seed)
    snake = [(N // 2, N // 2), (N // 2 - 1, N // 2), (N // 2 - 2, N // 2)]
    apple = (rng.randrange(N), rng.randrange(N))
    while apple in snake:
        apple = (rng.randrange(N), rng.randrange(N))
    score, since = 0, 0
    for t in range(max_steps):
        snake, apple, alive, ate = step_game(snake, apple, oracle(snake, apple), rng)
        score += ate; since = 0 if ate else since + 1
        if not alive or since > 4 * N * N:
            break
    return score, t + 1, alive


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="amyrmahdy/decima-small")
    ap.add_argument("--games", type=int, default=5)
    ap.add_argument("--no-room", action="store_true", help="the original sensor text without closed-pocket warnings")
    ap.add_argument("--oracle", action="store_true", help="play the flood-fill oracle instead of a model (ceiling)")
    a = ap.parse_args()
    ROOM[0] = not a.no_room
    if a.oracle:
        for g in range(a.games):
            s, steps, alive = play_oracle(g)
            print(f"game {g}: apples {s:3d}  steps {steps:4d}  {'still alive' if alive else 'crashed'}  oracle")
        return
    m = Decima.from_pretrained(a.model)
    for g in range(a.games):
        s, steps, alive, p50 = play(m, g)
        print(f"game {g}: apples {s:3d}  steps {steps:4d}  {'still alive' if alive else 'crashed'}  p50 {p50:.1f} ms/move")


if __name__ == "__main__":
    main()
