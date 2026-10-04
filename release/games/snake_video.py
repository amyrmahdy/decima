"""Record Decima playing Nokia-style Snake: one decision per move, rendered frame by frame into an MP4/GIF.

    uv run --with pillow --with imageio-ffmpeg python release/games/snake_video.py --model export/v2-int8 \
        --seed 3 --steps 260 --out snake.mp4

Left: the game (Nokia 3310 palette). Right: the text the game sends Decima this move and Decima's
probabilities over the four moves. Every move is one forward pass of the shipped weights; the game's
wording is its own (release/games/snake.py), not a training template.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
import snake as S  # noqa: E402

BG, INK, DIM, PANEL, ACCENT = (199, 240, 216), (67, 82, 61), (150, 176, 152), (24, 26, 30), (127, 226, 160)
W, H, CELL = 1280, 720, 36


def font(size, mono=False):
    for p in (["DejaVuSansMono.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"] if mono else
              ["DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


F_T, F_S, F_M, F_B = font(30), font(20), font(19, True), font(46)


LABEL = ["decima-base"]


def frame(snake, apple, text, probs, move, score, step, ms) -> Image.Image:
    im = Image.new("RGB", (W, H), PANEL)
    d = ImageDraw.Draw(im)
    gx, gy = 40, (H - S.N * CELL) // 2
    d.rounded_rectangle((gx - 14, gy - 14, gx + S.N * CELL + 14, gy + S.N * CELL + 14), 18, fill=BG)
    for x in range(S.N):
        for y in range(S.N):
            d.rectangle((gx + x * CELL + 15, gy + y * CELL + 15, gx + x * CELL + 20, gy + y * CELL + 20), fill=(186, 226, 202))
    ax, ay = apple
    ox, oy = gx + ax * CELL, gy + ay * CELL                      # pixel apple: body, stem, leaf
    for (cx, cy, w, h) in ((10, 12, 16, 16), (8, 14, 20, 12), (12, 10, 12, 20)):
        d.rectangle((ox + cx, oy + cy, ox + cx + w, oy + cy + h), fill=INK)
    d.rectangle((ox + 17, oy + 5, ox + 19, oy + 11), fill=INK)
    d.rectangle((ox + 20, oy + 6, ox + 25, oy + 9), fill=INK)
    for i, (x, y) in enumerate(snake):
        pad = 3 if i else 1
        d.rectangle((gx + x * CELL + pad, gy + y * CELL + pad, gx + x * CELL + CELL - pad, gy + y * CELL + CELL - pad), fill=INK)
    hx, hy = snake[0]                                            # eyes on the head, looking the way it moves
    dx, dy = S.DIRS[move] if move else (1, 0)
    cx, cy = gx + hx * CELL + CELL // 2, gy + hy * CELL + CELL // 2
    px_, py_ = -dy, dx
    for side in (-1, 1):
        ex, ey = cx + dx * 6 + px_ * 7 * side, cy + dy * 6 + py_ * 7 * side
        d.rectangle((ex - 3, ey - 3, ex + 3, ey + 3), fill=BG)
    px = gx + S.N * CELL + 60
    d.text((px, 36), "Decima plays Snake", font=F_T, fill=(236, 234, 230))
    d.text((px, 76), f"one decision per move · {LABEL[0]}", font=F_S, fill=(150, 156, 162))
    d.text((px, 128), f"apples {score}", font=F_B, fill=ACCENT)
    d.text((px + 280, 140), f"move {step}   {ms:.0f} ms", font=F_S, fill=(150, 156, 162))
    d.text((px, 214), "what the game tells Decima:", font=F_S, fill=(150, 156, 162))
    y = 244
    for line in text.split("\n"):
        d.text((px, y), line, font=F_M, fill=(220, 220, 214))
        y += 28
    d.text((px, y + 22), "Decima's answer:", font=F_S, fill=(150, 156, 162))
    y += 56
    for k, p in probs.items():
        on = k == move
        d.text((px, y), f"{k:>5}", font=F_M, fill=ACCENT if on else (190, 190, 186))
        d.rectangle((px + 80, y + 4, px + 80 + 360, y + 22), fill=(44, 48, 54))
        d.rectangle((px + 80, y + 4, px + 80 + int(360 * p), y + 22), fill=ACCENT if on else (100, 106, 114))
        d.text((px + 452, y), f"{p:.2f}", font=F_M, fill=(220, 220, 214))
        y += 34
    return im


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=260)
    ap.add_argument("--fps", type=int, default=18)
    ap.add_argument("--out", default="snake.mp4")
    ap.add_argument("--label", default="decima-base · CPU", help="model line shown under the title")
    a = ap.parse_args()
    from decima import Decima
    import imageio_ffmpeg
    model = Decima.from_pretrained(a.model)
    LABEL[0] = a.label
    frames = []
    S_STEP = [0]

    def on_step(snake, apple, text, probs, move, score):
        S_STEP[0] += 1
        frames.append((snake, apple, text, probs, move, score, S_STEP[0]))

    result = S.play(model, a.seed, max_steps=a.steps, on_step=on_step)
    ms = result[3]
    tmp = Path(a.out).with_suffix(".frames")
    tmp.mkdir(exist_ok=True)
    for i, f in enumerate(frames):
        frame(*f, ms).save(tmp / f"{i:05d}.png")
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ff, "-y", "-framerate", str(a.fps), "-i", str(tmp / "%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "18", "-movflags", "+faststart", a.out], check=True, capture_output=True)
    gif = Path(a.out).with_suffix(".gif")
    subprocess.run([ff, "-y", "-framerate", str(a.fps), "-i", str(tmp / "%05d.png"), "-vf",
                    "scale=800:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=64[p];[s1][p]paletteuse=dither=bayer",
                    str(gif)], check=True, capture_output=True)
    print(f"apples {result[0]}, moves {result[1]}, {'alive' if result[2] else 'crashed'}, p50 {ms:.1f} ms → {a.out}, {gif}")


if __name__ == "__main__":
    main()
