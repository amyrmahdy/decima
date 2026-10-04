"""Render a Decima platformer run to MP4 + GIF: original pixel-style art, the sensor text and Decima's answer."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
import platformer as P  # noqa: E402

W, H, T = 1280, 720, 48
VIEW_W = 17                                   # tiles visible
SKY_TOP, SKY_BOT = (110, 178, 238), (182, 222, 250)
GRASS, DIRT, PIPE, PIPE_D = (92, 184, 76), (156, 102, 60), (54, 168, 92), (36, 120, 64)
COIN, ENEMY, HERO, PANEL = (250, 204, 64), (138, 76, 196), (247, 128, 44), (24, 26, 30)
ACCENT = (127, 226, 160)
LABEL = ["decima-base · CPU"]


def font(size, mono=False):
    name = "DejaVuSansMono.ttf" if mono else "DejaVuSans.ttf"
    for p in (name, f"/usr/share/fonts/truetype/dejavu/{name}"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


F_T, F_S, F_M, F_B = font(28), font(18), font(17, True), font(34)


def draw_world(d: ImageDraw.ImageDraw, lv, hero, cam: float, gw: int, gh: int) -> None:
    for y in range(gh):
        t = y / gh
        d.line((0, y, gw, y), fill=tuple(int(a + (b - a) * t) for a, b in zip(SKY_TOP, SKY_BOT)))
    for cx in (3, 9, 15, 21, 27, 33, 39, 45, 51, 57, 63):       # parallax clouds
        x = (cx * T - cam * T * 0.5) % (gw + 200) - 100
        d.ellipse((x, 60 + (cx % 3) * 30, x + 120, 100 + (cx % 3) * 30), fill=(244, 249, 255))
    gy = P.GROUND * T
    for tx in range(int(cam) - 1, int(cam) + VIEW_W + 2):
        x = (tx - cam) * T
        if tx in lv.gaps or tx < 0 or tx > lv.length:
            continue
        d.rectangle((x, gy, x + T, gh), fill=DIRT)
        d.rectangle((x, gy, x + T, gy + 10), fill=GRASS)
        if tx in lv.pipes:
            h = lv.pipes[tx] * T
            d.rectangle((x + 4, gy - h, x + T - 4, gy), fill=PIPE)
            d.rectangle((x, gy - h, x + T, gy - h + 14), fill=PIPE_D)
        if tx in lv.coins:
            cy = gy - 2 * T - 10
            d.ellipse((x + 12, cy, x + T - 12, cy + 26), fill=COIN, outline=(196, 150, 30), width=2)
    fx = (lv.flag - cam) * T
    d.rectangle((fx + 20, gy - 4 * T, fx + 26, gy), fill=(230, 230, 230))
    d.polygon([(fx + 26, gy - 4 * T), (fx + 66, gy - 4 * T + 16), (fx + 26, gy - 4 * T + 32)], fill=(236, 72, 80))
    for ex, _ in lv.enemies:
        x = (ex - cam) * T
        if -T < x < gw:
            d.ellipse((x + 6, gy - 34, x + T - 6, gy + 2), fill=ENEMY)
            d.ellipse((x + 14, gy - 26, x + 22, gy - 16), fill="white"); d.ellipse((x + 27, gy - 26, x + 35, gy - 16), fill="white")
            d.ellipse((x + 16, gy - 22, x + 20, gy - 18), fill="black"); d.ellipse((x + 29, gy - 22, x + 33, gy - 18), fill="black")
    hx, hy = (hero.x - cam) * T, gy - hero.h * T
    d.rounded_rectangle((hx + 8, hy - 40, hx + T - 8, hy), 10, fill=HERO)
    d.rectangle((hx + 8, hy - 18, hx + T - 8, hy - 13), fill="white")                       # scarf
    d.ellipse((hx + 24, hy - 33, hx + 32, hy - 25), fill="white"); d.ellipse((hx + 27, hy - 31, hx + 31, hy - 27), fill="black")


def render(lv, hero, text, probs, action, cam, info) -> Image.Image:
    im = Image.new("RGB", (W, H), PANEL)
    gw, gh = VIEW_W * T, 11 * T
    game = Image.new("RGB", (gw, gh))
    draw_world(ImageDraw.Draw(game), lv, hero, cam, gw, gh)
    im.paste(game, (24, (H - gh) // 2))
    d = ImageDraw.Draw(im)
    px = gw + 52
    d.text((px, 30), "Decima plays a platformer", font=F_T, fill=(236, 234, 230))
    d.text((px, 66), f"one decision per landing · {LABEL[0]}", font=F_S, fill=(150, 156, 162))
    d.text((px, 104), f"coins {hero.coins}   stomps {hero.stomps}", font=F_B, fill=ACCENT)
    d.text((px, 160), "what the game tells Decima:", font=F_S, fill=(150, 156, 162))
    y = 188
    for line in (text or info.get("last_text", "")).split("\n")[:6]:
        d.text((px, y), line, font=F_M, fill=(220, 220, 214)); y += 26
    d.text((px, 360), "Decima's answer:", font=F_S, fill=(150, 156, 162))
    y = 390
    pr = probs or info.get("last_probs") or {}
    act = action or info.get("last_action")
    for k, label in P.ACTIONS.items():
        p = pr.get(k, 0.0)
        on = k == act
        d.text((px, y), label, font=F_M, fill=ACCENT if on else (190, 190, 186))
        d.rectangle((px, y + 24, px + 300, y + 34), fill=(44, 48, 54))
        d.rectangle((px, y + 24, px + int(300 * p), y + 34), fill=ACCENT if on else (100, 106, 114))
        d.text((px + 312, y + 16), f"{p:.2f}", font=F_M, fill=(220, 220, 214))
        y += 52
    if not hero.alive or hero.won:
        d.text((px, 620), "LEVEL CLEAR!" if hero.won else "ouch", font=F_B, fill=ACCENT if hero.won else (236, 96, 90))
    return im


def record(model, seed: int, out: str, fps: int = 24, sub: int = 3) -> None:
    import imageio_ffmpeg
    tmp = Path(out).with_suffix(".frames"); tmp.mkdir(exist_ok=True)
    for f in tmp.glob("*.png"):
        f.unlink()
    info, n = {}, [0]
    prev = {"x": 2.0, "h": 0.0}

    def on_tick(lv, hero, text, probs, action):
        if text:
            info.update(last_text=text, last_probs=probs, last_action=action)
        x0, h0 = prev["x"], prev["h"]
        for k in range(sub):                                   # interpolate between ticks for smooth motion
            t = (k + 1) / sub
            vx, vh = x0 + (hero.x - x0) * t, h0 + (hero.h - h0) * t
            ghost = P.Hero(x=vx, h=vh, coins=hero.coins, stomps=hero.stomps, alive=hero.alive, won=hero.won)
            cam = max(0.0, vx - 4)
            render(lv, ghost, text, probs, action, cam, info).save(tmp / f"{n[0]:05d}.png"); n[0] += 1
        prev.update(x=hero.x, h=hero.h)

    hero, ticks, p50 = P.play(model, seed, on_tick=on_tick)
    last = sorted(tmp.glob("*.png"))[-1]
    for k in range(fps):                                       # hold the last frame for a second
        (tmp / f"{n[0] + k:05d}.png").write_bytes(last.read_bytes())
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ff, "-y", "-framerate", str(fps), "-i", str(tmp / "%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "18", "-movflags", "+faststart", out], check=True, capture_output=True)
    subprocess.run([ff, "-y", "-framerate", str(fps), "-i", str(tmp / "%05d.png"), "-vf",
                    "scale=800:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=96[p];[s1][p]paletteuse=dither=bayer",
                    str(Path(out).with_suffix(".gif"))], check=True, capture_output=True)
    print(f"{'won' if hero.won else 'died'} at x={hero.x}, coins {hero.coins}, stomps {hero.stomps}, {n[0]} frames, p50 {p50:.1f} ms → {out}")
