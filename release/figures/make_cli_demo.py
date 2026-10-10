"""Render the README's terminal demo GIF from real `decima` CLI output (release/figures/decima-cli.gif).

    uv run --with pillow python release/figures/make_cli_demo.py OUTPUTS.txt release/figures/decima-cli.gif

OUTPUTS.txt holds `$ command` lines each followed by the command's real output, exactly as captured; nothing is edited.
"""

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, PAD, FS, LH = 1180, 28, 22, 31
BG, FG, DIM, PROMPT, WIN, BAR = (17, 19, 24), (226, 230, 236), (128, 136, 150), (98, 214, 145), (98, 214, 145), (150, 158, 172)


def font(bold=False):
    for p in (["DejaVuSansMono-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"] if bold else
              ["DejaVuSansMono.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"]):
        try:
            return ImageFont.truetype(p, FS)
        except OSError:
            pass
    return ImageFont.load_default()


F, FB = font(), font(True)


def scenes(path):
    out, cur = [], None
    for line in Path(path).read_text().splitlines():
        if line.startswith("$ "):
            cur = [line[2:], []]
            out.append(cur)
        elif cur is not None and line.strip():
            cur[1].append(line)
    return out


def wrap(text, width):
    """Split a command at spaces so each piece fits `width` pixels."""
    out, cur = [], ""
    for w in text.split(" "):
        t = (cur + " " + w).strip()
        if cur and F.getlength(t) > width:
            out.append(cur); cur = w
        else:
            cur = t
    return out + [cur]


def draw(lines, cursor=False):
    """lines: list of (kind, text) with kind in prompt / header / row."""
    h = PAD * 2 + 44 + LH * 10
    im = Image.new("RGB", (W, h), BG)
    d = ImageDraw.Draw(im)
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):          # window buttons
        d.ellipse((PAD + i * 26, 16, PAD + i * 26 + 14, 30), fill=c)
    d.text((W // 2 - 60, 12), "decima — CPU", font=F, fill=DIM)
    y = PAD + 30
    for kind, text in lines:
        if kind == "prompt":
            d.text((PAD, y), "$ ", font=FB, fill=PROMPT)
            parts = wrap(text, W - 2 * PAD - 60)
            for k, part in enumerate(parts):
                tail = "▌" if cursor and k == len(parts) - 1 else ""
                d.text((PAD + 30 + (24 if k else 0), y), part + tail, font=F, fill=FG)
                y += LH if k < len(parts) - 1 else 0
        elif kind == "header":
            d.text((PAD, y), text, font=F, fill=DIM)
        else:
            top = "█" in text and any(t in text for t in ("0.9", "1.00")) and text.split()[-1 if "p(yes)" not in text else -2].startswith(("0.9", "1.0"))
            d.text((PAD, y), text, font=F, fill=WIN if top else (FG if "█" in text else BAR))
        y += LH
    return im


def main():
    src, dst = sys.argv[1], sys.argv[2]
    frames, durs = [], []
    for cmd, output in scenes(src):
        typed = ""
        for k in range(0, len(cmd) + 1, 3):                                         # typing, 3 characters a frame
            typed = cmd[:k]
            frames.append(draw([("prompt", typed)], cursor=True)); durs.append(45)
        frames.append(draw([("prompt", cmd)])); durs.append(350)                    # "running"
        shown = [("prompt", cmd)]
        for j, line in enumerate(output[:8]):
            shown.append(("header" if j == 0 else "row", line))
            frames.append(draw(shown)); durs.append(70)
        durs[-1] = 2600                                                            # read the answer
    frames[0].save(dst, save_all=True, append_images=frames[1:], duration=durs, loop=0, optimize=True)
    print(f"{len(frames)} frames, {sum(durs) / 1000:.1f} s → {dst} ({Path(dst).stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
