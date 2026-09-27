"""Regenerate every Decima-small release figure from result files.

Run from the repo root:
    uv run --with matplotlib python release/figures/make_figures.py

Every plotted number is computed from these files (nothing is typed in):
    runs/v1i-{kev,laya,decima,jevtyped}.json   results[system][suite] -> acc, ece, flip, n
    runs/preds/laya--{v1i,laya}.jsonl          per-item probabilities (reliability diagram)
    runs/items/laya.jsonl                      gold labels / splits for those predictions
    runs/x86/scaling-v1i-int8-1t.json          x86 latency vs number of options
    runs/audit/calibration.json                the one headline ECE definition (scripts/audit/calibration.py)
The only hardcoded values are the five total parameter counts, copied from docs/EVAL.md §6
("Parameter counts (measured the same way for every system)").

Outputs (in this folder): <name>.png (@2x) + <name>.svg, and <name>-dark.png/.svg for dark timelines.
Design follows the dataviz reference palette (Decima = categorical slot 1, competitors muted gray).
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"

DECIMA = "v1i"
NAMES = {
    "v1i": "Decima-small",
    "kev-0.5b": "Kev-0.5B",
    "kev-0.8b": "Kev-0.8B",
    "laya": "Laya",
    "laya-multilingual": "Laya-multilingual",
}
# Total parameters as run, docs/EVAL.md §6 (safetensors headers / param_counts()).
PARAMS = {
    "v1i": 122_388_869,
    "kev-0.5b": 494_492_032,
    "kev-0.8b": 752_917_824,
    "laya": 421_293_830,
    "laya-multilingual": 321_908_998,
}
LANG = {
    "en": "English", "de": "German", "fr": "French", "es": "Spanish", "pt": "Portuguese",
    "ru": "Russian", "tr": "Turkish", "ar": "Arabic", "hi": "Hindi", "ta": "Tamil",
    "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "sw": "Swahili",
}

THEMES = {
    "light": dict(
        surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
        grid="#e1e0d9", axis="#c3c2b7", s1="#2a78d6", s2="#eb6834", other="#a8a69f",
    ),
    "dark": dict(
        surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781",
        grid="#2c2c2a", axis="#383835", s1="#3987e5", s2="#d95926", other="#7a7872",
    ),
}
FONT = ["Inter", "Ubuntu", "Liberation Sans", "DejaVu Sans"]


# ----------------------------------------------------------------------------- data
def load_results() -> dict:
    return {s: json.load(open(RUNS / f"v1i-{s}.json"))["results"] for s in ("kev", "laya", "decima", "jevtyped")}


def ok(v) -> bool:
    return v is not None and not (isinstance(v, float) and math.isnan(v))


def set_mean(res: dict, system: str, metric: str = "acc") -> tuple[float, int]:
    vals = [v[metric] for v in res[system].values() if v.get("n", 0) > 0 and ok(v.get(metric))]
    return sum(vals) / len(vals), len(vals)


# ----------------------------------------------------------------------------- style
UBUNTU_VF = Path("/usr/share/fonts/truetype/ubuntu/Ubuntu[wdth,wght].ttf")
MEDIUM_TTF: Path | None = None


def register_fonts():
    """Ubuntu ships as a variable font, which matplotlib renders at its default weight only.
    Instantiate a static Medium (wght 500) into a temp dir for titles and Decima's labels."""
    global MEDIUM_TTF
    if not UBUNTU_VF.exists():
        return
    import atexit
    import shutil
    import tempfile
    from fontTools.ttLib import TTFont  # fontTools ships with matplotlib
    from fontTools.varLib.instancer import instantiateVariableFont
    tmp = Path(tempfile.mkdtemp(prefix="decima-fig-"))
    atexit.register(shutil.rmtree, tmp, ignore_errors=True)
    MEDIUM_TTF = tmp / "Ubuntu-Medium.ttf"
    instantiateVariableFont(TTFont(str(UBUNTU_VF)), {"wght": 500, "wdth": 100}).save(str(MEDIUM_TTF))


def setup(theme: str):
    t = THEMES[theme]
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": FONT, "font.size": 10.5,
        "svg.fonttype": "path", "axes.unicode_minus": False,
        "figure.facecolor": t["surface"], "axes.facecolor": t["surface"],
        "savefig.facecolor": t["surface"], "text.color": t["ink"],
        "axes.edgecolor": t["axis"], "axes.labelcolor": t["ink2"], "axes.linewidth": 1.0,
        "xtick.color": t["muted"], "ytick.color": t["muted"],
        "xtick.labelcolor": t["ink2"], "ytick.labelcolor": t["ink2"],
        "xtick.major.size": 0, "ytick.major.size": 0, "xtick.minor.size": 0, "ytick.minor.size": 0,
        "xtick.major.pad": 6, "ytick.major.pad": 6,
        "axes.grid": False, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    })
    return t


def _width_in(fig, text, **kw) -> float:
    tmp = fig.text(0, 0, text, **kw)
    w = tmp.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
    tmp.remove()
    return w


def wrap(fig, text, max_w, **kw) -> str:
    """Greedy word wrap by rendered width (inches), keeping explicit newlines."""
    out = []
    for para in text.split("\n"):
        line = ""
        for word in para.split(" "):
            cand = f"{line} {word}".strip()
            if line and _width_in(fig, cand, **kw) > max_w:
                out.append(line)
                line = word
            else:
                line = cand
        out.append(line)
    return "\n".join(out)


def put(fig, x_in, y_in, text, max_w, va="top", **kw):
    """Place wrapped text at inch coordinates; return its height in inches."""
    W, H = fig.get_size_inches()
    txt = fig.text(x_in / W, y_in / H, wrap(fig, text, max_w, **kw), ha="left", va=va, linespacing=1.35, **kw)
    return txt.get_window_extent(fig.canvas.get_renderer()).height / fig.dpi


def emph(size, on=True):
    """Font for emphasis (title, Decima's labels). matplotlib cannot pick weights of a variable font,
    so the instantiated Medium face is addressed by file; elsewhere fall back to the bold of the sans stack."""
    from matplotlib.font_manager import FontProperties
    if not on:
        return FontProperties(family=FONT, size=size)
    if MEDIUM_TTF is not None:
        return FontProperties(fname=str(MEDIUM_TTF), size=size)
    return FontProperties(family=FONT, weight="bold", size=size)


def frame(t, title, subtitle, caption, size=(8, 4.5), left=0.95, right=0.35, xband=0.55, gap=0.35, width=None):
    """Takeaway title + subtitle on top, caption at the bottom, axes in the measured space between."""
    W, H = size
    fig = plt.figure(figsize=size)
    pad, maxw = 0.3, W - 0.6
    y = H - 0.28
    y -= put(fig, pad, y, title, maxw, color=t["ink"], fontproperties=emph(14.5)) + 0.08
    if subtitle:
        y -= put(fig, pad, y, subtitle, maxw, fontsize=10.5, color=t["ink2"])
    cap_h = put(fig, pad, 0.2, caption, maxw, va="bottom", fontsize=8.5, color=t["muted"]) if caption else 0
    bottom, top = 0.2 + cap_h + xband, y - gap
    w = width if width is not None else W - left - right
    ax = fig.add_axes([left / W, bottom / H, w / W, (top - bottom) / H])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    return fig, ax


def ygrid(ax, t):
    ax.yaxis.grid(True, color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def xgrid(ax, t):
    ax.xaxis.grid(True, color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, name, theme):
    stem = name if theme == "light" else f"{name}-dark"
    fig.savefig(OUT / f"{stem}.png", dpi=200)
    fig.savefig(OUT / f"{stem}.svg")
    plt.close(fig)


def dot(ax, x, y, color, t, size=9, zorder=3, marker="o", hollow=False):
    ax.plot([x], [y], marker=marker, ms=size, color=color, mfc=t["surface"] if hollow else color,
            mec=t["surface"] if not hollow else color, mew=2 if not hollow else 1.8, zorder=zorder, ls="none")


def pct(v, _=None):
    return f"{v * 100:.0f}%"


def legend_row(fig, t, items, x_in, y_in, gap_in=0.3):
    """Legend for >=2 series: marker + ink text, laid out left to right at inch coordinates."""
    from matplotlib.lines import Line2D
    W, H = fig.get_size_inches()
    x = x_in
    for label, color, hollow in items:
        fig.add_artist(Line2D([(x + 0.06) / W], [y_in / H], marker="o", ms=8, ls="none", color=color,
                              mfc=t["surface"] if hollow else color, mec=color if hollow else t["surface"],
                              mew=1.8 if hollow else 2, transform=fig.transFigure))
        fig.text((x + 0.18) / W, y_in / H, label, va="center", ha="left", fontsize=10, color=t["ink2"])
        x += 0.18 + _width_in(fig, label, fontsize=10) + gap_in


def axes_top_in(fig, ax):
    return ax.get_position().y1 * fig.get_size_inches()[1]


# ----------------------------------------------------------------------------- 1. size vs accuracy
def fig_size_vs_accuracy(res, theme, suite_set, name):
    t = setup(theme)
    r = res[suite_set]
    systems = [s for s in NAMES if s in r]
    acc = {s: set_mean(r, s)[0] for s in systems}
    nsuites = set_mean(r, DECIMA)[1]
    ranked = sorted(systems, key=lambda s: -acc[s])

    if suite_set == "laya":
        assert ranked[0] == DECIMA, "claim check: Decima is not first on the Laya protocol"
        langs = {v["lang"] for v in r[DECIMA].values()}
        title = (f"On Laya's MASSIVE + XNLI protocol, Decima-small scores {acc[DECIMA] * 100:.1f} % "
                 f"at {PARAMS[DECIMA] / 1e6:.0f}M parameters")
        subtitle = (f"Mean accuracy over Laya's {nsuites} suites (MASSIVE intent + XNLI), Laya's published protocol "
                    f"re-run by us; in-distribution for Decima")
        caption = (f"Decima trained on the MASSIVE and XNLI/MNLI train splits; Laya reports it did not. "
                   f"{nsuites} suites / {len(langs)} languages; test items unseen by Decima. Laya has "
                   f"{PARAMS['laya'] / PARAMS[DECIMA]:.1f}× Decima-small's parameters. Kev-0.8B was not run on this "
                   f"protocol. Parameters: total, docs/EVAL.md §6.")
    else:
        best = ranked[0]
        gap = (acc[best] - acc[DECIMA]) * 100
        pos = ranked.index(DECIMA) + 1
        assert best != DECIMA
        title = (f"Kev's published protocol: Decima-small is #{pos}, {gap:.1f} points behind "
                 f"{NAMES[best]} at 1/{PARAMS[best] / PARAMS[DECIMA]:.0f} the size")
        subtitle = f"Mean accuracy over Kev's {nsuites} English suites, Kev's published protocol re-run by us"
        caption = ("Kev protocol: banking77, AG News, BoolQ, MNLI, SST-5, Yelp + two yes/no variants. "
                   "Decima trained on the banking77, AG News, BoolQ, MNLI and SST-5 train splits (not Yelp). Parameters: total, docs/EVAL.md §6.")

    fig, ax = frame(t, title, subtitle, caption, left=0.95, right=0.4, xband=0.62)
    ygrid(ax, t)
    ax.set_xscale("log")
    ax.set_xlim(95e6, 1.0e9)
    ax.xaxis.set_major_locator(FixedLocator([1e8, 2e8, 3e8, 5e8, 1e9]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1e6:,.0f}M" if v < 1e9 else "1B"))
    hi = math.ceil((max(acc.values()) + 0.02) * 20) / 20
    ax.set_ylim(0, hi)  # zero baseline: a truncated axis exaggerates the gaps
    ax.yaxis.set_major_formatter(FuncFormatter(pct))
    ax.set_xlabel("Parameters (total, log scale)", color=t["ink2"], fontsize=9.5, labelpad=6)
    ax.set_ylabel("Accuracy", color=t["ink2"], fontsize=9.5)

    # label placement: right of the dot, unless two dots sit close in accuracy -> above / below
    place = {s: "right" for s in systems}
    for a in systems:
        for b in systems:
            if a < b and abs(acc[a] - acc[b]) < 0.03 and max(PARAMS[a], PARAMS[b]) / min(PARAMS[a], PARAMS[b]) < 2:
                hi_s, lo_s = (a, b) if acc[a] > acc[b] else (b, a)
                place[hi_s], place[lo_s] = "above", "below"
    for s in systems:
        is_d = s == DECIMA
        dot(ax, PARAMS[s], acc[s], t["s1"] if is_d else t["other"], t, size=12 if is_d else 9, zorder=4 if is_d else 3)
        off, ha, va = {"right": ((10, 0), "left", "center"), "above": ((0, 10), "center", "bottom"),
                       "below": ((0, -10), "center", "top")}[place[s]]
        ax.annotate(f"{NAMES[s]}  {acc[s] * 100:.1f}%", (PARAMS[s], acc[s]), xytext=off, textcoords="offset points",
                    ha=ha, va=va, fontproperties=emph(11 if is_d else 10, is_d), color=t["ink"] if is_d else t["ink2"])
    save(fig, name, theme)
    return {NAMES[s]: round(acc[s], 4) for s in ranked}, title


# ----------------------------------------------------------------------------- 2. option-order flips
def fig_flips(res, theme):
    t = setup(theme)
    flips, counts = {}, {}
    for s in NAMES:
        vals = [v["flip"] for st in ("kev", "laya", "decima") if s in res[st]
                for v in res[st][s].values() if v.get("n", 0) > 0 and ok(v.get("flip"))]
        flips[s], counts[s] = sum(vals) / len(vals), len(vals)
    dec_vals = [v["flip"] for st in ("kev", "laya", "decima") for v in res[st][DECIMA].values() if ok(v.get("flip"))]
    assert max(dec_vals) == 0.0, "claim check: Decima flipped on some suite"
    order = [DECIMA] + sorted([s for s in NAMES if s != DECIMA], key=lambda s: flips[s])

    title = "Shuffle the options: Decima's answer never changes"
    subtitle = "Share of answers that change when the same options are presented in a different order"
    caption = (f"Mean over every Kev, Laya and Decima-bench suite each model was run on "
               f"({min(counts.values())}–{max(counts.values())} suites per model; Decima-small: 0 flips on all "
               f"{counts[DECIMA]}). Decima scores each option independently, so this holds by construction.")
    fig, ax = frame(t, title, subtitle, caption, left=1.75, right=0.5, xband=0.45)
    xgrid(ax, t)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_color(t["axis"])
    n = len(order)
    ypos = list(range(n))[::-1]
    ax_h_px = ax.get_position().height * fig.get_figheight() * 100  # CSS px at 1x
    bar_h = min(0.6, 22 / (ax_h_px / n))  # cap bar thickness at ~22 px
    for y, s in zip(ypos, order):
        is_d = s == DECIMA
        if flips[s] > 0:
            ax.barh(y, flips[s], height=bar_h, color=t["other"], zorder=3)
        ax.annotate("0%" if is_d else f"{flips[s] * 100:.1f}%", (flips[s], y), xytext=(10 if is_d else 6, 0),
                    textcoords="offset points", va="center", ha="left", fontproperties=emph(10.5, is_d),
                    color=t["ink"] if is_d else t["ink2"])
    ax.plot([0], [ypos[0]], marker="o", ms=10, color=t["s1"], mec=t["surface"], mew=2, zorder=4, clip_on=False)
    ax.set_yticks(ypos)
    ax.set_yticklabels([NAMES[s] for s in order])
    for lbl, s in zip(ax.get_yticklabels(), order):
        if s == DECIMA:
            lbl.set_color(t["ink"])
            lbl.set_fontproperties(emph(10.5))
    ax.tick_params(axis="y", labelsize=10.5, pad=10)
    ax.set_xlim(0, max(flips.values()) * 1.15)
    ax.set_ylim(-0.6, n - 0.4)
    ax.xaxis.set_major_formatter(FuncFormatter(pct))
    save(fig, "option_order_flips", theme)
    return {NAMES[s]: (round(flips[s], 4), counts[s]) for s in order}, title


# ----------------------------------------------------------------------------- 3a. pairwise ECE
def pairwise_ece(res):
    """The headline ECE comes from runs/audit/calibration.json (scripts/audit/calibration.py), the single
    definition every document quotes; `res` is only used to check the file is not stale."""
    audit = json.load(open(RUNS / "audit" / "calibration.json"))["headline_as_shipped"]
    out = {c: (v["decima"], v[c], v["suites"]) for c, v in audit.items()}
    for c, v in audit.items():  # staleness check against the results files
        for suite in v["suite_ids"]:
            assert any(suite in r.get(DECIMA, {}) and suite in r.get(c, {}) for r in res.values()), suite
    return out


def fig_calibration(res, theme):
    t = setup(theme)
    pw = pairwise_ece(res)
    assert all(d < c for d, c, _ in pw.values()), "claim check: Decima ECE not lowest in every pairing"
    order = sorted(pw, key=lambda c: pw[c][1])
    title = "Lowest calibration error as shipped, head-to-head against every competitor"
    subtitle = "Expected calibration error (ECE, lower is better), mean over the suites both models were run on"
    caption = ("Probabilities as each model ships them (Kev raw, Laya with its shipped temperature); 15-bin ECE, mean "
               "of per-suite values. Suites from the Kev, Laya, Decima-bench and JevBench+typed sets; FarsTail excluded. "
               "Laya's MASSIVE/XNLI suites are in-distribution for Decima (Laya reports it did not train on them). "
               "With per-suite temperature scaling the systems are close. Source: runs/audit/calibration.json.")
    fig, ax = frame(t, title, subtitle, caption, left=2.35, right=0.45, xband=0.45, gap=0.7)
    legend_row(fig, t, [("Decima-small", t["s1"], False), ("Competitor", t["other"], False)],
               2.35, axes_top_in(fig, ax) + 0.3)
    xgrid(ax, t)
    ax.spines["bottom"].set_visible(False)
    n = len(order)
    ypos = list(range(n))[::-1]
    for y, c in zip(ypos, order):
        d, v, k = pw[c]
        ax.plot([d, v], [y, y], color=t["axis"], lw=2, zorder=2)
        dot(ax, v, y, t["other"], t, size=10)
        dot(ax, d, y, t["s1"], t, size=10, zorder=4)
        ax.annotate(f"{v:.3f}", (v, y), xytext=(9, 0), textcoords="offset points", va="center", ha="left",
                    fontsize=10, color=t["ink2"])
        ax.annotate(f"{d:.3f}", (d, y), xytext=(-9, 0), textcoords="offset points", va="center", ha="right",
                    fontproperties=emph(10), color=t["ink"])
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"vs {NAMES[c]}  ({pw[c][2]} suites)" for c in order])
    ax.tick_params(axis="y", labelsize=10)
    ax.set_xlim(-0.035, max(v for _, v, _ in pw.values()) * 1.12)
    ax.set_ylim(-0.6, n - 0.4)
    ax.xaxis.set_major_locator(FixedLocator([0, 0.1, 0.2, 0.3, 0.4]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}"))
    save(fig, "calibration", theme)
    return {NAMES[c]: (round(pw[c][0], 4), round(pw[c][1], 4), pw[c][2]) for c in order}, title


# ----------------------------------------------------------------------------- 3b. reliability diagram
def reliability(system, bins=10):
    items = {}
    for line in open(RUNS / "items" / "laya.jsonl"):
        it = json.loads(line)
        if it.get("split") == "eval" and "group" not in it:
            items[it["id"]] = it["gold"]
    rows = []
    for line in open(RUNS / "preds" / f"laya--{system}.jsonl"):
        p = json.loads(line)
        if p["id"] in items:
            probs = p["probs"]
            k = max(range(len(probs)), key=probs.__getitem__)
            rows.append((probs[k], k == items[p["id"]]))
    assert len(rows) == len(items), f"{system}: {len(rows)} preds for {len(items)} items"
    out = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [(c, y) for c, y in rows if (lo < c <= hi) or (b == 0 and c == 0)]
        if sel:
            out.append((sum(c for c, _ in sel) / len(sel), sum(y for _, y in sel) / len(sel), len(sel)))
    ece = sum(n * abs(c - a) for c, a, n in out) / len(rows)
    top = [(c, y) for c, y in rows if c > 0.99]
    return out, ece, len(rows), len(top) / len(rows), (sum(y for _, y in top) / len(top) if top else float("nan")), \
        (sum(y for _, y in top), len(top))


def fig_reliability(theme):
    t = setup(theme)
    need = [RUNS / "preds" / f"laya--{s}.jsonl" for s in ("v1i", "laya", "laya-multilingual")] + [RUNS / "items" / "laya.jsonl"]
    if not all(p.exists() for p in need):
        return None, None
    MIN_N = 20
    d_bins, d_ece, n, d_top, d_topacc, d_cnt = reliability("v1i")
    l_bins, l_ece, _, l_top, l_topacc, _ = reliability("laya")
    m_bins, m_ece, _, m_top, m_topacc, _ = reliability("laya-multilingual")
    dev = max(abs(c - a) for c, a, k in d_bins if k >= MIN_N)
    assert d_ece < min(l_ece, m_ece) and dev < 0.06, "claim check: Decima not visibly on the diagonal"
    title = "On Laya's protocol, Decima's confidence tracks its accuracy; both Laya checkpoints are overconfident"
    subtitle = ("Accuracy vs stated confidence, Laya's published protocol re-run by us, 10 confidence bins "
                "(on the diagonal = calibrated)")
    caption = (f"Decima trained on the MASSIVE and XNLI/MNLI train splits; Laya reports it did not. 29 suites, {n:,} eval "
               f"items (option-order copies excluded), probabilities as shipped; bins with <{MIN_N} items not drawn. "
               f"ECE here is pooled over all items with 10 bins, so it is not comparable with the per-suite 15-bin ECE "
               f"quoted elsewhere. Decima-small is >99% confident on {d_top * 100:.0f}% of answers and right on "
               f"{d_cnt[0]:,}/{d_cnt[1]:,} ({d_topacc * 100:.1f}%) of those.")
    fig, ax = frame(t, title, subtitle, caption, size=(8, 6.4), left=1.0, xband=0.62, gap=0.3)
    pos = ax.get_position()
    W, H = fig.get_size_inches()
    side = pos.height * H
    ax.set_position([pos.x0, pos.y0, side / W, pos.height])
    ygrid(ax, t)
    xgrid(ax, t)
    ax.plot([0, 1], [0, 1], color=t["axis"], lw=1.2, zorder=1)
    ax.annotate("perfectly calibrated", (0.62, 0.62), xytext=(-6, 6), textcoords="offset points", rotation=45,
                rotation_mode="anchor", ha="center", va="bottom", fontsize=9, color=t["muted"])
    series = ((l_bins, t["other"], "Laya", l_ece, 3, False, (14, -14)),
              (m_bins, t["s2"], "Laya-multilingual", m_ece, 3, False, (14, 14)),
              (d_bins, t["s1"], "Decima-small", d_ece, 5, True, (14, 0)))
    for bins_, color, lab, ece, z, is_d, off in series:
        pts = [(c, a) for c, a, k in bins_ if k >= MIN_N]
        ax.plot([c for c, _ in pts], [a for _, a in pts], color=color, lw=2, zorder=z)
        for c, a in pts:
            dot(ax, c, a, color, t, size=8, zorder=z + 1)
        c, a = pts[-1]
        ax.annotate(f"{lab}\npooled ECE {ece:.3f}", (c, a), xytext=off, textcoords="offset points", va="center",
                    ha="left", fontproperties=emph(10.5, is_d), linespacing=1.3,
                    color=t["ink"] if is_d else t["ink2"], annotation_clip=False)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.xaxis.set_major_formatter(FuncFormatter(pct))
    ax.yaxis.set_major_formatter(FuncFormatter(pct))
    ax.set_xlabel("Stated confidence (top answer)", color=t["ink2"], fontsize=9.5)
    ax.set_ylabel("Actual accuracy", color=t["ink2"], fontsize=9.5)
    save(fig, "calibration_reliability", theme)
    return {"decima_ece10": round(d_ece, 4), "laya_ece10": round(l_ece, 4), "laya_ml_ece10": round(m_ece, 4),
            "decima_top_counts": d_cnt, "laya_ml_top": round(m_top, 3), "laya_ml_topacc": round(m_topacc, 3),
            "decima_maxdev": round(dev, 3),
            "laya_top": round(l_top, 3), "laya_topacc": round(l_topacc, 3), "decima_top": round(d_top, 3),
            "decima_topacc": round(d_topacc, 3)}, title


# ----------------------------------------------------------------------------- 4. speed vs options
def fig_speed(theme):
    t = setup(theme)
    rows = json.load(open(RUNS / "x86" / "scaling-v1i-int8-1t.json"))
    rows = sorted((r for r in rows if r["precision"] == "int8" and r["threads"] == 1), key=lambda r: r["choices"])
    x = [r["choices"] for r in rows]
    y = [r["cached_p50_ms"] for r in rows]
    slope = (y[-1] - y[0]) / (x[-1] - x[0])  # end-to-end cost per extra option (measured end points)
    title = f"{y[0]:.0f} ms at {x[0]} options, ~{slope:.0f} ms per extra option, {y[-1] / 1000:.2f} s at {x[-1]:,}"
    subtitle = "Decima-small int8, 1 thread on an x86 laptop core, median ms per decision (option encodings cached)"
    caption = ("Intel Core Ultra 7 155H, onnxruntime, batch 1, short state (MASSIVE utterance), short intent-label "
               "options (the model-card speed table uses the benchmark option sets, e.g. AG News: 20 ms at 4). Under 50 ms holds up to "
               "roughly 20–40 options depending on option length; large option sets need a retrieval step. A new "
               "option set costs a one-time encode. Source: docs/BENCH-x86.md.")
    fig, ax = frame(t, title, subtitle, caption, left=0.95, right=0.45, xband=0.62)
    ygrid(ax, t)
    ax.plot(x, y, color=t["s1"], lw=2, zorder=3)
    for a, b in zip(x, y):
        dot(ax, a, b, t["s1"], t, size=8, zorder=4)
    ax.set_xlim(0, max(x) * 1.03)
    ax.set_ylim(0, max(y) * 1.15)
    ax.xaxis.set_major_locator(FixedLocator([0, 200, 400, 600, 800, 1000]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.yaxis.set_major_locator(FixedLocator([0, 250, 500, 750, 1000]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:g} s" if v >= 1000 else f"{v:.0f} ms"))
    ax.set_xlabel("Number of options", color=t["ink2"], fontsize=9.5)
    kw = dict(textcoords="offset points", fontsize=10.5, color=t["ink"])
    ax.annotate(f"{x[0]} options: {y[0]:.0f} ms", (x[0], y[0]), xytext=(6, 62), ha="left", va="bottom",
                arrowprops=dict(arrowstyle="-", color=t["muted"], lw=0.8, shrinkA=2, shrinkB=6), **kw)
    ax.annotate(f"{x[-1]:,} options: {y[-1] / 1000:.2f} s", (x[-1], y[-1]), xytext=(-12, 6), ha="right", va="bottom", **kw)
    save(fig, "speed_vs_options", theme)
    return {"slope_ms": round(slope, 3), "points": list(zip(x, y))}, title


# ----------------------------------------------------------------------------- 5. MASSIVE per language
def fig_massive(res, theme):
    t = setup(theme)
    r = res["laya"]
    suites = [s for s in r[DECIMA] if s.startswith("laya/massive-")]
    langs = [s.split("-", 1)[1] for s in suites]
    models = (DECIMA, "laya-multilingual", "laya")
    acc = {m: {lg: r[m][f"laya/massive-{lg}"]["acc"] for lg in langs} for m in models}
    order = sorted(langs, key=lambda lg: acc[DECIMA][lg])  # bottom -> top, best on top
    wins = sum(acc[DECIMA][lg] > max(acc["laya-multilingual"][lg], acc["laya"][lg]) for lg in langs)
    gap = [acc[DECIMA][lg] - acc["laya-multilingual"][lg] for lg in langs]
    if wins == len(langs):
        title = (f"Decima-small leads in all {len(langs)} MASSIVE languages, "
                 f"by {min(gap) * 100:.0f}–{max(gap) * 100:.0f} points over Laya-ML")
    else:
        title = f"Decima-small leads in {wins} of {len(langs)} MASSIVE languages"
    n_items = r[DECIMA][suites[0]]["n"]
    subtitle = (f"MASSIVE intent accuracy per language; Laya's published protocol re-run by us, 20 options, "
                f"{n_items} test items per language")
    caption = ("Caveat: Decima trained on MASSIVE's train split; Laya reports it did not. The test items are unseen by "
               "Decima, but this is in-distribution for Decima and zero-shot for Laya.")
    fig, ax = frame(t, title, subtitle, caption, size=(8, 6.6), left=1.25, right=0.45, xband=0.45, gap=0.7)
    legend_row(fig, t, [(NAMES[DECIMA], t["s1"], False), (NAMES["laya-multilingual"] + " (Laya-ML)", t["s2"], False),
                        (NAMES["laya"], t["other"], True)], 1.25, axes_top_in(fig, ax) + 0.32)
    xgrid(ax, t)
    ax.spines["bottom"].set_visible(False)
    ys = range(len(order))
    for y, lg in zip(ys, order):
        vals = [acc[m][lg] for m in models]
        ax.plot([min(vals), max(vals)], [y, y], color=t["grid"], lw=1.5, zorder=1)
        dot(ax, acc["laya"][lg], y, t["other"], t, size=8, hollow=True)
        dot(ax, acc["laya-multilingual"][lg], y, t["s2"], t, size=9)
        dot(ax, acc[DECIMA][lg], y, t["s1"], t, size=9, zorder=4)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([LANG.get(lg, lg) for lg in order])
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlim(0, 1.0)
    ax.xaxis.set_major_formatter(FuncFormatter(pct))
    save(fig, "multilingual_massive", theme)
    return {LANG[lg]: tuple(round(acc[m][lg], 3) for m in models) for lg in order[::-1]}, title


# ----------------------------------------------------------------------------- main
def main():
    register_fonts()
    res = load_results()
    summary = {}
    for theme in ("light", "dark"):
        summary["size_vs_accuracy"] = fig_size_vs_accuracy(res, theme, "laya", "size_vs_accuracy")
        summary["size_vs_accuracy_kev"] = fig_size_vs_accuracy(res, theme, "kev", "size_vs_accuracy_kev")
        summary["option_order_flips"] = fig_flips(res, theme)
        summary["calibration"] = fig_calibration(res, theme)
        summary["calibration_reliability"] = fig_reliability(theme)
        summary["speed_vs_options"] = fig_speed(theme)
        summary["multilingual_massive"] = fig_massive(res, theme)
    for k, (data, title) in summary.items():
        print(f"\n== {k}\n   title: {title}\n   data:  {data}")


if __name__ == "__main__":
    main()
