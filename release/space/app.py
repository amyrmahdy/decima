"""Decima Playground — a Hugging Face Space for Decima-small.

situation + question + free-text options → calibrated probabilities, on one CPU core.
Runs on ONNX Runtime (no PyTorch). Local test:

    OMP_NUM_THREADS=1 uv run --no-project --with-requirements requirements.txt python app.py
"""

from __future__ import annotations

import html
import json
import os
import random
import re
import statistics
import threading
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import gradio as gr  # noqa: E402
import plotly.graph_objects as go  # noqa: E402

from decima import Decima, Question  # noqa: E402

# ───────────────────────────── configuration ─────────────────────────────

MODEL = os.environ.get("DECIMA_MODEL", "amyrmahdy/decima-small")          # HF repo id (files under onnx/int8/)
LOCAL_FALLBACK = os.environ.get("DECIMA_LOCAL", "export/v1i-int8")
HF_URL = "https://huggingface.co/amyrmahdy/decima-small"                                             # model card
GITHUB_URL = "https://github.com/amyrmahdy/decima"
REPORT_URL = "https://github.com/amyrmahdy/decima/blob/main/docs/TECHNICAL-REPORT.md"                                     # technical report

HERE = Path(__file__).resolve().parent
EX = HERE / "examples"
MAX_OPTIONS = 200

T_LOAD = time.perf_counter()


def _load() -> tuple[Decima, str]:
    if "{{" not in MODEL:
        return Decima.from_pretrained(MODEL, threads=1), MODEL
    for cand in (Path(LOCAL_FALLBACK), HERE / LOCAL_FALLBACK, HERE.parents[1] / LOCAL_FALLBACK, HERE / "model"):
        if (cand / "encoder.onnx").exists():
            return Decima.from_pretrained(cand, threads=1), str(cand)
    raise FileNotFoundError("Set DECIMA_MODEL to the Hugging Face repo id, or DECIMA_LOCAL to an export folder.")


D, MODEL_SOURCE = _load()
LOAD_S = time.perf_counter() - T_LOAD
LOCK = threading.Lock()          # one decision at a time: the model runs on one thread by design

GALLERY = json.loads((EX / "gallery.json").read_text())
BUSINESS = json.loads((EX / "business_cases.json").read_text())
TICKETS = json.loads((EX / "tickets.json").read_text())
LABELS = {f: (EX / f).read_text().splitlines() for f in ("clinc150_intents.txt", "massive60_intents.txt", "banking77_intents.txt")}
for g in GALLERY["gallery"]:
    if g["choices"] is None:
        g["choices"] = LABELS[g["choices_file"]]

FEATURED = ["route-fa-informal", "route-ar", "route-zh", "route-ru", "route-es", "route-de",
            "clinc-reminder", "massive-fa", "urgency", "rank", "verify", "news"]
CARDS = [g for fid in FEATURED for g in GALLERY["gallery"] if g["id"] == fid]
CARD_BY_ID = {g["id"]: g for g in GALLERY["gallery"]}
SHUFFLE_PRESETS = {p["title"]: p for p in GALLERY["shuffle"]}

TEAMS = ["billing", "technical support", "sales", "account security"]
TEAM_Q = "Which team should handle this request?"
DEFAULT_STATE = "Someone logged into my account from another country."

KINDS = [
    ("choose — pick one of the options", "choose"),
    ("score — ordered levels, list them low → high", "score"),
    ("verify — yes or no, exactly two options", "verify"),
    ("rank — each option judged on its own", "rank"),
]
LANGS = [("auto-detect", "auto"), ("English", "en"), ("Persian", "fa"), ("Arabic", "ar"), ("Russian", "ru"),
         ("Chinese", "zh"), ("Spanish", "es"), ("German", "de"), ("French", "fr"), ("Turkish", "tr"),
         ("Hindi", "hi"), ("Japanese", "ja"), ("Korean", "ko"), ("Urdu", "ur"), ("Other", "xx")]
LANG_NAME = {v: k for k, v in LANGS} | {"xx": "other"}
LANG_NAME["auto"] = "auto"
LANG_NAME["latn"] = "Latin script"

# docs/BENCH-x86.md — Intel Core Ultra 7 155H, one P-core, int8, 1 thread, short state, p50 ms
X86_SCALING = {4: 16, 20: 26, 77: 75, 150: 130}
X86_CARD = {4: 20, 20: 42, 77: 83}


# ───────────────────────────── model helpers ─────────────────────────────

_PERSIAN_ONLY = re.compile(r"[پچژگکی]")


def detect_lang(text: str) -> str:
    """Only Persian vs Arabic changes what the model sees (script normalization); the rest is for display."""
    counts = {
        "ar": len(re.findall(r"[؀-ۿ]", text)),
        "ru": len(re.findall(r"[Ѐ-ӿ]", text)),
        "zh": len(re.findall(r"[一-鿿]", text)),
        "ja": len(re.findall(r"[぀-ヿ]", text)),
        "ko": len(re.findall(r"[가-힯]", text)),
        "hi": len(re.findall(r"[ऀ-ॿ]", text)),
        "el": len(re.findall(r"[Ͱ-Ͽ]", text)),
        "th": len(re.findall(r"[฀-๿]", text)),
    }
    best = max(counts, key=counts.get)
    if counts[best] == 0:
        return "latn"          # English, Spanish, German, … are treated identically by the model
    if best == "ar":
        return "fa" if _PERSIAN_ONLY.search(text) else "ar"
    if best == "zh" and counts["ja"]:
        return "ja"
    return best


def run_decision(state: str, q: Question):
    with LOCK:
        cold = (q.text, tuple(q.choices), q.lang) not in D._cache
        r = D.decide(state, q)
    return r, cold


def warm_up() -> None:
    """Encode the big option sets once so the first visitor does not pay for it."""
    for g in CARDS:
        run_decision(g["state"], Question(g["question"], g["choices"], kind=g["kind"], lang=g["lang"]))


# ───────────────────────────── HTML builders ─────────────────────────────

esc = html.escape


def pct(p: float) -> str:
    if p >= 0.9995:
        return "99.9%+" if p < 1 else "100%"
    return f"{p * 100:.1f}%" if p >= 0.001 else "<0.1%"


def conf_badge(p: float, kind: str) -> str:
    if kind == "rank":
        return '<span class="dz-badge dz-b-info">independent scores</span>'
    level = ("high", "High") if p >= 0.8 else ("mid", "Medium") if p >= 0.5 else ("low", "Low")
    return f'<span class="dz-badge dz-b-{level[0]}">{level[1]} confidence · {pct(p)}</span>'


def scale_html() -> str:
    ticks = "".join(f'<span style="left:{t}%">{t}</span>' for t in (0, 25, 50, 75, 100))
    return f'<div class="dz-row dz-scale" aria-hidden="true"><div></div><div class="dz-ticks">{ticks}</div><div></div></div>'


def rows_html(pairs: list[tuple[str, float]], top: str | None, limit: int | None = None) -> str:
    out = []
    for i, (c, p) in enumerate(pairs):
        if limit and i == limit:
            rest = len(pairs) - limit
            tail = "".join(_row(c2, p2, False) for c2, p2 in pairs[limit:])
            out.append(f'<details class="dz-more"><summary>Show the other {rest} options '
                       f'(all ≤ {pct(pairs[limit][1])})</summary>{tail}</details>')
            break
        out.append(_row(c, p, c == top))
    return "".join(out)


def _row(c: str, p: float, is_top: bool) -> str:
    w = max(p * 100, 0.35)
    return (f'<div class="dz-row{" is-top" if is_top else ""}"><div class="dz-label" dir="auto" title="{esc(c)}">{esc(c)}</div>'
            f'<div class="dz-track"><div class="dz-fill" style="width:{w:.2f}%"></div></div>'
            f'<div class="dz-pct">{pct(p)}</div></div>')


def result_html(r, kind: str, cold: bool, lang: str, detected: bool) -> str:
    pairs = list(zip(r.choices, r.probs))
    if kind != "score":
        pairs.sort(key=lambda t: -t[1])
    n = len(pairs)
    order_note = {"score": "levels in the order you gave them (low → high)",
                  "rank": "each option scored on its own — they need not add up to 100%"}.get(kind, "sorted by probability")
    lang_note = f"{LANG_NAME.get(lang, lang)}{' (detected)' if detected else ''}"
    ms = f"{r.latency_ms:.0f}"
    cold_note = (' <span class="dz-dim">— includes encoding this new option set once; it is cached now, '
                 'run it again to see the steady-state time</span>') if cold else ""
    return f"""
<div class="dz-card dz-result">
  <div class="dz-eyebrow">Decision · {esc(kind)} · {n} option{'s' if n != 1 else ''} · {esc(lang_note)}</div>
  <div class="dz-answer" dir="auto">{esc(r.top)}</div>
  <div class="dz-meta">{conf_badge(r.confidence, kind)}
    <span class="dz-time">decided in <b>{ms} ms</b> on one CPU core{cold_note}</span></div>
  <div class="dz-ledger">{scale_html()}{rows_html(pairs, r.top, limit=12 if n > 15 else None)}</div>
  <div class="dz-foot">{esc(order_note)}</div>
</div>"""


def message_html(title: str, body: str) -> str:
    return f'<div class="dz-card dz-result dz-msg"><div class="dz-eyebrow">{esc(title)}</div><p>{esc(body)}</p></div>'


EMPTY_RESULT = message_html("No decision yet", "Write a situation, a question and at least two options, then press Decide — "
                                              "or pick one of the examples below.")


def gallery_html() -> str:
    cards = []
    for g in CARDS:
        rec = g["recorded"]
        n = len(g["choices"])
        cards.append(f"""
<button class="dz-ex" data-id="{g['id']}" type="button">
  <span class="dz-ex-tag">{esc(g['tag'])}</span>
  <span class="dz-ex-title">{esc(g['title'])}</span>
  <span class="dz-ex-state" dir="auto">{esc(g['state'])}</span>
  <span class="dz-ex-out"><span dir="auto">→ <b>{esc(rec['top'])}</b> · {pct(rec['confidence'])}</span><span class="dz-ex-n">{n} options · recorded answer</span></span>
</button>""")
    return f'<div class="dz-gallery">{"".join(cards)}</div>'


CLICK_JS = """
element.addEventListener('click', (e) => {
  const card = e.target.closest('[data-id]');
  if (!card) return;
  element.querySelectorAll('[data-id]').forEach(c => c.classList.toggle('is-active', c === card));
  trigger('click', {id: card.dataset.id});
});
"""
GALLERY_JS = CLICK_JS.replace("trigger('click', {id: card.dataset.id});", """trigger('click', {id: card.dataset.id});
  const out = document.querySelector('.dz-try');
  if (out && out.getBoundingClientRect().top < 0) out.scrollIntoView({behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'start'});""")


# ───────────────────────────── tab 1 · try it ─────────────────────────────


def decide_ui(state: str, question: str, options: str, kind: str, lang: str):
    choices = [c.strip() for c in (options or "").splitlines() if c.strip()]
    choices = list(dict.fromkeys(choices))
    if not (state or "").strip():
        return message_html("Missing situation", "Describe the situation: a message, a ticket, a document excerpt.")
    if not (question or "").strip():
        return message_html("Missing question", "Ask a question about the situation, e.g. “Which team should handle this?”.")
    if len(choices) < 2:
        return message_html("Need options", "Give at least two options, one per line.")
    if len(choices) > MAX_OPTIONS:
        return message_html("Too many options", f"This demo takes up to {MAX_OPTIONS} options (cost grows ≈ 1 ms per option).")
    if kind == "verify" and len(choices) != 2:
        return message_html("verify takes exactly two options", "Use two options such as “yes” and “no”, or switch to choose.")
    detected = lang == "auto"
    lg = detect_lang(state) if detected else lang
    r, cold = run_decision(state.strip(), Question(question.strip(), choices, kind=kind, lang=lg))
    return result_html(r, kind, cold, lg, detected)


def load_card(evt: gr.EventData):
    g = CARD_BY_ID.get((evt._data or {}).get("id"))
    if g is None:
        return [gr.skip()] * 5
    return g["state"], g["question"], "\n".join(g["choices"]), g["kind"], g["lang"]


# ───────────────────────────── tab 2 · business cases ─────────────────────────────

DOMAIN_NAME = {"ecommerce": "E-commerce", "legal_contract": "Legal · contracts", "compliance_privacy": "Privacy & compliance",
               "it_ops": "IT operations", "product_feedback": "Product feedback", "rag_relevance": "Search relevance",
               "invoice_ap": "Accounts payable", "procurement": "Procurement", "fraud": "Fraud review", "sales_crm": "Sales CRM",
               "insurance": "Insurance", "real_estate": "Real estate", "code_review": "Code review", "it_helpdesk": "IT helpdesk",
               "email_ops": "Email operations", "travel_expense": "Travel & expenses", "banking": "Banking",
               "support_triage": "Support triage", "citation_support": "Citation check", "logistics": "Logistics", "hr": "HR",
               "data_quality": "Data quality", "public_services": "Public services", "education": "Education"}
LN = {"en": "English", "fa": "Persian", "ar": "Arabic", "ru": "Russian"}


def case_label(c: dict) -> tuple[str, str]:
    dom = DOMAIN_NAME.get(c["domain"], c["domain"].replace("_", " ").title())
    langs = LN[c["state_lang"]] + ("" if c["choice_lang"] == c["state_lang"] else f" → {LN[c['choice_lang']]} options")
    return dom, langs


def cases_html() -> str:
    items = []
    for i, c in enumerate(BUSINESS["cases"]):
        dom, langs = case_label(c)
        items.append(f"""
<button class="dz-case{' is-active' if i == 0 else ''}" data-id="{i}" type="button">
  <span class="dz-ex-tag">{esc(dom)} · {esc(langs)}</span>
  <span class="dz-case-title">{esc(c['scenario'][:1].upper() + c['scenario'][1:])}</span>
  <span class="dz-case-agree"><b>{c['agree']}/{c['n']}</b> answers match the teacher</span>
</button>""")
    return f'<div class="dz-cases">{"".join(items)}</div>'


def _argmax(p):
    return max(range(len(p)), key=p.__getitem__)


def _lines(text: str, cls: str = "") -> str:
    attr = f' class="{cls}"' if cls else ""
    return "".join(f'<div dir="auto"{attr}>{esc(line) or "&nbsp;"}</div>' for line in text.split("\n"))


def doc_html(state: str, cut: int | None) -> str:
    """The case document, with the point where Decima's 512-token window ends marked."""
    if not cut or cut >= len(state):
        return _lines(state)
    return (_lines(state[:cut]) + '<div class="dz-cut" role="separator"><span>Decima’s 512-token window ends about here — '
            'the teacher read the rest too</span></div>' + _lines(state[cut:], "dz-unread"))


def case_detail_html(i: int) -> str:
    c = BUSINESS["cases"][i]
    dom, langs = case_label(c)
    doc = doc_html(c["state"], c.get("window_end_char"))
    qs = []
    for k, q in enumerate(c["questions"], 1):
        td, dd = _argmax(q["teacher"]), _argmax(q["decima"])
        rows = []
        order = range(len(q["choices"])) if q["kind"] == "score" else sorted(range(len(q["choices"])), key=lambda j: -q["teacher"][j])
        for j in order:
            ch, pt, pd = q["choices"][j], q["teacher"][j], q["decima"][j]
            rows.append(f"""
<div class="dz-crow{' is-top' if j == dd else ''}{' is-ttop' if j == td else ''}">
  <div class="dz-label" dir="auto">{esc(ch)}</div>
  <div class="dz-pair">
    <div class="dz-track"><div class="dz-fill" style="width:{max(pd * 100, .35):.2f}%"></div></div>
    <div class="dz-track dz-teacher"><div class="dz-fill" style="width:{max(pt * 100, .35):.2f}%"></div></div>
  </div>
  <div class="dz-pct2"><span>{pct(pd)}</span><span class="dz-dim">{pct(pt)}</span></div>
</div>""")
        mark = ('<span class="dz-badge dz-b-agree">✓ same answer</span>' if q["agree"]
                else '<span class="dz-badge dz-b-disagree">✗ different answer</span>')
        kind_note = {"score": " · levels in order", "verify": "", "choose": ""}[q["kind"]]
        qs.append(f"""
<div class="dz-q">
  <div class="dz-q-head"><span class="dz-q-n">Q{k}</span><span class="dz-q-kind">{q['kind']}{kind_note}</span>{mark}</div>
  <div class="dz-q-text" dir="auto">{esc(q['question'])}</div>
  <div class="dz-cledger">{''.join(rows)}</div>
</div>""")
    return f"""
<div class="dz-case-detail">
  <div class="dz-case-meta"><span class="dz-eyebrow">{esc(dom)} · {esc(langs)} · {esc(c['format'])}</span></div>
  <div class="dz-doc" aria-label="The situation">{doc}</div>
  <div class="dz-legend"><span><i class="dz-sw dz-sw-d"></i>Decima-small · 122M · int8 on CPU</span>
    <span><i class="dz-sw dz-sw-t"></i>Teacher · Gemma-4-26B-A4B</span>
    <span class="dz-dim">rows sorted by the teacher; bold = Decima’s pick</span></div>
  {''.join(qs)}
</div>"""


def pick_case(evt: gr.EventData):
    try:
        i = int((evt._data or {}).get("id"))
    except (TypeError, ValueError):
        return gr.skip()
    return case_detail_html(i)


def business_summary_html() -> str:
    s, p = BUSINESS["shown"], BUSINESS["pool"]
    return f"""
<div class="dz-note">
  <p><b>{s['cases']} long cases, every question shown: Decima gives the teacher’s answer on {s['agree']} of {s['questions']}.</b>
  We picked these cases <i>because</i> Decima mostly agrees on them. Across <b>all {p['cases']:,}</b> held-out cases it agrees on
  only <b>{p['agree']:,} of {p['questions']:,}</b> questions ({p['agree'] / p['questions'] * 100:.0f}%) — most of those documents
  are longer than the 512 tokens Decima reads. Long, multi-fact decisions are this model’s weakest area; the teacher,
  a 26B model, is not always right either.</p>
  <p class="dz-dim">The model never trained on these cases: they come from rows of the synthetic set that were generated after
  the training snapshot. {esc(BUSINESS['attribution'])}</p>
</div>"""


# ───────────────────────────── tab 3 · shuffle ─────────────────────────────


def ordered_ledger(title: str, choices: list[str], probs: dict[str, float], top: str, perm_from: list[str] | None = None) -> str:
    rows = []
    for i, c in enumerate(choices, 1):
        p = probs[c]
        moved = ""
        if perm_from is not None:
            j = perm_from.index(c) + 1
            moved = f'<span class="dz-pos">was #{j}</span>' if j != i else '<span class="dz-pos">same place</span>'
        rows.append(f"""
<div class="dz-row{' is-top' if c == top else ''}"><div class="dz-label" dir="auto"><span class="dz-idx">{i}</span>{esc(c)} {moved}</div>
<div class="dz-track"><div class="dz-fill" style="width:{max(p * 100, .35):.2f}%"></div></div><div class="dz-pct">{pct(p)}</div></div>""")
    return f'<div class="dz-card dz-shuf"><div class="dz-eyebrow">{esc(title)}</div><div class="dz-ledger">{"".join(rows)}</div></div>'


def shuffle_view(preset_title: str, state: str, order: list[str]):
    p = SHUFFLE_PRESETS[preset_title]
    base_q = Question(p["question"], p["choices"], kind=p["kind"], lang=p["lang"])
    r0, _ = run_decision(state, base_q)
    r1, _ = run_decision(state, Question(p["question"], order, kind=p["kind"], lang=p["lang"]))
    a, b = r0.as_dict(), r1.as_dict()
    delta = max(abs(a[c] - b[c]) for c in p["choices"])
    return r0, r1, a, b, delta


def fmt_delta(x: float) -> str:
    if x == 0:
        return "0 (identical)"
    m, e = f"{x:.1e}".split("e")
    return f"{m} × 10<sup>{int(e)}</sup>"


def shuffle_html(preset_title, state, order, tally):
    p = SHUFFLE_PRESETS[preset_title]
    r0, r1, a, b, delta = shuffle_view(preset_title, state, order)
    changed = r0.top != r1.top
    left = ordered_ledger("Options as listed", p["choices"], a, r0.top)
    right = ordered_ledger("Same options, shuffled", order, b, r1.top, perm_from=p["choices"])
    t = tally
    return f"""
<div class="dz-shuf-grid">{left}{right}</div>
<div class="dz-verdict">
  <div><div class="dz-eyebrow">Answer</div><div class="dz-big" dir="auto">{'changed' if changed else 'unchanged'} — {esc(r1.top)}</div></div>
  <div><div class="dz-eyebrow">Largest probability change</div><div class="dz-big dz-mono">{fmt_delta(delta)}</div></div>
  <div><div class="dz-eyebrow">This session</div><div class="dz-big">{t['n']} shuffle{'s' if t['n'] != 1 else ''} · {t['flips']} answer change{'s' if t['flips'] != 1 else ''}</div>
  <div class="dz-dim">largest change seen: {fmt_delta(t['max'])}</div></div>
</div>"""


def shuffle_reset(preset_title):
    p = SHUFFLE_PRESETS[preset_title]
    order = list(p["choices"])
    tally = {"n": 0, "flips": 0, "max": 0.0}
    return p["state"], order, tally, shuffle_html(preset_title, p["state"], order, tally)


def shuffle_click(preset_title, state, order, tally):
    p = SHUFFLE_PRESETS[preset_title]
    state = (state or "").strip() or p["state"]
    new = list(p["choices"])
    while len(new) > 1 and new == list(p["choices"]) or new == order:
        random.shuffle(new)
    r0, r1, a, b, delta = shuffle_view(preset_title, state, new)
    tally = {"n": tally["n"] + 1, "flips": tally["flips"] + (r0.top != r1.top), "max": max(tally["max"], delta)}
    return new, tally, shuffle_html(preset_title, state, new, tally)


def shuffle_state_change(preset_title, state, order, tally):
    state = (state or "").strip() or SHUFFLE_PRESETS[preset_title]["state"]
    return shuffle_html(preset_title, state, order, tally)


# ───────────────────────────── tab 4 · auto-route ─────────────────────────────

TICKET_RESULTS: list[dict] = []


def score_tickets() -> None:
    for t in TICKETS:
        r, _ = run_decision(t["text"], Question(TEAM_Q, TEAMS, kind="choose", lang=t["lang"]))
        TICKET_RESULTS.append({**t, "top": r.top, "conf": r.confidence, "ok": r.top == t["gold"]})


INK2 = "#8b8e94"
ACCENT = "#2a78d6"
BAD = "#c2453d"


def route_figure(th: float) -> go.Figure:
    fig = go.Figure()
    rs = TICKET_RESULTS
    ys = [(i % 5 - 2) * 0.16 for i in range(len(rs))]
    for ok, name, color, symbol in ((True, "right team", ACCENT, "circle"), (False, "wrong team", BAD, "x-thin")):
        idx = [i for i, r in enumerate(rs) if r["ok"] == ok]
        fig.add_trace(go.Scatter(
            x=[rs[i]["conf"] for i in idx], y=[ys[i] for i in idx], mode="markers", name=name,
            marker=dict(size=11, color=color, symbol=symbol, line=dict(width=2 if not ok else 1.5, color=color if not ok else "rgba(255,255,255,0.9)")),
            customdata=[[rs[i]["text"][:70], rs[i]["top"], rs[i]["gold"]] for i in idx],
            hovertemplate="%{customdata[0]}<br>→ %{customdata[1]} (%{x:.3f})<br>expected: %{customdata[2]}<extra></extra>"))
    fig.add_vrect(x0=0.3, x1=th, fillcolor="rgba(139,142,148,0.12)", line_width=0, layer="below")
    fig.add_vline(x=th, line=dict(color=ACCENT, width=2))
    fig.add_annotation(x=th, y=0.52, text=f"threshold {th:.2f}", showarrow=False, xanchor="left", xshift=6,
                       font=dict(color=ACCENT, size=12, family="IBM Plex Mono"))
    fig.add_annotation(x=(0.3 + th) / 2, y=-0.52, text="escalate to a person", showarrow=False, font=dict(color=INK2, size=11))
    fig.add_annotation(x=(th + 1.0) / 2, y=-0.52, text="route automatically", showarrow=False, font=dict(color=INK2, size=11))
    fig.update_layout(
        height=250, margin=dict(l=10, r=10, t=10, b=36), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans, sans-serif", color=INK2, size=12),
        legend=dict(orientation="h", x=0, y=1.14, bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(range=[0.3, 1.0], title=dict(text="model confidence (top probability)", font=dict(size=12)),
                   gridcolor="rgba(139,142,148,0.18)", zeroline=False, tickformat=".1f", fixedrange=True),
        yaxis=dict(visible=False, range=[-0.62, 0.62], fixedrange=True), hoverlabel=dict(font_family="IBM Plex Sans"))
    return fig


def route_update(th: float):
    rs = TICKET_RESULTS
    auto = [r for r in rs if r["conf"] >= th]
    right = sum(r["ok"] for r in auto)
    esc_n = len(rs) - len(auto)
    caught = sum(not r["ok"] for r in rs if r["conf"] < th)
    wrong = sum(not r["ok"] for r in rs)
    tiles = f"""
<div class="dz-tiles">
  <div class="dz-tile"><div class="dz-tile-n">{len(auto)}<span>/{len(rs)}</span></div><div class="dz-tile-l">routed automatically</div></div>
  <div class="dz-tile"><div class="dz-tile-n">{right}<span>/{len(auto)}</span></div><div class="dz-tile-l">of those went to the right team</div></div>
  <div class="dz-tile"><div class="dz-tile-n">{esc_n}</div><div class="dz-tile-l">escalated to a person — {caught} of the {wrong} wrong answer{'s' if wrong != 1 else ''} among them</div></div>
</div>"""
    rows = []
    for r in sorted(rs, key=lambda r: -r["conf"]):
        a = r["conf"] >= th
        rows.append(f"""<tr class="{'is-auto' if a else 'is-esc'}"><td><span class="dz-badge {'dz-b-high' if a else 'dz-b-mid'}">{'auto' if a else 'escalate'}</span></td>
<td class="dz-mono">{r['conf']:.3f}</td><td dir="auto">{esc(r['text'])}</td><td>{esc(r['lang'])}</td><td>{esc(r['top'])}</td>
<td>{'<span class="dz-ok">✓</span>' if r['ok'] else f'<span class="dz-bad">✗</span> <span class="dz-dim">expected {esc(r["gold"])}</span>'}</td></tr>""")
    table = f"""<div class="dz-table-wrap"><table class="dz-table"><thead><tr><th>decision</th><th>confidence</th><th>ticket</th><th>lang</th>
<th>model’s team</th><th>right?</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>"""
    return tiles, route_figure(th), table


# ───────────────────────────── tab 5 · speed ─────────────────────────────

SPEED_SETS = {4: TEAMS, 20: LABELS["clinc150_intents.txt"][:20], 77: LABELS["banking77_intents.txt"],
              150: LABELS["clinc150_intents.txt"]}
SPEED_STATE = "I was charged twice for my subscription this month."


def cpu_name() -> str:
    try:
        for line in open("/proc/cpuinfo"):
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    import platform
    return f"{platform.machine()} CPU, {os.cpu_count()} logical cores"


def speed_figure(live: dict[int, float] | None) -> go.Figure:
    fig = go.Figure()
    xs = list(X86_SCALING)
    fig.add_trace(go.Scatter(x=xs, y=[X86_SCALING[x] for x in xs], name="x86 laptop core, reference (BENCH-x86)",
                             mode="lines+markers", line=dict(color=INK2, width=2, dash="dot"), marker=dict(size=8),
                             hovertemplate="%{x} options: %{y} ms (reference)<extra></extra>"))
    if live:
        fig.add_trace(go.Scatter(x=list(live), y=list(live.values()), name="this Space, live (p50)", mode="lines+markers",
                                 line=dict(color=ACCENT, width=2.5), marker=dict(size=9),
                                 hovertemplate="%{x} options: %{y:.1f} ms (live)<extra></extra>"))
    fig.update_layout(
        height=320, margin=dict(l=10, r=10, t=10, b=40), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans, sans-serif", color=INK2, size=12),
        legend=dict(orientation="h", x=0, y=1.12, bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(title=dict(text="number of options"), tickvals=xs, gridcolor="rgba(139,142,148,0.18)", zeroline=False,
                   fixedrange=True),
        yaxis=dict(title=dict(text="ms per decision"), rangemode="tozero", gridcolor="rgba(139,142,148,0.18)", zeroline=False,
                   fixedrange=True))
    return fig


def speed_run(progress=gr.Progress()):
    live, rows = {}, []
    reps = 12
    for n, ch in progress.tqdm(list(SPEED_SETS.items()), desc="timing"):
        q = Question("What is the customer asking about?", ch, kind="choose", lang="en")
        with LOCK:
            cold = (q.text, tuple(q.choices), q.lang) not in D._cache
            t0 = time.perf_counter()
            D.decide(SPEED_STATE, q)
            first = (time.perf_counter() - t0) * 1e3
            lat = sorted(D.decide(SPEED_STATE, q).latency_ms for _ in range(reps))
        p50, p95 = statistics.median(lat), lat[min(len(lat) - 1, round(0.95 * (len(lat) - 1)))]
        live[n] = p50
        rows.append(f"<tr><td class='dz-mono'>{n}</td><td class='dz-mono'><b>{p50:.1f}</b></td><td class='dz-mono'>{p95:.1f}</td>"
                    f"<td class='dz-mono'>{X86_SCALING[n]}</td><td class='dz-mono'>{first:.0f} <span class='dz-dim'>{'new option set' if cold else 'cached'}</span></td></tr>")
    table = f"""
<div class="dz-table-wrap"><table class="dz-table dz-num"><thead><tr><th>options</th><th>p50 ms · live</th><th>p95 ms · live</th>
<th>p50 ms · x86 reference</th><th>first call ms · live</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p class="dz-dim">Live: {esc(cpu_name())}, one thread, int8 ONNX, batch 1, {reps} timed calls per size after the first, short state.
Shared Space hardware is noisy — run it twice. Reference: Intel Core Ultra 7 155H, one P-core, 40 calls per size
(docs/BENCH-x86.md).</p>"""
    return speed_figure(live), table


# ───────────────────────────── page chrome ─────────────────────────────


def link(url: str, text: str) -> str:
    return f'<a href="{esc(url)}" target="_blank" rel="noopener">{esc(text)}</a>'


HEADER = f"""
<header class="dz-header">
  <div class="dz-brand">
    <div class="dz-word">Decima<span class="dz-word-sub">small</span></div>
    <p class="dz-pitch">Give it a situation, a question and your options.<br>Get back calibrated probabilities — on one CPU core.</p>
  </div>
  <div class="dz-side">
    <div class="dz-chips">
      <span class="dz-chip"><b>122M</b> parameters</span>
      <span class="dz-chip"><b>~20 ms</b> per decision on one CPU core (4 options)</span>
      <span class="dz-chip"><b>0%</b> answer changes when options are shuffled</span>
    </div>
    <nav class="dz-links">{link(HF_URL, "Model card")}{link(GITHUB_URL, "GitHub")}{link(REPORT_URL, "Technical report")}</nav>
  </div>
</header>"""

FOOTER = f"""
<footer class="dz-footer">
  <p><b>Read this before you judge the model by this page.</b> The examples are curated: we kept the ones that worked and
  list the ones that did not in our launch notes. For accuracy, see the benchmarks in the {link(HF_URL, "model card")}.
  Known limits: long, multi-fact business decisions and knowledge questions (facts, arithmetic) are weak at this size;
  adding a “none of the above” option can pull real answers into it — use the confidence instead.</p>
  <p class="dz-dim">Decima-small · Apache-2.0 · int8 ONNX on ONNX Runtime · loaded in {LOAD_S:.1f} s from {esc('the Hub' if '{{' not in MODEL else 'a local export')}.
  Synthetic business cases: CC BY 4.0.</p>
</footer>"""

CSS = (HERE / "style.css").read_text()
# dir="auto" on the free-text inputs so Persian/Arabic render right-to-left as you type
JS = """() => {
  const apply = () => document.querySelectorAll('.dz-auto-dir textarea, .dz-auto-dir input')
    .forEach(t => { if (t.getAttribute('dir') !== 'auto') t.setAttribute('dir', 'auto'); });
  apply();
  new MutationObserver(apply).observe(document.body, {childList: true, subtree: true});
}"""
HEAD = """
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Arabic:wght@400;500;600&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;0,700;1,400&display=swap" rel="stylesheet">
<meta name="description" content="Decima-small: a 122M decision model. Situation + question + options → calibrated probabilities on one CPU core.">
"""


def make_theme() -> gr.themes.Base:
    accent = gr.themes.Color(c50="#eef5fd", c100="#d9e8fa", c200="#b5d1f4", c300="#86b3ec", c400="#5693e1", c500="#2a78d6",
                             c600="#1f62b5", c700="#1b4f91", c800="#193f72", c900="#16335b", c950="#0e213c", name="decima")
    warm = gr.themes.Color(c50="#fcfcfb", c100="#f4f4f1", c200="#e8e7e3", c300="#d4d3ce", c400="#a9a7a1", c500="#7d7b75",
                           c600="#5e5c57", c700="#464541", c800="#2e2d2b", c900="#1e1e1c", c950="#141413", name="warm")
    return gr.themes.Base(primary_hue=accent, secondary_hue=accent, neutral_hue=warm, radius_size=gr.themes.sizes.radius_sm,
                          font=[gr.themes.GoogleFont("IBM Plex Sans"), "IBM Plex Sans Arabic", "system-ui", "sans-serif"],
                          font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"]).set(
        body_background_fill="#fcfcfb", body_background_fill_dark="#131415",
        block_background_fill="#ffffff", block_background_fill_dark="#1b1c1e",
        block_border_color="#e8e7e3", block_border_color_dark="#2c2d30", block_shadow="none", block_shadow_dark="none",
        input_background_fill="#ffffff", input_background_fill_dark="#161719",
        button_primary_background_fill="#2a78d6", button_primary_background_fill_hover="#1f62b5",
        button_primary_background_fill_dark="#3d86de", button_primary_background_fill_hover_dark="#5693e1",
        button_primary_text_color="#ffffff", button_primary_text_color_dark="#ffffff",
        slider_color="#2a78d6", slider_color_dark="#5693e1", body_text_color_subdued="#6b6a66",
        block_label_text_weight="500", block_title_text_weight="500",
    )


# ───────────────────────────── layout ─────────────────────────────


def build() -> gr.Blocks:
    first = SHUFFLE_PRESETS[next(iter(SHUFFLE_PRESETS))]
    with gr.Blocks(title="Decima Playground", fill_width=False) as demo:
        gr.HTML(HEADER, js_on_load=None, apply_default_css=False)
        with gr.Tabs(elem_classes="dz-tabs"):
            # 1 · try it
            with gr.Tab("Try it", id="try"):
                with gr.Row(equal_height=False, elem_classes="dz-try"):
                    with gr.Column(scale=5, min_width=320, elem_classes="dz-inputs"):
                        state = gr.Textbox(DEFAULT_STATE, label="Situation", lines=4, max_lines=14, rtl=False,
                                           elem_classes="dz-auto-dir", placeholder="A message, a ticket, an excerpt…")
                        question = gr.Textbox(TEAM_Q, label="Question", lines=1, elem_classes="dz-auto-dir")
                        options = gr.Textbox("\n".join(TEAMS), label="Options — one per line, any wording, any number",
                                             lines=5, max_lines=12, elem_classes="dz-auto-dir")
                        with gr.Row():
                            kind = gr.Radio(KINDS, value="choose", label="Kind of question", elem_classes="dz-kind")
                        with gr.Row(equal_height=False, elem_classes="dz-actions"):
                            lang = gr.Dropdown(LANGS, value="auto", label="Language of the situation",
                                               info="Only Persian and Arabic change the input (script normalization).",
                                               scale=3)
                            go_btn = gr.Button("Decide", variant="primary", scale=2, elem_classes="dz-go")
                    with gr.Column(scale=6, min_width=320):
                        out = gr.HTML(EMPTY_RESULT, js_on_load=None, apply_default_css=False, elem_classes="dz-out")
                gr.HTML('<div class="dz-section"><h2>Examples</h2><p>Each one was run before we put it here; the card shows '
                        'the recorded answer. Click to load it and run it live.</p></div>', js_on_load=None, apply_default_css=False)
                gallery = gr.HTML(gallery_html(), js_on_load=GALLERY_JS, apply_default_css=False)
                inputs = [state, question, options, kind, lang]
                go_btn.click(decide_ui, inputs, out, api_name="decide")
                state.submit(decide_ui, inputs, out, api_name=False)
                gallery.click(load_card, None, inputs, api_name=False).then(decide_ui, inputs, out, api_name=False)

            # 2 · business cases
            with gr.Tab("Business cases", id="cases"):
                gr.HTML('<div class="dz-section"><h2>Long cases the model never trained on</h2><p>One situation, several '
                        'questions about it. Next to each Decima answer is the answer of the 26B teacher that labelled the '
                        'training data — including where they disagree.</p></div>', js_on_load=None, apply_default_css=False)
                gr.HTML(business_summary_html(), js_on_load=None, apply_default_css=False)
                with gr.Row(equal_height=False):
                    with gr.Column(scale=3, min_width=260):
                        case_list = gr.HTML(cases_html(), js_on_load=CLICK_JS, apply_default_css=False)
                    with gr.Column(scale=8, min_width=320):
                        case_view = gr.HTML(case_detail_html(0), js_on_load=None, apply_default_css=False)
                case_list.click(pick_case, None, case_view, api_name=False)

            # 3 · shuffle
            with gr.Tab("Shuffle test", id="shuffle"):
                gr.HTML('<div class="dz-section"><h2>Reorder the options. The answer stays put.</h2><p>Each option is scored '
                        'against the situation on its own, so its position never enters the computation. Shuffle as often as '
                        'you like; the numbers below count every try in this session.</p></div>',
                        js_on_load=None, apply_default_css=False)
                order = gr.State(list(first["choices"]))
                tally = gr.State({"n": 0, "flips": 0, "max": 0.0})
                with gr.Row(equal_height=True):
                    preset = gr.Dropdown(list(SHUFFLE_PRESETS), value=first["title"], label="Question", scale=3)
                    shuf_btn = gr.Button("Shuffle options", variant="primary", scale=1, elem_classes="dz-go")
                shuf_state = gr.Textbox(first["state"], label="Situation (you can edit it)", lines=2, elem_classes="dz-auto-dir")
                shuf_out = gr.HTML(js_on_load=None, apply_default_css=False)
                shuf_btn.click(shuffle_click, [preset, shuf_state, order, tally], [order, tally, shuf_out], api_name="shuffle")
                preset.change(shuffle_reset, preset, [shuf_state, order, tally, shuf_out], api_name=False)
                shuf_state.submit(shuffle_state_change, [preset, shuf_state, order, tally], shuf_out, api_name=False)

            # 4 · auto-route
            with gr.Tab("Auto-route", id="route"):
                gr.HTML('<div class="dz-section"><h2>Automate the sure cases, hand over the rest</h2><p>Because the '
                        'probabilities are calibrated, the threshold is a dependable dial: raise it and fewer tickets are '
                        'automated, and fewer mistakes get through. Twenty realistic tickets in eight languages, expected team '
                        'written down before the model saw them.</p></div>', js_on_load=None, apply_default_css=False)
                th = gr.Slider(0.5, 0.95, value=0.8, step=0.01, label="Route automatically when confidence is at least")
                tiles = gr.HTML(js_on_load=None, apply_default_css=False)
                chart = gr.Plot(show_label=False, elem_classes="dz-plot")
                table = gr.HTML(js_on_load=None, apply_default_css=False)
                gr.HTML('<p class="dz-dim dz-small">Twenty tickets we wrote — a demonstration of the mechanism, not an '
                        'automation rate. Pick your threshold on your own labelled tickets. A threshold is not an off-topic '
                        'filter: a message about something else entirely can still score high on one of your options.</p>',
                        js_on_load=None, apply_default_css=False)
                th.change(route_update, th, [tiles, chart, table], api_name=False, show_progress="hidden")

            # 5 · speed
            with gr.Tab("Speed", id="speed"):
                gr.HTML('<div class="dz-section"><h2>Measured here, now</h2><p>Time one decision on this Space’s CPU with 4, '
                        '20, 77 and 150 options (one thread, option encodings cached). Cost grows about 1 ms per option; the '
                        'dotted line is our x86 laptop reference.</p></div>', js_on_load=None, apply_default_css=False)
                sp_btn = gr.Button("Run the latency sweep", variant="primary", elem_classes="dz-go dz-go-inline")
                sp_plot = gr.Plot(speed_figure(None), show_label=False, elem_classes="dz-plot")
                sp_table = gr.HTML(
                    '<div class="dz-note"><p><b>Reference, x86 laptop</b> (Intel Core Ultra 7 155H, one P-core, int8): '
                    '<span class="dz-mono">20 ms</span> with 4 options, <span class="dz-mono">42 ms</span> with 20, '
                    '<span class="dz-mono">83 ms</span> with 77 (model-card option sets). With short labels, as plotted: '
                    '<span class="dz-mono">16 ms</span> at 4 options and about 1 ms per extra option; 1,000 options take '
                    '<span class="dz-mono">1.06 s</span>.</p></div>', js_on_load=None, apply_default_css=False)
                sp_btn.click(speed_run, None, [sp_plot, sp_table], api_name="speed")

        gr.HTML(FOOTER, js_on_load=None, apply_default_css=False)
        demo.load(decide_ui, inputs, out, api_name=False)
        demo.load(route_update, th, [tiles, chart, table], api_name=False)
        demo.load(shuffle_reset, preset, [shuf_state, order, tally, shuf_out], api_name=False)
    return demo


score_tickets()
threading.Thread(target=warm_up, daemon=True).start()
demo = build()

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2).launch(
        theme=make_theme(), css=CSS, head=HEAD, js=JS,
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        footer_links=["api"], ssr_mode=False)
