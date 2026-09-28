#!/usr/bin/env python
"""Decima-small launch demos: reproduces every output shown in SCENARIOS.md, the articles and the videos.

    uv run python release/launch/demo/run_demos.py                   # all sections
    uv run python release/launch/demo/run_demos.py routing shuffle   # some sections
    uv run python release/launch/demo/run_demos.py --pause           # wait for Enter between sections (screen recording)
    uv run python release/launch/demo/run_demos.py --offline         # block all network access in this process first
    uv run python release/launch/demo/run_demos.py --json out.json   # also dump every decision

Sections: routing, multilingual, shuffle, threshold, clinc, crosslingual, urgency, verify, rank, news, speed
(default: all of these), and `rejected` (appendix A of SCENARIOS.md; only when named).
Runs on one CPU thread (the fastest setting at batch 1, docs/BENCH-x86.md). Probabilities are deterministic;
latencies depend on the machine. Every input below was written before running the model and was not edited
afterwards, with one exception: the non-English §2 lines were revised for naturalness after a native-level
language review (SENTENCES-REVIEW.md, which lists the model's output on both versions). Rejected scenarios are
listed in SCENARIOS.md, appendix A.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

# ─────────────────────────────── inputs ────────────────────────────────

TEAMS = ["billing", "technical support", "sales", "account security"]
TEAM_Q = "Which team should handle this request?"

ROUTING = [
    "Someone logged into my account from another country.",
    "I was charged twice for my subscription this month.",
    "The app crashes every time I open the settings page.",
    "Do you offer a discount if we buy 200 seats for our company?",
]

# four messages, each in English + 6 other languages; the options stay in English.
# Non-English lines were reviewed by a native-level reviewer (SENTENCES-REVIEW.md) before this run.
MULTILINGUAL = [
    ("account security", {
        "en": "Someone logged into my account from another country.",
        "fa": "یک نفر از کشور دیگری وارد حسابم شده است.",
        "ar": "أحدهم سجّل الدخول إلى حسابي من دولة أخرى.",
        "ru": "Кто-то вошёл в мой аккаунт из другой страны.",
        "es": "Alguien inició sesión en mi cuenta desde otro país.",
        "de": "Jemand hat sich aus einem anderen Land in mein Konto eingeloggt.",
        "zh": "有人从国外登录了我的账号。",
    }),
    ("technical support", {
        "en": "The app crashes every time I open the settings page.",
        "fa": "هر بار که صفحهٔ تنظیمات را باز می‌کنم، برنامه بسته می‌شود.",
        "ar": "التطبيق يُغلق فجأة كلما فتحت صفحة الإعدادات.",
        "ru": "Приложение вылетает каждый раз, когда я открываю настройки.",
        "es": "La aplicación se cierra sola cada vez que abro la configuración.",
        "de": "Die App stürzt jedes Mal ab, wenn ich die Einstellungen öffne.",
        "zh": "每次打开设置页面，应用就会崩溃。",
    }),
    ("sales", {
        "en": "Do you offer a discount if we buy 200 seats for our company?",
        "fa": "اگر برای شرکتمان ۲۰۰ اشتراک بخریم تخفیف می‌دهید؟",
        "ar": "هل تقدمون خصماً إذا اشترينا 200 ترخيص لشركتنا؟",
        "ru": "Есть ли скидка, если мы купим 200 лицензий для компании?",
        "es": "¿Hay descuento si compramos 200 licencias para nuestra empresa?",
        "de": "Gibt es Rabatt, wenn wir 200 Lizenzen für unsere Firma kaufen?",
        "zh": "如果我们公司买200个席位，有折扣吗？",
    }),
    ("account security", {   # informal 4th row: "…and it wasn't me!"
        "en": "Someone logged into my account from another country and it wasn't me!",
        "fa": "یه نفر از یه کشور دیگه وارد حسابم شده، کار من نبوده!",
        "ar": "وصلني تنبيه بتسجيل دخول إلى حسابي من دولة أخرى، ولم أكن أنا.",
        "ru": "Кто-то зашёл в мой аккаунт из другой страны, это был не я!",
        "es": "Alguien entró en mi cuenta desde otro país y no fui yo.",
        "de": "Da hat sich jemand aus dem Ausland in mein Konto eingeloggt, das war ich nicht!",
        "zh": "我的账号被人在国外登录了，不是我本人操作的！",
    }),
]

# 20 tickets with the correct team, written before the model saw them (not edited afterwards)
TICKETS = [
    ("en", "My invoice shows the wrong VAT number, can you reissue it?", "billing"),
    ("en", "I got a password reset email I never asked for. Is someone trying to get in?", "account security"),
    ("en", "The dashboard has been loading forever since your update last night.", "technical support"),
    ("en", "We're a team of 40 and want to know what the enterprise plan includes.", "sales"),
    ("en", "Why was I charged $49 when my plan is $29?", "billing"),
    ("en", "The API returns a 500 error whenever I upload a file larger than 10 MB.", "technical support"),
    ("en", "Can I get a demo for my manager next week?", "sales"),
    ("en", "Please turn on two-factor authentication for my account, I think my email was hacked.", "account security"),
    ("en", "How do I update the credit card you charge every month?", "billing"),
    ("en", "Is there a nonprofit discount?", "sales"),
    ("es", "No puedo sincronizar mis archivos desde ayer, la app se queda cargando.", "technical support"),
    ("de", "Ich möchte meine Rechnung als PDF bekommen, nicht per Post.", "billing"),
    ("fr", "Quelqu'un a changé l'adresse e-mail de mon compte sans mon accord.", "account security"),
    ("ru", "Сколько будет стоить годовая подписка для 15 сотрудников?", "sales"),
    ("fa", "رمز عبورم را عوض نکرده‌ام ولی دیگر نمی‌توانم وارد حسابم شوم.", "account security"),
    ("ar", "الإشعارات لا تصل إلى هاتفي منذ التحديث الأخير.", "technical support"),
    ("zh", "我想把按月付费改成按年付费。", "billing"),
    ("tr", "Şirketimiz için özel bir fiyat teklifi alabilir miyiz?", "sales"),
    ("en", "I cancelled last month but you charged me again.", "billing"),
    ("en", "After I changed my phone, the authenticator codes stopped working and I'm locked out.", "account security"),
]
THRESHOLD = 0.8

CLINC_Q = "What is the user's intent? (may be out of scope)"
CLINC = ["restaurant reviews", "nutrition info", "account blocked", "oil change how", "time", "weather", "redeem rewards", "interest rate", "gas type", "accept reservations", "smart home", "user name", "report lost card", "repeat", "whisper mode", "what are your hobbies", "order", "jump start", "schedule meeting", "meeting schedule", "freeze account", "what song", "meaning of life", "restaurant reservation", "traffic", "make call", "text", "bill balance", "improve credit score", "change language", "no", "measurement conversion", "timer", "flip coin", "do you have pets", "balance", "tell joke", "last maintenance", "exchange rate", "uber", "car rental", "credit limit", "none of the above (out of scope)", "shopping list", "expiration date", "routing", "meal suggestion", "tire change", "todo list", "card declined", "rewards balance", "change accent", "vaccines", "reminder update", "food last", "change ai name", "bill due", "who do you work for", "share location", "international visa", "calendar", "translate", "carry on", "book flight", "insurance change", "todo list update", "timezone", "cancel reservation", "transactions", "credit score", "report fraud", "spending history", "directions", "spelling", "insurance", "what is your name", "reminder", "where are you from", "distance", "payday", "flight status", "find phone", "greeting", "alarm", "order status", "confirm reservation", "cook time", "damaged card", "reset settings", "pin change", "replacement card duration", "new card", "roll dice", "income", "taxes", "date", "who made you", "pto request", "tire pressure", "how old are you", "rollover 401k", "pto request status", "how busy", "application status", "recipe", "calendar update", "play music", "yes", "direct deposit", "credit limit change", "gas", "pay bill", "ingredients list", "lost luggage", "goodbye", "what can i ask you", "book hotel", "are you a bot", "next song", "change speed", "plug type", "maybe", "w2", "oil change when", "thank you", "shopping list update", "pto balance", "order checks", "travel alert", "fun fact", "sync device", "schedule maintenance", "apr", "transfer", "ingredient substitution", "calories", "current location", "international fees", "calculator", "definition", "next holiday", "update playlist", "mpg", "min payment", "change user name", "restaurant suggestion", "travel notification", "cancel", "pto used", "travel suggestion", "change volume"]
CLINC_CASES = [
    ("can you move my dentist reminder to friday at 3", "reminder update"),
    ("how many calories are in a banana", "calories"),
    ("my card got declined at the grocery store", "card declined"),
    ("what's the plug type in japan", "plug type"),
    ("i need to rent a car in denver next weekend", "car rental"),
    ("how long will it take for my new card to arrive", "replacement card duration"),
    ("who won the 1998 world cup", "none of the above (out of scope)"),
    ("write me a poem about the sea", "none of the above (out of scope)"),   # the model is unsure here — shown on purpose
]

MASSIVE_Q = "What does the user want the assistant to do?"
MASSIVE = ["alarm query", "alarm remove", "alarm set", "audio volume down", "audio volume mute", "audio volume other", "audio volume up", "calendar query", "calendar remove", "calendar set", "cooking query", "cooking recipe", "datetime convert", "datetime query", "email addcontact", "email query", "email querycontact", "email sendemail", "general greet", "general joke", "general quirky", "iot cleaning", "iot coffee", "iot hue lightchange", "iot hue lightdim", "iot hue lightoff", "iot hue lighton", "iot hue lightup", "iot wemo off", "iot wemo on", "lists createoradd", "lists query", "lists remove", "music dislikeness", "music likeness", "music query", "music settings", "news query", "play audiobook", "play game", "play music", "play podcasts", "play radio", "qa currency", "qa definition", "qa factoid", "qa maths", "qa stock", "recommendation events", "recommendation locations", "recommendation movies", "social post", "social query", "takeaway order", "takeaway query", "transport query", "transport taxi", "transport ticket", "transport traffic", "weather query"]
PERSIAN_CASES = [
    ("فردا ساعت هفت صبح بیدارم کن", "wake me up at seven tomorrow morning", "alarm set"),
    ("چراغ‌های آشپزخانه را خاموش کن", "turn off the kitchen lights", "iot hue lightoff"),
    ("هوای تهران فردا چطور است؟", "what's the weather in Tehran tomorrow?", "weather query"),
    ("یک تاکسی برای فرودگاه بگیر", "get me a taxi to the airport", "transport taxi"),
    ("صدای موزیک را کم کن", "turn the music down", "audio volume down"),
    ("یه جوک برام بگو", "tell me a joke", "general joke"),
]

URGENCY_Q = "How urgent is this ticket?"
URGENCY = ["low", "medium", "high", "critical"]
URGENCY_CASES = [
    "Small typo on your pricing page: 'monthy' instead of 'monthly'.",
    "Could you change the font on my invoices sometime? No rush.",
    "The export to CSV is slow, it takes a few minutes.",
    "I can't log in since this morning and I have a client demo in an hour.",
    "Our checkout page is down and customers can't pay. We're losing sales every minute.",
]

VERIFY_CASES = [
    ("Does the customer ask for a refund?", "The blender stopped working after a week. I want my money back."),
    ("Does the customer ask for a refund?", "Great blender, just wanted to say thanks!"),
    ("Is the customer angry?", "This is the THIRD time I'm writing. Nobody answers. Unacceptable."),
    ("Is the customer angry?", "Hi, quick question about my invoice format, no rush."),
]

RANK_Q = "Which of these help articles are relevant to the customer's problem?"
RANK = ["How to reset your password", "Updating your billing address", "Enabling two-factor authentication",
        "Exporting reports to CSV", "What to do if you see an unfamiliar login"]
RANK_STATE = "Someone logged into my account from another country and I can't get back in."

NEWS_Q = "What is the topic of this news article?"
NEWS = ["World", "Sports", "Business", "Sci/Tech"]
NEWS_CASES = [
    "Central bank holds interest rates steady as inflation cools for a third month.",
    "Striker scores twice in stoppage time to send her club into the cup final.",
    "Researchers unveil a battery chemistry that charges an electric car in ten minutes.",
    "Ceasefire talks resume in Geneva as both delegations arrive for a second round.",
]

BANKING77 = ["activate my card", "age limit", "apple pay or google pay", "atm support", "automatic top up", "balance not updated after bank transfer", "balance not updated after cheque or cash deposit", "beneficiary not allowed", "cancel transfer", "card about to expire", "card acceptance", "card arrival", "card delivery estimate", "card linking", "card not working", "card payment fee charged", "card payment not recognised", "card payment wrong exchange rate", "card swallowed", "cash withdrawal charge", "cash withdrawal not recognised", "change pin", "compromised card", "contactless not working", "country support", "declined card payment", "declined cash withdrawal", "declined transfer", "direct debit payment not recognised", "disposable card limits", "edit personal details", "exchange charge", "exchange rate", "exchange via app", "extra charge on statement", "failed transfer", "fiat currency support", "get disposable virtual card", "get physical card", "getting spare card", "getting virtual card", "lost or stolen card", "lost or stolen phone", "order physical card", "passcode forgotten", "pending card payment", "pending cash withdrawal", "pending top up", "pending transfer", "pin blocked", "receiving money", "Refund not showing up", "request refund", "reverted card payment?", "supported cards and currencies", "terminate account", "top up by bank transfer charge", "top up by card charge", "top up by cash or cheque", "top up failed", "top up limits", "top up reverted", "topping up by card", "transaction charged twice", "transfer fee charged", "transfer into account", "transfer not received by recipient", "transfer timing", "unable to verify identity", "verify my identity", "verify source of funds", "verify top up", "virtual card not working", "visa or mastercard", "why verify identity", "wrong amount of cash received", "wrong exchange rate for cash withdrawal"]

# docs/BENCH-x86.md, "Scaling with the number of options": int8, 1 thread, short state, p50 ms (Core Ultra 7 155H)
X86_REF = {4: 16, 20: 26, 77: 75, 150: 130}

# ─────────────────────────────── output ────────────────────────────────

C = {"b": "\033[1m", "d": "\033[2m", "g": "\033[32m", "r": "\033[31m", "y": "\033[33m", "c": "\033[36m", "x": "\033[0m"}
LOG: list[dict] = []


def c(s: str, k: str) -> str:
    return f"{C[k]}{s}{C['x']}" if C["x"] else s


def header(title: str, sub: str = "") -> None:
    print()
    print(c("━" * 78, "d"))
    print(c(f"  {title}", "b"))
    if sub:
        print(c(f"  {sub}", "d"))
    print(c("━" * 78, "d"))


def bar(p: float, width: int = 24) -> str:
    n = round(p * width)
    return "█" * n + c("·" * (width - n), "d")


def dist(x, top_k: int | None = None, indent: str = "    ") -> None:
    items = list(zip(x.choices, x.probs))
    if top_k:
        items = sorted(items, key=lambda t: -t[1])[:top_k]
    w = max(len(ch) for ch, _ in items)
    for ch, p in items:
        line = f"{indent}{ch:<{w}}  {bar(p)} {p:6.3f}"
        print(c(line, "b") if ch == x.top else line)


def mark(ok: bool) -> str:
    return c("✓", "g") if ok else c("✗", "r")


def decide(d, state: str, q, section: str, gold: str | None = None, repeats: int = 5):
    """Warm the option-set cache, then time `repeats` calls and keep the median latency."""
    d.decide(state, q)
    runs = [d.decide(state, q) for _ in range(repeats)]
    x = runs[-1]
    x.latency_ms = statistics.median(r.latency_ms for r in runs)
    LOG.append({"section": section, "state": state, "question": q.text, "kind": q.kind, "lang": q.lang,
                "n_options": len(q.choices), "gold": gold, "top": x.top, "confidence": round(x.confidence, 4),
                "probs": {ch: round(p, 6) for ch, p in zip(x.choices, x.probs)}, "latency_ms_p50": round(x.latency_ms, 1)})
    return x


# ─────────────────────────────── sections ──────────────────────────────


def s_routing(d, Q):
    header("1 · Support routing", f'"{TEAM_Q}"  options: {TEAMS}')
    q = Q(TEAM_Q, TEAMS, kind="choose", lang="en")
    for s in ROUTING:
        x = decide(d, s, q, "routing")
        print(f'\n  "{s}"   {c(f"{x.latency_ms:.0f} ms", "c")}')
        dist(x)


def s_multilingual(d, Q):
    header("2 · Same English options, seven languages", "the message changes language; the options and the code do not")
    n = ok = 0
    for gold, by_lang in MULTILINGUAL:
        print(f"\n  expected: {c(gold, 'b')}")
        for lang, s in by_lang.items():
            x = decide(d, s, Q(TEAM_Q, TEAMS, kind="choose", lang=lang), "multilingual", gold)
            n += 1
            ok += x.top == gold
            print(f"    {mark(x.top == gold)} [{lang}] {x.top:<18} {x.confidence:.3f}  {c(f'{x.latency_ms:4.0f} ms', 'c')}  {s}")
    print(f"\n  {ok}/{n} routed to the expected team")


def s_shuffle(d, Q):
    header("3 · The shuffle test", "reorder the options: the probabilities move with them, unchanged")
    s = ROUTING[0]
    orders = [TEAMS, TEAMS[::-1], ["sales", "account security", "billing", "technical support"]]
    for order in orders:
        x = decide(d, s, Q(TEAM_Q, order), "shuffle")
        print(f"\n  options in this order: {order}")
        dist(x)
    # bulk check: every ticket, 24 option orders each (all permutations of 4)
    import itertools

    flips = worst = 0
    total = 0
    for lang, t, _ in TICKETS:
        ref = d.decide(t, Q(TEAM_Q, TEAMS, lang=lang)).as_dict()
        for perm in itertools.permutations(TEAMS):
            x = d.decide(t, Q(TEAM_Q, list(perm), lang=lang))
            total += 1
            flips += x.top != max(ref, key=ref.get)
            worst = max(worst, max(abs(x.as_dict()[k] - ref[k]) for k in TEAMS))
    print(f"\n  all 20 tickets × all 24 orders = {total} decisions: answer changed {c(str(flips), 'b')} times; "
          f"largest probability difference {worst:.1e}")
    LOG.append({"section": "shuffle-bulk", "decisions": total, "answer_changes": flips, "max_abs_prob_diff": worst})


def s_threshold(d, Q):
    header(f"4 · Act only when confident: auto-route at ≥ {THRESHOLD}, escalate the rest", "20 tickets, 8 languages, labelled before the run")
    rows = []
    for lang, s, gold in TICKETS:
        x = decide(d, s, Q(TEAM_Q, TEAMS, kind="choose", lang=lang), "threshold", gold, repeats=3)
        auto = x.confidence >= THRESHOLD
        rows.append((x, gold, auto))
        tag = c("AUTO    ", "g") if auto else c("ESCALATE", "y")
        print(f"  {tag} {mark(x.top == gold)} {x.confidence:.3f} {x.top:<18} [{lang}] {s[:56]}")
    auto = [r for r in rows if r[2]]
    right = sum(r[0].top == r[1] for r in auto)
    allright = sum(r[0].top == r[1] for r in rows)
    print(f"\n  auto-routed {c(str(len(auto)), 'b')}/20, of which right {c(str(right), 'b')}/{len(auto)}; "
          f"escalated {20 - len(auto)} (all 20 at top-1: {allright}/20 right)")
    for th in (0.5, 0.7, 0.9):
        a = [r for r in rows if r[0].confidence >= th]
        print(c(f"    at {th}: auto {len(a)}/20, right {sum(r[0].top == r[1] for r in a)}/{len(a)}", "d"))
    LOG.append({"section": "threshold-summary", "threshold": THRESHOLD, "auto": len(auto), "auto_right": right, "top1_right": allright})


def s_clinc(d, Q):
    header("5 · 151 intents in one call (CLINC150 label set, incl. out-of-scope)", "no fine-tuning, no prompt to fill — the labels are just a list of strings")
    q = Q(CLINC_Q, CLINC, kind="choose", lang="en")
    for s, gold in CLINC_CASES:
        x = decide(d, s, q, "clinc", gold, repeats=3)
        top3 = sorted(zip(x.choices, x.probs), key=lambda t: -t[1])[:3]
        t3 = ", ".join(f"{ch} {p:.3f}" for ch, p in top3)
        print(f"  {mark(x.top == gold)} {x.confidence:.3f}  {c(f'{x.latency_ms:4.0f} ms', 'c')}  \"{s}\"")
        print(c(f"       top 3: {t3}", "d"))


def s_crosslingual(d, Q):
    header("6 · Persian in, English labels out (60 MASSIVE intents)", "the message is Persian; the 60 options are English intent names")
    q = Q(MASSIVE_Q, MASSIVE, kind="choose", lang="fa")
    for s, en, gold in PERSIAN_CASES:
        x = decide(d, s, q, "crosslingual", gold, repeats=3)
        print(f"  {mark(x.top == gold)} {s}  {c('(' + en + ')', 'd')}")
        print(f"      → {c(x.top, 'b')} {x.confidence:.3f}   {c(f'{x.latency_ms:.0f} ms', 'c')}")


def s_urgency(d, Q):
    header("7 · Ordered levels: the `score` kind", f'"{URGENCY_Q}"  levels: {URGENCY} (ordinal head)')
    q = Q(URGENCY_Q, URGENCY, kind="score", lang="en")
    for s in URGENCY_CASES:
        x = decide(d, s, q, "urgency")
        print(f'\n  "{s}"   {c(f"{x.latency_ms:.0f} ms", "c")}')
        dist(x)


def s_verify(d, Q):
    header("8 · Yes / no: the `verify` kind")
    for qt, s in VERIFY_CASES:
        x = decide(d, s, Q(qt, ["yes", "no"], kind="verify", lang="en"), "verify")
        print(f"  {qt:<38} {c(x.top, 'b'):>3} {x.confidence:.3f}   \"{s}\"")


def s_rank(d, Q):
    header("9 · Independent relevance per option: the `rank` kind", "each option gets its own probability; they need not sum to 1")
    x = decide(d, RANK_STATE, Q(RANK_Q, RANK, kind="rank", lang="en"), "rank")
    print(f'  "{RANK_STATE}"\n')
    for ch, p in sorted(zip(x.choices, x.probs), key=lambda t: -t[1]):
        print(f"    {ch:<44} {bar(p)} {p:.3f}")


def s_news(d, Q):
    header("10 · News topic (AG News label names)")
    q = Q(NEWS_Q, NEWS, kind="choose", lang="en")
    for s in NEWS_CASES:
        x = decide(d, s, q, "news")
        print(f"  {c(x.top, 'b'):<8} {x.confidence:.3f}  {s}")


def cpu_name() -> str:
    try:
        for line in open("/proc/cpuinfo"):
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def s_speed(d, Q):
    header("11 · Speed on this machine (int8, 1 thread, batch 1)",
           f"{cpu_name()} · {os.cpu_count()} logical CPUs")
    sets = {4: TEAMS, 20: CLINC[:20], 77: BANKING77, 150: CLINC[:150]}
    state = "I was charged twice for my subscription this month."
    print(f"  {'options':>7}  {'p50 ms':>7}  {'p95 ms':>7}  {'x86 laptop core (BENCH-x86)':>28}")
    for n, ch in sets.items():
        q = Q("What is the customer asking about?", ch, kind="choose", lang="en")
        t0 = time.perf_counter()
        d.decide(state, q)                       # first call: encodes and caches the option set
        cold = (time.perf_counter() - t0) * 1e3
        lat = sorted(d.decide(state, q).latency_ms for _ in range(30))
        p50, p95 = statistics.median(lat), lat[int(0.95 * len(lat)) - 1]
        print(f"  {n:>7}  {p50:>7.1f}  {p95:>7.1f}  {X86_REF[n]:>25} ms   {c(f'(first call with a new option set: {cold:.0f} ms)', 'd')}")
        LOG.append({"section": "speed", "n_options": n, "p50_ms": round(p50, 1), "p95_ms": round(p95, 1), "cold_ms": round(cold, 1)})
    print(c("\n  Reference: ≈ 15 ms + 1 ms per option on one x86 laptop core (docs/BENCH-x86.md).", "d"))


# ───────────── appendix A: rejected scenarios (not run by default; `run_demos.py rejected`) ─────────────

CHARGED_TWICE = {
    "fa": "این ماه دو بار از کارت من پول کم شده است.",
    "ar": "لقد تم خصم المبلغ مرتين من بطاقتي هذا الشهر.",
    "ru": "С моей карты дважды списали деньги в этом месяце.",
    "es": "Me cobraron dos veces este mes.",
    "de": "Mir wurde diesen Monat doppelt abgebucht.",
    "zh": "这个月我的卡被扣了两次款。",
    "fr": "On m'a débité deux fois ce mois-ci.",
    "tr": "Bu ay kartımdan iki kez ödeme alındı.",
    "hi": "इस महीने मेरे कार्ड से दो बार पैसे कट गए।",
    "ja": "今月、カードから二重に引き落とされました。",
}
# A1 with "subscription" kept and no "from my card" (native-level review, SENTENCES-REVIEW.md appendix A)
CHARGED_TWICE_FAITHFUL = {
    "fa": "این ماه دو بار بابت اشتراکم از من پول کم شده است.",
    "ar": "تم خصم رسوم اشتراكي مرتين هذا الشهر.",
    "ru": "В этом месяце с меня дважды списали деньги за подписку.",
    "es": "Este mes me cobraron dos veces la suscripción.",
    "de": "Mir wurde diesen Monat das Abo doppelt abgebucht.",
    "zh": "这个月我的订阅费被扣了两次。",
    "fr": "On m'a facturé deux fois mon abonnement ce mois-ci.",
}
# A7: weaknesses found in the language review (§1 options); expected team in the last field
LANG_WEAK = [
    ("fa", "loanword «اکانت» (account)", "یکی از یه کشور دیگه وارد اکانتم شده!", "account security"),
    ("fa", "same line with «حسابم»", "یکی از یه کشور دیگه وارد حسابم شده!", "account security"),
    ("fa", "§2 sales line, as shown", "اگر برای شرکتمان ۲۰۰ اشتراک بخریم تخفیف می‌دهید؟", "sales"),
    ("fa", "same with the optional comma", "اگر برای شرکتمان ۲۰۰ اشتراک بخریم، تخفیف می‌دهید؟", "sales"),
]
OFF_TOPIC = [
    ("en", "What's the best recipe for banana bread?"),
    ("en", "Can you recommend a good hotel in Lisbon?"),
    ("en", "Your office dog is adorable, give him a treat from me!"),
    ("en", "Who won the 1998 world cup?"),
    ("fa", "بهترین رستوران ایتالیایی تهران کجاست؟"),
    ("es", "¿Qué tiempo hará mañana en Madrid?"),
]
POLICY = "Refund policy: unused items can be returned for a refund within 30 days of purchase.\nCustomer: "
POLICY_CASES = [("I bought the headphones 12 days ago and never opened the box.", "yes"),
                ("I bought the headphones 45 days ago and never opened the box.", "no"),
                ("I've used them every day for three weeks and now I don't like them.", "no")]
STARS = ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]
STAR_CASES = [("Absolutely terrible. The blender broke after two days and support never answered.", "1 star"),
              ("It works, but it's louder than I expected and the lid is flimsy.", "2 stars or 3 stars"),
              ("Does the job. Nothing special, nothing wrong.", "3 stars"),
              ("Really good blender, smoothies come out smooth. Only wish it were a bit quieter.", "4 stars"),
              ("Best kitchen purchase I've made in years. Powerful, quiet, easy to clean. Love it!", "5 stars")]
SHELL = [("rm -rf /var/lib/postgresql/data", "yes"), ("ls -la /var/log", "no"), ("git push --force origin main", "yes"),
         ("cat README.md", "no"), ("dd if=/dev/zero of=/dev/sda bs=1M", "yes"), ("DROP TABLE users;", "yes")]


def s_rejected(d, Q):
    header("Appendix A · rejected scenarios", "reproduces SCENARIOS.md appendix A; none of these belong in a demo")
    print(c("\n  A1 · 'charged twice' in 10 languages (expected: billing)", "b"))
    for lang, s in CHARGED_TWICE.items():
        x = decide(d, s, Q(TEAM_Q, TEAMS, lang=lang), "rejected-A1", "billing", repeats=1)
        print(f"    {mark(x.top == 'billing')} [{lang}] {x.top:<18} {x.confidence:.3f}  billing={x.as_dict()['billing']:.3f}")
    print(c("  A1 · faithful translations (keep 'subscription', no 'from my card')", "d"))
    for lang, s in CHARGED_TWICE_FAITHFUL.items():
        x = decide(d, s, Q(TEAM_Q, TEAMS, lang=lang), "rejected-A1-faithful", "billing", repeats=1)
        print(f"    {mark(x.top == 'billing')} [{lang}] {x.top:<18} {x.confidence:.3f}  billing={x.as_dict()['billing']:.3f}")
    print(c("\n  A2 · adding a 'none of the above' option to the 4 teams", "b"))
    for extra in [None, "none of the above", "none of the above (out of scope)", "other"]:
        ch = TEAMS + ([extra] if extra else [])
        ok = sum(d.decide(s, Q(TEAM_Q, ch, lang=l)).top == g for l, s, g in TICKETS)
        if extra:
            caught = sum(d.decide(s, Q(TEAM_Q, ch, lang=l)).top == extra for l, s in OFF_TOPIC)
            print(f"    + {extra!r:36} 20 tickets right: {ok}/20   off-topic caught: {caught}/6")
        else:
            confs = [d.decide(s, Q(TEAM_Q, ch, lang=l)).confidence for l, s in OFF_TOPIC]
            print(f"    (no extra option){'':21} 20 tickets right: {ok}/20   off-topic top confidence: "
                  + ", ".join(f"{p:.2f}" for p in confs))
    print(c("\n  A3 · applying a numeric policy (verify)", "b"))
    for s, gold in POLICY_CASES:
        x = decide(d, POLICY + s, Q("Is the customer eligible for a refund?", ["yes", "no"], kind="verify"), "rejected-A3", gold, repeats=1)
        print(f"    {mark(x.top == gold)} {x.top:<3} {x.confidence:.3f}  {s}")
    print(c("\n  A4 · 1–5 star ratings (score)", "b"))
    for s, gold in STAR_CASES:
        x = decide(d, s, Q("How many stars would this customer give?", STARS, kind="score"), "rejected-A4", gold, repeats=1)
        print(f"    {mark(x.top in gold)} {x.top:<8} {x.confidence:.3f}  (expected {gold})  {s[:56]}")
    print(c("\n  A5 · 'is this shell command destructive?' — never demo or claim tool-risk gating", "b"))
    for s, gold in SHELL:
        x = decide(d, s, Q("Is this shell command destructive?", ["yes", "no"], kind="verify"), "rejected-A5", gold, repeats=1)
        print(f"    {mark(x.top == gold)} {x.top:<3} {x.confidence:.3f}  {s}")
    print(c("\n  A6 · borderline cases trimmed from kept scenarios", "b"))
    x = decide(d, "The blender stopped working after a week. Can you send me a replacement part?",
               Q("Does the customer ask for a refund?", ["yes", "no"], kind="verify"), "rejected-A6", "no", repeats=1)
    print(f"    verify, replacement request: {x!r}")
    x = decide(d, "Chipmaker shares jump 12% after it raises its full-year revenue forecast.", Q(NEWS_Q, NEWS), "rejected-A6", None, repeats=1)
    print(f"    news, chipmaker earnings:   {x!r}")
    gate = "Is this a request our software company's support team can help with?"
    p_in = [d.decide(s, Q(gate, ["yes", "no"], kind="verify", lang=l)).probs[0] for l, s, _ in TICKETS]
    p_off = [d.decide(s, Q(gate, ["yes", "no"], kind="verify", lang=l)).probs[0] for l, s in OFF_TOPIC]
    print(f"    verify gate p(yes): 20 real tickets median {statistics.median(p_in):.2f} (max {max(p_in):.2f}); "
          f"6 off-topic median {statistics.median(p_off):.2f}")
    print(c("\n  A7 · Persian loanword and punctuation sensitivity (§1 options)", "b"))
    for lang, what, s, gold in LANG_WEAK:
        x = decide(d, s, Q(TEAM_Q, TEAMS, lang=lang), "rejected-A7", gold, repeats=1)
        print(f"    {mark(x.top == gold)} [{lang}] {x.top:<18} {x.confidence:.3f}  p({gold})={x.as_dict()[gold]:.3f}  {what}: {s}")


SECTIONS = {"routing": s_routing, "multilingual": s_multilingual, "shuffle": s_shuffle, "threshold": s_threshold,
            "clinc": s_clinc, "crosslingual": s_crosslingual, "urgency": s_urgency, "verify": s_verify,
            "rank": s_rank, "news": s_news, "speed": s_speed, "rejected": s_rejected}
DEFAULT = [k for k in SECTIONS if k != "rejected"]


def block_network() -> None:
    import socket

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    def refuse(*a, **k):
        raise OSError("network disabled by run_demos.py --offline")

    socket.socket.connect = refuse          # type: ignore[method-assign]
    socket.create_connection = refuse       # type: ignore[assignment]
    socket.getaddrinfo = refuse             # type: ignore[assignment]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sections", nargs="*", help=f"any of: {' '.join(SECTIONS)} (default: all)")
    local = ROOT / "export/v1i-int8"
    ap.add_argument("--model", default=str(local) if local.exists() else "amyrmahdy/decima-small",
                    help="local export dir or HF repo id (default: the local export if present, else the published "
                         "model; with --offline, run once online first so the Hub cache is filled)")
    ap.add_argument("--pause", action="store_true", help="wait for Enter between sections")
    ap.add_argument("--offline", action="store_true", help="disable all network access in this process")
    ap.add_argument("--no-color", action="store_true")
    ap.add_argument("--json", help="write every decision to this file")
    a = ap.parse_args()
    unknown = [s for s in a.sections if s not in SECTIONS]
    if unknown:
        ap.error(f"unknown section(s) {unknown}; choose from {list(SECTIONS)}")
    if a.no_color or not sys.stdout.isatty():
        for k in C:
            C[k] = ""
    if a.offline:
        block_network()
        print(c("network: disabled for this process (sockets refuse to connect)", "y"))

    from decima import Decima, Question

    t0 = time.perf_counter()
    d = Decima.from_pretrained(a.model, threads=1)
    print(c(f"loaded {a.model} in {time.perf_counter() - t0:.1f} s · one CPU thread · int8 ONNX", "d"))
    random.seed(0)
    for i, name in enumerate(a.sections or DEFAULT):
        if a.pause and i:
            input(c("\n  [Enter] ", "d"))
        SECTIONS[name](d, Question)
    if a.json:
        Path(a.json).write_text(json.dumps(LOG, ensure_ascii=False, indent=1))
        print(c(f"\nwrote {len(LOG)} records to {a.json}", "d"))


if __name__ == "__main__":
    main()
