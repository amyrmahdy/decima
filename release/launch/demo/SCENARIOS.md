# Decima-small — launch demo scenarios (verified)

Every output below is **real**: Decima-small int8 ONNX (`export/v1i-int8`, run v1i), one CPU thread,
batch 1, run on 2026-09-27 with `release/launch/demo/run_demos.py`. Rerun it and you get the same
probabilities; latencies depend on the machine.

```bash
uv run python release/launch/demo/run_demos.py                  # all kept scenarios (≈ 1 min)
uv run python release/launch/demo/run_demos.py --offline        # same, with networking disabled in-process
uv run python release/launch/demo/run_demos.py --pause          # Enter between sections (screen recording)
uv run python release/launch/demo/run_demos.py rejected         # appendix A
```

**Rules we followed.** Every input (tickets, headlines, utterances) was written *before* the model saw it and
was not reworded afterwards — with one documented exception: the non-English §2 lines were revised for
naturalness after a native-level language review ([SENTENCES-REVIEW.md](SENTENCES-REVIEW.md), which shows the
model's output on the old and new wording), and a fourth informal §2 message was added from that review. The expected answer for the 20-ticket set was written down first. Scenarios where
the model was wrong or unconvincing were dropped and are listed in [appendix A](#appendix-a--rejected-scenarios)
with their real outputs, so nobody can say we hid them.

**About the latencies in this file.** They were measured on the GX10's Grace CPU (aarch64, one thread),
with other work running on the same box. They are here to show the relative cost
of each scenario, **not** for public quoting. Public speed claims use only docs/BENCH-x86.md (Intel Core
Ultra 7 155H, one P-core): **~20 ms per decision (int8, 4 options, short input); with a short state, 16 ms at
4 options, ~1 ms per extra option, 1.06 s at 1,000.**

| # | scenario | result | kept for |
|---|---|---|---|
| 1 | Support routing (EN) | 4/4, confidence 0.887–0.985 | every format |
| 2 | Same English options, 7 languages | 28/28, confidence ≥ 0.912 | every format |
| 3 | The shuffle test | 0 answer changes in 480 decisions; max Δp 1.2 × 10⁻⁷ | every format (hero) |
| 4 | Confidence threshold on 20 tickets | ≥ 0.8: 14 auto-routed, 14/14 right; 6 escalated incl. the only error | article, long video, HN |
| 5 | 151 intents in one call (CLINC150 labels) | 7/8; the miss has confidence 0.200 | long video, article 2 |
| 6 | Persian message → 60 English intent names | 6/6 | reel, short, long video |
| 7 | Ordered levels (`score`): ticket urgency | sensible ordering, honest uncertainty | long video |
| 8 | Yes/no (`verify`) on simple properties | 4/4, 0.899–0.983 | long video |
| 9 | `rank`: which help articles are relevant | right article on top (0.936), irrelevant ≤ 0.036 | long video |
| 10 | News topic | 4/4, 0.679–0.977 | long video (brief) |
| 11 | Speed vs number of options | local p50 + BENCH-x86 reference | every format (with x86 numbers) |

---

## 1 · Support routing (EN)

**Goal.** The one-glance "what is this" demo: free-text options, probabilities out.

**Input.** Question `"Which team should handle this request?"`, kind `choose`, options
`["billing", "technical support", "sales", "account security"]`.

| message | output (rounded as `repr` prints it) | p50 (GX10) |
|---|---|---:|
| Someone logged into my account from another country. | `{'billing': 0.004, 'technical support': 0.033, 'sales': 0.001, 'account security': 0.962}` | 9 ms |
| I was charged twice for my subscription this month. | `{'billing': 0.887, 'technical support': 0.053, 'sales': 0.019, 'account security': 0.041}` | 10 ms |
| The app crashes every time I open the settings page. | `{'billing': 0.007, 'technical support': 0.972, 'sales': 0.014, 'account security': 0.008}` | 10 ms |
| Do you offer a discount if we buy 200 seats for our company? | `{'billing': 0.012, 'technical support': 0.001, 'sales': 0.985, 'account security': 0.001}` | 10 ms |

**Proves.** Options are plain strings chosen at call time; no training, no label set, no prompt.


## 2 · Same English options, seven languages

**Goal.** Multilingual routing with *one* option set in English; the incoming message changes language.

**Input.** Same question and options as §1; `lang` set to the message language. Four messages, each in
EN / FA / AR / RU / ES / DE / ZH. Translations written by us, then checked by a native-level reviewer
([SENTENCES-REVIEW.md](SENTENCES-REVIEW.md), decisions applied 2026-09-27): six lines were reworded to sound
natural (fa ×2, ar ×2, zh, es), and the fourth, informal row was added.

| expected | en | fa | ar | ru | es | de | zh |
|---|---:|---:|---:|---:|---:|---:|---:|
| account security ("someone logged in from another country") | 0.962 | 0.956 | 0.981 | 0.921 | 0.927 | 0.952 | 0.973 |
| technical support ("app crashes when I open settings") | 0.972 | 0.982 | 0.995 | 0.980 | 0.993 | 0.983 | 0.982 |
| sales ("discount for 200 seats?") | 0.985 | 0.912 | 0.987 | 0.990 | 0.964 | 0.973 | 0.991 |
| account security, informal ("…logged into my account from another country and it wasn't me!") | 0.982 | 0.978 | 0.962 | 0.981 | 0.984 | 0.991 | 0.970 |

All 28 routed to the expected team (the cell is the probability of the expected team, which was the top
answer every time). p50 9–12 ms on the GX10. The exact sentences are in `run_demos.py` (`MULTILINGUAL`).

**Known wording traps (not for the demo, A7).** The Persian informal line uses «حسابم» on purpose: with the
common loanword «اکانت» (`یکی از یه کشور دیگه وارد اکانتم شده!`) the model answers technical support 0.519
(account security 0.209); with «حسابم» it is account security 0.924. The Persian sales line has no comma after
the condition; the grammatically optional comma drops it from 0.912 to 0.837.

**Proves.** Cross-lingual decisions with no translation step and no per-language setup.


## 3 · The shuffle test (hero demo)

**Goal.** Show that option order cannot change the answer — the four open decision models we compared
changed their answer on 10–27 % of shuffled questions in our evaluation (mean over suites).

**Input.** "Someone logged into my account from another country." with the §1 options in three orders:

```
order: billing, technical support, sales, account security   → account security 0.962, technical support 0.033, billing 0.004, sales 0.001
order: account security, sales, technical support, billing   → account security 0.962, technical support 0.033, billing 0.004, sales 0.001
order: sales, account security, billing, technical support   → account security 0.962, technical support 0.033, billing 0.004, sales 0.001
```

To six decimals, in every order: account security 0.962196, technical support 0.033204, billing 0.003557,
sales 0.001044.

**Bulk check (also in the script).** All 20 tickets from §4 × all 24 orders of the 4 options = 480
decisions: the answer changed **0** times; the largest probability difference was **1.2 × 10⁻⁷**.

**Proves.** Permutation invariance is architectural: each option is scored against the message on its
own, so its position cannot enter the computation.


## 4 · Act only when confident (20 tickets)

**Goal.** Show what calibrated probabilities are *for*: automate the confident cases, send the rest to
a human.

**Input.** 20 realistic tickets, 8 languages (EN ×12, ES, DE, FR, RU, FA, AR, ZH, TR), expected team written
down before the run; §1 question and options; threshold 0.8.

| decision | ticket (abridged) | top | conf | right? |
|---|---|---|---:|---|
| escalate | My invoice shows the wrong VAT number, can you reissue it? | billing | 0.457 | ✓ |
| auto | I got a password reset email I never asked for… | account security | 0.856 | ✓ |
| auto | The dashboard has been loading forever since your update… | technical support | 0.945 | ✓ |
| auto | We're a team of 40 and want to know what the enterprise plan includes. | sales | 0.947 | ✓ |
| auto | Why was I charged $49 when my plan is $29? | billing | 0.941 | ✓ |
| auto | The API returns a 500 error whenever I upload a file larger than 10 MB. | technical support | 0.992 | ✓ |
| auto | Can I get a demo for my manager next week? | sales | 0.805 | ✓ |
| auto | Please turn on two-factor authentication… my email was hacked. | account security | 0.987 | ✓ |
| escalate | How do I update the credit card you charge every month? | billing | 0.671 | ✓ |
| auto | Is there a nonprofit discount? | sales | 0.953 | ✓ |
| auto | [es] No puedo sincronizar mis archivos desde ayer… | technical support | 0.987 | ✓ |
| escalate | [de] Ich möchte meine Rechnung als PDF bekommen, nicht per Post. | billing | 0.491 | ✓ |
| auto | [fr] Quelqu'un a changé l'adresse e-mail de mon compte… | account security | 0.903 | ✓ |
| auto | [ru] Сколько будет стоить годовая подписка для 15 сотрудников? | sales | 0.961 | ✓ |
| escalate | [fa] رمز عبورم را عوض نکرده‌ام ولی دیگر نمی‌توانم وارد حسابم شوم. | account security | 0.675 | ✓ |
| escalate | [ar] الإشعارات لا تصل إلى هاتفي منذ التحديث الأخير. | technical support | 0.692 | ✓ |
| auto | [zh] 我想把按月付费改成按年付费。 | billing | 0.848 | ✓ |
| auto | [tr] Şirketimiz için özel bir fiyat teklifi alabilir miyiz? | sales | 0.993 | ✓ |
| auto | I cancelled last month but you charged me again. | billing | 0.936 | ✓ |
| escalate | After I changed my phone, the authenticator codes stopped working and I'm locked out. | technical support | 0.641 | **✗** (expected account security) |

**Result.** Top-1 right on 19/20. At ≥ 0.8: **14 auto-routed, 14/14 right; 6 escalated, including the one
wrong answer.** Other thresholds: ≥ 0.5 → 18 auto, 17/18 right; ≥ 0.7 → 14, 14/14; ≥ 0.9 → 11, 11/11.

**Proves.** On this set, the model's confidence separates the answers you can automate from the ones you
should not. This is the practical face of the calibration numbers (figure `calibration.png`; if you show
`calibration_reliability.png`, its caveat "Decima trained on the MASSIVE and XNLI/MNLI train splits; Laya
reports it did not" must be on screen).


## 5 · 151 intents in one call (CLINC150 label set)

**Goal.** A big, real label set passed as a Python list — the case where prompt-based classifiers run out
of context or slow down.

**Input.** Question `"What is the user's intent? (may be out of scope)"`; the 151 CLINC150 intent names
(incl. `"none of the above (out of scope)"`); new utterances written by us.

| utterance | top (conf) | runner-up | p50 (GX10) |
|---|---|---|---:|
| can you move my dentist reminder to friday at 3 | reminder update (0.943) | calendar update 0.021 | 183 ms |
| how many calories are in a banana | calories (0.994) | none of the above 0.001 | 181 ms |
| my card got declined at the grocery store | card declined (0.993) | shopping list update 0.000 | 166 ms |
| what's the plug type in japan | plug type (0.994) | none of the above 0.001 | 165 ms |
| i need to rent a car in denver next weekend | car rental (0.981) | tire change 0.001 | 165 ms |
| how long will it take for my new card to arrive | replacement card duration (0.959) | order status 0.020 | 165 ms |
| who won the 1998 world cup | none of the above (out of scope) (0.954) | yes 0.004 | 163 ms |
| write me a poem about the sea | **tell joke (0.200)** ✗ | fun fact 0.149; out of scope 0.038 | 162 ms |

**Proves.** Large free-text option sets work in one call, and a miss can come with an honest, low
confidence (0.200) that a threshold would catch.


## 6 · Persian message → 60 English intent names

**Goal.** Cross-lingual: the user writes Persian, the app's labels stay English.

**Input.** Question `"What does the user want the assistant to do?"`, `lang="fa"`, the 60 MASSIVE intent
names in English.

| Persian (meaning) | top (conf) | runner-up | p50 (GX10) |
|---|---|---|---:|
| فردا ساعت هفت صبح بیدارم کن (wake me up at seven tomorrow) | alarm set (0.851) | alarm query 0.062 | 43 ms |
| چراغ‌های آشپزخانه را خاموش کن (turn off the kitchen lights) | iot hue lightoff (0.935) | iot wemo off 0.024 | 42 ms |
| هوای تهران فردا چطور است؟ (weather in Tehran tomorrow?) | weather query (0.906) | news query 0.019 | 42 ms |
| یک تاکسی برای فرودگاه بگیر (get me a taxi to the airport) | transport taxi (0.981) | transport ticket 0.010 | 43 ms |
| صدای موزیک را کم کن (turn the music down) | audio volume down (0.654) | audio volume up 0.200 | 42 ms |
| یه جوک برام بگو (tell me a joke — colloquial) | general joke (0.991) | general quirky 0.006 | 42 ms |

**Proves.** State and options in different languages, no translation layer.


## 7 · Ordered levels (`score`): ticket urgency

**Goal.** Show the ordinal kind: probabilities that respect the order of the levels.

**Input.** `"How urgent is this ticket?"`, kind `score`, levels `["low", "medium", "high", "critical"]`.

| ticket | low | medium | high | critical |
|---|---:|---:|---:|---:|
| Small typo on your pricing page: 'monthy' instead of 'monthly'. | **0.744** | 0.224 | 0.027 | 0.005 |
| Could you change the font on my invoices sometime? No rush. | **0.646** | 0.324 | 0.028 | 0.003 |
| The export to CSV is slow, it takes a few minutes. | 0.048 | **0.456** | 0.441 | 0.054 |
| I can't log in since this morning and I have a client demo in an hour. | 0.013 | 0.090 | **0.453** | 0.444 |
| Our checkout page is down and customers can't pay. We're losing sales every minute. | 0.003 | 0.024 | 0.292 | **0.681** |

p50 9–11 ms (GX10).

**Proves.** The mass sits on adjacent levels, never split between "low" and "critical"; genuinely
borderline tickets come out borderline.


## 8 · Yes / no (`verify`) on simple properties

| question | message | output |
|---|---|---|
| Does the customer ask for a refund? | The blender stopped working after a week. I want my money back. | yes 0.954 |
| Does the customer ask for a refund? | Great blender, just wanted to say thanks! | no 0.983 |
| Is the customer angry? | This is the THIRD time I'm writing. Nobody answers. Unacceptable. | yes 0.924 |
| Is the customer angry? | Hi, quick question about my invoice format, no rush. | no 0.899 |

p50 7–9 ms (GX10).

**Proves.** One-property checks on a message.

## 9 · `rank`: which help articles are relevant

**Input.** "Someone logged into my account from another country and I can't get back in." —
`"Which of these help articles are relevant to the customer's problem?"`, kind `rank`.

```
What to do if you see an unfamiliar login   0.936
How to reset your password                  0.794
Enabling two-factor authentication          0.424
Exporting reports to CSV                    0.036
Updating your billing address               0.024
```

p50 15 ms (GX10). **Proves.** Independent per-option probabilities (they do not sum to 1), useful for
picking several.

## 10 · News topic (AG News label names)

| headline (written by us) | output |
|---|---|
| Central bank holds interest rates steady as inflation cools for a third month. | Business 0.934 |
| Striker scores twice in stoppage time to send her club into the cup final. | Sports 0.922 |
| Researchers unveil a battery chemistry that charges an electric car in ten minutes. | Sci/Tech 0.679 |
| Ceasefire talks resume in Geneva as both delegations arrive for a second round. | World 0.977 |

p50 11 ms (GX10). Brief B-roll only; AG News is in-distribution. A chipmaker-earnings headline split
Business 0.535 / Sci/Tech 0.422 (A6) — a fair split, but not a demo.

## 11 · Speed vs number of options

**Measured here (GX10 Grace core, int8, 1 thread, short state, cached option set, 30 calls; a GPU training
run on 2026-09-27):** 4 options 8.3 ms · 20 options 17.4 ms · 77 options 55.0 ms · 150 options 111.8 ms
(p50). A new option set costs extra once (e.g. ≈ 0.9 s to encode 150 options).

**What to say publicly (docs/BENCH-x86.md, Intel Core Ultra 7 155H, one P-core):** ≈ **20 ms** per decision
with 4 options (p95 23 ms); 42 ms at 20; 83 ms at 77; with short intent labels, 16 ms at 4 options, ~1 ms per
extra option, 1.06 s at 1,000. Under 50 ms holds up to roughly 20–40 options depending on option length.
Figure: `speed_vs_options.png`. For the laptop-with-Wi-Fi-off shot, run `run_demos.py --offline` on the x86
laptop — it disables sockets in-process and prints its own timings; quote what the screen shows.


---

## Appendix A — rejected scenarios

Reproduce with `run_demos.py rejected`. None of these may appear in a demo, video or post.

**A1 · "I was charged twice this month" in 10 languages** (expected billing). Right in **4/10**
(ar 0.705, es 0.798, tr 0.594, de 0.365 — barely); wrong in fa, ru, zh, hi, ja (→ account security,
0.465–0.662) and fr (→ technical support 0.553). The English version is fine (billing 0.887). Every wrong
answer is ≤ 0.662, so a 0.8 threshold escalates all six.
**Part of this was our translation, not the model.** The English source says "charged twice *for my
subscription*"; every translation dropped "subscription", and most said "money was taken *from my card*"
(fa, ru, zh, …), which does read like possible fraud. Faithful versions from the native-level review
(SENTENCES-REVIEW.md; also in `run_demos.py rejected`) are better but still weak: right in 5 of 7 (fa 0.619,
ar 0.797, ru 0.554, es 0.859, zh 0.836), mostly below 0.8 (only es and zh reach it); German «das Abo doppelt
abgebucht» fails (→ technical support 0.357) and so does French (→ account security 0.450). **Stays rejected.**

**A2 · "none of the above" as an extra option.** It catches off-topic messages (6/6: recipes, hotels,
football, Persian restaurant, Madrid weather) but **pulls real tickets into it**: with
`"none of the above"` the 20 tickets drop from 19/20 to **11/20** right (NOTA at up to 0.933 on a genuine
notification bug); `"other"` 11/20; `"none of the above (out of scope)"` 16/20. Without a none option the six
off-topic messages get top confidence 0.38–0.81 — one (Madrid weather → technical support 0.812) would pass
a 0.8 threshold. A verify gate ("Is this a request our support team can help with?") does not work either:
real tickets median p(yes) 0.21. Out-of-scope handling for small option sets is a real gap.

**A3 · Applying a written policy** ("refunds within 30 days for unused items"). Bought 45 days ago →
yes 0.785; used daily for three weeks → yes 0.698. Rule application with numbers is multi-fact reasoning,
a documented weakness.

**A4 · 1–5 star ratings (`score`).** "Does the job. Nothing special, nothing wrong." → 4 stars 0.583
(expected 3); a furious review is only 0.552 on 1 star. Consistent with SST-5 ≈ 0.49–0.51. Replaced by the
urgency demo (§7), which claims nothing.

**A5 · "Is this shell command destructive?"** `rm -rf /var/lib/postgresql/data` → **no** 0.693;
`git push --force origin main` → **no** 0.704. Never demo, imply or claim tool-risk / agent-safety gating.

**A6 · Borderline cases trimmed from kept scenarios.** Verify "does the customer ask for a refund?" on a
replacement-part request: yes 0.488 / no 0.512 (right, not convincing). Chipmaker-earnings headline:
Business 0.535 / Sci/Tech 0.422.

**A7 · Wording traps found in the language review (Persian, §1 options).** The colloquial loanword «اکانت»
("account"): `یکی از یه کشور دیگه وارد اکانتم شده!` → **technical support 0.519** (account security 0.209);
the same line with «حسابم» → account security 0.924. Optional punctuation: the §2 Persian sales line gives
sales 0.912; with the grammatically optional comma after the condition it drops to 0.837. Both are recorded
as limitations in the model card.

**Not attempted as demos** (documented weaknesses, docs/EVAL.md §5): long multi-fact business decisions
(JevBench, typed-decisions), knowledge questions, English NLI against Laya.
