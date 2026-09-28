# Demo sentences — native-level language review

> **Decisions applied (2026-09-27).** The owner approved: the six §2 rewrites marked *use proposed* /
> *use proposed (optional)* (fa 2a «حسابم», fa 2b reordered, ar 2a «أحدهم … دولة أخرى», ar 2b «يُغلق فجأة كلما»,
> zh 2a «国外 … 账号», es 2b «se cierra sola»); the 4th §2 row "…and it wasn't me!" in all 7 languages (the
> **var** rows marked *add (4th row)*); every other original kept (§4 tickets, §6 Persian, no «آهنگ» change,
> English §7 unchanged); the «اکانت» loanword and Persian comma sensitivity recorded as limitations (model card,
> TECHNICAL-REPORT §10); A1's explanation corrected in SCENARIOS.md. Applied in `run_demos.py`, SCENARIOS.md
> (§2 now 28/28, new appendix A7), the Space gallery (rebuilt with `_build/build_gallery.py`, new card
> `route-fa-informal`), both READMEs and the video scripts. Every output was re-run on `export/v1i-int8`, one
> thread; all reproduce the numbers below. The text below is the review as submitted and keeps the old
> sentences for the record.

Reviewer pass over every non-English input in `run_demos.py` (source of truth), SCENARIOS.md and the Space
files `release/space/examples/gallery.json` + `tickets.json`. Business cases skipped as asked. **Nothing has
been edited yet — the owner decides.**

## Summary for the owner (10 lines)

1. Reviewed **41 non-English sentences** (FA 12, AR 5, RU 5, ES 6, DE 5, ZH 5, FR 2, TR 1) and ~40 English ones. Orthography is clean: Persian uses ی/ک (no Arabic ي/ك), ZWNJ is correct everywhere (می‌کنم، چراغ‌ها، نکرده‌ام), digits are consistent per sentence.
2. Overall the set is already good — **no blocking errors, nothing a native would laugh at.** Most sentences are "correct but slightly written/formal"; I kept those.
3. **Proposed edits (text only): FA 2 (+1 optional), AR 2, ZH 1, ES 1, DE 0, RU 0, EN 1.** Every proposed edit keeps the right answer, and every proposed sentence scores ≥ 0.956.
4. **§4 tickets: keep all 8 non-English as they are.** They are natural, and SCENARIOS.md promises "not reworded afterwards".
5. **§6 Persian: keep all 6.** The "turn the music down" line (0.654) is the weakest. Rewording doesn't help it (0.58–0.68 for every phrasing), so leave it out of the reel/short.
6. **Enhancement:** an informal "…and it wasn't me!" account-security line works in all 7 languages (0.962–0.991). It's a good optional 4th row for §2.
7. **Do NOT use** the Persian informal line with **«اکانت»**: `یکی از یه کشور دیگه وارد اکانتم شده!` → *technical support 0.519* (wrong). The same line with «حسابم» → 0.924. Real Iranians write «اکانت» all the time, so this is a real weakness to record, not a sentence problem.
8. The model is **sensitive to Persian punctuation**. Adding the grammatically optional comma to the sales line drops it from 0.912 to 0.837. Keep the original.
9. **A1 ("charged twice") was partly our translation's fault.** The translations dropped "for my subscription" and said "money was taken from my card", which reads as fraud. Faithful versions get 5 of the 6 reviewed languages right (fa, ar, ru, es, zh), but only es/zh reach ≥ 0.8 and DE still fails ("Abo"). Keep A1 rejected, but correct its explanation.
10. Persian digits vs Latin digits (۲۰۰ / 200) and Arabic-Indic vs Latin (٢٠٠ / 200) give **identical** outputs, so the digit style is purely editorial.

**Method.** Every original and every proposed/variant sentence was re-run on `export/v1i-int8`, 1 CPU thread,
with the same question, options and `lang` as the scenario. The cells show top answer and its probability
(for a wrong answer, p(expected) is in brackets). Originals reproduce SCENARIOS.md exactly. Scripts:
scratchpad `review_run.py` / `diag.py`; they are not in the repo.
Legend: **prop** = replaces the original, **var** = optional extra sentence, **test** = diagnostic only, never for a demo.

---

## §2 · Same English options, seven languages
Question `"Which team should handle this request?"`, options `billing / technical support / sales / account security`.
The Space gallery reuses six of these (route-fa, route-ar, route-zh, route-ru, route-es, route-de), so the same decision applies there.

### 2a · account security — "Someone logged into my account from another country."

| lang | original | proposed / variant | why | model: original → proposed | recommendation |
|---|---|---|---|---|---|
| fa | یک نفر از کشور دیگری وارد حساب من شده است. | **prop** یک نفر از کشور دیگری وارد حسابم شده است. | Correct already. «حسابم» is what people actually write; «حساب من» sounds translated. Minor. | account security 0.955 → 0.956 | use proposed (optional) |
| fa | — | **var** یه نفر از یه کشور دیگه وارد حسابم شده، کار من نبوده! | Everyday chat/ticket register ("…it wasn't me!"). | — → account security 0.978 | add (if the §2 4th row is used) |
| fa | — | **test** یکی از یه کشور دیگه وارد اکانتم شده! | «اکانت» is the most common colloquial word for account. | — → **technical support 0.519 ✗** (p(acc. sec.) 0.209); same sentence with «حسابم»: 0.924 ✓ | **do not use**; record the loanword weakness |
| fa | — | **test** یک نفر از کشور دیگری وارد حساب کاربری‌ام شده است. | Formal «حساب کاربری». | — → account security 0.881 | not needed |
| ar | شخص ما سجّل الدخول إلى حسابي من بلد آخر. | **prop** أحدهم سجّل الدخول إلى حسابي من دولة أخرى. | «شخص ما» is correct MSA but reads like a calque of "someone"; «أحدهم» is what people write. «دولة» is more usual than «بلد» in this context. | account security 0.969 → 0.981 | use proposed |
| ar | — | **var** وصلني تنبيه بتسجيل دخول إلى حسابي من دولة أخرى، ولم أكن أنا. | Realistic ("I got a login alert … it wasn't me"). Dialect-neutral. | — → account security 0.962 | add (4th row) |
| ru | Кто-то вошёл в мой аккаунт из другой страны. | keep | Natural. | account security 0.921 | keep original |
| ru | — | **var** Кто-то зашёл в мой аккаунт из другой страны, это был не я! | «зашёл» is the everyday verb. | — → account security 0.981 | add (4th row) |
| es | Alguien inició sesión en mi cuenta desde otro país. | keep | Natural and neutral across Spain/LatAm. | account security 0.927 | keep original |
| es | — | **var** Alguien entró en mi cuenta desde otro país y no fui yo. | Informal («entró en»; LatAm would say «entró a»). | — → account security 0.984 | add (4th row) |
| de | Jemand hat sich aus einem anderen Land in mein Konto eingeloggt. | keep | Natural colloquial German. | account security 0.952 | keep original |
| de | — | **var** Da hat sich jemand aus dem Ausland in mein Konto eingeloggt, das war ich nicht! | «aus dem Ausland» is the idiomatic way to say it. | — → account security 0.991 | add (4th row) |
| zh | 有人从另一个国家登录了我的账户。 | **prop** 有人从国外登录了我的账号。 | «另一个国家» is translation-ese. Users say «国外». For app accounts «账号» is far more common than «账户» (which suggests a bank account). | account security 0.985 → 0.973 | use proposed |
| zh | — | **var** 我的账号被人在国外登录了，不是我本人操作的！ | Very typical complaint phrasing. | — → account security 0.970 | add (4th row) |
| en | — | **var** Someone logged into my account from another country and it wasn't me! | English counterpart for the 4th row. | — → account security 0.982 | add (4th row) |

### 2b · technical support — "The app crashes every time I open the settings page."

| lang | original | proposed / variant | why | model: original → proposed | recommendation |
|---|---|---|---|---|---|
| fa | برنامه هر بار که صفحه تنظیمات را باز می‌کنم بسته می‌شود. | **prop** هر بار که صفحهٔ تنظیمات را باز می‌کنم، برنامه بسته می‌شود. | The original puts the subject far from its verb and has no comma, so it reads clumsily. The fronted clause + comma is the natural order. Ezafe «صفحهٔ» is optional. | technical support 0.992 → 0.982 | use proposed |
| fa | — | **var** هر بار تنظیمات رو باز می‌کنم، اپ کرش می‌کنه. | How users actually type it («اپ», «کرش می‌کنه»). | — → technical support 0.953 | add (optional) |
| ar | التطبيق يتوقف في كل مرة أفتح فيها صفحة الإعدادات. | **prop** التطبيق يُغلق فجأة كلما فتحت صفحة الإعدادات. | «يتوقف» = "stops/freezes", not quite "crashes". «يُغلق فجأة» carries the crash meaning. «كلما» is lighter than «في كل مرة … فيها». | technical support 0.994 → 0.995 | use proposed |
| ar | — | **var** كلما فتحت الإعدادات يتعطّل التطبيق ويُغلق من تلقاء نفسه. | Alternative, a little more descriptive. | — → technical support 0.950 | optional |
| ru | Приложение вылетает каждый раз, когда я открываю настройки. | keep | Natural («вылетает» is exactly right). | technical support 0.980 | keep original |
| ru | — | **var** Приложение вылетает, как только захожу в настройки. | Shorter, everyday. | — → technical support 0.987 | optional |
| es | La aplicación se cierra cada vez que abro la configuración. | **prop** La aplicación se cierra sola cada vez que abro la configuración. | Without «sola» it can read as normal closing. «se cierra sola» is the idiomatic way to say crash. | technical support 0.993 → 0.993 | use proposed |
| es | — | **var** La app se me cierra cada vez que entro en los ajustes. | Informal, Spain («ajustes»). | — → technical support 0.972 | optional |
| de | Die App stürzt jedes Mal ab, wenn ich die Einstellungen öffne. | keep | Perfect. | technical support 0.983 | keep original |
| de | — | **var** Jedes Mal, wenn ich die Einstellungen aufmache, stürzt die App ab. | Spoken register. | — → technical support 0.960 | optional |
| zh | 每次打开设置页面，应用就会崩溃。 | keep | Correct and fine. «崩溃» is slightly technical. | technical support 0.982 | keep original |
| zh | — | **var** 一打开设置页面，App就闪退。 | «闪退» is *the* everyday word for an app crash. | — → technical support 0.918 | optional |

### 2c · sales — "Do you offer a discount if we buy 200 seats for our company?"

| lang | original | proposed / variant | why | model: original → proposed | recommendation |
|---|---|---|---|---|---|
| fa | اگر برای شرکتمان ۲۰۰ اشتراک بخریم تخفیف می‌دهید؟ | **test** …بخریم، تخفیف می‌دهید؟ (comma added) | Comma after the condition is standard but optional. | sales 0.912 → **0.837** | **keep original** (comma costs 0.075) |
| fa | — | **var** اگه برای شرکتمون ۲۰۰ تا لایسنس بگیریم، تخفیف دارید؟ | Colloquial («لایسنس», «۲۰۰ تا»). | — → sales 0.816 | skip (weaker than original) |
| fa | — | **test** same as original with Latin «200» | Digit style check. | sales 0.837 (= Persian-digit comma version, identical) | digits don't matter |
| ar | هل تقدمون خصماً إذا اشترينا 200 ترخيص لشركتنا؟ | keep | Correct («200 ترخيصٍ», singular after the hundreds). «رخصة» is more common, but «ترخيص» is fine. | sales 0.987 | keep original |
| ar | — | **var** نريد شراء 200 رخصة لشركتنا، هل يوجد خصم؟ | Natural customer order (state the need, then ask). | — → sales 0.978 | optional |
| ar | — | **test** original with «٢٠٠» | Arabic-Indic digits. | sales 0.987 (identical) | digits don't matter |
| ru | Есть ли скидка, если мы купим 200 лицензий для компании? | keep | Natural. | sales 0.990 | keep original |
| ru | — | **var** Если возьмём 200 лицензий на компанию, скидку дадите? | Informal. | — → sales 0.930 | optional |
| es | ¿Hay descuento si compramos 200 licencias para nuestra empresa? | keep | Natural. | sales 0.964 | keep original |
| es | — | **var** Somos una empresa y queremos 200 licencias, ¿nos hacen algún descuento? | Very typical sales-inquiry phrasing. | — → sales 0.967 | optional |
| de | Gibt es Rabatt, wenn wir 200 Lizenzen für unsere Firma kaufen? | keep («einen Rabatt» optional) | «Gibt es Rabatt» is fine in speech; «einen Rabatt» is more standard in writing. | sales 0.973 → 0.972 (with «einen») | keep original |
| de | — | **var** Gibt es Mengenrabatt, wenn wir 200 Lizenzen für unsere Firma kaufen? | «Mengenrabatt» (volume discount) is what a German buyer writes. | — → sales 0.924 | optional |
| de | — | **test** Bekommen wir Mengenrabatt, wenn wir 200 Lizenzen für unsere Firma nehmen? | Equally natural. | — → sales **0.706** (billing 0.160) | don't use |
| zh | 如果我们公司买200个席位，有折扣吗？ | keep | «席位» is standard SaaS wording (used by Chinese SaaS vendors). | sales 0.991 | keep original |
| zh | — | **var** 我们公司想买200个账号，能打折吗？ | How a non-technical buyer asks. | — → sales 0.909 | optional |

**§2 bottom line.** With the proposals, the 21 cells stay 21/21 and the minimum is 0.912 (unchanged; FA sales
is kept as is). The optional 4th row ("…and it wasn't me!", all 7 languages): 7/7, 0.962–0.991, using the
«حسابم» Persian line, **not** «اکانت».

---

## §4 · 20-ticket auto-route set (non-English tickets; `tickets.json` is identical)

SCENARIOS.md promises the tickets were "not reworded afterwards", so the bar for changing them is high. None needs it.

| lang | original | proposed | why | model (original) | recommendation |
|---|---|---|---|---|---|
| es | No puedo sincronizar mis archivos desde ayer, la app se queda cargando. | keep | Natural («se queda cargando» is exactly what people say). | technical support 0.987 | keep |
| de | Ich möchte meine Rechnung als PDF bekommen, nicht per Post. | keep | Natural. | billing 0.491 (escalated) | keep |
| fr | Quelqu'un a changé l'adresse e-mail de mon compte sans mon accord. | keep | Natural (FR is outside my core list, but I'm confident). | account security 0.903 | keep |
| ru | Сколько будет стоить годовая подписка для 15 сотрудников? | keep | Natural («на 15 сотрудников» also fine). | sales 0.961 | keep |
| fa | رمز عبورم را عوض نکرده‌ام ولی دیگر نمی‌توانم وارد حسابم شوم. | keep | Natural, ZWNJ correct. | account security 0.675 (escalated) | keep |
| ar | الإشعارات لا تصل إلى هاتفي منذ التحديث الأخير. | keep | Natural MSA. | technical support 0.692 (escalated) | keep |
| zh | 我想把按月付费改成按年付费。 | keep | Natural. | billing 0.848 | keep |
| tr | Şirketimiz için özel bir fiyat teklifi alabilir miyiz? | keep | Natural (TR is outside my core list). | sales 0.993 | keep |

---

## §6 · Persian message → 60 English MASSIVE intents (gallery `massive-fa` and shuffle `persian-8` use lines 2 and 4)

All six originals are correct Persian. Five use written register («را», «است») and one is colloquial («یه جوک برام بگو»).
That mix is realistic for voice-assistant text, so it's fine. Colloquial variants are listed in case the video wants "how people really talk".

| meaning | original | proposed / variant | why | model: original → proposed | recommendation |
|---|---|---|---|---|---|
| wake me up at 7 tomorrow | فردا ساعت هفت صبح بیدارم کن | **var** برای فردا ساعت ۷ صبح زنگ بذار | Colloquial "set the alarm". | alarm set 0.851 → 0.784 (2nd: calendar set 0.087) | keep original |
| turn off the kitchen lights | چراغ‌های آشپزخانه را خاموش کن | **var** چراغای آشپزخونه رو خاموش کن | Spoken form. | iot hue lightoff 0.935 → 0.874; persian-8 shuffle: 0.969 → 0.923 | keep original |
| weather in Tehran tomorrow | هوای تهران فردا چطور است؟ | **var** فردا هوای تهران چطوره؟ | Spoken form. | weather query 0.906 → 0.909 | keep original (var fine) |
| taxi to the airport | یک تاکسی برای فرودگاه بگیر | **var** برام یه تاکسی بگیر برم فرودگاه | The original is acceptable. The variant is how people say it. | transport taxi 0.981 → 0.984 | keep original; var is a good informal alt |
| turn the music down | صدای موزیک را کم کن | **prop** صدای آهنگ را کم کن | «آهنگ» is slightly more common than «موزیک», but both are fine. | audio volume down 0.654 → 0.682 | optional. Either way, **keep out of reel/short** |
| turn the music down | — | **test** صدای آهنگو کم کن / صدای موزیک رو یه کم کم کن / صدا رو کم کن | Colloquial forms. | 0.626 / 0.580 / 0.673, runner-up always *audio volume up* 0.20–0.26 | the intent is weak whatever the phrasing |
| tell me a joke | یه جوک برام بگو | keep | Natural. | general joke 0.991 | keep |

---

## Appendix A (rejected) — language notes

**A1 · "charged twice".** The English source is *"I was charged twice **for my subscription** this month."* Every
translation dropped "subscription". Most of them turned it into "money was taken **from my card**" (fa «از کارت من
پول کم شده», ru «С моей карты … списали», zh «我的卡被扣了两次款»), which really does read like fraud. Faithful versions:

| lang | original (A1) | faithful proposal | original → faithful |
|---|---|---|---|
| fa | این ماه دو بار از کارت من پول کم شده است. | این ماه دو بار بابت اشتراکم از من پول کم شده است. | ✗ account security 0.465 → ✓ billing 0.619 |
| fa | — | (informal) این ماه دو بار پول اشتراکمو ازم کم کردید! | — → ✓ billing 0.433 |
| ar | لقد تم خصم المبلغ مرتين من بطاقتي هذا الشهر. | تم خصم رسوم اشتراكي مرتين هذا الشهر. | ✓ billing 0.705 → ✓ billing 0.797 |
| ru | С моей карты дважды списали деньги в этом месяце. | В этом месяце с меня дважды списали деньги за подписку. | ✗ account security 0.615 → ✓ billing 0.554 |
| es | Me cobraron dos veces este mes. | Este mes me cobraron dos veces la suscripción. | ✓ billing 0.798 → ✓ billing 0.859 |
| de | Mir wurde diesen Monat doppelt abgebucht. | Mir wurde diesen Monat das Abo doppelt abgebucht. | ✓ billing 0.365 → **✗ technical support 0.357**. Also ✗ with «mein Abo … zweimal» (0.485) and «Ihr habt mir … das Abo zweimal abgebucht» (0.469) |
| zh | 这个月我的卡被扣了两次款。 | 这个月我的订阅费被扣了两次。 | ✗ account security 0.565 → ✓ billing 0.836 |
| fr | On m'a débité deux fois ce mois-ci. | On m'a facturé deux fois mon abonnement ce mois-ci. | ✗ technical support 0.553 → ✗ account security 0.450 |

hi/ja were not reviewed (outside my languages). **Recommendation:** keep A1 **rejected**. Faithful versions are right
in 5/6 of my languages, but only es/zh reach ≥ 0.8 and de/fr are still wrong. Do change the A1 explanation in SCENARIOS.md
("our translations dropped 'subscription' and said 'from my card'"), because as written it blames the model for part of our translation.
The German «Abo» failure (3 phrasings) is worth noting as a real weakness.

**A2 off-topic.** fa «بهترین رستوران ایتالیایی تهران کجاست؟» and es «¿Qué tiempo hará mañana en Madrid?» are natural. Keep.

---

## English

| where | original | proposed | why | model: original → proposed | recommendation |
|---|---|---|---|---|---|
| §7 urgency | I can't log in since this morning and I have a client demo in an hour. | I haven't been able to log in since this morning and I have a client demo in an hour. | "can't … since" is a non-native tense (a German/Persian calque). A native writes present perfect. | high 0.453 / critical 0.444 → **critical 0.475** / high 0.429 | use proposed. The table stays borderline high/critical, but the top flips to *critical*, so update SCENARIOS §7 |

Everything else in §1, §4, §5, §7–§10, the gallery and the rejected appendix reads as natural English (CLINC
utterances are lowercase on purpose, matching the dataset style). No change.

---

## Proposed final multilingual set (if the owner accepts)

- **§2 main table (21 cells):** apply fa-2a, fa-2b, ar-2a, ar-2b, es-2b and zh-2a. Keep everything else. Still 21/21, min 0.912.
- **§2 optional 4th row "…and it wasn't me!"** (en/fa/ar/ru/es/de/zh): 0.982 / 0.978 / 0.962 / 0.981 / 0.984 / 0.991 / 0.970.
- **§4 and §6:** no text changes. Persian "turn the music down" goes in the long video only.
- **Space gallery:** mirror the §2 edits in route-fa (2a), route-ar (2b) and route-es (2b). route-zh (2c), route-ru and route-de are unchanged. Re-record with `_build/build_gallery.py`.
- **Record as known weaknesses:** Persian loanword «اکانت» (→ wrong team), punctuation sensitivity in Persian (one comma: −0.075), German «Abo».
