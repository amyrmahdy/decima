"""Gold-labelled intent / topic / sentiment data for Decima, rendered as decisions.

Why: Decima trails Kev on banking77, SST-5, AG News and on the derived yes/no questions
("Is this article about sports?"), and MASSIVE needs more multilingual signal. These rows carry
the datasets' own human labels, so no teacher is involved. Companion of teacher/gold.py (NLI /
BoolQ) — same row schema, id hashing, smoothing, decontamination and CLI.

Sources — TRAIN SPLITS ONLY (validation/test are our eval sets; never read here):
  en  banking77  mteb/banking77              data/train (9,993)
  en  clinc150   clinc/clinc_oos  plus       train (15,250 incl. 250 oos → "none of the above (out of scope)")
  xx  massive    mteb/amazon_massive_intent  train/<locale>.json.gz, all 51 locales (11,514 each)
  en  agnews     fancyzhx/ag_news            train (120,000); titles/descriptions from sh0416/ag_news
                                              train.jsonl (same rows, same order, checked)
  en  sst5       SetFit/sst5                 train.jsonl (8,544)
  Deliberately NOT used: anything that is in BTZSC but not already in our training data (Yelp,
  Amazon polarity, IMDB, app reviews, Yahoo topics, emotion, empathetic dialogues, financial
  phrasebank, biasframes, wikitoxic, manifesto, capsotu, trueteacher) — BTZSC stays zero-shot.

Render modes (chosen per row by a seeded RNG; the mode is not stored in the rows):
  full    choose over the whole label set (canonical order or shuffled); clinc always carries its
          out-of-scope option.
  subset  choose over a random 4–25-label slice incl. gold (AG News: 2–3 + gold); ~40 % of intent
          rows; sometimes with a non-gold "none of the above"/"other" exit.
  none    like subset but the gold label is removed and "none of the above"/"other" is gold
          (labels sharing a word with gold are kept out, so the exit is really correct);
          clinc oos texts in subsets are this mode too.
  verify  derived yes/no ("Is this article about Sports?", "Does the user want the assistant to set
          an alarm?", "Is this review positive?"), balanced yes/no, order varies, bare or
          "yes: …"/"no: …" choices.
  rank    ~5 %: 4–10 candidate tags, only gold relevant; probs independent (gold 0.9, others 0.05),
          as teacher rank rows store them.
  score   SST-5 only: five ordered levels, lowest → highest, several wordings.
  choose  SST-5 only: the five levels shuffled, or coarse negative/neutral/positive.
  Label wording: prettified ("alarm_set" → "alarm set"), raw keys, Title Case, Laya-style
  "key: description", Kev-style mixes, own short descriptions (all MASSIVE intents, ~40 banking77
  intents, AG News topics), and for MASSIVE fa/ar/ru ~30 % translated label names (own
  translations). fa/ar/ru MASSIVE rows get a native-language question ~30 % of the time.
  States: raw text mostly; also ticket/message/JSON/`- role:` wrappers; MASSIVE as
  {"utterance": …} (Laya) or "User: …"; AG News sometimes as title + description.

Targets: probs = 0.92 on gold, 0.08 spread evenly over the rest (rank: see above).

Size (row targets): ~300k. banking77 42k, clinc150 42k, massive ~134k (en 20k, fa/ar/ru 15k each,
the 10 other Laya locales 2k, the remaining 36 locales 1.4k), agnews 45k, sst5 32k. Train texts
are rendered several times (different modes) where the quota exceeds the pool.

Decontamination: exactly teacher.gold.Contam (every runs/items/*.jsonl "state"/"question" split into
segments, normalized, exact + 200-char-prefix match). A text is dropped if it (or, for AG News, its
title or description) hits. Counted over the whole train pool of each source → <out>/stats.json.

Usage:  uv run python -m teacher.gold_cls --out data/gold-cls/ --seed 0
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from teacher.gold import Contam, YESNO, _cut, _hub, _parquet, norm, smooth
from teacher.prompts import NONE_TEXT

REPO = Path(__file__).resolve().parents[1]

# ─────────────────────────────── quotas ───────────────────────────────
MASSIVE_LOCALES = ["af", "am", "ar", "az", "bn", "cy", "da", "de", "el", "en", "es", "fa", "fi", "fr", "he", "hi", "hu",
                   "hy", "id", "is", "it", "ja", "jv", "ka", "km", "kn", "ko", "lv", "ml", "mn", "ms", "my", "nb", "nl",
                   "pl", "pt", "ro", "ru", "sl", "sq", "sv", "sw", "ta", "te", "th", "tl", "tr", "ur", "vi", "zh-CN", "zh-TW"]
LANG_OF = {"zh-CN": "zh", "zh-TW": "zh-tw"}
MASSIVE_MAIN = {"en": 20_000, "fa": 15_000, "ar": 15_000, "ru": 15_000}
LAYA_LOCALES = {"de", "fr", "es", "pt", "tr", "hi", "ta", "zh-CN", "ja", "ko", "sw"}  # Laya eval locales
# (domain, locale) → number of ROWS to produce
QUOTA: dict[tuple[str, str], int] = {
    ("banking77", "en"): 42_000,
    ("clinc150", "en"): 42_000,
    **{("massive", l): MASSIVE_MAIN.get(l, 2_000 if l in LAYA_LOCALES else 1_400) for l in MASSIVE_LOCALES},
    ("agnews", "en"): 45_000,
    ("sst5", "en"): 32_000,
}
OOS_EXTRA_RENDERS = 3   # clinc oos texts are rare (250); render them this many extra times

# mode mix per domain: (mode, weight)
MODES = {
    "banking77": [("full", 0.40), ("subset", 0.34), ("none", 0.06), ("verify", 0.15), ("rank", 0.05)],
    "clinc150": [("full", 0.43), ("subset", 0.34), ("none", 0.06), ("verify", 0.12), ("rank", 0.05)],
    "massive": [("full", 0.43), ("subset", 0.34), ("none", 0.06), ("verify", 0.12), ("rank", 0.05)],
    "agnews": [("full", 0.52), ("subset", 0.04), ("none", 0.04), ("verify", 0.35), ("rank", 0.05)],
    "sst5": [("score", 0.55), ("choose", 0.15), ("verify", 0.30)],
}

# ─────────────────────────────── loading ───────────────────────────────


def _item(domain, lang, text, y, check=None, extra=None):
    return {"domain": domain, "lang": lang, "text": text.strip(), "y": y, "check": check or [], "extra": extra or {}}


def load_source(domain: str, locale: str) -> tuple[list[dict], list[str]]:
    """→ (items, catalogue of label keys in canonical order)."""
    if domain == "banking77":
        rows = _parquet("mteb/banking77", "data/train-00000-of-00001.parquet", ["text", "label_text"])
        cat = sorted({r["label_text"] for r in rows})
        return [_item(domain, "en", r["text"], r["label_text"]) for r in rows if r["text"] and r["text"].strip()], cat
    if domain == "clinc150":
        import pyarrow.parquet as pq
        t = pq.read_table(_hub("clinc/clinc_oos", "plus/train-00000-of-00001.parquet"))
        names = json.loads(t.schema.metadata[b"huggingface"])["info"]["features"]["intent"]["names"]
        return [_item(domain, "en", r["text"], names[r["intent"]]) for r in t.to_pylist() if r["text"].strip()], list(names)
    if domain == "massive":
        rows = [json.loads(l) for l in gzip.open(_hub("mteb/amazon_massive_intent", f"train/{locale}.json.gz"), "rt", encoding="utf-8")]
        lang = LANG_OF.get(locale, locale)
        return [_item(domain, lang, r["text"], r["label"]) for r in rows if r["text"].strip()], sorted(MASSIVE_DESC)
    if domain == "agnews":
        rows = _parquet("fancyzhx/ag_news", "data/train-00000-of-00001.parquet", ["text", "label"])
        td = [json.loads(l) for l in open(_hub("sh0416/ag_news", "train.jsonl"), encoding="utf-8")]
        out = []
        for r, s in zip(rows, td):
            extra = {}
            if (s["title"] + " " + s["description"]).strip() == r["text"].strip():   # same row, verified
                extra = {"title": s["title"].strip(), "desc": s["description"].strip()}
            out.append(_item(domain, "en", r["text"], r["label"], [extra.get("title", ""), extra.get("desc", "")], extra))
        return out, [0, 1, 2, 3]
    if domain == "sst5":
        rows = [json.loads(l) for l in open(_hub("SetFit/sst5", "train.jsonl"), encoding="utf-8")]
        return [_item(domain, "en", r["text"], int(r["label"])) for r in rows if r["text"].strip()], [0, 1, 2, 3, 4]
    raise KeyError(domain)


# ─────────────────────────────── label wording ───────────────────────────────
# MASSIVE: short verb-phrase descriptions (usable after "want the assistant to …") + fa/ar/ru names
MASSIVE_DESC = {
    "alarm_query": "check my alarms", "alarm_remove": "remove an alarm", "alarm_set": "set an alarm",
    "audio_volume_down": "turn the volume down", "audio_volume_mute": "mute the audio",
    "audio_volume_other": "change other volume settings", "audio_volume_up": "turn the volume up",
    "calendar_query": "check the calendar", "calendar_remove": "delete a calendar event",
    "calendar_set": "add a calendar event or reminder", "cooking_query": "answer a cooking question",
    "cooking_recipe": "find a recipe", "datetime_convert": "convert a time or time zone",
    "datetime_query": "tell the date or time", "email_addcontact": "add an email contact", "email_query": "check emails",
    "email_querycontact": "look up a contact's details", "email_sendemail": "send an email",
    "general_greet": "greet the user or chat", "general_joke": "tell a joke",
    "general_quirky": "respond to small talk or an odd question", "iot_cleaning": "start the robot vacuum",
    "iot_coffee": "make coffee", "iot_hue_lightchange": "change the light colour", "iot_hue_lightdim": "dim the lights",
    "iot_hue_lightoff": "turn the lights off", "iot_hue_lighton": "turn the lights on",
    "iot_hue_lightup": "brighten the lights", "iot_wemo_off": "turn off a smart plug", "iot_wemo_on": "turn on a smart plug",
    "lists_createoradd": "create a list or add to one", "lists_query": "read out a list",
    "lists_remove": "remove something from a list", "music_dislikeness": "note that the user dislikes some music",
    "music_likeness": "remember that the user likes some music", "music_query": "identify or describe the music",
    "music_settings": "change playback settings such as shuffle or repeat", "news_query": "give the news",
    "play_audiobook": "play an audiobook", "play_game": "start a game", "play_music": "play music",
    "play_podcasts": "play a podcast", "play_radio": "play the radio", "qa_currency": "give an exchange rate",
    "qa_definition": "define a word", "qa_factoid": "answer a general-knowledge question", "qa_maths": "do a calculation",
    "qa_stock": "give a stock price", "recommendation_events": "recommend events",
    "recommendation_locations": "recommend places", "recommendation_movies": "recommend a movie",
    "social_post": "post on social media", "social_query": "check social media updates",
    "takeaway_order": "order takeaway food", "takeaway_query": "check on a takeaway order",
    "transport_query": "give transport information", "transport_taxi": "book a taxi",
    "transport_ticket": "book a train or travel ticket", "transport_traffic": "report traffic conditions",
    "weather_query": "give the weather",
}
MASSIVE_TR = {  # key → (fa, ar, ru)
    "alarm_query": ("پرسش درباره هشدار", "الاستعلام عن المنبه", "запрос о будильнике"),
    "alarm_remove": ("حذف هشدار", "حذف المنبه", "удалить будильник"),
    "alarm_set": ("تنظیم هشدار", "ضبط المنبه", "установить будильник"),
    "audio_volume_down": ("کم کردن صدا", "خفض الصوت", "уменьшить громкость"),
    "audio_volume_mute": ("بی‌صدا کردن", "كتم الصوت", "выключить звук"),
    "audio_volume_other": ("سایر تنظیمات صدا", "إعدادات صوت أخرى", "другие настройки звука"),
    "audio_volume_up": ("زیاد کردن صدا", "رفع الصوت", "увеличить громкость"),
    "calendar_query": ("پرسش درباره تقویم", "الاستعلام عن التقويم", "запрос к календарю"),
    "calendar_remove": ("حذف رویداد تقویم", "حذف حدث من التقويم", "удалить событие из календаря"),
    "calendar_set": ("افزودن رویداد به تقویم", "إضافة حدث إلى التقويم", "добавить событие в календарь"),
    "cooking_query": ("پرسش آشپزی", "سؤال عن الطبخ", "вопрос о готовке"),
    "cooking_recipe": ("دستور پخت", "وصفة طبخ", "рецепт"),
    "datetime_convert": ("تبدیل ساعت یا منطقه زمانی", "تحويل الوقت أو المنطقة الزمنية", "перевод времени или часового пояса"),
    "datetime_query": ("پرسش درباره تاریخ یا ساعت", "الاستعلام عن التاريخ أو الوقت", "вопрос о дате или времени"),
    "email_addcontact": ("افزودن مخاطب ایمیل", "إضافة جهة اتصال للبريد", "добавить контакт"),
    "email_query": ("بررسی ایمیل‌ها", "الاستعلام عن البريد الإلكتروني", "проверить почту"),
    "email_querycontact": ("پرسش درباره مخاطب", "الاستعلام عن جهة اتصال", "информация о контакте"),
    "email_sendemail": ("ارسال ایمیل", "إرسال بريد إلكتروني", "отправить письмо"),
    "general_greet": ("احوال‌پرسی", "تحية", "приветствие"),
    "general_joke": ("جوک گفتن", "إلقاء نكتة", "рассказать шутку"),
    "general_quirky": ("گفت‌وگوی متفرقه", "دردشة عامة", "болтовня"),
    "iot_cleaning": ("روشن کردن جاروی رباتیک", "تشغيل المكنسة الذكية", "включить робот-пылесос"),
    "iot_coffee": ("درست کردن قهوه", "تحضير القهوة", "сварить кофе"),
    "iot_hue_lightchange": ("تغییر رنگ چراغ", "تغيير لون الإضاءة", "изменить цвет света"),
    "iot_hue_lightdim": ("کم‌نور کردن چراغ", "تخفيف الإضاءة", "приглушить свет"),
    "iot_hue_lightoff": ("خاموش کردن چراغ", "إطفاء الأضواء", "выключить свет"),
    "iot_hue_lighton": ("روشن کردن چراغ", "تشغيل الأضواء", "включить свет"),
    "iot_hue_lightup": ("پرنورتر کردن چراغ", "زيادة الإضاءة", "сделать свет ярче"),
    "iot_wemo_off": ("خاموش کردن پریز هوشمند", "إطفاء المقبس الذكي", "выключить умную розетку"),
    "iot_wemo_on": ("روشن کردن پریز هوشمند", "تشغيل المقبس الذكي", "включить умную розетку"),
    "lists_createoradd": ("ساختن فهرست یا افزودن به آن", "إنشاء قائمة أو الإضافة إليها", "создать список или добавить в него"),
    "lists_query": ("پرسش درباره فهرست", "الاستعلام عن قائمة", "просмотреть список"),
    "lists_remove": ("حذف از فهرست", "الحذف من قائمة", "удалить из списка"),
    "music_dislikeness": ("نپسندیدن آهنگ", "عدم الإعجاب بالموسيقى", "не нравится музыка"),
    "music_likeness": ("پسندیدن آهنگ", "الإعجاب بالموسيقى", "нравится музыка"),
    "music_query": ("پرسش درباره آهنگ", "الاستعلام عن الموسيقى", "вопрос о музыке"),
    "music_settings": ("تنظیمات پخش موسیقی", "إعدادات تشغيل الموسيقى", "настройки воспроизведения"),
    "news_query": ("اخبار", "الأخبار", "новости"),
    "play_audiobook": ("پخش کتاب صوتی", "تشغيل كتاب صوتي", "включить аудиокнигу"),
    "play_game": ("بازی کردن", "تشغيل لعبة", "запустить игру"),
    "play_music": ("پخش موسیقی", "تشغيل الموسيقى", "включить музыку"),
    "play_podcasts": ("پخش پادکست", "تشغيل بودكاست", "включить подкаст"),
    "play_radio": ("پخش رادیو", "تشغيل الراديو", "включить радио"),
    "qa_currency": ("نرخ ارز", "سعر العملة", "курс валют"),
    "qa_definition": ("معنی واژه", "تعريف كلمة", "значение слова"),
    "qa_factoid": ("پرسش اطلاعات عمومی", "سؤال معلومات عامة", "фактический вопрос"),
    "qa_maths": ("محاسبه ریاضی", "عملية حسابية", "математический расчёт"),
    "qa_stock": ("قیمت سهام", "سعر السهم", "цена акций"),
    "recommendation_events": ("پیشنهاد رویداد", "اقتراح فعاليات", "рекомендация мероприятий"),
    "recommendation_locations": ("پیشنهاد مکان", "اقتراح أماكن", "рекомендация мест"),
    "recommendation_movies": ("پیشنهاد فیلم", "اقتراح أفلام", "рекомендация фильмов"),
    "social_post": ("ارسال پست در شبکه اجتماعی", "النشر على وسائل التواصل", "пост в соцсети"),
    "social_query": ("پرسش درباره شبکه‌های اجتماعی", "الاستعلام عن وسائل التواصل", "новости из соцсетей"),
    "takeaway_order": ("سفارش غذای بیرون‌بر", "طلب طعام سفري", "заказать еду на вынос"),
    "takeaway_query": ("پرسش درباره سفارش غذا", "الاستعلام عن طلب الطعام", "вопрос о заказе еды"),
    "transport_query": ("پرسش درباره حمل‌ونقل", "الاستعلام عن المواصلات", "вопрос о транспорте"),
    "transport_taxi": ("درخواست تاکسی", "طلب سيارة أجرة", "вызвать такси"),
    "transport_ticket": ("خرید بلیت سفر", "حجز تذكرة سفر", "купить билет"),
    "transport_traffic": ("وضعیت ترافیک", "حالة المرور", "пробки на дорогах"),
    "weather_query": ("وضع هوا", "الطقس", "погода"),
}
TR_IDX = {"fa": 0, "ar": 1, "ru": 2}
# banking77: own short descriptions (noun phrases) for a subset of intents
BANKING_DESC = {
    "Refund_not_showing_up": "a refund that has not appeared yet", "activate_my_card": "activating a new card",
    "age_limit": "the minimum age to open an account", "apple_pay_or_google_pay": "using Apple Pay or Google Pay",
    "atm_support": "which ATMs the card works at", "automatic_top_up": "setting up automatic top-ups",
    "beneficiary_not_allowed": "a payee that cannot be added", "cancel_transfer": "cancelling a transfer",
    "card_arrival": "when the ordered card will arrive", "card_delivery_estimate": "how long card delivery takes",
    "card_not_working": "a card that does not work", "card_swallowed": "an ATM that kept the card",
    "cash_withdrawal_charge": "a fee for withdrawing cash", "change_pin": "changing the PIN",
    "compromised_card": "a card that may have been compromised", "contactless_not_working": "contactless payment not working",
    "country_support": "which countries are supported", "declined_card_payment": "a declined card payment",
    "edit_personal_details": "updating personal details", "exchange_charge": "fees for exchanging currency",
    "exchange_rate": "the exchange rate used", "failed_transfer": "a transfer that failed",
    "getting_spare_card": "getting an additional card", "lost_or_stolen_card": "a lost or stolen card",
    "lost_or_stolen_phone": "a lost or stolen phone", "passcode_forgotten": "a forgotten passcode",
    "pending_transfer": "a transfer that is still pending", "pin_blocked": "a blocked PIN",
    "receiving_money": "receiving money into the account", "request_refund": "asking for a refund",
    "terminate_account": "closing the account", "top_up_failed": "a top-up that failed", "top_up_limits": "limits on top-ups",
    "transaction_charged_twice": "being charged twice for one payment", "transfer_timing": "how long a transfer takes",
    "unable_to_verify_identity": "problems verifying identity", "verify_my_identity": "how to verify identity",
    "virtual_card_not_working": "a virtual card that does not work", "visa_or_mastercard": "whether the card is Visa or Mastercard",
    "wrong_amount_of_cash_received": "an ATM giving the wrong amount of cash",
}
DESC = {"banking77": BANKING_DESC, "massive": MASSIVE_DESC, "clinc150": {}}
OOS = "oos"
OOS_TEXT = {"pretty": "none of the above (out of scope)", "key": "out_of_scope", "title": "None of the above (out of scope)",
            "keydesc": "out_of_scope: the request is outside what the assistant supports", "desc": "something out of scope"}
STOP = {"a", "an", "the", "of", "to", "or", "and", "my", "by", "for", "on", "in", "up", "is", "are", "you", "your", "not", "how", "what"}

# AG News: label order World, Sports, Business, Sci/Tech
AG_SETS = [
    ["World", "Sports", "Business", "Sci/Tech"], ["world", "sports", "business", "scitech"],
    ["World news", "Sports", "Business", "Science and technology"], ["world", "sports", "business", "science/technology"],
    ["international news", "sports", "business and economy", "science and technology"],
    ["Politics & World", "Sports", "Business & Finance", "Tech & Science"], ["World", "Sports", "Business", "Technology"],
]
AG_KEYS = ["world", "sports", "business", "scitech"]
AG_DESC = [
    ["World news: politics, international affairs, conflicts", "Sports: games, athletes, teams, results",
     "Business: companies, markets, economy, finance", "Science and technology: research, gadgets, software, space"],
    ["news about countries, governments, wars and diplomacy", "news about sports events, teams and players",
     "news about companies, earnings, markets and the economy", "news about science, computing, the internet and new technology"],
]
AG_TOPIC = [["world news", "international affairs", "world politics", "world events"], ["sports", "sport"],
            ["business", "business and the economy", "companies and markets", "finance"],
            ["science and technology", "technology", "science and tech", "sci/tech"]]
AG_TR_NONE = ["other", "none of the above", "none of these topics"]

# SST-5: ordered level wordings, lowest → highest
SST_SCORE = [
    ["very negative", "negative", "neutral", "positive", "very positive"],
    ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"],
    ["terrible", "bad", "okay", "good", "excellent"],
    ["1 star: terrible", "2 stars: poor", "3 stars: average", "4 stars: good", "5 stars: excellent"],
    ["strongly negative", "somewhat negative", "neutral or mixed", "somewhat positive", "strongly positive"],
    ["hated it", "disliked it", "mixed feelings", "liked it", "loved it"],
    ["1 - very negative", "2 - negative", "3 - neutral", "4 - positive", "5 - very positive"],
    ["Very negative", "Negative", "Neutral", "Positive", "Very positive"],
]
SST_COARSE = [["negative", "neutral", "positive"], ["negative: the reviewer dislikes the film", "neutral: mixed or no clear opinion",
                                                    "positive: the reviewer likes the film"], ["bad", "mixed", "good"]]

# ─────────────────────────────── question templates ───────────────────────────────
CHOOSE_Q = {
    "banking77": ["Which banking intent best describes this customer message?", "What is the customer asking about?",
                  "Classify the customer's request.", "What does the customer need help with?",
                  "Which support topic fits this message?", "Pick the intent of this banking query.",
                  "Route this message to the right intent.", "What is this customer message about?",
                  "Which category best matches the query?", "Label the intent of the message.",
                  "What is the customer's banking issue?", "Choose the option that best answers the request.",
                  "Which intent should this ticket be tagged with?", "What does this bank customer want?"],
    "clinc150": ["What is the user's intent? (may be out of scope)", "What does the user want?",
                 "Which intent does this request express?", "Classify this virtual-assistant request.",
                 "Which skill should handle this request?", "What is the user asking the assistant to do?",
                 "Pick the intent; choose out of scope if none fits.", "Which of these best describes what the user wants?",
                 "Label the user's request.", "Identify the intent of the utterance.",
                 "Which category does this request belong to?", "What is this request about?"],
    "massive": ["What does the user want the assistant to do?", "What is the intent of this voice command?",
                "Classify the command.", "Which action should the voice assistant take?", "What is the user asking for?",
                "Identify the intent of the utterance.", "Which intent matches this request?",
                "Which of these best describes what the user wants?", "Label the intent of this utterance.",
                "What is this command about?", "Which skill should handle this request?"],
    "agnews": ["What is the topic of this article?", "What is the topic of this news article?",
               "Which section of the newspaper does this story belong to?", "Classify the news article by topic.",
               "What category is this news story?", "Which topic best describes the text?", "What is this news about?",
               "Assign a topic to this article.", "Which desk would cover this story?", "What kind of news is this?"],
    "sst5": ["What is the sentiment of this review?", "How does the reviewer feel about the movie?",
             "Classify the sentiment of this sentence.", "What is the sentiment of this review sentence?",
             "What opinion does this movie review express?", "Which label fits the tone of this review?"],
}
JSON_Q_MASSIVE = ["What is the user asking for in `utterance`?", "What intent does `utterance` express?",
                  "Classify `utterance`."]
CHOOSE_Q_NATIVE = {
    "fa": ["کاربر از دستیار چه می‌خواهد؟", "هدف این درخواست چیست؟", "این دستور صوتی در کدام دسته قرار می‌گیرد؟",
           "منظور کاربر از این جمله چیست؟"],
    "ar": ["ماذا يريد المستخدم من المساعد؟", "ما هو قصد هذا الطلب؟", "إلى أي فئة ينتمي هذا الأمر الصوتي؟",
           "ماذا يقصد المستخدم بهذه الجملة؟"],
    "ru": ["Что пользователь хочет от ассистента?", "Какое намерение у этого запроса?",
           "К какой категории относится эта голосовая команда?", "Что имеет в виду пользователь?"],
}
SCORE_Q = ["What is the sentiment of this review sentence?", "How many stars would this reviewer give?",
           "Rate the sentiment of the review.", "How positive is this movie review?",
           "What rating does this critic's sentence imply?", "How much did the reviewer like the film?",
           "Rate this review from most negative to most positive.", "What is the sentiment of this review?"]
RANK_Q = ["Which of the following tags are applicable to this state? (Select all that apply)",
          "Which of these labels apply to the message?", "Mark every tag that fits.",
          "Which of these categories are relevant? Select all that apply.", "Which labels describe this text?",
          "Tag the message: which of these apply?"]
FOCUS = ["Consider the whole message.", "Pick the single best fit.", "Use only the information given."]

VERIFY_Q = {  # {L}: label text · {D}: description (verb phrase for massive, noun phrase for banking)
    "banking77": {"L": ['Is the customer\'s intent "{L}"?', "Does this message belong to the intent {L}?",
                        "Intent: {L}\nDoes the message match this intent?", "Would you tag this query as {L}?",
                        'Is this message about "{L}"?', "Is the category of this request {L}?"],
                  "D": ["Is the customer asking about {D}?", "Does the customer need help with {D}?",
                        "Is this message about {D}?", "Is the customer's issue {D}?"]},
    "clinc150": {"L": ['Is the user\'s intent "{L}"?', "Does this request fall under {L}?",
                       "Intent: {L}\nDoes the utterance match this intent?", "Should this request be routed to {L}?",
                       'Is this request about "{L}"?', "Would you label this request {L}?"], "D": []},
    "massive": {"L": ['Is the intent of this command "{L}"?', "Does this utterance belong to the intent {L}?",
                      "Intent: {L}\nDoes the utterance match this intent?", "Would you label this request {L}?"],
                "D": ["Does the user want the assistant to {D}?", "Is the user asking the assistant to {D}?",
                      "Should the assistant {D}?", "Is this a request to {D}?"]},
}
VERIFY_Q_NATIVE = {
    "fa": ["آیا هدف این درخواست «{L}» است؟", "آیا کاربر این را می‌خواهد: {L}؟", "آیا این جمله در دسته «{L}» قرار می‌گیرد؟"],
    "ar": ["هل قصد هذا الطلب «{L}»؟", "هل يريد المستخدم هذا: {L}؟", "هل تنتمي هذه الجملة إلى فئة «{L}»؟"],
    "ru": ["Намерение этого запроса — «{L}»?", "Пользователь хочет следующее: {L}?", "Относится ли этот запрос к категории «{L}»?"],
}
OOS_VERIFY_Q = ["Is this request outside what the assistant supports?", "Is this request out of scope?",
                "Does this request fall outside all the supported intents?"]
AG_VERIFY_Q = ["Is this article about {t}?", "Is this a {t} story?", "Does this news item belong in the {t} section?",
               "Topic: {t}\nDoes the article match this topic?", "Is the main subject of this text {t}?",
               "Is this news about {t}?", "Would this story run in the {t} pages?"]
SST_VERIFY = [  # (question, set of labels answered yes)
    ("Is this review positive?", {3, 4}), ("Is the sentiment of this review positive?", {3, 4}),
    ("Would this reviewer recommend the movie?", {3, 4}), ("Did the reviewer like the film?", {3, 4}),
    ("Is this review negative?", {0, 1}), ("Did the reviewer dislike the film?", {0, 1}),
    ("Is the sentiment of this sentence negative?", {0, 1}), ("Is this review very positive?", {4}),
    ("Is this review strongly negative?", {0}), ("Is the reviewer's opinion neutral or mixed?", {2}),
    ("Is the tone of this review neither clearly positive nor clearly negative?", {2}),
    ("Is this at least a somewhat positive review?", {3, 4}),
]
YN_DESC = {
    "intent": [("yes: the message matches this intent", "no: it is about something else"),
               ("yes: This is what the user wants", "no: The user wants something else")],
    "agnews": [("yes: That is the article's topic", "no: The article is about something else")],
    "sst5": [("yes: Clearly the case", "no: Not the case"), ("yes: that describes the review", "no: that does not describe the review")],
}
EN_YN = [("yes", "no"), ("Yes", "No"), ("yes", "no"), ("true", "false")]

# ─────────────────────────────── state wrapping ───────────────────────────────
WRAP = {
    "intent": ["Customer: {t}", "User: {t}", "message: {t}", "document: {t}", "ticket:\n  channel: chat\n  body: {t}",
               "ticket:\n  channel: email\n  body: {t}", "- role: customer\n  content: {t}", 'Customer message: "{t}"',
               "JSON:message", "JSON:text", "JSON:query"],
    "massive": ["User: {t}", "user: {t}", "Command: {t}", "document: {t}", "- role: user\n  content: {t}",
                "JSON:text", "JSON:query"],
    "agnews": ["document: {t}", "Article: {t}", "News: {t}", "ticket:\n  channel: email\n  body: {t}", "JSON:text"],
    "sst5": ["Review: {t}", "document: {t}", "Movie review: {t}", "- role: customer\n  content: {t}",
             "ticket:\n  channel: email\n  body: {t}", "JSON:review", "Critic: {t}"],
}
AG_TD = ["Title: {a}\nDescription: {b}", "Headline: {a}\nLead: {b}", "{a}\n\n{b}", "title: {a}\nbody: {b}", "JSON"]


def _wrap(rng, it) -> tuple[str, str]:
    """→ (state, wrap tag)."""
    dom, t = it["domain"], _cut(it["text"])
    r = rng.random()
    if dom == "massive":
        if r < 0.55:
            return t, "raw"
        if r < 0.75:
            return json.dumps({"utterance": t}, ensure_ascii=False), "utterance"
        w = rng.choice(WRAP["massive"])
    elif dom == "agnews":
        ex = it["extra"]
        if r < 0.22 and ex.get("title") and ex.get("desc"):
            w = rng.choice(AG_TD)
            if w == "JSON":
                return json.dumps({"title": ex["title"], "description": _cut(ex["desc"])}, ensure_ascii=False), "title+desc"
            return w.format(a=ex["title"], b=_cut(ex["desc"])), "title+desc"
        if r < 0.65:
            return t, "raw"
        w = rng.choice(WRAP["agnews"])
    else:
        if r < 0.6:
            return t, "raw"
        w = rng.choice(WRAP["sst5" if dom == "sst5" else "intent"])
    if w.startswith("JSON:"):
        return json.dumps({w[5:]: t}, ensure_ascii=False), "json"
    return w.format(t=t), "wrap"


# ─────────────────────────────── rendering ───────────────────────────────


def pretty(k: str) -> str:
    return re.sub(r"\s+", " ", k.replace("_", " ").replace("?", "")).strip().lower()


def _words(k) -> set[str]:
    return {w for w in re.split(r"[_\s/]+", str(k).lower().replace("?", "")) if w and w not in STOP}


def _far(dom, gold, cat):
    """Labels that share no content word with gold (safe distractors when gold is absent / rank)."""
    if dom in ("agnews", "sst5"):
        return [k for k in cat if k != gold]
    gw = _words(gold)
    return [k for k in cat if k != gold and k != OOS and not (_words(k) & gw)]


def _near(dom, gold, cat):
    gw = _words(gold)
    return [k for k in cat if k != gold and k != OOS and _words(k) & gw]


def _style(rng, dom, lang):
    """Per-row label-wording style for intent domains."""
    if dom == "massive" and lang in TR_IDX and rng.random() < 0.30:
        return "tr"
    r = rng.random()
    if r < 0.33:
        return "pretty"
    if r < 0.48:
        return "key"
    if r < 0.56:
        return "title"
    if r < 0.76:
        return "keydesc"
    if r < 0.86:
        return "mix"
    return "desc"


def _label(dom, k, style, lang, sub):
    """sub: per-row choice of fallback description for keydesc ('laya' | 'kev' | 'own')."""
    if k == OOS:
        if style == "tr":
            return NONE_TEXT[lang]
        return OOS_TEXT.get(style if style in OOS_TEXT else "keydesc", OOS_TEXT["pretty"])
    d = DESC[dom].get(k)
    if style == "pretty":
        return pretty(k)
    if style == "key":
        return k
    if style == "title":
        return pretty(k).title()
    if style == "tr":
        return MASSIVE_TR[k][TR_IDX[lang]] if k in MASSIVE_TR else pretty(k)
    if style == "desc":
        return d if d else pretty(k)
    # keydesc
    if sub == "own" and d:
        return f"{k}: {d}"
    if sub == "kev":
        return f"{k}: Request related to {pretty(k)}"
    return f"{k}: {pretty(k)}"


def _labels(rng, dom, keys, style, lang):
    sub = rng.choice(["laya", "kev", "own", "own"])
    if style == "mix":
        out = [_label(dom, k, "keydesc" if rng.random() < 0.5 else "key", lang, sub) for k in keys]
    else:
        out = [_label(dom, k, style, lang, sub) for k in keys]
    if len(set(out)) != len(out):  # a description collided — fall back to prettified names
        out = [_label(dom, k, "pretty", lang, sub) for k in keys]
    return out


def _none_text(rng, lang, tr):
    if tr and lang in NONE_TEXT:
        return rng.choice([NONE_TEXT[lang], {"fa": "سایر", "ar": "أخرى", "ru": "другое"}[lang]])
    return rng.choice(["none of the above", "none of the above", "other", "none of these", "something else"])


def _q(rng, q):
    if rng.random() < 0.08:
        return f"question: {q}\nfocus: {rng.choice(FOCUS)}"
    return q


def _yn(rng, lang, fam, yes_gold, native_q):
    if lang in YESNO and (native_q and rng.random() < 0.7 or not native_q and rng.random() < 0.10):
        y, n = YESNO[lang]
        tr = True
    else:
        tr = False
        y, n = rng.choice(EN_YN) if rng.random() < 0.7 else rng.choice(YN_DESC[fam])
    items = [(y, yes_gold), (n, not yes_gold)]
    if rng.random() < 0.5:
        items.reverse()
    return [c for c, _ in items], next(i for i, (_, g) in enumerate(items) if g), tr


def _pick_mode(rng, dom, avoid=None):
    for _ in range(8):
        r, acc = rng.random(), 0.0
        for m, w in MODES[dom]:
            acc += w
            if r < acc:
                break
        if m != avoid:
            return m
    return m


def render(it: dict, cat: list, rng: random.Random, avoid: str | None = None) -> tuple[dict, str, str]:
    """→ (row, mode, wrap tag)."""
    dom, lang, y = it["domain"], it["lang"], it["y"]
    mode = _pick_mode(rng, dom, avoid)
    if dom == "clinc150" and y == OOS and mode in ("subset", "rank"):
        mode = "none"
    state, wrap = _wrap(rng, it)
    native_q = dom == "massive" and lang in CHOOSE_Q_NATIVE and rng.random() < 0.30
    none_idx, tr = None, False
    if dom in ("sst5",):
        return _render_sst(it, rng, mode, state, wrap)
    if dom == "agnews":
        return _render_ag(it, rng, mode, state, wrap)

    # ── intent domains: banking77 / clinc150 / massive ──
    def choose_q():
        if native_q:
            return rng.choice(CHOOSE_Q_NATIVE[lang])
        if wrap == "utterance" and rng.random() < 0.5:
            return rng.choice(JSON_Q_MASSIVE)
        return _q(rng, rng.choice(CHOOSE_Q[dom]))

    if mode == "full":
        kind = "choose"
        style = _style(rng, dom, lang)
        tr = style == "tr"
        keys = list(cat)
        if rng.random() < 0.6:
            rng.shuffle(keys)
        choices = _labels(rng, dom, keys, style, lang)
        gold = keys.index(y)
        none_idx = keys.index(OOS) if OOS in keys else None
        question = choose_q()
    elif mode in ("subset", "none"):
        kind = "choose"
        style = _style(rng, dom, lang)
        tr = style == "tr"
        k = rng.randint(4, min(25, len(cat)))
        if mode == "none":
            far = _far(dom, y, cat)
            keys = rng.sample(far, min(k - 1, len(far)))
            rng.shuffle(keys)
            choices = _labels(rng, dom, keys, style, lang)
            nt = OOS_TEXT["pretty"] if dom == "clinc150" and rng.random() < 0.5 and not tr else _none_text(rng, lang, tr)
            pos = len(choices) if rng.random() < 0.85 else rng.randint(0, len(choices))
            choices.insert(pos, nt)
            gold = none_idx = pos
        else:
            pool = [c for c in cat if c != y and c != OOS]
            keys = rng.sample(pool, k - 1) + [y]
            rng.shuffle(keys)
            choices = _labels(rng, dom, keys, style, lang)
            gold = keys.index(y)
            if rng.random() < (0.5 if dom == "clinc150" else 0.3):   # a non-gold exit
                nt = OOS_TEXT["pretty"] if dom == "clinc150" and not tr else _none_text(rng, lang, tr)
                if nt not in choices:
                    choices.append(nt)
                    none_idx = len(choices) - 1
        question = choose_q()
    elif mode == "rank":
        kind = "rank"
        style = _style(rng, dom, lang)
        if style == "tr":
            style = "pretty"
        far = _far(dom, y, cat)
        keys = rng.sample(far, min(rng.randint(3, 9), len(far))) + [y]
        rng.shuffle(keys)
        choices = _labels(rng, dom, keys, style, lang)
        gold = keys.index(y)
        question = rng.choice(RANK_Q)
        native_q = False
        row = _row(it, kind, state, question, choices, gold, lang, "en", None,
                   [0.9 if i == gold else 0.05 for i in range(len(choices))])
        return row, mode, wrap
    else:  # verify
        kind = "verify"
        if dom == "clinc150" and (y == OOS or rng.random() < 0.02) and rng.random() < 0.6:
            yes = y == OOS
            question = _q(rng, rng.choice(OOS_VERIFY_Q))
            native_q = False
        else:
            yes = rng.random() < 0.5 if y != OOS else False
            if yes:
                tgt = y
            else:
                near = _near(dom, y, cat)
                tgt = rng.choice(near) if near and rng.random() < 0.3 else rng.choice([c for c in cat if c != y and c != OOS])
            d = DESC[dom].get(tgt)
            if native_q:
                tr_l = rng.random() < 0.6
                L = MASSIVE_TR[tgt][TR_IDX[lang]] if tr_l else (pretty(tgt) if rng.random() < 0.6 else tgt)
                question = rng.choice(VERIFY_Q_NATIVE[lang]).format(L=L)
            elif dom == "massive" and lang in TR_IDX and rng.random() < 0.30:
                tr = True
                question = rng.choice(VERIFY_Q[dom]["L"]).format(L=MASSIVE_TR[tgt][TR_IDX[lang]])
            elif d and rng.random() < 0.55:
                question = _q(rng, rng.choice(VERIFY_Q[dom]["D"]).format(D=d))
            else:
                L = rng.choice([pretty(tgt), pretty(tgt), tgt, pretty(tgt).title()])
                question = _q(rng, rng.choice(VERIFY_Q[dom]["L"]).format(L=L))
        choices, gold, tr_yn = _yn(rng, lang, "intent", yes, native_q)
        tr = tr or tr_yn
    choice_lang = lang if (native_q or tr) else "en"
    return _row(it, kind, state, question, choices, gold, lang, choice_lang, none_idx), mode, wrap


def _render_ag(it, rng, mode, state, wrap):
    y = it["y"]
    q = lambda: _q(rng, rng.choice(CHOOSE_Q["agnews"]))
    none_idx = None

    def names(keys):
        r = rng.random()
        if r < 0.55:
            s = rng.choice(AG_SETS)
            return [s[k] for k in keys]
        d = rng.choice(AG_DESC)
        if r < 0.75:
            return [f"{AG_KEYS[k]}: {d[k]}" for k in keys]
        if r < 0.9:   # Kev-style mix
            return [f"{AG_KEYS[k]}: {d[k]}" if rng.random() < 0.5 else AG_KEYS[k] for k in keys]
        return [d[k] for k in keys]

    if mode == "full":
        keys = [0, 1, 2, 3]
        if rng.random() < 0.7:
            rng.shuffle(keys)
        choices, gold, kind, question = names(keys), keys.index(y), "choose", q()
    elif mode in ("subset", "none"):
        others = [k for k in range(4) if k != y]
        if mode == "none":
            keys = rng.sample(others, rng.randint(2, 3))
            choices = names(keys)
            choices.append(rng.choice(AG_TR_NONE))
            gold = none_idx = len(choices) - 1
        else:
            keys = rng.sample(others, rng.randint(1, 2)) + [y]
            rng.shuffle(keys)
            choices = names(keys)
            gold = keys.index(y)
            if rng.random() < 0.5:
                choices.append(rng.choice(AG_TR_NONE))
                none_idx = len(choices) - 1
        kind, question = "choose", q()
    elif mode == "rank":
        keys = [0, 1, 2, 3]
        rng.shuffle(keys)
        choices, gold = names(keys), keys.index(y)
        row = _row(it, "rank", state, rng.choice(RANK_Q), choices, gold, "en", "en", None,
                   [0.9 if i == gold else 0.05 for i in range(4)])
        return row, mode, wrap
    else:  # verify, balanced
        yes = rng.random() < 0.5
        tgt = y if yes else rng.choice([k for k in range(4) if k != y])
        t = rng.choice(AG_TOPIC[tgt] + [rng.choice(AG_SETS)[tgt]])
        tq = rng.choice(AG_VERIFY_Q)
        if "section" in tq or "pages" in tq:
            t = rng.choice(AG_SETS)[tgt]
        question = _q(rng, tq.format(t=t))
        choices, gold, _ = _yn(rng, "en", "agnews", yes, False)
        kind = "verify"
    return _row(it, kind, state, question, choices, gold, "en", "en", none_idx), mode, wrap


def _render_sst(it, rng, mode, state, wrap):
    y = it["y"]
    if mode == "score":
        choices = list(rng.choice(SST_SCORE))
        return _row(it, "score", state, _q(rng, rng.choice(SCORE_Q)), choices, y, "en", "en", None), mode, wrap
    if mode == "choose":
        if rng.random() < 0.5:
            levels = rng.choice(SST_COARSE)
            gold_c = 0 if y <= 1 else 1 if y == 2 else 2
            keys = [0, 1, 2]
        else:
            levels = rng.choice(SST_SCORE)
            gold_c = y
            keys = [0, 1, 2, 3, 4]
        rng.shuffle(keys)
        choices = [levels[k] for k in keys]
        return _row(it, "choose", state, _q(rng, rng.choice(CHOOSE_Q["sst5"])), choices, keys.index(gold_c), "en", "en", None), mode, wrap
    # verify, balanced: pick the answer first, then a question compatible with it
    yes = rng.random() < 0.5
    cands = [(q, s) for q, s in SST_VERIFY if (y in s) == yes]
    q, _ = rng.choice(cands)
    choices, gold, _ = _yn(rng, "en", "sst5", yes, False)
    return _row(it, "verify", state, _q(rng, q), choices, gold, "en", "en", None), mode, wrap


def _row(it, kind, state, question, choices, gold, lang, choice_lang, none_idx, probs=None):
    dom = it["domain"]
    key = json.dumps([dom, lang, state, question, choices], ensure_ascii=False)
    return {
        "id": hashlib.sha1(key.encode()).hexdigest()[:20], "source": "gold", "domain": dom, "kind": kind,
        "state": state, "question": question, "choices": choices, "gold": gold,
        "probs": probs if probs is not None else smooth(len(choices), gold),
        "state_lang": lang, "choice_lang": choice_lang, "none_idx": none_idx, "batch": f"{dom}/{lang}|gold",
    }


# ─────────────────────────────── main ───────────────────────────────


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="data/gold-cls/")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--items", default=str(REPO / "runs/items"))
    ap.add_argument("--scale", type=float, default=1.0, help="multiply every quota (for quick tests)")
    ap.add_argument("--only", default="", help="comma-separated domains to build (default: all)")
    a = ap.parse_args(argv)
    try:
        import pyarrow as pa
        pa.set_cpu_count(2); pa.set_io_thread_count(2)
    except Exception:
        pass

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    print("decontamination index …", file=sys.stderr, flush=True)
    cont = Contam(Path(a.items))
    print(f"  {len(cont.files)} item files, {len(cont.exact):,} segments, {len(cont.prefix):,} prefixes", file=sys.stderr)

    only = set(filter(None, a.only.split(",")))
    drops: dict[str, dict] = {}
    seen_rows: set[str] = set()
    YES = ("yes", "true", "بله", "نعم", "да", "ja", "oui", "sí", "evet", "是", "हाँ")
    # rows are written one domain at a time (memory); only compact records are kept for stats
    recs: list[tuple] = []           # (domain, lang, kind, mode, wrap, yes-gold?, has none, none gold, cross-lang)
    reservoir: dict[str, list[dict]] = defaultdict(list)
    n_seen_mode: Counter = Counter()
    srng = random.Random(a.seed)
    cur_dom, cur_rows = None, []

    def flush():
        if cur_dom is None:
            return
        random.Random(f"{a.seed}/write/{cur_dom}").shuffle(cur_rows)
        with open(out / f"{cur_dom}.jsonl", "w", encoding="utf-8") as f:
            for r in cur_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  wrote {len(cur_rows):,} rows → {out / (cur_dom + '.jsonl')}", file=sys.stderr, flush=True)

    for (dom, loc), quota in QUOTA.items():
        if only and dom not in only:
            continue
        if dom != cur_dom:
            flush()
            cur_dom, cur_rows = dom, []
        quota = max(1, int(quota * a.scale))
        lang = LANG_OF.get(loc, loc)
        key = f"{dom}/{lang}"
        print(f"{key}: loading …", file=sys.stderr, flush=True)
        pool, cat = load_source(dom, loc)
        n_pool = len(pool)
        clean, n_cont, n_dup, seen_txt = [], 0, 0, set()
        for it in pool:
            if cont.hit([it["text"]] + [c for c in it["check"] if c]):
                n_cont += 1
                continue
            k = norm(it["text"])
            if k in seen_txt:
                n_dup += 1
                continue
            seen_txt.add(k)
            clean.append(it)
        rng = random.Random(f"{a.seed}/{key}")
        rng.shuffle(clean)
        del pool
        # render slots: pass 0 over every text, pass 1 …; rare clinc oos texts get extra passes up front
        reps = -(-quota // max(len(clean), 1))
        slots = [(it, r) for it in clean if it["y"] == OOS for r in range(reps, reps + OOS_EXTRA_RENDERS)]
        slots += [(it, r) for r in range(reps) for it in clean]
        n_rows, used = 0, set()
        last_mode: dict[int, str] = {}
        for it, rep in slots:
            if n_rows >= quota:
                break
            rrng = random.Random(f"{a.seed}/{key}/{it['text']}/{rep}")
            row, mode, wrap = render(it, cat, rrng, avoid=last_mode.get(id(it)))
            last_mode[id(it)] = mode
            if row["id"] in seen_rows:
                continue
            seen_rows.add(row["id"]); used.add(id(it))
            cur_rows.append(row); n_rows += 1
            yes = row["choices"][row["gold"]].split(":")[0].strip().lower() in YES if row["kind"] == "verify" else None
            ni = row["none_idx"]
            recs.append((dom, lang, row["kind"], mode, wrap, yes, ni is not None, ni is not None and ni == row["gold"],
                         row["state_lang"] != row["choice_lang"]))
            n_seen_mode[mode] += 1          # reservoir sample (≤ 40 per mode) for the printout
            res = reservoir[mode]
            if len(res) < 40:
                res.append(row)
            elif (j := srng.randrange(n_seen_mode[mode])) < 40:
                res[j] = row
        drops[key] = {"pool": n_pool, "contaminated": n_cont, "duplicate_texts": n_dup, "clean": len(clean),
                      "texts_used": len(used), "rows": n_rows}
        print(f"  {key}: pool {n_pool:,}  contaminated {n_cont:,}  dup {n_dup:,}  used {len(used):,}  rows {n_rows:,}", file=sys.stderr, flush=True)
    flush()
    cur_rows = []

    # stats
    total = len(recs)
    dom_c = Counter(r[0] for r in recs)
    lang_c = Counter(r[1] for r in recs)
    kind_c = Counter(r[2] for r in recs)
    mode_c = Counter(r[3] for r in recs)
    wrap_c = Counter(r[4] for r in recs)
    cell = Counter((r[0], r[1], r[2], r[3]) for r in recs)
    ver = [r for r in recs if r[5] is not None]
    yes_share = sum(r[5] for r in ver) / max(len(ver), 1)
    n_none, n_none_gold, n_cross = sum(r[6] for r in recs), sum(r[7] for r in recs), sum(r[8] for r in recs)
    stats = {
        "total": total, "by_domain": dict(dom_c.most_common()), "by_lang": dict(lang_c.most_common()),
        "by_mode": dict(mode_c.most_common()), "by_kind": dict(kind_c), "by_state_wrap": dict(wrap_c.most_common()),
        "verify_yes_share": round(yes_share, 3), "rows_with_none": n_none, "none_is_gold": n_none_gold,
        "cross_lang_rows": n_cross, "sources": drops,
        "cells": [{"domain": d, "lang": l, "kind": k, "mode": m, "n": n} for (d, l, k, m), n in sorted(cell.items())],
    }
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1))

    print(f"\n=== {total:,} rows → {out}/ ===")
    print("by domain: " + "  ".join(f"{d} {n:,}" for d, n in dom_c.most_common()))
    print("by lang:   " + "  ".join(f"{l} {n:,}" for l, n in lang_c.most_common()))
    print("by mode:   " + "  ".join(f"{m} {n:,} ({n / total:.1%})" for m, n in mode_c.most_common()))
    print("by kind:   " + "  ".join(f"{k} {n:,}" for k, n in kind_c.most_common()))
    print("by wrap:   " + "  ".join(f"{k} {n:,}" for k, n in wrap_c.most_common()))
    print(f"verify yes share {yes_share:.3f} · rows with a none option {n_none:,} (gold {n_none_gold:,}) · cross-lang {n_cross:,}")
    print("\ndecontamination / pool:")
    for k, d in drops.items():
        print(f"  {k:14s} pool {d['pool']:>7,}  contaminated {d['contaminated']:>5,}  dup {d['duplicate_texts']:>5,}  "
              f"used {d['texts_used']:>7,}  rows {d['rows']:>7,}")
    print("\ndomain × lang × kind × mode:")
    for (d, l, k, m), n in sorted(cell.items()):
        print(f"  {d:9s} {l:5s} {k:7s} {m:6s} {n:>7,}")
    print("\nsamples:")
    for m in sorted(reservoir):
        print(f"\n--- mode {m} ---")
        for r in srng.sample(reservoir[m], min(2, len(reservoir[m]))):
            s = dict(r); s["state"] = s["state"][:300]
            if len(s["choices"]) > 12:
                s["choices"] = s["choices"][:12] + [f"… (+{len(r['choices']) - 12})"]
                s["probs"] = f"[{len(r['probs'])} probs, gold {r['probs'][r['gold']]}]"
            print(json.dumps(s, ensure_ascii=False))

if __name__ == "__main__":
    main()
