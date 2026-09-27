"""Script normalization for Persian and Arabic.

Persian and Arabic share letters that differ only in Unicode code point (ی/ي, ک/ك …).
Left alone, the same word tokenizes differently depending on the keyboard it was typed
on. We map to the canonical form of *the language being processed*, strip tashkeel
(vowel diacritics) and tatweel, and unify digits. Other languages pass through untouched.
"""

from __future__ import annotations

import re
import unicodedata

_TASHKEEL = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
_TATWEEL = "ـ"

# Arabic-Indic (٠..٩) and Extended Arabic-Indic (۰..۹) → ASCII
_DIGITS = {ord(c): str(i) for i, c in enumerate("٠١٢٣٤٥٦٧٨٩")}
_DIGITS.update({ord(c): str(i) for i, c in enumerate("۰۱۲۳۴۵۶۷۸۹")})

_TO_PERSIAN = str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک", "ة": "ه", "ؤ": "و", "إ": "ا", "أ": "ا"})
_TO_ARABIC = str.maketrans({"ی": "ي", "ک": "ك"})


def normalize(text: str, lang: str = "en") -> str:
    text = unicodedata.normalize("NFC", text)
    if lang not in ("fa", "ar"):
        return text.strip()
    text = _TASHKEEL.sub("", text).replace(_TATWEEL, "")
    text = text.translate(_DIGITS)
    text = text.translate(_TO_PERSIAN if lang == "fa" else _TO_ARABIC)
    return re.sub(r"[ \t]+", " ", text).strip()
