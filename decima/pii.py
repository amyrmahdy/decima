"""decima-pii: find and redact personal data in Arabic, Persian, Urdu, Turkish and English text. CPU, ONNX, no PyTorch.

    from decima import PII
    pii = PII.from_pretrained()                 # amyrmahdy/decima-pii at tag v0.2.1, int8 ONNX (148 MB), downloaded once
    pii = PII.from_pretrained(revision="v0.1")  # the previous model, with its own decoding (decoder="v1")
    pii.find("My name is Sara Ahmadi, کد ملی من ۰۰۱۲۳۴۵۶۷۸ است")   # [{"start", "end", "label", "text"}, ...]
    pii.redact("...")                           # "My name is [PERSON_NAME], کد ملی من [NATIONAL_ID] است"

    decima pii "..."                            # the same from the command line

Not the same thing as the `pii` preset: the preset asks decima-base a yes/no question ("does this text contain personal
data, and which kind?"); this module runs a token tagger that returns the exact spans, so they can be masked.

The model tags tokens; this module turns the tags into character spans. Two steps matter for accuracy:
- bridging: the tokenizer emits a bare space token before ASCII digits, so "5185 5272 2255 5565" is several tokens; a
  space token between two pieces of the same entity stays inside it;
- post-filter: drop spans that cannot be what they claim (a PHONE with 3 digits, an EMAIL without "@", a span that
  starts or ends inside a word).
Texts longer than the model's window are tagged in overlapping windows.

decoder="v2" (from decima-pii v0.2): a name keeps the period of a final initial ("Anna S."), and number
words count toward the digit minimum, so a phone or ID written in words survives the post-filter. decoder="v1" gives
v0.1's output exactly. The Hub model is pinned to a tag (DEFAULT_REVISION) so a model update never changes results
silently; a new tag comes with a new package version.
decoder="v2.1" (the default from v0.2.1; revision="v0.2" keeps "v2"): v2 plus a cue-gated phone backstop that can only ADD PHONE spans the model missed,
for numbers right after a phone word such as جوالي / رقم جوالي / موبايلي / شماره موبایلم / cep / my mobile (or right
before "(جوال)", "mobile" …) that fit a real phone format; see phone_backstop. With "v2" the output is v0.2's exactly.
"""

from __future__ import annotations

import json
import re
import unicodedata as _ud
from pathlib import Path

import numpy as np

DEFAULT_REPO = "amyrmahdy/decima-pii"
DEFAULT_REVISION = "v0.2.1"                # the model tag this package version was tested with
DEFAULT_ONNX = "model_nbits8.onnx"
DECODER_FOR_REVISION = {"v0.1": "v1", "v0.2": "v2"}  # every other tag uses decoder v2.1
LABELS = ["PERSON_NAME", "PHONE", "EMAIL", "NATIONAL_ID", "IBAN", "CARD", "PASSPORT", "ADDRESS", "DATE_OF_BIRTH", "IP_ADDRESS"]
TRAIL = set(".,;:!?،؛؟»)\"'۔")
LEAD = set("«\"'")
MARKS = "‎‏‌"                     # LRM, RLM, ZWNJ
DIGITS_RE = re.compile(r"[0-9٠-٩۰-۹]")
MIN_DIGITS = {"PHONE": 6, "CARD": 12, "IBAN": 10, "NATIONAL_ID": 5, "PASSPORT": 5, "DATE_OF_BIRTH": 2, "IP_ADDRESS": 4}
CLITICS = set("وبلفكس")
NUM_WORDS = set("""zero oh one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen
eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred double triple
sıfır bir iki üç dört beş altı yedi sekiz dokuz on yirmi otuz kırk elli altmış yetmiş seksen doksan yüz
صفر واحد اثنين اثنان اتنين ثلاثة تلاتة أربعة اربعة خمسة ستة سبعة ثمانية تمانية تسعة عشرة
یک دو سه چهار پنج شش هفت هشت نه ده یازده دوازده سیزده چهارده پانزده شانزده هفده هجده نوزده بیست سی چهل پنجاه شصت هفتاد هشتاد نود
صد دویست سیصد چهارصد پانصد ششصد هفتصد هشتصد نهصد
ایک تین چار پانچ چھ سات آٹھ نو""".split())
WORD_RE = re.compile(r"[^\W\d_]+")


def _is_initial_period(text: str, s: int, i: int) -> bool:
    return text[i] == "." and i - 1 >= s and text[i - 1].isalpha() and (i - 1 == s or text[i - 2] in " .\u00a0")


def _trim(text: str, s: int, e: int, label: str | None = None, initials: bool = True) -> tuple[int, int]:
    while s < e and (text[s].isspace() or text[s] in LEAD or text[s] in MARKS):
        s += 1
    while e > s and (text[e - 1].isspace() or text[e - 1] in TRAIL or text[e - 1] in MARKS):
        if initials and label == "PERSON_NAME" and _is_initial_period(text, s, e - 1):
            break                                                  # "Anna S." keeps the period of the initial
        e -= 1
    return s, e


def decode(text: str, offsets: list[tuple[int, int]], tags: list[str], initials: bool = True) -> list[dict]:
    """Token BIO tags -> character spans, bridging bare space tokens inside an entity. initials=False is decoder v1."""
    spans, cur = [], None
    for (a, b), lab in zip(offsets, tags):
        if a == b:
            continue
        if lab == "O":
            if cur and not text[a:b].strip():
                continue
            if cur:
                spans.append(cur)
                cur = None
            continue
        tag, typ = lab.split("-", 1)
        if cur and tag == "I" and cur["label"] == typ:
            cur["end"] = b
        else:
            if cur:
                spans.append(cur)
            cur = {"start": a, "end": b, "label": typ}
    if cur:
        spans.append(cur)
    out = []
    for s in spans:
        a, b = _trim(text, s["start"], s["end"], s["label"], initials)
        if a < b:
            out.append({"start": a, "end": b, "label": s["label"]})
    return out


def _num_count(v: str, spelled: bool) -> int:
    n = len(DIGITS_RE.findall(v))
    if spelled:                                                    # a spelled-out phone / ID: a number word stands for 1-3 digits
        n += 2 * sum(w.lower() in NUM_WORDS for w in WORD_RE.findall(v))
    return n


def postfilter(text: str, spans: list[dict], spelled: bool = True) -> list[dict]:
    """Drop spans with too few digits for their type (number words count for PHONE / NATIONAL_ID unless spelled=False,
    which is decoder v1), emails without '@', one-character names, and spans that cut through a word (a one-letter
    Arabic-script clitic such as و before a name is allowed)."""
    out = []
    for p in spans:
        a, b, lab = p["start"], p["end"], p["label"]
        v = text[a:b]
        if lab in MIN_DIGITS and _num_count(v, spelled and lab in ("PHONE", "NATIONAL_ID")) < MIN_DIGITS[lab]:
            continue
        if (lab == "EMAIL" and "@" not in v) or (lab in ("PERSON_NAME", "ADDRESS") and len(v) < 2):
            continue
        left = a > 0 and text[a - 1].isalnum() and text[a].isalnum() and not (text[a - 1] in CLITICS and (a < 2 or not text[a - 2].isalnum()))
        right = b < len(text) and text[b].isalnum() and text[b - 1].isalnum()
        if left or right:
            continue
        out.append(p)
    return out


# ── decoder v2.1: cue-gated phone backstop ───────────────────────────────────────────────────────────────────────────
# Runs after the model + post-filter and can only ADD PHONE spans. A digit run becomes PHONE only when (a) a phone cue
# word sits right before it (at most 3 filler words / 25 characters between, no blocker word such as order / invoice /
# ID / price near the cue) or right after it, or it follows another phone after "/", "or", ","; (b) with Arabic-Indic
# and Persian digits read as 0-9 and separators removed, it fits a national or international phone pattern (toll-free
# and hotline prefixes excluded); (c) it overlaps no span already found and is not a date or card by format, nor glued
# to letters (IBANs), and no currency or unit follows it. Kept identical in decima_pii.py, decima/pii.py, pii-core.js.

_TO_ASCII = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_LETTER = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ی": "ي", "ک": "ك", "ة": "ه", "ۀ": "ه", "ہ": "ه", "ھ": "ه"})
_AR_PRE = ("", "و", "ب", "ل", "ف", "ك", "ال", "وال", "بال", "لل", "فال", "كال", "ولل", "وب", "ول", "فب")
# "*": the word may carry up to 3 more letters (جوال* → جوالي, جوالك; phone* → phones); "~": no Arabic prefix (و ب ل ال …)
PHONE_CUES_STRONG = """جوال* موبايل* موبيل* مبايل* محمول* تلفون* تليفون* تلفن* هاتف* نقال* خلوي* بورطابل* تيليفون* واتساب* واتس* وتساب* وتس*
واتساپ* موبائل* فون* واٹس* واٹساپ* telefon* tel cep cebim* gsm whatsapp* whats* wp phone* mobile* mob cell cellphone* telephone* ph
jawal* jawwal* mobil mobilim mobilam mobaily mobayl* tlfn tlf tlfoni tlfon* tilifon* talafon* watsap* wats* whatsap*"""
PHONE_CUES_WEAK = """~رقمي ~رقمك ~رقمه ~رقمها ~رقمنا ~نمرتي ~نمرتك اتصل* تواصل* كلمني كلموني كلمي رن رنلي دق دقلي عيط* نومرو* احكيني
شمارم* شماره_ام شماره_من شماره_جديدم شماره_جديد شماره_همراه تماس* زنگ* ميرا_نمبر رابطه* كال* numaram* ara arayın arayin arayınız
arayabilirsiniz ulaşın ulasin ulaşabilirsiniz ulasabilirsiniz iletişim* iletisim* my_number new_number contact_number call contact text reach
dial بيت منزل* خانه home ev اتصال* rakmi ra2mi raqmi ra8mi kalimni kallemni kalemni shomaram* shomare_man shomare_jadidam tamas zang"""
PHONE_BLOCKERS = """طلب* اوردر* فاتور* هوي* اقام* مدني قومي* حساب* ايبان* بطاق* جواز* شحن* تتبع مرجع* تسلسل* سيريال* موديل* طراز اصدار
شكوي بلاغ* عقد حجز* تذكر* لوحه عضوي* سعر* ثمن* قيم* مبلغ* كود* رمز* التحقق تفعيل* سري* قسط* باقه جيجا كوموند* خدمه عملاء طوارئ مركز
ساخن سفارش* فاكتور* كارت* ملي كد پيگيري* شبا پاسپورت گذرنامه پرسنلي دانشجويي پرونده پلاك قيمت* سريال مدل* بليت بليط رزرو قرارداد
عضويت شارژ بسته گيگ تاييد پشتيباني اورژانس ارڈر شناختي اكاؤنٹ كارڈ پاسپورٹ ماڈل بل رسيد پيكج ٹريكنگ sipariş* siparis* fatura* kimlik
tc tckn hesap hesab* iban kart* pasaport* kargo* takip abone* sicil* seri model* fiyat* tutar* ücret* ucret* bilet* rezervasyon* poliçe*
police dosya* plaka* şifre* sifre* kod* doğrulama dogrulama vergi hizmetleri* order* invoice* account* acct card* id passport* tracking ref
reference serial imei price* cost* total amount ticket* booking* reservation policy sku otp pin password code verification bill* plan*
receipt* transaction* ssn service center centre hotline support emergency talab gheymat sefaresh نسخه version* sürüm* surum*
شناسه* بيمه* گواهينامه مالياتي وظيف* ضريب* رخصه tax license* licence* ehliyet* insurance sigorta*"""
# person-role words: block only the generic cues ("öğrenci numaram …", "customer number …" is not a phone)
PHONE_ROLE_WORDS = """müşteri* musteri* öğrenci* ogrenci* çalışan* calisan* personel* üye* uye* hasta* customer* student* employee* member*
patient* staff عميل* طالب* موظف* عضو* مريض* مشترك* مشتري* دانشجو* پرسنل* كارمند* بيمار* ملازم* گاهك*"""
PHONE_UNITS = """ريال* درهم* دينار* جنيه* ليره تومان روپے روپي tl lira sar aed qar kwd bhd omr egp usd dollar* riyal* toman rial rs pkr gb mb
جيجا ميجا گيگ جي kg mah adet"""
_OR_WORDS = {"او", "or", "veya", "ya", "يا", "يان"}
_INTL = {"966": r"5\d{8}|1[1-7]\d{7}", "971": r"5\d{8}|[2-9]\d{7}", "974": r"[3-7]\d{7}", "965": r"[2569]\d{7}", "973": r"[136]\d{7}",
         "968": r"[279]\d{7}", "20": r"1[0125]\d{8}|[2-9]\d{7,8}", "962": r"7[789]\d{7}|[2-6]\d{7}", "212": r"[5-7]\d{8}",
         "98": r"9\d{9}|[1-8]\d{9}", "92": r"3\d{9}|[2-9]\d{8,9}", "90": r"5\d{9}|[2-4]\d{9}", "44": r"7\d{9}|[1-3]\d{8,9}",
         "1": r"[2-9]\d{2}[2-9]\d{6}", "961": r"[3-9]\d{6,7}", "963": r"9\d{8}|[1-5]\d{7,8}", "964": r"7\d{9}|[1-6]\d{7,8}",
         "967": r"7\d{8}|[1-6]\d{6,7}", "213": r"[5-7]\d{8}|[2-4]\d{7}", "216": r"[2-9]\d{7}", "218": r"9\d{8}|[2-8]\d{7,8}",
         "249": r"[19]\d{8}", "970": r"5\d{8}", "91": r"[6-9]\d{9}", "93": r"7\d{8}"}
_INTL_RE = {k: re.compile(v) for k, v in _INTL.items()}
_CC = sorted(_INTL, key=len, reverse=True)
_CC_NOPLUS = ("966", "971", "974", "965", "973", "968", "20", "962")       # written without "+" (WhatsApp exports, forms)
_NATIONAL_RE = re.compile(r"05\d{8}|01[1-7]\d{7}|5\d{8}|0[2-9]\d{7}|[1-79]\d{7}|01[0125]\d{8}|0[2-9]\d{8}|07[789]\d{7}|0[5-7]\d{8}"
                          r"|09\d{9}|0[1-8]\d{9}|03\d{9}|0[2-9]\d{9}|05\d{9}|0[2-4]\d{9}|5\d{9}|07\d{9}|0[1-3]\d{8,9}"
                          r"|1?[2-9]\d{2}[2-9]\d{6}|07\d{9}|[6-9]\d{9}")
_TOLL_FREE = ("800", "0800", "080", "0850", "0845", "0870", "0871", "0900", "9200", "1800", "1888", "1877", "1866", "1855", "1844",
              "888", "877", "866", "855", "844", "4444")
_SEPS, _PRE_BAD, _CUR, _STOPS = " \u00a0-./()", set("/=#@_&?$€£₺%"), set("$€£₺%@"), set(".,،؛;!?\n|")


def _wkey(w: str) -> str:
    """Word key for matching: lower case, marks dropped, Arabic/Persian/Urdu letter variants unified."""
    return "".join(c for c in w.lower() if not _ud.category(c).startswith("M")).translate(_LETTER)


def _cue_list(s: str) -> list[tuple]:
    out = []
    for e in s.split():
        bare = e.startswith("~")
        out.append((tuple((_wkey(w.rstrip("*")), w.endswith("*")) for w in e.lstrip("~").split("_")), not bare))
    return out


PHONE_CUES_AFTER = "تماس* زنگ* numaralı_telefon* numarali_telefon* nolu_telefon* numaralı_cep*"   # may also follow the number
_STRONG, _WEAK, _BLOCK, _UNITS = _cue_list(PHONE_CUES_STRONG), _cue_list(PHONE_CUES_WEAK), _cue_list(PHONE_BLOCKERS), _cue_list(PHONE_UNITS)
_AFTER, _ROLE = _STRONG + _cue_list(PHONE_CUES_AFTER), _cue_list(PHONE_ROLE_WORDS)


def _wmatch(key: str, pat: str, star: bool, pre: bool) -> bool:
    for p in (_AR_PRE if pre else ("",)):
        if key.startswith(p):
            rest = key[len(p):]
            if rest == pat or (star and rest.startswith(pat) and len(rest) - len(pat) <= 3):
                return True
    return False


def _tokchar(c: str) -> bool:
    return c.isalpha() or c.isdecimal() or _ud.category(c).startswith("M")


def _tokens(t: str) -> list[tuple]:
    """(start, end, key, has_letter) for every run of letters, marks and digits."""
    out, i, n = [], 0, len(t)
    while i < n:
        if _tokchar(t[i]):
            j = i
            while j < n and _tokchar(t[j]):
                j += 1
            w = t[i:j]
            out.append((i, j, _wkey(w), any(c.isalpha() for c in w)))
            i = j
        else:
            i += 1
    return out


def _phrase_at(toks: list, k: int, lst: list, forward: bool = False):
    """Start index of an entry of lst whose last word is token k (forward: whose first word is token k), else -1."""
    for words, pre in lst:
        L = len(words)
        a = k if forward else k - L + 1
        if a < 0 or a + L > len(toks):
            continue
        ok = True
        for m in range(L):
            if m and toks[a + m][0] - toks[a + m - 1][1] > 2:
                ok = False; break
            if not _wmatch(toks[a + m][2], words[m][0], words[m][1], pre):
                ok = False; break
        if ok:
            return a
    return -1


def _is_phone(d: str, intl: bool) -> bool:
    if intl:
        s = d[2:] if d.startswith("00") else d
        for cc in _CC:
            if s.startswith(cc):
                nsn = s[len(cc):]
                if nsn.startswith("0"):
                    nsn = nsn[1:]                                       # "+44 (0) 7700 …"
                return _INTL_RE[cc].fullmatch(nsn) is not None
        return 8 <= len(s) <= 15 and s[0] != "0"
    if d.startswith(_TOLL_FREE):
        return False
    if _NATIONAL_RE.fullmatch(d):
        return True
    return any(d.startswith(cc) and _INTL_RE[cc].fullmatch(d[len(cc):]) is not None for cc in _CC_NOPLUS)


def _sep_ok(s: str) -> bool:
    return (1 <= len(s) <= 3 and all(c in _SEPS for c in s) and sum(c in "-./" for c in s) <= 1 and s.count("(") <= 1
            and s.count(")") <= 1 and sum(c in " \u00a0" for c in s) <= 2)


def _date_like(t: str, grp: list) -> bool:
    if len(grp) != 3:
        return False
    seps = [t[grp[i][1]:grp[i + 1][0]] for i in range(2)]
    if not all(len(s) == 1 and s in "/.-" for s in seps):
        return False
    L = [b - a for a, b in grp]
    return (L[0] <= 2 and L[1] <= 2 and L[2] in (2, 4)) or (L[0] == 4 and L[1] <= 2 and L[2] <= 2)


def _blocked(t: str, toks: list, p: int, weak: bool = False) -> bool:
    """A blocker word (for a generic cue also a person-role word) among the words right before position p (same clause:
    no . , ; ! ? or newline in between, at most 24 characters back)?"""
    q = max([k for k, x in enumerate(toks) if x[1] <= p], default=-1)
    nxt = p
    while q >= 0 and toks[q][0] >= p - 24 and toks[q][3] and not any(c in _STOPS for c in t[toks[q][1]:nxt]):
        if _phrase_at(toks, q, _BLOCK, True) == q or weak and _phrase_at(toks, q, _ROLE, True) == q:
            return True
        nxt = toks[q][0]
        q -= 1
    return False


def phone_backstop(text: str, spans: list[dict]) -> list[dict]:
    """Decoder v2.1: spans + cue-gated PHONE spans the model missed (see the comment above). Returns `spans` itself
    when nothing is added."""
    t = text.translate(_TO_ASCII)
    n = len(t)
    groups, i = [], 0
    while i < n:
        if "0" <= t[i] <= "9":
            j = i
            while j < n and "0" <= t[j] <= "9":
                j += 1
            groups.append((i, j)); i = j
        else:
            i += 1
    if not groups:
        return spans
    runs, cur = [], [groups[0]]
    for g in groups[1:]:
        if _sep_ok(t[cur[-1][1]:g[0]]):
            cur.append(g)
        else:
            runs.append(cur); cur = [g]
    runs.append(cur)
    toks = None
    taken = [(s["start"], s["end"]) for s in spans]
    phone_ends = [s["end"] for s in spans if s["label"] == "PHONE"]
    added = []
    for run in runs:
        if len(run) >= 4 and sum(b - a == 4 for a, b in run) >= 3:              # 4-4-4-4: a card or an account number
            continue
        if len(run) >= 3 and _date_like(t, run[:3]) or sum("." in t[x[1]:y[0]] for x, y in zip(run, run[1:])) >= 3:
            continue                                                             # a date / timestamp; an IP or version
        k = 0
        while k < len(run):
            a0 = run[k][0]
            s = a0
            plus = k == 0 and s > 0 and t[s - 1] == "+"
            if plus:
                s -= 1
            if k == 0 and s > 0 and t[s - 1] == "(" and ")" in t[a0:run[-1][1]]:
                s -= 1
            best = -1
            for j in range(len(run) - 1, k - 1, -1):
                d = "".join(t[x:y] for x, y in run[k:j + 1])
                if not _date_like(t, run[k:j + 1]) and _is_phone(d, plus or d.startswith("00")):
                    best = j; break
            if best < 0:
                break
            e = run[best][1]
            first, k = k == 0, best + 1
            if s > 0 and (t[s - 1].isalnum() or t[s - 1] in _PRE_BAD) or e < n and (t[e].isalnum() or t[e] in _CUR):
                continue
            if first and s > 1 and t[s - 1] in "-./" and t[s - 2].isalnum():       # "INV-2026-7741", "HSP-27321011"
                continue
            if e + 1 < n and t[e] == ":" and "0" <= t[e + 1] <= "9":                # a time "… 09:14"
                continue
            if any(x < e and s < y for x, y in taken):
                continue
            if toks is None:
                toks = _tokens(t)
            after = [x for x in toks if x[0] >= e]
            if after and after[0][0] - e <= 2 and _phrase_at(toks, toks.index(after[0]), _UNITS, True) >= 0:
                continue
            before = [q for q, x in enumerate(toks) if x[1] <= s]
            cued = False
            pe = max([p for p in phone_ends if p <= s], default=-1)
            if pe >= 0 and s - pe <= 8:                                          # "… 0551234567 / 0501234567", "… or …"
                gap = [x for x in toks if x[0] >= pe and x[1] <= s]
                cued = all(x[2] in _OR_WORDS for x in gap) and all(c in " \u00a0,،/-|;؛\n" for c in t[pe:s] if not _tokchar(c))
            if not cued and before:
                last = before[-1]
                for g in range(0, 4):                                            # g filler words between cue and number
                    c = last - g
                    if c < 0 or s - toks[c][1] > 25:
                        break
                    gap = toks[c + 1:last + 1]
                    if any(not x[3] or _phrase_at(toks, q, _BLOCK, True) == q for q, x in zip(range(c + 1, last + 1), gap)):
                        break
                    a, weak = _phrase_at(toks, c, _STRONG), False
                    if a < 0:
                        a, weak = _phrase_at(toks, c, _WEAK), True
                    if a >= 0:                                                   # no clause break between cue and number
                        gtxt = t[toks[c][1]:s]                                   # ("Tel. 0551…" keeps its period)
                        cued = (not any(ch in ";؛!?" for ch in gtxt) and "." not in (gtxt[1:] if not weak else gtxt)
                                and not (weak and any(_phrase_at(toks, q, _ROLE, True) == q for q in range(c + 1, last + 1)))
                                and not _blocked(t, toks, toks[a][0], weak))
                        break
            if not cued and after and after[0][0] - e <= 12 and "\n" not in t[e:after[0][0]]:   # "0551234567 (جوال)": cue right after,
                q = toks.index(after[0])                                         # unless a number follows that cue
                if _phrase_at(toks, q, _AFTER, True) == q and all(x[3] for x in toks[q + 1:q + 4]):
                    cued = not _blocked(t, toks, s)
            if cued:
                added.append({"start": s, "end": e, "label": "PHONE"})
                taken.append((s, e)); phone_ends.append(e)
    if not added:
        return spans
    return sorted(spans + added, key=lambda x: x["start"])


def window_starts(n: int, w: int, stride: int) -> list[int]:
    """Start offsets of the overlapping windows that cover n tokens with windows of w tokens (n > w)."""
    return list(range(0, n - w, stride)) + [n - w]


def merge_windows(n: int, w: int, stride: int, tag_fn) -> list[int]:
    """Tag n tokens in overlapping windows; each token keeps the tag from the window where it is most central.
    tag_fn(start, end) -> tags for tokens[start:end]."""
    pred, best = [0] * n, [-1.0] * n
    for s in window_starts(n, w, stride):
        for k, lab in enumerate(tag_fn(s, s + w)):
            centrality = min(k, w - 1 - k)
            if centrality > best[s + k]:
                best[s + k], pred[s + k] = centrality, lab
    return pred


def redact_spans(text: str, spans: list[dict], fmt: str = "[{label}]") -> str:
    """Replace each span with fmt (keys: label, text); spans must not overlap."""
    for s in sorted(spans, key=lambda s: -s["start"]):
        text = text[:s["start"]] + fmt.format(label=s["label"], text=text[s["start"]:s["end"]]) + text[s["end"]:]
    return text


class DecimaPII:
    """Token tagger for personal data. `find(text)` -> spans, `redact(text)` -> text with spans replaced."""

    def __init__(self, path: str | Path, onnx: str = DEFAULT_ONNX, threads: int = 4, window: int = 256, stride: int = 192,
                 decoder: str = "v2.1"):
        if decoder not in ("v1", "v2", "v2.1"):
            raise ValueError(f"decoder must be 'v1', 'v2' or 'v2.1', not {decoder!r}")
        import onnxruntime as ort
        from tokenizers import Tokenizer

        path = Path(path)
        self.path = path
        self.tok = Tokenizer.from_file(str(path / "tokenizer.json"))
        self.tok.no_truncation()                                       # the file carries a 256-token cut; we window instead
        self.tok.no_padding()
        self.labels = json.loads((path / "pii_meta.json").read_text())["labels"]
        so = ort.SessionOptions()
        so.intra_op_num_threads, so.inter_op_num_threads = threads, 1
        self.sess = ort.InferenceSession(str(path / "onnx" / onnx), so, providers=["CPUExecutionProvider"])
        self.window, self.stride = window, stride
        self.decoder = decoder
        self.cls, self.sep = self.tok.token_to_id("[CLS]"), self.tok.token_to_id("[SEP]")
        if self.cls is None or self.sep is None:                       # take them from an encoded empty string
            e = self.tok.encode("")
            self.cls, self.sep = e.ids[0], e.ids[-1]

    @classmethod
    def from_pretrained(cls, name_or_path: str | Path = DEFAULT_REPO, onnx: str = DEFAULT_ONNX, threads: int = 4,
                        local_files_only: bool = False, revision: str | None = None, decoder: str | None = None,
                        **kw) -> "DecimaPII":
        """A local directory (tokenizer.json, pii_meta.json, onnx/<onnx>) or a Hugging Face repo id.
        revision: the Hub tag or commit; for the default repo it defaults to DEFAULT_REVISION ("v0.2.1"), so the files
        never change under you. revision="v0.1" loads the previous model. decoder: "v1", "v2" or "v2.1"; by default "v1" for
        v0.1, "v2" for v0.2 and "v2.1" otherwise. local_files_only=True (or HF_HUB_OFFLINE=1) never touches the network: a repo id must
        already be cached."""
        p = Path(name_or_path)
        if not p.exists():
            from huggingface_hub import snapshot_download

            if revision is None and str(name_or_path) == DEFAULT_REPO:
                revision = DEFAULT_REVISION
            p = Path(snapshot_download(str(name_or_path), revision=revision,
                                       allow_patterns=["tokenizer.json", "pii_meta.json", f"onnx/{onnx}"],
                                       local_files_only=local_files_only))
        decoder = decoder or DECODER_FOR_REVISION.get(revision or "", "v2.1")
        return cls(p, onnx=onnx, threads=threads, decoder=decoder, **kw)

    def _tags(self, ids: list[int]) -> list[int]:
        x = np.array([[self.cls] + ids + [self.sep]], dtype=np.int64)
        logits = self.sess.run(None, {"input_ids": x, "attention_mask": np.ones_like(x)})[0][0]
        return logits[1:-1].argmax(-1).tolist()

    def find(self, text: str) -> list[dict]:
        """Personal-data spans: [{"start", "end", "label", "text"}], character offsets into `text`."""
        enc = self.tok.encode(text, add_special_tokens=False)
        ids, offsets = enc.ids, [tuple(o) for o in enc.offsets]
        n, w = len(ids), self.window - 2
        if n <= w:
            pred = self._tags(ids) if n else []
        else:
            pred = merge_windows(n, w, self.stride, lambda a, b: self._tags(ids[a:b]))
        v2 = self.decoder != "v1"
        spans = postfilter(text, decode(text, offsets, [self.labels[p] for p in pred], initials=v2), spelled=v2)
        if self.decoder == "v2.1":
            spans = phone_backstop(text, spans)
        for s in spans:
            s["text"] = text[s["start"]:s["end"]]
        return spans

    def redact(self, text: str, fmt: str = "[{label}]") -> str:
        """`text` with every span replaced by fmt ("[{label}]" by default; "{label}" and "{text}" are available)."""
        return redact_spans(text, self.find(text), fmt)

    def __call__(self, text: str) -> list[dict]:
        return self.find(text)


PII = DecimaPII
