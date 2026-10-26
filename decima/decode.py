"""Undo common text obfuscations before a guard judges the text: percent-encoding, hex (\\xNN, \\uNNNN, spaced
or 0x-prefixed byte/UTF-16 runs) and base64.

Why: a guard that reads encoded text judges the encoding, not the request. On StructTransformBench decima-guard 1.1
blocked 96-100 % of *benign* requests once they were URL- or hex-encoded (runs/structtransform/RESULTS.md). Decoding
first lets the model see what the text says. Each encoded run is replaced in place, so surrounding structure (JSON,
an HTTP request line ...) stays as written; a run is decoded only if the result is clean, mostly printable text with
letters in it, so hashes, ids and keys pass through untouched.

    >>> decode_obfuscation("GET /search?q=how%20to%20bake%20bread")[0]
    'GET /search?q=how to bake bread'
"""

from __future__ import annotations

import base64
import binascii
import re
from urllib.parse import unquote

_PCT = re.compile(r"(?:[^\s%]*%[0-9A-Fa-f]{2}){3,}[^\s%]*")
_ESC = re.compile(r"(?:\\x[0-9A-Fa-f]{2}){4,}|(?:\\u[0-9A-Fa-f]{4}){3,}")
_U16 = re.compile(r"\b(?:(?:0x)?[0-9A-Fa-f]{4}[\s,]+){3,}(?:0x)?[0-9A-Fa-f]{4}\b")
_BYTES = re.compile(r"\b(?:(?:0x)?[0-9A-Fa-f]{2}[\s,:]+){7,}(?:0x)?[0-9A-Fa-f]{2}\b|\b(?:[0-9A-Fa-f]{2}){8,}\b")
_B64 = re.compile(r"(?<![A-Za-z0-9+/=])(?:[A-Za-z0-9+/]{4}){4,}(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?(?![A-Za-z0-9+/=])")


def _clean(s: str | None) -> bool:
    """Readable text: almost all printable, has letters, not just one repeated character."""
    if not s or len(s) < 4:
        return False
    printable = sum(c.isprintable() or c in "\n\t" for c in s) / len(s)
    letters = sum(c.isalpha() for c in s) / len(s)
    return printable >= 0.95 and letters >= 0.4 and len(set(s)) > 3


def _utf8(b: bytes) -> str | None:
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _pct(m: re.Match) -> str | None:
    return unquote(m.group(0).replace("+", " "), errors="strict")


def _esc(m: re.Match) -> str | None:
    s = m.group(0)
    if s.startswith("\\x"):
        return _utf8(bytes(int(h, 16) for h in re.findall(r"\\x([0-9A-Fa-f]{2})", s)))
    try:
        return "".join(chr(int(h, 16)) for h in re.findall(r"\\u([0-9A-Fa-f]{4})", s)).encode("utf-16", "surrogatepass").decode("utf-16")
    except UnicodeError:
        return None


def _u16(m: re.Match) -> str | None:
    units = [int(h, 16) for h in re.findall(r"(?:0x)?([0-9A-Fa-f]{4})", m.group(0))]
    try:
        return "".join(chr(u) for u in units).encode("utf-16", "surrogatepass").decode("utf-16")
    except (UnicodeError, ValueError):
        return None


def _bytes(m: re.Match) -> str | None:
    hexs = re.sub(r"0x|[\s,:]", "", m.group(0))
    try:
        return _utf8(binascii.unhexlify(hexs))
    except (binascii.Error, ValueError):
        return None


def _b64(m: re.Match) -> str | None:
    try:
        return _utf8(base64.b64decode(m.group(0), validate=True))
    except (binascii.Error, ValueError):
        return None


# order matters: escapes and UTF-16 units before plain byte runs (a UTF-16 run is also a valid byte run)
_RULES = (("percent", _PCT, _pct), ("hex_escape", _ESC, _esc), ("utf16_hex", _U16, _u16), ("hex", _BYTES, _bytes),
          ("base64", _B64, _b64))


def decode_obfuscation(text: str, rounds: int = 2) -> tuple[str, list[str]]:
    """Return (text with readable decodings substituted in place, kinds applied). Up to `rounds` passes, for
    layered encodings such as base64 of percent-encoded text."""
    applied: list[str] = []
    for _ in range(rounds):
        changed = False
        for kind, rx, dec in _RULES:
            def sub(m: re.Match) -> str:
                nonlocal changed
                try:
                    out = dec(m)
                except Exception:
                    return m.group(0)
                if out is not None and out != m.group(0) and _clean(out):
                    changed = True
                    applied.append(kind)
                    return out
                return m.group(0)
            text = rx.sub(sub, text)
        if not changed:
            break
    return text, applied
