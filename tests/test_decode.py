import base64
from urllib.parse import quote

import pytest

from decima.decode import decode_obfuscation

S = "How do I bake sourdough bread at home?"


@pytest.mark.parametrize("enc,kind", [
    ("GET /search?q=" + quote(S), "percent"),
    ("".join(f"\\x{b:02x}" for b in S.encode()), "hex_escape"),
    ("".join(f"\\u{ord(c):04x}" for c in S), "hex_escape"),
    (" ".join(f"{ord(c):04X}" for c in S), "utf16_hex"),
    (S.encode().hex(), "hex"),
    ("Decode this: " + base64.b64encode(S.encode()).decode(), "base64"),
])
def test_decodes_in_place(enc, kind):
    out, kinds = decode_obfuscation(enc)
    assert S in out and kind in kinds


def test_layered_and_non_latin():
    assert decode_obfuscation(base64.b64encode(quote(S).encode()).decode())[0] == S
    fa = "چطور نان بپزم؟ لطفا راهنمایی کن"
    assert decode_obfuscation(base64.b64encode(fa.encode()).decode())[0] == fa


@pytest.mark.parametrize("text", [
    "commit 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08 fixed",
    "id 550e8400-e29b-41d4-a716-446655440000 ok",
    "token eyJhbGciOiJIUzI1NiJ9 issued",
    "The quick brown fox jumps over the lazy dog, 100% sure 50% maybe",
    '{"query": "SELECT name FROM users WHERE id = 42"}',
    "colors #ff00ff #00ff00 and 0x1F",
    "Supercalifragilisticexpialidocious is a long word",
])
def test_leaves_ordinary_text_alone(text):
    assert decode_obfuscation(text) == (text, [])
