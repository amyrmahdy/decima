"""decima-pii: find and redact personal data in Arabic, Persian, Urdu, Turkish and English text. CPU, ONNX, no PyTorch.

    from decima import PII
    pii = PII.from_pretrained()                 # amyrmahdy/decima-pii, int8 ONNX (148 MB), downloaded once
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
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

DEFAULT_REPO = "amyrmahdy/decima-pii"
DEFAULT_ONNX = "model_nbits8.onnx"
LABELS = ["PERSON_NAME", "PHONE", "EMAIL", "NATIONAL_ID", "IBAN", "CARD", "PASSPORT", "ADDRESS", "DATE_OF_BIRTH", "IP_ADDRESS"]
TRAIL = set(".,;:!?،؛؟»)\"'۔")
LEAD = set("«\"'")
MARKS = "‎‏‌"                     # LRM, RLM, ZWNJ
DIGITS_RE = re.compile(r"[0-9٠-٩۰-۹]")
MIN_DIGITS = {"PHONE": 6, "CARD": 12, "IBAN": 10, "NATIONAL_ID": 5, "PASSPORT": 5, "DATE_OF_BIRTH": 2, "IP_ADDRESS": 4}
CLITICS = set("وبلفكس")


def _trim(text: str, s: int, e: int) -> tuple[int, int]:
    while s < e and (text[s].isspace() or text[s] in LEAD or text[s] in MARKS):
        s += 1
    while e > s and (text[e - 1].isspace() or text[e - 1] in TRAIL or text[e - 1] in MARKS):
        e -= 1
    return s, e


def decode(text: str, offsets: list[tuple[int, int]], tags: list[str]) -> list[dict]:
    """Token BIO tags -> character spans, bridging bare space tokens inside an entity."""
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
        a, b = _trim(text, s["start"], s["end"])
        if a < b:
            out.append({"start": a, "end": b, "label": s["label"]})
    return out


def postfilter(text: str, spans: list[dict]) -> list[dict]:
    """Drop spans with too few digits for their type, emails without '@', one-character names, and spans that cut
    through a word (a one-letter Arabic-script clitic such as و before a name is allowed)."""
    out = []
    for p in spans:
        a, b, lab = p["start"], p["end"], p["label"]
        v = text[a:b]
        if lab in MIN_DIGITS and len(DIGITS_RE.findall(v)) < MIN_DIGITS[lab]:
            continue
        if (lab == "EMAIL" and "@" not in v) or (lab in ("PERSON_NAME", "ADDRESS") and len(v) < 2):
            continue
        left = a > 0 and text[a - 1].isalnum() and text[a].isalnum() and not (text[a - 1] in CLITICS and (a < 2 or not text[a - 2].isalnum()))
        right = b < len(text) and text[b].isalnum() and text[b - 1].isalnum()
        if left or right:
            continue
        out.append(p)
    return out


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

    def __init__(self, path: str | Path, onnx: str = DEFAULT_ONNX, threads: int = 4, window: int = 256, stride: int = 192):
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
        self.cls, self.sep = self.tok.token_to_id("[CLS]"), self.tok.token_to_id("[SEP]")
        if self.cls is None or self.sep is None:                       # take them from an encoded empty string
            e = self.tok.encode("")
            self.cls, self.sep = e.ids[0], e.ids[-1]

    @classmethod
    def from_pretrained(cls, name_or_path: str | Path = DEFAULT_REPO, onnx: str = DEFAULT_ONNX, threads: int = 4,
                        local_files_only: bool = False, **kw) -> "DecimaPII":
        """A local directory (tokenizer.json, pii_meta.json, onnx/<onnx>) or a Hugging Face repo id.
        local_files_only=True (or HF_HUB_OFFLINE=1) never touches the network: a repo id must already be cached."""
        p = Path(name_or_path)
        if not p.exists():
            from huggingface_hub import snapshot_download

            p = Path(snapshot_download(str(name_or_path), allow_patterns=["tokenizer.json", "pii_meta.json", f"onnx/{onnx}"],
                                       local_files_only=local_files_only))
        return cls(p, onnx=onnx, threads=threads, **kw)

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
        spans = postfilter(text, decode(text, offsets, [self.labels[p] for p in pred]))
        for s in spans:
            s["text"] = text[s["start"]:s["end"]]
        return spans

    def redact(self, text: str, fmt: str = "[{label}]") -> str:
        """`text` with every span replaced by fmt ("[{label}]" by default; "{label}" and "{text}" are available)."""
        return redact_spans(text, self.find(text), fmt)

    def __call__(self, text: str) -> list[dict]:
        return self.find(text)


PII = DecimaPII
