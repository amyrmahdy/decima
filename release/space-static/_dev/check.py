"""Browser-vs-Python check for the static Playground (not uploaded to the Space: `_dev/` is excluded).

    export PATH=$HOME/.local/bin:$PATH
    OMP_NUM_THREADS=2 uv run --no-sync --with playwright python release/space-static/_dev/check.py [export/v1i-int8]

1. Tokenizer: tokenizers.js (the tokenizer inside transformers.js) + decima.js `normalize` against the Python
   tokenizer + decima/normalize.py on 200 varied strings (20 languages incl. fa/ar/zh, edge cases, truncation).
2. Probabilities: the 12 featured gallery examples, the 20 auto-route tickets and all 40 business-case
   questions, run in headless Chromium through the page (`window.DZ`), against `DecimaOnnx` on the same export.
   ONNX Runtime Web computes MatMulNBits with float activations; native ORT honours the export's
   `accuracy_level=4` (int8 activations). So the exact reference is the same weights with accuracy_level=0,
   which this script builds in a temp dir; the shipped export is reported alongside.
3. Timing: first load (empty cache), repeat visit (Cache API), per-decision latency.
"""

from __future__ import annotations

import asyncio
import functools
import json
import random
import shutil
import sys
import tempfile
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from decima.normalize import normalize  # noqa: E402
from decima.runtime import DecimaOnnx  # noqa: E402
from decima.types import Question  # noqa: E402

EX = ROOT / "release/space-static/examples"
FEATURED = ["route-fa-informal", "route-ar", "route-zh", "route-ru", "route-es", "route-de",
            "clinc-reminder", "massive-fa", "urgency", "rank", "verify", "news"]
TEAMS = ["billing", "technical support", "sales", "account security"]


def items() -> list[dict]:
    g = json.loads((EX / "gallery.json").read_text())
    labels = {f: (EX / f).read_text().splitlines() for f in ("clinc150_intents.txt", "massive60_intents.txt")}
    by = {x["id"]: x for x in g["gallery"]}
    out = []
    for fid in FEATURED:
        x = by[fid]
        out.append({"group": "gallery", "id": fid, "state": x["state"],
                    "q": {"text": x["question"], "choices": x["choices"] or labels[x["choices_file"]], "kind": x["kind"], "lang": x["lang"]}})
    for i, t in enumerate(json.loads((EX / "tickets.json").read_text())):
        out.append({"group": "tickets", "id": f"t{i}", "state": t["text"],
                    "q": {"text": "Which team should handle this request?", "choices": TEAMS, "kind": "choose", "lang": t["lang"]}})
    for ci, c in enumerate(json.loads((EX / "business_cases.json").read_text())["cases"]):
        for k, q in enumerate(c["questions"]):
            out.append({"group": "business", "id": f"c{ci}q{k}", "state": c["state"],
                        "q": {"text": q["question"], "choices": q["choices"], "kind": q["kind"], "lang": c["state_lang"]}})
    return out


def tok_cases(tok) -> list[dict]:
    rnd = random.Random(7)
    by: dict[str, list] = {}
    for s in ("decima", "laya"):
        for line in open(ROOT / f"runs/items/{s}.jsonl"):
            r = json.loads(line)
            by.setdefault(r["lang"], []).append(r)
    raw = [("", "en"), ("   ", "en"), ("Hello\tworld\n\nnew  line", "en"), ("😀 emoji 👍🏽 test ✓ — “quotes” ‘x’ …", "en"),
           ("كيف يمكنني إلغاء بطاقتي؟ ٣٤٥ ـــ مَرْحَبًا", "ar"), ("كارت بانكي من گم شده ي ك ۱۲۳ ٤٥ ة", "fa"),
           ("我的银行卡被偷了，请帮我冻结。ABC１２３", "zh"), ("ｆｕｌｌｗｉｄｔｈ　ｔｅｘｔ ﬁ ligature ½ ①", "en"), ("Café naïve résumé Ångström ß", "de"),
           ("日本語のテキストです。カタカナ", "ja"), ("한국어 문장입니다", "ko"), ("नमस्ते, मेरा कार्ड खो गया", "hi"), ("a" * 3000, "en"),
           ("zero​width‌join‍ x nbsp", "en"), ("<s> </s> <pad> <mask> <unk>", "en"), ("اکانت من هک شده", "fa"),
           ("مرحبا ‏ RTL mark", "ar"), ("é combining", "fr")]
    for it in items():
        raw.append((it["state"], it["q"]["lang"]))
    for lang, rs in sorted(by.items()):
        raw += [(r["state"], lang) for r in rnd.sample(rs, 6)]
    out = []
    for i, (t, lang) in enumerate(raw[:200]):
        txt = ("query: " if i % 2 == 0 else "passage: ") + normalize(t, lang)
        out.append({"raw": t, "lang": lang, "text": txt, "norm": normalize(t, lang), "ids": tok(txt)["input_ids"],
                    "t512": tok([txt], truncation=True, max_length=512)["input_ids"][0],
                    "t64": tok([txt], truncation=True, max_length=64)["input_ids"][0]})
    return out


class H(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *a):
        pass


H.extensions_map.update({".mjs": "text/javascript", ".onnx": "application/octet-stream"})

TOK_JS = """async ([cases, tokUrl]) => {
  const { Tokenizer } = await import("https://cdn.jsdelivr.net/npm/@huggingface/tokenizers@0.2.0/dist/tokenizers.min.mjs");
  const { normalize } = await import("./decima.js");
  const j = (f) => fetch(tokUrl + f).then((r) => r.json());
  const tok = new Tokenizer(await j("tokenizer.json"), await j("tokenizer_config.json"));
  const eq = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);
  const trunc = (ids, L) => (ids.length <= L ? ids : [...ids.slice(0, L - 1), 2]);
  let ids = 0, norm = 0, t512 = 0, t64 = 0;
  for (const c of cases) {
    const e = tok.encode(c.text).ids;
    ids += eq(e, c.ids); t512 += eq(trunc(e, 512), c.t512); t64 += eq(trunc(e, 64), c.t64); norm += normalize(c.raw, c.lang) === c.norm;
  }
  return { n: cases.length, ids, norm, t512, t64 };
}"""


async def browser(url: str, model_url: str, its: list[dict], cases: list[dict]) -> dict:
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await (await b.new_context()).new_page()
        errors: list[str] = []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        t0 = time.time()
        await pg.goto(f"{url}?model={model_url}")
        await pg.wait_for_function("window.DZ", timeout=60000)
        stats = await pg.evaluate("DZ.ready.then(() => DZ.stats())")
        first = time.time() - t0
        tok = await pg.evaluate(TOK_JS, [cases, model_url + "tokenizer/"])
        res = await pg.evaluate("""async (items) => { const out = [];
          for (const it of items) { const r = await DZ.decide(it.state, it.q); const r2 = await DZ.decide(it.state, it.q);
            out.push({ probs: r.probs, cold: r.cold, ms: r.latency_ms, warm: r2.latency_ms }); }
          return out; }""", [{"state": i["state"], "q": i["q"]} for i in its])
        t0 = time.time()
        await pg.reload()
        await pg.wait_for_function("window.DZ", timeout=60000)
        stats2 = await pg.evaluate("DZ.ready.then(() => DZ.stats())")
        again = time.time() - t0
        await b.close()
    return {"stats": stats, "first_s": first, "stats2": stats2, "again_s": again, "tok": tok, "res": res, "errors": errors}


def main() -> None:
    export = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "export/v1i-int8")
    its = items()
    tmp = Path(tempfile.mkdtemp(prefix="decima-acc0-"))
    try:
        import onnx

        for f in ("scorer.onnx", "decima.json"):
            shutil.copy(export / f, tmp / f)
        shutil.copytree(export / "tokenizer", tmp / "tokenizer")
        m = onnx.load(str(export / "encoder.onnx"))
        for node in m.graph.node:
            for a in node.attribute:
                if node.op_type == "MatMulNBits" and a.name == "accuracy_level":
                    a.i = 0
        onnx.save(m, str(tmp / "encoder.onnx"))
        refs = {}
        for tag, d in (("shipped export (native accuracy_level=4)", export), ("same weights, accuracy_level=0", tmp)):
            rt = DecimaOnnx(d, threads=2)
            refs[tag] = [rt.decide(i["state"], Question(i["q"]["text"], i["q"]["choices"], kind=i["q"]["kind"], lang=i["q"]["lang"])).probs
                         for i in its]
        cases = tok_cases(rt.tok)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(H, directory=str(ROOT)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}/"
    r = asyncio.run(browser(base + "release/space-static/index.html", base + str(export.relative_to(ROOT)) + "/", its, cases))
    srv.shutdown()

    t = r["tok"]
    print(f"tokenizer: ids {t['ids']}/{t['n']} · truncated@512 {t['t512']}/{t['n']} · @64 {t['t64']}/{t['n']} · normalize {t['norm']}/{t['n']}")
    am = lambda p: max(range(len(p)), key=p.__getitem__)  # noqa: E731
    js = [x["probs"] for x in r["res"]]
    for tag, ref in refs.items():
        for g in ("gallery", "tickets", "business", None):
            idx = [k for k, i in enumerate(its) if g is None or i["group"] == g]
            d = max(max(abs(a - b) for a, b in zip(js[k], ref[k])) for k in idx)
            same = sum(am(js[k]) == am(ref[k]) for k in idx)
            print(f"  vs {tag:<42} {g or 'ALL':<9} n={len(idx):>2}  max|Δp| {d:.1e}  top-1 identical {same}/{len(idx)}")
    for g in ("gallery", "tickets", "business"):
        w = sorted(x["warm"] for x, i in zip(r["res"], its) if i["group"] == g)
        print(f"latency {g:<9} warm p50 {w[len(w) // 2]:.0f} ms (min {w[0]:.0f}, max {w[-1]:.0f})")
    s = r["stats"]
    print(f"first load {r['first_s']:.1f} s ({s['bytes'] / 1e6:.1f} MB, download {s['download_ms'] / 1e3:.1f} s + init {s['init_ms'] / 1e3:.1f} s, "
          f"{s['threads']} thread) · repeat visit {r['again_s']:.1f} s (cached: {r['stats2']['fromCache']})")
    print("page errors:", r["errors"] or "none")


if __name__ == "__main__":
    main()
