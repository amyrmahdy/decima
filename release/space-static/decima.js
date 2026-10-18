// Decima runtime for the browser — a line-for-line port of decima/runtime.py (DecimaOnnx), decima/render.py and
// decima/normalize.py (decima-ai 2.2.0) on ONNX Runtime Web (WASM) + tokenizers.js. No server: everything runs here.
// Works for every published model: decima-small (multilingual-e5-small) and decima-base / decima-agent (mmBERT-base).
//
//   const d = await Decima.load(MODEL_URL, {ort, Tokenizer, onProgress});
//   d.decide("My card was stolen", {text: "What does the customer want?", choices: ["block card", "check balance"],
//            kind: "choose", lang: "en"})   // → {probs, choices, latency_ms, cold}

// ───────────────────────── normalize.py ─────────────────────────

const TASHKEEL = /[ؐ-ًؚ-ٰٟۖ-ۭ]/gu;
const TATWEEL = /ـ/g;
const DIGITS = {};
[..."٠١٢٣٤٥٦٧٨٩"].forEach((c, i) => (DIGITS[c] = String(i)));
[..."۰۱۲۳۴۵۶۷۸۹"].forEach((c, i) => (DIGITS[c] = String(i)));
const TO_PERSIAN = { "ي": "ی", "ى": "ی", "ك": "ک", "ة": "ه", "ؤ": "و", "إ": "ا", "أ": "ا" };
const TO_ARABIC = { "ی": "ي", "ک": "ك" };
// Python's str.strip() whitespace set (differs from JS trim(): includes \x1c-\x1f and \x85, excludes ﻿)
const PY_WS = "\\t\\n\\x0b\\x0c\\r\\x1c-\\x1f \\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000";
const PY_STRIP = new RegExp(`^[${PY_WS}]+|[${PY_WS}]+$`, "g");
const pyStrip = (s) => s.replace(PY_STRIP, "");
const PY_SPLIT = new RegExp(`[${PY_WS}]+`);
const pySplit = (s) => s.split(PY_SPLIT).filter(Boolean);              // str.split()
const pyLen = (s) => [...s].length;                                      // len(str): code points
const mapChars = (s, table) => s.replace(/./gsu, (c) => table[c] ?? c);

export function normalize(text, lang = "en") {
  text = text.normalize("NFC");
  if (lang !== "fa" && lang !== "ar") return pyStrip(text);
  text = text.replace(TASHKEEL, "").replace(TATWEEL, "");
  text = mapChars(text, DIGITS);
  text = mapChars(text, lang === "fa" ? TO_PERSIAN : TO_ARABIC);
  return pyStrip(text.replace(/[ \t]+/g, " "));
}

// Only Persian vs Arabic changes what the model sees; the rest is for display (same as app.py detect_lang).
export function detectLang(text) {
  const n = (re) => (text.match(re) || []).length;
  const counts = {
    ar: n(/[؀-ۿ]/g), ru: n(/[Ѐ-ӿ]/g), zh: n(/[一-鿿]/g), ja: n(/[぀-ヿ]/g),
    ko: n(/[가-힯]/g), hi: n(/[ऀ-ॿ]/g), el: n(/[Ͱ-Ͽ]/g), th: n(/[฀-๿]/g),
  };
  let best = "ar";
  for (const k in counts) if (counts[k] > counts[best]) best = k;
  if (counts[best] === 0) return "latn";
  if (best === "ar") return /[پچژگکی]/.test(text) ? "fa" : "ar";
  if (best === "zh" && counts.ja) return "ja";
  return best;
}

// ───────────────────────── math (runtime.py helpers) ─────────────────────────

function logSoftmax(x) {
  const m = Math.max(...x);
  const y = x.map((v) => v - m);
  const lse = Math.log(y.reduce((a, v) => a + Math.exp(v), 0));
  return y.map((v) => v - lse);
}
const sigmoid = (x) => 1 / (1 + Math.exp(-x));
const softplus = (x) => Math.max(x, 0) + Math.log1p(Math.exp(-Math.abs(x))); // np.logaddexp(0, x)

// ───────────────────────── model download with cache ─────────────────────────

const CACHE_NAME = "decima-small-model-v1.1";   // bump on every model release: cached files are keyed by URL

async function openCache(name = CACHE_NAME) {
  try { return typeof caches !== "undefined" ? await caches.open(name) : null; } catch { return null; }
}

/** True when every URL is in the named cache (a model downloaded on an earlier visit). */
export async function isCached(urls, name = CACHE_NAME) {
  const cache = await openCache(name);
  if (!cache) return false;
  for (const u of urls) if (!(await cache.match(u).catch(() => null))) return false;
  return true;
}

/** Fetch files, reporting combined progress; cached with the Cache API so the next visit skips the network. */
export async function fetchFiles(files, onProgress = () => {}, cacheName = CACHE_NAME) {
  const cache = await openCache(cacheName);
  const state = files.map((f) => ({ loaded: 0, total: f.size || 0, cached: false }));
  const report = () => {
    const loaded = state.reduce((a, s) => a + s.loaded, 0), total = state.reduce((a, s) => a + s.total, 0);
    onProgress({ loaded, total, cached: state.every((s) => s.cached) });
  };
  return Promise.all(files.map(async (f, i) => {
    const s = state[i];
    let res = cache ? await cache.match(f.url).catch(() => null) : null;
    if (res) {
      const buf = await res.arrayBuffer();
      Object.assign(s, { loaded: buf.byteLength, total: buf.byteLength, cached: true });
      report();
      return buf;
    }
    res = await fetch(f.url, { mode: "cors" });
    if (!res.ok) throw new Error(`${f.url}: HTTP ${res.status}`);
    const len = Number(res.headers.get("content-length")) || 0;
    if (len && !res.headers.get("content-encoding")) s.total = len;
    const reader = res.body.getReader();
    const parts = [];
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      parts.push(value);
      s.loaded += value.byteLength;
      if (s.loaded > s.total) s.total = s.loaded;
      report();
    }
    const buf = new Uint8Array(s.loaded);
    let o = 0;
    for (const p of parts) { buf.set(p, o); o += p.byteLength; }
    s.total = s.loaded;
    report();
    if (cache) cache.put(f.url, new Response(buf.slice().buffer, { headers: { "content-type": "application/octet-stream" } })).catch(() => {});
    return buf.buffer;
  }));
}

export async function clearCache(name = CACHE_NAME) {
  try { return await caches.delete(name); } catch { return false; }
}

/** Remove every cached Decima model (all caches named decima-*). */
export async function clearAllCaches() {
  try {
    const names = (await caches.keys()).filter((n) => n.startsWith("decima-"));
    await Promise.all(names.map((n) => caches.delete(n)));
    return names.length > 0;
  } catch { return false; }
}

// ───────────────────────── runtime.py · DecimaOnnx ─────────────────────────

// approximate sizes of the int8 export, for the progress bar before Content-Length arrives
const SIZES = { "encoder.onnx": 122164246, "scorer.onnx": 4915472, "decima.json": 22464,
                "tokenizer/tokenizer.json": 17100000, "tokenizer/tokenizer_config.json": 700 };

export const MODEL_FILES = ["encoder.onnx", "scorer.onnx", "decima.json", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"];

export class Decima {
  static async load(baseUrl, { ort, Tokenizer, onProgress, numThreads = 1, cacheName = CACHE_NAME, sizes = SIZES } = {}) {
    if (!baseUrl.endsWith("/")) baseUrl += "/";
    const names = MODEL_FILES;
    const t0 = performance.now();
    let fromCache = true;
    const progress = (p) => { fromCache = p.cached; onProgress?.(p); };
    const bufs = await fetchFiles(names.map((n) => ({ url: baseUrl + n, size: sizes[n] })), progress, cacheName);
    const tDownload = performance.now() - t0;
    // The Hub counts a model download when the repo's config.json is requested; do that once per real download.
    const repoRoot = baseUrl.match(/^(.*\/resolve\/[^/]+\/)/)?.[1];
    if (!fromCache && repoRoot) fetch(repoRoot + "config.json", { method: "HEAD", mode: "cors" }).catch(() => {});
    const dec = new TextDecoder();
    const json = (b) => JSON.parse(dec.decode(b));
    const opts = { executionProviders: ["wasm"], graphOptimizationLevel: "all" };
    ort.env.wasm.numThreads = numThreads;
    const enc = await ort.InferenceSession.create(new Uint8Array(bufs[0]), opts);
    const sc = await ort.InferenceSession.create(new Uint8Array(bufs[1]), opts);
    // The runtime truncates explicitly (as transformers does per call); a truncation or padding setting saved
    // inside tokenizer.json (mmBERT's has max_length 64) must not apply to plain encodes.
    const tj = json(bufs[3]);
    delete tj.truncation;
    delete tj.padding;
    const tc = json(bufs[4]);
    const d = new Decima(ort, enc, sc, json(bufs[2]), new Tokenizer(tj, tc), tc);
    d.stats = { download_ms: tDownload, init_ms: performance.now() - t0 - tDownload,
                bytes: bufs.reduce((a, b) => a + b.byteLength, 0), fromCache };
    return d;
  }

  constructor(ort, enc, sc, cfg, tok, tokCfg = {}) {
    this.ort = ort; this.enc = enc; this.sc = sc; this.cfg = cfg; this.tok = tok;
    const o = cfg.ordinal;
    this.ord = [Float64Array.from(o.ord_g_w), o.ord_g_b, Float64Array.from(o.ord_gap_w), o.ord_gap_b];
    this.cache = new Map();
    const padTok = typeof tokCfg.pad_token === "string" ? tokCfg.pad_token : tokCfg.pad_token?.content;
    this.pad = (padTok && tok.token_to_id(padTok)) ?? tok.token_to_id("<pad>") ?? 1;
    // mmBERT (Gemma tokenizer): spaces become "▁" and Metaspace splits before every "▁", so a run of spaces is one
    // piece per space in the Rust tokenizer, while tokenizers.js keeps the run together. Encoding the text in
    // segments that each start at the second space of a run gives the Rust pieces (each segment's leading space is
    // its own prefix, so no extra "▁" is prepended).
    this.splitRuns = !!cfg.backbone?.includes("mmBERT");
  }

  /** Token ids with special tokens, as the Python (Rust) tokenizer gives them. */
  encodeIds(text) {
    if (!this.splitRuns || !text.includes("  ")) return this.tok.encode(text).ids;
    const segs = text.split(/(?<= )(?= )/);
    const first = this.tok.encode(segs[0]).ids;
    const rest = segs.slice(1).flatMap((t) => this.tok.encode(t, { add_special_tokens: false }).ids);
    return [...first.slice(0, -1), ...rest, first[first.length - 1]];
  }

  async release() {
    await Promise.all([this.enc.release?.(), this.sc.release?.()].filter(Boolean)).catch(() => {});
  }

  /** Number of tokens (with special tokens) the model sees for `text`, untruncated. */
  count(text) {
    return this.encodeIds(text).length;
  }

  /** Token ids with special tokens, truncated like HF `truncation=True, max_length` (right side, keeping the
   *  closing special token: </s> for XLM-R / e5, <eos> for mmBERT). */
  ids(text, maxLen) {
    const ids = this.encodeIds(text);
    return ids.length <= maxLen ? ids : [...ids.slice(0, maxLen - 1), ids[ids.length - 1]];
  }

  /** render.py · choice_texts: "question option", cut to max_choice_tokens. fit (choice_fit models) keeps the
   *  option whole and shortens only this copy of the question; earlier models keep their right-hand cut unless it
   *  makes two options identical. */
  choiceTexts(qt, choices, lang) {
    const pre = this.cfg.choice_prefix ?? "passage: ", L = this.cfg.max_choice_tokens;
    if (!this.cfg.choice_fit) {
      const legacy = choices.map((c) => pre + normalize(pyStrip(`${qt} ${c}`), lang));
      if (new Set(legacy.map((t) => this.ids(t, L).join(","))).size === legacy.length) return legacy;
    }
    return choices.map((c) => {
      const full = pre + normalize(pyStrip(`${qt} ${c}`), lang);
      if (this.count(full) <= L) return full;   // token count only (byte fallback: rare scripts and emoji)
      const opt = normalize(c, lang);
      const words = pySplit(normalize(qt, lang));
      let lo = 0, hi = words.length;
      while (lo < hi) {                                   // the longest question head that fits
        const mid = (lo + hi + 1) >> 1;
        if (this.count(`${pre}${words.slice(0, mid).join(" ")} … ${opt}`) <= L) lo = mid;
        else hi = mid - 1;
      }
      return lo ? `${pre}${words.slice(0, lo).join(" ")} … ${opt}` : pre + opt;
    });
  }

  async encode(texts, maxLen) {
    const rows = texts.map((t) => this.ids(t, maxLen));
    const L = Math.max(...rows.map((r) => r.length));
    const ids = new BigInt64Array(rows.length * L).fill(BigInt(this.pad));
    const mask = new BigInt64Array(rows.length * L);
    rows.forEach((r, i) => r.forEach((id, j) => { ids[i * L + j] = BigInt(id); mask[i * L + j] = 1n; }));
    const T = this.ort.Tensor;
    const m = new T("int64", mask, [rows.length, L]);
    const out = await this.enc.run({ input_ids: new T("int64", ids, [rows.length, L]), attention_mask: m });
    return [out.token_states, m];
  }

  async choices(q) {
    const qt = this.cfg.question_in_choices ?? true ? q.text : "";
    const key = JSON.stringify([qt, q.choices, q.lang]);
    if (this.cache.has(key)) return [this.cache.get(key), false];
    if (this.cache.size > 256) this.cache.clear();
    const v = await this.encode(this.choiceTexts(qt, q.choices, q.lang), this.cfg.max_choice_tokens);
    this.cache.set(key, v);
    return [v, true];
  }

  stateText(state, q) {
    const body = this.cfg.question_in_state ? `${q.text}\n${state}` : state;
    return (this.cfg.state_prefix ?? "query: ") + normalize(body, q.lang);
  }

  hasChoices(q) {
    const qt = this.cfg.question_in_choices ?? true ? q.text : "";
    return this.cache.has(JSON.stringify([qt, q.choices, q.lang]));
  }

  ordinal(s, z, D) {
    const [gw, gb, pw, pb] = this.ord;
    const K = s.length;
    const sm = logSoftmax(s).map(Math.exp);
    const expected = sm.reduce((a, p, k) => a + p * k, 0) - (K - 1) / 2;
    let g = gb;
    for (let j = 0; j < D; j++) {
      let mean = 0;
      for (let k = 0; k < K; k++) mean += z[k * D + j];
      g += (mean / K) * gw[j];
    }
    g += expected;
    const gaps = [];
    for (let k = 0; k < K; k++) {
      let a = pb;
      for (let j = 0; j < D; j++) a += z[k * D + j] * pw[j];
      gaps.push(softplus(a) + 1e-3);
    }
    const total = gaps.reduce((a, b) => a + b, 0);
    const theta = [];
    let c = 0;
    for (let k = 0; k < K - 1; k++) { c += gaps[k]; theta.push(c - total / 2); }
    const sig = theta.map((t) => sigmoid(g - t));
    const p = [];
    for (let k = 0; k < K; k++) p.push((k === 0 ? 1 : sig[k - 1]) - (k === K - 1 ? 0 : sig[k]));
    return logSoftmax(p.map((v) => Math.log(Math.max(v, 1e-7))));
  }

  async decideLogits(state, q) {
    const [[ch, cm], cold] = await this.choices(q);
    const [h, m] = await this.encode([this.stateText(state, q)], this.cfg.max_state_tokens);
    const r = await this.sc.run({ state_h: h, state_mask: m, choice_h: ch, choice_mask: cm });
    const s = Array.from(r.scores.data, (v) => v / this.cfg.temperature);
    let lp;
    if (q.kind === "score") lp = this.ordinal(s, r.z.data, r.z.dims[1]);
    else if (q.kind === "rank") lp = s.map((v) => -softplus(-v));
    else lp = logSoftmax(s);
    return [lp, cold];
  }

  /** Probabilities over the question's choices (independent sigmoids for `rank`). */
  async decide(state, q) {
    const t0 = performance.now();
    const [lp, cold] = await this.decideLogits(state, q);
    let probs;
    if (q.kind === "rank") probs = lp.map(Math.exp);
    else { const m = Math.max(...lp); const e = lp.map((v) => Math.exp(v - m)); const z = e.reduce((a, b) => a + b, 0); probs = e.map((v) => v / z); }
    return { probs, choices: [...q.choices], latency_ms: performance.now() - t0, cold };
  }
}
