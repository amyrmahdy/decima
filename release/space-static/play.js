// Decima Playground, main page: the three published models and the presets of the `decima` package, in the browser.
// The model runs in a Web Worker (worker.js → decima.js, ONNX Runtime Web); router.js shapes questions exactly as
// decima.Router does. Nothing loads until the first decision, and only the chosen model is downloaded.

import { detectLang, isCached, MODEL_FILES } from "./decima.js";
import { asState, answer, plan, pyDumps } from "./router.js";

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const params = new URLSearchParams(location.search);

// Exports pinned to the release tags (the same files as each repo's main today). ?decima-base=<url> etc. override
// a model's location for local testing.
const HUB = "https://huggingface.co/amyrmahdy";
const MODELS = {
  "decima-agent": {
    version: "2.2", params: "321M", url: `${HUB}/decima-agent/resolve/v2.2/onnx/int8/`, cache: "decima-agent-model-v2.2",
    sizes: { "encoder.onnx": 322447126, "scorer.onnx": 14436752, "decima.json": 40309, "tokenizer/tokenizer.json": 34363440, "tokenizer/tokenizer_config.json": 735 },
    blurb: "Agent gates and knowledge-graph judgments. English. Reads up to 2,048 tokens.",
  },
  "decima-base": {
    version: "2.0", params: "321M", url: `${HUB}/decima-base/resolve/v2.0/onnx/int8/`, cache: "decima-base-model-v2.0",
    sizes: { "encoder.onnx": 322447126, "scorer.onnx": 14436752, "decima.json": 45015, "tokenizer/tokenizer.json": 34363440, "tokenizer/tokenizer_config.json": 735 },
    blurb: "General decisions in many languages: triage, routing, moderation, personal data. Reads up to 512 tokens.",
  },
  "decima-small": {
    // the same URL and cache as the Decima-small playground (small.html), so a visitor downloads it once
    version: "1.1", params: "122M", url: `${HUB}/decima-small/resolve/main/onnx/int8/`, cache: "decima-small-model-v1.1",
    sizes: { "encoder.onnx": 122164246, "scorer.onnx": 4915472, "decima.json": 20151, "tokenizer/tokenizer.json": 17082987, "tokenizer/tokenizer_config.json": 596 },
    blurb: "First generation: smallest and fastest. Reads up to 512 tokens.",
  },
};
for (const [k, m] of Object.entries(MODELS)) {
  m.key = k;
  m.mb = Object.values(m.sizes).reduce((a, b) => a + b, 0) / 1e6;
  if (params.get(k)) m.url = params.get(k).replace(/\/?$/, "/");
}
const DEFAULT_MODEL = "decima-base";              // decima.Router's default for your own questions
const shortName = (repo) => String(repo || "").split("/").pop();

// ───────────────────────────── worker ─────────────────────────────

const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
let seq = 0;
const pending = new Map();
worker.onmessage = ({ data }) => {
  const p = pending.get(data.id);
  if (!p) return;
  if (data.progress) return p.onProgress?.(data.progress);
  pending.delete(data.id);
  data.error ? p.reject(new Error(data.error)) : p.resolve(data.result);
};
worker.onerror = (e) => { for (const p of pending.values()) p.reject(new Error(e.message || "worker failed")); pending.clear(); };
const call = (op, args = {}, onProgress) => new Promise((resolve, reject) => {
  const id = ++seq;
  pending.set(id, { resolve, reject, onProgress });
  worker.postMessage({ id, op, ...args });
});

const mb = (b) => (b / 1e6).toFixed(0);
let loaded = null;                 // key of the model in the worker
let loading = null;                // {key, promise}
const onDevice = {};               // key → true when its files are in the browser cache

function showLoader(m, p) {
  const L = $("#loader"), bar = $("#load-bar"), txt = $("#load-text"), num = $("#load-num");
  L.hidden = false;
  L.classList.remove("is-ready", "is-error");
  const name = `${m.key} ${m.version}`;
  if (!p) { txt.innerHTML = `<b>Loading ${name}</b>`; num.textContent = ""; bar.classList.add("is-indeterminate"); return; }
  if (p.cached || (p.total && p.loaded >= p.total)) {
    bar.classList.add("is-indeterminate");
    bar.firstElementChild.style.width = "";
    txt.innerHTML = p.cached ? `<b>Loading ${name} from this device</b>, saved on an earlier visit.`
                             : `<b>Preparing ${name}</b>: compiling it for this browser.`;
    num.textContent = `${mb(p.total)} MB`;
    return;
  }
  const frac = p.total ? Math.min(1, p.loaded / p.total) : 0;
  bar.classList.remove("is-indeterminate");
  bar.firstElementChild.style.width = `${(frac * 100).toFixed(1)}%`;
  bar.setAttribute("aria-valuenow", Math.round(frac * 100));
  txt.innerHTML = `<b>Downloading ${name}</b>, once; after that it is cached on this device.`;
  num.textContent = `${mb(p.loaded)} / ${mb(p.total)} MB · ${Math.round(frac * 100)}%`;
}

/** Load a model into the worker (one at a time: switching releases the previous one). */
function ensureModel(key) {
  if (loaded === key && !loading) return Promise.resolve();
  if (loading?.key === key) return loading.promise;
  const m = MODELS[key];
  const t0 = performance.now();
  showLoader(m, null);
  const prev = loading?.promise || Promise.resolve();
  const promise = prev.catch(() => {}).then(() => call("load", { url: m.url, cacheName: m.cache, sizes: m.sizes }, (p) => showLoader(m, p)))
    .then((s) => {
      loaded = key;
      m.info = s.cfg;
      onDevice[key] = true;
      const L = $("#loader");
      L.classList.add("is-ready");
      const secs = ((performance.now() - t0) / 1000).toFixed(1);
      $("#load-text").innerHTML = `<b>${m.key} ${m.version} ready</b> in ${secs} s · ${s.fromCache ? "loaded from this device" : `downloaded ${mb(s.bytes)} MB once, now cached`}, running in this tab`;
      $("#load-num").textContent = s.threads > 1 ? `${s.threads} threads` : "";
      renderModels();
    })
    .catch((e) => {
      loaded = null;
      const L = $("#loader");
      L.classList.add("is-error");
      $("#load-text").innerHTML = `<b>${esc(m.key)} could not be loaded.</b> ${esc(e.message)}. Try reloading, or a recent Chrome, Edge, Firefox or Safari.`;
      throw e;
    })
    .finally(() => { if (loading?.promise === promise) loading = null; });
  loading = { key, promise };
  return promise;
}

// ───────────────────────────── state ─────────────────────────────

let PRESETS = {};
let EXAMPLES = [];
let RECORDED = {};
const ui = { model: DEFAULT_MODEL, mode: "preset", preset: "triage", modelPinned: false };

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#x27;" }[c]));
}
function pct(p) {
  if (p >= 0.9995) return p < 1 ? "99.9%+" : "100%";
  return p >= 0.001 ? `${(p * 100).toFixed(1)}%` : "<0.1%";
}

// ───────────────────────────── models ─────────────────────────────

function renderModels() {
  $("#models").innerHTML = Object.values(MODELS).map((m) => {
    const sel = m.key === ui.model;
    const rec = ui.mode === "preset" && shortName(PRESETS[ui.preset]?.model) === m.key;
    const status = loaded === m.key ? "loaded in this tab" : onDevice[m.key] ? "saved on this device" : `${mb(m.mb * 1e6)} MB download, once`;
    return `
<button type="button" class="pg-model${sel ? " is-on" : ""}" role="radio" aria-checked="${sel}" data-key="${m.key}" tabindex="${sel ? 0 : -1}">
  <span class="pg-model-head"><span class="pg-model-name">${m.key}</span><span class="pg-model-ver">${m.version}</span>${rec ? '<span class="pg-rec">preset\'s choice</span>' : ""}</span>
  <span class="pg-model-blurb">${esc(m.blurb)}</span>
  <span class="pg-model-meta"><span>${m.params} parameters</span><span class="${loaded === m.key || onDevice[m.key] ? "pg-ok" : ""}">${status}</span></span>
</button>`;
  }).join("");
  const m = MODELS[ui.model];
  $("#go-note").textContent = loaded === ui.model ? "" : onDevice[ui.model] ? `${m.key} loads from this device` : `First run downloads ${m.key} (${mb(m.mb * 1e6)} MB) once`;
}

function selectModel(key, pinned = true) {
  if (!MODELS[key]) return;
  ui.model = key;
  if (pinned) ui.modelPinned = true;
  renderModels();
  renderCode();
}

// ───────────────────────────── presets / own question ─────────────────────────────

const TYPE_NAME = { choice: "choice", noul: "yes / no", score: "score" };

function presetHint(p) {
  const tpl = p.state;
  if (!tpl || tpl === "{text}") return "Plain text. A JSON object or array is used as it is.";
  const shape = pyDumps(tpl).replace('"{text}"', "your text");
  const all = new Set(Object.values(p.questions).flatMap((q) => q.requires || []));
  const tplKeys = new Set(Object.keys(tpl));
  const other = [...all].filter((k) => !tplKeys.has(k));
  return `Plain text becomes ${shape}.` + (other.length ? ` Paste a JSON object with ${other.map((k) => `"${k}"`).join(", ")} for the other questions.` : "");
}

function renderPreset() {
  const p = PRESETS[ui.preset];
  $("#preset").value = ui.preset;
  $("#preset-desc").textContent = `${p.description}. Recommended model: ${shortName(p.model)}.`;
  const qs = Object.entries(p.questions);
  $("#preset-qs").innerHTML = `<summary>${qs.length} question${qs.length > 1 ? "s" : ""}: ${qs.map(([k]) => esc(k)).join(", ")}</summary>` +
    `<ul>${qs.map(([k, q]) => `<li><b>${esc(k)}</b> <span class="dz-dim">${TYPE_NAME[q.type] || q.type}${q.requires ? ` · needs ${q.requires.map(esc).join(", ")}` : ""}</span><br>${esc(q.instructions || k)}</li>`).join("")}</ul>`;
  $("#text-hint").textContent = presetHint(p);
}

function setMode(mode) {
  ui.mode = mode;
  for (const m of ["preset", "own"]) {
    const on = m === mode;
    $(`#mode-${m}-tab`).setAttribute("aria-selected", on);
    $(`#mode-${m}-tab`).tabIndex = on ? 0 : -1;
    $(`#mode-${m}`).hidden = !on;
  }
  if (mode === "preset") renderPreset();
  else $("#text-hint").textContent = "Plain text. A JSON object or array is used as it is.";
  renderModels();
  renderCode();
}

function selectPreset(name, followModel = true) {
  ui.preset = name;
  renderPreset();
  if (followModel) { ui.model = shortName(PRESETS[name].model) || DEFAULT_MODEL; ui.modelPinned = false; }
  renderModels();
  renderCode();
}

function ownQuestions() {
  const t = $('input[name="own-type"]:checked').value;
  const instructions = $("#own-q").value.trim();
  if (!instructions) throw new Error("Write a question.");
  if (t === "noul") return { answer: { type: "noul", instructions } };
  const opts = [...new Set($("#own-opts").value.split(/\r\n|\n|\r/).map((s) => s.trim()).filter(Boolean))];
  if (t === "score") {
    if (opts.length < 2 || opts.length > 10) throw new Error("A score needs 2 to 10 levels, one per line, lowest first.");
    return { answer: { type: "score", instructions, criteria: opts } };
  }
  if (opts.length < 2) throw new Error("Give at least two options, one per line.");
  if (opts.length > 255) throw new Error("Up to 255 options.");
  const criteria = {};
  for (const o of opts) {
    const i = o.indexOf(": ");
    if (i > 0) criteria[o.slice(0, i).trim()] = o.slice(i + 2).trim(); else criteria[o] = null;
  }
  return { answer: { type: "choice", instructions, criteria } };
}

function langOf(text) {
  const v = $("#lang").value;
  if (v !== "auto") return v;
  const d = detectLang(text);
  return d === "fa" || d === "ar" ? d : "en";
}

// ───────────────────────────── code snippet ─────────────────────────────

const shq = (s) => `'${s.replace(/'/g, `'\\''`)}'`;

function renderCode() {
  const text = $("#text").value.trim() || "…";
  const lang = langOf(text);
  const model = `amyrmahdy/${ui.model}`;
  const langArg = lang !== "en" ? `, lang="${lang}"` : "";
  let py, cli;
  if (ui.mode === "preset") {
    py = `r.decide_text(${JSON.stringify(text)}, preset="${ui.preset}"${langArg})`;
    cli = `decima ${shq(text)} --preset ${ui.preset}${ui.model !== shortName(PRESETS[ui.preset]?.model) ? ` --model ${ui.model}` : ""}${lang !== "en" ? ` --lang ${lang}` : ""}`;
  } else {
    let qs;
    try { qs = ownQuestions(); } catch { qs = { answer: { type: "choice", instructions: "…", criteria: ["…", "…"] } }; }
    py = `r.predict(${JSON.stringify(text)}, ${JSON.stringify(qs, null, 1).replace(/\n\s*/g, " ").replace(/\bnull\b/g, "None")}${langArg})`;
    cli = null;
  }
  $("#code").innerHTML = esc([
    "pip install decima-ai",
    ...(cli ? ["", `# command line (int8 ONNX on CPU; downloads the model from the Hub once)`, cli] : []),
    "", "# Python",
    "from decima import Router",
    `r = Router(model="${model}")`,
    py,
  ].join("\n"));
}

// ───────────────────────────── results ─────────────────────────────

function confBadge(c) {
  const [k, t] = c >= 0.8 ? ["high", "High"] : c >= 0.5 ? ["mid", "Medium"] : ["low", "Low"];
  return `<span class="dz-badge dz-b-${k}" title="TypeSafe confidence: (K·p_max − 1) / (K − 1)">${t} confidence · ${c.toFixed(2)}</span>`;
}
function row(label, p, isTop, title = label) {
  const w = Math.max(p * 100, 0.35);
  return `<div class="dz-row${isTop ? " is-top" : ""}"><div class="dz-label" dir="auto" title="${esc(title)}">${esc(label)}</div>` +
    `<div class="dz-track"><div class="dz-fill" style="width:${w.toFixed(2)}%"></div></div><div class="dz-pct">${pct(p)}</div></div>`;
}

function answerHtml(qid, spec, a, p, keys) {
  const ins = spec.instructions ?? qid;
  let head, rows;
  if (a.type === "choice") {
    const pairs = keys.map((k, i) => [k, p[i]]).sort((x, y) => y[1] - x[1]);
    head = `<div class="pg-ans" dir="auto">${esc(a.choice)}</div>${confBadge(a.confidence)}`;
    const desc = (k) => { const d = spec.criteria && !Array.isArray(spec.criteria) ? spec.criteria[k] : null; return d ? `${k}: ${d}` : k; };
    rows = pairs.map(([k, v]) => row(k, v, k === a.choice, desc(k))).join("");
  } else if (a.type === "score") {
    const best = p.reduce((b, v, i) => (v > p[b] ? i : b), 0);
    head = `<div class="pg-ans">score ${a.score.toFixed(2)} <span class="dz-dim">of ${p.length - 1}</span></div>${confBadge(a.confidence)}`;
    rows = p.map((v, i) => row(`${i} · ${a.legend[i]}`, v, i === best)).join("");
  } else {
    const yes = a.noul >= 0.5;
    head = `<div class="pg-ans ${yes ? "pg-yes" : "pg-no"}">${yes ? "yes" : "no"}</div><span class="dz-badge ${yes ? "dz-b-high" : "dz-b-low"}">p(yes) ${a.noul.toFixed(2)}</span>`;
    rows = row("yes", p[0], yes) + row("no", p[1], !yes);
  }
  return `
<div class="dz-q pg-a">
  <div class="dz-q-head"><span class="pg-qid">${esc(qid)}</span><span class="dz-q-kind">${TYPE_NAME[a.type]}</span></div>
  <div class="dz-q-text" dir="auto">${esc(ins)}</div>
  <div class="pg-ans-row">${head}</div>
  <div class="dz-ledger">${rows}</div>
</div>`;
}

function resultHtml(res, plans, meta) {
  const warn = (res.warnings || []).map((w) => `<div class="pg-warn">${esc(w)}</div>`).join("");
  return `
<div class="dz-card dz-result pg-result">
  <div class="dz-eyebrow">${esc(res.model)} ${esc(MODELS[res.model].version)} · ${esc(meta.label)} · ${res.usage.input_tokens} tokens · ${esc(meta.lang)}</div>
  <div class="dz-meta"><span class="dz-time">decided in <b>${Math.round(meta.ms)} ms</b> in your browser${meta.cold ? ' <span class="dz-dim">(includes encoding new option sets once; they are cached now)</span>' : ""}</span></div>
  ${warn}
  <div class="pg-answers">${plans.map((x) => answerHtml(x.qid, x.spec, res.answers[x.qid], x.p, x.keys)).join("")}</div>
  <details class="pg-json"><summary>Response as <span class="dz-mono">/v1/systemone</span> JSON</summary><pre>${esc(JSON.stringify(res, null, 2))}</pre></details>
</div>`;
}
const messageHtml = (title, body) => `<div class="dz-card dz-result dz-msg"><div class="dz-eyebrow">${esc(title)}</div><p>${esc(body)}</p></div>`;
const busyHtml = (text) => `<div class="dz-card dz-result dz-msg"><span class="dz-busy">${esc(text)}</span></div>`;

// ───────────────────────────── decide ─────────────────────────────

let runToken = 0;

/** decima.Router.predict on one state, in the browser: same questions, same `/v1/systemone` answers. */
async function predict(text, { model, preset, questions, lang }, onStep = () => {}) {
  const ps = preset ? PRESETS[preset] : null;
  const state = asState(text, ps);
  const { s, qs } = plan(state, questions, ps, lang);
  await ensureModel(model);
  const specs = { ...(ps?.questions || {}), ...(questions || {}) };
  const t0 = performance.now();
  const probs = [];
  let cold = false;
  for (let i = 0; i < qs.length; i++) {
    onStep(i, qs.length);
    const r = await call("decide", { state: s, q: qs[i].q });
    probs.push(r.probs);
    cold ||= r.cold;
  }
  const ms = performance.now() - t0;
  const m = MODELS[model];
  const [n, ...qn] = await call("count", { texts: [s, ...qs.map((x) => x.q.text)] });
  const res = { model, answers: Object.fromEntries(qs.map((x, i) => [x.qid, answer(x.type, x.keys, probs[i])])),
                usage: { input_tokens: n, output_tokens: 0 }, latency_ms: Math.round(ms * 10) / 10 };
  const nq = m.info?.question_in_state ? Math.max(...qn) : 0;
  const limit = m.info?.max_state_tokens ?? 512;
  if (n + nq > limit) res.warnings = [`state is ${n} tokens; ${model} reads the first ${limit}`];
  return { res, plans: qs.map((x, i) => ({ qid: x.qid, spec: specs[x.qid], p: probs[i], keys: x.keys })), ms, cold };
}

async function decideUi() {
  const text = $("#text").value;
  const out = $("#out");
  if (!text.trim()) return (out.innerHTML = messageHtml("Missing text", "Paste a message, a ticket, a command or a sentence."));
  let questions = null;
  try { if (ui.mode === "own") questions = ownQuestions(); } catch (e) { return (out.innerHTML = messageHtml("Check the question", e.message)); }
  const lang = langOf(text);
  const token = ++runToken;
  const go = $("#go");
  go.setAttribute("aria-busy", "true");
  out.innerHTML = busyHtml(loaded === ui.model ? "Deciding…" : `Loading ${ui.model}…`);
  try {
    const r = await predict(text.trim(), { model: ui.model, preset: ui.mode === "preset" ? ui.preset : null, questions, lang },
      (i, n) => { if (token === runToken && n > 1) out.innerHTML = busyHtml(`Deciding: question ${i + 1} of ${n}…`); });
    if (token !== runToken) return;
    const label = ui.mode === "preset" ? `preset ${ui.preset}` : "your question";
    out.innerHTML = resultHtml(r.res, r.plans, { label, lang: lang === "en" ? "en / other" : lang, ms: r.ms, cold: r.cold });
    window.DZ2.last = r.res;
  } catch (e) {
    if (token === runToken) out.innerHTML = messageHtml("Could not decide", e.message);
  } finally {
    if (token === runToken) go.removeAttribute("aria-busy");
    renderModels();
  }
}

// ───────────────────────────── examples ─────────────────────────────

function exampleSummary(rec) {
  const [qid, a] = Object.entries(rec.answers)[0];
  const v = a.type === "choice" ? `<b>${esc(a.choice)}</b>` : a.type === "score" ? `<b>${a.score.toFixed(2)}</b>` : `<b>${a.noul >= 0.5 ? "yes" : "no"}</b> · ${a.noul.toFixed(2)}`;
  return `${esc(qid)} → ${v}`;
}

function galleryHtml() {
  return EXAMPLES.map((g) => `
<button class="dz-ex" data-id="${esc(g.id)}" type="button">
  <span class="dz-ex-tag">${esc(g.tag)}</span>
  <span class="dz-ex-title">${esc(g.title)}</span>
  <span class="dz-ex-state" dir="auto">${esc(g.text)}</span>
  <span class="dz-ex-out"><span dir="auto">${exampleSummary(g.recorded)}</span><span class="dz-ex-n">${esc(g.model)} · ${g.preset ? `preset ${esc(g.preset)}` : "own question"}</span></span>
</button>`).join("");
}

function loadExample(g) {
  if (g.preset) {
    setMode("preset");
    selectPreset(g.preset, false);
  } else {
    setMode("own");
    const a = g.questions.answer;
    $(`input[name="own-type"][value="${a.type}"]`).checked = true;
    $("#own-q").value = a.instructions;
    const c = a.criteria;
    $("#own-opts").value = Array.isArray(c) ? c.join("\n") : Object.entries(c || {}).map(([k, v]) => (v ? `${k}: ${v}` : k)).join("\n");
    $("#own-opts-field").hidden = a.type === "noul";
  }
  $("#text").value = g.text;
  $("#lang").value = g.lang === "fa" || g.lang === "ar" ? g.lang : "auto";
  selectModel(g.model, false);
  $$(".dz-ex").forEach((c) => c.classList.toggle("is-active", c.dataset.id === g.id));
  const f = $("#form");
  if (f.getBoundingClientRect().top < 0) f.scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
  renderCode();
  return decideUi();
}

// ───────────────────────────── page ─────────────────────────────

function initTheme() {
  const root = document.documentElement;
  $("#theme").addEventListener("click", () => {
    const dark = !root.classList.contains("dark");
    root.classList.toggle("dark", dark);
    try { localStorage.setItem("dz-theme", dark ? "dark" : "light"); } catch {}
  });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", (e) => {
    let saved = null;
    try { saved = localStorage.getItem("dz-theme"); } catch {}
    if (!saved) root.classList.toggle("dark", e.matches);
  });
}

async function main() {
  initTheme();
  const [presets, ex] = await Promise.all([fetch("presets.json").then((r) => r.json()), fetch("examples/models.json").then((r) => r.json())]);
  PRESETS = presets;
  EXAMPLES = ex.examples;
  RECORDED = ex.recorded_with;
  $("#ex-note").textContent = `Each one was run with the published Python package (${RECORDED.package}, native int8 on CPU) before we put it here; the card shows that answer. Click to load it and run it live, in your browser.`;

  $("#preset").innerHTML = Object.entries(PRESETS).map(([k, p]) => `<option value="${esc(k)}">${esc(k)} · ${esc(shortName(p.model))}</option>`).join("");
  $("#preset").addEventListener("change", (e) => selectPreset(e.target.value));
  $("#mode-preset-tab").addEventListener("click", () => setMode("preset"));
  $("#mode-own-tab").addEventListener("click", () => setMode("own"));
  $$('.pg-seg [role="tab"]').forEach((b) => b.addEventListener("keydown", (e) => {
    if (e.key === "ArrowRight" || e.key === "ArrowLeft") { e.preventDefault(); const m = ui.mode === "preset" ? "own" : "preset"; setMode(m); $(`#mode-${m}-tab`).focus(); }
  }));
  $$('input[name="own-type"]').forEach((r) => r.addEventListener("change", () => { $("#own-opts-field").hidden = r.value === "noul" && r.checked; renderCode(); }));
  for (const id of ["#own-q", "#own-opts", "#text", "#lang"]) $(id).addEventListener("input", renderCode);

  $("#models").addEventListener("click", (e) => { const b = e.target.closest("[data-key]"); if (b) selectModel(b.dataset.key); });
  $("#models").addEventListener("keydown", (e) => {
    const d = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key];
    if (!d) return;
    e.preventDefault();
    const keys = Object.keys(MODELS);
    const k = keys[(keys.indexOf(ui.model) + d + keys.length) % keys.length];
    selectModel(k);
    $(`.pg-model[data-key="${k}"]`).focus();
  });

  $("#form").addEventListener("submit", (e) => { e.preventDefault(); decideUi(); });
  $("#text").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) { e.preventDefault(); decideUi(); } });
  $("#gallery").innerHTML = galleryHtml();
  $("#gallery").addEventListener("click", (e) => { const c = e.target.closest("[data-id]"); if (c) loadExample(EXAMPLES.find((g) => g.id === c.dataset.id)); });
  $("#forget").addEventListener("click", async () => {
    const ok = await call("clearCache", { all: true }).catch(() => false);
    for (const k in onDevice) onDevice[k] = loaded === k;
    renderModels();
    $("#forget").textContent = ok ? "Removed. The next run downloads the model again." : "Nothing cached on this device.";
  });

  // first view: the first example's inputs, nothing downloaded until the visitor asks for a decision
  const first = EXAMPLES[0];
  $("#text").value = first.text;
  selectPreset(first.preset);
  setMode("preset");
  $("#out").innerHTML = `<div class="dz-card dz-result dz-msg"><div class="dz-eyebrow">Ready when you are</div>
    <p>Pick a model, a preset or your own question, and press <b>Decide</b>. The first decision downloads the chosen model
    once (${mb(MODELS["decima-small"].mb * 1e6)} MB for decima-small, ${mb(MODELS["decima-base"].mb * 1e6)} MB for decima-base or decima-agent); after that it is cached on this device.
    Or click an example below.</p></div>`;

  // which models are already saved in this browser (Cache API; no network)
  await Promise.all(Object.values(MODELS).map(async (m) => { onDevice[m.key] = await isCached(MODEL_FILES.map((f) => m.url + f), m.cache); }));
  renderModels();
}

// hook for automated checks (parity tests, screenshots)
window.DZ2 = { predict, ensureModel, call, MODELS, presets: () => PRESETS, examples: () => EXAMPLES, ready: null };
window.DZ2.ready = main();
