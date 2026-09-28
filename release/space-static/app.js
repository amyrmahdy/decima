// Decima Playground — static version. The model runs in this browser tab (ONNX Runtime Web, WebAssembly);
// the page talks to it through a Web Worker (worker.js) so the interface stays responsive.

// Where the int8 export lives: encoder.onnx, scorer.onnx, decima.json, tokenizer/{tokenizer.json, tokenizer_config.json}.
// Override for local testing with ?model=<url>.
const MODEL_URL = "https://huggingface.co/amyrmahdy/decima-small/resolve/main/onnx/int8/";

import { detectLang } from "./decima.js";

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const params = new URLSearchParams(location.search);
const modelUrl = params.get("model") || MODEL_URL;

// ───────────────────────────── the model, in a worker ─────────────────────────────

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

const mb = (b) => (b / 1e6).toFixed(1);
let loadStats = null;
let fromCache = false;
const tLoad0 = performance.now();

const ready = call("load", { url: modelUrl }, (p) => {
  fromCache = p.cached;
  const bar = $("#load-bar"), txt = $("#load-text"), num = $("#load-num");
  if (p.cached || (p.total && p.loaded >= p.total)) {
    bar.classList.add("is-indeterminate");
    bar.firstElementChild.style.width = "";
    bar.removeAttribute("aria-valuenow");
  }
  if (p.cached) {
    txt.innerHTML = "<b>Loading the model from this device</b> — it was saved on your first visit.";
    num.textContent = `${mb(p.total)} MB`;
    return;
  }
  const frac = p.total ? Math.min(1, p.loaded / p.total) : 0;
  if (frac >= 1) {
    txt.innerHTML = "<b>Preparing the model</b> — compiling it for this browser.";
    num.textContent = `${mb(p.total)} MB`;
    return;
  }
  bar.classList.remove("is-indeterminate");
  bar.firstElementChild.style.width = `${(frac * 100).toFixed(1)}%`;
  bar.setAttribute("aria-valuenow", Math.round(frac * 100));
  txt.innerHTML = "<b>Downloading the model</b> — once; after that it is cached on this device.";
  num.textContent = `${mb(p.loaded)} / ${mb(p.total)} MB · ${Math.round(frac * 100)}%`;
}).then((s) => {
  loadStats = { ...s, fromCache, total_ms: performance.now() - tLoad0 };
  const L = $("#loader");
  L.classList.add("is-ready");
  const secs = (loadStats.total_ms / 1000).toFixed(1);
  $("#load-text").innerHTML = fromCache
    ? `<b>Model ready</b> in ${secs} s · loaded from this device, running in this tab`
    : `<b>Model ready</b> in ${secs} s · downloaded ${mb(s.bytes)} MB once, now cached on this device`;
  $("#load-num").textContent = s.threads > 1 ? `${s.threads} threads` : "";
  $("#load-foot").textContent = `This visit: model ready in ${(loadStats.total_ms / 1000).toFixed(1)} s.`;
  return s;
}).catch((e) => {
  const L = $("#loader");
  L.classList.add("is-error");
  $("#load-bar").hidden = true;
  $("#load-text").innerHTML = `<b>The model could not be loaded.</b> ${esc(e.message)} — try reloading, or a recent Chrome, Edge, Firefox or Safari.`;
  throw e;
});
ready.catch(() => {});

async function decide(state, q) {
  await ready;
  return call("decide", { state, q });
}

// ───────────────────────────── data ─────────────────────────────

const TEAMS = ["billing", "technical support", "sales", "account security"];
const TEAM_Q = "Which team should handle this request?";
const FEATURED = ["route-fa-informal", "route-ar", "route-zh", "route-ru", "route-es", "route-de",
                  "clinc-reminder", "massive-fa", "urgency", "rank", "verify", "news"];
const MAX_OPTIONS = 200;
const X86_SCALING = { 4: 16, 20: 26, 77: 75, 150: 130 };   // docs/BENCH-x86.md, short labels, p50 ms, one P-core
const LANG_NAME = { auto: "auto", latn: "Latin script", en: "English", fa: "Persian", ar: "Arabic", ru: "Russian", zh: "Chinese",
                    es: "Spanish", de: "German", fr: "French", tr: "Turkish", hi: "Hindi", ja: "Japanese", ko: "Korean",
                    ur: "Urdu", xx: "other", el: "Greek", th: "Thai" };
const DOMAIN_NAME = { ecommerce: "E-commerce", legal_contract: "Legal · contracts", compliance_privacy: "Privacy & compliance",
  it_ops: "IT operations", product_feedback: "Product feedback", rag_relevance: "Search relevance", invoice_ap: "Accounts payable",
  procurement: "Procurement", fraud: "Fraud review", sales_crm: "Sales CRM", insurance: "Insurance", real_estate: "Real estate",
  code_review: "Code review", it_helpdesk: "IT helpdesk", email_ops: "Email operations", travel_expense: "Travel & expenses",
  banking: "Banking", support_triage: "Support triage", citation_support: "Citation check", logistics: "Logistics", hr: "HR",
  data_quality: "Data quality", public_services: "Public services", education: "Education" };
const LN = { en: "English", fa: "Persian", ar: "Arabic", ru: "Russian" };

const getJSON = (p) => fetch(p).then((r) => r.json());
const getLines = (p) => fetch(p).then((r) => r.text()).then((t) => { const l = t.split(/\r\n|\n|\r/); if (l.at(-1) === "") l.pop(); return l; });

const dataReady = (async () => {
  const [gallery, business, tickets, clinc, massive, banking] = await Promise.all([
    getJSON("examples/gallery.json"), getJSON("examples/business_cases.json"), getJSON("examples/tickets.json"),
    getLines("examples/clinc150_intents.txt"), getLines("examples/massive60_intents.txt"), getLines("examples/banking77_intents.txt")]);
  const labels = { "clinc150_intents.txt": clinc, "massive60_intents.txt": massive, "banking77_intents.txt": banking };
  for (const g of gallery.gallery) if (g.choices == null) g.choices = labels[g.choices_file];
  const byId = Object.fromEntries(gallery.gallery.map((g) => [g.id, g]));
  return { gallery, business, tickets, labels, byId, cards: FEATURED.map((id) => byId[id]).filter(Boolean),
           speedSets: { 4: TEAMS, 20: clinc.slice(0, 20), 77: banking, 150: clinc } };
})();

// ───────────────────────────── HTML builders (ported from app.py) ─────────────────────────────

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#x27;" }[c]));
}
function pct(p) {
  if (p >= 0.9995) return p < 1 ? "99.9%+" : "100%";
  return p >= 0.001 ? `${(p * 100).toFixed(1)}%` : "<0.1%";
}
const argmax = (p) => p.reduce((b, v, i) => (v > p[b] ? i : b), 0);
const top = (r) => r.choices[argmax(r.probs)];
const confidence = (r) => Math.max(...r.probs);

function confBadge(p, kind) {
  if (kind === "rank") return '<span class="dz-badge dz-b-info">independent scores</span>';
  const [c, t] = p >= 0.8 ? ["high", "High"] : p >= 0.5 ? ["mid", "Medium"] : ["low", "Low"];
  return `<span class="dz-badge dz-b-${c}">${t} confidence · ${pct(p)}</span>`;
}
function scaleHtml() {
  const ticks = [0, 25, 50, 75, 100].map((t) => `<span style="left:${t}%">${t}</span>`).join("");
  return `<div class="dz-row dz-scale" aria-hidden="true"><div></div><div class="dz-ticks">${ticks}</div><div></div></div>`;
}
function row(c, p, isTop) {
  const w = Math.max(p * 100, 0.35);
  return `<div class="dz-row${isTop ? " is-top" : ""}"><div class="dz-label" dir="auto" title="${esc(c)}">${esc(c)}</div>` +
    `<div class="dz-track"><div class="dz-fill" style="width:${w.toFixed(2)}%"></div></div>` +
    `<div class="dz-pct">${pct(p)}</div></div>`;
}
function rowsHtml(pairs, topC, limit) {
  const out = [];
  for (let i = 0; i < pairs.length; i++) {
    const [c, p] = pairs[i];
    if (limit && i === limit) {
      const tail = pairs.slice(limit).map(([c2, p2]) => row(c2, p2, false)).join("");
      out.push(`<details class="dz-more"><summary>Show the other ${pairs.length - limit} options (all ≤ ${pct(pairs[limit][1])})</summary>${tail}</details>`);
      break;
    }
    out.push(row(c, p, c === topC));
  }
  return out.join("");
}
function resultHtml(r, kind, lang, detected) {
  let pairs = r.choices.map((c, i) => [c, r.probs[i]]);
  if (kind !== "score") pairs = pairs.sort((a, b) => b[1] - a[1]);
  const n = pairs.length, t = top(r);
  const orderNote = { score: "levels in the order you gave them (low → high)",
                      rank: "each option scored on its own — they need not add up to 100%" }[kind] || "sorted by probability";
  const langNote = `${LANG_NAME[lang] || lang}${detected ? " (detected)" : ""}`;
  const coldNote = r.cold ? ' <span class="dz-dim">— includes encoding this new option set once; it is cached now, run it again to see the steady-state time</span>' : "";
  return `
<div class="dz-card dz-result">
  <div class="dz-eyebrow">Decision · ${esc(kind)} · ${n} option${n !== 1 ? "s" : ""} · ${esc(langNote)}</div>
  <div class="dz-answer" dir="auto">${esc(t)}</div>
  <div class="dz-meta">${confBadge(confidence(r), kind)}
    <span class="dz-time">decided in <b>${Math.round(r.latency_ms)} ms</b> in your browser${coldNote}</span></div>
  <div class="dz-ledger">${scaleHtml()}${rowsHtml(pairs, t, n > 15 ? 12 : 0)}</div>
  <div class="dz-foot">${esc(orderNote)}</div>
</div>`;
}
const messageHtml = (title, body) => `<div class="dz-card dz-result dz-msg"><div class="dz-eyebrow">${esc(title)}</div><p>${esc(body)}</p></div>`;
const busyHtml = (text) => `<div class="dz-card dz-result dz-msg"><span class="dz-busy">${esc(text)}</span></div>`;
const waitText = () => (loadStats ? "Deciding…" : "Waiting for the model to finish loading…");

// ───────────────────────────── tab 1 · try it ─────────────────────────────

let tryToken = 0;

async function decideUi() {
  const state = $("#in-state").value.trim(), question = $("#in-question").value.trim();
  const kind = $('input[name="kind"]:checked').value, langSel = $("#in-lang").value;
  const choices = [...new Set($("#in-options").value.split(/\r\n|\n|\r/).map((c) => c.trim()).filter(Boolean))];
  const out = $("#out");
  if (!state) return (out.innerHTML = messageHtml("Missing situation", "Describe the situation: a message, a ticket, a document excerpt."));
  if (!question) return (out.innerHTML = messageHtml("Missing question", "Ask a question about the situation, e.g. “Which team should handle this?”."));
  if (choices.length < 2) return (out.innerHTML = messageHtml("Need options", "Give at least two options, one per line."));
  if (choices.length > MAX_OPTIONS) return (out.innerHTML = messageHtml("Too many options", `This demo takes up to ${MAX_OPTIONS} options (cost grows with every option).`));
  if (kind === "verify" && choices.length !== 2) return (out.innerHTML = messageHtml("verify takes exactly two options", "Use two options such as “yes” and “no”, or switch to choose."));
  const detected = langSel === "auto";
  const lang = detected ? detectLang(state) : langSel;
  const token = ++tryToken;
  const go = $("#go");
  go.setAttribute("aria-busy", "true");
  if (!out.querySelector(".dz-result") || !loadStats) out.innerHTML = busyHtml(waitText());
  try {
    const r = await decide(state, { text: question, choices, kind, lang });
    if (token === tryToken) out.innerHTML = resultHtml(r, kind, lang, detected);
  } catch (e) {
    if (token === tryToken) out.innerHTML = messageHtml("Could not decide", e.message);
  } finally {
    if (token === tryToken) go.removeAttribute("aria-busy");
  }
}

function galleryHtml(cards) {
  return cards.map((g) => {
    const rec = g.recorded;
    return `
<button class="dz-ex" data-id="${esc(g.id)}" type="button">
  <span class="dz-ex-tag">${esc(g.tag)}</span>
  <span class="dz-ex-title">${esc(g.title)}</span>
  <span class="dz-ex-state" dir="auto">${esc(g.state)}</span>
  <span class="dz-ex-out"><span dir="auto">→ <b>${esc(rec.top)}</b> · ${pct(rec.confidence)}</span><span class="dz-ex-n">${g.choices.length} options · recorded answer</span></span>
</button>`;
  }).join("");
}

function loadCard(g) {
  $("#in-state").value = g.state;
  $("#in-question").value = g.question;
  $("#in-options").value = g.choices.join("\n");
  $(`input[name="kind"][value="${g.kind}"]`).checked = true;
  $("#in-lang").value = g.lang;
  $$(".dz-ex").forEach((c) => c.classList.toggle("is-active", c.dataset.id === g.id));
  const out = $("#try-form");
  if (out.getBoundingClientRect().top < 0) out.scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
  return decideUi();
}

// ───────────────────────────── tab 2 · business cases ─────────────────────────────

function caseLabel(c) {
  const dom = DOMAIN_NAME[c.domain] || c.domain.replace(/_/g, " ").replace(/\b\w/g, (m) => m.toUpperCase());
  const langs = LN[c.state_lang] + (c.choice_lang === c.state_lang ? "" : ` → ${LN[c.choice_lang]} options`);
  return [dom, langs];
}
function casesHtml(cases) {
  return cases.map((c, i) => {
    const [dom, langs] = caseLabel(c);
    return `
<button class="dz-case${i === 0 ? " is-active" : ""}" data-id="${i}" type="button">
  <span class="dz-ex-tag">${esc(dom)} · ${esc(langs)}</span>
  <span class="dz-case-title">${esc(c.scenario[0].toUpperCase() + c.scenario.slice(1))}</span>
  <span class="dz-case-agree"><b>${c.agree}/${c.n}</b> answers match the teacher</span>
</button>`;
  }).join("");
}
const lines = (text, cls = "") => text.split("\n").map((l) => `<div dir="auto"${cls ? ` class="${cls}"` : ""}>${esc(l) || "&nbsp;"}</div>`).join("");
function docHtml(state, cut) {
  if (!cut || cut >= state.length) return lines(state);
  return lines(state.slice(0, cut)) + '<div class="dz-cut" role="separator"><span>Decima’s 512-token window ends about here — the teacher read the rest too</span></div>' + lines(state.slice(cut), "dz-unread");
}
function questionHtml(q, k, dec) {
  const td = argmax(q.teacher), dd = dec ? argmax(dec) : -1;
  const idx = q.choices.map((_, j) => j);
  const order = q.kind === "score" ? idx : idx.sort((a, b) => q.teacher[b] - q.teacher[a]);
  const rows = order.map((j) => {
    const pt = q.teacher[j], pd = dec ? dec[j] : 0;
    return `
<div class="dz-crow${j === dd ? " is-top" : ""}${j === td ? " is-ttop" : ""}${dec ? "" : " is-pending"}">
  <div class="dz-label" dir="auto">${esc(q.choices[j])}</div>
  <div class="dz-pair">
    <div class="dz-track">${dec ? `<div class="dz-fill" style="width:${Math.max(pd * 100, 0.35).toFixed(2)}%"></div>` : ""}</div>
    <div class="dz-track dz-teacher"><div class="dz-fill" style="width:${Math.max(pt * 100, 0.35).toFixed(2)}%"></div></div>
  </div>
  <div class="dz-pct2"><span>${dec ? pct(pd) : "…"}</span><span class="dz-dim">${pct(pt)}</span></div>
</div>`;
  }).join("");
  const mark = !dec ? '<span class="dz-busy">Decima is reading…</span>'
    : dd === td ? '<span class="dz-badge dz-b-agree">✓ same answer</span>' : '<span class="dz-badge dz-b-disagree">✗ different answer</span>';
  const kindNote = q.kind === "score" ? " · levels in order" : "";
  return `
<div class="dz-q" data-q="${k}">
  <div class="dz-q-head"><span class="dz-q-n">Q${k + 1}</span><span class="dz-q-kind">${q.kind}${kindNote}</span>${mark}</div>
  <div class="dz-q-text" dir="auto">${esc(q.question)}</div>
  <div class="dz-cledger">${rows}</div>
</div>`;
}
function caseDetailHtml(c) {
  const [dom, langs] = caseLabel(c);
  return `
<div class="dz-case-detail">
  <div class="dz-case-meta"><span class="dz-eyebrow">${esc(dom)} · ${esc(langs)} · ${esc(c.format)}</span></div>
  <div class="dz-doc" aria-label="The situation">${docHtml(c.state, c.window_end_char)}</div>
  <div class="dz-legend"><span><i class="dz-sw dz-sw-d"></i>Decima-small · 122M · int8, live in your browser</span>
    <span><i class="dz-sw dz-sw-t"></i>Teacher · Gemma-4-26B-A4B (recorded)</span>
    <span class="dz-dim">rows sorted by the teacher; bold = Decima’s pick</span></div>
  ${c.questions.map((q, k) => questionHtml(q, k, null)).join("")}
</div>`;
}
let caseToken = 0;
const caseResults = new Map();   // "case:question" → live Decima probabilities
async function showCase(D, i) {
  const c = D.business.cases[i];
  const token = ++caseToken;
  $$(".dz-case").forEach((b) => b.classList.toggle("is-active", Number(b.dataset.id) === i));
  $("#case-view").innerHTML = caseDetailHtml(c);
  for (let k = 0; k < c.questions.length; k++) {
    const q = c.questions[k];
    const key = `${i}:${k}`;
    let probs = caseResults.get(key);
    if (!probs) {
      try {
        probs = (await decide(c.state, { text: q.question, choices: q.choices, kind: q.kind, lang: c.state_lang })).probs;
        caseResults.set(key, probs);
      } catch (e) { return; }
    }
    if (token !== caseToken) return;
    $(`.dz-q[data-q="${k}"]`).outerHTML = questionHtml(q, k, probs);
  }
  // The card counts were recorded with native ONNX Runtime; show the count from this browser's run instead.
  // (WebAssembly computes the int8 layers with float activations, so a near-tie can land the other way.)
  const agree = c.questions.filter((q, k) => argmax(caseResults.get(`${i}:${k}`)) === argmax(q.teacher)).length;
  const b = $(`.dz-case[data-id="${i}"] .dz-case-agree b`);
  if (b) b.textContent = `${agree}/${c.questions.length}`;
}
function businessSummaryHtml(B) {
  const s = B.shown, p = B.pool;
  return `
<div class="dz-note">
  <p><b>${s.cases} long cases, every question shown: Decima gives the teacher’s answer on ${s.agree} of ${s.questions}.</b>
  We picked these cases <i>because</i> Decima mostly agrees on them. Across <b>all ${p.cases.toLocaleString("en")}</b> held-out cases it agrees on
  only <b>${p.agree.toLocaleString("en")} of ${p.questions.toLocaleString("en")}</b> questions (${Math.round(p.agree / p.questions * 100)}%) — most of those documents
  are longer than the 512 tokens Decima reads. Long, multi-fact decisions are this model’s weakest area; the teacher,
  a 26B model, is not always right either.</p>
  <p class="dz-dim">The model never trained on these cases: they come from rows of the synthetic set that were generated after
  the training snapshot. ${esc(B.attribution)}</p>
</div>`;
}

// ───────────────────────────── tab 3 · shuffle ─────────────────────────────

const shuf = { preset: null, order: [], tally: { n: 0, flips: 0, max: 0 }, token: 0 };
function orderedLedger(title, choices, probs, topC, permFrom) {
  const rows = choices.map((c, i) => {
    const p = probs[c];
    let moved = "";
    if (permFrom) {
      const j = permFrom.indexOf(c) + 1;
      moved = j !== i + 1 ? `<span class="dz-pos">was #${j}</span>` : '<span class="dz-pos">same place</span>';
    }
    return `
<div class="dz-row${c === topC ? " is-top" : ""}"><div class="dz-label" dir="auto"><span class="dz-idx">${i + 1}</span>${esc(c)} ${moved}</div>
<div class="dz-track"><div class="dz-fill" style="width:${Math.max(p * 100, 0.35).toFixed(2)}%"></div></div><div class="dz-pct">${pct(p)}</div></div>`;
  }).join("");
  return `<div class="dz-card dz-shuf"><div class="dz-eyebrow">${esc(title)}</div><div class="dz-ledger">${rows}</div></div>`;
}
function fmtDelta(x) {
  if (x === 0) return "0 (identical)";
  const [m, e] = x.toExponential(1).split("e");
  return `${m} × 10<sup>${Number(e)}</sup>`;
}
const asDict = (r) => Object.fromEntries(r.choices.map((c, i) => [c, r.probs[i]]));
const same = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);

async function shuffleRender(p, state, order, bump) {
  const token = ++shuf.token;
  const out = $("#shuf-out");
  if (!out.firstElementChild || !loadStats) out.innerHTML = busyHtml(waitText());
  const q = { text: p.question, kind: p.kind, lang: p.lang };
  let r0, r1;
  try {
    r0 = await decide(state, { ...q, choices: p.choices });
    r1 = await decide(state, { ...q, choices: order });
  } catch (e) { out.innerHTML = messageHtml("Could not decide", e.message); return; }
  if (token !== shuf.token) return;
  const a = asDict(r0), b = asDict(r1);
  const delta = Math.max(...p.choices.map((c) => Math.abs(a[c] - b[c])));
  const t0 = top(r0), t1 = top(r1), changed = t0 !== t1;
  if (bump) shuf.tally = { n: shuf.tally.n + 1, flips: shuf.tally.flips + (changed ? 1 : 0), max: Math.max(shuf.tally.max, delta) };
  const t = shuf.tally;
  out.innerHTML = `
<div class="dz-shuf-grid">${orderedLedger("Options as listed", p.choices, a, t0)}${orderedLedger("Same options, shuffled", order, b, t1, p.choices)}</div>
<div class="dz-verdict">
  <div><div class="dz-eyebrow">Answer</div><div class="dz-big" dir="auto">${changed ? "changed" : "unchanged"} — ${esc(t1)}</div></div>
  <div><div class="dz-eyebrow">Largest probability change</div><div class="dz-big dz-mono">${fmtDelta(delta)}</div></div>
  <div><div class="dz-eyebrow">This session</div><div class="dz-big">${t.n} shuffle${t.n !== 1 ? "s" : ""} · ${t.flips} answer change${t.flips !== 1 ? "s" : ""}</div>
  <div class="dz-dim">largest change seen: ${fmtDelta(t.max)}</div></div>
</div>`;
}
function shuffleReset(D) {
  const p = D.gallery.shuffle[Number($("#shuf-preset").value)];
  shuf.preset = p; shuf.order = [...p.choices]; shuf.tally = { n: 0, flips: 0, max: 0 };
  $("#shuf-state").value = p.state;
  return shuffleRender(p, p.state, shuf.order, false);
}
function shuffleClick() {
  const p = shuf.preset;
  const state = $("#shuf-state").value.trim() || p.state;
  let next = [...p.choices];
  const shuffleArr = (a) => { for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } };
  if (next.length > 1) while (same(next, p.choices) || same(next, shuf.order)) shuffleArr(next);
  shuf.order = next;
  return shuffleRender(p, state, next, true);
}

// ───────────────────────────── charts (inline SVG) ─────────────────────────────

const tip = document.createElement("div");
tip.className = "dz-tip"; tip.hidden = true; tip.setAttribute("role", "tooltip");
document.body.appendChild(tip);
function bindTips(svg) {
  svg.addEventListener("pointermove", (e) => {
    const t = e.target.closest("[data-tip]");
    if (!t) { tip.hidden = true; return; }
    tip.innerHTML = t.dataset.tip;
    tip.hidden = false;
    const w = tip.offsetWidth, h = tip.offsetHeight;
    tip.style.left = `${Math.min(innerWidth - w - 8, e.clientX + 12)}px`;
    tip.style.top = `${Math.max(8, e.clientY - h - 12)}px`;
  });
  svg.addEventListener("pointerleave", () => (tip.hidden = true));
}

function routeChart(rs, th, width) {
  const W = Math.max(300, width), H = 250, L = 14, R = 14, T = 14, B = 44;
  const x = (v) => L + ((Math.max(0.3, Math.min(1, v)) - 0.3) / 0.7) * (W - L - R);
  const y = (v) => T + ((0.62 - v) / 1.24) * (H - T - B);
  const ticks = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
  let g = ticks.map((t) => `<line class="grid" x1="${x(t)}" x2="${x(t)}" y1="${T}" y2="${H - B}"/><text class="axis-t" x="${x(t)}" y="${H - B + 16}" text-anchor="middle">${t.toFixed(1)}</text>`).join("");
  g += `<rect class="band" x="${x(0.3)}" y="${T}" width="${x(th) - x(0.3)}" height="${H - B - T}"/>`;
  g += `<text class="zone" x="${(x(0.3) + x(th)) / 2}" y="${y(-0.52)}" text-anchor="middle">escalate to a person</text>`;
  g += `<text class="zone" x="${(x(th) + x(1)) / 2}" y="${y(-0.52)}" text-anchor="middle">route automatically</text>`;
  g += `<line class="th" x1="${x(th)}" x2="${x(th)}" y1="${T}" y2="${H - B}"/>`;
  const anchorEnd = th > 0.85;
  g += `<text class="th-t" x="${x(th) + (anchorEnd ? -6 : 6)}" y="${y(0.52)}" text-anchor="${anchorEnd ? "end" : "start"}">threshold ${th.toFixed(2)}</text>`;
  // spread marks over five rows by confidence rank, so neighbours on the x axis never share a row
  const rank = new Map([...rs].sort((a, b) => a.conf - b.conf).map((r, i) => [r, i]));
  rs.forEach((r) => {
    const cx = x(r.conf), cy = y(((rank.get(r) % 5) - 2) * 0.16);
    const tipText = `${esc(r.text.slice(0, 70))}${r.text.length > 70 ? "…" : ""}<br>→ <b>${esc(r.top)}</b> (${r.conf.toFixed(3)})<br>expected: ${esc(r.gold)}`;
    g += r.ok ? `<circle class="ok" cx="${cx}" cy="${cy}" r="5.5"/>`
              : `<path class="bad" d="M${cx - 5} ${cy - 5}L${cx + 5} ${cy + 5}M${cx - 5} ${cy + 5}L${cx + 5} ${cy - 5}"/>`;
    g += `<circle class="hit" cx="${cx}" cy="${cy}" r="11" data-tip="${esc(tipText)}"/>`;
  });
  g += `<text class="axis-l" x="${(L + W - R) / 2}" y="${H - 6}" text-anchor="middle">model confidence (top probability)</text>`;
  return `<svg class="dz-chart" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="Twenty tickets placed by model confidence; the threshold line splits automatic routing from escalation. The table below lists the same data.">${g}</svg>`;
}

function speedChart(live, width) {
  const W = Math.max(300, width), H = 300, L = 48, R = 96, T = 14, B = 46;
  const xs = [4, 20, 77, 150];
  const ymax = Math.max(140, ...Object.values(live || {}).map((v) => v * 1.12));
  const step = ymax > 600 ? 200 : ymax > 300 ? 100 : 50;
  const x = (v) => L + (v / 150) * (W - L - R);
  const y = (v) => T + (1 - v / ymax) * (H - T - B);
  let g = "";
  for (let t = 0; t <= ymax; t += step) g += `<line class="grid" x1="${L}" x2="${W - R}" y1="${y(t)}" y2="${y(t)}"/><text class="axis-t" x="${L - 8}" y="${y(t) + 4}" text-anchor="end">${t}</text>`;
  g += xs.map((v) => `<text class="axis-t" x="${x(v)}" y="${H - B + 17}" text-anchor="middle">${v}</text>`).join("");
  g += `<text class="axis-l" x="${(L + W - R) / 2}" y="${H - 6}" text-anchor="middle">number of options</text>`;
  g += `<text class="axis-l" transform="translate(13 ${(T + H - B) / 2}) rotate(-90)" text-anchor="middle">ms per decision</text>`;
  const series = (vals, cls, name) => {
    const pts = xs.filter((v) => vals[v] != null).map((v) => [x(v), y(vals[v]), v, vals[v]]);
    let s = `<path class="${cls}" d="${pts.map((p, i) => `${i ? "L" : "M"}${p[0]} ${p[1]}`).join("")}"/>`;
    for (const [px, py, v, ms] of pts) {
      s += `<circle class="${cls}-dot" cx="${px}" cy="${py}" r="4.5"/>`;
      s += `<circle class="hit" cx="${px}" cy="${py}" r="12" data-tip="${esc(`<b>${name}</b><br>${v} options: ${ms.toFixed(1)} ms`)}"/>`;
    }
    const last = pts.at(-1);
    if (last) s += `<text class="lbl" x="${last[0] + 10}" y="${last[1] + 4}">${Math.round(last[3])} ms</text>`;
    return s;
  };
  g += series(X86_SCALING, "ref", "x86 laptop core, reference");
  if (live) g += series(live, "live", "your browser, live (p50)");
  return `<svg class="dz-chart" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="Milliseconds per decision against the number of options: x86 reference${live ? " and your browser" : ""}. The table lists the same numbers.">${g}</svg>`;
}

// ───────────────────────────── tab 4 · auto-route ─────────────────────────────

let routeResults = null;
async function scoreTickets(D) {
  if (routeResults) return routeResults;
  const out = $("#route-out");
  out.innerHTML = busyHtml(loadStats ? `Scoring ${D.tickets.length} tickets in your browser…` : waitText());
  await ready;
  out.innerHTML = busyHtml(`Scoring ${D.tickets.length} tickets in your browser…`);
  const items = D.tickets.map((t) => ({ state: t.text, q: { text: TEAM_Q, choices: TEAMS, kind: "choose", lang: t.lang } }));
  const t0 = performance.now();
  const rs = await call("decideMany", { items });
  const ms = performance.now() - t0;
  routeResults = D.tickets.map((t, i) => ({ ...t, top: top(rs[i]), conf: confidence(rs[i]), ok: top(rs[i]) === t.gold, probs: rs[i].probs }));
  routeResults.ms = ms;
  return routeResults;
}
function routeUpdate() {
  const rs = routeResults;
  if (!rs) return;
  const th = Number($("#th").value);
  $("#th-val").textContent = th.toFixed(2);
  const auto = rs.filter((r) => r.conf >= th);
  const right = auto.filter((r) => r.ok).length;
  const escN = rs.length - auto.length;
  const caught = rs.filter((r) => !r.ok && r.conf < th).length;
  const wrong = rs.filter((r) => !r.ok).length;
  const tiles = `
<div class="dz-tiles">
  <div class="dz-tile"><div class="dz-tile-n">${auto.length}<span>/${rs.length}</span></div><div class="dz-tile-l">routed automatically</div></div>
  <div class="dz-tile"><div class="dz-tile-n">${right}<span>/${auto.length}</span></div><div class="dz-tile-l">of those went to the right team</div></div>
  <div class="dz-tile"><div class="dz-tile-n">${escN}</div><div class="dz-tile-l">escalated to a person — ${caught} of the ${wrong} wrong answer${wrong !== 1 ? "s" : ""} among them</div></div>
</div>`;
  const rows = [...rs].sort((a, b) => b.conf - a.conf).map((r) => {
    const a = r.conf >= th;
    return `<tr class="${a ? "is-auto" : "is-esc"}"><td><span class="dz-badge ${a ? "dz-b-high" : "dz-b-mid"}">${a ? "auto" : "escalate"}</span></td>
<td class="dz-mono">${r.conf.toFixed(3)}</td><td dir="auto">${esc(r.text)}</td><td>${esc(r.lang)}</td><td>${esc(r.top)}</td>
<td>${r.ok ? '<span class="dz-ok">✓</span>' : `<span class="dz-bad">✗</span> <span class="dz-dim">expected ${esc(r.gold)}</span>`}</td></tr>`;
  }).join("");
  const out = $("#route-out");
  const width = out.clientWidth - 26;
  out.innerHTML = `${tiles}
<div class="dz-plot"><div class="dz-legend-row"><span><i class="dz-k-ok"></i>right team</span><span><i class="dz-k-bad"></i>wrong team</span>
<span class="dz-dim">scored in your browser in ${(rs.ms / 1000).toFixed(1)} s · hover a mark for the ticket</span></div>${routeChart(rs, th, width)}</div>
<div class="dz-table-wrap"><table class="dz-table"><thead><tr><th>decision</th><th>confidence</th><th>ticket</th><th>lang</th>
<th>model’s team</th><th>right?</th></tr></thead><tbody>${rows}</tbody></table></div>`;
  bindTips($("#route-out svg"));
}

// ───────────────────────────── tab 5 · speed ─────────────────────────────

let speedLive = null;
function drawSpeed() {
  const el = $("#sp-plot");
  const legend = `<div class="dz-legend-row"><span><i class="dz-k-ref"></i>x86 laptop core, reference (BENCH-x86)</span>${speedLive ? '<span><i class="dz-k-live"></i>your browser, live (p50)</span>' : ""}</div>`;
  el.innerHTML = legend + speedChart(speedLive, el.clientWidth - 26);
  bindTips($("svg", el));
}
async function speedRun(D) {
  const btn = $("#sp-go");
  if (btn.getAttribute("aria-busy")) return;
  btn.setAttribute("aria-busy", "true");
  const reps = 12;
  const rows = [];
  const live = {};
  const table = $("#sp-table");
  try {
    table.innerHTML = busyHtml(waitText());
    await ready;
    for (const n of [4, 20, 77, 150]) {
      table.innerHTML = busyHtml(`Timing ${n} options… (${reps + 1} decisions)`);
      const q = { text: "What is the customer asking about?", choices: D.speedSets[n], kind: "choose", lang: "en" };
      const { first, lat } = await call("bench", { state: "I was charged twice for my subscription this month.", q, reps });
      lat.sort((a, b) => a - b);
      const p50 = lat.length % 2 ? lat[(lat.length - 1) / 2] : (lat[lat.length / 2 - 1] + lat[lat.length / 2]) / 2;
      const p95 = lat[Math.min(lat.length - 1, Math.round(0.95 * (lat.length - 1)))];
      live[n] = p50;
      speedLive = { ...live };
      drawSpeed();
      rows.push(`<tr><td class='dz-mono'>${n}</td><td class='dz-mono'><b>${p50.toFixed(1)}</b></td><td class='dz-mono'>${p95.toFixed(1)}</td>` +
        `<td class='dz-mono'>${X86_SCALING[n]}</td><td class='dz-mono'>${Math.round(first)}</td></tr>`);
    }
    const ua = navigator.userAgentData?.brands?.filter((b) => !/Not.A.Brand/i.test(b.brand)).map((b) => b.brand).at(-1) || "";
    table.innerHTML = `
<div class="dz-table-wrap"><table class="dz-table dz-num"><thead><tr><th>options</th><th>p50 ms · your browser</th><th>p95 ms · your browser</th>
<th>p50 ms · x86 reference</th><th>first call ms · your browser</th></tr></thead><tbody>${rows.join("")}</tbody></table></div>
<p class="dz-dim dz-small">Live: this tab${ua ? ` (${esc(ua)})` : ""}, ${navigator.hardwareConcurrency || "?"} logical cores reported, ${loadStats.threads} WebAssembly thread${loadStats.threads > 1 ? "s" : ""},
int8 ONNX on ONNX Runtime Web, batch 1, ${reps} timed calls per size after the first, short state. The first call also encodes the option
set (once per set). Background tabs and battery saving slow it down — run it twice. Reference: Intel Core Ultra 7 155H, one P-core,
native ONNX Runtime, 40 calls per size (docs/BENCH-x86.md).</p>`;
  } catch (e) {
    table.innerHTML = messageHtml("The sweep stopped", e.message);
  } finally {
    btn.removeAttribute("aria-busy");
  }
}

// ───────────────────────────── page wiring ─────────────────────────────

const TABS = ["try", "cases", "shuffle", "route", "speed"];
const opened = new Set();
function selectTab(name, D, focus = false) {
  if (!TABS.includes(name)) name = "try";
  for (const t of TABS) {
    const on = t === name;
    const b = $(`#tab-${t}`);
    b.setAttribute("aria-selected", on);
    b.tabIndex = on ? 0 : -1;
    $(`#p-${t}`).hidden = !on;
    if (on && focus) b.focus();
  }
  if (location.hash.slice(1) !== name) history.replaceState(null, "", name === "try" ? location.pathname + location.search : `#${name}`);
  if (opened.has(name)) { if (name === "route") routeUpdate(); if (name === "speed") drawSpeed(); return; }
  opened.add(name);
  if (name === "cases") showCase(D, 0);
  if (name === "shuffle") shuffleReset(D);
  if (name === "route") scoreTickets(D).then(routeUpdate).catch((e) => ($("#route-out").innerHTML = messageHtml("Could not score the tickets", e.message)));
  if (name === "speed") drawSpeed();
}

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
  $("#out").innerHTML = busyHtml("Waiting for the model to finish loading…");
  const D = await dataReady;

  // try it
  $("#gallery").innerHTML = galleryHtml(D.cards);
  $("#gallery").addEventListener("click", (e) => {
    const card = e.target.closest("[data-id]");
    if (card) loadCard(D.byId[card.dataset.id]);
  });
  $("#try-form").addEventListener("submit", (e) => { e.preventDefault(); decideUi(); });
  $("#in-state").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) { e.preventDefault(); decideUi(); }
  });

  // business cases
  $("#cases-summary").innerHTML = businessSummaryHtml(D.business);
  $("#case-list").innerHTML = casesHtml(D.business.cases);
  $("#case-list").addEventListener("click", (e) => {
    const b = e.target.closest("[data-id]");
    if (b) showCase(D, Number(b.dataset.id));
  });

  // shuffle
  $("#shuf-preset").innerHTML = D.gallery.shuffle.map((p, i) => `<option value="${i}">${esc(p.title)}</option>`).join("");
  $("#shuf-preset").addEventListener("change", () => shuffleReset(D));
  $("#shuf-go").addEventListener("click", () => shuffleClick());
  $("#shuf-state").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      shuffleRender(shuf.preset, $("#shuf-state").value.trim() || shuf.preset.state, shuf.order, false);
    }
  });

  // auto-route + speed
  $("#th").addEventListener("input", routeUpdate);
  $("#sp-go").addEventListener("click", () => speedRun(D));
  let rt;
  addEventListener("resize", () => {
    clearTimeout(rt);
    rt = setTimeout(() => { if (!$("#p-route").hidden) routeUpdate(); if (!$("#p-speed").hidden) drawSpeed(); }, 150);
  });

  // tabs
  $$(".dz-tab").forEach((b, i) => {
    b.addEventListener("click", () => selectTab(TABS[i], D));
    b.addEventListener("keydown", (e) => {
      const d = { ArrowRight: 1, ArrowLeft: -1 }[e.key];
      if (d) { e.preventDefault(); selectTab(TABS[(i + d + TABS.length) % TABS.length], D, true); }
    });
  });
  addEventListener("hashchange", () => selectTab(location.hash.slice(1), D));
  selectTab(location.hash.slice(1), D);

  $("#forget").addEventListener("click", async () => {
    const ok = await call("clearCache").catch(() => false);
    $("#forget").textContent = ok ? "Removed — the next visit downloads it again." : "Nothing cached on this device.";
  });

  // first decision, like the Gradio version's on-load run
  ready.then(() => decideUi(), () => ($("#out").innerHTML = messageHtml("The model could not be loaded", "See the message at the top of the page.")));
}

// hook for automated checks (parity tests, screenshots)
window.DZ = { ready, decide: (state, q) => decide(state, q), data: dataReady, stats: () => loadStats, call };

main();
