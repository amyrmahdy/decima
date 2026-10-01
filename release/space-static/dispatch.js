// Decima · Live city dispatch. Invented city reports stream in; for each one the model answers three
// Jev-style questions (which service? how severe? is someone in danger?) in this browser tab.
// ?video=1 plays one pass with a title card and an end card (used to record the demo video).

import { detectLang } from "./decima.js";

const MODEL_URL = "https://huggingface.co/amyrmahdy/decima-small/resolve/main/onnx/int8/";
const params = new URLSearchParams(location.search);
const VIDEO = params.has("video");
const $ = (s) => document.querySelector(s);

// ───────────────────────── the model, in a worker (same protocol as the Playground) ─────────────────────────

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
const call = (op, args = {}, onProgress) => new Promise((resolve, reject) => {
  const id = ++seq;
  pending.set(id, { resolve, reject, onProgress });
  worker.postMessage({ id, op, ...args });
});

// ───────────────────────── services, severity, districts ─────────────────────────

const SERVICE = {
  fire: { icon: "🔥", name: "Fire" },
  police: { icon: "🚓", name: "Police" },
  medical: { icon: "🚑", name: "Ambulance" },
  traffic: { icon: "🚦", name: "Traffic" },
  utilities: { icon: "⚡", name: "Utilities" },
  city_services: { icon: "🏙️", name: "City services" },
};
const SEV_NAME = ["low", "moderate", "high", "critical"];
const SEV_VAR = ["--sev0", "--sev1", "--sev2", "--sev3"];
const DISTRICTS = {
  "Northside": [520, 95], "University": [185, 150], "Old Town": [420, 250], "Market": [640, 255],
  "Eastgate": [870, 190], "Central Station": [290, 385], "Riverside": [560, 420], "Parkview": [150, 520], "Harbor": [800, 520],
};
const REVIEW_SERVICE = 0.6;              // below this, the routing goes to a person
const REVIEW_DANGER = [0.35, 0.65];      // a danger probability in this band goes to a person too

// ───────────────────────── map ─────────────────────────

const NS = "http://www.w3.org/2000/svg";
const svg = $("#map");
const el = (tag, attrs = {}, parent = svg) => {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  parent.appendChild(e);
  return e;
};

function rng(seed) {                       // small deterministic PRNG so the city looks the same every time
  let s = seed >>> 0;
  return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 2 ** 32);
}

function drawMap() {
  el("rect", { x: 0, y: 0, width: 1000, height: 640, class: "m-land" });
  const r = rng(7);
  for (let x = 20; x < 1000; x += 62) {
    for (let y = 18; y < 640; y += 54) {
      if (r() < 0.12) continue;
      el("rect", { x: x + r() * 4, y: y + r() * 4, width: 50 - r() * 12, height: 42 - r() * 10, rx: 3, class: "m-block" });
    }
  }
  el("path", { class: "m-park", d: "M40 450 C 90 420 230 430 260 470 C 290 520 250 600 170 610 C 90 620 30 560 40 450 Z" });
  el("path", { class: "m-park", d: "M860 60 C 930 50 980 90 975 140 C 940 150 880 140 860 60 Z" });
  el("path", { class: "m-water", d: "M650 640 C 700 560 820 560 1000 575 L 1000 640 Z" });
  el("path", { class: "m-river", d: "M-20 300 C 150 320 250 250 360 320 C 470 390 560 330 660 360 C 760 390 760 520 700 640" });
  for (const d of ["M0 205 L1000 175", "M470 0 L520 640", "M0 470 C 300 450 600 480 1000 430"]) el("path", { class: "m-road major", d });
  for (const d of ["M230 0 L260 640", "M760 0 L720 640", "M0 95 L1000 60", "M0 360 L1000 330", "M880 0 L940 640"]) el("path", { class: "m-road", d });
  for (const [name, [x, y]] of Object.entries(DISTRICTS)) {
    const t = el("text", { x, y: y - 22, class: "m-label", "text-anchor": "middle" });
    t.textContent = name;
  }
}

const pinLayer = () => svg.querySelector("#pins") || el("g", { id: "pins" });

function addPin(item) {
  const [cx, cy] = DISTRICTS[item.district] || [500, 320];
  const jr = rng(item.n * 97 + 13);
  const x = cx + (jr() - 0.5) * 130, y = cy + 4 + jr() * 58;
  const color = getComputedStyle(document.documentElement).getPropertyValue(SEV_VAR[item.sev]).trim();
  const g = el("g", { class: "pin new", "data-n": item.n }, pinLayer());
  if (item.danger >= 0.5) el("circle", { cx: x, cy: y, r: 15, class: "pulse" }, g);
  el("circle", { cx: x, cy: y, r: 15, fill: color, class: "ring", stroke: item.danger >= 0.5 ? "var(--sev3)" : "rgba(0,0,0,0.35)" }, g);
  const t = el("text", { x, y: y + 1 }, g);
  t.textContent = SERVICE[item.service]?.icon || "•";
  g.addEventListener("mouseenter", () => highlight(item.n, true));
  g.addEventListener("mouseleave", () => highlight(item.n, false));
  requestAnimationFrame(() => requestAnimationFrame(() => g.classList.remove("new")));
  const pins = [...pinLayer().children];
  pins.slice(0, Math.max(0, pins.length - 12)).forEach((p) => p.classList.add("old"));
  if (pins.length > 40) pins[0].remove();
}

function highlight(n, on) {
  document.querySelectorAll(`[data-n="${n}"]`).forEach((e) => e.classList.toggle("hl", on));
}

// ───────────────────────── feed ─────────────────────────

const feed = $("#feed");
const stats = { reports: 0, decisions: 0, review: 0, ms: [] };

function renderStats() {
  $("#st-reports").textContent = stats.reports;
  $("#st-decisions").textContent = stats.decisions;
  $("#st-review").textContent = stats.review;
  if (stats.ms.length) {
    const s = [...stats.ms].sort((a, b) => a - b);
    $("#st-ms").textContent = Math.round(s[Math.floor(s.length / 2)]);
  }
}

function card(item) {
  const li = document.createElement("li");
  li.className = "card";
  li.dataset.n = item.n;
  const now = new Date();
  li.innerHTML = `<div class="card-top"><span>${now.toTimeString().slice(0, 8)}</span><span class="card-lang">${item.lang.toUpperCase()}</span><span>${item.district}</span></div>
    <div class="card-text" dir="auto"></div><div class="card-out"><span class="thinking">deciding…</span></div>`;
  li.querySelector(".card-text").textContent = item.text;
  li.addEventListener("mouseenter", () => highlight(item.n, true));
  li.addEventListener("mouseleave", () => highlight(item.n, false));
  feed.prepend(li);
  while (feed.children.length > 30) feed.lastChild.remove();
  return li;
}

function chip(html, cls = "") {
  const s = document.createElement("span");
  s.className = `chip ${cls}`;
  s.innerHTML = html;
  return s;
}

const fmt = (p) => p.toFixed(2);

// ───────────────────────── deciding ─────────────────────────

let Q = null;          // the three questions in model form
let serviceKeys = [];

function buildQuestions(questions) {
  const svc = questions.service;
  serviceKeys = Object.keys(svc.criteria);
  Q = {
    service: { text: svc.text, choices: serviceKeys.map((k) => `${k}: ${svc.criteria[k]}`), kind: "choose" },
    severity: { text: questions.severity.text, choices: questions.severity.levels, kind: "score" },
    danger: { text: questions.danger.text, choices: ["yes", "no"], kind: "verify" },
  };
}

const argmax = (a) => a.reduce((b, v, i) => (v > a[b] ? i : b), 0);

async function decideReport(item) {
  const lang = ["fa", "ar"].includes(item.lang) ? item.lang : "en";
  const li = card(item);
  const out = [];
  for (const k of ["service", "severity", "danger"]) out.push(await call("decide", { state: item.text, q: { ...Q[k], lang } }));
  const [s, v, d] = out;
  const si = argmax(s.probs), vi = argmax(v.probs);
  Object.assign(item, { service: serviceKeys[si], pService: s.probs[si], sev: vi, pSev: v.probs[vi], danger: d.probs[0] });
  const ms = out.reduce((a, r) => a + r.latency_ms, 0) / out.length;
  stats.reports += 1; stats.decisions += 3; stats.ms.push(ms);
  const review = item.pService < REVIEW_SERVICE || (item.danger > REVIEW_DANGER[0] && item.danger < REVIEW_DANGER[1]);
  if (review) stats.review += 1;

  const box = li.querySelector(".card-out");
  box.innerHTML = "";
  const svc = SERVICE[item.service];
  box.append(chip(`${svc.icon} <b>${svc.name}</b><span class="p">${fmt(item.pService)}</span>`));
  box.append(chip(`<span class="dot" style="background:var(${SEV_VAR[vi]})"></span>${SEV_NAME[vi]}<span class="p">${fmt(item.pSev)}</span>`));
  box.append(item.danger >= 0.5
    ? chip(`person in danger<span class="p">${fmt(item.danger)}</span>`, "danger-yes")
    : chip(`no one in danger<span class="p">${fmt(1 - item.danger)}</span>`));
  if (review) box.append(chip("⚠ check by a person", "review"));
  box.append(chip(`${ms.toFixed(0)} ms each`, "ms"));
  li.classList.add(`sev-${vi}`);
  addPin(item);
  renderStats();
}

// ───────────────────────── the stream ─────────────────────────

let reports = [];
let playing = true;
let speed = 1;
let counter = 0;
const ownQueue = [];
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function waitWhilePaused() { while (!playing) await sleep(150); }

async function stream() {
  let i = 0;
  for (;;) {
    await waitWhilePaused();
    const next = ownQueue.length ? ownQueue.shift() : reports[i++];
    if (!next) {
      if (VIDEO) break;
      i = 0;
      await sleep(3000);
      continue;
    }
    const t0 = performance.now();
    await decideReport({ ...next, n: ++counter });
    const gap = (VIDEO ? 2300 : 2600) / speed - (performance.now() - t0);
    if (gap > 0) await sleep(gap);
  }
  if (VIDEO) {
    await sleep(2500);
    $("#outro").hidden = false;
    document.title = "DONE";
  }
}

$("#play").addEventListener("click", () => {
  playing = !playing;
  $("#play").textContent = playing ? "Pause" : "Play";
});
$("#speed").addEventListener("change", (e) => { speed = Number(e.target.value); });
$("#own").addEventListener("submit", (e) => {
  e.preventDefault();
  const text = $("#own-text").value.trim();
  if (!text) return;
  const names = Object.keys(DISTRICTS);
  const lang = detectLang(text);
  ownQueue.push({ text, lang: ["latn", "auto", "xx"].includes(lang) ? "en" : lang, district: names[Math.floor(Math.random() * names.length)] });
  $("#own-text").value = "";
  if (!playing) { playing = true; $("#play").textContent = "Pause"; }
});

// ───────────────────────── start ─────────────────────────

async function main() {
  drawMap();
  const data = await (await fetch("examples/dispatch.json")).json();
  reports = data.reports;
  buildQuestions(data.questions);
  const res = await call("load", { url: params.get("model") || MODEL_URL }, (p) => {
    const pct = p.total ? Math.min(100, (100 * p.loaded) / p.total) : 0;
    $("#load-bar").style.width = `${pct}%`;
    $("#load-num").textContent = p.cached ? "from cache" : `${(p.loaded / 1e6).toFixed(0)} MB`;
  });
  $("#engine-where").textContent = `this tab, ${res.threads} thread${res.threads > 1 ? "s" : ""}`;
  // encode each question's options once (a one-time cache fill) so the stream shows warm latencies
  for (const k of Object.keys(Q)) await call("decide", { state: "warm-up", q: { ...Q[k], lang: "en" } });
  $("#loader").classList.add("done");
  if (VIDEO) {
    window.__tIntro = performance.now();     // lets the recorder trim the loading screen
    $("#intro").hidden = false;
    await sleep(4200);
    $("#intro").hidden = true;
  }
  stream();
}

main().catch((e) => {
  $("#load-num").textContent = "";
  $("#loader").querySelector("p").textContent = `Could not load the model: ${e.message}`;
});
