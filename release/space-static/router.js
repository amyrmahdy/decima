// decima.Router for the browser — a port of decima/router.py and decima/systemone.py (decima-ai 2.2.0):
// TypeSafe-style questions (or a preset's) in, `/v1/systemone`-shaped answers out. The model calls go to the
// worker; everything here is plain data shaping, kept line for line with the Python package so that the same
// preset on the same text asks the model exactly the same questions.

// ───────────────────────── json.dumps(…, ensure_ascii=False) ─────────────────────────

/** Python's json.dumps with default separators (", " and ": ") and ensure_ascii=False. Floats that are whole
 *  numbers print as "1" here and "1.0" in Python: JSON.parse cannot tell them apart. */
export function pyDumps(v) {
  if (v === null || v === undefined) return "null";
  if (typeof v === "boolean") return v ? "true" : "false";
  if (typeof v === "number") return Number.isFinite(v) ? String(v) : v > 0 ? "Infinity" : v < 0 ? "-Infinity" : "NaN";
  if (typeof v === "string") return JSON.stringify(v);
  if (Array.isArray(v)) return `[${v.map(pyDumps).join(", ")}]`;
  return `{${Object.entries(v).map(([k, x]) => `${JSON.stringify(k)}: ${pyDumps(x)}`).join(", ")}}`;
}

const text = (v) => (v === null || v === undefined ? "" : typeof v === "string" ? v : pyDumps(v));   // systemone._text
const pyStrip = (s) => s.replace(/^\s+|\s+$/g, "");

// ───────────────────────── router.py ─────────────────────────

/** Put `text` wherever the template has the string "{text}". */
export function fill(template, t) {
  if (template === "{text}") return t;
  if (Array.isArray(template)) return template.map((v) => fill(v, t));
  if (template && typeof template === "object") return Object.fromEntries(Object.entries(template).map(([k, v]) => [k, fill(v, t)]));
  return template;
}

/** Drop questions whose `requires` keys the state lacks; strip `requires` from the rest. */
export function applicable(questions, state) {
  const keys = state && typeof state === "object" && !Array.isArray(state) ? new Set(Object.keys(state)) : null;
  const out = {};
  for (const [qid, spec] of Object.entries(questions)) {
    const req = spec && typeof spec === "object" ? spec.requires : null;
    if (req && req.length && keys && !req.every((k) => keys.has(k))) continue;
    if (req && req.length) { const { requires, ...rest } = spec; out[qid] = rest; } else out[qid] = spec;
  }
  return out;
}

/** Text from a user → state: a JSON object or array stays structured; plain text goes through the preset's
 *  `state` template. */
export function asState(t, preset) {
  const s = t.replace(/^\s+/, "");
  if (s.startsWith("{") || s.startsWith("[")) {
    try { return JSON.parse(t); } catch { /* plain text */ }
  }
  return preset ? fill(preset.state ?? "{text}", t) : t;
}

// ───────────────────────── systemone.py ─────────────────────────

/** TypeSafe question → {type, q: Decima question, keys}. Throws on a question that does not fit the schema. */
export function toQuestion(qid, spec, lang = "en") {
  if (!spec || typeof spec !== "object" || Array.isArray(spec)) throw new Error(`questions.${qid}: must be an object`);
  const t = spec.type;
  let qt = spec.instructions === undefined || spec.instructions === null ? qid : text(spec.instructions);
  let crit = spec.criteria;
  if (t === "choice") {
    if (Array.isArray(crit)) crit = Object.fromEntries([...new Set(crit.map(String))].map((c) => [c, null]));
    if (!crit || typeof crit !== "object" || Object.keys(crit).length < 2 || Object.keys(crit).length > 255)
      throw new Error(`questions.${qid}: a choice needs 2–255 options`);
    const labels = Object.keys(crit);
    const opts = labels.map((k) => (pyStrip(text(crit[k])) ? `${k}: ${text(crit[k])}` : k));
    return { type: t, q: { text: qt, choices: opts, kind: "choose", lang }, keys: labels };
  }
  if (t === "score") {
    if (!Array.isArray(crit) || crit.length < 2 || crit.length > 10) throw new Error(`questions.${qid}: a score needs 2–10 levels`);
    const levels = crit.map(text);
    return { type: t, q: { text: qt, choices: levels, kind: "score", lang }, keys: levels };
  }
  if (t === "noul") {
    if (crit && typeof crit === "object" && !Array.isArray(crit)) {
      const tr = pyStrip(text(crit.true)), fa = pyStrip(text(crit.false));
      if (tr || fa) qt = `${qt}\nTrue if: ${tr || "—"}\nFalse if: ${fa || "—"}`;
    }
    return { type: t, q: { text: qt, choices: ["yes", "no"], kind: "verify", lang }, keys: ["yes", "no"] };
  }
  throw new Error(`questions.${qid}: unknown type ${JSON.stringify(t)} (choice, score or noul)`);
}

const r4 = (x) => Math.round(x * 1e4) / 1e4;
export function confidence(p) {
  const k = p.length;
  return k > 1 ? Math.max(0, Math.min(1, (k * Math.max(...p) - 1) / (k - 1))) : 1;
}

/** One `/v1/systemone` answer from probabilities over the question's options. */
export function answer(t, keys, p) {
  if (t === "choice") {
    const i = p.reduce((b, v, j) => (v > p[b] ? j : b), 0);
    return { type: "choice", choice: keys[i], confidence: r4(confidence(p)), probabilities: Object.fromEntries(keys.map((k, j) => [k, r4(p[j])])) };
  }
  if (t === "score") {
    return { type: "score", score: r4(p.reduce((a, v, i) => a + i * v, 0)), confidence: r4(confidence(p)),
             legend: Object.fromEntries(keys.map((k, i) => [String(i), k])), probabilities: Object.fromEntries(p.map((v, i) => [String(i), r4(v)])) };
  }
  return { type: "noul", noul: r4(p[0]) };
}

/** Router._plan: the state string the model reads and the questions that apply to it. */
export function plan(state, questions, preset, lang = "en") {
  if (state === null || ["boolean", "number"].includes(typeof state)) throw new Error("state must be a string, object or array");
  const qs = applicable({ ...(preset?.questions || {}), ...(questions || {}) }, state);
  if (!Object.keys(qs).length) {
    const keys = state && typeof state === "object" && !Array.isArray(state) ? Object.keys(state).sort().join(", ") : "text";
    throw new Error(`no question applies to this state (keys: ${keys})`);
  }
  const s = typeof state === "string" ? state : pyDumps(state);
  return { s, qs: Object.entries(qs).map(([qid, spec]) => ({ qid, ...toQuestion(qid, spec, lang) })) };
}
