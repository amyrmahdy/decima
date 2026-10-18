// Runs the model off the main thread so the page stays responsive while it loads and decides.
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.wasm.min.mjs";
import { Tokenizer } from "https://cdn.jsdelivr.net/npm/@huggingface/tokenizers@0.2.0/dist/tokenizers.min.mjs";
import { Decima, clearCache, clearAllCaches } from "./decima.js";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";
ort.env.logLevel = "error";

let model = null;
let modelUrl = null;                 // one model at a time: loading another releases the previous one (memory)
let chain = Promise.resolve();       // one operation at a time: an ORT session runs one call at a time

async function handle({ id, op, ...a }) {
  const reply = (msg) => postMessage({ id, ...msg });
  try {
    if (op === "load") {
      // Threads need cross-origin isolation (COOP/COEP), which static hosting usually lacks: then one thread.
      const numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
      if (!model || modelUrl !== a.url) {
        if (model) { const old = model; model = null; modelUrl = null; await old.release(); }
        model = await Decima.load(a.url, { ort, Tokenizer, numThreads, onProgress: (p) => reply({ progress: p }),
                                           ...(a.cacheName ? { cacheName: a.cacheName } : {}), ...(a.sizes ? { sizes: a.sizes } : {}) });
        modelUrl = a.url;
      }
      reply({ result: { ...model.stats, threads: numThreads,
                        cfg: { max_state_tokens: model.cfg.max_state_tokens, question_in_state: !!model.cfg.question_in_state } } });
    } else if (op === "count") {
      reply({ result: a.texts.map((t) => model.count(t)) });
    } else if (op === "decide") {
      reply({ result: await model.decide(a.state, a.q) });
    } else if (op === "decideMany") {
      const out = [];
      for (const it of a.items) out.push(await model.decide(it.state, it.q));
      reply({ result: out });
    } else if (op === "bench") {
      // first call (may encode the option set), then `reps` timed calls with the option encodings cached
      const t0 = performance.now();
      await model.decide(a.state, a.q);
      const first = performance.now() - t0;
      const lat = [];
      for (let i = 0; i < a.reps; i++) lat.push((await model.decide(a.state, a.q)).latency_ms);
      reply({ result: { first, lat } });
    } else if (op === "clearCache") {
      reply({ result: a.all ? await clearAllCaches() : await clearCache() });
    } else throw new Error(`unknown op ${op}`);
  } catch (e) {
    reply({ error: String((e && e.message) || e) });
  }
}

onmessage = (e) => { chain = chain.then(() => handle(e.data)); };
