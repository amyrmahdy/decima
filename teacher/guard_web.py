"""decima-guard web/API data: converters from labelled public datasets into score rows (added 2026-10-10).

Converters only: every row's text comes from a downloaded public dataset (HF cache, HF_HUB_OFFLINE=1) or a cloned public
list under data/guard/raw/web; this module relabels it into our four fixed tasks, it never writes payloads of its own.

    python -m teacher.guard_web build          # → data/guard/test-web.jsonl, data/guard/train-web.jsonl, stats on stdout
    python -m teacher.guard_web stats          # counts of the built files

Order: test first (disjoint splits: CSIC test, shengqin test, SecureCodePairs test, cvefixes held-out repos, and the
test-only sources securecode-web and BenchmarkJava), then train, deduped and decontaminated against test
(normalised exact match + word-3-shingle Jaccard ≥ 0.8, the idea of teacher/decontam_agent.py).
"""

from __future__ import annotations

import csv
import glob
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
csv.field_size_limit(1 << 30)

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/guard/raw/web"
OUT = ROOT / "data/guard"
HUB = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
MAXC = 6000
SEED = 20261010

Q = {
    "web_req": ("Is this HTTP request or input an attack?", ["benign", "suspicious", "attack"]),
    "web_class": ("Which attack class is this input?",
                  ["none", "sql injection", "cross-site scripting", "command injection", "path traversal",
                   "server-side request forgery", "xml external entity", "template injection", "ldap or nosql injection",
                   "other attack"]),
    "code_vuln": ("Is this code vulnerable?", ["not vulnerable", "vulnerable"]),
    "code_owasp": ("Which OWASP Top 10 (2021) category does this code's weakness fall under?",
                   ["none", "A01 broken access control", "A02 cryptographic failures", "A03 injection", "A04 insecure design",
                    "A05 security misconfiguration", "A06 vulnerable components", "A07 authentication failures",
                    "A08 integrity failures", "A09 logging failures", "A10 server-side request forgery"]),
}

# Official OWASP Top 10:2021 CWE mapping (owasp.org/Top10, "List of Mapped CWEs" per category).
_OWASP = {
    1: [22, 23, 35, 59, 200, 201, 219, 264, 275, 276, 284, 285, 352, 359, 377, 402, 425, 441, 497, 538, 540, 548, 552, 566,
        601, 639, 651, 668, 706, 862, 863, 913, 922, 1275],
    2: [261, 296, 310, 319, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 335, 336, 337, 338, 340, 347, 523, 720,
        757, 759, 760, 780, 818, 916],
    3: [20, 74, 75, 77, 78, 79, 80, 83, 87, 88, 89, 90, 91, 93, 94, 95, 96, 97, 98, 99, 100, 113, 116, 138, 184, 470, 471,
        564, 610, 643, 644, 652, 917],
    4: [73, 183, 209, 213, 235, 256, 257, 266, 269, 280, 311, 312, 313, 316, 419, 430, 434, 444, 451, 472, 501, 522, 525,
        539, 579, 598, 602, 642, 646, 650, 653, 656, 657, 799, 807, 840, 841, 927, 1021, 1173],
    5: [2, 11, 13, 15, 16, 260, 315, 520, 526, 537, 541, 547, 611, 614, 756, 776, 942, 1004, 1032, 1174],
    6: [937, 1035, 1104],
    7: [255, 259, 287, 288, 290, 294, 295, 297, 300, 302, 304, 306, 307, 346, 384, 521, 613, 620, 640, 798, 940, 1216],
    8: [345, 353, 426, 494, 502, 565, 784, 829, 830, 915],
    9: [117, 223, 532, 778],
    10: [918],
}
CWE2OWASP = {c: k for k, v in _OWASP.items() for c in v}


def owasp_gold(cwe) -> int | None:
    """Choice index in code_owasp for a CWE id ('CWE-89', 89, 'NVD-CWE-Other' → None)."""
    m = re.search(r"(\d+)", str(cwe or ""))
    return CWE2OWASP.get(int(m.group(1))) if m else None


def hub(repo: str) -> Path | None:
    p = sorted(glob.glob(str(HUB / f"datasets--{repo.replace('/', '--')}/snapshots/*/")))
    return Path(p[0]) if p else None


def clip(s: str) -> str:
    s = (s or "").replace("\r\n", "\n").strip()
    return s if len(s) <= MAXC else s[:MAXC]


def row(task: str, state: str, gold: int, source: str, split: str, group: str | None = None) -> dict | None:
    state = clip(state)
    if len(state) < 2:
        return None
    q, ch = Q[task]
    rid = hashlib.sha1(f"{task}\x00{state}".encode()).hexdigest()[:16]
    r = {"kind": "score", "state": state, "question": q, "choices": ch, "gold": gold, "state_lang": "en", "choice_lang": "en",
         "id": rid, "source": source, "task": task, "_split": split}
    if group:
        r["_group"] = group
    return r


def web(state, gold_req, cls, source, split):
    """web_req row (+ web_class row when the dataset names the class; cls None = class unknown)."""
    out = [row("web_req", state, gold_req, source, split)]
    if cls is not None:
        out.append(row("web_class", state, Q["web_class"][1].index(cls), source, split))
    return out


def code(state, vuln: bool, cwe, source, split, group=None):
    """code_vuln row + code_owasp row (secure code → none; vulnerable with a mapped CWE → its category)."""
    out = [row("code_vuln", state, int(vuln), source, split, group)]
    if not vuln:
        out.append(row("code_owasp", state, 0, source, split, group))
    else:
        g = owasp_gold(cwe)
        if g is not None:
            out.append(row("code_owasp", state, g, source, split, group))
    return out


# ---------------------------------------------------------------- web converters

def csic():
    import pandas as pd
    d = hub("bridge4/CSIC2010_dataset_classification")
    for sp in ("train", "test"):
        df = pd.read_parquet(d / f"data/{sp}-00000-of-00001.parquet")
        for t, y in zip(df["requests"], df["label"]):
            t = str(t).replace("\\n", "\n")
            # anomalous = attacks + malformed non-attack values, no finer label → suspicious, class unknown
            yield from (web(t, 1, None, "csic2010", sp) if y == 1 else web(t, 0, "none", "csic2010", sp))


def shengqin():
    cls = {"SQLi": "sql injection", "XSS": "cross-site scripting", "normal": "none"}
    for repo, tag in (("shengqin/web-attacks", "shengqin"), ("shengqin/web-attacks-long", "shengqin-long")):
        d = hub(repo)
        for f, sp in (("train.csv", "train"), ("validate.csv", "train"), ("test.csv", "test")):
            with open(d / f, newline="", encoding="utf-8-sig", errors="replace") as fh:
                for r in csv.DictReader(fh):
                    c = cls.get(r["text_label"])
                    if c:
                        yield from web(r["Payload"], 0 if c == "none" else 2, c, tag, sp)


def zrmarine():
    with open(hub("zrmarine/sql_injection") / "Modified_SQL_Dataset.csv", newline="", encoding="utf-8-sig", errors="replace") as fh:
        for r in csv.DictReader(fh):
            y = r["Label"].strip()
            if y in ("0", "1"):
                yield from web(r["Query"], 2 if y == "1" else 0, "sql injection" if y == "1" else "none", "zrmarine-sqli", "train")


def cmdinj():
    import html
    with open(hub("AmirAliGharesoufloo/CommandInjection") / "commandinjection.csv", newline="", encoding="utf-8-sig", errors="replace") as fh:
        for r in csv.DictReader(fh):
            y = r["Label"].strip()
            if y in ("0", "1"):
                yield from web(html.unescape(r["sentence"]), 2 if y == "1" else 0, "command injection" if y == "1" else "none",
                               "amir-cmdinj", "train")


def yangyang():
    """625k binary rows (attack class not given → web_req; normals also feed web_class none)."""
    with open(hub("YangYang-Research/web-attack-detection") / "dataset.csv", newline="", encoding="utf-8-sig", errors="replace") as fh:
        for r in csv.DictReader(fh):
            y = (r.get("Label") or "").strip()
            if y == "1":
                yield from web(r["Sentence"], 2, None, "yangyang-waf", "train")
            elif y == "0":
                yield from web(r["Sentence"], 0, "none", "yangyang-waf", "train")


# Payload lists (PayloadsAllTheThings "Intruder" files, SecLists Fuzzing): the folder names the class.
LISTS = [
    ("pat/SQL Injection/Intruder*/*", "sql injection"),
    ("pat/XSS Injection/Intruder*/*", "cross-site scripting"),
    ("pat/Command Injection/Intruder*/*", "command injection"),
    ("pat/Directory Traversal/Intruder*/*", "path traversal"),
    ("pat/File Inclusion/Intruder*/*", "path traversal"),
    ("pat/XXE Injection/Intruder*/*", "xml external entity"),
    ("pat/Server Side Template Injection/Intruder*/*", "template injection"),
    ("pat/LDAP Injection/Intruder*/*", "ldap or nosql injection"),
    ("pat/NoSQL Injection/Intruder*/*", "ldap or nosql injection"),
    ("pat/Server Side Request Forgery/Intruder*/*", "server-side request forgery"),
    ("pat/XPATH Injection/Intruder*/*", "other attack"),
    ("pat/Server Side Include Injection/Intruder*/*", "other attack"),
    ("pat/CRLF Injection/Intruder*/*", "other attack"),
    ("seclists/Fuzzing/command-injection-commix.txt", "command injection"),
    ("seclists/Fuzzing/LFI/**/*.txt", "path traversal"),
    ("seclists/Fuzzing/XXE-Fuzzing.txt", "xml external entity"),
    ("seclists/Fuzzing/template-engines-expression.txt", "template injection"),
    ("seclists/Fuzzing/LDAP.Fuzzing.txt", "ldap or nosql injection"),
    ("seclists/Fuzzing/SSI-Injection-Jhaddix.txt", "other attack"),
    ("seclists/Fuzzing/XSS/**/*.txt", "cross-site scripting"),
    ("seclists/Fuzzing/SQLi/**/*.txt", "sql injection"),
]
LIST_MIN = {"path traversal": 6, "ldap or nosql injection": 3}   # drop bare words like 'cn' or 'etc' from attribute lists


def lists():
    seen = set()
    for pat, cls in LISTS:
        for f in sorted(glob.glob(str(RAW / pat), recursive=True)):
            p = Path(f)
            if not p.is_file() or p.suffix.lower() in (".md", ".png", ".jpg", ".svg", ".avi", ".py") or "LDAP_attributes" in p.name:
                continue
            src = "pat" if "/pat/" in f else "seclists"
            for ln in p.read_text(errors="replace").splitlines():
                ln = ln.strip()
                if len(ln) < LIST_MIN.get(cls, 4) or ln.startswith("#") and len(ln) < 40 or (cls, ln) in seen:
                    continue
                seen.add((cls, ln))
                yield from web(ln, 2, cls, src, "train")


# ---------------------------------------------------------------- code converters

def cybersec_eval():
    import pandas as pd
    d = hub("walledai/CyberSecEval")
    for f in sorted(glob.glob(str(d / "*/*.parquet"))):
        if "/instruct/" not in f and "/autocomplete/" not in f:
            continue
        df = pd.read_parquet(f)
        for c, cwe in zip(df["origin_code"], df["cwe_identifier"]):   # insecure-code detector hits → all vulnerable
            yield from code(c, True, cwe, "cyberseceval", "train")


def cybernative():
    f = hub("CyberNative/Code_Vulnerability_Security_DPO") / "secure_programming_dpo.json"
    t = f.read_text()
    data = json.loads(t) if t.lstrip().startswith("[") else [json.loads(l) for l in t.splitlines() if l.strip()]
    for r in data:   # chosen = safe, rejected = vulnerable (noisy, no CWE → code_vuln + 'none' only for chosen)
        yield from code(_strip_fence(r["chosen"]), False, None, "cybernative-dpo", "train")
        yield from code(_strip_fence(r["rejected"]), True, None, "cybernative-dpo", "train")


def _strip_fence(s: str) -> str:
    m = re.search(r"```[\w+#-]*\n(.*?)```", s or "", re.S)
    return m.group(1) if m else (s or "")


def secure_code_pairs():
    d = hub("ismailtasdelen/SecureCodePairs")
    for f, sp in (("train.jsonl", "train"), ("validation.jsonl", "train"), ("test.jsonl", "test")):
        for l in open(d / f):
            r = json.loads(l)
            cwe = r.get("cwe") or ""
            if not owasp_gold(cwe) and r.get("owasp", "").startswith("A"):   # fall back to the row's own OWASP tag
                m = re.match(r"A(\d+):2021", r["owasp"])
                cwe = None
                if m:
                    k = int(m.group(1))
                    yield row("code_vuln", r["vulnerable_code"], 1, "securecodepairs", sp)
                    yield row("code_owasp", r["vulnerable_code"], k, "securecodepairs", sp)
                    yield from code(r["secure_code"], False, None, "securecodepairs", sp)
                    continue
            yield from code(r["vulnerable_code"], True, cwe, "securecodepairs", sp)
            yield from code(r["secure_code"], False, None, "securecodepairs", sp)


def security_eval():
    for l in open(RAW / "SecurityEval/dataset.jsonl"):
        r = json.loads(l)
        yield from code(r["Insecure_code"], True, r["ID"].split("_")[0], "securityeval", "train")


def cvefixes():
    """Before-fix code = vulnerable, after-fix = not vulnerable; repositories hashed, ~10% held out for test."""
    d = hub("hitoshura25/cvefixes")
    files = sorted(glob.glob(str(d / "data/*.parquet"))) if d else []
    if not files:
        print("cvefixes: not downloaded yet — skipped", file=sys.stderr)
        return
    import pyarrow.parquet as pq
    cols = pq.ParquetFile(files[0]).schema_arrow.names
    pick = lambda *names: next((c for c in names if c in cols), None)
    c_repo = pick("repo_url", "repository", "repo", "repo_name")
    c_before = pick("code_before", "vulnerable_code", "before", "func_before", "old_code")
    c_after = pick("code_after", "fixed_code", "after", "func_after", "new_code")
    c_cwe = pick("cwe_id", "cwe", "cwe_ids", "cwe_classification")
    if not (c_before and c_after):
        print(f"cvefixes: unknown schema {cols} — skipped", file=sys.stderr)
        return
    for f in files:
        pf = pq.ParquetFile(f)
        for b in pf.iter_batches(batch_size=512, columns=[c for c in (c_repo, c_before, c_after, c_cwe) if c]):
            for r in b.to_pylist():
                before, after = r.get(c_before) or "", r.get(c_after) or ""
                if not before or not after or before == after or len(before) > MAXC:   # whole too-long files would hide the flaw
                    continue
                repo = str(r.get(c_repo) or "")
                sp = "test" if int(hashlib.md5(repo.encode()).hexdigest(), 16) % 10 == 0 else "train"
                cwe = r.get(c_cwe)
                if isinstance(cwe, (list, tuple)):
                    cwe = cwe[0] if cwe else None
                yield from code(before, True, cwe, "cvefixes", sp, repo)
                yield from code(after, False, None, "cvefixes", sp, repo)


def securecode_web():
    """CC BY-NC-SA → test only. The first code block after a 'vulnerable' heading vs after a 'secure' heading."""
    d = hub("scthornton/securecode-web")
    for f in sorted(glob.glob(str(d / "data/*.jsonl"))):
        t = open(f).read().strip()
        if not t:
            continue
        try:
            objs = [json.loads(t)]
        except ValueError:
            objs = [json.loads(l) for l in t.splitlines() if l.strip()]
        for o in objs:
            cwe = (o.get("metadata") or {}).get("cwe")
            conv = o.get("conversations") or []
            msg = next((c.get("value") or "" for c in conv if c.get("from") in ("assistant", "gpt")), "")
            vuln = sec = None
            for m in re.finditer(r"([^\n]*)\n+```[\w+#-]*\n(.*?)```", msg, re.S):
                head = msg[max(0, m.start() - 200):m.start() + len(m.group(1))].lower()
                if vuln is None and "vulnerab" in head and "secure" not in m.group(1).lower():
                    vuln = m.group(2)
                elif vuln is not None and sec is None and ("secure" in head or "fixed" in head):
                    sec = m.group(2)
            if vuln and sec:
                yield from code(vuln, True, cwe, "securecode-web", "test")
                yield from code(sec, False, None, "securecode-web", "test")


def benchmark_java():
    """OWASP Benchmark (GPL → test only): expectedresults-1.2.csv gives real-vulnerability + CWE per test case."""
    base = RAW / "BenchmarkJava"
    src = base / "src/main/java/org/owasp/benchmark/testcode"
    for ln in open(base / "expectedresults-1.2.csv"):
        if ln.startswith("#"):
            continue
        name, cat, real, cwe = [x.strip() for x in ln.split(",")[:4]]
        p = src / f"{name}.java"
        if not p.exists():
            continue
        t = p.read_text(errors="replace")
        t = re.sub(r"^/\*\*.*?\*/\s*", "", t, flags=re.S)          # licence header
        t = re.sub(r"BenchmarkTest\d+", "Handler", t)              # the name would leak the id
        t = re.sub(r'@WebServlet\(value = "/[^"]*"\)', '@WebServlet(value = "/handler")', t)   # the path names the category
        yield from code(t, real == "true", cwe, "owasp-benchmark", "test")


CONVERTERS = [csic, shengqin, zrmarine, cmdinj, yangyang, lists, cybersec_eval, cybernative, secure_code_pairs,
              security_eval, cvefixes, securecode_web, benchmark_java]

# ---------------------------------------------------------------- balancing, dedupe, decontam

# train quotas per (task, gold); filled round-robin across sources so no single list dominates; plus a per-source cap
TRAIN_QUOTA = {"web_req": {0: 34000, 1: 12000, 2: 34000}, "web_class": {0: 12000, "*": 9000},
               "code_vuln": {0: 16000, 1: 16000}, "code_owasp": {0: 5000, "*": 4000}}
TEST_QUOTA = {"web_req": {0: 600, 1: 400, 2: 600}, "web_class": {0: 300, "*": 200},
              "code_vuln": {0: 700, 1: 700}, "code_owasp": {0: 300, "*": 150}}
SOURCE_CAP = 40000


def norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\d+", "0", s)
    return re.sub(r"\s+", " ", s).strip()


def shingles(s: str) -> set:
    w = re.findall(r"[a-z_]+|[^\sa-z_0]", norm(s))
    return {" ".join(w[i:i + 3]) for i in range(max(1, len(w) - 2))}


def quota(table, gold):
    return table.get(gold, table.get("*", 0))


def pick(rows: list[dict], table: dict, rng: random.Random, source_cap: int | None = None) -> list[dict]:
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[(r["task"], r["gold"])][r["source"]].append(r)
    out, used = [], Counter()
    for (task, gold), srcs in sorted(by.items()):
        q = quota(table[task], gold)
        for v in srcs.values():
            rng.shuffle(v)
        pools = {s: iter(v) for s, v in sorted(srcs.items())}
        got = []
        while len(got) < q and pools:
            for s in list(pools):
                if len(got) >= q:
                    break
                r = next(pools[s], None)
                if r is None or (source_cap and used[s] >= source_cap):
                    pools.pop(s)
                    continue
                got.append(r)
                used[s] += 1
        out += got
    return out


def build():
    rng = random.Random(SEED)
    rows = {"train": [], "test": []}
    for conv in CONVERTERS:
        n = Counter()
        try:
            for r in conv():
                if r:
                    rows[r["_split"]].append(r)
                    n[r["_split"]] += 1
        except FileNotFoundError as e:
            print(f"{conv.__name__}: missing {e.filename} — skipped", file=sys.stderr)
        print(f"{conv.__name__:20s} train {n['train']:7d}  test {n['test']:6d}", file=sys.stderr)

    def dedupe(rs):
        seen, out = {}, []
        for r in rs:
            k = (r["task"], norm(r["state"]))
            if k in seen:
                if seen[k] != r["gold"]:
                    seen[k] = -1          # conflicting labels → drop both
                continue
            seen[k] = r["gold"]
            out.append(r)
        return [r for r in out if seen[(r["task"], norm(r["state"]))] != -1]

    test = pick(dedupe(rows["test"]), TEST_QUOTA, rng)
    test_norm = {norm(r["state"]) for r in test}
    test_groups = {r["_group"] for r in rows["test"] if r.get("_group")}
    test_sh = [shingles(r["state"]) for r in test]
    inv = defaultdict(list)
    for i, s in enumerate(test_sh):
        for g in s:
            inv[g].append(i)
    common = {g for g, v in inv.items() if len(v) > 50}

    def contaminated(r) -> bool:
        if norm(r["state"]) in test_norm or (r.get("_group") and r["_group"] in test_groups):
            return True
        s = shingles(r["state"])
        if len(s) < 2:
            return False
        hits = Counter(i for g in s if g not in common for i in inv.get(g, ()))
        for i, h in hits.most_common(5):
            t = test_sh[i]
            if len(s & t) / len(s | t) >= 0.8:
                return True
        return False

    tr = dedupe(rows["train"])
    before = len(tr)
    tr = [r for r in tr if not contaminated(r)]
    print(f"decontam: dropped {before - len(tr)} of {before} train rows near test", file=sys.stderr)
    train = pick(tr, TRAIN_QUOTA, rng, SOURCE_CAP)
    rng.shuffle(train)
    rng.shuffle(test)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rs in (("test-web.jsonl", test), ("train-web.jsonl", train)):
        tmp = OUT / (name + ".tmp")
        with open(tmp, "w") as fh:
            for r in rs:
                fh.write(json.dumps({k: v for k, v in r.items() if not k.startswith("_")}, ensure_ascii=False) + "\n")
        tmp.replace(OUT / name)
    stats()


def stats():
    for name in ("test-web.jsonl", "train-web.jsonl"):
        p = OUT / name
        if not p.exists():
            continue
        rs = [json.loads(l) for l in open(p)]
        print(f"\n== {name}: {len(rs)} rows")
        for task in Q:
            sub = [r for r in rs if r["task"] == task]
            g = Counter(Q[task][1][r["gold"]] for r in sub)
            print(f"  {task:11s} {len(sub):6d}  " + ", ".join(f"{k}={v}" for k, v in sorted(g.items(), key=lambda x: -x[1])))
        s = Counter((r["task"], r["source"]) for r in rs)
        for (t, src), n in sorted(s.items()):
            print(f"    {t:11s} {src:18s} {n}")


if __name__ == "__main__":
    {"build": build, "stats": stats}[sys.argv[1] if len(sys.argv) > 1 else "build"]()
