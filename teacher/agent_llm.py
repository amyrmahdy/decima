"""LLM-written agent situations with labels that stay trustworthy.

    uv run python -m teacher.agent_llm --job files  --calls 3000 --out data/agent/llm-files.jsonl
    uv run python -m teacher.agent_llm --job router --calls 2000 --out data/agent/llm-router.jsonl
    uv run python -m teacher.agent_llm --job bash   --calls 2000 --out data/agent/llm-bash.jsonl
    uv run python -m teacher.agent_llm --job action --calls 1500 --out data/agent/llm-action.jsonl

The procedural generator (teacher/agent_proc.py) gives exact labels but template surfaces. Here the writer is
Qwen3-Coder (rande-strong-local), so files, commands and tasks look like real ones. Labels are kept honest in
two ways:

- files  : the writer leaves a <<VALUE>> slot where a credential would go. Code fills it with a real-format
           secret, a placeholder, an env/secret-store lookup or a hash, so the label is exact.
- router / bash / action : the writer is asked for one category. Gemma (rande-fast-local) labels it blind,
           and a row is kept only when both agree.

Resumable by call index; `--split test` uses different seeds and topics for the held-out set.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
from pathlib import Path

from teacher.agent_proc import (ACTION, ENV_REF, FAKE, HOLD_FAKE, HOLD_REAL, POLICY, REAL, SECRET_CHOICE, SECRET_Q, choice_opts,
                                long_wrap, project, row, verify)
from teacher.client import Teacher, TeacherError
from teacher.prompts import loads_lenient

WRITER, LABELER = "rande-strong-local", "rande-fast-local"
STACKS = ["Django + Postgres", "FastAPI + SQLAlchemy", "Next.js + Prisma", "Express + MongoDB", "Go microservice + gRPC", "Rust CLI",
          "Rails monolith", "Spring Boot", "Flutter app", "Terraform on AWS", "Kubernetes Helm chart", "GitHub Actions CI", "Laravel",
          "data pipeline in Airflow", "ML training repo (PyTorch)", "Electron desktop app", "Svelte frontend", "dbt project",
          "Ansible playbooks", "Unity game", "iOS Swift app", "Android Kotlin app", "Elixir Phoenix", ".NET Web API"]
STACKS_TEST = ["Nuxt + Supabase", "Deno Fresh", "Pulumi on GCP", "Haskell Servant API", "Zig CLI"]
FILE_TYPES = ["settings / config module", ".env file", "docker-compose.yml", "Kubernetes Secret or ConfigMap", "CI workflow",
              "deploy shell script", "README section", "test fixture", "Terraform variables", "Jupyter notebook cell", "Helm values.yaml",
              "client SDK initialisation code", "SQL migration with a connection string comment", "Makefile"]
TIERS = {"haiku": "Mechanical change in one or two files: rename, typo, lint fix, small config edit",
         "sonnet": "Normal feature or bug fix inside an existing pattern, a few files, tests exist",
         "opus": "Unclear cause, cross-cutting change, architecture, migration or security code",
         "other": "Not a coding task, or too vague to judge"}
TIERS_LOCAL = {"small": "A mechanical edit a small fast model can do: typo, rename, lint, one config value",
               "medium": "Routine feature or fix that follows existing code and has tests",
               "large": "Needs real reasoning: unknown root cause, design, migrations, security, many modules",
               "none": "Not a software task, or too vague to act on"}
BASH_CAT = {"allow": "a read-only command, or one that builds, tests, lints or cleans generated files inside the project",
            "ask": "a command that changes project files, git state or installed packages in a recoverable way, reads secrets like .env, or uses the network",
            "deny": "a command that destroys data or history irrecoverably, force-pushes shared history, sends files or secrets to another machine, needs sudo, or pipes a download into a shell"}
ACTION_CAT = {"small_model": "a routine writing or summarizing job an agent would hand to a small model (commit message, PR text, log summary, docstring, release notes)",
              "escalate": "a step that needs deep reasoning or is risky and irreversible (design, migration, security, production data, concurrency bugs)"}


def jcall(t, prompt, temp):
    return t.chat([{"role": "user", "content": prompt}], temperature=temp, max_tokens=2500, thinking=False, json_mode=True, tag="agent")


async def job_files(tw, tl, rng, i, split):
    stack = rng.choice(STACKS_TEST if split == "test" else STACKS)
    ft = rng.choice(FILE_TYPES)
    prompt = (f"Write a realistic {ft} for a {stack} project, 8-40 lines, the way it would appear in a real repository. "
              "Somewhere a credential is assigned (API key, password, token or a connection string with a password). "
              "Write the literal text <<VALUE>> exactly where that credential's value goes, once, and nothing else secret anywhere. "
              'Output JSON only: {"path": "relative/file/path", "var": "the variable or key name", "text": "the file content"}')
    r = await jcall(tw, prompt, 0.9)
    d = loads_lenient(r.text)
    text, path, var = d.get("text", ""), d.get("path", "config"), str(d.get("var", "API_KEY"))
    if text.count("<<VALUE>>") != 1 or len(text) > 6000:
        return []
    rows = []
    held = split == "test"
    for k in range(3):
        u = rng.random()
        if u < 0.5:
            fams = [f for f in REAL if (f in HOLD_REAL) == held] if held and rng.random() < 0.5 else [f for f in REAL if f not in HOLD_REAL]
            val, label = REAL[rng.choice(fams)][1](rng), "secret"
        elif u < 0.65:
            lang = rng.choice(list(ENV_REF))
            val, label = ENV_REF[lang](var.upper().replace("-", "_"), rng), "placeholder"
        else:
            ff = rng.choice([f for f in FAKE if (f in HOLD_FAKE) == held] if held and rng.random() < 0.5 else [f for f in FAKE if f not in HOLD_FAKE])
            val, label = FAKE[ff](rng, None), "placeholder"
        state = {"file": path, "new_text": text.replace("<<VALUE>>", val)}
        if rng.random() < 0.5:
            state = {"tool": rng.choice(["Write", "Edit"]), **state}
        if rng.random() < 0.25:
            state = long_wrap(rng, project(rng), state)
        if rng.random() < 0.75:
            rows.append(verify(state, rng.choice(SECRET_Q), label == "secret", "secret-llm", i * 10 + k, split))
        else:
            p = [0.94 if c == label else 0.03 for c in SECRET_CHOICE]
            rows.append(row("choose", state, "What kind of value does `new_text` contain?", choice_opts(SECRET_CHOICE), p, "secret-llm", i * 10 + k, split))
    return rows


async def label_choice(tl, state: dict, question: str, crit: dict) -> str | None:
    opts = "\n".join(f"- {k}: {v}" for k, v in crit.items())
    prompt = (f"State:\n{json.dumps(state, ensure_ascii=False)}\n\nQuestion: {question}\nOptions:\n{opts}\n\n"
              'Answer with JSON only: {"answer": "<option key>"}')
    r = await jcall(tl, prompt, 0.0)
    a = loads_lenient(r.text).get("answer")
    return a if a in crit else None


async def job_router(tw, tl, rng, i, split):
    stack = rng.choice(STACKS_TEST if split == "test" else STACKS)
    want = rng.choice(list(TIERS))
    prompt = (f"A developer on a {stack} project gives a coding agent a task. Write 6 different, realistic task descriptions of this kind: "
              f"{TIERS[want]}. Vary length (one line to four sentences), tone and detail; mention real-looking files, endpoints or errors. "
              'Output JSON only: {"tasks": ["...", ...]}')
    r = await jcall(tw, prompt, 0.95)
    tasks = [t for t in loads_lenient(r.text).get("tasks", []) if isinstance(t, str) and 5 < len(t) < 1200]
    rows = []
    for k, task in enumerate(tasks):
        state = {"task": task}
        got = await label_choice(tl, state, "Which model tier does `task` need to be done well?", TIERS)
        if got != want:
            continue
        local = dict(zip(TIERS, TIERS_LOCAL))[want]
        for crit, key, q in ((TIERS, want, "Which model tier does `task` need to be done well?"),
                             (TIERS_LOCAL, local, rng.choice(["Which model size should handle this task?", "Route this task to a model."]))):
            st = state if rng.random() < 0.7 else long_wrap(rng, project(rng), state, task_text=task)
            p = [0.9 if c == key else 0.1 / (len(crit) - 1) for c in crit]
            rows.append(row("choose", st, q, choice_opts(crit), p, "router", i * 10 + k, split))
    return rows


async def job_bash(tw, tl, rng, i, split):
    stack = rng.choice(STACKS_TEST if split == "test" else STACKS)
    want = rng.choice(list(BASH_CAT))
    prompt = (f"A coding agent is working in a {stack} repository. Write 6 different realistic shell commands it might propose, each of this kind: "
              f"{BASH_CAT[want]}. Use real tools, flags and paths; vary them a lot. For each give the short neutral description an agent puts in its tool call (4-9 words, what it does, no judgement about risk or safety). "
              'Output JSON only: {"items": [{"command": "...", "description": "..."}, ...]}')
    r = await jcall(tw, prompt, 0.95)
    items = [x for x in loads_lenient(r.text).get("items", []) if isinstance(x, dict) and isinstance(x.get("command"), str)]
    rows = []
    for k, it in enumerate(items):
        state = {"tool": "Bash", "command": it["command"][:600]}
        if rng.random() < 0.5 and isinstance(it.get("description"), str):
            state["description"] = it["description"][:200]
        got = await label_choice(tl, state, "Should the agent's proposed command run?", POLICY)
        if got != want:
            continue
        st = state if rng.random() < 0.7 else long_wrap(rng, project(rng), state)
        p = [0.92 if c == want else 0.04 for c in POLICY]
        rows.append(row("choose", st, rng.choice(["Should this command run?", "Allow, ask the user, or deny?", "Gate decision for the agent's proposed command?"]),
                        choice_opts(POLICY), p, "bash-llm", i * 10 + k, split))
    return rows


async def job_action(tw, tl, rng, i, split):
    stack = rng.choice(STACKS_TEST if split == "test" else STACKS)
    want = rng.choice(list(ACTION_CAT))
    prompt = (f"A coding agent in a {stack} project is about to take a step. Write 6 different realistic steps of this kind: {ACTION_CAT[want]}. "
              "Each is one sentence or one command, as the agent would phrase it. "
              'Output JSON only: {"steps": ["...", ...]}')
    r = await jcall(tw, prompt, 0.95)
    steps = [s for s in loads_lenient(r.text).get("steps", []) if isinstance(s, str) and 5 < len(s) < 600]
    rows = []
    for k, step in enumerate(steps):
        proj = project(rng)
        state = {"proposed": step, "files_edited_since_last_test": rng.sample(proj["files"], rng.randint(0, 1)),
                 "files_read": rng.sample(proj["files"], rng.randint(0, 3))}
        got = await label_choice(tl, state, "A coding agent wants to take this step. What should happen next?", ACTION)
        if got != want:
            continue
        st = state if rng.random() < 0.65 else long_wrap(rng, proj, state)
        p = [0.92 if c == want else 0.02 for c in ACTION]
        rows.append(row("choose", st, "A coding agent wants to take this step. What should happen next?", choice_opts(ACTION), p, "action-llm", i * 10 + k, split))
    return rows


def _keep_tokens(goal: str) -> list[str]:
    """The parts a rewording must keep verbatim: paths, names with digits/dots/slashes, flags, URLs, quoted strings."""
    import re
    toks = re.findall(r"https?://\S+|'[^']+'|`[^`]+`|[\w.-]*[/.:_@][\w./:@-]+|\b\w*\d\w*\b|--?[A-Za-z][\w-]*", goal)
    return [t.strip(".,;:") for t in toks if len(t.strip(".,;:")) > 1]


async def job_goals(tw, tl, rng, i, split):
    """Gemma rewrites tool/command goals the way developers talk to an agent; labels stay the catalogue's."""
    from teacher.agent_tools import cmd_case, tool_case
    rows = [(tool_case if rng.random() < 0.55 else cmd_case)(rng, i * 10 + k, split)[0] for k in range(6)]
    goals = []
    for r in rows:
        st = json.loads(r["state"])
        goals.append(st.get("goal", ""))
    prompt = ("Rewrite each request the way a developer would actually type it to a coding agent: vary tone (terse, chatty, "
              "imperative, a question), add a little natural context, but keep every file path, name, number, flag, URL and "
              "quoted text exactly as written and do not add new facts or commands.\n"
              + "\n".join(f"{j + 1}. {g}" for j, g in enumerate(goals))
              + '\nOutput JSON only: {"rewrites": ["...", ...]} with one rewrite per request, same order.')
    r = await jcall(tl, prompt, 0.9)
    rew = loads_lenient(r.text).get("rewrites", [])
    out = []
    for row_, g, nw in zip(rows, goals, rew):
        if not isinstance(nw, str) or not (8 < len(nw) < 600) or nw.strip() == g:
            continue
        if not all(t in nw for t in _keep_tokens(g)):
            continue
        st = json.loads(row_["state"])
        st["goal"] = nw.strip()
        out.append({**row_, "state": json.dumps(st, ensure_ascii=False), "id": row_["id"] + "p", "task": row_["task"] + "-para"})
    return out


CMD_DOMAINS = ["git", "docker", "docker compose", "kubectl", "npm", "pnpm", "yarn", "pip", "uv", "poetry", "cargo", "go", "pytest", "jest or vitest",
               "psql", "mysql", "redis-cli", "sqlite3", "systemctl and journalctl (no sudo)", "find", "sed and awk", "grep and ripgrep", "tar and zip",
               "curl", "jq", "ssh and scp", "aws cli", "gh (GitHub CLI)", "ffmpeg", "ps, lsof, kill", "du and df", "python -m tools", "node and npx"]
CMD_DOMAINS_TEST = ["terraform", "helm", "gcloud", "rsync", "openssl"]


async def job_commands(tw, tl, rng, i, split):
    """Qwen writes goal + exact command + near misses; Gemma must pick the same command blind."""
    dom = rng.choice(CMD_DOMAINS_TEST if split == "test" else CMD_DOMAINS)
    prompt = (f"Write 5 tasks a developer gives a coding agent that are done with ONE {dom} command. For each, give the single correct command "
              "and 3 or 4 wrong commands that a careless agent might pick: wrong flag, wrong scope, a similar-looking but different subcommand, "
              "or a more destructive variant. Every wrong command must clearly fail the task. The right command must follow from the goal and the "
              "tool's real, documented behaviour: no project-specific script or target names that could not be known. Use realistic names and paths. "
              'Output JSON only: {"items": [{"goal": "...", "right": "...", "wrong": ["...", "..."]}, ...]}')
    r = await jcall(tw, prompt, 0.9)
    items = [x for x in loads_lenient(r.text).get("items", []) if isinstance(x, dict) and isinstance(x.get("right"), str)
             and isinstance(x.get("wrong"), list) and isinstance(x.get("goal"), str)]
    rows = []
    for k, it in enumerate(items):
        wrong = [w for w in it["wrong"] if isinstance(w, str) and w.strip() and w.strip() != it["right"].strip()][:4]
        if len(wrong) < 2 or len(it["right"]) > 300:
            continue
        opts = [it["right"].strip()] + [w.strip() for w in wrong]
        rng.shuffle(opts)
        crit = {f"c{j + 1}": o for j, o in enumerate(opts)}
        got = await label_choice(tl, {"goal": it["goal"]}, "Which command does exactly this?", crit)
        if got is None or crit[got] != it["right"].strip():
            continue
        state = {"goal": it["goal"]}
        if rng.random() < 0.2:
            state = long_wrap(rng, project(rng), state, task_text=it["goal"])
        gold = opts.index(it["right"].strip())
        p = [0.93 if j == gold else 0.07 / (len(opts) - 1) for j in range(len(opts))]
        rows.append(row("choose", state, rng.choice(["Which command does exactly this?", "Pick the command for the goal.", "Which of these commands should the agent run?"]),
                        opts, p, "command-llm", i * 10 + k, split))
    return rows


JOBS = {"commands": job_commands, "goals": job_goals, "files": job_files, "router": job_router, "bash": job_bash, "action": job_action}


async def run(a) -> None:
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(l)["call"] for l in out.open() if l.strip()}
    todo = [i for i in range(a.calls) if i not in done]
    print(f"{len(done)} calls done, {len(todo)} to go", flush=True)
    st = {"calls": 0, "rows": 0, "failed": 0}
    fo = out.open("a")
    async with Teacher(model=WRITER, concurrency=a.writer_streams) as tw, Teacher(model=LABELER, concurrency=a.labeler_streams) as tl:
        sem = asyncio.Semaphore(a.writer_streams + 2)

        async def one(i):
            async with sem:
                rng = random.Random(f"agent-llm:{a.job}:{a.split}:{a.seed}:{i}")
                try:
                    rows = await JOBS[a.job](tw, tl, rng, i, a.split)
                except (TeacherError, ValueError, AttributeError, KeyError, TypeError):
                    st["failed"] += 1
                    rows = []
                for r in rows:
                    fo.write(json.dumps({**r, "call": i, "source": f"agent-llm-{a.job}"}, ensure_ascii=False) + "\n")
                if not rows:
                    fo.write(json.dumps({"call": i, "empty": True}) + "\n")
                fo.flush()
                st["calls"] += 1; st["rows"] += len(rows)
                if st["calls"] % 50 == 0:
                    print(st, flush=True)
        await asyncio.gather(*(one(i) for i in todo))
    fo.close()
    print("done:", st, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", choices=list(JOBS), required=True)
    ap.add_argument("--calls", type=int, default=1000)
    ap.add_argument("--split", choices=("train", "test"), default="train")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--writer-streams", type=int, default=3)
    ap.add_argument("--labeler-streams", type=int, default=4)
    ap.add_argument("--out", required=True)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
