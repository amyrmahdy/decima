"""Exact-label data for choosing tools and commands in an agent loop.

    uv run python -m teacher.agent_tools --n 90000 --out data/agent/tools-train.jsonl
    uv run python -m teacher.agent_tools --n 3000 --split test --out data/agent/tools-test.jsonl

Three decisions, each with the options rebuilt from the live state (the shape every Jev-style loop uses):

- tool    : which tool fits this step? Claude Code's own tools, sometimes mixed with MCP tools (GitHub, Slack, Sentry,
            Postgres…). Only the tools "available" in that state are offered, so the same intent can have a different
            right answer (open a PR: github.create_pull_request if present, else Bash).
- command : a goal and 4–7 candidate commands; the wrong ones are near misses (reset --soft vs --hard, --lf vs -x,
            npm ci vs npm install), wrong scope, or the destructive variant.
- step    : after an event (a test failed at file:line, an edit, a green run, an empty search), which concrete action next.

Labels come from the catalogue, never from a model. The test split holds out a quarter of the command catalogue
entries, some tools, and its own phrasings.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from teacher.agent_proc import MODS, long_wrap, project, row

TOOLS = {
    "Read": "Read a file whose path you already know",
    "Glob": "Find files by name or path pattern",
    "Grep": "Search file contents for a string or regex",
    "Edit": "Replace an exact piece of text in an existing file",
    "Write": "Create a new file, or replace a whole file",
    "Bash": "Run a shell command: tests, builds, git, package managers, scripts",
    "WebFetch": "Fetch and read a specific URL",
    "WebSearch": "Search the web when you do not know where the answer is",
    "Task": "Hand a broad, multi-step investigation to a sub-agent",
    "TodoWrite": "Write or update the task checklist",
    "AskUserQuestion": "Ask the user when the decision is theirs or the requirement is unclear",
}
MCP = {
    "github.create_pull_request": "Open a pull request on GitHub",
    "github.search_issues": "Search GitHub issues and pull requests",
    "slack.post_message": "Post a message to a Slack channel",
    "sentry.get_issue": "Read a Sentry error by its issue ID",
    "postgres.query": "Run a read-only SQL query on the app database",
    "linear.create_issue": "Create a Linear ticket",
    "browser.screenshot": "Open a URL in a browser and take a screenshot",
}
FILES = ["src/{m}/service.py", "src/{m}/views.py", "app/models/{m}.rb", "src/{m}.ts", "internal/{m}/handler.go", "lib/{m}.ex", "tests/test_{m}.py"]
FUNCS = ["parse_date", "calculateTotal", "get_user", "applyCoupon", "retry_with_backoff", "render_invoice", "validateEmail", "flush_cache"]
INTENTS = [   # (tool, [goal templates], extra state builder)
    ("Read", ["Look at what is in {f}", "Check how {fn} in {f} works", "Open {f} around line {n}", "See the current contents of {f} before changing it"], "known"),
    ("Glob", ["Find all the test files for {m}", "Where are the database migration files?", "List every Dockerfile in the repo", "Which files are named {m}_config.*?"], None),
    ("Grep", ["Find every place that calls {fn}", "Search the codebase for the string '{s}'", "Which files import the {m} module?", "Where is the env var {env} read?"], None),
    ("Edit", ["Change the retry count in {f} from 3 to 5", "Fix the typo 'recieve' in {f}", "Rename the local variable tmp to total inside {fn} in {f}"], "read"),
    ("Write", ["Create a new file {nf} with the CLI entry point", "Add a brand-new {nf} for the {m} package", "Write the first version of {nf}"], "new"),
    ("Bash", ["Run the {m} tests", "Install the {pkg} package", "See what git thinks has changed", "Build the project", "Check which Python version is installed"], None),
    ("WebFetch", ["Read the changelog at https://github.com/{pkg}/{pkg}/releases", "Check the API reference at https://docs.{pkg}.dev/api", "Look at the migration guide at https://{pkg}.io/upgrade"], None),
    ("WebSearch", ["Find out what the error '{err}' from {pkg} means", "Look up whether {pkg} supports Python 3.13", "Find the recommended way to paginate in {pkg}"], None),
    ("Task", ["Investigate across the whole repository why requests sometimes take 30 seconds, it touches many services",
              "Survey every module that writes to the orders table and summarize how each one does it"], None),
    ("TodoWrite", ["Break the database migration into steps and track them", "Write down the remaining steps for this refactor so none get lost"], None),
    ("AskUserQuestion", ["Decide between Postgres and MySQL for the new service; the user has not said which", "The spec doesn't say whether deleted users keep their invoices",
                         "Choose the public name of the new API endpoint; product hasn't decided"], None),
    ("github.create_pull_request", ["Open a pull request for this branch"], "mcp"),
    ("github.search_issues", ["Check whether someone already reported the login loop bug"], "mcp"),
    ("slack.post_message", ["Tell #eng that the deploy is done"], "mcp"),
    ("sentry.get_issue", ["Read what Sentry issue {sentry} says"], "mcp"),
    ("postgres.query", ["How many orders were created today?", "Count the users who signed up this week"], "mcp"),
    ("linear.create_issue", ["File a ticket for the flaky date test"], "mcp"),
    ("browser.screenshot", ["See what the checkout page looks like now at https://staging.acme.io/checkout"], "mcp"),
]
FALLBACK = {"github.create_pull_request": "Bash", "github.search_issues": "WebSearch", "postgres.query": "Bash", "browser.screenshot": "WebFetch",
            "sentry.get_issue": "WebFetch", "slack.post_message": "AskUserQuestion", "linear.create_issue": "TodoWrite"}
HOLD_TOOLS = {"browser.screenshot", "linear.create_issue"}


def ents(rng, proj):
    m = rng.choice(MODS)
    return dict(m=m, f=rng.choice(FILES).format(m=m), fn=rng.choice(FUNCS), n=rng.randint(10, 400), s=rng.choice(["TODO(remove)", "legacy_mode", "X-Request-Id"]),
                env=rng.choice(["DATABASE_URL", "STRIPE_KEY", "LOG_LEVEL"]), nf=rng.choice(["cli/main.py", f"src/{m}/README.md", f"packages/{m}/index.ts"]),
                pkg=rng.choice(["fastapi", "prisma", "pydantic", "axios", "tokio", "celery"]), err=rng.choice(["ECONNRESET", "MaxRetryError", "P2002 Unique constraint failed"]),
                sentry=f"ACME-{rng.randint(100, 9999)}")


def tool_case(rng, i, split):
    test = split == "test"
    proj = project(rng)
    intents = [x for x in INTENTS if (x[0] in HOLD_TOOLS) == test] if test and rng.random() < 0.3 else [x for x in INTENTS if x[0] not in HOLD_TOOLS]
    tool, goals, kind = rng.choice(intents)
    e = ents(rng, proj)
    goal = rng.choice(goals).format(**e)
    avail = dict(TOOLS)
    mcp_on = kind == "mcp" or rng.random() < 0.3
    if mcp_on:
        for k in rng.sample(list(MCP), rng.randint(2, len(MCP))):
            avail[k] = MCP[k]
    if kind == "mcp" and rng.random() < 0.3:                  # the MCP tool is not connected → the built-in fallback is right
        avail.pop(tool, None)
        tool = FALLBACK[tool]
    elif kind == "mcp":
        avail[tool] = MCP[tool]
    if rng.random() < 0.5:                                    # a smaller tool palette, always containing the answer
        keep = {tool} | set(rng.sample([k for k in avail if k != tool], rng.randint(3, min(8, len(avail) - 1))))
        avail = {k: v for k, v in avail.items() if k in keep}
    items = list(avail.items())
    rng.shuffle(items)
    avail = dict(items)
    state = {"goal": goal}
    if kind == "known":
        state["known_files"] = [e["f"]] + rng.sample(proj["files"], 2)
    if kind == "read":
        state["files_read"] = [e["f"]]
    if kind == "new":
        state["note"] = f"{e['nf']} does not exist yet"
    if rng.random() < 0.3:
        state = long_wrap(rng, proj, state, task_text=goal)
    q = rng.choice(["Which tool should the agent use for this step?", "Pick the tool for the next step.", "What is the right tool here?"])
    p = [0.92 if k == tool else 0.08 / (len(avail) - 1) for k in avail]
    return [row("choose", state, q, [f"{k}: {v}" for k, v in avail.items()], p, "tool", i, split)]


# goal → right command → near-miss distractors. {b} branch, {f} file, {t} test name, {p} package, {c} container, {pod} pod, {u} URL
CMD = [
    (["undo the last commit but keep its changes", "take back the last commit, keep the work in the tree"], "git reset --soft HEAD~1",
     ["git reset --hard HEAD~1", "git revert HEAD", "git checkout HEAD~1", "git commit --amend"]),
    (["show the changes that are staged for commit", "what exactly is staged right now?"], "git diff --staged",
     ["git diff", "git status", "git log -p -1", "git show HEAD"]),
    (["run only {t} in {tf}", "run just the one test {t}"], "pytest {tf}::{t}",
     ["pytest {tf}", "pytest tests/", "pytest -k cart", "pytest --lf"]),
    (["stop the test run at the first failure", "fail fast on the first broken test"], "pytest -x",
     ["pytest -v", "pytest --lf", "pytest -q", "pytest --maxfail=10"]),
    (["rerun only the tests that failed last time", "just the previously failing tests again"], "pytest --lf",
     ["pytest -x", "pytest --ff", "pytest -v", "pytest"]),
    (["create a branch {b} and switch to it", "start new branch {b} from here"], "git switch -c {b}",
     ["git branch {b}", "git checkout {b}", "git push -u origin {b}", "git switch {b}"]),
    (["throw away my local edits to {f}", "restore {f} to the last commit"], "git restore {f}",
     ["git reset --hard", "git rm {f}", "git stash", "git checkout ."]),
    (["unstage {f} but keep the edits", "remove {f} from the index, keep my changes"], "git restore --staged {f}",
     ["git restore {f}", "git rm {f}", "git reset --hard", "git rm --cached -r ."]),
    (["see which commit last changed line 42 of {f}", "who touched line 42 of {f}?"], "git blame -L 42,42 {f}",
     ["git log {f}", "git show HEAD:{f}", "git diff HEAD~1 {f}", "git log -p"]),
    (["rebase my branch onto the latest main", "bring my branch up to date with origin main by rebasing"], "git fetch origin && git rebase origin/main",
     ["git rebase main", "git merge main", "git pull --force", "git reset --hard origin/main"]),
    (["fix the message of my last commit (not pushed yet)", "reword the last unpushed commit"], "git commit --amend -m \"Fix rounding in totals\"",
     ["git revert HEAD", "git reset --hard HEAD~1", "git rebase -i --root", "git commit -m \"Fix rounding in totals\""]),
    (["push the new branch {b} and track it", "publish {b} with upstream set"], "git push -u origin {b}",
     ["git push", "git push --force", "git push origin main", "git push --all"]),
    (["save my uncommitted work for later", "park the current changes"], "git stash push -m \"wip\"",
     ["git reset --hard", "git commit -am wip && git push", "git checkout .", "git clean -fd"]),
    (["show the last 5 commits, one line each"], "git log --oneline -5",
     ["git log -5 -p", "git reflog", "git show HEAD~5", "git shortlog"]),
    (["preview what git clean would delete", "dry-run the clean of untracked files"], "git clean -n",
     ["git clean -fd", "git clean -fdx", "git status --ignored", "git rm -r --cached ."]),
    (["list all containers, including stopped ones"], "docker ps -a",
     ["docker ps", "docker images", "docker container prune", "docker compose ps"]),
    (["follow the logs of container {c}", "stream {c}'s logs"], "docker logs -f {c}",
     ["docker logs {c}", "docker attach {c}", "docker inspect {c}", "docker exec -it {c} sh"]),
    (["remove only dangling images"], "docker image prune",
     ["docker system prune -a --volumes", "docker rmi $(docker images -q)", "docker volume prune", "docker container prune"]),
    (["build the image tagged {p}:dev from this folder"], "docker build -t {p}:dev .",
     ["docker run {p}:dev", "docker pull {p}:dev", "docker compose down -v", "docker tag {p} {p}:dev"]),
    (["install exactly version 2.8.1 of {p}", "pin {p} to 2.8.1"], "pip install {p}==2.8.1",
     ["pip install {p}", "pip install -U {p}", "pip install \"{p}>=2.8.1\"", "pip download {p}"]),
    (["add {p} as a dev dependency", "install {p} for development only"], "npm install -D {p}",
     ["npm install {p}", "npm install -g {p}", "npm ci", "npm update {p}"]),
    (["install exactly what the lockfile says", "clean install from package-lock"], "npm ci",
     ["npm install", "npm update", "npm audit fix", "npm install --force"]),
    (["see why pod {pod} keeps crashing", "get the logs from the crashed container of {pod}"], "kubectl logs {pod} --previous",
     ["kubectl get pods", "kubectl delete pod {pod}", "kubectl describe nodes", "kubectl logs {pod} -f"]),
    (["find files bigger than 100 MB", "which files are over 100MB?"], "find . -type f -size +100M",
     ["du -sh *", "ls -lS", "find . -size 100M", "df -h"]),
    (["search for TODO but skip node_modules"], "grep -rn TODO . --exclude-dir=node_modules",
     ["grep -rn TODO .", "find . -name TODO", "grep TODO *", "cat * | grep TODO"]),
    (["which process is using port 3000?", "port 3000 is taken, by what?"], "lsof -i :3000",
     ["kill -9 3000", "netstat -r", "ps aux", "lsof +D ."]),
    (["type-check the TypeScript without writing output"], "npx tsc --noEmit",
     ["npx tsc", "npm run build", "npx tsc --watch", "npx eslint ."]),
    (["run the Go tests with the race detector"], "go test -race ./...",
     ["go test ./...", "go vet ./...", "go build -race", "go run -race ."]),
    (["lint the Rust code and fail on warnings"], "cargo clippy -- -D warnings",
     ["cargo check", "cargo fmt", "cargo build", "cargo test"]),
    (["create a virtual environment in .venv"], "python3 -m venv .venv",
     ["pip install venv", "python3 -m pip install .venv", "virtualenv /", "source .venv/bin/activate"]),
    (["run just the jest tests in {tf}"], "npx jest {tf}",
     ["npm test", "npx jest --watchAll", "npx jest -u", "npx jest --coverage"]),
    (["update the jest snapshots on purpose after the UI change"], "npx jest -u",
     ["npx jest", "npm test -- --ci", "rm -rf __snapshots__", "npx jest --watch"]),
    (["list the outdated Python packages"], "pip list --outdated",
     ["pip freeze", "pip install -U -r requirements.txt", "pip check", "pip show pip"]),
    (["apply the pending Django migrations"], "python manage.py migrate",
     ["python manage.py makemigrations", "python manage.py flush", "python manage.py sqlmigrate app 0001", "python manage.py migrate app zero"]),
    (["create a migration after changing the Django models"], "python manage.py makemigrations",
     ["python manage.py migrate", "python manage.py flush", "python manage.py showmigrations", "python manage.py squashmigrations app 0001"]),
    (["generate an alembic migration from the model changes"], "alembic revision --autogenerate -m \"add orders index\"",
     ["alembic upgrade head", "alembic downgrade -1", "alembic stamp head", "alembic revision -m \"add orders index\""]),
    (["roll back just the last alembic migration"], "alembic downgrade -1",
     ["alembic downgrade base", "alembic upgrade head", "alembic stamp -1", "alembic revision --autogenerate"]),
    (["show the last 100 lines of app.log and keep following it"], "tail -n 100 -f app.log",
     ["cat app.log", "head -n 100 app.log", "tail -n 100 app.log", "less app.log"]),
    (["replace foo with bar in every .py file under src"], "grep -rl foo src --include=*.py | xargs sed -i 's/foo/bar/g'",
     ["sed -i 's/foo/bar/g' src", "sed 's/foo/bar/g' src/*.py", "grep -rn foo src", "find src -name '*.py' -delete"]),
    (["make deploy.sh executable"], "chmod +x deploy.sh",
     ["chmod -R 777 .", "sudo ./deploy.sh", "chown root deploy.sh", "bash deploy.sh"]),
    (["print only the HTTP status code of {u}"], "curl -s -o /dev/null -w \"%{{http_code}}\" {u}",
     ["curl {u}", "wget {u}", "curl -I {u} -o out.html", "ping {u}"]),
    (["set the git email for this repository only"], "git config user.email \"dev@acme.io\"",
     ["git config --global user.email \"dev@acme.io\"", "git config --system user.email \"dev@acme.io\"", "git commit --author dev", "export GIT_EMAIL=dev@acme.io"]),
    (["kill whatever is listening on port 8080"], "kill $(lsof -t -i:8080)",
     ["killall node", "kill -9 -1", "pkill -f .", "lsof -i :8080"]),
    (["show disk usage of each top-level folder"], "du -sh */",
     ["df -h", "ls -la", "du -a /", "du -sh"]),
]


def cmd_case(rng, i, split):
    n_hold = len(CMD) // 4
    pool = CMD[-n_hold:] if split == "test" and rng.random() < 0.5 else CMD[:-n_hold] if split == "train" else CMD
    goals, right, wrong = rng.choice(pool)
    proj = project(rng)
    e = dict(b=f"feat/{rng.choice(MODS)}-{rng.randint(10, 99)}", f=rng.choice(proj["files"]), t=f"test_{rng.choice(MODS)}_{rng.choice(['empty', 'rounding', 'retry'])}",
             tf=rng.choice(["tests/test_cart.py", "tests/unit/test_orders.py", "src/cart.test.ts", "web/src/checkout.test.tsx"]),
             p=rng.choice(["requests", "zod", "pydantic", "axios", "prettier"]), c=rng.choice(["api", "worker", "db", "redis"]),
             pod=f"{rng.choice(['api', 'payments', 'web'])}-{rng.randint(1000, 9999)}-{rng.choice('abcdefgh')}{rng.choice('xyz')}", u="https://staging.acme.io/health")
    goal = rng.choice(goals).format(**e)
    opts = [right.format(**e)] + [w.format(**e) for w in rng.sample(wrong, rng.randint(3, len(wrong)))]
    opts = list(dict.fromkeys(opts))
    rng.shuffle(opts)
    gold = opts.index(right.format(**e))
    state = {"goal": goal}
    if rng.random() < 0.4:
        state["cwd"] = proj["root"]
    if rng.random() < 0.25:
        state = long_wrap(rng, proj, state, task_text=goal)
    q = rng.choice(["Which command does exactly this?", "Pick the command for the goal.", "Which of these commands should the agent run?"])
    p = [0.93 if j == gold else 0.07 / (len(opts) - 1) for j in range(len(opts))]
    return [row("choose", state, q, opts, p, "command", i, split)]


def step_case(rng, i, split):
    proj = project(rng)
    f = rng.choice(proj["files"])
    other = rng.choice([x for x in proj["files"] if x != f] or [f])
    fn = rng.choice(FUNCS)
    test = proj["test"]
    situ = rng.choice(["fail_unread", "fail_read", "edited", "green", "grep_empty", "unclear", "cmd_missing"])
    done = []
    if situ == "fail_unread":
        last = f"`{test}` failed:\n  File \"{f}\", line {rng.randint(10, 300)}, in {fn}\n    AssertionError: expected 10.00, got 9.99"
        done = [f"Read {other}"]
        gold = f"Read {f}"
    elif situ == "fail_read":
        last = f"`{test}` failed:\n  File \"{f}\", line {rng.randint(10, 300)}, in {fn}\n    TypeError: unsupported operand type(s) for +: 'int' and 'NoneType'"
        done = [f"Read {f}"]
        gold = f"Edit {f}"
    elif situ == "edited":
        last = f"Edited {f}: changed {fn} to handle the empty case"
        done = [f"Read {f}", f"Edit {f}"]
        gold = f"Run `{test}`"
    elif situ == "green":
        last = f"`{test}`: all {rng.randint(20, 400)} tests passed"
        done = [f"Read {f}", f"Edit {f}", f"Run `{test}`"]
        gold = "Finish: summarize the change for the user"
    elif situ == "grep_empty":
        last = f"Grep for `{fn}(` in src/: 0 matches"
        gold = f"Grep for `{fn}` without the parenthesis across the whole repo"
    elif situ == "unclear":
        last = "The ticket says 'make invoices follow the new rules' but no rules are linked anywhere in the repo or the ticket"
        gold = "Ask the user which rules apply"
    else:
        last = f"`{test}` → bash: {test.split()[0]}: command not found"
        gold = f"Ask the user before installing {test.split()[0]}"
    cands = [f"Read {f}", f"Edit {f}", f"Run `{test}`", "Finish: summarize the change for the user", f"Grep for `{fn}` without the parenthesis across the whole repo",
             "Ask the user which rules apply", f"Ask the user before installing {test.split()[0]}", f"Read {other}", f"Write a new file {f}.bak with a copy",
             "git push origin HEAD"]
    opts = [gold] + rng.sample([c for c in cands if c != gold], rng.randint(3, 6))
    rng.shuffle(opts)
    state = {"goal": f"Fix the {rng.choice(MODS)} bug", "done_so_far": done, "last_result": last}
    if rng.random() < 0.3:
        state = long_wrap(rng, proj, state)
    p = [0.92 if o == gold else 0.08 / (len(opts) - 1) for o in opts]
    return [row("choose", state, rng.choice(["What should the agent do next?", "Next step?", "Choose the agent's next action."]), opts, p, "step", i, split)]


GENS = [(tool_case, 0.4), (cmd_case, 0.35), (step_case, 0.25)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=90000)
    ap.add_argument("--split", choices=("train", "test"), default="train")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = random.Random(f"agent-tools:{a.split}:{a.seed}")
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    seen, n, i = set(), 0, 0
    with out.open("w") as fo:
        while n < a.n:
            gen = rng.choices([g for g, _ in GENS], [w for _, w in GENS])[0]
            for r in gen(rng, i, a.split):
                k = (r["state"], r["question"], tuple(r["choices"]))
                if k not in seen:
                    seen.add(k); fo.write(json.dumps(r, ensure_ascii=False) + "\n"); n += 1
            i += 1
    print(f"{n} rows → {out}")


if __name__ == "__main__":
    main()
