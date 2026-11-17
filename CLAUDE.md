# CLAUDE.md: ROCm Docs Agent

Rules for Claude Code sessions in this repo. Speed on plumbing; design picks are written down
with their reasons.

## Always
- Never invent or estimate results. Every number in README.md comes from `rocm-agent eval` or
  `scripts/run_answers.py` output saved in evals/results.
- Design questions (chunking, retrieval, tools, schema, prompts, metrics): give options with
  trade-offs before changing anything.
- After writing plumbing (scaffold, Dockerfile, CI, fetching and cleaning, API, eval runner,
  plots, README formatting), give one sentence on what it does.
- Commits written with Claude carry a `Co-Authored-By: Claude` trailer.

## Which model does what

Fable decides and checks; cheaper models type and run.

| Work | Model |
| --- | --- |
| Design picks and trade-offs, the written spec for a build step, review of a finished step, reading an eval miss | Fable 5.2 |
| Code from a spec, tests, lint and dependency fixes, running chunk / index / eval / batch runs, README formatting, diagrams, code walkthroughs | Sonnet 5.5 |
| Renames, reformatting, summaries, concept drills | Haiku 5.5 |
| A bug Sonnet has failed on twice | Opus 5.5, then Fable |

In a Fable session:
- Write a brief (files to touch, the contract, what done looks like, what to report) and hand
  the typing to a subagent with `model: "sonnet"` (or `"haiku"`). Review its result and its
  numbers; do not redo its keystrokes.
- Do not read large files, logs or command output when a subagent can return ten lines.
- One subagent per independent piece. Two writers never share a file.
- Never delegate: a design pick, a number that goes in the README (re-check it against the
  saved run), human labels, or anything that leaves the machine (push, publish).

In a Sonnet or Haiku session: if a task turns out to need a design pick, stop, list the options
with their trade-offs, and say it needs Fable.

## Project facts
- Windows 11 dev machine, Python 3.13 via `py`, plain venv + pip. No Docker locally: the Docker
  image is built in CI.
- Corpus settings live in config.toml, so a different document set is a config change.
- Commands: `rocm-agent fetch-docs`, `clean-docs` (data/clean/rocm/*.md with front matter),
  `chunk` (data/index/chunks.jsonl), `index` (Chroma in data/index/chroma), `search "q"`,
  `eval [--mode hybrid|vector|bm25] [--k N] [--label NAME]`, `ask [--agent] "q"`, `serve`.
  `scripts/run_answers.py` runs plain RAG and the agent over the eval set.
- Embeddings are local (fastembed, model in config.toml, cached in data/models); only `ask` and
  the agent need an API key in `.env`. The console code page is cp932: run Python with
  `PYTHONUTF8=1`.
- Docs come from github.com/ROCm/ROCm, `docs/` folder; the exact commit is in
  data/raw/rocm/SOURCE.txt. Remote-content sources are in data/raw/rocm/_remote/SOURCES.txt.
