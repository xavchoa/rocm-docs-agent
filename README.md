# ROCm Docs Agent

An agentic RAG assistant that answers AMD ROCm developer questions from the official ROCm
documentation, calls tools for structured lookups, and returns cited answers. It says
"not in the docs" instead of guessing.

**Status:** in progress. Built so far: docs ingestion (fetch and clean) and a retrieval eval
harness (page-level hit rate@k and MRR). In progress: chunking, retrieval and the agent loop.
Results will appear here once the eval runs on a real eval set.

## How this was built

This is a learning project, built with Claude Code under explicit rules in
[CLAUDE.md](CLAUDE.md):

- **I write the core:** chunking, retrieval, the agent graph, tool schemas, and the eval
  questions and labels. Claude reviews these and asks questions, but does not write them.
- **Claude writes the plumbing:** scaffold, docs fetching and cleaning, the eval runner, API,
  CI and README formatting. Those commits carry a `Co-Authored-By: Claude` line.

## Setup (Windows, PowerShell)

```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pre-commit install
copy .env.example .env   # then add your API key
rocm-agent fetch-docs    # downloads the ROCm docs; commit recorded in data/raw/rocm/SOURCE.txt
pytest
```

## Commands

| Command | What it does |
| --- | --- |
| `rocm-agent fetch-docs` | Downloads the docs folder named in `config.toml` |
| `rocm-agent clean-docs` | Converts them to one clean Markdown file per page in `data/clean/` |
| `rocm-agent eval` | Scores retrieval on `evals/questions.jsonl` and saves each run |
| `rocm-agent serve` | Starts the HTTP API on port 8000 |

## Layout

```
config.toml          corpus and paths (swap the corpus here)
src/rocm_agent/      package: config, fetch, cli, api (+ chunking, retrieval, graph, tools to come)
evals/               questions, rubric, labels and results
docs/design.md       design note
```
