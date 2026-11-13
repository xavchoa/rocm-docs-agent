# ROCm Docs Agent

An agentic RAG assistant that answers AMD ROCm developer questions from the official ROCm
documentation, calls tools for structured lookups, and returns cited answers. It says
"not in the docs" instead of guessing.

**Status:** in progress (scaffold only). Results will appear here once the eval runs.

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
| `rocm-agent eval` | Runs the eval set and prints every reported number (coming soon) |
| `rocm-agent serve` | Starts the HTTP API on port 8000 |

## Layout

```
config.toml          corpus and paths (swap the corpus here)
src/rocm_agent/      package: config, fetch, cli, api (+ chunking, retrieval, graph, tools to come)
evals/               questions, rubric, labels and results
docs/design.md       design note
LEARNING.md          decisions and lessons log
```
