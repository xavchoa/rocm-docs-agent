# ROCm Docs Agent

A retrieval-augmented assistant for AMD ROCm developer questions, built on the official ROCm
documentation. The goal is an agent that calls tools for structured lookups, returns cited
answers, and says "not in the docs" instead of guessing. The agent and its two tools are
built and tested with a scripted fake model; they have not run against a real model yet.

**Status:** in progress. Working: fetching and cleaning the docs, chunking (700 chunks from
48 pages), indexing (bge-small-en-v1.5 vectors in Chroma), search, hybrid retrieval (BM25 and
vectors merged by reciprocal rank fusion), and a retrieval eval. Three chunks are longer than
the embedding model's 512-token input. The answer step (`rocm-agent ask`, `POST /ask`) is
written but has never run against a real model, so there are no answer-quality or refusal
numbers. The LangGraph agent (`rocm-agent ask --agent`) loops over two tools, `search_docs`
and `lookup_gpu_support` (exact product names against AMD's compatibility data), then
answers through the same cited answer step. Not built: the answer-level eval. The repo is not
on GitHub yet, so CI and the Docker build have never run. 76 tests pass.

## Results so far

Retrieval eval on 21 scored questions, top 5.

| Retriever | Expected page in top 5 | MRR |
| --- | --- | --- |
| BM25 only | 0.762 (16 of 21) | 0.593 |
| Vector only | 0.857 (18 of 21) | 0.704 |
| Hybrid (BM25 + vectors) | 0.952 (20 of 21) | 0.726 |

With 21 questions, one question is 4.8 points. Hybrid at top 10 scores the same as at top 5
(0.952, 0.726). Question q06 is missed by every retriever: its expected page is mostly a list
of links.

Every number here comes from `rocm-agent eval`. Result files are in `evals/results/`:
`20261115-162904-bm25-v2.json`, `20261115-162905-vector-v2.json`,
`20261115-162907-hybrid-v2.json` and `20261115-162909-hybrid-k10-v2.json`.

## How this was built

Built with Claude Code; the rules are in [CLAUDE.md](CLAUDE.md).

- **Mine:** the evaluation set (questions from real user threads, page labels, and the rules
  for what counts as a correct or a refused answer) and the chunking rules: the heading path
  on every chunk, tables split between rows with the header repeated, visible "Applies to"
  labels, and a chunks file between stages.
- **Claude's:** the code, including the chunker, retrieval and the answer step, and these
  design picks: the embedding model, the BM25 tokenizer and rank fusion. I planned to
  hand-write the core and handed it over to ship sooner. Claude's commits carry a
  `Co-Authored-By: Claude` line.

## Setup (Windows, PowerShell)

```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pre-commit install
copy .env.example .env   # only `ask` needs the API key; fetch, chunk, index, search and eval do not
rocm-agent fetch-docs    # downloads the ROCm docs; commit recorded in data/raw/rocm/SOURCE.txt
pytest
```

## Commands

| Command | What it does |
| --- | --- |
| `rocm-agent fetch-docs` | Downloads the docs folder named in `config.toml` |
| `rocm-agent clean-docs` | Converts them to one clean Markdown file per page in `data/clean/` |
| `rocm-agent chunk` | Splits the clean pages into chunks and writes `chunks.jsonl` |
| `rocm-agent index` | Embeds the chunks and builds the vector index |
| `rocm-agent search` | Prints the top chunks for a query |
| `rocm-agent ask [--agent]` | Answers a question with citations (needs the API key); `--agent` lets the model call tools |
| `rocm-agent eval` | Scores retrieval on `evals/questions.jsonl` and saves each run. Options: `--mode`, `--k`, `--label` |
| `rocm-agent serve` | Starts the HTTP API on port 8000 |

## Layout

```
config.toml          corpus and paths (swap the corpus here)
src/rocm_agent/      package: config, fetch, cli, api, chunking, embedding, retrieval, answer
                     (graph and tools to come)
evals/               questions, rubric, labels and results
docs/design.md       design note
```
