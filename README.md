# ROCm Docs Agent

A retrieval-augmented assistant for AMD ROCm developer questions, built on the official ROCm
documentation. It returns cited answers, calls tools for structured lookups, and says "not in
the docs" instead of guessing.

**Status:** in progress. Working end to end:
- fetching and cleaning the docs;
- chunking (700 chunks from 48 pages);
- hybrid retrieval (BM25 and bge-small-en-v1.5 vectors, merged by reciprocal rank fusion);
- a cited answer step (`rocm-agent ask`, `POST /ask`);
- a LangGraph agent (`rocm-agent ask --agent`) with two tools: `search_docs`, and
  `lookup_gpu_support` (exact product names against AMD's compatibility data).

Both answer modes have run against a real model (Claude Sonnet 5.5) on the full eval set.
Answer correctness is not scored yet: the 25 answers still need hand labels. 76 tests pass.
The repo has not run CI on GitHub yet.

## Results so far

### Retrieval

21 scored questions, top 5.

| Retriever | Expected page in top 5 | MRR |
| --- | --- | --- |
| BM25 only | 0.762 (16 of 21) | 0.593 |
| Vector only | 0.857 (18 of 21) | 0.704 |
| Hybrid (BM25 + vectors) | 0.952 (20 of 21) | 0.726 |

With 21 questions, one question is 4.8 points. Hybrid at top 10 scores the same as at top 5
(0.952, 0.726). Question q06 is missed by every retriever: its expected page is mostly a list
of links.

From `rocm-agent eval`. Files: `evals/results/20261115-162904-bm25-v2.json`,
`20261115-162905-vector-v2.json`, `20261115-162907-hybrid-v2.json` and
`20261115-162909-hybrid-k10-v2.json`.

### Answers (plain RAG vs agent)

All 25 questions, one run each, from `scripts/run_answers.py`.

| | Plain RAG | Agent |
| --- | --- | --- |
| Errors | 0 of 25 | 0 of 25 |
| Out-of-scope questions answered "not in the docs" | 3 of 3 | 3 of 3 |
| Vague question answered with a clarifying question | 1 of 1 | 1 of 1 |
| In-scope questions answered | 18 of 21 | 18 of 21 |
| Answers without a citation | 0 | 0 |
| Median time per question | 3.2 s | 7.9 s |

The agent made 68 tool calls in total and used 1 to 4 research rounds (most often 3).

In-scope questions that were not answered:
- Plain RAG: q03 and q20 ("not in the docs"), q17 (asked for clarification).
- Agent: q09 and q20 ("not in the docs"), q17 (asked for clarification).

Whether the 18 answers are right is not measured yet. That needs hand labels against the
expected answers in `evals/questions.jsonl`.

Files: `evals/results/20261116-212234-answers-plain.jsonl` and
`20261116-212234-answers-agent.jsonl`.

## What was tried, changed or dropped

Each entry gives what happened, why, and the evidence.

- **Docling for chunking: tried, added, then replaced.**
  - A spike on three real pages cut table rows mid-cell, garbled the repeated table header, and
    cut two code blocks.
  - It also dropped every selector condition, because they were HTML comments at the time.
  - Once the conditions were visible text, 17% of the command chunks on the install page still
    lost their "Applies to" line. The line appears once under a heading, and Docling did not
    repeat it on later chunks.
  - It went in anyway (commit "add Docling for chunking"). It was then replaced before the next
    commit by a chunker written for these docs: split at headings, never cut a block, split
    tables by row with the header repeated, and restate the "Applies to" line on each piece.
  - Docling and its CPU torch install were removed from the dependencies, CI and the Dockerfile.
- **Selector conditions: from hidden comments to visible labels.** Install pages show different
  commands per OS and install method. The cleaner first marked these with HTML comments, which
  chunkers and embeddings ignore. It now writes a visible "**Applies to: ...**" line, the same
  approach AMD's own LLM-oriented docs export uses.
- **Tables: dropped, then rebuilt.**
  - Tables with bullet lists in their cells came out as "[TABLE]". A pandoc filter now flattens
    the list cells.
  - `.. matrix::` tables printed one cell per line. They are now rebuilt as real tables, with
    merged cells copied down.
- **No overlap for tables.** A paragraph that has to be cut repeats its last sentence in the
  next piece; table pieces repeat nothing but the header. Each row is complete once the header
  and heading path are repeated, so overlap only duplicated rows.
- **Chunk size: from characters to tokens.**
  - A 1,200-character limit left 68 of 661 chunks longer than the embedding model's 512-token
    input, so vector search never saw their tails. Commands and tables carry more tokens per
    character than prose.
  - Size is now measured in tokens too (budget 440): 700 chunks, 3 over 512.
  - Retrieval scores did not move on this set, so the fix removed a silent truncation without
    a measurable gain.
- **BM25 tokenizer: a plural bug.** "GPUs" and "GPU" were different words. After the fix, MRR
  went down slightly (hybrid 0.730 to 0.726, BM25 0.619 to 0.595), and hits did not change. The
  fix was kept: it is a correctness fix, not tuning, and one question is 4.8 points.
  - Before: `20261115-155757-bm25.json`, `20261115-155801-hybrid.json`.
  - After: `20261115-160231-bm25-fix1.json`, `20261115-160236-hybrid-fix1.json`.
- **Eval set: a refusal question dropped, then restored.** The question about
  `HSA_OVERRIDE_GFX_VERSION` was dropped, then restored with a second one like it. Without them
  the set had no true "not in the docs" question, so the claim that the assistant refuses
  instead of guessing would have gone untested.
- **q06 left as a miss.** Its page is mostly links. BM25 ranks it 9th, and vector search does
  not have it in the top 30. Tuning for one question was not worth it.

## How this was built

Built with Claude Code; the rules are in [CLAUDE.md](CLAUDE.md).

- **Mine:** the evaluation set (questions from real user threads, page labels, and the rules
  for what counts as a correct or a refused answer) and the chunking rules: the heading path
  on every chunk, tables split between rows with the header repeated, visible "Applies to"
  labels, and a chunks file between stages.
- **Claude's:** the code, including the chunker, retrieval, the answer step and the agent, and
  these design picks: the embedding model, the BM25 tokenizer and rank fusion. I planned to
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
| `rocm-agent clean-docs` | Converts the docs to one clean Markdown file per page in `data/clean/` |
| `rocm-agent chunk` | Splits the clean pages into chunks and writes `chunks.jsonl` |
| `rocm-agent index` | Embeds the chunks and builds the vector index |
| `rocm-agent search` | Prints the top chunks for a query |
| `rocm-agent ask [--agent]` | Answers a question with citations (needs the API key); `--agent` lets the model call tools |
| `rocm-agent eval` | Scores retrieval on `evals/questions.jsonl` and saves each run. Options: `--mode`, `--k`, `--label` |
| `rocm-agent serve` | Starts the HTTP API on port 8000 |
| `python scripts/run_answers.py` | Runs plain RAG and the agent over the eval set and saves both |

## Layout

```
config.toml          corpus, chunking, retrieval and model settings (swap the corpus here)
src/rocm_agent/      package: config, fetch, clean, chunking, embedding, retrieval, answer,
                     graph and agent (the LangGraph loop), tools/, evals, cli, api
scripts/             run_answers.py (answer runs over the eval set)
evals/               questions.jsonl, candidates.md (where the questions came from), results/
tests/               unit tests; the agent is tested with a scripted fake model
```
