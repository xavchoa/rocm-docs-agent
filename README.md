# ROCm Docs Agent

[![ci](https://github.com/xavchoa/rocm-docs-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/xavchoa/rocm-docs-agent/actions/workflows/ci.yml)

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

Both answer modes have run against a real model (Claude Sonnet 5.5) on the full eval set, and
all 50 answers are hand-labelled for correctness and citations. An LLM judge
(`scripts/judge.py`) was checked against those labels. CI runs lint, the 76 tests and a Docker
build on every push.

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

Files: `evals/results/20261116-212234-answers-plain.jsonl` and
`20261116-212234-answers-agent.jsonl`.

### Answer correctness (hand labels)

Every answer was labelled by hand against the docs: a verdict (`correct`, `partial`, `wrong`)
and whether its citations back up its facts. The rules, with an example for each, are in
[evals/rubric.md](evals/rubric.md); the labels are in `evals/human_labels.jsonl`.

| | Plain RAG | Agent |
| --- | --- | --- |
| In-scope questions (21): correct | 12 | 17 |
| partial | 6 | 3 |
| wrong | 3 | 1 |
| Out-of-scope questions declined correctly | 3 of 3 | 3 of 3 |
| Vague question met with a useful clarifying question | 1 of 1 | 1 of 1 |
| Answers with a fact missing its citation | 2 | 1 |
| Answers citing a section that does not support them | 0 | 0 |

The agent did better than plain RAG on seven questions (q03, q07, q08, q12, q13, q19, q23),
mostly where its extra searches found a page or section plain RAG's single search missed. It
did worse on two: q10 (left out the community-builds link) and q04, its one `wrong` answer,
which says the docs give no RHEL or SLES uninstall commands; they do, in chunks of a 14-chunk
section that retrieval did not return. One run of each mode, so a
question is 4.8 points and the gap is indicative, not measured variance.

### LLM judge

`scripts/judge.py` grades the same answers with a fixed rubric prompt (Claude Opus 5.5, a
different model from the one that wrote the answers). It never sees the hand labels. On the
first-round labels it agreed on 40 of 50 verdicts and on all 36 citation calls where either
side gave one. It was never more lenient than the hand labels: 7 of its 10 disagreements were
`correct` → `partial`. Reviewing those 10 led to 8 changed labels and the rule changes listed
in the rubric. File: `evals/results/20261117-170758-judge.jsonl`.

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

Built with Claude Code; the rules are in [CLAUDE.md](CLAUDE.md). Claude's commits carry a
`Co-Authored-By: Claude` line.

- **Mine:**
  - **The evaluation:** the 25 questions from real user threads, their expected pages and
    answers, the 50 hand labels, and the labelling rules in [evals/rubric.md](evals/rubric.md)
    (decided by me while reviewing the judge's disagreements; written up with Claude).
  - **The chunking rules:** the heading path on every chunk, tables split between rows with
    the header repeated, visible "Applies to" labels, and a chunks file between stages.
  - **Reviewing the results:** tracing failures to their causes (for example, q04's retrieval
    miss).
- **Claude's:**
  - **The code:** the cleaner, chunker, retrieval, the answer step, the LangGraph agent and its
    tools, the API, the eval and judge scripts.
  - **These design picks:** the embedding model, the BM25 tokenizer, rank fusion, and the
    judge's model and prompt.

I planned to hand-write the core and handed it over to ship sooner.

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
| `python scripts/judge.py` | Grades a saved answer run with the LLM judge and reports agreement with the hand labels (needs the API key) |

## Layout

```
config.toml          corpus, chunking, retrieval and model settings (swap the corpus here)
src/rocm_agent/      package: config, fetch, clean, chunking, embedding, retrieval, answer,
                     graph and agent (the LangGraph loop), tools/, evals, cli, api
scripts/             run_answers.py (answer runs), judge.py (LLM judge)
evals/               questions.jsonl, candidates.md (where the questions came from),
                     rubric.md, human_labels.jsonl (+ round 1), results/
tests/               unit tests; the agent is tested with a scripted fake model
```
