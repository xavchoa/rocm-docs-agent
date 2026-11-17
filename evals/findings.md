# Findings and reflection

Notes on the first labelled run: what it shows, what it can't, and what I'd change. The numbers
come from the files in the right-hand column below, and the labelling rules are in
[rubric.md](rubric.md).

## What the run shows

| Area | Result | Source |
| --- | --- | --- |
| Retrieval (hit@5, 21 questions) | hybrid 20, vector only 18, BM25 only 16 | `results/*-v2.json` |
| In-scope answers correct (21) | agent 17, plain RAG 12 | `human_labels.jsonl` |
| Out-of-scope questions declined | 6 of 6 (3 questions, both modes) | `human_labels.jsonl` |
| Vague question met with a useful clarifying question | 2 of 2 (1 question, both modes) | `human_labels.jsonl` |
| Citations pointing at a section that does not support them | 0 | `human_labels.jsonl` |
| Answers with a fact missing its citation | 3 (q02 plain, q17 both) | `human_labels.jsonl` |
| Median time per answer | plain RAG 3.2 s, agent 7.9 s | `results/20261116-212234-*` |
| LLM judge vs first-round hand labels | 40 of 50 verdicts, 36 of 36 citation calls | `results/20261117-170758-judge.jsonl` |

Hybrid retrieval beats either method on its own, which is what I expected. Refusals held up:
when a question was out of scope, both modes said so instead of answering from memory. When an
answer cited something, the citation backed it up. The agent gave the better answer more often
than plain RAG, at a bit over twice the time.

## Why answers failed

Thirteen in-scope answers weren't `correct`:

| Cause | Plain RAG | Agent |
| --- | --- | --- |
| Retrieval did not bring in the section that answers it | 7 (q03, q07, q08, q12, q13, q19, q23) | 1 (q04) |
| Left out the TheRock community-builds pointer for an unlisted GPU | 1 (q09) | 2 (q09, q10) |
| Missed the key point of a question that needs its premise corrected | 1 (q20) | 1 (q20) |

Plain RAG mostly failed on retrieval, and the agent's extra searches covered most of those. The
agent's own misses look different, though this is from reading the prompts and code, not from
testing a fix:

- q04: the uninstall section is split into 14 chunks. The agent seems to have got the WSL and
  Ubuntu ones and stopped searching, which the research prompt allows ("stop once the excerpts
  cover the question"). It then said the docs have no RHEL or SLES uninstall commands. They do,
  in chunks it never saw, and nothing in the answer prompt tells it not to call an uncovered case
  undocumented.
- q09 and q10 probably come from the GPU lookup tool. For an unlisted product it names the
  closest listed ones but never mentions TheRock community builds, whereas plain RAG's doc search
  pulled in the excerpt that does. I haven't verified this yet.

## What the run can't tell me

Whether the agent's lead is real. Question by question it did better on 7 and worse on 2, and a
sign test on that gives p ≈ 0.18. The direction is consistent and the reason makes sense, but 21
questions and one run aren't enough to show it.

Whether hybrid gives better answers, as opposed to better retrieval. Both answer runs used
hybrid, so I haven't run the BM25-only or vector-only versions end to end.

There are also gaps in what was recorded:

- how much answers vary between runs (one run per mode, and Claude Sonnet 5.5 doesn't accept a
  temperature setting);
- faithfulness to context, since the runs log which pages came back but not which chunks, so I
  can't check an answer against exactly what the model saw;
- the agent's full cost, because its token counts leave out the research calls.

## Reflection

I wanted a quick MVP, so in the interest of time I graded each answer with a single verdict plus
a citation check. The first review was a bit shaky and I had to
go through the disagreements with the judge a second time, because one verdict was covering
several different things: an answer can be right but uncited (q02), or the model can be faithful
to what it was given while retrieval missed the part that mattered (q04). I ended up writing the
rules as I went. That works, but it should have been defined better from the start.

If I did it again I'd use more atomic metrics and define them up front: faithfulness,
correctness, relevance, citation accuracy and refusal accuracy as separate scores. I'd try the
rubric on about 10 answers first, with the judge alongside, before labelling the whole set. And
I'd log the chunks each answer actually saw, so faithfulness can be checked properly.

My instinct going in was "garbage in, garbage out": get the right text to the model and the
answers follow. That seems to be true for plain RAG, since most of its misses were retrieval,
but the agent makes me less sure. I'd want to test the fixes below before saying how much of
its misses come from the prompts and the tool.

## Next steps

1. Increase the dataset. With 21 questions nothing is significant, so this comes first.
2. Work on retrieval, then chunking if needed. The chunking rules held up; the problem was
   coverage. First thing to try: when a chunk matches, also return its section or its
   neighbours (parent-section expansion), which targets q04. Pages that are mostly links (q06)
   need their own handling.
3. The cheap agent fixes: a TheRock pointer in the GPU tool's not-found reply (q09, q10), and a
   prompt rule against calling an uncovered case undocumented (q04).
4. Measure instead of assuming: three runs per mode graded by the judge, plus the BM25-only and
   vector-only answer runs.
