# Answer labelling: rubric and method

How the answers in `evals/results/20261116-212234-answers-{plain,agent}.jsonl` were labelled by
hand (`evals/human_labels.jsonl`) and how the LLM judge (`scripts/judge.py`) was checked against
those labels.

## What is graded

Each of the 25 questions was answered twice, by plain RAG and by the agent: 50 answers. Each
answer gets two independent fields:

- **verdict**: is the answer right? `correct`, `partial` or `wrong`.
- **citation_ok**: can the user check it? `true`, `false` or `null`.

The docs are the ground truth. An answer is judged against what the ROCm docs say, not against
what is true in the real world, and against the user's experience (end to end), not against what
retrieval happened to give the model. When retrieval caused a miss, that goes in the note.

## Verdict

1. **Must-have**: the one thing the user needs to walk away with. Taken from the expected answer
   in `questions.jsonl`, or from the docs when none is written.
2. **Claims**: each main claim is supported (the docs say it, in the same words or plainly in
   meaning), contradicted (the docs say otherwise) or unsupported (not in the docs, even if
   true). A **main claim** is one a user would act on; a side remark is one they would not.
3. **Decide**:
   - `wrong`: a main claim is contradicted, or it answers a different question.
   - `partial`: the must-have is missing, a main claim is unsupported, or a side remark is
     contradicted.
   - `correct`: the must-have is present and nothing is contradicted.

Specific rules, each fixed the first time the case came up:

| Case | Verdict | Example |
| --- | --- | --- |
| An example command assembled from documented pieces | supported | q01 agent |
| Covers some methods correctly, leaves others out, nothing false | `correct` | q04 plain |
| Says the docs lack something they contain, and a user would act on it | `wrong` | q04 agent: "no RHEL/SLES uninstall commands" |
| Says the docs lack something they contain, as a side remark | `partial` | q23 plain: "no vLLM row", while giving the versions |
| Declines, and the docs really do not answer it | `correct` | q14, q15, q21 |
| Declines, saying the docs do not answer it, and they clearly do | `wrong` | q03 plain |
| Declines, says nothing false, but misses a key point the docs give | `partial` | q09 agent, q20 |
| Out-of-scope question answered from general knowledge | `wrong` | none in this run |
| Vague question: useful clarifying questions (a true tip alongside is fine) | `correct` | q02 |
| Normal question answered only with questions that point at the must-have steps | `correct` | q17 |
| Normal question answered only with generic questions | `partial` | none in this run |

## citation_ok

- `true`: every fact about the topic has a citation, and each cited section supports what the
  answer uses it for. An extra citation that supports some part of the answer is fine.
- `false`: a fact about the topic has no citation (even a true one), or a cited section supports
  nothing in the answer.
- `null`: the answer makes no factual claims about the topic (a pure refusal, or only questions).

Remarks about what the excerpts do or do not cover never need a citation; whether they are true
is judged in the verdict. Citation problems never change the verdict, so one flaw is not counted
twice. Example: q02 plain gives a true `rocm-sdk init` tip taken from its first excerpt but does
not cite it: `correct` / `false`.

## Method

1. **Round 1**: all 50 answers labelled by hand with the rules above in their first form (verdict
   tree, refusal rule, citations checked only where present). The first question was worked
   through as a calibration example. Saved as `evals/human_labels_round1.jsonl`.
2. **LLM judge**: `scripts/judge.py` grades the same answers with a fixed rubric prompt. Judge
   model Claude Opus 5.5 (a different model from the one that wrote the answers, Claude Sonnet
   5.5), effort `high`, structured output. It sees the question, the expected answer, the answer,
   and the text of that answer's own cited sections plus the best-matching sections of the
   expected page. It never sees the human labels: they are read only after every call, to report
   agreement.
3. **Review**: each disagreement was looked at against the docs and settled by the labeller.
   Where a case had no rule yet, a rule was written and applied to every answer it covers.
4. **Re-run**: the judge prompt was updated to the final rules and run again.

## Revisions

| Change | Why |
| --- | --- |
| citation_ok: a missing citation on a fact is `false` (was `null`) | Keep "is it right" and "can the user check it" separate |
| Remarks about what the excerpts cover need no citation | They describe the context, not the topic |
| Incomplete but accurate answers are `correct` | The docs do not rank install methods; the judge had assumed one was "the most common" |
| "The docs lack X" when they contain X: main-claim test | Separates harmful false statements (q04 agent) from side remarks (q23 plain) |
| A refusal that says nothing false but misses a key point is `partial` (q20 was `wrong`) | Distinguishes it from a refusal that claims the docs say nothing (q03 plain) |
| Clarifying questions on a normal question can be `correct` | When the questions name the steps to check, the user learns the same thing (q17) |
| Judge sees GPU-lookup data as text | Those citations are compatibility data, not doc pages, and were shown empty (first judge run, `20261117-170220-judge.jsonl`; the fixed run is `20261117-170758-judge.jsonl`) |
| Judge sees only each answer's own sources | Pooled sources held plain RAG to facts only the agent's tool had found (q10 plain) |

Eight labels changed between round 1 and the final file: q02 plain, q04 agent, q07 plain, q17
(both), q20 (both), q23 plain.

## Limits

- One labeller, and the second rater is an LLM, not a second person.
- One run of each mode at the model's default sampling; Claude Sonnet 5.5 does not accept a
  temperature setting. Run-to-run variation is not measured yet.
- The runs record which pages were retrieved, not which chunks, so an answer cannot be graded
  against exactly what the model saw (faithfulness to context). Example: q04 agent cites a
  14-chunk section and most likely saw only the WSL and Ubuntu chunks.
- The saved judge run (`20261117-170758-judge.jsonl`, 40 of 50) used the original rubric and
  still showed each answer the sources of both modes; the per-answer sources and the final
  rules came after it. The re-run with those changes stopped part-way (API credit), so the
  judge has not yet been scored on the final rubric. That score would also be fitted to these
  50 answers, since the final rules were written while reviewing them; the 40 of 50 is the
  blind figure.
