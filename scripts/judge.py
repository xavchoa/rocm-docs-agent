"""LLM judge for the answer runs: grades each saved answer against the docs with a fixed rubric.

Usage: python scripts/judge.py [--run 20261116-212234] [--model claude-opus-5-5]

The judge sees the question, the expected answer and pages, the answer, and the text of the
sections it cited plus the best-matching sections of the expected pages. It never sees human
labels: those are read only after every call is done, to report agreement.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from rocm_agent import config
from rocm_agent.retrieval import retrieve
from rocm_agent.tools.gpu_support import _facts, load_products

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "evals" / "results"

RUBRIC = """You grade answers from a documentation assistant for AMD ROCm. The docs are the
ground truth: judge whether the answer says what the SOURCE TEXT says, not whether it is true in
the real world.

Steps:
1. Decide the must-have: the one thing a good answer must contain to solve the question (use the
   expected answer if given, otherwise the source text).
2. List the answer's main claims and mark each: supported (the sources say it, in the same words
   or plainly in meaning), contradicted (the sources say otherwise), or unsupported (not in any
   source, even if true). An example command assembled from documented pieces counts as
   supported if nothing contradicts it. A main claim is one a user would act on; a side remark
   is one they would not. Uncited side details (a version list, a tip) are handled by
   citation_ok, not by the verdict.
3. Verdict:
   - wrong: a main claim contradicted, or it answers a different question
   - partial: must-have missing, a main claim unsupported, or a side remark contradicted
   - correct: must-have present and nothing contradicted
   Leaving out some methods or cases is fine: an answer that covers some correctly and omits
   others, with nothing false, is correct. Do not use outside knowledge to decide which method
   matters most; the docs do not rank them.
4. Saying the docs lack something they contain is a contradicted claim. It is a main claim
   (wrong) if a user would act on it, for example stop looking for steps that exist; a side
   remark (partial) if not, for example when the answer still gives the needed facts or points
   to the right page.
5. citation_ok (judged separately from the verdict): true if every factual claim about the topic
   has a citation and each cited section supports what the answer uses it for (an extra citation
   is fine if it supports any part of the answer); false if a cited section supports nothing in
   the answer, or a factual claim about the topic has no citation (even a true one); null when
   the answer makes no factual claims about the topic. Remarks about what the excerpts do or do
   not cover never need a citation; whether they are true is judged in the verdict.

Special cases:
- The answer declines ("the docs do not cover it"): correct if the sources really do not answer
  the question; wrong if it says the docs do not answer the question and they clearly do;
  partial if what it says is true but it misses a key point the sources give.
- Question type "refusal" (out of scope): correct if the answer declines or says the docs do not
  cover it; wrong if it answers from general knowledge, even correctly.
- Question type "vague": correct if it asks a useful clarifying question (error message, OS,
  install method...); partial if the question it asks is too generic to help; wrong if it guesses
  a fix instead of asking. A true tip added next to the questions does not lower the verdict.
- A normal question answered only with clarifying questions: correct if the questions point at
  the must-have steps (the user learns what to check); partial if they are generic.
"""

NO_EXPECTED = "(none written: use the sources)"

SCHEMA = {
    "type": "object",
    "properties": {
        "must_have": {"type": "string"},
        "reason": {"type": "string"},
        "verdict": {"type": "string", "enum": ["correct", "partial", "wrong"]},
        "citation_ok": {"type": ["boolean", "null"]},
    },
    "required": ["must_have", "reason", "verdict", "citation_ok"],
    "additionalProperties": False,
}


_PRODUCTS: dict | None = None


def gpu_facts(heading_path: str) -> str:
    """Text of a lookup_gpu_support citation (compatibility data, not a docs chunk)."""
    global _PRODUCTS
    prefix = "Compatibility matrix > "
    if not heading_path.startswith(prefix):
        return ""
    if _PRODUCTS is None:
        settings = config.load()
        data = settings.raw_dir / settings.corpus.name / "data"
        _PRODUCTS = {p.name: p for p in load_products(data)}
    product = _PRODUCTS.get(heading_path[len(prefix) :])
    return "\n".join(_facts(product, None)) if product else ""


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def build_sources(question: dict, answers: dict, chunks: dict) -> tuple[list[dict], dict]:
    """Cited sections (S1, S2, ...) for both modes, then up to 2 expected-page sections."""
    keys: list[tuple[str, str]] = []
    tags: dict[str, list[str]] = {}
    for mode, row in answers.items():
        tags[mode] = []
        for cite in row.get("citations") or []:
            key = (cite["url"], cite["heading_path"])
            if key not in keys:
                keys.append(key)
            tags[mode].append(f"S{keys.index(key) + 1}")
    expected = question.get("expected_source")
    expected = [expected] if isinstance(expected, str) else (expected or [])
    expected = [e for e in expected if e != "none"]
    extra: list[tuple[str, str]] = []
    if expected:
        for hit in retrieve(question["question"], k=30):
            key = (hit["url"], hit["heading_path"])
            if hit.get("path") in expected and key not in keys and key not in extra:
                extra.append(key)
            if len(extra) >= 2:
                break
    sources = []
    for i, key in enumerate(keys + extra):
        pieces = chunks.get(key, [])
        text = "\n\n".join(p["text"] for p in pieces) or gpu_facts(key[1])
        sources.append(
            {
                "tag": f"S{i + 1}" if i < len(keys) else "expected page",
                "title": (pieces[0]["path"] + " > " if pieces else "") + key[1],
                "text": text,
            }
        )
    return sources, tags


def prompt_for(question: dict, row: dict, cited: list[str], sources: list[dict]) -> str:
    src = (
        "\n\n".join(
            f'<source tag="{s["tag"]}" title="{s["title"]}">\n{s["text"]}\n</source>'
            for s in sources
        )
        or "(no sources)"
    )
    return (
        f"Question type: {question.get('type', 'normal')}\n"
        f"Question: {question['question']}\n"
        f"Expected page(s): {question.get('expected_source')}\n"
        f"Expected answer: {question.get('expected_answer') or NO_EXPECTED}\n\n"
        f'<answer cites="{", ".join(cited) or "nothing"}">\n'
        f"{row.get('answer') or ''}\n</answer>\n\n"
        f"SOURCE TEXT:\n{src}"
    )


def judge(client: anthropic.Anthropic, model: str, text: str) -> dict:
    response = client.messages.create(
        model=model,
        max_tokens=16000,
        system=RUBRIC,
        output_config={"effort": "high", "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": text}],
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("judge refused")
    body = next(b.text for b in response.content if b.type == "text")
    out = json.loads(body)
    out["input_tokens"] = response.usage.input_tokens
    out["output_tokens"] = response.usage.output_tokens
    return out


def agreement(judged: list[dict], labels_path: Path, exclude: set[str]) -> None:
    if not labels_path.exists():
        print("no human labels yet: skipping agreement")
        return
    human = {(x["id"], x["mode"]): x for x in load_jsonl(labels_path)}
    pairs = [(j, human[(j["id"], j["mode"])]) for j in judged if (j["id"], j["mode"]) in human]
    for name, subset in (
        ("all", pairs),
        ("blind only", [p for p in pairs if p[0]["id"] not in exclude]),
    ):
        same = sum(j["verdict"] == h["verdict"] for j, h in subset)
        cit = [
            (j, h)
            for j, h in subset
            if h["citation_ok"] is not None or j["citation_ok"] is not None
        ]
        cit_same = sum(j["citation_ok"] == h["citation_ok"] for j, h in cit)
        print(
            f"{name}: verdict agreement {same}/{len(subset)}, "
            f"citation_ok agreement {cit_same}/{len(cit)}"
        )
    print(
        "confusion (human -> judge):", dict(Counter((h["verdict"], j["verdict"]) for j, h in pairs))
    )
    print("disagreements:")
    for j, h in pairs:
        if j["verdict"] != h["verdict"] or j["citation_ok"] != h["citation_ok"]:
            print(
                f"  {j['id']} {j['mode']}: human {h['verdict']}/{h['citation_ok']}, "
                f"judge {j['verdict']}/{j['citation_ok']} - {j['reason']}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="20261116-212234")
    parser.add_argument("--model", default="claude-opus-5-5")
    parser.add_argument(
        "--exclude", default="", help="comma-separated ids to report separately as 'blind only'"
    )
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")

    questions = load_jsonl(ROOT / "evals" / "questions.jsonl")
    runs = {
        m: {r["id"]: r for r in load_jsonl(RESULTS / f"{args.run}-answers-{m}.jsonl")}
        for m in ("plain", "agent")
    }
    chunks: dict = {}
    for c in load_jsonl(ROOT / "data" / "index" / "chunks.jsonl"):
        chunks.setdefault((c["url"], c["heading_path"]), []).append(c)

    client = anthropic.Anthropic()
    out_path = RESULTS / f"{datetime.now():%Y%m%d-%H%M%S}-judge.jsonl"
    judged, times = [], []
    with out_path.open("w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {"meta": {"run": args.run, "model": args.model, "effort": "high", "rubric": RUBRIC}}
            )
            + "\n"
        )
        for q in questions:
            answers = {m: runs[m][q["id"]] for m in ("plain", "agent")}
            for mode in ("plain", "agent"):
                # each answer is judged only with its own citations plus the expected page
                sources, tags = build_sources(q, {mode: answers[mode]}, chunks)
                start = time.perf_counter()
                row = {"id": q["id"], "mode": mode}
                try:
                    row |= judge(
                        client, args.model, prompt_for(q, answers[mode], tags[mode], sources)
                    )
                except Exception as e:  # keep going; a failed call is recorded, not hidden
                    row |= {"error": repr(e), "verdict": None, "citation_ok": None, "reason": ""}
                times.append(time.perf_counter() - start)
                judged.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"{q['id']} {mode}: {row['verdict']}", flush=True)
    errors = sum(1 for r in judged if r.get("error"))
    print(
        f"saved {out_path.name}; errors {errors}; median {statistics.median(times):.1f}s per call"
    )
    agreement(
        [r for r in judged if not r.get("error")],
        ROOT / "evals" / "human_labels.jsonl",
        set(args.exclude.split(",")),
    )


if __name__ == "__main__":
    main()
