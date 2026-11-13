"""Eval runner: retrieval metrics today, answer metrics later.

Reads evals/questions.jsonl (one JSON object per line, hand-written labels):
    {"id": "q01", "question": "...", "expected_source": "install/rocm.md",
     "expected_answer": "...", "category": "install"}
Only id, question and expected_source are needed for retrieval metrics. Labels are at
page level (the cleaned file's path) so they survive chunking changes.

Needs a retriever: rocm_agent.retrieval.retrieve(query: str, k: int) -> list of chunks,
where each chunk is a dict with at least "source" (same path format as expected_source).
Change this contract freely; it is the only thing the runner assumes.

Writes evals/results/<timestamp>.json with per-question results and the summary, so every
number in the README traces back to a saved run.
"""

import json
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

Retriever = Callable[[str, int], list[dict]]


def load_questions(path: Path) -> list[dict]:
    rows = []
    for n, line in enumerate(path.read_text("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        missing = {"id", "question", "expected_source"} - row.keys()
        if missing:
            raise ValueError(f"{path.name} line {n}: missing {sorted(missing)}")
        rows.append(row)
    return rows


def _norm(source: str) -> str:
    return source.replace("\\", "/").split("#")[0].removesuffix(".rst").removesuffix(".md")


def score_retrieval(questions: list[dict], retrieve: Retriever, k: int) -> dict:
    per_question, by_cat = [], defaultdict(list)
    for q in questions:
        got = [_norm(c["source"]) for c in retrieve(q["question"], k)]
        want = _norm(q["expected_source"])
        rank = next((i for i, s in enumerate(got, 1) if s == want), None)
        row = {
            "id": q["id"],
            "category": q.get("category", "none"),
            "rank": rank,
            "hit": rank is not None,
            "retrieved": got,
        }
        per_question.append(row)
        by_cat[row["category"]].append(row)

    def summary(rows: list[dict]) -> dict:
        n = len(rows)
        return {
            "n": n,
            f"hit_rate@{k}": round(sum(r["hit"] for r in rows) / n, 3) if n else None,
            "mrr": round(sum(1 / r["rank"] for r in rows if r["rank"]) / n, 3) if n else None,
        }

    return {
        "k": k,
        "overall": summary(per_question),
        "by_category": {c: summary(r) for c, r in sorted(by_cat.items())},
        "questions": per_question,
    }


def run(evals_dir: Path, k: int = 5, retrieve: Retriever | None = None, label: str = "") -> dict:
    if retrieve is None:
        try:
            from rocm_agent.retrieval import retrieve  # hand-written module, see CLAUDE.md
        except ImportError as exc:
            msg = f"No retriever yet ({exc}). Build rocm_agent.retrieval.retrieve."
            raise SystemExit(msg) from exc
    questions = load_questions(evals_dir / "questions.jsonl")
    result = score_retrieval(questions, retrieve, k)
    result["label"] = label
    result["run_at"] = datetime.now(UTC).isoformat(timespec="seconds")

    out_dir = evals_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out = out_dir / f"{stamp}{'-' + label if label else ''}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["saved_to"] = str(out)
    return result


def print_summary(result: dict) -> None:
    k, o = result["k"], result["overall"]
    print(
        f"Retrieval, {o['n']} questions, k={k}: hit_rate@{k}={o[f'hit_rate@{k}']}  MRR={o['mrr']}"
    )
    for cat, s in result["by_category"].items():
        print(f"  {cat:<16} n={s['n']:<3} hit_rate@{k}={s[f'hit_rate@{k}']}  MRR={s['mrr']}")
    misses = [q["id"] for q in result["questions"] if not q["hit"]]
    if misses:
        print(f"  missed: {', '.join(misses)}")
    print(f"Saved: {result['saved_to']}")
