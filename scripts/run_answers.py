"""Run the answer step (plain RAG) and the agent on every eval question; save one JSON line each.

Usage: PYTHONUTF8=1 .venv/Scripts/python scripts/run_answers.py [--k 5] [--mode hybrid]
Writes evals/results/<timestamp>-answers-plain.jsonl and ...-answers-agent.jsonl.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

from rocm_agent import config
from rocm_agent.agent import run_agent
from rocm_agent.answer import answer_question


def record(q: dict, ans, error: str | None, elapsed: float) -> dict:
    row = {
        "id": q["id"],
        "question": q["question"],
        "type": q.get("type", "normal"),
        "expected_source": q.get("expected_source"),
        "expected_answer": q.get("expected_answer"),
        "error": error,
    }
    if ans is not None:
        row.update(
            status=ans.status,
            answer=ans.answer,
            citations=[{"url": c.url, "heading_path": c.heading_path} for c in ans.citations],
            retrieved=[r.path for r in ans.retrieved],
            tool_calls=ans.tool_calls,
            rounds=ans.rounds,
            latency_s=ans.latency_s,
            input_tokens=ans.input_tokens,
            output_tokens=ans.output_tokens,
            uncited=ans.uncited,
        )
    else:
        row.update(status="error", latency_s=round(elapsed, 2))
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--mode", default="hybrid")
    args = parser.parse_args()

    settings = config.load()
    path = settings.evals_dir / "questions.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    questions = [json.loads(line) for line in lines if line.strip()]
    out_dir = settings.evals_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    runners = {
        "plain": lambda text: answer_question(text, args.k, args.mode),
        "agent": lambda text: run_agent(text),
    }
    for name, run in runners.items():
        out = out_dir / f"{stamp}-answers-{name}.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for q in questions:
                start = time.perf_counter()
                try:
                    row = record(q, run(q["question"]), None, 0.0)
                except Exception as exc:  # keep going; the error goes in the row
                    error = f"{type(exc).__name__}: {exc}"
                    row = record(q, None, error, time.perf_counter() - start)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                print(f"{name} {q['id']} {row['status']}", flush=True)
        print("wrote", out)


if __name__ == "__main__":
    main()
