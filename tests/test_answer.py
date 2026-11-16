from typing import Any

import pytest
from fastapi.testclient import TestClient

from rocm_agent import api, config
from rocm_agent.answer import UNSUPPORTED, Answer, Draft, answer_question, format_answer

SETTINGS = config.load()


def chunk(n: int, applies: str = "") -> dict[str, Any]:
    return {
        "id": f"c{n}",
        "path": f"page{n}.md",
        "title": f"Page {n}",
        "url": f"https://rocm.docs.amd.com/page{n}.html",
        "heading_path": f"Page {n} > Section",
        "applies": applies,
        "text": f"{applies}\n\nBody of chunk {n}.".strip(),
        "rocm_version": "7.0",
    }


class FakeLlm:
    def __init__(self, draft: Draft) -> None:
        self.draft = draft
        self.calls: list[tuple[str, str]] = []

    def __call__(self, system: str, user: str) -> tuple[Draft, dict[str, int]]:
        self.calls.append((system, user))
        return self.draft, {"input_tokens": 100, "output_tokens": 20}


def ask(draft: Draft, chunks: list[dict[str, Any]]) -> tuple[Answer, FakeLlm]:
    llm = FakeLlm(draft)
    result = answer_question(
        "How do I install?",
        retrieve=lambda query, k, mode=None: chunks,
        llm=llm,
        settings=SETTINGS,
    )
    return result, llm


def test_context_numbers_excerpts_and_keeps_applies_lines():
    applies = "**Applies to: install_method=pkgman; os=ubuntu,debian**"
    _, llm = ask(Draft(status="answered", answer="x", sources=[1]), [chunk(1, applies), chunk(2)])
    system, user = llm.calls[0]
    assert "ROCm 7.0 docs" in system
    assert f"[1] Page 1 > Section\nURL: https://rocm.docs.amd.com/page1.html\n\n{applies}" in user
    assert "[2] Page 2 > Section\nURL: https://rocm.docs.amd.com/page2.html\n\n" in user
    assert user.index("[1]") < user.index("[2]") < user.index("Question: How do I install?")
    assert "\n----" in user


def test_source_numbers_map_to_citations():
    result, _ = ask(
        Draft(status="answered", answer="Use apt.", sources=[2, 1]), [chunk(1), chunk(2)]
    )
    assert result.status == "answered"
    assert [c.n for c in result.citations] == [2, 1]
    assert result.citations[0].url == "https://rocm.docs.amd.com/page2.html"
    assert result.citations[0].title == "Page 2"
    assert result.citations[0].heading_path == "Page 2 > Section"
    assert [r.n for r in result.retrieved] == [1, 2]
    assert (result.input_tokens, result.output_tokens) == (100, 20)
    assert result.uncited is False


def test_out_of_range_and_repeated_numbers_are_dropped():
    draft = Draft(status="answered", answer="ok", sources=[0, 2, 2, 9, -1, 1])
    result, _ = ask(draft, [chunk(1), chunk(2)])
    assert [c.n for c in result.citations] == [2, 1]
    assert result.uncited is False


def test_answered_without_valid_source_becomes_not_in_docs():
    result, _ = ask(Draft(status="answered", answer="From memory.", sources=[7]), [chunk(1)])
    assert result.status == "not_in_docs"
    assert result.answer == UNSUPPORTED
    assert result.citations == []
    assert result.uncited is True


def test_no_chunks_skips_the_model():
    result, llm = ask(Draft(status="answered", answer="x", sources=[1]), [])
    assert llm.calls == []
    assert result.status == "not_in_docs"
    assert result.answer == UNSUPPORTED
    assert result.retrieved == []
    assert result.input_tokens is None
    assert result.uncited is False


def test_needs_clarification_passes_through():
    draft = Draft(status="needs_clarification", answer="Which OS?", sources=[1])
    result, _ = ask(draft, [chunk(1)])
    assert result.status == "needs_clarification"
    assert result.answer == "Which OS?"
    assert result.uncited is False


def test_format_answer_shows_status_only_when_not_answered():
    answered, _ = ask(Draft(status="answered", answer="Use apt.", sources=[1]), [chunk(1)])
    assert format_answer(answered).splitlines() == [
        "Use apt.",
        "[1] Page 1 > Section (https://rocm.docs.amd.com/page1.html)",
    ]
    vague, _ = ask(Draft(status="needs_clarification", answer="Which OS?", sources=[]), [chunk(1)])
    assert format_answer(vague).splitlines() == ["Status: needs_clarification", "Which OS?"]


def test_post_ask(monkeypatch: pytest.MonkeyPatch):
    seen: list[tuple[str, int | None]] = []

    def fake(question: str, k: int | None = None) -> Answer:
        seen.append((question, k))
        return Answer(
            question=question,
            status="answered",
            answer="Use apt.",
            citations=[],
            retrieved=[],
            model="fake",
            latency_s=0.1,
            input_tokens=None,
            output_tokens=None,
            uncited=False,
        )

    monkeypatch.setattr(api, "answer_question", fake)
    response = TestClient(api.app).post("/ask", json={"question": "Install?", "k": 3})
    assert response.status_code == 200
    assert response.json()["answer"] == "Use apt."
    assert seen == [("Install?", 3)]


@pytest.mark.parametrize("question", ["", "   "])
def test_blank_question_is_422(question: str):
    response = TestClient(api.app).post("/ask", json={"question": question})
    assert response.status_code == 422
