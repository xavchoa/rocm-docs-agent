import json

import pytest

from rocm_agent.evals import load_questions, run, score_retrieval

QUESTIONS = [
    {"id": "q1", "question": "install", "expected_source": "install/rocm.md", "category": "a"},
    {"id": "q2", "question": "what", "expected_source": "about/what-is-rocm.md", "category": "a"},
    {"id": "q3", "question": "specs", "expected_source": "reference/gpu-specs.md", "category": "b"},
]
RESULTS = {
    "install": ["install/rocm.md", "about/release-notes.md"],
    "what": ["reference/glossary.md", "components/core.md", "about/what-is-rocm.md"],
    "specs": ["install/rocm.md"],
}


def fake_retrieve(query, k):
    return [{"source": s} for s in RESULTS[query][:k]]


def test_hit_rate_and_mrr_match_hand_calculation():
    r = score_retrieval(QUESTIONS, fake_retrieve, k=5)
    assert r["overall"]["hit_rate@5"] == pytest.approx(0.667, abs=1e-3)
    assert r["overall"]["mrr"] == pytest.approx((1 + 1 / 3 + 0) / 3, abs=1e-3)
    assert r["by_category"]["b"]["hit_rate@5"] == 0


def test_unanswerable_questions_skip_retrieval_scoring():
    qs = QUESTIONS + [{"id": "q4", "question": "cuda", "expected_source": "none"}]
    r = score_retrieval(qs, fake_retrieve, k=5)
    assert r["overall"]["n"] == 3


def test_sources_match_across_extensions_and_anchors():
    q = [{"id": "q", "question": "install", "expected_source": "install\\rocm.rst#ubuntu"}]
    assert score_retrieval(q, fake_retrieve, k=5)["overall"]["mrr"] == 1


def test_list_of_pages_is_a_hit_if_any_is_retrieved():
    q = [
        {
            "id": "q",
            "question": "what",
            "expected_source": ["about/license.md", "components/core.md", "about/what-is-rocm.md"],
        }
    ]
    r = score_retrieval(q, fake_retrieve, k=5)
    assert r["questions"][0]["rank"] == 2  # first listed page found, not the last
    assert r["overall"]["hit_rate@5"] == 1


def test_list_with_no_page_retrieved_is_a_miss():
    q = [{"id": "q", "question": "specs", "expected_source": ["about/license.md", "index.md"]}]
    assert score_retrieval(q, fake_retrieve, k=5)["overall"]["hit_rate@5"] == 0


def test_none_in_a_list_counts_as_unanswerable():
    q = [{"id": "q", "question": "cuda", "expected_source": ["none"]}]
    assert score_retrieval(q, fake_retrieve, k=5)["overall"]["n"] == 0


def test_run_saves_results(tmp_path):
    (tmp_path / "questions.jsonl").write_text(
        "\n".join(json.dumps(q) for q in QUESTIONS), encoding="utf-8"
    )
    r = run(tmp_path, k=2, retrieve=fake_retrieve, label="test")
    saved = json.loads(open(r["saved_to"], encoding="utf-8").read())
    assert saved["overall"]["hit_rate@2"] == pytest.approx(0.333, abs=1e-3)


def test_missing_fields_are_reported(tmp_path):
    p = tmp_path / "questions.jsonl"
    p.write_text('{"id": "q1", "question": "x"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="expected_source"):
        load_questions(p)
