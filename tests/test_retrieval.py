import json
import math
import uuid
from dataclasses import replace

import chromadb
import pytest

from rocm_agent import config
from rocm_agent.retrieval import (
    Bm25Index,
    DenseIndex,
    Retriever,
    build_index,
    rrf,
    tokenize,
)


def test_tokenize_splits_a_gpu_name_into_words():
    assert tokenize("RX 7600 XT") == ["rx", "7600", "xt"]


def test_tokenize_keeps_wsl2_and_adds_wsl():
    assert tokenize("WSL2") == ["wsl2", "wsl"]


def test_tokenize_keeps_a_variable_name_whole_and_by_parts():
    assert tokenize("HSA_OVERRIDE_GFX_VERSION") == [
        "hsa_override_gfx_version",
        "hsa",
        "override",
        "gfx",
        "version",
    ]


def test_tokenize_keeps_a_version_as_one_token():
    assert tokenize("6.18.4") == ["6.18.4"]


def test_tokenize_does_not_split_a_model_number_off_a_one_letter_prefix():
    tokens = tokenize("PRO W6800")
    assert tokens == ["pro", "w6800"] and "6800" not in tokens


def test_tokenize_singularises_plurals_and_drops_stopwords():
    assert tokenize("GPUs") == ["gpu"]
    assert tokenize("CPUs and APUs") == ["cpu", "apu"]
    assert tokenize("drivers") == ["driver"]
    assert tokenize("status") == ["status"]  # "us", "is" and "ss" endings are left alone
    assert tokenize("What is the status of the driver?") == ["status", "driver"]


def test_rrf_by_hand():
    out = rrf([["a", "b", "c"], ["b", "d"]], k=60)
    assert [i for i, _ in out] == ["b", "a", "d", "c"]
    scores = dict(out)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["a"] == pytest.approx(1 / 61)
    assert scores["d"] == pytest.approx(1 / 62)
    assert scores["c"] == pytest.approx(1 / 63)


def test_rrf_ties_break_by_id():
    assert [i for i, _ in rrf([["b"], ["a"]], k=60)] == ["a", "b"]


def make_chunk(n: int, text: str) -> dict:
    return {
        "id": f"id{n}",
        "path": f"p{n}.md",
        "chunk_index": 0,
        "title": f"P{n}",
        "url": f"https://example.test/p{n}",
        "source": f"p{n}.md",
        "heading_path": f"P{n}",
        "applies": "",
        "text": text,
        "search_text": f"P{n}\n\n{text}",
    }


CHUNKS = [
    make_chunk(0, "Install ROCm on Ubuntu with the package manager."),
    make_chunk(1, "The gfx1151 target needs the HSA_OVERRIDE_GFX_VERSION variable."),
    make_chunk(2, "Remove everything with the uninstall command."),
    make_chunk(3, "Docker images ship the runtime."),
]


def test_bm25_finds_the_exact_string_and_leaves_out_zero_scores():
    index = Bm25Index(CHUNKS)
    assert index.search("gfx1151", 5) == ["id1"]
    assert index.search("uninstall command", 2) == ["id2"]


def test_bm25_with_no_tokens_returns_nothing():
    assert Bm25Index(CHUNKS).search("what is the", 5) == []


def test_bm25_ties_keep_corpus_order():
    same = [make_chunk(n, "identical words here") for n in range(2)]
    other = [make_chunk(n, f"unrelated filler {n}") for n in range(2, 6)]
    assert Bm25Index(same + other).search("identical", 5) == ["id0", "id1"]


def never_called(query: str, n: int) -> list[str]:
    raise AssertionError("the dense side must not be used")


def make_retriever(dense, mode="hybrid") -> Retriever:
    return Retriever(CHUNKS, dense, candidates=10, rrf_k=60, default_mode=mode)


def test_retriever_hybrid_fuses_both_lists_and_reports_ranks():
    retriever = make_retriever(lambda q, n: ["id2", "id1"])
    out = retriever.search("gfx1151", 2)
    assert [c["id"] for c in out] == ["id1", "id2"]
    assert out[0]["ranks"] == {"bm25": 1, "vector": 2}
    assert out[0]["score"] == round(1 / 61 + 1 / 62, 5)
    assert out[1]["ranks"] == {"bm25": None, "vector": 1}
    assert out[0]["text"] == CHUNKS[1]["text"]


def test_retriever_vector_mode_uses_only_the_dense_list():
    out = make_retriever(lambda q, n: ["id3", "id0"]).search("anything", 5, mode="vector")
    assert [c["id"] for c in out] == ["id3", "id0"]
    assert out[0]["ranks"] == {"vector": 1}
    assert out[0]["score"] == round(1 / 61, 5)


def test_retriever_bm25_mode_never_calls_the_dense_side():
    out = make_retriever(never_called).search("gfx1151", 5, mode="bm25")
    assert [c["id"] for c in out] == ["id1"]
    assert out[0]["ranks"] == {"bm25": 1}


def test_retriever_uses_its_default_mode_and_rejects_unknown_ones():
    retriever = make_retriever(never_called, mode="bm25")
    assert retriever.search("gfx1151", 1)[0]["id"] == "id1"
    with pytest.raises(ValueError):
        retriever.search("gfx1151", 1, mode="fuzzy")


def test_retriever_returns_copies():
    retriever = make_retriever(never_called, mode="bm25")
    retriever.search("gfx1151", 1)[0]["text"] = "changed"
    assert CHUNKS[1]["text"].startswith("The gfx1151")


VOCAB = ["install", "uninstall", "gfx1151", "docker"]


class FakeEmbedder:
    """Unit vectors from word counts: no model, the same input always gives the same vector."""

    def __init__(self) -> None:
        self.doc_calls = 0

    def _vec(self, text: str) -> list[float]:
        words = text.lower().split()
        raw = [float(words.count(w)) for w in VOCAB] + [0.1]
        norm = math.sqrt(sum(x * x for x in raw))
        return [x / norm for x in raw]

    def embed_docs(self, texts: list[str]) -> list[list[float]]:
        self.doc_calls += 1
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)

    def count_truncated(self, texts: list[str]) -> int:
        return 0


def write_chunks(settings, chunks) -> None:
    lines = [json.dumps(c) for c in chunks]
    settings.chunks_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_settings(tmp_path, chunks):
    base = config.load()
    # one collection name per test: EphemeralClient state is shared inside the process
    corpus = replace(base.corpus, name=f"t{uuid.uuid4().hex[:8]}")
    settings = replace(base, index_dir=tmp_path, corpus=corpus)
    write_chunks(settings, chunks)
    return settings


def test_dense_index_builds_and_ranks_by_similarity(tmp_path):
    settings = make_settings(tmp_path, CHUNKS)
    client = chromadb.EphemeralClient()
    info = build_index(settings, FakeEmbedder(), client)
    assert info["chunks"] == 4 and info["truncated"] == 0
    index = DenseIndex(settings, FakeEmbedder(), client)
    assert index.search("uninstall", 2)[0] == "id2"
    assert index.search("gfx1151", 4)[0] == "id1"
    assert len(index.search("docker", 10)) == 4  # n larger than the collection is fine


def test_dense_index_builds_itself_when_missing(tmp_path, capsys):
    settings = make_settings(tmp_path, CHUNKS)
    index = DenseIndex(settings, FakeEmbedder(), chromadb.EphemeralClient())
    assert index.search("docker", 1) == ["id3"]
    assert "rebuilding" in capsys.readouterr().err


def test_changed_chunks_trigger_a_rebuild(tmp_path, capsys):
    settings = make_settings(tmp_path, CHUNKS)
    client = chromadb.EphemeralClient()
    embedder = FakeEmbedder()
    DenseIndex(settings, embedder, client).search("docker", 1)
    assert embedder.doc_calls == 1
    DenseIndex(settings, embedder, client).search("docker", 1)
    assert embedder.doc_calls == 1  # fingerprint matches: no rebuild
    capsys.readouterr()

    extra = make_chunk(4, "A new docker chunk about docker docker.")
    write_chunks(settings, [*CHUNKS, extra])
    assert len(DenseIndex(settings, embedder, client).search("docker", 10)) == 5
    assert embedder.doc_calls == 2
    assert "rebuilding" in capsys.readouterr().err
