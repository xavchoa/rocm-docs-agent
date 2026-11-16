import pytest

from rocm_agent import config
from rocm_agent.tools.gpu_support import load_products, lookup_gpu_support
from rocm_agent.tools.schemas import LookupGpuSupportArgs, SearchDocsArgs
from rocm_agent.tools.search import search_docs

SETTINGS = config.load()
DATA = SETTINGS.raw_dir / SETTINGS.corpus.name / "data"
needs_data = pytest.mark.skipif(
    not (DATA / "gpus.yaml").exists() or not (DATA / "os-support.yaml").exists(),
    reason="ROCm compatibility data not fetched",
)


def lookup(gpu: str, os: str | None = None):
    return lookup_gpu_support(LookupGpuSupportArgs(gpu=gpu, os=os), SETTINGS)


def names(result) -> list[str]:
    return [e["id"].removeprefix("gpu-support:") for e in result.excerpts]


@needs_data
def test_exact_product_is_found_with_its_gfx_target():
    result = lookup("RX 7900 XTX")
    assert names(result) == ["AMD Radeon RX 7900 XTX"]
    assert "listed as officially supported" in result.text
    assert "gfx1100" in result.text
    excerpt = result.excerpts[0]
    assert excerpt["path"] == "compatibility/compatibility-matrix.md"
    assert excerpt["url"].endswith("compatibility/compatibility-matrix.html")
    assert excerpt["heading_path"] == "Compatibility matrix > AMD Radeon RX 7900 XTX"


@needs_data
def test_a_longer_name_is_not_the_listed_product():
    listed = {p.name for p in load_products(DATA)}
    result = lookup("RX 7600 XT")
    assert result.excerpts == []
    assert "not in the official" in result.text
    if "AMD Radeon RX 7600" in listed:
        assert "AMD Radeon RX 7600" in result.text


@needs_data
def test_partial_name_does_not_match_a_pro_card():
    result = lookup("RX 6800")
    assert result.excerpts == []
    assert "not in the official" in result.text
    assert lookup("6800").excerpts == []


@needs_data
def test_gfx_target_returns_every_product_with_it():
    result = lookup("gfx1151")
    assert len(result.excerpts) > 1
    assert all("gfx1151" in e["text"] for e in result.excerpts)


@needs_data
def test_os_filter_keeps_only_that_os():
    result = lookup("RX 7900 XTX", os="windows")
    assert "Windows" in result.text
    assert "Ubuntu" not in result.text
    assert "Ubuntu" not in result.excerpts[0]["text"]


@needs_data
def test_os_filter_with_no_match_says_so():
    result = lookup("RX 7900 XTX", os="solaris")
    assert "solaris is not listed" in result.text


@needs_data
def test_os_filter_matches_abbreviation():
    assert "Red Hat" in lookup("AMD Instinct MI300X", os="rhel").text


def test_search_docs_wraps_retrieve(monkeypatch: pytest.MonkeyPatch):
    chunks = [{"id": "a", "heading_path": "A", "text": "x"}]
    seen = []

    def fake(query, k):
        seen.append((query, k))
        return chunks

    monkeypatch.setattr("rocm_agent.retrieval.retrieve", fake)
    result = search_docs(SearchDocsArgs(query="install"), k=3)
    assert seen == [("install", 3)]
    assert result.text == "1 excerpts found for install."
    assert result.excerpts == chunks

    monkeypatch.setattr("rocm_agent.retrieval.retrieve", lambda q, k: [])
    assert search_docs(SearchDocsArgs(query="x")).text == "No excerpts found."
