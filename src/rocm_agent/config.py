"""Loads config.toml from the project root."""

import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Corpus:
    name: str
    repo: str
    ref: str
    subdir: str
    site_url: str


@dataclass(frozen=True)
class Chunking:
    max_chars: int
    max_tokens: int


@dataclass(frozen=True)
class Retrieval:
    embedding_model: str
    query_prefix: str  # added to the question before embedding; model-specific
    mode: str  # "hybrid" | "vector" | "bm25"
    candidates: int
    rrf_k: int
    top_k: int


@dataclass(frozen=True)
class Settings:
    corpus: Corpus
    raw_dir: Path
    clean_dir: Path
    index_dir: Path
    models_dir: Path
    evals_dir: Path
    chunking: Chunking
    retrieval: Retrieval
    llm_provider: str
    llm_model: str

    @property
    def chunks_path(self) -> Path:
        return self.index_dir / "chunks.jsonl"


def load(path: Path | None = None) -> Settings:
    data = tomllib.loads((path or ROOT / "config.toml").read_text(encoding="utf-8"))
    paths = data["paths"]
    return Settings(
        corpus=Corpus(**data["corpus"]),
        raw_dir=ROOT / paths["raw"],
        clean_dir=ROOT / paths["clean"],
        index_dir=ROOT / paths["index"],
        models_dir=ROOT / paths["models"],
        evals_dir=ROOT / paths["evals"],
        chunking=Chunking(**data["chunking"]),
        retrieval=Retrieval(**data["retrieval"]),
        llm_provider=data["llm"]["provider"],
        llm_model=data["llm"]["model"],
    )
