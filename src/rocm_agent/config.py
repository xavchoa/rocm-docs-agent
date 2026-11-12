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
class Settings:
    corpus: Corpus
    raw_dir: Path
    index_dir: Path
    evals_dir: Path
    llm_provider: str


def load(path: Path | None = None) -> Settings:
    data = tomllib.loads((path or ROOT / "config.toml").read_text(encoding="utf-8"))
    paths = data["paths"]
    return Settings(
        corpus=Corpus(**data["corpus"]),
        raw_dir=ROOT / paths["raw"],
        index_dir=ROOT / paths["index"],
        evals_dir=ROOT / paths["evals"],
        llm_provider=data["llm"]["provider"],
    )
