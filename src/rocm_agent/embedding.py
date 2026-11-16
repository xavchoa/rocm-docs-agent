"""The local embedding model (fastembed) and the token counts that go with it.

An embedding model reads at most MAX_TOKENS tokens of a text and silently ignores the rest, so
the chunker needs to know how long a text is in the model's own tokens: `token_counter`.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from rocm_agent.config import Settings

MAX_TOKENS = 512  # input length of the embedding model; longer text is cut off silently


class Embedder(Protocol):
    def embed_docs(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...

    def count_truncated(self, texts: list[str]) -> int: ...


class FastEmbedder:
    """The local embedding model. Loaded on first use, so importing this module stays cheap."""

    def __init__(self, model_name: str, query_prefix: str, cache_dir: Path) -> None:
        self.model_name = model_name
        self.query_prefix = query_prefix
        self.cache_dir = cache_dir
        self._model = None
        self._tokenizer = None

    @property
    def model(self):
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self.model_name, cache_dir=str(self.cache_dir))
        return self._model

    @property
    def tokenizer(self):
        """A copy of the model's tokenizer with the 512-token cut-off and padding switched off."""
        if self._tokenizer is None:
            from tokenizers import Tokenizer

            self._tokenizer = Tokenizer.from_str(self.model.model.tokenizer.to_str())
            self._tokenizer.no_truncation()
            self._tokenizer.no_padding()
        return self._tokenizer

    def embed_docs(self, texts: list[str]) -> list[list[float]]:
        return [vec.tolist() for vec in self.model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        # the prefix goes on the question only, never on the chunks
        return next(iter(self.model.embed([self.query_prefix + text]))).tolist()

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text).ids)

    def count_truncated(self, texts: list[str]) -> int:
        """How many texts are longer than the model's input, i.e. have their tail ignored."""
        return sum(len(enc.ids) > MAX_TOKENS for enc in self.tokenizer.encode_batch(texts))


def default_embedder(settings: Settings) -> FastEmbedder:
    r = settings.retrieval
    return FastEmbedder(r.embedding_model, r.query_prefix, settings.models_dir)


def token_counter(settings: Settings) -> Callable[[str], int]:
    """Text -> number of tokens the embedding model sees (start and end markers included)."""
    return default_embedder(settings).count_tokens
