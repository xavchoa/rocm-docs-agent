"""Finds the chunks that answer a question: keyword search, vector search, and a fusion of both.

The parts, and why each exists:

- BM25 (keyword search) scores a chunk by how many of the question's words it contains, with
  rare words counting more. It is the part that finds exact strings such as "gfx1151" or
  "RX 7600 XT", which an embedding model may blur into "some AMD GPU". `tokenize` decides what
  counts as a word: it keeps "HSA_OVERRIDE_GFX_VERSION" whole and also by its parts, and
  "WSL2" also as "wsl".
- Vector search turns each chunk and the question into a list of numbers (an embedding) such
  that texts with similar meaning land close together. It finds the right chunk when the
  question uses other words than the docs ("remove ROCm" vs "uninstall"). The embedding model
  runs locally (fastembed); the vectors live in Chroma, a small on-disk vector database.
- Rank fusion (reciprocal rank fusion, RRF) merges the two result lists. BM25 scores and
  vector distances are on different scales, so adding them would let one side swamp the other.
  RRF ignores the scores and uses only the positions: a chunk earns 1 / (k + position) from
  each list it appears in, so a chunk that both lists like beats a chunk only one list likes.

chunks.jsonl (written by `rocm-agent chunk`) is the single source of truth for chunk text.
Chroma holds only vectors and a few fields to show in results. The index remembers a
fingerprint of what it was built from, and is rebuilt if chunks, model or prefix change.
"""

import hashlib
import re
import sys
from collections.abc import Callable
from functools import lru_cache

import chromadb
from chromadb.errors import NotFoundError
from rank_bm25 import BM25Okapi

from rocm_agent import config
from rocm_agent.chunking import load_chunks
from rocm_agent.config import Settings
from rocm_agent.embedding import MAX_TOKENS, Embedder, default_embedder

MODES = ("hybrid", "vector", "bm25")
BATCH = 500  # rows per Chroma insert

STOPWORDS = frozenset(
    "a an and are as at be but by can could did do does for from had has have how i if in into "
    "is it its me my of on or our should so than that the their then there these they this those "
    "to was we were what when where which who why will with would you your".split()
)
ACRONYM_PLURAL = re.compile(r"\b([A-Z]{2,})s\b")  # GPUs -> GPU, APUs -> APU
TOKEN = re.compile(r"[a-z0-9]+(?:[._+-][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lower-case words for BM25: names kept whole and by parts, a plural cut to its singular."""
    tokens = []
    text = ACRONYM_PLURAL.sub(r"\1", text)  # before lower-casing: only acronyms, not "status"
    for whole in TOKEN.findall(text.lower()):
        forms = [whole]
        if re.search(r"[._+-]", whole):  # "therock-dist-windows" also matches its parts
            forms += [p for p in re.split(r"[._+-]", whole) if any(c.isalpha() for c in p)]
        for form in list(forms):  # "wsl2" also matches "wsl"
            m = re.fullmatch(r"([a-z]{3,})\d+", form)
            if m:
                forms.append(m.group(1))
        for form in forms:
            if len(form) > 3 and form.endswith("s") and not form.endswith(("ss", "us", "is")):
                form = form[:-1]  # drivers -> driver
            if len(form) > 1 and form not in STOPWORDS:
                tokens.append(form)
    return tokens


class Bm25Index:
    """Keyword search over the chunks' search_text."""

    def __init__(self, chunks: list[dict]) -> None:
        self.ids = [c["id"] for c in chunks]
        self.bm25 = BM25Okapi([tokenize(c["search_text"]) for c in chunks])

    def search(self, query: str, n: int) -> list[str]:
        """Ids of the best chunks, best first. Chunks with no word in common are left out."""
        words = tokenize(query)
        if not words:
            return []
        scores = self.bm25.get_scores(words)
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))  # ties: corpus order
        return [self.ids[i] for i in order[:n] if scores[i] > 0]


def fingerprint(settings: Settings) -> str:
    """Identifies what an index was built from: the chunks file, the model and the prefix."""
    r = settings.retrieval
    digest = hashlib.sha256()
    digest.update(settings.chunks_path.read_bytes())
    digest.update(r.embedding_model.encode())
    digest.update(r.query_prefix.encode())
    return digest.hexdigest()[:16]


def default_client(settings: Settings):
    return chromadb.PersistentClient(path=str(settings.index_dir / "chroma"))


def build_index(settings: Settings, embedder: Embedder | None = None, client=None) -> dict:
    """Embeds every chunk and stores the vectors in Chroma, replacing any earlier index."""
    chunks = load_chunks(settings.chunks_path)
    embedder = embedder or default_embedder(settings)
    client = client or default_client(settings)
    texts = [c["search_text"] for c in chunks]
    vectors = embedder.embed_docs(texts)

    name = settings.corpus.name
    # No rmtree: Windows keeps the files locked. Drop and recreate the collection instead.
    client.get_or_create_collection(name)
    client.delete_collection(name)
    collection = client.create_collection(
        name,
        embedding_function=None,  # we supply the vectors
        metadata={
            "fingerprint": fingerprint(settings),
            "embedding_model": settings.retrieval.embedding_model,
        },
    )
    for start in range(0, len(chunks), BATCH):
        batch = chunks[start : start + BATCH]
        collection.add(
            ids=[c["id"] for c in batch],
            embeddings=vectors[start : start + BATCH],
            metadatas=[
                {
                    "path": c["path"],
                    "title": c["title"],
                    "url": c["url"],
                    "heading_path": c["heading_path"],
                    "applies": c["applies"],
                    "chunk_index": c["chunk_index"],
                }
                for c in batch
            ],
        )  # no documents stored: chunks.jsonl is the source of truth
    return {
        "chunks": len(chunks),
        "embedding_model": settings.retrieval.embedding_model,
        "max_tokens": MAX_TOKENS,
        "truncated": embedder.count_truncated(texts),
        "saved_to": str(settings.index_dir / "chroma"),
    }


class DenseIndex:
    """Vector search over the Chroma collection, rebuilt first if it is missing or out of date."""

    def __init__(self, settings: Settings, embedder: Embedder | None = None, client=None) -> None:
        self.settings = settings
        self.embedder = embedder or default_embedder(settings)
        self._client = client
        self._collection = None

    @property
    def client(self):
        if self._client is None:  # opened on first use, so a bm25-only run never touches Chroma
            self._client = default_client(self.settings)
        return self._client

    def _is_current(self) -> bool:
        try:
            collection = self.client.get_collection(self.settings.corpus.name)
        except NotFoundError:
            return False
        return (collection.metadata or {}).get("fingerprint") == fingerprint(self.settings)

    def _open(self):
        if self._collection is None:
            if not self._is_current():
                print("Vector index missing or out of date; rebuilding it.", file=sys.stderr)
                build_index(self.settings, self.embedder, self.client)
            self._collection = self.client.get_collection(self.settings.corpus.name)
        return self._collection

    def search(self, query: str, n: int) -> list[str]:
        """Ids of the n nearest chunks, best first."""
        collection = self._open()
        vector = self.embedder.embed_query(query)
        # The vectors have length 1, so Chroma's default distance (squared L2) orders results
        # exactly like cosine similarity does. Nothing here depends on the hnsw space setting.
        found = collection.query(query_embeddings=[vector], n_results=min(n, collection.count()))
        return found["ids"][0]


def rrf(rankings: list[list[str]], k: int) -> list[tuple[str, float]]:
    """Reciprocal rank fusion: (id, score) best first, from lists of ids each best first."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, 1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


class Retriever:
    def __init__(
        self,
        chunks: list[dict],
        dense_search: Callable[[str, int], list[str]],
        candidates: int,
        rrf_k: int,
        default_mode: str,
    ) -> None:
        self.by_id = {c["id"]: c for c in chunks}
        self.bm25 = Bm25Index(chunks)
        self.dense_search = dense_search  # (query, n) -> ids; only called in vector and hybrid
        self.candidates = candidates
        self.rrf_k = rrf_k
        self.default_mode = default_mode

    def search(self, query: str, k: int, mode: str | None = None) -> list[dict]:
        """The top k chunks (copies) with "score" and "ranks" (position in each list used)."""
        mode = mode or self.default_mode
        if mode not in MODES:
            raise ValueError(f"unknown retrieval mode {mode!r}; use one of {MODES}")
        n = self.candidates if mode == "hybrid" else k
        lists: dict[str, list[str]] = {}
        if mode in ("hybrid", "bm25"):
            lists["bm25"] = self.bm25.search(query, n)
        if mode in ("hybrid", "vector"):
            lists["vector"] = self.dense_search(query, n)
        results = []
        for chunk_id, score in rrf(list(lists.values()), self.rrf_k)[:k]:
            ranks = {
                name: ids.index(chunk_id) + 1 if chunk_id in ids else None
                for name, ids in lists.items()
            }
            results.append({**self.by_id[chunk_id], "score": round(score, 5), "ranks": ranks})
        return results


@lru_cache(maxsize=1)
def _default_retriever() -> Retriever:
    settings = config.load()
    r = settings.retrieval
    dense = DenseIndex(settings)  # opens nothing until the first vector search
    chunks = load_chunks(settings.chunks_path)
    return Retriever(chunks, dense.search, r.candidates, r.rrf_k, r.mode)


def retrieve(query: str, k: int = 5, mode: str | None = None) -> list[dict]:
    """The function the eval runner and the agent call."""
    return _default_retriever().search(query, k, mode)


def describe(settings: Settings, mode: str) -> dict:
    """The settings behind a result, saved next to the eval numbers."""
    r = settings.retrieval
    return {
        "mode": mode,
        "embedding_model": r.embedding_model,
        "query_prefix": r.query_prefix,
        "candidates": r.candidates,
        "rrf_k": r.rrf_k,
        "max_chars": settings.chunking.max_chars,
        "max_tokens": settings.chunking.max_tokens,
        "chunks": len(load_chunks(settings.chunks_path)),
        "fingerprint": fingerprint(settings),
    }
