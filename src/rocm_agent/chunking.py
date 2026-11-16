"""Splits the cleaned Markdown pages into retrieval chunks: data/index/chunks.jsonl.

Rules (docs/design.md section 1 and the 15 Nov entries in the learning log):

- A chunk never crosses a heading. Its searched text starts with the heading path, so a chunk
  still says where it sits: "Install AMD ROCm 10.0.0 > Prerequisites > Install Python".
- Blocks stay whole. A paragraph, a list, a code block or a table row is never cut to fit a
  chunk. Only one block that is larger than the limit on its own is split, and then at its own
  boundaries: table rows (header row repeated, no overlap), code lines, sentences.
- "**Applies to: ...**" labels say which OS or install method a step is for. They stay in the
  text, and when a section is split the label in force is printed again at the top of the next
  piece, so a command is never separated from its condition.
- A lead-in line ("Run the following command:") stays with the block it introduces.
- Size is measured so that a chunk fits the embedding model's input. Prose is limited by
  characters. Commands, paths and tables hold more tokens per character than prose does, so
  they are counted by the model's own tokens instead (see `effective_measure`).

One JSON object per line, so a chunk can be opened and read when an eval question misses.
"""

import json
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from statistics import median

from rocm_agent.config import Settings

FENCE = re.compile(r"^\s*(`{3,}|~{3,})")
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
LABEL = re.compile(r"^\s*\*\*Applies to: (.+?)\*\*\s*$")
RULE = re.compile(r"^\s*([-*_])\1{2,}\s*$")
TABLE_RULE = re.compile(r"^\s*\|[\s|:-]+\|\s*$")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

LEAD_IN_MAX = 300  # a short paragraph ending in ":" introduces the next block
OVERLAP_MAX = 200  # longest sentence repeated when one paragraph has to be cut
MIN_CHARS = 25  # a prose-only chunk shorter than this is a leftover, not content

Measure = Callable[[str], int]  # the size of a piece of text; plain length unless told otherwise


@dataclass(frozen=True)
class Block:
    kind: str  # "heading" | "label" | "text" | "code" | "table"
    text: str
    level: int = 0  # headings only


@dataclass(frozen=True)
class Item:
    """One block of content plus the "Applies to" condition in force where it stands."""

    kind: str  # "text" | "code" | "table"
    text: str
    label: str | None  # None = applies to everything


def split_front_matter(text: str) -> tuple[dict[str, str], str]:
    """The cleaner's front matter (title, source, url, ...) and the page body."""
    if not text.startswith("---\n"):
        return {}, text
    head, sep, body = text[4:].partition("\n---\n")
    if not sep:
        return {}, text
    meta = {}
    for line in head.splitlines():
        key, colon, value = line.partition(":")
        if colon:
            meta[key.strip()] = value.strip()
    return meta, body


def parse_blocks(body: str) -> list[Block]:
    """Markdown as a flat list of blocks. Lines inside a code fence are never read as markup."""
    lines = body.split("\n")
    blocks: list[Block] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip() or RULE.match(line):
            i += 1
            continue
        fence = FENCE.match(line)
        if fence:
            marker = fence.group(1)[0] * 3
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith(marker):
                j += 1
            blocks.append(Block("code", "\n".join(lines[i : j + 1])))
            i = j + 1
            continue
        heading = HEADING.match(line)
        if heading:
            blocks.append(Block("heading", heading.group(2), level=len(heading.group(1))))
            i += 1
            continue
        label = LABEL.match(line)
        if label:
            blocks.append(Block("label", label.group(1).strip()))
            i += 1
            continue
        j = i
        if line.lstrip().startswith("|"):
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                j += 1
            blocks.append(Block("table", "\n".join(lines[i:j])))
        else:  # paragraph or list: runs to a blank line or to the start of another kind of block
            while j < len(lines) and lines[j].strip() and (j == i or not _starts_block(lines[j])):
                j += 1
            blocks.append(Block("text", "\n".join(lines[i:j])))
        i = j
    return blocks


def _starts_block(line: str) -> bool:
    return bool(
        FENCE.match(line) or HEADING.match(line) or LABEL.match(line) or line.lstrip()[:1] == "|"
    )


def _join_up_to(
    units: list[str],
    limit: int,
    sep: str,
    head: str = "",
    tail: str = "",
    measure: Measure = len,
) -> list[str]:
    """Joins units into pieces that measure at most `limit`, head and tail included.

    A unit is never cut, so one unit larger than the limit becomes a piece on its own.
    """
    pieces: list[str] = []
    cur: list[str] = []
    for unit in units:
        if cur and measure(head + sep.join([*cur, unit]) + tail) > limit:
            pieces.append(head + sep.join(cur) + tail)
            cur = []
        cur.append(unit)
    if cur:
        pieces.append(head + sep.join(cur) + tail)
    return pieces


def split_block(block: Block, limit: int, measure: Measure = len) -> list[str]:
    """A block as one or more pieces. Only a block larger than the limit is split at all."""
    text = block.text
    if measure(text) <= limit:
        return [text]
    lines = text.split("\n")
    if block.kind == "table":  # between rows, header row repeated in every piece, no overlap
        if len(lines) > 2 and TABLE_RULE.match(lines[1]):
            head = "\n".join(lines[:2]) + "\n"
            return _join_up_to(lines[2:], limit, "\n", head=head, measure=measure)
        return _join_up_to(lines, limit, "\n", measure=measure)
    if block.kind == "code":  # between lines, each piece fenced again
        closed = len(lines) > 1 and FENCE.match(lines[-1])
        body = lines[1:-1] if closed else lines[1:]
        closer = lines[-1] if closed else lines[0].split("`")[0] + "```"
        head, tail = lines[0] + "\n", "\n" + closer
        return _join_up_to(body, limit, "\n", head=head, tail=tail, measure=measure)
    if len(lines) > 1:  # a list: between lines
        return _join_up_to(lines, limit, "\n", measure=measure)
    pieces = _join_up_to(SENTENCE_END.split(text), limit, " ", measure=measure)
    out = pieces[:1]  # one paragraph cut mid-way: repeat the previous sentence for context
    for prev, cur in zip(pieces, pieces[1:], strict=False):
        last = SENTENCE_END.split(prev)[-1]
        out.append(f"{last} {cur}" if len(last) <= OVERLAP_MAX else cur)
    return out


def render(items: list[Item]) -> str:
    """Chunk text: the items in order, with a label line wherever the condition changes."""
    out: list[str] = []
    prev: str | None = None  # a chunk starts unconditional, so a leading "all" is not printed
    for item in items:
        if item.label != prev:
            out.append(f"**Applies to: {item.label or 'all'}**")
        out.append(item.text)
        prev = item.label
    return "\n\n".join(out)


def _is_lead_in(item: Item) -> bool:
    return (
        item.kind == "text" and len(item.text) <= LEAD_IN_MAX and item.text.rstrip().endswith(":")
    )


def pack(items: list[Item], limit: int, measure: Measure = len) -> list[list[Item]]:
    """Groups a section's items into chunks that measure at most `limit`, whole items only."""
    chunks: list[list[Item]] = []
    cur: list[Item] = []
    for item in items:
        only_lead_ins = all(_is_lead_in(c) for c in cur)
        if cur and not only_lead_ins and measure(render([*cur, item])) > limit:
            carried = [cur.pop()] if _is_lead_in(cur[-1]) else []  # keep it with what it introduces
            chunks.append(cur)
            cur = carried
        cur.append(item)
    if cur:
        chunks.append(cur)
    return chunks


def chunk_page(body: str, title: str, limit: int, measure: Measure = len) -> list[dict]:
    """Chunks of one page: {"heading_path": [...], "text": str, "applies": [labels]}."""
    chunks: list[dict] = []
    stack: list[tuple[int, str]] = []  # open headings as (level, text)
    items: list[Item] = []
    label: str | None = None

    def flush() -> None:
        path = [title, *(text for _, text in stack if text != title)]
        for group in pack(items, limit, measure):
            structured = any(item.kind != "text" for item in group)
            if not structured and sum(len(item.text) for item in group) < MIN_CHARS:
                continue
            if all(_is_lead_in(item) for item in group):
                continue  # "The following are the changes:" with the content under sub-headings
            labels = list(dict.fromkeys(item.label for item in group if item.label))
            chunks.append({"heading_path": path, "text": render(group), "applies": labels})
        items.clear()

    for block in parse_blocks(body):
        if block.kind == "heading":
            flush()
            while stack and stack[-1][0] >= block.level:
                stack.pop()
            stack.append((block.level, block.text))
            label = None  # the cleaner's convention: a label holds until the next label or heading
        elif block.kind == "label":
            label = None if block.text.lower() == "all" else block.text
        else:
            pieces = split_block(block, limit, measure)
            items.extend(Item(block.kind, piece, label) for piece in pieces)
    flush()
    return chunks


def effective_measure(settings: Settings) -> Measure:
    """Size in "effective characters": prose counts by its length, token-dense text by more.

    The embedding model reads a fixed number of tokens and silently ignores the rest. Prose has
    more characters per token than commands, paths and tables do, so one character limit lets
    the dense kind overflow. Counting `tokens * max_chars / max_tokens` as well makes a chunk
    full at `max_tokens` tokens or at `max_chars` characters, whichever comes first.
    """
    from rocm_agent.embedding import token_counter  # loads the tokenizer, so only when needed

    count = token_counter(settings)
    per_token = settings.chunking.max_chars / settings.chunking.max_tokens
    return lambda text: max(len(text), round(count(text) * per_token))


def build_chunks(settings: Settings, measure: Measure | None = None) -> list[dict]:
    """Every clean page as chunk rows, in a stable order with stable ids."""
    measure = measure or effective_measure(settings)
    root = settings.clean_dir / settings.corpus.name
    pages = sorted(root.rglob("*.md"))
    if not pages:
        raise SystemExit(f"No clean pages under {root}. Run `rocm-agent clean-docs` first.")
    rows: list[dict] = []
    for page in pages:
        meta, body = split_front_matter(page.read_text("utf-8"))
        rel = page.relative_to(root).as_posix()
        title = meta.get("title") or rel
        for n, chunk in enumerate(chunk_page(body, title, settings.chunking.max_chars, measure)):
            heading_path = " > ".join(chunk["heading_path"])
            rows.append(
                {
                    # same id on every run for the same page and position: re-indexing replaces
                    "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{rel}#{n}")),
                    "path": rel,
                    "chunk_index": n,
                    "title": title,
                    "url": meta.get("url", ""),
                    "source": meta.get("source", rel),
                    "rocm_version": meta.get("rocm_version", ""),
                    "templated": meta.get("templated") == "true",
                    "heading_path": heading_path,
                    "applies": " | ".join(chunk["applies"]),
                    "text": chunk["text"],
                    # what BM25 and the embedding model see: the heading path, then the text
                    "search_text": f"{heading_path}\n\n{chunk['text']}",
                }
            )
    return rows


def write_chunks(settings: Settings) -> dict:
    """Builds the chunks, writes chunks.jsonl and returns the numbers worth looking at."""
    measure = effective_measure(settings)
    rows = build_chunks(settings, measure)
    out = settings.chunks_path
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    sizes = sorted(len(r["text"]) for r in rows)
    limit = settings.chunking.max_chars
    return {
        "pages": len({r["path"] for r in rows}),
        "chunks": len(rows),
        "chars_min": sizes[0],
        "chars_median": int(median(sizes)),
        "chars_p95": sizes[int(len(sizes) * 0.95)],
        "chars_max": sizes[-1],
        "over_limit": sum(measure(r["text"]) > limit for r in rows),  # unsplittable blocks
        "with_applies_label": sum(bool(r["applies"]) for r in rows),
        "limit": limit,
        "saved_to": str(out),
    }


def load_chunks(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"{path} not found. Run `rocm-agent chunk` first.")
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def print_stats(s: dict) -> None:
    print(f"{s['chunks']} chunks from {s['pages']} pages (limit {s['limit']} characters)")
    print(
        f"  characters: min {s['chars_min']}, median {s['chars_median']}, "
        f"p95 {s['chars_p95']}, max {s['chars_max']}"
    )
    print(f"  over the limit (one unsplittable block): {s['over_limit']}")
    print(f"  carrying an 'Applies to' label: {s['with_applies_label']}")
    print(f"Saved: {s['saved_to']}")
