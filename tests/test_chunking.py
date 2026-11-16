from dataclasses import replace

from rocm_agent import config
from rocm_agent.chunking import (
    Block,
    build_chunks,
    chunk_page,
    parse_blocks,
    split_block,
    split_front_matter,
)

LONG = "This sentence is long enough to count as content. "


def test_front_matter_is_split_from_the_body():
    meta, body = split_front_matter("---\ntitle: Install\nsource: a.rst\n---\nBody text.\n")
    assert meta == {"title": "Install", "source": "a.rst"}
    assert body == "Body text.\n"
    assert split_front_matter("No front matter.") == ({}, "No front matter.")


def test_heading_path_nests_and_starts_with_the_title():
    body = f"# T\n\n{LONG}\n\n## A\n\n{LONG}\n\n### B\n\n{LONG}\n\n## C\n\n{LONG}\n"
    paths = [c["heading_path"] for c in chunk_page(body, "T", 1200)]
    assert paths == [["T"], ["T", "A"], ["T", "A", "B"], ["T", "C"]]


def test_label_is_dropped_at_the_next_heading():
    body = f"## A\n\n**Applies to: os=ubuntu**\n\n{LONG}\n\n## B\n\n{LONG}\n"
    first, second = chunk_page(body, "T", 1200)
    assert first["applies"] == ["os=ubuntu"]
    assert second["applies"] == [] and "Applies to" not in second["text"]


def test_split_section_restates_the_label_in_force():
    paragraphs = "\n\n".join(f"Paragraph {n}. " + "word " * 30 for n in range(4))
    body = f"## A\n\n**Applies to: os=ubuntu**\n\n{paragraphs}\n"
    chunks = chunk_page(body, "T", 400)
    assert len(chunks) > 1
    assert all(c["text"].startswith("**Applies to: os=ubuntu**") for c in chunks)


def test_all_is_not_printed_first_but_is_printed_when_a_condition_ends():
    body = (
        f"## A\n\n{LONG}\n\n**Applies to: os=ubuntu**\n\nUbuntu only. {LONG}\n\n"
        f"**Applies to: all**\n\nEveryone again. {LONG}\n"
    )
    (chunk,) = chunk_page(body, "T", 1200)
    assert not chunk["text"].startswith("**Applies to")
    assert "**Applies to: os=ubuntu**" in chunk["text"]
    assert "**Applies to: all**\n\nEveryone again." in chunk["text"]
    assert chunk["applies"] == ["os=ubuntu"]


def test_comment_line_in_a_code_fence_is_not_a_heading():
    body = "## H\n\n```bash\n# install it\napt install foo\n```\n"
    blocks = parse_blocks(body)
    assert [b.kind for b in blocks] == ["heading", "code"]
    (chunk,) = chunk_page(body, "T", 1200)
    assert chunk["heading_path"] == ["T", "H"] and "# install it" in chunk["text"]


def test_long_table_splits_between_rows_and_repeats_the_header():
    rows = [f"| row{n:02d} | {'x' * 20} |" for n in range(20)]
    table = "\n".join(["| Name | Value |", "|---|---|", *rows])
    pieces = split_block(Block("table", table), 300)
    assert len(pieces) > 1
    for piece in pieces:
        assert piece.startswith("| Name | Value |\n|---|---|\n")
    body_lines = [ln for p in pieces for ln in p.split("\n")[2:]]
    assert body_lines == rows  # every row once, in order, none cut


def test_lead_in_stays_with_the_block_after_it():
    body = f"## H\n\n{'a' * 150}\n\nRun the following command:\n\n```bash\n{'b' * 150}\n```\n"
    chunks = chunk_page(body, "T", 300)
    assert len(chunks) == 2
    assert not chunks[0]["text"].rstrip().endswith(":")
    assert chunks[1]["text"].startswith("Run the following command:\n\n```bash")


def test_section_holding_only_a_lead_in_is_dropped():
    body = f"## A\n\nThe following are the changes:\n\n## B\n\n{LONG}\n"
    chunks = chunk_page(body, "T", 1200)
    assert [c["heading_path"] for c in chunks] == [["T", "B"]]


def test_ids_are_identical_across_two_builds(tmp_path):
    page = tmp_path / "rocm" / "install" / "a.md"
    page.parent.mkdir(parents=True)
    page.write_text(f"---\ntitle: A\n---\n# A\n\n{LONG}\n\n## B\n\n{LONG}\n", encoding="utf-8")
    settings = replace(config.load(), clean_dir=tmp_path)
    first, second = build_chunks(settings), build_chunks(settings)
    assert [r["id"] for r in first] == [r["id"] for r in second]
    assert len({r["id"] for r in first}) == len(first) == 2


def test_packing_follows_the_measure_not_the_character_count():
    from rocm_agent.chunking import chunk_page

    body = "# T\n\n" + "\n\n".join(f"Paragraph number {n} with some words." for n in range(6))
    by_chars = chunk_page(body, "T", limit=400)
    dense = chunk_page(body, "T", limit=400, measure=lambda text: len(text) * 4)
    assert len(by_chars) == 1  # 6 short paragraphs fit in 400 characters
    assert len(dense) > 1  # the same text counted four times heavier does not
