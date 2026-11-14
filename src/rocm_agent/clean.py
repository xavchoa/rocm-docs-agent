"""Turns the raw docs tree into one clean Markdown file per page.

- .rst pages go through pandoc (rst -> GitHub-flavoured Markdown).
- .md pages (MyST) get a light regex cleanup: substitutions, admonitions, roles, labels,
  {grid} card markup, <style> blocks. Raw HTML tables go through pandoc like the rst tables.
- `.. remote-content::` sections are filled from the files fetch-docs downloaded (remote.py).
- `.. selected::` blocks (website selector) become explicit condition markers, see
  selected_blocks.lua:
  <!-- applies: install_method=pkgman; os=ubuntu,debian --> ... <!-- /applies -->
- Every output file starts with a small front matter block: title, source path, public URL.

What this deliberately does NOT do: render the Jinja/YAML-generated tables (compatibility
matrix, install selectors). Those pages are flagged as "templated" in the report so the
corpus owner can decide how the structured data is used.
"""

import html
import re
from dataclasses import dataclass
from pathlib import Path

import pypandoc

from rocm_agent.config import Settings
from rocm_agent.remote import splice_remote

SKIP_DIRS = {
    "exclude",
    "images",
    "sphinx",
    "extension",
    "data",
    "contribute",
    "include",
    "redirect",
    "_build",
    "_remote",  # remote-content sources, spliced into the pages that use them
}
TABLE_FILTER = Path(__file__).with_name("flatten_tables.lua")  # list cells -> one line
SELECTED_FILTER = Path(__file__).with_name("selected_blocks.lua")  # selector -> applies markers
TEMPLATED_MARKERS = (".. datatemplate", ".. selector", "{datatemplate", ".jinja")


@dataclass
class PageReport:
    source: str
    out_chars: int
    templated: bool
    error: str | None = None
    missing_remote: tuple[str, ...] = ()


def _version(raw_root: Path) -> str:
    m = re.search(r'^ROCM_VERSION\s*=\s*"([^"]+)"', (raw_root / "conf.py").read_text("utf-8"), re.M)
    return m.group(1) if m else "unknown"


def _html_table(m: re.Match) -> str:
    md = pypandoc.convert_text(
        m.group(0),
        "gfm-raw_html",
        format="html",
        extra_args=["--wrap=none", "--quiet", f"--lua-filter={TABLE_FILTER}"],
    )
    return "\n" + md.strip() + "\n"


def _grid_card(m: re.Match) -> str:
    title = m.group(1).strip()
    return f"**{title}**\n" if title else ""


def _clean_myst(text: str, version: str) -> str:
    text = re.sub(r"\{\{\s*ROCM_VERSION\s*\}\}", version, text)
    text = re.sub(r"<table\b.*?</table>", _html_table, text, flags=re.S | re.I)
    # {grid} / {grid-item-card} layout: keep each card's title (bold) and body, drop the fences
    text = re.sub(r"^:{3,}\{grid-item-card\}[ \t]*(.*)$", _grid_card, text, flags=re.M)
    text = re.sub(r"^:{3,}\{grid\}.*$", "", text, flags=re.M)
    text = re.sub(r"^:{3,}[ \t]*$", "", text, flags=re.M)  # closing fences
    text = re.sub(r"^\+\+\+[ \t]*$", "", text, flags=re.M)  # card footer separator
    text = re.sub(r"^\(.+?\)=\s*$", "", text, flags=re.M)  # (label)= targets
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)

    def admonition(m: re.Match) -> str:
        kind = m.group(1).strip().capitalize()
        body = "\n".join(f"> {line}" if line else ">" for line in m.group(2).strip().splitlines())
        return f"> **{kind}:**\n{body}\n"

    text = re.sub(
        r"^```\{(note|important|warning|tip|caution|Note)\}[^\n]*\n(.*?)^```",
        admonition,
        text,
        flags=re.M | re.S,
    )
    text = re.sub(r"^```\{code-block\}\s*(\w*)", r"```\1", text, flags=re.M)
    text = re.sub(r"^```\{include\}[^\n]*\n.*?^```\n?", "", text, flags=re.M | re.S)
    text = re.sub(r"^:[\w-]+:.*$", "", text, flags=re.M)  # directive options left in fences
    text = re.sub(r"\{(?:doc|ref)\}`([^<`]+?)\s*<[^>]+>`", r"\1", text)  # {doc}`text <path>`
    text = re.sub(r"\{(?:doc|ref)\}`([^`]+)`", r"\1", text)
    return text


def _clean_rst(path: Path, version: str, raw_root: Path | None = None) -> tuple[str, list[str]]:
    """Returns (markdown, remote-content files that could not be filled in)."""
    src = path.read_text("utf-8")
    missing: list[str] = []
    if raw_root is not None:
        src, missing = splice_remote(src, raw_root)
    src = src.replace("|ROCM_VERSION|", version)
    src = re.sub(r"^:selector[\w-]*:.*$", "", src, flags=re.M)  # page-level selector options
    # `.. _label:` directly followed by a directive: docutils accepts it, pandoc then reads the
    # directive as a block quote of bullet lists. A blank line in between fixes that.
    src = re.sub(r"^(\.\. _[^\n:]+:[ \t]*\n)(?=\.\. )", r"\1\n", src, flags=re.M)
    # gfm-raw_html drops `.. raw:: html` blocks (layout markup, not content)
    md = pypandoc.convert_text(
        src,
        "gfm-raw_html",
        format="rst",
        extra_args=[
            "--wrap=none",
            "--quiet",
            f"--lua-filter={SELECTED_FILTER}",
            f"--lua-filter={TABLE_FILTER}",
        ],
        cworkdir=str(path.parent),
    )
    md = re.sub(r"`([^`<\n]+?)\s+<[^>`\n]+>`", r"\1", md)  # :doc:`text <path>` leftovers
    return md, missing


HTML_TAG = re.compile(
    r"</?(?:p|span|div|a|br|img|i|b|strong|em|sup|sub|svg|path|ul|li|table|thead|tbody|tr|td|th)"
    r"\b[^>]*>"
)


def _tidy(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"<style\b.*?</style>\s*", "", text, flags=re.S | re.I)  # page CSS
    text = html.unescape(HTML_TAG.sub("", text))  # layout HTML that survived conversion
    text = re.sub(r"[ \t]+$", "", text, flags=re.M)
    # Text pulled out of nested HTML keeps deep indentation; outside code fences, 8+ leading
    # spaces would render as a code block, so drop it.
    lines, in_fence = [], False
    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        lines.append(line if in_fence else re.sub(r"^ {8,}", "", line))
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


def _title(md: str, fallback: str) -> str:
    m = re.search(r"^#\s+(.+)$", md, flags=re.M)
    return m.group(1).strip() if m else fallback


def clean_docs(settings: Settings) -> list[PageReport]:
    c = settings.corpus
    raw_root = settings.raw_dir / c.name
    out_root = settings.clean_dir / c.name
    version = _version(raw_root)
    reports: list[PageReport] = []

    for path in sorted(raw_root.rglob("*")):
        rel = path.relative_to(raw_root)
        if path.suffix not in (".rst", ".md") or SKIP_DIRS.intersection(rel.parts[:-1]):
            continue
        raw = path.read_text("utf-8")
        templated = any(marker in raw for marker in TEMPLATED_MARKERS)
        missing: list[str] = []
        try:
            if path.suffix == ".rst":
                body, missing = _clean_rst(path, version, raw_root)
            else:
                body = _clean_myst(raw, version)
            body = _tidy(body)
        except Exception as exc:  # keep going; the report lists failures
            reports.append(PageReport(str(rel), 0, templated, repr(exc)))
            continue

        url = c.site_url + rel.with_suffix(".html").as_posix()
        front = (
            f"---\ntitle: {_title(body, rel.stem)}\nsource: {rel.as_posix()}\n"
            f"url: {url}\nrocm_version: {version}\n"
            f"templated: {str(templated).lower()}\n---\n\n"
        )
        out = out_root / rel.with_suffix(".md")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(front + body, encoding="utf-8")
        reports.append(PageReport(str(rel), len(body), templated, missing_remote=tuple(missing)))
    return reports


def print_report(reports: list[PageReport]) -> None:
    ok = [r for r in reports if not r.error]
    print(f"Cleaned {len(ok)} pages, {sum(r.out_chars for r in ok):,} characters in total.")
    for r in reports:
        if r.error:
            print(f"  FAILED    {r.source}: {r.error}")
    for r in ok:
        for m in r.missing_remote:
            print(f"  NO REMOTE {r.source}: {m} (not fetched, or missing upstream)")
    for r in ok:
        if r.templated or r.out_chars < 400:
            why = "templated" if r.templated else "short"
            print(f"  {why:<9} {r.source} ({r.out_chars} chars)")
