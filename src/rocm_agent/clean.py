"""Turns the raw docs tree into one clean Markdown file per page.

- .rst pages go through pandoc (rst -> GitHub-flavoured Markdown).
- .md pages (MyST) get a light regex cleanup: substitutions, admonitions, roles, labels.
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
}
TABLE_FILTER = Path(__file__).with_name("flatten_tables.lua")  # list cells -> one line
TEMPLATED_MARKERS =(".. datatemplate", ".. selector", "{datatemplate", ".jinja")


@dataclass
class PageReport:
    source: str
    out_chars: int
    templated: bool
    error: str | None = None


def _version(raw_root: Path) -> str:
    m = re.search(r'^ROCM_VERSION\s*=\s*"([^"]+)"', (raw_root / "conf.py").read_text("utf-8"), re.M)
    return m.group(1) if m else "unknown"


def _clean_myst(text: str, version: str) -> str:
    text = re.sub(r"\{\{\s*ROCM_VERSION\s*\}\}", version, text)
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


def _clean_rst(path: Path, version: str) -> str:
    src = path.read_text("utf-8").replace("|ROCM_VERSION|", version)
    src = re.sub(r"^:selector[\w-]*:.*$", "", src, flags=re.M)  # page-level selector options
    # gfm-raw_html drops `.. raw:: html` blocks (layout markup, not content)
    md = pypandoc.convert_text(
        src,
        "gfm-raw_html",
        format="rst",
        extra_args=["--wrap=none", "--quiet", f"--lua-filter={TABLE_FILTER}"],
        cworkdir=str(path.parent),
    )
    md = re.sub(r"`([^`<\n]+?)\s+<[^>`\n]+>`", r"\1", md)  # :doc:`text <path>` leftovers
    return md


HTML_TAG = re.compile(
    r"</?(?:p|span|div|a|br|img|i|b|strong|em|sup|sub|svg|path|ul|li|table|thead|tbody|tr|td|th)"
    r"\b[^>]*>"
)


def _tidy(text: str) -> str:
    text = text.replace("\r\n", "\n")
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
        try:
            body = _clean_rst(path, version) if path.suffix == ".rst" else _clean_myst(raw, version)
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
        reports.append(PageReport(str(rel), len(body), templated))
    return reports


def print_report(reports: list[PageReport]) -> None:
    ok = [r for r in reports if not r.error]
    print(f"Cleaned {len(ok)} pages, {sum(r.out_chars for r in ok):,} characters in total.")
    for r in reports:
        if r.error:
            print(f"  FAILED    {r.source}: {r.error}")
    for r in ok:
        if r.templated or r.out_chars < 400:
            why = "templated" if r.templated else "short"
            print(f"  {why:<9} {r.source} ({r.out_chars} chars)")
