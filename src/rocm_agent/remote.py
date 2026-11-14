"""Fills in `.. remote-content::` sections, which the site pulls from other repos at build time.

Without this, those sections (HIP and ROCR env var tables, the CDNA system optimization page)
are empty headings in the corpus.

- fetch_remote(): finds every remote-content directive in the raw docs, pins each remote file
  to its last commit at or before the docs commit's date (so the corpus is one snapshot in time,
  not "whatever develop is today"), downloads the files into <raw>/_remote/ and writes
  _remote/SOURCES.txt (status, repo@branch, commit, path).
- splice_remote(): used by clean-docs, swaps each directive for the downloaded file's text,
  applying the directive's :start_line: and :replace: options the way the site does.

A file that does not exist upstream at the pinned commit is recorded as "missing", and the
section stays empty (that is what the site showed at the time too).
"""

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REMOTE_DIR = "_remote"
DIRECTIVE = re.compile(r"^\.\. remote-content::[ \t]*\n((?:[ \t]+:[\w_]+:.*\n?)*)", re.M)
OPTION = re.compile(r"^[ \t]+:([\w_]+):[ \t]*(.*)$", re.M)


@dataclass(frozen=True)
class Remote:
    repo: str
    path: str
    branch: str
    start_line: int = 0
    replace: str = ""

    @property
    def local(self) -> Path:
        return Path(REMOTE_DIR, *self.repo.split("/"), *self.path.strip("/").split("/"))


def parse_options(block: str) -> Remote:
    opts = dict(OPTION.findall(block))
    return Remote(
        repo=opts["repo"].strip(),
        path=opts["path"].strip(),
        branch=opts.get("default_branch", "develop").strip(),
        start_line=int(opts.get("start_line", 0) or 0),
        replace=opts.get("replace", "").strip(),
    )


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "rocm-docs-agent"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def _api(path: str):
    return json.loads(_get("https://api.github.com/" + path))


def _docs_commit(raw_root: Path) -> tuple[str, str]:
    source = (raw_root / "SOURCE.txt").read_text("utf-8")
    repo = re.search(r"^repo: https://github.com/(.+?)(?:\.git)?$", source, re.M).group(1)
    sha = re.search(r"^commit: (\w+)", source, re.M).group(1)
    return repo, sha


def fetch_remote(raw_root: Path) -> None:
    remotes: set[Remote] = set()
    for page in raw_root.rglob("*.rst"):
        if REMOTE_DIR in page.relative_to(raw_root).parts:
            continue
        for m in DIRECTIVE.finditer(page.read_text("utf-8")):
            remotes.add(parse_options(m.group(1)))
    if not remotes:
        return

    docs_repo, docs_sha = _docs_commit(raw_root)
    until = _api(f"repos/{docs_repo}/commits/{docs_sha}")["commit"]["committer"]["date"]
    print(f"Fetching {len(remotes)} remote-content files, pinned to {until} (docs commit date)...")

    lines = [f"# remote-content files, each pinned to its last commit at or before {until}"]
    for r in sorted(remotes, key=lambda r: (r.repo, r.path)):
        # Pin per file (last commit that touched it), not per repo: in merge-heavy repos such as
        # llvm-project the newest commit by date can be an upstream commit without the file.
        path = r.path.strip("/")
        hits = _api(f"repos/{r.repo}/commits?sha={r.branch}&path={path}&until={until}&per_page=1")
        if not hits:
            lines.append(f"missing\t{r.repo}@{r.branch}\t-\t{r.path}")
            print(f"  missing {r.repo}/{path} (no commit before {until})")
            continue
        sha = hits[0]["sha"]
        try:
            data = _get(f"https://raw.githubusercontent.com/{r.repo}/{sha}/{path}")
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise
            status = "missing"
        else:
            out = raw_root / r.local
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(data)
            status = "ok"
        lines.append(f"{status}\t{r.repo}@{r.branch}\t{sha}\t{r.path}")
        print(f"  {status:<7} {r.repo}/{path}")
    (raw_root / REMOTE_DIR).mkdir(exist_ok=True)
    (raw_root / REMOTE_DIR / "SOURCES.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def splice_remote(src: str, raw_root: Path) -> tuple[str, list[str]]:
    """Replace remote-content directives with the fetched text. Returns (text, unresolved)."""
    unresolved: list[str] = []

    def sub(m: re.Match) -> str:
        r = parse_options(m.group(1))
        local = raw_root / r.local
        if not local.exists():
            unresolved.append(f"{r.repo}/{r.path.strip('/')}")
            return "\n"
        text = local.read_text("utf-8")
        for spec in filter(None, (s.strip() for s in r.replace.split(";;"))):
            if "|" in spec:
                old, new = spec.split("|", 1)
                text = text.replace(old.strip(), new.strip())
        text = "\n".join(text.splitlines()[r.start_line :])
        return f"\n{text}\n\n"

    return DIRECTIVE.sub(sub, src), unresolved
