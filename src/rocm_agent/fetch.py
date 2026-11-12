"""Downloads the docs folder of the configured corpus repo (shallow, sparse git clone).

Result: <raw_dir>/<corpus name>/ holds the docs tree, and SOURCE.txt records the exact commit,
so every eval number can be traced back to a fixed version of the docs.
"""

import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from rocm_agent.config import Settings


def _git(*args: str, cwd: Path | None = None) -> str:
    out = subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)
    return out.stdout.strip()


def fetch_docs(settings: Settings, force: bool = False) -> Path:
    c = settings.corpus
    dest = settings.raw_dir / c.name
    if dest.exists():
        if not force:
            print(f"{dest} already exists; use --force to download again.")
            return dest
        shutil.rmtree(dest, onexc=_make_writable)

    checkout = settings.raw_dir / f".{c.name}-checkout"
    if checkout.exists():
        shutil.rmtree(checkout, onexc=_make_writable)
    settings.raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"Cloning {c.repo}@{c.ref} ({c.subdir}/ only)...")
    _git(
        "clone",
        "--depth",
        "1",
        "--filter=blob:none",
        "--sparse",
        "--branch",
        c.ref,
        c.repo,
        str(checkout),
    )
    _git("sparse-checkout", "set", c.subdir, cwd=checkout)
    sha = _git("rev-parse", "HEAD", cwd=checkout)

    shutil.move(str(checkout / c.subdir), str(dest))
    shutil.rmtree(checkout, onexc=_make_writable)

    (dest / "SOURCE.txt").write_text(
        f"repo: {c.repo}\nref: {c.ref}\ncommit: {sha}\n"
        f"fetched: {datetime.now(UTC).isoformat(timespec='seconds')}\n",
        encoding="utf-8",
    )
    counts: dict[str, int] = {}
    for p in dest.rglob("*"):
        if p.is_file():
            counts[p.suffix or "(none)"] = counts.get(p.suffix or "(none)", 0) + 1
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:6]
    print(f"Saved to {dest} at commit {sha[:10]}. Files by type: {top}")
    return dest


def _make_writable(func, path, _exc):  # git marks pack files read-only on Windows
    Path(path).chmod(0o700)
    func(path)
