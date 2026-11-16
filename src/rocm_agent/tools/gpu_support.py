"""lookup_gpu_support: exact-name lookup in the ROCm compatibility data.

The data is gpus.yaml and os-support.yaml.
gpus.yaml maps family ("AMD Instinct", "AMD Radeon", "AMD Ryzen") -> product -> details (gfx
target, series, architecture), and os-support.yaml maps the same keys to the supported operating
systems. Both use YAML anchors and `<<:` merge keys, which safe_load resolves. A product matches
only by its exact normalised name or gfx target: a similar name is a different product.
"""

import difflib
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from rocm_agent import config
from rocm_agent.tools.schemas import LookupGpuSupportArgs, ToolResult

BRANDS = ("radeon", "instinct", "ryzen")
PAGE = "compatibility/compatibility-matrix"


@dataclass(frozen=True)
class OsEntry:
    name: str
    abbr: str
    versions: tuple[str, ...]  # "24.04.4 (kernel GA 6.8)"


@dataclass(frozen=True)
class Product:
    name: str
    family: str
    gfx: str
    series: str
    arch: str
    oses: tuple[OsEntry, ...]


def normalise(name: str) -> str:
    words = re.sub(r"[-\s]+", " ", name.lower()).split()
    return " ".join(w for w in words if w != "amd")


def _without_brand(norm: str) -> str:
    return " ".join(w for w in norm.split() if w not in BRANDS)


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _version_text(item) -> str:
    if isinstance(item, dict):
        text = str(item.get("version", ""))
        return f"{text} (kernel {item['kernel']})" if item.get("kernel") else text
    return str(item)


@lru_cache(maxsize=4)
def load_products(data_dir: Path) -> tuple[Product, ...]:
    gpus = _load_yaml(data_dir / "gpus.yaml")
    systems = _load_yaml(data_dir / "os-support.yaml")
    products = []
    for family, members in gpus.items():
        if family.startswith("_") or not isinstance(members, dict):
            continue
        for name, info in members.items():
            oses = tuple(
                OsEntry(
                    os_name,
                    str(entry.get("abbr", "")),
                    tuple(_version_text(v) for v in entry.get("versions", [])),
                )
                for os_name, entry in (systems.get(family, {}).get(name) or {}).items()
            )
            products.append(
                Product(
                    name=name,
                    family=family,
                    gfx=str(info.get("gfx", "")),
                    series=(info.get("series") or {}).get("name", ""),
                    arch=(info.get("arch") or {}).get("name", ""),
                    oses=oses,
                )
            )
    return tuple(products)


def _rocm_version(raw_root: Path) -> str:
    try:
        text = (raw_root / "conf.py").read_text("utf-8")
    except OSError:
        return ""
    m = re.search(r'^ROCM_VERSION\s*=\s*"([^"]+)"', text, re.M)
    return m.group(1) if m else ""


def _matches(query: str, product: Product) -> bool:
    norm = normalise(query)
    own = normalise(product.name)
    return norm in (own, _without_brand(own), product.gfx.lower())


def _filter_os(product: Product, wanted: str | None) -> list[OsEntry]:
    if not wanted:
        return list(product.oses)
    w = wanted.lower().strip()
    return [o for o in product.oses if w in o.name.lower() or (o.abbr and w in o.abbr.lower())]


def _os_line(entry: OsEntry) -> str:
    return f"{entry.name}: {', '.join(entry.versions)}"


def _facts(product: Product, wanted: str | None) -> list[str]:
    lines = [f"{product.name} is listed as officially supported."]
    lines.append(
        f"Family: {product.family}. Series: {product.series}. Architecture: {product.arch}."
    )
    lines.append(f"GFX target: {product.gfx}.")
    found = _filter_os(product, wanted)
    if wanted and not found:
        lines.append(f"{wanted} is not listed for this product.")
    elif found:
        lines.append("Supported operating systems:")
        lines += [_os_line(o) for o in found]
    return lines


def lookup_gpu_support(args: LookupGpuSupportArgs, settings: config.Settings) -> ToolResult:
    raw_root = settings.raw_dir / settings.corpus.name
    products = load_products(raw_root / "data")
    hits = [p for p in products if _matches(args.gpu, p)]

    if not hits:
        # compare with the name as listed and with the brand word dropped, as users write it
        names = {}
        for p in products:
            full = normalise(p.name)
            names[full] = names[_without_brand(full)] = p.name
        close = difflib.get_close_matches(normalise(args.gpu), list(names), n=10, cutoff=0.5)
        nearest = list(dict.fromkeys(names[c] for c in close))[:5]
        text = f"{args.gpu} is not in the official ROCm compatibility list."
        if nearest:
            text += " Closest listed products: " + ", ".join(nearest) + "."
        return ToolResult(text=text)

    if len(hits) == 1:
        text = "\n".join(_facts(hits[0], args.os))
    else:
        shown = ", ".join(p.name for p in hits[:10]) + (
            f" and {len(hits) - 10} more" if len(hits) > 10 else ""
        )
        text = (
            f"{len(hits)} listed products match {args.gpu} (gfx target {hits[0].gfx}): {shown}. "
            "Details are in the excerpts."
        )

    version = _rocm_version(raw_root)
    base = settings.corpus.site_url + PAGE
    excerpts = [
        {
            "id": f"gpu-support:{p.name}",
            "path": f"{PAGE}.md",
            "title": "ROCm compatibility matrix",
            "url": base + ".html",
            "heading_path": f"Compatibility matrix > {p.name}",
            "text": "\n".join(_facts(p, args.os)),
            "rocm_version": version,
        }
        for p in hits
    ]
    return ToolResult(text=text, excerpts=excerpts)
