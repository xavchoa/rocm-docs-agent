"""The agent's tools as the model sees them: a name, a description and typed arguments.

The model never runs a tool. It reads these definitions, and when it wants one it replies with
a tool call: the tool's name plus arguments as JSON. Our code checks those arguments against
the classes below, runs the tool and sends the result back (see graph.py). So a description is
a prompt: it is the only thing that tells the model when this tool is the right one.
"""

from dataclasses import dataclass

from pydantic import BaseModel, Field


class SearchDocsArgs(BaseModel):
    query: str = Field(
        description=(
            "What to look for: a few key terms or a short question. Keep exact names such as "
            "error messages, package names, GPU models and gfx targets as the user wrote them."
        )
    )


class LookupGpuSupportArgs(BaseModel):
    gpu: str = Field(
        description=(
            "The GPU, APU or gfx target exactly as the user wrote it, for example "
            '"RX 7900 XTX", "Ryzen AI Max 395" or "gfx1151".'
        )
    )
    os: str | None = Field(
        default=None,
        description=(
            'Operating system to check, if the user named one, for example "Ubuntu", "RHEL" '
            'or "Windows". Leave it out to list every supported one.'
        ),
    )


class ToolResult(BaseModel):
    """What a tool hands back to the graph.

    `text` is a short line for the model. `excerpts` are the records an answer may cite. Each
    has the fields of a chunk that the answer step reads: id, path, title, url, heading_path
    and text. The graph numbers them, so a tool never numbers anything itself.
    """

    text: str
    excerpts: list[dict] = []


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args: type[BaseModel]

    def for_model(self) -> dict:
        """The tool definition that is sent to the model with every request."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.args.model_json_schema(),
        }


SEARCH_DOCS = ToolSpec(
    name="search_docs",
    description=(
        "Search the ROCm documentation (install guides, release notes, reference pages, "
        "environment variables) and get back the most relevant excerpts. Use it for anything "
        "the docs explain in prose: how to install or uninstall, what an error means, what "
        "changed in a release. The docs may use other words than the user does, so if the "
        "first results do not cover the question, search again with different terms."
    ),
    args=SearchDocsArgs,
)

LOOKUP_GPU_SUPPORT = ToolSpec(
    name="lookup_gpu_support",
    description=(
        "Check the official compatibility data for one GPU, APU or gfx target: whether that "
        "exact product is listed as supported, and on which operating systems and versions. "
        "Use it when the question is whether specific hardware is supported. It matches exact "
        "product names only. A similar name is a different product, and the result names the "
        "listed products that are close."
    ),
    args=LookupGpuSupportArgs,
)

TOOL_SPECS = (SEARCH_DOCS, LOOKUP_GPU_SUPPORT)
