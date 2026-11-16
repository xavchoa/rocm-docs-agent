"""The answer step: retrieved excerpts in, a cited answer out.

The model sees the excerpts as a numbered list and cites them by number. The code, not the
model, turns those numbers into titles and URLs, so a link in the result always comes from a
chunk that was retrieved and can never be invented. Numbers that do not exist are dropped.

An "answered" reply with no valid number left is not grounded in the docs. It is turned into
"not_in_docs" and flagged `uncited`, so the eval can count how often that happens.
"""

import os
import time
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel

from rocm_agent import config

SYSTEM_PROMPT = """\
You answer questions about AMD ROCm for developers, using only the documentation excerpts in the user message. They come from the ROCm {version} docs and are the only source you may use, even when you know more: an answer from memory cannot be cited and may be out of date.

How to answer:
- Be direct and brief. Copy commands, version numbers, package names, paths and URLs exactly as they appear in the excerpts.
- Many steps depend on the operating system, install method or GPU. Excerpts mark them with lines such as "**Applies to: install_method=pkgman; os=ubuntu,debian**". Give only the steps that match what the user told you. If they did not say which case applies and the steps differ, give each variant with its condition, or ask which one they need.
- Product names are exact. A model with an extra prefix or suffix is a different product. Say a product is supported only if that exact name is in the excerpts. If a name looks like a mix-up of products that are listed, say so and name the listed ones.
- If the excerpts do not answer the question but point to another page or link for it, say that these docs do not cover it, give that link, keep status "answered" and cite the excerpt that holds the link.
- If the excerpts neither answer the question nor point anywhere, set status to "not_in_docs" and say so in one sentence. Do not fill the gap from memory, even when the excerpts are about a related topic.
- If the question is too vague to answer (no error message, no operating system, no step that failed), set status to "needs_clarification" and ask for the one or two details you need.

In "sources", list the numbers of the excerpts your answer relies on. Leave it empty for "not_in_docs" and "needs_clarification".
"""  # noqa: E501

UNSUPPORTED = "The documentation I retrieved does not support an answer to this."
SEPARATOR = "\n\n" + "-" * 40 + "\n\n"
# Room for the model's own reasoning as well as the reply: thinking tokens count against this
# limit, and a reply cut off in the middle of its JSON fails to parse.
MAX_TOKENS = 4096
KEY_VARIABLES = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}

Status = Literal["answered", "not_in_docs", "needs_clarification"]
Llm = Callable[[str, str], tuple["Draft", dict[str, int]]]


class Draft(BaseModel):
    """What the model returns."""

    status: Status
    answer: str
    sources: list[int]  # numbers of the excerpts the answer relies on


class Citation(BaseModel):
    n: int
    title: str
    url: str
    heading_path: str


class Retrieved(BaseModel):
    n: int
    path: str
    heading_path: str
    url: str


class Answer(BaseModel):
    """What callers get."""

    question: str
    status: Status
    answer: str
    citations: list[Citation]
    retrieved: list[Retrieved]
    model: str
    latency_s: float
    input_tokens: int | None
    output_tokens: int | None
    uncited: bool
    rounds: int | None = None  # research turns taken (agent only)
    tool_calls: list[str] = []  # names of the tools the model asked for, in order (agent only)


def build_system(chunks: list[dict[str, Any]]) -> str:
    version = chunks[0].get("rocm_version") or "" if chunks else ""
    return SYSTEM_PROMPT.format(version=version)


def format_excerpt(n: int, chunk: dict[str, Any]) -> str:
    """One numbered excerpt, as the model sees it in a prompt or in a tool result."""
    return f"[{n}] {chunk['heading_path']}\nURL: {chunk['url']}\n\n{chunk['text']}"


def build_user(question: str, chunks: list[dict[str, Any]]) -> str:
    excerpts = [format_excerpt(n, c) for n, c in enumerate(chunks, 1)]
    return SEPARATOR.join(excerpts) + SEPARATOR + f"Question: {question}"


def valid_numbers(sources: list[int], count: int) -> list[int]:
    """Keep numbers that name a retrieved excerpt, once each, in the order given."""
    kept: list[int] = []
    for n in sources:
        if 1 <= n <= count and n not in kept:
            kept.append(n)
    return kept


def make_citations(numbers: list[int], chunks: list[dict[str, Any]]) -> list[Citation]:
    return [
        Citation(
            n=n,
            title=chunks[n - 1]["title"],
            url=chunks[n - 1]["url"],
            heading_path=chunks[n - 1]["heading_path"],
        )
        for n in numbers
    ]


def require_key(provider: str) -> None:
    from dotenv import load_dotenv

    load_dotenv(config.ROOT / ".env")
    variable = KEY_VARIABLES[provider]
    if not os.environ.get(variable):
        raise SystemExit(f"{variable} is not set: add it to .env (see .env.example).")


def make_llm(settings: config.Settings) -> Llm:
    """Build the model call: (system, user) -> (Draft, token usage)."""
    require_key(settings.llm_provider)
    if settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        # json_schema uses Anthropic's structured output; the default (forced tool call)
        # is rejected by Sonnet 5.5 and Opus 5.5.
        chat = ChatAnthropic(model=settings.llm_model, max_tokens=MAX_TOKENS)
        structured = chat.with_structured_output(Draft, include_raw=True, method="json_schema")
    else:
        from langchain_openai import ChatOpenAI

        chat = ChatOpenAI(model=settings.llm_model, max_tokens=MAX_TOKENS)
        structured = chat.with_structured_output(Draft, include_raw=True)

    def call(system: str, user: str) -> tuple[Draft, dict[str, int]]:
        result = structured.invoke([("system", system), ("user", user)])
        if result["parsed"] is None:
            raise ValueError(f"Model reply did not match the schema: {result['parsing_error']}")
        usage = getattr(result["raw"], "usage_metadata", None) or {}
        return result["parsed"], {
            key: usage[key] for key in ("input_tokens", "output_tokens") if key in usage
        }

    return call


def ground(draft: Draft, chunks: list[dict[str, Any]]) -> tuple[Draft, list[Citation], bool]:
    """Map source numbers to citations; an answer left with none is not_in_docs."""
    if draft.status != "answered":
        return draft, [], False  # a refusal or a question back cites nothing
    numbers = valid_numbers(draft.sources, len(chunks))
    if not numbers:
        return Draft(status="not_in_docs", answer=UNSUPPORTED, sources=[]), [], True
    return draft, make_citations(numbers, chunks), False


def retrieved_summary(chunks: list[dict[str, Any]]) -> list[Retrieved]:
    return [
        Retrieved(n=n, path=c["path"], heading_path=c["heading_path"], url=c["url"])
        for n, c in enumerate(chunks, 1)
    ]


def answer_question(
    question: str,
    k: int | None = None,
    mode: str | None = None,
    *,
    retrieve: Callable[..., list[dict[str, Any]]] | None = None,
    llm: Llm | None = None,
    settings: config.Settings | None = None,
) -> Answer:
    start = time.perf_counter()
    settings = settings or config.load()
    if retrieve is None:
        from rocm_agent.retrieval import retrieve as default_retrieve

        retrieve = default_retrieve
    chunks = retrieve(question, k or settings.retrieval.top_k, mode=mode)

    uncited = False
    usage: dict[str, int] = {}
    if chunks:
        ask = llm or make_llm(settings)
        draft, usage = ask(build_system(chunks), build_user(question, chunks))
        draft, citations, uncited = ground(draft, chunks)
    else:
        draft, citations = Draft(status="not_in_docs", answer=UNSUPPORTED, sources=[]), []

    return Answer(
        question=question,
        status=draft.status,
        answer=draft.answer,
        citations=citations,
        retrieved=retrieved_summary(chunks),
        model=settings.llm_model,
        latency_s=round(time.perf_counter() - start, 3),
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
        uncited=uncited,
    )


def format_answer(result: Answer) -> str:
    """Plain-text view for the terminal: status if not answered, the answer, the citations."""
    lines = [] if result.status == "answered" else [f"Status: {result.status}"]
    lines.append(result.answer)
    lines += [f"[{c.n}] {c.heading_path} ({c.url})" for c in result.citations]
    return "\n".join(lines)
