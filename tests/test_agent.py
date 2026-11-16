from typing import Any

from langchain_core.messages import AIMessage, ToolMessage

from rocm_agent import config
from rocm_agent.agent import run_agent
from rocm_agent.answer import Draft
from rocm_agent.tools.schemas import ToolResult

SETTINGS = config.load()


def chunk(n: int) -> dict[str, Any]:
    return {
        "id": f"c{n}",
        "path": f"page{n}.md",
        "title": f"Page {n}",
        "url": f"https://rocm.docs.amd.com/page{n}.html",
        "heading_path": f"Page {n} > Section",
        "text": f"Body of chunk {n}.",
        "rocm_version": "7.0",
    }


def request(*calls: tuple[str, dict], prefix: str = "t") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {"name": name, "args": args, "id": f"{prefix}{i}", "type": "tool_call"}
            for i, (name, args) in enumerate(calls)
        ],
    )


DONE = AIMessage(content="DONE")


class FakeChat:
    """Replays scripted replies; repeats the last one when the script runs out."""

    def __init__(self, *replies: AIMessage) -> None:
        self.replies = list(replies)
        self.seen: list[list] = []

    def __call__(self, messages):
        self.seen.append(list(messages))
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        # a fresh id each time, as a real model reply has; `add_messages` merges equal ids
        return reply.model_copy(update={"id": None})


class FakeLlm:
    def __init__(self, draft: Draft) -> None:
        self.draft = draft
        self.calls: list[tuple[str, str]] = []

    def __call__(self, system, user):
        self.calls.append((system, user))
        return self.draft, {"input_tokens": 10, "output_tokens": 5}


def run(chat, tools, draft, max_rounds=4):
    llm = FakeLlm(draft)
    result = run_agent(
        "How do I install?",
        settings=SETTINGS,
        chat=chat,
        tools=tools,
        write_answer=llm,
        max_rounds=max_rounds,
    )
    return result, llm


def search_tool(*chunks):
    return lambda args: ToolResult(text=f"{len(chunks)} excerpts found.", excerpts=list(chunks))


ANSWERED = Draft(status="answered", answer="Use the installer.", sources=[2])


def tool_messages(chat: FakeChat) -> list[ToolMessage]:
    return [m for m in chat.seen[-1] if isinstance(m, ToolMessage)]


def test_one_search_then_done_maps_citations():
    chat = FakeChat(request(("search_docs", {"query": "install"})), DONE)
    result, llm = run(chat, {"search_docs": search_tool(chunk(1), chunk(2))}, ANSWERED)
    assert result.status == "answered"
    assert [c.n for c in result.citations] == [2]
    assert result.citations[0].url.endswith("page2.html")
    assert [r.n for r in result.retrieved] == [1, 2]
    assert result.tool_calls == ["search_docs"]
    assert result.rounds == 2
    assert result.input_tokens == 10 and result.output_tokens == 5
    assert "[2] Page 2 > Section" in llm.calls[0][1]


def test_overlapping_chunk_is_not_repeated():
    chat = FakeChat(
        request(("search_docs", {"query": "a"}), prefix="a"),
        request(("search_docs", {"query": "b"}), prefix="b"),
        DONE,
    )
    results = iter([[chunk(1), chunk(2)], [chunk(2), chunk(3)]])
    tools = {"search_docs": lambda args: ToolResult(text="found", excerpts=next(results))}
    result, llm = run(chat, tools, Draft(status="answered", answer="x", sources=[3]))
    assert [r.n for r in result.retrieved] == [1, 2, 3]
    second = tool_messages(chat)[1].content
    assert "[2] (shown above) Page 2 > Section" in second
    assert "[3] Page 3 > Section" in second
    assert second.count("Body of chunk 2.") == 0
    assert llm.calls[0][1].count("Body of chunk 2.") == 1
    assert result.citations[0].url.endswith("page3.html")


def test_unknown_tool_and_invalid_args_become_error_messages():
    chat = FakeChat(
        request(("nope", {}), ("search_docs", {"wrong": 1}), ("search_docs", {"query": "ok"})),
        DONE,
    )
    result, _ = run(chat, {"search_docs": search_tool(chunk(1), chunk(2))}, ANSWERED)
    contents = [m.content for m in tool_messages(chat)]
    assert contents[0].startswith("Unknown tool 'nope'")
    assert contents[1].startswith("Invalid arguments for search_docs")
    assert "2 excerpts found." in contents[2]
    assert result.status == "answered"


def test_model_that_never_stops_is_cut_off_and_still_answered():
    n = {"i": 0}

    def tool(args):
        n["i"] += 1
        return ToolResult(text="found", excerpts=[chunk(n["i"]), chunk(n["i"] + 100)])

    chat = FakeChat(request(("search_docs", {"query": "again"})))
    result, llm = run(chat, {"search_docs": tool}, ANSWERED, max_rounds=3)
    assert result.rounds == 4
    assert len(llm.calls) == 1
    assert result.status == "answered"
    assert len(result.retrieved) == 6  # three executed rounds, two excerpts each


def test_no_tool_calls_still_calls_the_answer_step():
    draft = Draft(status="needs_clarification", answer="Which error?", sources=[])
    chat = FakeChat(DONE)
    result, llm = run(chat, {}, draft)
    assert len(llm.calls) == 1
    assert result.status == "needs_clarification"
    assert result.citations == [] and result.retrieved == []
    assert result.tool_calls == []


def test_failing_tool_is_reported_not_raised():
    def boom(args):
        raise RuntimeError("index missing")

    chat = FakeChat(request(("search_docs", {"query": "x"})), DONE)
    draft = Draft(status="not_in_docs", answer="No.", sources=[])
    result, _ = run(chat, {"search_docs": boom}, draft)
    assert tool_messages(chat)[0].content == "search_docs failed: index missing"
    assert result.status == "not_in_docs"


def test_every_tool_message_answers_its_request():
    chat = FakeChat(
        request(("search_docs", {"query": "a"}), ("nope", {}), prefix="x"),
        request(("search_docs", {"query": "b"}), prefix="y"),
        DONE,
    )
    run(chat, {"search_docs": search_tool(chunk(1), chunk(2))}, ANSWERED)
    final = chat.seen[-1]
    asked = [c["id"] for m in final if isinstance(m, AIMessage) for c in m.tool_calls]
    answered = [m.tool_call_id for m in final if isinstance(m, ToolMessage)]
    assert asked == answered == ["x0", "x1", "y0"]
