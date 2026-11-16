"""Runs the LangGraph agent (graph.py) and maps its final state to an `Answer`."""

import time
from functools import partial

from langchain_core.messages import AIMessage

from rocm_agent import config
from rocm_agent.answer import Answer, Llm, make_llm, require_key, retrieved_summary
from rocm_agent.graph import Chat, ToolFn, build_graph, initial_state
from rocm_agent.tools.gpu_support import lookup_gpu_support
from rocm_agent.tools.schemas import TOOL_SPECS
from rocm_agent.tools.search import search_docs

MAX_TOKENS = 4096


def make_chat(settings: config.Settings) -> Chat:
    """One research turn: messages in, the model's reply (text or tool requests) out."""
    require_key(settings.llm_provider)
    if settings.llm_provider != "anthropic":
        raise SystemExit("The agent needs llm.provider = 'anthropic' in config.toml.")
    from langchain_anthropic import ChatAnthropic

    model = ChatAnthropic(model=settings.llm_model, max_tokens=MAX_TOKENS)
    bound = model.bind_tools([spec.for_model() for spec in TOOL_SPECS])
    return lambda messages: bound.invoke(messages)


def default_tools(settings: config.Settings) -> dict[str, ToolFn]:
    return {
        "search_docs": search_docs,
        "lookup_gpu_support": partial(lookup_gpu_support, settings=settings),
    }


def run_agent(
    question: str,
    *,
    settings: config.Settings | None = None,
    chat: Chat | None = None,
    tools: dict[str, ToolFn] | None = None,
    write_answer: Llm | None = None,
    max_rounds: int = 4,
) -> Answer:
    start = time.perf_counter()
    settings = settings or config.load()
    chat = chat or make_chat(settings)
    tools = tools or default_tools(settings)
    write_answer = write_answer or make_llm(settings)

    graph = build_graph(chat, TOOL_SPECS, tools, write_answer, max_rounds)
    state = graph.invoke(initial_state(question), config={"recursion_limit": 4 * max_rounds + 10})

    draft, usage = state["draft"], state["usage"]
    return Answer(
        question=question,
        status=draft.status,
        answer=draft.answer,
        citations=state["citations"],
        retrieved=retrieved_summary(state["excerpts"]),
        model=settings.llm_model,
        latency_s=round(time.perf_counter() - start, 3),
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
        uncited=state["uncited"],
        rounds=state["rounds"],
        tool_calls=[
            call["name"]
            for message in state["messages"]
            if isinstance(message, AIMessage)
            for call in message.tool_calls
        ],
    )
