"""The agent as a LangGraph graph: a research loop that gathers excerpts, then one answer step.

    START -> research -> tools -> research -> ... -> answer -> END

- `research` calls the model with the tool definitions attached. The model either asks for
  tools or stops.
- `tools` runs what the model asked for. The model never runs a tool: it returns a request
  (a name and arguments), this node executes it, and the result goes back as a tool message
  that carries the same id as the request.
- `answer` writes the reply from every excerpt gathered, with the same rules and the same
  citation check as the plain RAG path in answer.py. The agent changes how excerpts are found,
  not how an answer is grounded.

The state is what the nodes share. A node returns only the keys it changes, and a reducer says
how a change is merged: `messages` and `excerpts` are appended to, the rest are overwritten.

Which node runs next is plain Python (`after_research`), not a decision the model makes. The
loop is bounded by `max_rounds`, counted in the state, so a model that keeps searching is
stopped and answered with what it has gathered, instead of failing with a recursion error.
"""

import operator
from collections.abc import Callable, Sequence
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ValidationError

from rocm_agent.answer import (
    SEPARATOR,
    Citation,
    Draft,
    Llm,
    build_system,
    build_user,
    format_excerpt,
    ground,
)
from rocm_agent.tools.schemas import ToolResult, ToolSpec

RESEARCH_PROMPT = """\
You are the research step of a documentation assistant for AMD ROCm. A later step writes the answer from the excerpts you gather, so your job is only to gather them. Nothing you know from memory counts: only what the tools return can be cited.

- For anything the docs explain in prose (installing, uninstalling, errors, concepts, release notes, environment variables), call search_docs. If the results do not cover the question, search again with different words, because the docs may not use the user's terms.
- When the question is whether a specific GPU, APU or gfx target is supported, or on which operating systems, call lookup_gpu_support with the product name exactly as the user wrote it. Search the docs as well when the user also asks how to do something.
- Stop as soon as the excerpts cover the question, or when one more search would not add anything. If the question is too vague to search for (no error message, no product, no step that failed), stop at once: the answer step will ask the user for details.
- When you stop, reply with the single word DONE.
"""  # noqa: E501

Chat = Callable[[list[AnyMessage]], AIMessage]  # one model call with the tools attached
ToolFn = Callable[[BaseModel], ToolResult]  # a tool's implementation


class State(TypedDict):
    question: str
    messages: Annotated[list[AnyMessage], add_messages]  # the research conversation; appended to
    excerpts: Annotated[list[dict], operator.add]  # all the tools found, in order; appended to
    rounds: int  # research turns taken so far
    draft: Draft | None  # the final reply, set by `answer`
    citations: list[Citation]
    uncited: bool
    usage: dict[str, int]  # token usage of the answer step


def initial_state(question: str) -> State:
    return {
        "question": question,
        "messages": [HumanMessage(question)],
        "excerpts": [],
        "rounds": 0,
        "draft": None,
        "citations": [],
        "uncited": False,
        "usage": {},
    }


def build_graph(
    chat: Chat,
    specs: Sequence[ToolSpec],
    tools: dict[str, ToolFn],
    write_answer: Llm,
    max_rounds: int = 4,
):
    """Wires the three nodes. Everything that talks to a model or to data is passed in."""
    by_name = {spec.name: spec for spec in specs}

    def research(state: State) -> dict:
        reply = chat([SystemMessage(RESEARCH_PROMPT), *state["messages"]])
        return {"messages": [reply], "rounds": state["rounds"] + 1}

    def after_research(state: State) -> str:
        last = state["messages"][-1]
        wants_tools = isinstance(last, AIMessage) and bool(last.tool_calls)
        # Past the limit the request is left unanswered on purpose: `answer` builds its own
        # prompt from the question and the excerpts, not from these messages.
        return "tools" if wants_tools and state["rounds"] <= max_rounds else "answer"

    def run_tools(state: State) -> dict:
        # number of every excerpt found so far, so a second search can point back to it
        numbers = {excerpt["id"]: n for n, excerpt in enumerate(state["excerpts"], 1)}
        found: list[dict] = []

        def run_one(call: dict) -> str:
            spec = by_name.get(call["name"])
            if spec is None:
                return f"Unknown tool {call['name']!r}. Available: {', '.join(by_name)}."
            try:
                args = spec.args(**call["args"])
            except ValidationError as error:  # tell the model what was wrong so it can retry
                return f"Invalid arguments for {spec.name}: {error.errors(include_url=False)}"
            try:
                result = tools[spec.name](args)
            except Exception as error:  # a failing tool must not end the run
                return f"{spec.name} failed: {error}"
            parts = [result.text]
            for excerpt in result.excerpts:
                if excerpt["id"] in numbers:
                    n = numbers[excerpt["id"]]
                    parts.append(f"[{n}] (shown above) {excerpt['heading_path']}")
                else:
                    n = numbers[excerpt["id"]] = len(numbers) + 1
                    found.append(excerpt)
                    parts.append(format_excerpt(n, excerpt))
            return SEPARATOR.join(parts)

        request = state["messages"][-1]
        replies = [
            ToolMessage(content=run_one(call), tool_call_id=call["id"])
            for call in request.tool_calls
        ]
        return {"messages": replies, "excerpts": found}

    def answer(state: State) -> dict:
        excerpts = state["excerpts"]
        # Called even with no excerpts: for a vague question the right reply is to ask for
        # details, and that is the answer step's call. `ground` still refuses an uncited answer.
        draft, usage = write_answer(build_system(excerpts), build_user(state["question"], excerpts))
        draft, citations, uncited = ground(draft, excerpts)
        return {"draft": draft, "citations": citations, "uncited": uncited, "usage": usage}

    graph = StateGraph(State)
    graph.add_node("research", research)
    graph.add_node("tools", run_tools)
    graph.add_node("answer", answer)
    graph.add_edge(START, "research")
    graph.add_conditional_edges("research", after_research, ["tools", "answer"])
    graph.add_edge("tools", "research")
    graph.add_edge("answer", END)
    return graph.compile()
