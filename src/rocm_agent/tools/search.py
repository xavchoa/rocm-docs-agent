"""search_docs: the documentation retriever as an agent tool."""

from rocm_agent.tools.schemas import SearchDocsArgs, ToolResult


def search_docs(args: SearchDocsArgs, k: int = 5) -> ToolResult:
    from rocm_agent.retrieval import retrieve

    chunks = retrieve(args.query, k)
    if not chunks:
        return ToolResult(text="No excerpts found.")
    return ToolResult(text=f"{len(chunks)} excerpts found for {args.query}.", excerpts=chunks)
