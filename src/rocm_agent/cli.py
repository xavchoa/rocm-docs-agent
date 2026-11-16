"""Command line entry point: `rocm-agent <command>`."""

import argparse
import sys

from rocm_agent import config

MODES = ("hybrid", "vector", "bm25")


def main(argv: list[str] | None = None) -> None:
    # Doc text holds characters a Windows console code page cannot print; never crash on output.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(prog="rocm-agent")
    sub = parser.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser("fetch-docs", help="download the docs named in config.toml")
    p_fetch.add_argument("--force", action="store_true", help="download again even if present")

    sub.add_parser("clean-docs", help="convert the downloaded docs to clean Markdown")
    sub.add_parser("chunk", help="split the clean pages into chunks (data/index/chunks.jsonl)")
    sub.add_parser("index", help="embed the chunks into the Chroma index")

    p_eval = sub.add_parser("eval", help="run the eval set and print every reported number")
    p_eval.add_argument("--k", type=int, default=5, help="chunks retrieved per question")
    p_eval.add_argument("--label", default="", help="name for this run, e.g. baseline or option-d")
    p_eval.add_argument(
        "--mode", choices=MODES, default=None, help="retriever to score (default: config.toml)"
    )

    p_search = sub.add_parser("search", help="show the chunks retrieved for a question")
    p_search.add_argument("question")
    p_search.add_argument("--k", type=int, default=5)
    p_search.add_argument("--mode", choices=MODES, default=None)

    p_ask = sub.add_parser("ask", help="answer a question from the docs, with citations")
    p_ask.add_argument("question")
    p_ask.add_argument(
        "--k", type=int, default=None, help="chunks retrieved (default: config.toml)"
    )
    p_ask.add_argument("--mode", choices=MODES, default=None)
    p_ask.add_argument("--json", action="store_true", help="print the full Answer as JSON")
    p_ask.add_argument(
        "--agent", action="store_true", help="let the model call tools (search, GPU lookup)"
    )

    p_serve = sub.add_parser("serve", help="run the HTTP API")
    p_serve.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)
    settings = config.load()

    if args.command == "fetch-docs":
        from rocm_agent.fetch import fetch_docs

        fetch_docs(settings, force=args.force)
    elif args.command == "clean-docs":
        from rocm_agent.clean import clean_docs, print_report

        print_report(clean_docs(settings))
    elif args.command == "chunk":
        from rocm_agent.chunking import print_stats, write_chunks

        print_stats(write_chunks(settings))
    elif args.command == "index":
        from rocm_agent.retrieval import build_index

        info = build_index(settings)
        print(f"Indexed {info['chunks']} chunks with {info['embedding_model']}")
        print(f"  longer than the model's {info['max_tokens']}-token input: {info['truncated']}")
        print(f"Saved: {info['saved_to']}")
    elif args.command == "eval":
        from functools import partial

        from rocm_agent.evals import print_summary, run
        from rocm_agent.retrieval import describe, retrieve

        mode = args.mode or settings.retrieval.mode
        result = run(
            settings.evals_dir,
            k=args.k,
            retrieve=partial(retrieve, mode=mode),
            label=args.label,
            meta=describe(settings, mode),
        )
        print_summary(result)
    elif args.command == "search":
        from rocm_agent.retrieval import retrieve

        for n, c in enumerate(retrieve(args.question, args.k, mode=args.mode), 1):
            print(f"[{n}] {c['path']}  ({c['heading_path']})  ranks={c['ranks']}")
            print("    " + c["text"][:300].replace("\n", "\n    "))
    elif args.command == "ask":
        from rocm_agent.answer import answer_question, format_answer

        if args.agent:
            from rocm_agent.agent import run_agent

            result = run_agent(args.question, settings=settings)
        else:
            result = answer_question(args.question, args.k, args.mode, settings=settings)
        if args.json:
            print(result.model_dump_json(indent=2))
        else:
            print(format_answer(result))
            if args.agent:
                print("Tools: " + (", ".join(result.tool_calls) or "none"))
    elif args.command == "serve":
        import uvicorn

        uvicorn.run("rocm_agent.api:app", host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
