"""Command line entry point: `rocm-agent <command>`."""

import argparse

from rocm_agent import config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rocm-agent")
    sub = parser.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser("fetch-docs", help="download the docs named in config.toml")
    p_fetch.add_argument("--force", action="store_true", help="download again even if present")

    sub.add_parser("clean-docs", help="convert the downloaded docs to clean Markdown")

    p_eval = sub.add_parser("eval", help="run the eval set and print every reported number")
    p_eval.add_argument("--k", type=int, default=5, help="chunks retrieved per question")
    p_eval.add_argument("--label", default="", help="name for this run, e.g. baseline or option-d")

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
    elif args.command == "eval":
        from rocm_agent.evals import print_summary, run

        print_summary(run(settings.evals_dir, k=args.k, label=args.label))
    elif args.command == "serve":
        import uvicorn

        uvicorn.run("rocm_agent.api:app", host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
