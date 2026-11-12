"""Command line entry point: `rocm-agent <command>`."""

import argparse

from rocm_agent import config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rocm-agent")
    sub = parser.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser("fetch-docs", help="download the docs named in config.toml")
    p_fetch.add_argument("--force", action="store_true", help="download again even if present")

    sub.add_parser("eval", help="run the eval set and print every reported number")

    p_serve = sub.add_parser("serve", help="run the HTTP API")
    p_serve.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)
    settings = config.load()

    if args.command == "fetch-docs":
        from rocm_agent.fetch import fetch_docs

        fetch_docs(settings, force=args.force)
    elif args.command == "eval":
        raise SystemExit("Not built yet: the eval runner arrives on Friday (see LEARNING.md).")
    elif args.command == "serve":
        import uvicorn

        uvicorn.run("rocm_agent.api:app", host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
