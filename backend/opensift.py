from __future__ import annotations

import argparse
import os
import sys


def run_ui(host: str, port: int, reload: bool) -> None:
    import uvicorn  # type: ignore

    uvicorn.run(
        "ui_app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


def run_terminal(argv: list[str]) -> None:
    from cli_chat import main as cli_main  # type: ignore

    old_argv = sys.argv[:]
    try:
        sys.argv = ["cli_chat.py", *argv]
        cli_main()
    finally:
        sys.argv = old_argv


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenSift Launcher (UI or Terminal)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ui = sub.add_parser("ui", help="Run localhost web UI")
    p_ui.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    p_ui.add_argument("--port", type=int, default=8001, help="Port (default: 8001)")
    p_ui.add_argument("--reload", action="store_true", help="Enable auto-reload")

    p_term = sub.add_parser("terminal", help="Run terminal chatbot")
    p_term.add_argument("--owner", default="default", help="Owner/namespace")
    p_term.add_argument("--mode", default="study_guide", help="Mode")
    p_term.add_argument("--provider", default="claude_code", choices=["openai", "claude", "claude_code"])
    p_term.add_argument("--model", default="", help="Model override (optional)")
    p_term.add_argument("--k", type=int, default=8, help="Top-k retrieval")
    p_term.add_argument("--wrap", type=int, default=100, help="Wrap width")
    p_term.add_argument("--history-turns", type=int, default=10, help="How many messages to include as history")
    p_term.add_argument("--no-stream", action="store_true", help="Disable streaming output")
    p_term.add_argument("--no-sources", action="store_true", help="Disable sources printing")

    args, extras = parser.parse_known_args()

    this_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(this_dir)

    if args.cmd == "ui":
        run_ui(args.host, args.port, args.reload)

    if args.cmd == "terminal":
        forwarded = [
            "--owner",
            args.owner,
            "--mode",
            args.mode,
            "--provider",
            args.provider,
            "--model",
            args.model,
            "--k",
            str(args.k),
            "--wrap",
            str(args.wrap),
            "--history-turns",
            str(args.history_turns),
        ]
        if args.no_stream:
            forwarded.append("--no-stream")
        if args.no_sources:
            forwarded.append("--no-sources")

        run_terminal(forwarded + extras)


if __name__ == "__main__":
    main()