from __future__ import annotations

import argparse
import sys

from graspscope import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="graspscope", description="GraspScope CLI")
    parser.add_argument("--version", action="store_true", help="Print version")
    sub = parser.add_subparsers(dest="cmd")

    p_ui = sub.add_parser("ui", help="Launch the GraspScope dashboard")
    p_ui.add_argument("--host", default="127.0.0.1")
    p_ui.add_argument("--port", type=int, default=8787)
    p_ui.add_argument("--no-open", action="store_true", help="Do not open a browser")
    p_ui.set_defaults(func=_cmd_ui)

    args = parser.parse_args(argv)
    if args.version or args.cmd == "version":
        print(__version__)
        return 0
    if not args.cmd:
        parser.print_help()
        return 2
    return int(args.func(args))


def _cmd_ui(args: argparse.Namespace) -> int:
    import webbrowser

    try:
        import uvicorn
    except ImportError:
        print(
            'ERROR: UI extras missing. Install with: pip install -e ".[ui]"',
            file=sys.stderr,
        )
        return 2

    from graspscope.ui.app import create_app

    url = f"http://{args.host}:{args.port}/"
    print(f"GraspScope UI → {url}")
    if not args.no_open:
        webbrowser.open(url)
    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
