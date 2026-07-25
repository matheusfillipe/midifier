"""Run the HTTP API, or the MCP server, from one entry point."""

from __future__ import annotations

import argparse

from midifier.config import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(prog="midifier")
    parser.add_argument("surface", choices=("api", "mcp"), nargs="?", default="api")
    args = parser.parse_args()
    settings = get_settings()

    if args.surface == "mcp":
        from midifier.mcp import mcp

        mcp.run()
        return

    import uvicorn

    uvicorn.run(
        "midifier.api:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
    )


if __name__ == "__main__":
    main()
