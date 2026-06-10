"""CryoSPARC MCP Server — main entry point.

Usage (stdio, default for Claude Code):
    python -m cryosparc_mcp.server

Usage (HTTP for remote access):
    python -m cryosparc_mcp.server --transport streamable-http --port 8200 --host 0.0.0.0
"""
from __future__ import annotations

import argparse
import sys
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from mcp.server.fastmcp import FastMCP

from .connection import get_client, reset_client


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[dict]:
    """Startup: try to connect to CryoSPARC. Shutdown: reset client."""
    try:
        cs = get_client()
        # On utilise base_url si disponible, sinon on affiche host/port
        url = getattr(cs, 'base_url', None) or f"{getattr(cs, 'host', 'unknown')}:{getattr(cs, 'base_port', 'unknown')}"
        print(f"[cryosparc-mcp] Connected to CryoSPARC at {url}", file=sys.stderr)
    except Exception as exc:
        print(
            f"[cryosparc-mcp] WARNING: {exc}\n"
            "[cryosparc-mcp] Tools will attempt to connect on first use.",
            file=sys.stderr,
        )
    try:
        yield {}
    finally:
        reset_client()


def create_server() -> FastMCP:
    """Build the fully-configured MCP server."""
    mcp = FastMCP("cryosparc_mcp", lifespan=app_lifespan)

    from .tools import register_all_tools
    from .tools_extra import register_extra_tools
    from .tools_report import register_report_tools
    register_all_tools(mcp)
    register_extra_tools(mcp)
    register_report_tools(mcp)

    return mcp


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="CryoSPARC MCP Server")
    parser.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    parser.add_argument("--port", type=int, default=8200)
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to in HTTP mode")
    args = parser.parse_args()

    server = create_server()

    if args.transport == "streamable-http":
        print(f"[cryosparc-mcp] HTTP mode on {args.host}:{args.port}", file=sys.stderr)
        server.run(transport="streamable-http", host=args.host, port=args.port)
    else:
        server.run(transport="stdio")


if __name__ == "__main__":
    main()