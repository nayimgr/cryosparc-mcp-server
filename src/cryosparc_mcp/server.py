"""CryoSPARC MCP Server — main entry point (cryosparc-tools v5).

Usage (stdio, default for Claude Code):
    python -m cryosparc_mcp.server

Usage (HTTP, local only by default):
    python -m cryosparc_mcp.server --transport streamable-http --port 8200

The HTTP transport is UNAUTHENTICATED and carries your CryoSPARC credentials,
so it binds to 127.0.0.1 unless you explicitly pass --host. Prefer an SSH
tunnel over exposing the port.
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastmcp import FastMCP

from .connection import describe_client, get_client, reset_client


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[dict]:
    """Startup: try to connect to CryoSPARC. Shutdown: reset the client."""
    try:
        cs = get_client()
        print(f"[cryosparc-mcp] Connected to CryoSPARC at {describe_client(cs)}", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001
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

    # MCP Tasks extension (io.modelcontextprotocol/tasks, spec 2026-07-28).
    # Required because cryosparc_wait_for_done / wait_for_status /
    # generate_live_report are declared with task=TASK_MODE.
    from .helpers import TASKS_AVAILABLE

    if TASKS_AVAILABLE:
        from fastmcp_tasks import TasksExtension

        mcp.add_extension(TasksExtension())

    from .tools import register_all_tools
    from .tools_extra import register_extra_tools
    from .tools_report import register_report_tools
    from .tools_results import register_results_tools

    register_all_tools(mcp)
    register_extra_tools(mcp)
    register_report_tools(mcp)
    register_results_tools(mcp)

    return mcp


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="CryoSPARC MCP Server (cryosparc-tools v5)")
    parser.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    parser.add_argument("--port", type=int, default=8200)
    parser.add_argument(
        "--host", type=str, default="127.0.0.1",
        help="Host to bind in HTTP mode. Defaults to 127.0.0.1; binding to 0.0.0.0 "
             "exposes an UNAUTHENTICATED server that drives your CryoSPARC instance.",
    )
    args = parser.parse_args()

    server = create_server()

    if args.transport == "streamable-http":
        if args.host not in ("127.0.0.1", "localhost", "::1"):
            print(
                f"[cryosparc-mcp] WARNING: binding to {args.host} exposes an "
                f"UNAUTHENTICATED server that can drive CryoSPARC with the credentials "
                f"in your .env. Use a firewall or an SSH tunnel instead.",
                file=sys.stderr, flush=True,
            )
        # FastMCP 4 forwards extra keyword arguments to the transport.
        print(f"[cryosparc-mcp] HTTP mode on {args.host}:{args.port}", file=sys.stderr)
        server.run(transport="streamable-http", host=args.host, port=args.port)
    else:
        # stdio: keep stdout clean for the JSON-RPC stream.
        server.run(transport="stdio", show_banner=False)


if __name__ == "__main__":
    main()
