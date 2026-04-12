#!/usr/bin/env bash
# CryoSPARC MCP Server launcher — activates venv and starts the server.
# Usage:  ./start_server.sh                                    (stdio mode)
#         ./start_server.sh --transport streamable-http --port 8200  (HTTP mode)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$SCRIPT_DIR/.venv/bin/activate" ] && source "$SCRIPT_DIR/.venv/bin/activate"
export PYTHONPATH="$SCRIPT_DIR/src:$PYTHONPATH"
exec python3 -m cryosparc_mcp.server "$@"
