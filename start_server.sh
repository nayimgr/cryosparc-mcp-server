#!/usr/bin/env bash
# CryoSPARC MCP Server launcher — activates venv and starts the server.
# Usage:  ./start_server.sh                                    (stdio mode)
#         ./start_server.sh --transport streamable-http --port 8200   (HTTP mode, binds 127.0.0.1)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate virtual environment if it exists in the project directory
if [ -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
fi

export PYTHONPATH="$SCRIPT_DIR/src:$PYTHONPATH"
exec python3 -m cryosparc_mcp.server "$@"
