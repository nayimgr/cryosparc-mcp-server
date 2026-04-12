# CryoSPARC MCP Server – CryoSPARC v4.7.1 Edition

MCP server for **CryoSPARC v4.7.1** lets Claude Code (or any MCP client) control a CryoSPARC instance through `cryosparc-tools` (v4.7.1).

```
┌─────────────┐     stdio / HTTP      ┌──────────────────┐     REST API     ┌──────────────┐
│ Claude Code  │◄─────────────────────►│  cryosparc-mcp   │◄───────────────►│  CryoSPARC   │
│   (client)   │        MCP            │   (this server)   │  cryosparc-tools│   instance   │
└─────────────┘                        └──────────────────┘                  └──────────────┘
```

## 34 MCP Tools

### Navigation (read-only)
| Tool | Description |
|------|-------------|
| `cryosparc_test_connection` | Verify connection to CryoSPARC |
| `cryosparc_list_projects` | List all projects |
| `cryosparc_find_project` | Get project details (UID, title, directory) |
| `cryosparc_list_workspaces` | List workspaces in a project |
| `cryosparc_list_jobs` | List jobs, filterable by workspace / type |
| `cryosparc_find_job` | Get job details (params, inputs, outputs, status) |
| `cryosparc_get_job_spec` | Get parameter / input / output specification |
| `cryosparc_list_job_types` | List all job types available on the instance |
| `cryosparc_get_lanes` | List scheduler lanes |
| `cryosparc_get_targets` | List GPU targets |

### Job Management
| Tool | Description |
|------|-------------|
| `cryosparc_create_workspace` | Create a new workspace |
| `cryosparc_create_job` | Create a job with type, params and connections |
| `cryosparc_set_param` | Set a single parameter |
| `cryosparc_set_params_batch` | Set multiple parameters at once |
| `cryosparc_connect` | Connect an output → input between jobs |
| `cryosparc_queue_job` | Queue a job for execution |
| `cryosparc_wait_for_done` | Wait for job completion |
| `cryosparc_wait_for_status` | Wait for a specific status (e.g. 'waiting') |
| `cryosparc_interact` | Interactive actions (Curate, Inspect Picks, Select 2D) |
| `cryosparc_kill_job` | Kill a running job |
| `cryosparc_clear_job` | Reset a job to 'building' status |

### Data Inspection
| Tool | Description |
|------|-------------|
| `cryosparc_dataset_summary` | Get dataset schema (columns, types, row count) |
| `cryosparc_load_output` | Load paginated data with slot filtering |
| `cryosparc_list_files` | List files in a project or job directory |
| `cryosparc_list_assets` | List image/plot assets for a job |

### External Jobs
| Tool | Description |
|------|-------------|
| `cryosparc_create_external_job` | Create an External Job |
| `cryosparc_external_connect` | Connect with slot filtering |
| `cryosparc_external_add_output` | Add output with pre-allocation |
| `cryosparc_external_start` | Start the external job |
| `cryosparc_external_save_output` | Persist output data |
| `cryosparc_external_stop` | Mark as completed |
| `cryosparc_external_load_input` | Load data from a connected input |

### Workflow Helpers
| Tool | Description |
|------|-------------|
| `cryosparc_workflow_import_movies` | Import movies in one step |
| `cryosparc_workflow_preprocess` | Motion Correction + CTF Estimation |

## Installation

```bash
git clone https://github.com/kdursunnizam-art/cryopsarc-mcp-server.git
cd ~/cryosparc-mcp-server
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
chmod +x start_server.sh
```

> **Note:** `pip install -e .` installs `cryosparc-tools`, which requires a C compiler.
> If you get a build error: `sudo apt install build-essential python3-dev`

### CryoSPARC Authentication (v4.7.1)

This server works with CryoSPARC v4.7.1 using host + base_port + email/password + license.
Create a .env file in the project root (or set environment variables):

```bash
# .env file
CRYOSPARC_HOST=your.cryosparc.host
CRYOSPARC_BASE_PORT=61000
CRYOSPARC_EMAIL=your_email@example.com
CRYOSPARC_PASSWORD=your_password
CRYOSPARC_LICENSE=your-license-uuid
CRYOSPARC_TIMEOUT=300
```

The CRYOSPARC_LICENSE is mandatory for v4.7.1. You can find it in your CryoSPARC license file or web UI.


## Usage

### With Claude Code (using `claude mcp` command, no `claude.json` manual configuration)

#### For stdio (local use)

From your terminal, run this command:

```bash
claude mcp add-json cryosparc '{"command":"/path/to/cryosparc-mcp-server/start_server.sh","args":[],"env":{}}' --scope user
```

credentials are read from .env.


```

Verify it's registered:

```bash
claude mcp list
```

To remove and reconfigure:

```bash
claude mcp remove cryosparc
```

#### For http use (remote) EXPERIMENTAL

1. Start the server manually in a terminal

```bash
cd /path/to/cryosparc-mcp-server
source .venv/bin/activate
./start_server.sh --transport streamable-http --port 8200 --host 0.0.0.0 
```

Keep this terminal open. (It is preferable to change --host 0.0.0.0 to your own IP address) The server will read credentials from .env

2. Add the server to Claude Code

```bash
claude mcp add --transport http cryosparc http://YOUR.IP.ADDRESS:8200/mcp --scope user
```

3. Verify

```bash
claude mcp list
```

Should show cryosparc with transport streamable-http and URL http://YOUR.IP.ADDRESS:8200/mcp.

4. Restart Claude Code and test.

Then in Claude Code:

```
> Use cryosparc_test_connection to check we're connected
> List available projects with cryosparc_list_projects
> Show me all jobs in project P1

```

### With OpenClaw and Claude Desktop

#### For stdio (local use)

From your terminal, run this command:

```bash
openclaw mcp add --transport stdio --scope user cryosparc --cmd /path/to/cryosparc-mcp-server/start.server.sh 
```

Verify it's registered:

```bash
openclaw mcp list
```
or edit manually claude_desktop_config.json (mandatory for Claude Dektop and only in stdio)

claude_desktop_config.json
Windows : %APPDATA%\Claude\claude_desktop_config.json
macOS : ~/Library/Application Support/Claude/claude_desktop_config.json
```json
{
  "mcpServers": {
    "cryosparc": {
      "command": "/path/to/start_server.sh",
      "args": [],
    }
  }
}
```


#### For http (remote use)

Start the server in HTTP mode:

```bash
cd /path/to/cryosparc-mcp-server
source .venv/bin/activate
./start_server.sh --transport streamable-http --port 8200 --host 0.0.0.0
```
Keep this terminal open. (It is preferable to change --host 0.0.0.0 to your own IP address)

Configure using openclaw mcp command:

```bash
openclaw mcp add --transport http --scope user cryosparc http://YOUR.IP.ADDRESS:8200/mcp
```

Or configure manually in `openclaw.json` (both stdio and http):

```json
"skills": {
  "install": {
    "nodeManager": "npm"
  },
  "entries": {
    "mcp-integration": {
      "enabled": true,
      "config": {
        "servers": [
          {
            "name": "cryosparc-stdio",
            "transport": "stdio",
            "command": "/path/to/cryosparc-mcp-server/server.py",
            "args": [],
            "env": {
              "CRYOSPARC_BASE_URL": "http://localhost:39000"
            }
          },
          {
            "name": "cryosparc-http",
            "transport": "streamable-http",
            "url": "http://YOUR.IP.ADDRESS:8200/mcp"
          }
        ],
        "toolPrefix": true
      }
    }
  }
}
```

## Project Structure

```
cryosparc-mcp-server/
├── start_server.sh          # Launcher (activates venv + starts server)
├── pyproject.toml
├── README.md
├── .env.example             # Template for credentials
└── src/cryosparc_mcp/
    ├── __init__.py
    ├── __main__.py
    ├── connection.py        # CryoSPARC connection (v4.7.1)
    ├── helpers.py           # JSON serialisation, error formatting
    ├── server.py            # FastMCP server + lifespan
    └── tools.py             # All 34 MCP tools
```

## Compatibility

- **CryoSPARC**: v4.7.1 (tested; may work with other v4.x but not v5.x)
- **cryosparc-tools**: vv4.7.1
- **Python**: 3.10+

## License

MIT
