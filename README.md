# CryoSPARC MCP Server – CryoSPARC v5 Edition

MCP server for **CryoSPARC v5** lets Claude Code (or any MCP client) control a CryoSPARC instance through `cryosparc-tools` (v5.x).

```
┌─────────────┐     stdio / HTTP      ┌──────────────────┐     REST API     ┌──────────────┐
│ Claude Code  │◄─────────────────────►│  cryosparc-mcp   │◄───────────────►│  CryoSPARC   │
│   (client)   │        MCP            │   (this server)   │  cryosparc-tools│   instance   │
└─────────────┘                        └──────────────────┘                  └──────────────┘
```


## 41 MCP Tools

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

### Data Inspection & Download
| Tool | Description |
|------|-------------|
| `cryosparc_dataset_summary` | Get dataset schema (columns, types, row count) |
| `cryosparc_load_output` | Load paginated data with slot filtering |
| `cryosparc_list_files` | List files in a project or job directory |
| `cryosparc_list_assets` | List image/plot assets for a job |
| `cryosparc_download_mrc` | Download a .mrc file as NumPy array (header + shape) |
| `cryosparc_download_file` | Download any file from the project to local disk |

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

### Workflow & Parameter Helpers
| Tool | Description |
|------|-------------|
| `cryosparc_workflow_import_movies` | Import movies in one step |
| `cryosparc_workflow_preprocess` | Motion Correction + CTF Estimation |
| `cryosparc_parse_workflow` | Parse a CryoSPARC workflow JSON and extract its structure |
| `cryosparc_suggest_params` | Retrieve reference parameters for common target types (e.g., GPCR) |
| `cryosparc_generate_live_report` | PDF report on all completed jobs inside a workspace saved in `/tmp/cryosparc_report_<project_uid>_<workspace_uid>.pdf` |

### Results & Decision Tools
| Tool | Description |
|------|-------------|
| `cryosparc_job_results` | Key metrics of a finished job: **per-class particle distribution** (2D/3D), CTF defocus/astigmatism/fit-resolution stats, motion stats, pick scores, particle counts. Use this instead of hand-parsing datasets |
| `cryosparc_render_class_averages` | Renders 2D class averages as an **inline PNG montage** so a vision-capable agent can see which classes are good |

## Installation

```bash
git clone https://github.com/kdursunnizam-art/cryosparc-mcp-server.git
cd cryosparc-mcp-server
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[report]'   # 'report' adds matplotlib/numpy/pillow for the PDF + image tools
chmod +x start_server.sh
```

> **Note:** `pip install -e .` installs `cryosparc-tools`, which requires a C compiler.
> If you get a build error: `sudo apt install build-essential python3-dev`

### CryoSPARC Authentication (v5)

Create a .env file in the project root (or set environment variables). Use **either**
`CRYOSPARC_HOST` + `CRYOSPARC_BASE_PORT`, **or** `CRYOSPARC_BASE_URL`:

```bash
# .env file
CRYOSPARC_HOST=your.cryosparc.host
CRYOSPARC_BASE_PORT=61000
CRYOSPARC_EMAIL=your_email@example.com
CRYOSPARC_PASSWORD=your_password
CRYOSPARC_LICENSE=your-license-uuid
CRYOSPARC_TIMEOUT=300

# Alternative to HOST+BASE_PORT:
# CRYOSPARC_BASE_URL=http://localhost:39000
```

`CRYOSPARC_LICENSE` is required by most instances — find it in your CryoSPARC license file or web UI.


## Usage

### With Claude Code (using `claude mcp` command, no `claude.json` manual configuration)

#### For stdio (local use)

From your terminal, run this command:

```bash
claude mcp add-json cryosparc '{"command":"/path/to/cryosparc-mcp-server/start_server.sh","args":[],"env":{}}' --scope user
```

credentials are read from .env.

Verify it's registered:

```bash
claude mcp list
```

To remove and reconfigure:

```bash
claude mcp remove cryosparc
```

#### For http use (remote) EXPERIMENTAL

> ⚠️ **Security warning:** The HTTP transport has **no authentication**. Anyone who can
> reach the port drives your CryoSPARC instance with the credentials stored in `.env`,
> and can read/write files in your project directories via the data tools. Do **not**
> bind to `0.0.0.0` on an untrusted network. Prefer binding to `127.0.0.1` and reaching
> it through an SSH tunnel (`ssh -L 8200:localhost:8200 user@host`), or restrict access
> with a firewall / reverse proxy that adds authentication.

1. Start the server manually in a terminal

```bash
cd /path/to/cryosparc-mcp-server
source .venv/bin/activate
./start_server.sh --transport streamable-http --port 8200
```

Keep this terminal open. The server binds to 127.0.0.1 by default; reach it from another machine with an SSH tunnel (`ssh -L 8200:localhost:8200 user@host`). Credentials are read from .env.

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
openclaw mcp add --transport stdio --scope user cryosparc --cmd /path/to/cryosparc-mcp-server/start_server.sh 
```

Verify it's registered:

```bash
openclaw mcp list
```

edit manually claude_desktop_config.json (mandatory for Claude Dektop and only in stdio)

claude_desktop_config.json
Windows : %APPDATA%\Claude\claude_desktop_config.json
macOS : ~/Library/Application Support/Claude/claude_desktop_config.json

```json
{
  "mcpServers": {
    "cryosparc": {
      "command": "/path/to/start_server.sh",
      "args": []
    }
  }
}
```


#### For http (remote use) (EXPERIMENTAL For Claude Desktop)

Start the server in HTTP mode:

```bash
cd /path/to/cryosparc-mcp-server
source .venv/bin/activate
./start_server.sh --transport streamable-http --port 8200
```
Keep this terminal open. The server binds to 127.0.0.1 by default; use an SSH tunnel for remote access.

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
├── .env.example             # Template for credentials change the name after edition
└── src/cryosparc_mcp/
    ├── __init__.py
    ├── __main__.py
    ├── connection.py        # CryoSPARC connection (v5: host+port or base_url, + license)
    ├── helpers.py           # JSON serialisation, MCP errors, v5 adapters
    ├── server.py            # FastMCP server + lifespan
    ├── tools.py             # Navigation, job management, data inspection (29 tools)
    ├── tools_extra.py       # External Jobs + workflow helpers (9 tools)
    ├── tools_report.py      # PDF report generator (1 tool)
    └── tools_results.py     # Results interpretation + class-average rendering (2 tools)
tests/
└── test_cryosparc_mcp.py    # Offline tests against a mocked CryoSPARC client
```

## MCP conformance (specification 2026-07-28)

Built on **FastMCP 4.x** (`fastmcp[tasks]>=4.0`), which targets the current **MCP
specification 2026-07-28** (stateless core, extensions framework).

- Every tool declares a human-readable **`title`** alongside its machine `name`.
- **Structured output:** 40 of the 41 tools declare an `outputSchema` and return
  `structuredContent`, so an agent reads fields instead of re-parsing JSON text.
  (The exception is `cryosparc_render_class_averages`, which returns an image.)
- **`openWorldHint: true`** on every tool — they all reach a remote CryoSPARC instance.
- **MCP Tasks extension** (`io.modelcontextprotocol/tasks`) on the long-running tools —
  `cryosparc_wait_for_done`, `cryosparc_wait_for_status` and
  `cryosparc_generate_live_report`. The client receives a durable task handle and polls
  `tasks/get` instead of holding a request open for hours. Requires the `[tasks]` extra;
  without it those tools degrade to ordinary blocking calls.
- **Errors are real MCP errors** (`isError`), never success payloads containing "Error".
- Transport is **Streamable HTTP** (HTTP+SSE is deprecated), and no deprecated MCP
  feature (Roots, Sampling, Logging) is used.

## Compatibility

- **MCP specification**: 2026-07-28 (via FastMCP 4.x)
- **CryoSPARC**: v5.x
- **cryosparc-tools**: `>=5.0,<6` — this server uses the v5 API (`find_projects`,
  `find_jobs`, `job.params/.inputs/.outputs`, `job_register`), which does not exist in 4.x
- **FastMCP**: `fastmcp[tasks]>=4.0`
- **Python**: 3.10+
- Optional extra `[report]`: matplotlib, numpy, pillow

> **Note on tomography:** CryoSPARC has no native tomography / subtomogram-averaging
> pipeline (the official job-type catalogue and the v5.0 release notes contain no
> tomography jobs), so this server covers single-particle analysis only.

## Changelog

### 0.5.0 — MCP specification 2026-07-28
- **Migrated from the official `mcp` SDK v1 to FastMCP 4.x**, moving the server from
  protocol 2025-11-25 to **2026-07-28**. (`mcp` 2.x renamed `FastMCP` to `MCPServer`
  and does not expose the tasks extension on its decorator.)
- **Tool `title`** on all 41 tools; **`openWorldHint: true`** on all 41.
- **Structured output**: tools now return dictionaries, so the MCP layer emits
  `structuredContent` with a generated `outputSchema` (40 tools).
- **MCP Tasks extension** enabled for `cryosparc_wait_for_done`,
  `cryosparc_wait_for_status` and `cryosparc_generate_live_report`.
- HTTP host/port now passed as FastMCP 4 transport kwargs; stdio runs with the
  banner suppressed to keep the JSON-RPC stream clean.

### 0.4.0 — CryoSPARC v5 migration
- **Migrated to the cryosparc-tools v5 API.** The code previously called v5-only methods
  (`find_projects`, `find_workspaces`, `find_jobs`, `job_register`,
  `job.params/.title/.outputs/.full_spec`) while pinning `cryosparc-tools==4.7.1`, where
  none of them exist — so most navigation, inspection and reporting tools could not work.
- **Fixed `Dataset.columns()` → `fields()`** (`columns()` exists in neither 4.x nor 5.x).
- **Fixed HTTP mode**, which raised `TypeError`: `FastMCP.run()` accepts only
  `(transport, mount_path)`, so host/port now go through `server.settings`.
- **`--host` now defaults to `127.0.0.1`** instead of `0.0.0.0` (the old default
  contradicted the security warning below).
- **Tool failures are now real MCP errors** (`isError`) instead of success payloads
  containing the word "Error", which an agent could not distinguish from a valid result.
- **The PDF report no longer claims success when every job was skipped**, and reports
  per-job skip reasons.
- **Report performance:** sample micrographs are chosen *before* downloading, instead of
  downloading every micrograph in the job to render ~10 of them.
- **Fixed `get_job_spec`**, which read a non-existent `value` field instead of `default`;
  it now also returns descriptions, allowed values and the current value.
- **New:** `cryosparc_job_results` and `cryosparc_render_class_averages`.
- **New:** offline test suite against a mocked CryoSPARC client.

## License

MIT
