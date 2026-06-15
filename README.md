# Headless Colab MCP Server

`colab-mcp` is a FastMCP server for controlling Google Colab notebooks through a secure, headless WebSocket architecture. Agents can create, edit, run, and inspect notebook cells without browser automation or UI scraping.

---

## Features

- **Headless Operation** — Run notebook operations through a secure WebSocket proxy
- **Zero Browser Management** — No Chromium, browser profiles, or DOM scraping
- **ML-Ready Tooling** — Workspace setup, dataset handling, and pipeline execution
- **Structured Results** — Typed stdout, stderr, file paths, and error details
- **FastMCP Integration** — Clean MCP server interface for tool composition

---

## Architecture

The server uses two cooperating layers:

### ColabSessionProxy

- Starts a localhost WebSocket server
- Generates a one-time connection URL with `mcpProxyToken` and `mcpProxyPort`
- Waits for an authenticated Colab tab to attach

### NotebookController

- Exposes the stable MCP tool surface
- Discovers proxy capabilities from the connected Colab frontend
- Maps server-owned tools to proxy-backed cell operations
- Falls back to direct runtime execution only when needed

---

## Requirements

| Requirement | Version |
|-------------|---------|
| Python | 3.13+ |
| uv | Latest |
| Google Colab | Active browser session |

---

## Installation

```bash
uv sync
uv run colab-mcp
```

---

## Configuration

Add this to your MCP configuration:

```json
{
  "mcpServers": {
    "colab-mcp-local": {
      "command": "uv",
      "args": ["run", "colab-mcp"],
      "cwd": "${workspaceFolder}",
      "timeout": 30000
    }
  }
}
```

---

## API Reference

### Core Notebook Tools

| Tool | Description |
|------|-------------|
| `connect_colab(notebook_url?)` | Initialize connection and retrieve proxy URL |
| `list_colab_cells()` | List all cells in the notebook |
| `read_colab_cell(cell_id)` | Read a specific cell |
| `write_colab_cell(code, cell_id?, mode?)` | Write code to a cell |
| `run_colab_cell(cell_id?, wait?, timeout_seconds?)` | Execute a cell |
| `run_colab_code(code, mode?, wait?, timeout_seconds?)` | Write and execute code in one step |
| `get_colab_output(cell_id?)` | Retrieve execution output |
| `save_colab_notebook()` | Save the notebook |
| `run_runtime_code(code)` | Execute code directly in the runtime |

### ML Workflow Tools

| Tool | Description |
|------|-------------|
| `setup_ml_workspace(packages)` | Install packages and create standard data directories |
| `fetch_remote_dataset(download_url, extract_to)` | Download and extract datasets |
| `execute_ml_pipeline(code_block)` | Execute Python blocks with structured results |

---

## Usage

1. Start the MCP server.
2. Call `connect_colab` to get a `connect_url`, `proxy_token`, and `proxy_port`.
3. Paste the `connect_url` into an active Colab tab.
4. Wait for the proxy connection to establish.
5. Run notebook operations through the MCP tools.

### Example

```text
uv run colab-mcp
connect_colab()
setup_ml_workspace(["pandas", "scikit-learn"])
fetch_remote_dataset(url, "/content/data")
execute_ml_pipeline(training_code)
get_colab_output()
```

---

## Development & Verification

```bash
PYTHONPATH=src python scripts/smoke_test.py
PYTHONPATH=src py -m pytest
cat RELEASE_CHECKLIST.md
```

### Test coverage

- Proxy capability discovery
- Native Colab argument mapping
- Cell ID extraction
- ML tool routing through proxy
- Execution result normalization

---

## License

This project is licensed under the [Apache License 2.0](LICENSE).

---

## Acknowledgments

This headless WebSocket proxy architecture was inspired by the open-source work provided by the Google Colab team.
