# Headless Colab MCP Server

`colab-mcp` is a FastMCP server for controlling an active Google Colab notebook without Playwright, DOM scraping, or browser automation.

Instead of launching a visible browser or trying to automate the Colab UI, this server establishes a secure local WebSocket proxy that an already-authenticated Colab tab can attach to. Once that tunnel is live, MCP tools can write code cells, run them, read structured execution results, and drive machine learning workflows entirely through the Colab kernel and proxy APIs.

## What This Project Does

- Exposes a local MCP server through `colab-mcp`
- Connects to an active Colab browser tab through a tokenized localhost WebSocket proxy
- Runs notebook operations headlessly after the tunnel is established
- Avoids Playwright, Chromium, DOM scraping, and browser-profile management
- Provides ML-oriented tools for workspace setup, dataset download, and pipeline execution

## Architecture

This project has two cooperating layers:

1. `ColabSessionProxy`
   - Starts a localhost WebSocket server
   - Generates a one-time connection URL containing `mcpProxyToken` and `mcpProxyPort`
   - Waits for an already-authenticated Colab tab to attach

2. `NotebookController`
   - Exposes the stable MCP tool surface
   - Discovers proxy capabilities from the connected Colab frontend
   - Maps server-owned tools to proxy-backed cell operations
   - Prefers proxy execution for notebook-visible work
   - Falls back to direct runtime execution only when explicitly needed

The result is a headless architecture:

- No Playwright
- No Chromium install
- No DOM scraping
- No standalone OAuth flow required for proxy-backed notebook execution

## Headless Connection Model

The browser does not auto-discover the proxy. A human performs one short handshake step:

1. Start the MCP server.
2. Call `connect_colab`.
3. The server returns a `connect_url`, `proxy_token`, and `proxy_port`.
4. Paste the returned `connect_url` into an existing Colab tab, or append its fragment to your current notebook URL.
5. Once the tab connects, the rest of the workflow is headless and API-driven.

This design keeps authentication inside the user’s existing Colab session while still allowing autonomous tool execution afterward.

## Tool Surface

### Core Notebook Tools

- `connect_colab(notebook_url: str | None = None)`
- `list_colab_cells()`
- `read_colab_cell(cell_id: str)`
- `write_colab_cell(code: str, cell_id: str | None = None, mode: "append" | "replace" = "append")`
- `run_colab_cell(cell_id: str | None = None, wait: bool = True, timeout_seconds: int = 120)`
- `run_colab_code(code: str, mode: "append" | "replace" = "append", wait: bool = True, timeout_seconds: int = 120)`
- `get_colab_output(cell_id: str | None = None)`
- `save_colab_notebook()`
- `run_runtime_code(code: str)`

### ML Workflow Tools

- `setup_ml_workspace(packages: list[str])`
  - Installs packages in the attached Colab environment
  - Creates `/content/data/scour`
  - Creates `/content/data/concrete`

- `fetch_remote_dataset(download_url: str, extract_to: str)`
  - Downloads CSV or ZIP assets into `/content`
  - Extracts ZIP archives automatically

- `execute_ml_pipeline(code_block: str)`
  - Executes a Python block through the connected Colab session
  - Returns structured stdout, stderr, generated file paths, and error details

## Why This Replaced The Old Browser-Automation Approach

The previous browser-automation model had the usual failure modes:

- visible browser dependency
- flaky DOM selectors
- Colab UI drift
- local browser profile issues
- poor fit for a real MCP server

This rewrite removes those classes of problems by using:

- FastMCP for the server surface
- a localhost WebSocket tunnel for authenticated Colab access
- structured proxy tool calls instead of UI scraping
- direct kernel-style execution payloads for ML workflows

## Installation

### Prerequisites

- Python
- `uv`
- an active Google Colab session in your browser

Install dependencies:

```bash
uv sync
```

Run the server:

```bash
uv run colab-mcp
```

## Example MCP Configuration

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

## End-To-End Flow

1. Start the local MCP server.
2. Call `connect_colab`.
3. Copy the returned `connect_url`.
4. Open that URL in your Colab browser tab.
5. Wait for the proxy to connect.
6. Use `setup_ml_workspace`, `fetch_remote_dataset`, and `execute_ml_pipeline`.
7. Read back structured stdout and artifact paths from the tool results.

## Development

Run tests:

```bash
PYTHONPATH=src py -m pytest
```

Current regression coverage includes:

- proxy capability discovery
- native Colab argument mapping
- real `cellId` extraction
- ML tool routing through the proxy
- execution-result normalization for list-based stream output

## Repository Notes

This repository intentionally reflects the headless MCP architecture. Browser automation dependencies and Playwright-era implementation paths are not part of the supported design.
