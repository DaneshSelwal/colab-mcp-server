# Headless Colab MCP Agent Guide

This repository exposes a headless Google Colab MCP server. Use it when you need an AI agent to control a live Colab notebook through MCP without browser automation.

## Mental Model

There are two distinct phases:

1. Human-assisted tunnel setup
2. Fully headless notebook execution

The human step is only needed to attach an already-authenticated Colab browser tab to the local proxy. After that, the agent should operate strictly through MCP tool calls.

## Never Do This

- Do not use Playwright
- Do not scrape the DOM
- Do not launch Chromium
- Do not ask for a standalone Colab OAuth file when a proxy-connected browser tab is available
- Do not spin up a second temporary proxy server outside the real MCP server process

## Preferred Tool Order

### 1. Establish Proxy Connectivity

Call:

```text
connect_colab()
```

Interpret the result:

- If `proxy_connected` is `false`, the user must complete the browser handshake.
- If `connected` is `true`, notebook tools are available.
- If only `proxy_connected` is `true`, inspect `capabilities` to see whether execution is available.

The response includes:

- `proxy_token`
- `proxy_port`
- `connect_url`

## Human-In-The-Loop Handshake

When `connect_colab()` returns a `connect_url`, tell the user to:

1. Copy the exact `connect_url`
2. Paste it into an already-open Colab browser tab
3. Load or refresh that URL

That is the required manual handshake. A normal notebook refresh without the fragment parameters is not enough.

## After The Tunnel Is Live

Once the Colab tab is attached, stay fully headless and use MCP tools only.

### General Notebook Tools

- `list_colab_cells()`
- `read_colab_cell(cell_id)`
- `write_colab_cell(code, cell_id=None, mode="append")`
- `run_colab_cell(cell_id=None, wait=True, timeout_seconds=120)`
- `run_colab_code(code, mode="append", wait=True, timeout_seconds=120)`
- `get_colab_output(cell_id=None)`
- `save_colab_notebook()`

### ML Tools

- `setup_ml_workspace(packages)`
- `fetch_remote_dataset(download_url, extract_to)`
- `execute_ml_pipeline(code_block)`

## ML Workflow Pattern

Use this sequence:

1. `connect_colab()`
2. `setup_ml_workspace([...])`
3. `fetch_remote_dataset(...)`
4. `execute_ml_pipeline(...)`

The ML tools are proxy-first. If the connected Colab proxy exposes execution capability, they route code through the notebook-backed proxy path instead of the standalone runtime OAuth path.

## Result Handling

`execute_ml_pipeline` returns structured information:

- `status`
- `stdout`
- `stderr`
- `generated_file_paths`
- `error_name`
- `error_value`
- `traceback`

Always read `stdout` for user-facing results like model metrics. Use `generated_file_paths` to surface saved artifacts such as `.pkl` files or chart images.

## Operational Rules For Agents

- Prefer `run_colab_code` and the ML tools over raw runtime execution when the proxy is connected.
- Use `run_runtime_code` only when the task explicitly needs the direct runtime path.
- Treat real Colab `cellId` values as opaque strings.
- Do not assume cell indices can substitute for `cellId`.
- If a tool returns `connect_url`, stop and ask the human to perform the browser step.
- Once connected, continue autonomously.

## Troubleshooting

### `proxy_not_connected`

The Colab tab is not attached to the local MCP proxy. Re-run `connect_colab()` and have the user open the exact `connect_url`.

### `No cell was found with ID ...`

The agent is likely using a numeric index where Colab expects the browser-generated `cellId`. Re-read the created cell metadata and use the returned alphanumeric ID.

### Missing Tool Arguments Like `cellIndex`, `language`, or `cellId`

The proxy adapter must match the browser-native API exactly:

- `add_code_cell` requires `cellIndex` and `language`
- `run_code_cell` requires `cellId`

### Runtime OAuth Errors

If the proxy is connected, do not route ML execution through the standalone runtime path. Use the proxy-backed execution flow.
