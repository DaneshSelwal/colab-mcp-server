#!/usr/bin/env python3
"""Lightweight smoke test for colab-mcp.

Run from the repository root:

    PYTHONPATH=src python scripts/smoke_test.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from colab_mcp import runtime
from colab_mcp.models import ColabExecutionResult, MLPipelineResult
from colab_mcp.notebook_control import NotebookController


def check_execution_result_normalization() -> None:
    result = ColabExecutionResult.from_outputs(
        [
            {"output_type": "stream", "name": "stdout", "text": "hello\n"},
            {"output_type": "stream", "name": "stderr", "text": "warn\n"},
        ]
    )
    assert result.stdout == "hello\n"
    assert result.stderr == "warn\n"


def check_runtime_code_builders() -> None:
    tool = runtime.ColabRuntimeTool(
        client_oauth_config="colab-mcp-oauth-config.json",
        token_path=str(ROOT / ".smoke-test-token.json"),
    )
    setup_code = tool.build_setup_ml_workspace_code(["pandas", "numpy"])
    assert "pandas" in setup_code
    assert "numpy" in setup_code
    assert "/content/data/scour" in setup_code

    pipeline_code = tool.build_execute_ml_pipeline_code("print('training')")
    assert "__COLAB_MCP_ML_PIPELINE__" in pipeline_code
    assert "print('training')" in pipeline_code


async def check_controller_delegation() -> None:
    runtime_tool = mock.Mock()
    runtime_tool.run_runtime_code.return_value = ColabExecutionResult(
        status="ok",
        stdout="runtime\n",
    )
    runtime_tool.build_setup_ml_workspace_code.return_value = "setup-code"
    runtime_tool.build_fetch_remote_dataset_code.return_value = "fetch-code"
    runtime_tool.build_execute_ml_pipeline_code.return_value = "pipeline-code"
    runtime_tool.parse_ml_pipeline_result.return_value = MLPipelineResult(
        status="ok",
        stdout="pipeline\n",
    )

    controller = NotebookController(runtime_tool=runtime_tool)

    setup_result = await controller.setup_ml_workspace(["pandas"])
    fetch_result = await controller.fetch_remote_dataset(
        "https://example.com/data.zip", "/content/data"
    )
    pipeline_result = await controller.execute_ml_pipeline("print('done')")

    runtime_tool.build_setup_ml_workspace_code.assert_called_once_with(["pandas"])
    runtime_tool.build_fetch_remote_dataset_code.assert_called_once_with(
        "https://example.com/data.zip", "/content/data"
    )
    runtime_tool.build_execute_ml_pipeline_code.assert_called_once_with("print('done')")

    assert setup_result.stdout == "runtime\n"
    assert fetch_result.stdout == "runtime\n"
    assert pipeline_result.stdout == "pipeline\n"


def main() -> None:
    check_execution_result_normalization()
    check_runtime_code_builders()
    asyncio.run(check_controller_delegation())
    print("Smoke test passed")


if __name__ == "__main__":
    main()
