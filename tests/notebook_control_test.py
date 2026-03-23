import asyncio
from unittest import mock

import pytest

from colab_mcp.models import (
    CellDetail,
    CellSummary,
    ColabExecutionResult,
    MLPipelineResult,
)
from colab_mcp.notebook_control import (
    NotebookController,
    ProxyNotebookBackend,
    VisibleNotebookUnavailableError,
)


@pytest.fixture
def controller():
    session_proxy = mock.Mock()
    session_proxy.wss = mock.Mock(token="test-token", port=8765)
    session_proxy.proxy_client = mock.Mock()
    session_proxy.is_connected.return_value = False
    session_proxy.build_connect_url.return_value = (
        "https://colab.research.google.com/notebooks/empty.ipynb#mcpProxyToken=test-token&mcpProxyPort=8765"
    )

    runtime_tool = mock.Mock()
    runtime_tool.run_runtime_code.return_value = ColabExecutionResult(
        status="ok", stdout="runtime\n"
    )
    runtime_tool.build_setup_ml_workspace_code.return_value = "setup_code"
    runtime_tool.build_fetch_remote_dataset_code.return_value = "fetch_code"
    runtime_tool.build_execute_ml_pipeline_code.return_value = "pipeline_code"
    runtime_tool.parse_ml_pipeline_result.return_value = MLPipelineResult(
        status="ok", stdout="0.8\n", generated_file_paths=[]
    )

    with mock.patch("colab_mcp.notebook_control.FastMCP"):
        return NotebookController(session_proxy=session_proxy, runtime_tool=runtime_tool)


@pytest.mark.asyncio
async def test_connect_colab_uses_proxy_when_available(controller):
    proxy_backend = mock.Mock()
    proxy_backend.backend_name = "proxy"
    proxy_backend.connect = mock.AsyncMock(
        return_value=[
            "get_output",
            "list_cells",
            "read_cell",
            "run_cell",
            "save_notebook",
            "write_cell",
        ]
    )
    proxy_backend.has_required_capabilities.return_value = True

    controller.session_proxy.is_connected.return_value = True
    controller._make_proxy_backend = mock.Mock(return_value=proxy_backend)

    status = await controller.connect_colab()

    assert status.connected is True
    assert status.backend == "proxy"
    assert status.browser_attached is False


@pytest.mark.asyncio
async def test_connect_colab_reports_missing_proxy(controller):
    controller.session_proxy.is_connected.return_value = False

    status = await controller.connect_colab()

    assert status.connected is False
    assert "run_runtime_code" in status.message


@pytest.mark.asyncio
async def test_connect_colab_reports_missing_capabilities(controller):
    proxy_backend = mock.Mock()
    proxy_backend.backend_name = "proxy"
    proxy_backend.connect = mock.AsyncMock(return_value=["list_cells"])
    proxy_backend.has_required_capabilities.return_value = False

    controller.session_proxy.is_connected.return_value = True
    controller._make_proxy_backend = mock.Mock(return_value=proxy_backend)

    status = await controller.connect_colab()

    assert status.connected is False
    assert "required headless notebook tools" in status.message


@pytest.mark.asyncio
async def test_run_colab_code_falls_back_to_runtime_when_no_proxy(controller):
    result = await controller.run_colab_code("print('hello')")

    assert result.stdout == "runtime\n"
    controller.runtime_tool.run_runtime_code.assert_called_once_with("print('hello')")


@pytest.mark.asyncio
async def test_run_colab_code_writes_then_runs_when_proxy_connected(controller):
    backend = mock.Mock()
    backend.write_cell = mock.AsyncMock(
        return_value=CellDetail(cell_id="cell-1", code="print('hello')")
    )
    backend.run_cell = mock.AsyncMock(
        return_value=ColabExecutionResult(status="ok", cell_id="cell-1", stdout="hello\n")
    )
    controller.visible_backend = backend
    controller.execution_backend = backend

    result = await controller.run_colab_code("print('hello')", mode="append")

    backend.write_cell.assert_awaited_once_with("print('hello')", None, "append")
    backend.run_cell.assert_awaited_once_with(
        cell_id="cell-1", wait=True, timeout_seconds=120
    )
    assert result.stdout == "hello\n"


def test_run_runtime_code_delegates_to_runtime_tool(controller):
    result = asyncio.run(controller.run_runtime_code("print('hello')"))
    assert result.stdout == "runtime\n"


@pytest.mark.asyncio
async def test_ml_tools_route_through_proxy_run_colab_code_when_connected(controller):
    controller.visible_backend = mock.Mock()
    controller.execution_backend = controller.visible_backend
    controller.run_colab_code = mock.AsyncMock(
        return_value=ColabExecutionResult(status="ok", stdout="proxy\n")
    )

    setup_result = await controller.setup_ml_workspace(["pandas"])
    fetch_result = await controller.fetch_remote_dataset(
        "https://example.com/data.csv", "/content"
    )
    pipeline_result = await controller.execute_ml_pipeline("print('done')")

    controller.runtime_tool.build_setup_ml_workspace_code.assert_called_once_with(
        ["pandas"]
    )
    controller.runtime_tool.build_fetch_remote_dataset_code.assert_called_once_with(
        "https://example.com/data.csv", "/content"
    )
    controller.runtime_tool.build_execute_ml_pipeline_code.assert_called_once_with(
        "print('done')"
    )
    controller.run_colab_code.assert_any_await("setup_code", mode="append", wait=True)
    controller.run_colab_code.assert_any_await("fetch_code", mode="append", wait=True)
    controller.run_colab_code.assert_any_await("pipeline_code", mode="append", wait=True)
    controller.runtime_tool.parse_ml_pipeline_result.assert_called_once()
    assert setup_result.stdout == "proxy\n"
    assert fetch_result.stdout == "proxy\n"
    assert pipeline_result.stdout == "0.8\n"


@pytest.mark.asyncio
async def test_ml_tools_route_through_proxy_execution_backend_without_full_visible_backend(
    controller,
):
    controller.visible_backend = None
    controller.execution_backend = mock.Mock()
    controller.run_colab_code = mock.AsyncMock(
        return_value=ColabExecutionResult(status="ok", stdout="proxy-only\n")
    )

    result = await controller.setup_ml_workspace(["pandas"])

    controller.runtime_tool.run_runtime_code.assert_not_called()
    controller.run_colab_code.assert_awaited_once_with(
        "setup_code", mode="append", wait=True
    )
    assert result.stdout == "proxy-only\n"


@pytest.mark.asyncio
async def test_cell_operations_fail_without_proxy(controller):
    with pytest.raises(VisibleNotebookUnavailableError):
        await controller.list_colab_cells()


def test_execution_result_normalizes_streams_and_errors():
    outputs = [
        {"output_type": "stream", "name": "stdout", "text": "hello\n"},
        {"output_type": "stream", "name": "stderr", "text": "warn\n"},
        {"output_type": "display_data", "data": {"text/plain": "42"}},
        {
            "output_type": "error",
            "ename": "ValueError",
            "evalue": "bad",
            "traceback": ["line 1"],
        },
    ]

    result = ColabExecutionResult.from_outputs(outputs)

    assert result.status == "error"
    assert result.stdout == "hello\n"
    assert result.stderr == "warn\n"
    assert result.text_result == "42"
    assert result.error_name == "ValueError"
    assert result.traceback == ["line 1"]


@pytest.mark.asyncio
async def test_proxy_backend_add_code_cell_includes_cell_index_and_language():
    backend = ProxyNotebookBackend(mock.Mock())
    backend.tool_map = {"write_cell": "add_code_cell"}
    backend._resolve_cell_index = mock.AsyncMock(return_value=3)
    backend._invoke_tool_name = mock.AsyncMock(
        return_value={"result": {"cellId": "cell-3"}}
    )

    result = await backend.write_cell("print('hello')", None, "append")

    backend._invoke_tool_name.assert_awaited_once_with(
        "add_code_cell",
        cellIndex=3,
        language="python",
        code="print('hello')",
        source="print('hello')",
        content="print('hello')",
        text="print('hello')",
    )
    assert result.cell_id == "cell-3"
    assert result.code == "print('hello')"


@pytest.mark.asyncio
async def test_proxy_backend_prefers_nested_cell_id_over_top_level_numeric_id():
    backend = ProxyNotebookBackend(mock.Mock())
    backend.tool_map = {"write_cell": "add_code_cell"}
    backend._resolve_cell_index = mock.AsyncMock(return_value=3)
    backend._invoke_tool_name = mock.AsyncMock(
        return_value={"id": 1, "result": {"cellId": "aB3_XyZ9"}}
    )

    result = await backend.write_cell("print('hello')", None, "append")

    assert result.cell_id == "aB3_XyZ9"


@pytest.mark.asyncio
async def test_proxy_backend_recovers_real_cell_id_from_list_after_add():
    backend = ProxyNotebookBackend(mock.Mock())
    backend.tool_map = {
        "write_cell": "add_code_cell",
        "list_cells": "list_cells",
    }
    backend._resolve_cell_index = mock.AsyncMock(return_value=1)
    backend._invoke_tool_name = mock.AsyncMock(return_value={"id": 1, "result": {}})
    backend.list_cells = mock.AsyncMock(
        side_effect=[
            [CellSummary(cell_id="old-cell", preview="")],
            [
                CellSummary(cell_id="old-cell", preview=""),
                CellSummary(cell_id="aB3_XyZ9", preview=""),
            ],
        ]
    )

    result = await backend.write_cell("print('hello')", None, "append")

    assert result.cell_id == "aB3_XyZ9"


@pytest.mark.asyncio
async def test_proxy_backend_run_cell_uses_cell_index_for_native_run_tool():
    backend = ProxyNotebookBackend(mock.Mock())
    backend.tool_map = {"run_cell": "run_cell"}
    backend._resolve_cell_index = mock.AsyncMock(return_value=4)
    backend._invoke_tool_name = mock.AsyncMock(return_value={"status": "ok"})

    await backend.run_cell("cell-4", wait=False, timeout_seconds=120)

    backend._invoke_tool_name.assert_awaited_once_with("run_cell", cellIndex=4)


@pytest.mark.asyncio
async def test_proxy_backend_run_code_cell_uses_cell_id_for_native_run_tool():
    backend = ProxyNotebookBackend(mock.Mock())
    backend.tool_map = {"run_cell": "run_code_cell"}
    backend._invoke_tool_name = mock.AsyncMock(return_value={"status": "ok"})

    await backend.run_cell("cell-abc123", wait=False, timeout_seconds=120)

    backend._invoke_tool_name.assert_awaited_once_with(
        "run_code_cell", cellId="cell-abc123"
    )


@pytest.mark.asyncio
async def test_proxy_backend_run_code_cell_resolves_numeric_index_to_real_id():
    backend = ProxyNotebookBackend(mock.Mock())
    backend.tool_map = {"run_cell": "run_code_cell", "list_cells": "list_cells"}
    backend._invoke_tool_name = mock.AsyncMock(return_value={"status": "ok"})
    backend.list_cells = mock.AsyncMock(
        return_value=[
            CellSummary(cell_id="cell-0", preview=""),
            CellSummary(cell_id="aB3_XyZ9", preview=""),
        ]
    )

    await backend.run_cell("1", wait=False, timeout_seconds=120)

    backend._invoke_tool_name.assert_awaited_once_with(
        "run_code_cell", cellId="aB3_XyZ9"
    )
