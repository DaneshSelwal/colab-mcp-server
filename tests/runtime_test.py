# Copyright 2026 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest import mock

import pytest
from colab_mcp import runtime


@pytest.fixture
def runtime_tool():
    with mock.patch("fastmcp.FastMCP"):
        return runtime.ColabRuntimeTool()


def test_session_property(runtime_tool):
    mock_session = mock.Mock()
    with mock.patch("colab_mcp.auth.get_credentials", return_value=mock_session):
        assert runtime_tool.session == mock_session
        # Test memoization
        assert runtime_tool.session == mock_session


def test_colab_prod_client_property(runtime_tool):
    mock_session = mock.Mock()
    mock_client_instance = mock.Mock()
    with (
        mock.patch.object(
            runtime.ColabRuntimeTool, "session", new_callable=mock.PropertyMock
        ) as mock_session_prop,
        mock.patch("colab_mcp.client.ColabClient", return_value=mock_client_instance),
    ):
        mock_session_prop.return_value = mock_session
        assert runtime_tool.colab_prod_client == mock_client_instance
        # Test memoization
        assert runtime_tool.colab_prod_client == mock_client_instance


def test_assignment_property(runtime_tool):
    mock_client = mock.Mock()
    mock_assignment = mock.Mock()
    mock_client.assign.return_value = mock_assignment

    with mock.patch.object(
        runtime.ColabRuntimeTool, "colab_prod_client", new_callable=mock.PropertyMock
    ) as mock_client_prop:
        mock_client_prop.return_value = mock_client
        assert runtime_tool.assignment == mock_assignment
        mock_client.assign.assert_called_once()
        # Test memoization
        assert runtime_tool.assignment == mock_assignment
        assert mock_client.assign.call_count == 1


def test_kernel_client_property(runtime_tool):
    mock_assignment = mock.Mock()
    mock_assignment.runtime_proxy_info.url = "http://server"
    mock_assignment.runtime_proxy_info.token = "token123"

    mock_kc_instance = mock.Mock()

    with (
        mock.patch.object(
            runtime.ColabRuntimeTool, "assignment", new_callable=mock.PropertyMock
        ) as mock_assignment_prop,
        mock.patch("jupyter_kernel_client.KernelClient", return_value=mock_kc_instance),
    ):
        mock_assignment_prop.return_value = mock_assignment
        assert runtime_tool.kernel_client == mock_kc_instance
        mock_kc_instance.start.assert_called_once()
        # Test memoization
        assert runtime_tool.kernel_client == mock_kc_instance
        assert mock_kc_instance.start.call_count == 1


def test_start(runtime_tool):
    mock_kc = mock.Mock()
    mock_assignment = mock.Mock()
    mock_assignment.endpoint = "vm-endpoint"

    with (
        mock.patch.object(
            runtime.ColabRuntimeTool, "kernel_client", new_callable=mock.PropertyMock
        ) as mock_kc_prop,
        mock.patch.object(
            runtime.ColabRuntimeTool, "assignment", new_callable=mock.PropertyMock
        ) as mock_assignment_prop,
    ):
        mock_kc_prop.return_value = mock_kc
        mock_assignment_prop.return_value = mock_assignment

        runtime_tool.start()

        mock_kc.execute.assert_called_once_with("_colab_mcp = True")
        runtime_tool.start()
        assert mock_kc.execute.call_count == 1


def test_stop(runtime_tool):
    mock_client = mock.Mock()
    mock_assignment = mock.Mock()
    mock_assignment.endpoint = "vm-endpoint"
    runtime_tool._ColabRuntimeTool__assignment = mock_assignment

    # Test stop when assignment exists
    with mock.patch.object(
        runtime.ColabRuntimeTool,
        "colab_prod_client",
        new_callable=mock.PropertyMock,
    ) as mock_client_prop:
        mock_client_prop.return_value = mock_client
        runtime_tool.stop()

        mock_client.unassign.assert_called_once_with("vm-endpoint")
        mock_client.unassign.reset_mock()
        runtime_tool.stop()
        mock_client.unassign.assert_not_called()

    # Test stop when assignment is None
    runtime_tool._ColabRuntimeTool__assignment = None
    mock_client.unassign.reset_mock()
    with mock.patch.object(
        runtime.ColabRuntimeTool,
        "colab_prod_client",
        new_callable=mock.PropertyMock,
    ) as mock_client_prop:
        mock_client_prop.return_value = mock_client
        runtime_tool.stop()
        mock_client.unassign.assert_not_called()


def test_execute_code(runtime_tool):
    mock_kc = mock.Mock()
    mock_kc.execute.side_effect = [
        {"status": "ok"},
        {
            "status": "ok",
            "execution_count": 7,
            "outputs": [{"output_type": "stream", "name": "stdout", "text": "hello\n"}],
        },
    ]

    with mock.patch.object(
        runtime.ColabRuntimeTool, "kernel_client", new_callable=mock.PropertyMock
    ) as mock_kc_prop, mock.patch.object(
        runtime.ColabRuntimeTool, "assignment", new_callable=mock.PropertyMock
    ) as mock_assignment_prop:
        mock_kc_prop.return_value = mock_kc
        mock_assignment_prop.return_value = mock.Mock(endpoint="vm-endpoint")

        result = runtime_tool.execute_code("print('hello')")

        assert result.status == "ok"
        assert result.execution_count == 7
        assert result.stdout == "hello\n"
        assert mock_kc.execute.call_count == 2
        mock_kc.execute.assert_any_call("print('hello')")


def test_execute_code_no_outputs(runtime_tool):
    mock_kc = mock.Mock()
    mock_kc.execute.side_effect = [{"status": "ok"}, {"status": "ok"}]

    with mock.patch.object(
        runtime.ColabRuntimeTool, "kernel_client", new_callable=mock.PropertyMock
    ) as mock_kc_prop, mock.patch.object(
        runtime.ColabRuntimeTool, "assignment", new_callable=mock.PropertyMock
    ) as mock_assignment_prop:
        mock_kc_prop.return_value = mock_kc
        mock_assignment_prop.return_value = mock.Mock(endpoint="vm-endpoint")

        result = runtime_tool.execute_code("print('hello')")

        assert result.status == "ok"
        assert result.stdout == ""
        assert result.display_items == []


def test_execute_code_empty_reply(runtime_tool):
    mock_kc = mock.Mock()
    mock_kc.execute.side_effect = [{"status": "ok"}, None]

    with mock.patch.object(
        runtime.ColabRuntimeTool, "kernel_client", new_callable=mock.PropertyMock
    ) as mock_kc_prop, mock.patch.object(
        runtime.ColabRuntimeTool, "assignment", new_callable=mock.PropertyMock
    ) as mock_assignment_prop:
        mock_kc_prop.return_value = mock_kc
        mock_assignment_prop.return_value = mock.Mock(endpoint="vm-endpoint")

        result = runtime_tool.execute_code("print('hello')")

        assert result.status == "ok"
        assert result.raw_backend_payload is None


def test_setup_ml_workspace_formats_execution_payload(runtime_tool):
    with mock.patch.object(
        runtime.ColabRuntimeTool,
        "run_runtime_code",
        return_value=mock.Mock(status="ok"),
    ) as mock_run:
        runtime_tool.setup_ml_workspace(["numpy", "pandas"])

    payload = mock_run.call_args.args[0]
    assert "pip" in payload
    assert "numpy" in payload
    assert "pandas" in payload
    assert "/content/data/scour" in payload
    assert "/content/data/concrete" in payload


def test_setup_ml_workspace_flattens_nested_package_lists(runtime_tool):
    payload = runtime_tool.build_setup_ml_workspace_code(
        [["scikit-learn", "pandas"], "matplotlib"]
    )

    assert 'raw_packages = ["scikit-learn", "pandas", "matplotlib"]' in payload
    assert 'install_spec = " ".join(packages)' in payload
    assert "_flatten_packages" in payload


def test_fetch_remote_dataset_formats_execution_payload(runtime_tool):
    with mock.patch.object(
        runtime.ColabRuntimeTool,
        "run_runtime_code",
        return_value=mock.Mock(status="ok"),
    ) as mock_run:
        runtime_tool.fetch_remote_dataset(
            "https://example.com/data.zip", "/content/data/scour"
        )

    payload = mock_run.call_args.args[0]
    assert "urllib.request" in payload
    assert "zipfile.is_zipfile" in payload
    assert "https://example.com/data.zip" in payload
    assert "/content/data/scour" in payload


def test_execute_ml_pipeline_formats_execution_payload_and_parses_result(runtime_tool):
    execution_result = runtime.ColabExecutionResult(
        status="ok",
        stdout='before\n__COLAB_MCP_ML_PIPELINE__{"status":"ok","stdout":"train\\n","stderr":"","generated_file_paths":["/content/model.pkl"],"error_name":null,"error_value":null,"traceback":[]}\nafter',
        stderr="",
    )
    with mock.patch.object(
        runtime.ColabRuntimeTool,
        "run_runtime_code",
        return_value=execution_result,
    ) as mock_run:
        result = runtime_tool.execute_ml_pipeline("print('train')")

    payload = mock_run.call_args.args[0]
    assert "<colab_mcp_ml_pipeline>" in payload
    assert "generated_file_paths" in payload
    assert "print('train')" in payload
    assert result.status == "ok"
    assert result.stdout == "train\n"
    assert result.generated_file_paths == ["/content/model.pkl"]


def test_execution_result_normalizes_list_stream_text():
    result = runtime.ColabExecutionResult.from_outputs(
        [
            {
                "output_type": "stream",
                "name": "stdout",
                "text": ['{"installed_packages": ["scikit-learn", "pandas"]}\n'],
            }
        ]
    )

    assert result.stdout == '{"installed_packages": ["scikit-learn", "pandas"]}\n'
