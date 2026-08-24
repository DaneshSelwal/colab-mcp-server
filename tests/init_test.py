import argparse
import asyncio
import logging
import sys
from unittest import mock
import pytest

from colab_mcp import parse_args, init_logger, main_async, main

def test_parse_args_defaults():
    args = parse_args([])
    assert args.enable_runtime is False
    assert args.enable_proxy is True
    assert args.client_oauth_config == "colab-mcp-oauth-config.json"
    assert args.log is not None

def test_parse_args_custom():
    args = parse_args(["-r", "--disable-proxy", "-c", "custom.json"])
    assert args.enable_runtime is True
    assert args.enable_proxy is False
    assert args.client_oauth_config == "custom.json"

@mock.patch("colab_mcp.os.makedirs")
@mock.patch("colab_mcp.logging.basicConfig")
def test_init_logger(mock_basic_config, mock_makedirs):
    init_logger("/tmp/test_log_dir")
    mock_makedirs.assert_called_once_with("/tmp/test_log_dir", exist_ok=True)
    mock_basic_config.assert_called_once()


@pytest.mark.asyncio
@mock.patch("colab_mcp.sys.argv", ["colab-mcp"])
@mock.patch("colab_mcp.init_logger")
@mock.patch("colab_mcp.ColabSessionProxy")
@mock.patch("colab_mcp.NotebookController")
@mock.patch("colab_mcp.mcp")
async def test_main_async_defaults(mock_mcp, mock_notebook_controller, mock_session_proxy, mock_init_logger):
    mock_session_instance = mock.AsyncMock()
    mock_session_proxy.return_value = mock_session_instance
    mock_session_instance.middleware = []

    mock_controller_instance = mock.Mock()
    mock_notebook_controller.return_value = mock_controller_instance

    # FastMCP.run_async needs an AsyncMock
    mock_mcp.run_async = mock.AsyncMock()

    await main_async()

    mock_init_logger.assert_called_once()
    mock_session_proxy.assert_called_once()
    mock_session_instance.start_proxy_server.assert_called_once()
    mock_notebook_controller.assert_called_once()
    mock_mcp.run_async.assert_called_once()
    mock_session_instance.cleanup.assert_called_once()


@pytest.mark.asyncio
@mock.patch("colab_mcp.sys.argv", ["colab-mcp", "-r"])
@mock.patch("colab_mcp.init_logger")
@mock.patch("colab_mcp.auth.get_credentials")
@mock.patch("colab_mcp.runtime.ColabRuntimeTool")
@mock.patch("colab_mcp.ColabSessionProxy")
@mock.patch("colab_mcp.NotebookController")
@mock.patch("colab_mcp.mcp")
async def test_main_async_runtime_enabled(mock_mcp, mock_notebook_controller, mock_session_proxy, mock_runtime_tool, mock_get_credentials, mock_init_logger):
    mock_session_instance = mock.AsyncMock()
    mock_session_proxy.return_value = mock_session_instance
    mock_session_instance.middleware = []

    mock_runtime_instance = mock.Mock()
    mock_runtime_tool.return_value = mock_runtime_instance

    mock_controller_instance = mock.Mock()
    mock_notebook_controller.return_value = mock_controller_instance

    mock_mcp.run_async = mock.AsyncMock()

    await main_async()

    mock_get_credentials.assert_called_once()
    mock_runtime_tool.assert_called_once()
    mock_runtime_instance.stop.assert_called_once()


@pytest.mark.asyncio
@mock.patch("colab_mcp.sys.argv", ["colab-mcp", "-r"])
@mock.patch("colab_mcp.init_logger")
@mock.patch("colab_mcp.auth.get_credentials", side_effect=PermissionError("Test Permission Error"))
@mock.patch("colab_mcp.sys.exit")
@mock.patch("colab_mcp.mcp")
async def test_main_async_credentials_error(mock_mcp, mock_exit, mock_get_credentials, mock_init_logger):
    mock_mcp.run_async = mock.AsyncMock()
    # Mock sys.exit to raise SystemExit so the execution stops,
    # mirroring actual behavior. This avoids main_async continuing
    # and trying to initialize other parts unmocked.
    mock_exit.side_effect = SystemExit
    with pytest.raises(SystemExit):
        await main_async()
    mock_exit.assert_called_once()

@mock.patch("colab_mcp.asyncio.run")
def test_main(mock_asyncio_run):
    main()
    mock_asyncio_run.assert_called_once()
