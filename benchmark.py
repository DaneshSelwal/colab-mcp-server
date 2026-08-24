import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

from colab_mcp.notebook_control import ProxyNotebookBackend, CellSummary
from colab_mcp.session import ColabSessionProxy

async def main():
    session_proxy_mock = MagicMock(spec=ColabSessionProxy)
    backend = ProxyNotebookBackend(session_proxy_mock)

    backend.tool_map = {"list_cells": "list_cells", "write_cell": "add_code_cell"}

    # Mock list_cells to simulate latency and count calls
    call_count = 0
    async def mock_list_cells():
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.1) # Simulate 100ms network latency
        return [CellSummary(cell_id="abc", cell_type="code", preview="test")]

    backend.list_cells = mock_list_cells
    backend._invoke_tool_name = AsyncMock(return_value={"cellId": "new_id"})

    start_time = time.time()
    await backend.write_cell(code="print(1)", cell_id="abc", mode="append")
    end_time = time.time()

    duration = end_time - start_time
    print(f"Time taken: {duration:.4f}s")
    print(f"list_cells called {call_count} times")

if __name__ == "__main__":
    asyncio.run(main())
