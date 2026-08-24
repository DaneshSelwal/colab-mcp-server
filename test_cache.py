import asyncio
from colab_mcp.notebook_control import ProxyNotebookBackend
from unittest.mock import AsyncMock, MagicMock

async def test_cache():
    backend = ProxyNotebookBackend(MagicMock())
    backend.tool_map = {"list_cells": "foo", "read_cell": "read_cellid"}
    backend._invoke_tool = AsyncMock(return_value=[{"cell_id": "a"}, {"cell_id": "b"}])
    backend._invoke_tool_name = AsyncMock(return_value={"cellId": "a"})

    # Simulate a read_cell
    cell = await backend.read_cell("0")
    print(f"cell id: {cell.cell_id}")
    print(f"list_cells API calls via _invoke_tool: {backend._invoke_tool.call_count}")

if __name__ == "__main__":
    asyncio.run(test_cache())
