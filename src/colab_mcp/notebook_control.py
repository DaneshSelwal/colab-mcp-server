from __future__ import annotations

import abc
import inspect
import json
import logging
from typing import Any, Literal

from fastmcp import FastMCP

from colab_mcp.models import (
    CellDetail,
    CellSummary,
    ColabExecutionResult,
    ConnectionStatus,
    MLPipelineResult,
    SaveResult,
)
from colab_mcp.session import ColabSessionProxy

REQUIRED_PROXY_CAPABILITIES = [
    "list_cells",
    "read_cell",
    "write_cell",
    "run_cell",
    "get_output",
    "save_notebook",
]

EXECUTION_PROXY_CAPABILITIES = [
    "write_cell",
    "run_cell",
]

PROXY_TOOL_PATTERNS = {
    "list_cells": [("list", "cell"), ("notebook", "cell")],
    "read_cell": [("read", "cell"), ("get", "cell"), ("cell", "source")],
    "write_cell": [
        ("write", "cell"),
        ("update", "cell"),
        ("replace", "cell"),
        ("append", "cell"),
        ("insert", "cell"),
        ("set", "cell"),
    ],
    "run_cell": [("run", "cell"), ("execute", "cell")],
    "get_output": [("output",), ("cell", "result"), ("execution", "result")],
    "save_notebook": [("save", "notebook"), ("save",)],
}


class VisibleNotebookUnavailableError(RuntimeError):
    pass


async def _maybe_await(value):
    if inspect.isawaitable(value):
        return await value
    return value


def _tool_name(tool: Any) -> str:
    if isinstance(tool, dict):
        return str(tool.get("name", ""))
    return str(getattr(tool, "name", ""))


def _tool_description(tool: Any) -> str:
    if isinstance(tool, dict):
        return str(tool.get("description", ""))
    return str(getattr(tool, "description", ""))


def _normalize_tool_list(payload: Any) -> list[Any]:
    if payload is None:
        return []
    if isinstance(payload, list):
        return payload
    if isinstance(payload, tuple):
        return list(payload)
    for attr in ("tools", "items"):
        value = getattr(payload, attr, None)
        if value is not None:
            return _normalize_tool_list(value)
    if isinstance(payload, dict):
        for key in ("tools", "items", "result", "data"):
            if key in payload:
                return _normalize_tool_list(payload[key])
    return []


def _extract_structured_payload(payload: Any) -> Any:
    if payload is None:
        return None
    if isinstance(payload, dict):
        if "structured_content" in payload:
            return payload["structured_content"]
        if "structuredContent" in payload:
            return payload["structuredContent"]
        if "result" in payload:
            return payload["result"]
        return payload

    for attr in ("structured_content", "structuredContent", "data", "result"):
        value = getattr(payload, attr, None)
        if value is not None:
            return value

    content = getattr(payload, "content", None)
    if content:
        texts = []
        for item in content:
            text = getattr(item, "text", None)
            if text is not None:
                texts.append(text)
        if len(texts) == 1:
            try:
                return json.loads(texts[0])
            except json.JSONDecodeError:
                return texts[0]
        if texts:
            return "\n".join(texts)
    return payload


def _normalize_cells(payload: Any) -> list[CellSummary]:
    if isinstance(payload, dict):
        payload = payload.get("cells", payload)
    if not isinstance(payload, list):
        return []

    cells: list[CellSummary] = []
    for index, item in enumerate(payload):
        if isinstance(item, CellSummary):
            cells.append(item)
            continue

        if not isinstance(item, dict):
            cells.append(
                CellSummary(cell_id=str(index), preview=str(item), cell_type="code")
            )
            continue

        code = str(item.get("code", ""))
        preview = str(item.get("preview", code[:80]))
        cells.append(
            CellSummary(
                cell_id=str(
                    item.get("cell_id")
                    or item.get("id")
                    or item.get("cellId")
                    or index
                ),
                cell_type=str(item.get("cell_type") or item.get("type") or "code"),
                execution_count=item.get("execution_count")
                or item.get("executionCount"),
                preview=preview,
            )
        )
    return cells


def _is_real_cell_id(value: Any) -> bool:
    if value is None:
        return False
    if not isinstance(value, str):
        value = str(value)
    value = value.strip()
    if not value or value.lower() == "none":
        return False
    return not value.isdigit()


def _extract_cell_id(payload: Any) -> str | None:
    prioritized_keys = ("cellId", "cell_id")

    def walk(value: Any, *, allow_generic_id: bool = False) -> str | None:
        if isinstance(value, dict):
            for key in prioritized_keys:
                candidate = value.get(key)
                if _is_real_cell_id(candidate):
                    return str(candidate)

            cell_value = value.get("cell")
            if isinstance(cell_value, dict):
                nested = walk(cell_value, allow_generic_id=True)
                if nested is not None:
                    return nested

            result_value = value.get("result")
            if isinstance(result_value, dict):
                nested = walk(result_value, allow_generic_id=True)
                if nested is not None:
                    return nested

            for nested_value in value.values():
                nested = walk(nested_value)
                if nested is not None:
                    return nested

            if allow_generic_id:
                generic_id = value.get("id")
                if _is_real_cell_id(generic_id):
                    return str(generic_id)

        if isinstance(value, list):
            for item in value:
                nested = walk(item)
                if nested is not None:
                    return nested
        return None

    return walk(payload, allow_generic_id=True)


def _normalize_cell_detail(payload: Any) -> CellDetail:
    if isinstance(payload, CellDetail):
        return payload
    if not isinstance(payload, dict):
        return CellDetail(cell_id=str(payload), code=str(payload))

    code = str(payload.get("code", payload.get("source", "")))
    cell_id = _extract_cell_id(payload)
    return CellDetail(
        cell_id=str(cell_id),
        cell_type=str(payload.get("cell_type") or payload.get("type") or "code"),
        execution_count=payload.get("execution_count")
        or payload.get("executionCount"),
        preview=str(payload.get("preview") or code[:80]),
        code=code,
        outputs=list(payload.get("outputs") or []),
    )


def _build_execution_result(
    payload: Any,
    *,
    fallback_cell_id: str | None = None,
) -> ColabExecutionResult:
    if isinstance(payload, ColabExecutionResult):
        return payload
    if isinstance(payload, dict):
        if set(payload.keys()) >= {"status", "stdout", "stderr"}:
            return ColabExecutionResult.model_validate(payload)
        outputs = payload.get("outputs")
        if isinstance(outputs, list):
            return ColabExecutionResult.from_outputs(
                outputs,
                status=str(payload.get("status", "ok")),
                cell_id=str(
                    payload.get("cell_id")
                    or payload.get("cellId")
                    or fallback_cell_id
                    or ""
                )
                or None,
                execution_count=payload.get("execution_count")
                or payload.get("executionCount"),
                raw_backend_payload=payload,
            )
        return ColabExecutionResult(
            status=str(payload.get("status", "ok")),
            cell_id=str(
                payload.get("cell_id") or payload.get("cellId") or fallback_cell_id or ""
            )
            or None,
            execution_count=payload.get("execution_count")
            or payload.get("executionCount"),
            stdout=str(payload.get("stdout", "")),
            stderr=str(payload.get("stderr", "")),
            text_result=payload.get("text_result") or payload.get("textResult"),
            display_items=list(
                payload.get("display_items") or payload.get("displayItems") or []
            ),
            error_name=payload.get("error_name") or payload.get("errorName"),
            error_value=payload.get("error_value") or payload.get("errorValue"),
            traceback=list(payload.get("traceback") or []),
            raw_backend_payload=payload,
        )
    if isinstance(payload, list):
        return ColabExecutionResult.from_outputs(
            payload, cell_id=fallback_cell_id, raw_backend_payload=payload
        )
    return ColabExecutionResult(
        status="ok",
        cell_id=fallback_cell_id,
        text_result=str(payload) if payload is not None else None,
        raw_backend_payload=payload,
    )


class HeadlessNotebookBackend(abc.ABC):
    backend_name: str

    @abc.abstractmethod
    async def connect(self) -> list[str]:
        raise NotImplementedError

    @abc.abstractmethod
    async def list_cells(self) -> list[CellSummary]:
        raise NotImplementedError

    @abc.abstractmethod
    async def read_cell(self, cell_id: str) -> CellDetail:
        raise NotImplementedError

    @abc.abstractmethod
    async def write_cell(
        self, code: str, cell_id: str | None, mode: Literal["append", "replace"]
    ) -> CellDetail:
        raise NotImplementedError

    @abc.abstractmethod
    async def run_cell(
        self, cell_id: str | None, wait: bool, timeout_seconds: int
    ) -> ColabExecutionResult:
        raise NotImplementedError

    @abc.abstractmethod
    async def get_output(self, cell_id: str | None) -> ColabExecutionResult:
        raise NotImplementedError

    @abc.abstractmethod
    async def save_notebook(self) -> SaveResult:
        raise NotImplementedError


class ProxyNotebookBackend(HeadlessNotebookBackend):
    backend_name = "proxy"

    def __init__(self, session_proxy: ColabSessionProxy):
        self.session_proxy = session_proxy
        self.tool_map: dict[str, str] = {}

    def has_required_capabilities(self) -> bool:
        return all(name in self.tool_map for name in REQUIRED_PROXY_CAPABILITIES)

    def has_execution_capabilities(self) -> bool:
        return all(name in self.tool_map for name in EXECUTION_PROXY_CAPABILITIES)

    async def connect(self) -> list[str]:
        self.tool_map = await self.discover_capabilities()
        return sorted(self.tool_map)

    async def _get_client(self):
        client = self.session_proxy.get_connected_client()
        if client is None:
            raise VisibleNotebookUnavailableError(
                "No headless Colab notebook proxy is connected."
            )
        return client

    async def _call_client_method(self, names: list[str], *args, **kwargs):
        client = await self._get_client()
        last_error = None
        for name in names:
            method = getattr(client, name, None)
            if method is None:
                continue
            try:
                return await _maybe_await(method(*args, **kwargs))
            except TypeError as exc:
                last_error = exc
        if last_error:
            raise last_error
        raise AttributeError(f"Proxy client does not implement any of: {names}")

    async def discover_capabilities(self) -> dict[str, str]:
        tools_payload = await self._call_client_method(["list_tools", "get_tools"])
        tools = _normalize_tool_list(tools_payload)
        discovered: dict[str, str] = {}
        for capability, patterns in PROXY_TOOL_PATTERNS.items():
            best_name = None
            best_score = 0
            for tool in tools:
                haystack = (
                    f"{_tool_name(tool)} {_tool_description(tool)}".strip().lower()
                )
                score = 0
                for pattern in patterns:
                    if all(keyword in haystack for keyword in pattern):
                        score = max(score, len(pattern))
                if score > best_score:
                    best_score = score
                    best_name = _tool_name(tool)
            if best_name:
                discovered[capability] = best_name
        return discovered

    async def _invoke_tool(self, capability: str, **arguments):
        tool_name = self.tool_map.get(capability)
        if tool_name is None:
            raise VisibleNotebookUnavailableError(
                f"The connected headless notebook proxy does not expose '{capability}'."
            )

        return await self._invoke_tool_name(tool_name, **arguments)

    async def _invoke_tool_name(self, tool_name: str, **arguments):
        try:
            payload = await self._call_client_method(
                ["call_tool"], tool_name, arguments=arguments
            )
        except TypeError:
            payload = await self._call_client_method(
                ["call_tool"], tool_name, arguments
            )
        return _extract_structured_payload(payload)

    async def _resolve_cell_index(
        self, cell_id: str | None = None, *, append: bool = False
    ) -> int:
        if append:
            if "list_cells" in self.tool_map:
                return len(await self.list_cells())
            return 9999

        if cell_id is not None:
            try:
                return int(cell_id)
            except (TypeError, ValueError):
                if "list_cells" in self.tool_map:
                    cells = await self.list_cells()
                    for index, cell in enumerate(cells):
                        if cell.cell_id == cell_id:
                            return index

        if "list_cells" in self.tool_map:
            cells = await self.list_cells()
            return max(len(cells) - 1, 0)

        return 9999

    async def _resolve_real_cell_id(self, cell_id: str | None) -> str:
        if _is_real_cell_id(cell_id):
            return str(cell_id)

        if cell_id is not None:
            try:
                target_index = int(str(cell_id))
            except (TypeError, ValueError):
                target_index = None
            else:
                if "list_cells" in self.tool_map:
                    cells = await self.list_cells()
                    if 0 <= target_index < len(cells):
                        candidate = cells[target_index].cell_id
                        if _is_real_cell_id(candidate):
                            return str(candidate)

        raise VisibleNotebookUnavailableError(
            "Unable to resolve a real Colab cellId from the proxy response."
        )

    async def list_cells(self) -> list[CellSummary]:
        return _normalize_cells(await self._invoke_tool("list_cells"))

    async def read_cell(self, cell_id: str) -> CellDetail:
        tool_name = self.tool_map.get("read_cell")
        if tool_name and "cell" in tool_name.lower():
            resolved_cell_id = await self._resolve_real_cell_id(cell_id)
            if "cellid" in tool_name.lower() or "code_cell" in tool_name.lower():
                return _normalize_cell_detail(
                    await self._invoke_tool_name(tool_name, cellId=resolved_cell_id)
                )
        return _normalize_cell_detail(
            await self._invoke_tool("read_cell", cell_id=cell_id)
        )

    async def write_cell(
        self, code: str, cell_id: str | None, mode: Literal["append", "replace"]
    ) -> CellDetail:
        tool_name = self.tool_map.get("write_cell")
        if tool_name and "add_code_cell" in tool_name.lower():
            before_cells = await self.list_cells() if "list_cells" in self.tool_map else []
            cell_index = await self._resolve_cell_index(
                cell_id,
                append=(mode == "append" or cell_id is None),
            )
            payload = await self._invoke_tool_name(
                tool_name,
                cellIndex=cell_index,
                language="python",
                code=code,
                source=code,
                content=code,
                text=code,
            )
            detail = _normalize_cell_detail(payload)
            if not _is_real_cell_id(detail.cell_id) and "list_cells" in self.tool_map:
                after_cells = await self.list_cells()
                before_ids = {cell.cell_id for cell in before_cells if _is_real_cell_id(cell.cell_id)}
                new_cells = [
                    cell for cell in after_cells if _is_real_cell_id(cell.cell_id) and cell.cell_id not in before_ids
                ]
                if 0 <= cell_index < len(after_cells):
                    indexed_cell = after_cells[cell_index]
                    if _is_real_cell_id(indexed_cell.cell_id):
                        detail.cell_id = indexed_cell.cell_id
                if not _is_real_cell_id(detail.cell_id) and new_cells:
                    detail.cell_id = new_cells[-1].cell_id
            if not _is_real_cell_id(detail.cell_id):
                raise VisibleNotebookUnavailableError(
                    "Unable to resolve the real Colab cellId returned by add_code_cell."
                )
            detail.code = code
            detail.preview = code[:80]
            return detail

        payload = await self._invoke_tool(
            "write_cell", code=code, cell_id=cell_id, mode=mode
        )
        return _normalize_cell_detail(payload)

    async def run_cell(
        self, cell_id: str | None, wait: bool, timeout_seconds: int
    ) -> ColabExecutionResult:
        tool_name = self.tool_map.get("run_cell")
        if tool_name and "run_code_cell" in tool_name.lower():
            resolved_cell_id = await self._resolve_real_cell_id(cell_id)
            payload = await self._invoke_tool_name(
                tool_name,
                cellId=str(resolved_cell_id),
            )
            result = _build_execution_result(payload, fallback_cell_id=resolved_cell_id)
            if wait and "get_output" in self.tool_map:
                try:
                    return await self.get_output(str(resolved_cell_id))
                except Exception:
                    return result
            return result

        if tool_name and (
            "run_cell" in tool_name.lower() or "execute_cell" in tool_name.lower()
        ):
            cell_index = await self._resolve_cell_index(cell_id)
            payload = await self._invoke_tool_name(tool_name, cellIndex=cell_index)
            result = _build_execution_result(payload, fallback_cell_id=cell_id)
            if wait and "get_output" in self.tool_map:
                try:
                    return await self.get_output(cell_id or str(cell_index))
                except Exception:
                    return result
            return result

        payload = await self._invoke_tool(
            "run_cell",
            cell_id=cell_id,
            wait=wait,
            timeout_seconds=timeout_seconds,
        )
        return _build_execution_result(payload, fallback_cell_id=cell_id)

    async def get_output(self, cell_id: str | None) -> ColabExecutionResult:
        tool_name = self.tool_map.get("get_output")
        if tool_name and cell_id is not None and "cell" in tool_name.lower():
            resolved_cell_id = await self._resolve_real_cell_id(cell_id)
            if "cellid" in tool_name.lower() or "code_cell" in tool_name.lower():
                payload = await self._invoke_tool_name(tool_name, cellId=resolved_cell_id)
                return _build_execution_result(payload, fallback_cell_id=resolved_cell_id)
        payload = await self._invoke_tool("get_output", cell_id=cell_id)
        return _build_execution_result(payload, fallback_cell_id=cell_id)

    async def save_notebook(self) -> SaveResult:
        payload = await self._invoke_tool("save_notebook")
        if isinstance(payload, SaveResult):
            return payload
        if isinstance(payload, dict):
            return SaveResult.model_validate(payload)
        return SaveResult(success=bool(payload), message=str(payload))


class NotebookController:
    def __init__(
        self,
        session_proxy: ColabSessionProxy | None = None,
        runtime_tool: Any | None = None,
        runtime_tool_factory: Any | None = None,
    ):
        self.session_proxy = session_proxy
        self.runtime_tool = runtime_tool
        self.runtime_tool_factory = runtime_tool_factory
        self.visible_backend: HeadlessNotebookBackend | None = None
        self.execution_backend: HeadlessNotebookBackend | None = None
        self.visible_backend_name: str | None = None
        self.capabilities: list[str] = []
        self.logger = logging.getLogger(__name__)
        self.mcp = FastMCP("colab")
        self.mcp.tool(self.connect_colab)
        self.mcp.tool(self.list_colab_cells)
        self.mcp.tool(self.read_colab_cell)
        self.mcp.tool(self.write_colab_cell)
        self.mcp.tool(self.run_colab_cell)
        self.mcp.tool(self.run_colab_code)
        self.mcp.tool(self.get_colab_output)
        self.mcp.tool(self.save_colab_notebook)
        self.mcp.tool(self.run_runtime_code)
        self.mcp.tool(self.setup_ml_workspace)
        self.mcp.tool(self.fetch_remote_dataset)
        self.mcp.tool(self.execute_ml_pipeline)

    def _get_runtime_tool(self):
        if self.runtime_tool is None and self.runtime_tool_factory is not None:
            self.runtime_tool = self.runtime_tool_factory()
        if self.runtime_tool is None:
            raise VisibleNotebookUnavailableError(
                "The direct runtime backend is not available."
            )
        return self.runtime_tool

    def _make_proxy_backend(self) -> ProxyNotebookBackend | None:
        if self.session_proxy is None or not self.session_proxy.is_connected():
            return None
        return ProxyNotebookBackend(self.session_proxy)

    async def connect_colab(self, notebook_url: str | None = None) -> ConnectionStatus:
        proxy_token = None
        proxy_port = None
        connect_url = None
        if self.session_proxy is not None and self.session_proxy.wss is not None:
            proxy_token = self.session_proxy.wss.token
            proxy_port = self.session_proxy.wss.port
            connect_url = self.session_proxy.build_connect_url(notebook_url)

        proxy_backend = self._make_proxy_backend()
        if proxy_backend is None:
            self.visible_backend = None
            self.execution_backend = None
            self.visible_backend_name = None
            self.capabilities = []
            return ConnectionStatus(
                connected=False,
                backend=None,
                notebook_url=notebook_url,
                browser_attached=False,
                proxy_connected=False,
                proxy_token=proxy_token,
                proxy_port=proxy_port,
                connect_url=connect_url,
                capabilities=[],
                message="No headless Colab notebook proxy is connected. Runtime execution remains available through run_runtime_code.",
            )

        capabilities = await proxy_backend.connect()
        self.capabilities = capabilities
        self.execution_backend = (
            proxy_backend if proxy_backend.has_execution_capabilities() else None
        )
        if proxy_backend.has_required_capabilities():
            self.visible_backend = proxy_backend
            self.visible_backend_name = proxy_backend.backend_name
            return ConnectionStatus(
                connected=True,
                backend=self.visible_backend_name,
                notebook_url=notebook_url,
                browser_attached=False,
                proxy_connected=True,
                proxy_token=proxy_token,
                proxy_port=proxy_port,
                connect_url=connect_url,
                capabilities=capabilities,
                message="Connected to the headless Colab notebook proxy.",
            )

        self.visible_backend = None
        self.visible_backend_name = None
        return ConnectionStatus(
            connected=False,
            backend=None,
            notebook_url=notebook_url,
            browser_attached=False,
            proxy_connected=True,
            proxy_token=proxy_token,
            proxy_port=proxy_port,
            connect_url=connect_url,
            capabilities=capabilities,
            message="A Colab proxy is connected, but it does not expose the required headless notebook tools.",
        )

    def _require_visible_backend(self) -> HeadlessNotebookBackend:
        if self.visible_backend is None:
            raise VisibleNotebookUnavailableError(
                "Headless notebook operations are unavailable. Connect a Colab proxy that exposes notebook tools, or use run_runtime_code for direct kernel execution."
            )
        return self.visible_backend

    def _get_execution_backend(self) -> HeadlessNotebookBackend | None:
        return self.execution_backend

    async def list_colab_cells(self) -> list[CellSummary]:
        return await self._require_visible_backend().list_cells()

    async def read_colab_cell(self, cell_id: str) -> CellDetail:
        return await self._require_visible_backend().read_cell(cell_id)

    async def write_colab_cell(
        self,
        code: str,
        cell_id: str | None = None,
        mode: Literal["append", "replace"] = "append",
    ) -> CellDetail:
        return await self._require_visible_backend().write_cell(code, cell_id, mode)

    async def run_colab_cell(
        self,
        cell_id: str | None = None,
        wait: bool = True,
        timeout_seconds: int = 120,
    ) -> ColabExecutionResult:
        return await self._require_visible_backend().run_cell(
            cell_id, wait, timeout_seconds
        )

    async def run_colab_code(
        self,
        code: str,
        mode: Literal["append", "replace"] = "append",
        wait: bool = True,
        timeout_seconds: int = 120,
    ) -> ColabExecutionResult:
        execution_backend = self._get_execution_backend()
        if execution_backend is None:
            return await self.run_runtime_code(code)

        cell = await execution_backend.write_cell(code, None, mode)
        result = await execution_backend.run_cell(
            cell_id=cell.cell_id,
            wait=wait,
            timeout_seconds=timeout_seconds,
        )
        if result.cell_id is None:
            result.cell_id = cell.cell_id
        return result

    async def get_colab_output(
        self, cell_id: str | None = None
    ) -> ColabExecutionResult:
        return await self._require_visible_backend().get_output(cell_id)

    async def save_colab_notebook(self) -> SaveResult:
        return await self._require_visible_backend().save_notebook()

    async def run_runtime_code(self, code: str) -> ColabExecutionResult:
        runtime_tool = self._get_runtime_tool()
        result = runtime_tool.run_runtime_code(code)
        return await _maybe_await(result)

    async def _run_ml_code_via_best_backend(self, code: str) -> ColabExecutionResult:
        if self._get_execution_backend() is not None:
            return await self.run_colab_code(code, mode="append", wait=True)
        return await self.run_runtime_code(code)

    async def setup_ml_workspace(self, packages: list[str]) -> ColabExecutionResult:
        runtime_tool = self._get_runtime_tool()
        code = runtime_tool.build_setup_ml_workspace_code(packages)
        return await self._run_ml_code_via_best_backend(code)

    async def fetch_remote_dataset(
        self, download_url: str, extract_to: str
    ) -> ColabExecutionResult:
        runtime_tool = self._get_runtime_tool()
        code = runtime_tool.build_fetch_remote_dataset_code(download_url, extract_to)
        return await self._run_ml_code_via_best_backend(code)

    async def execute_ml_pipeline(self, code_block: str) -> MLPipelineResult:
        runtime_tool = self._get_runtime_tool()
        code = runtime_tool.build_execute_ml_pipeline_code(code_block)
        execution_result = await self._run_ml_code_via_best_backend(code)
        return runtime_tool.parse_ml_pipeline_result(execution_result)
