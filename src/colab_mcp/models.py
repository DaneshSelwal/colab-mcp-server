from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ConnectionStatus(BaseModel):
    connected: bool
    backend: str | None = None
    notebook_url: str | None = None
    browser_attached: bool = False
    proxy_connected: bool = False
    proxy_token: str | None = None
    proxy_port: int | None = None
    connect_url: str | None = None
    capabilities: list[str] = Field(default_factory=list)
    message: str | None = None


class CellSummary(BaseModel):
    cell_id: str
    cell_type: str = "code"
    execution_count: int | None = None
    preview: str = ""


class CellDetail(CellSummary):
    code: str = ""
    outputs: list[Any] = Field(default_factory=list)


class SaveResult(BaseModel):
    success: bool
    notebook_url: str | None = None
    message: str | None = None


class ColabExecutionResult(BaseModel):
    status: str
    cell_id: str | None = None
    execution_count: int | None = None
    stdout: str = ""
    stderr: str = ""
    text_result: str | None = None
    display_items: list[Any] = Field(default_factory=list)
    error_name: str | None = None
    error_value: str | None = None
    traceback: list[str] = Field(default_factory=list)
    raw_backend_payload: Any = None

    @classmethod
    def from_outputs(
        cls,
        outputs: list[Any] | None,
        *,
        status: str = "ok",
        cell_id: str | None = None,
        execution_count: int | None = None,
        raw_backend_payload: Any = None,
    ) -> "ColabExecutionResult":
        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        display_items: list[Any] = []
        text_result: str | None = None
        error_name: str | None = None
        error_value: str | None = None
        traceback: list[str] = []

        for output in outputs or []:
            if not isinstance(output, dict):
                display_items.append(output)
                continue

            output_type = output.get("output_type")
            if output_type == "stream":
                text = output.get("text") or ""
                if isinstance(text, list):
                    text = "".join(str(item) for item in text)
                else:
                    text = str(text)
                if output.get("name") == "stderr":
                    stderr_parts.append(text)
                else:
                    stdout_parts.append(text)
                continue

            if output_type == "error":
                error_name = output.get("ename")
                error_value = output.get("evalue")
                traceback = [str(line) for line in output.get("traceback") or []]
                continue

            data = output.get("data")
            if isinstance(data, dict):
                plain_text = data.get("text/plain")
                if isinstance(plain_text, list):
                    plain_text = "".join(str(item) for item in plain_text)
                if plain_text and text_result is None:
                    text_result = str(plain_text)

            if "text" in output and text_result is None:
                value = output.get("text")
                if isinstance(value, list):
                    value = "".join(str(item) for item in value)
                text_result = str(value)

            display_items.append(output)

        final_status = "error" if error_name else status
        return cls(
            status=final_status,
            cell_id=cell_id,
            execution_count=execution_count,
            stdout="".join(stdout_parts),
            stderr="".join(stderr_parts),
            text_result=text_result,
            display_items=display_items,
            error_name=error_name,
            error_value=error_value,
            traceback=traceback,
            raw_backend_payload=raw_backend_payload if raw_backend_payload is not None else outputs,
        )


class MLPipelineResult(BaseModel):
    status: str
    stdout: str = ""
    stderr: str = ""
    generated_file_paths: list[str] = Field(default_factory=list)
    error_name: str | None = None
    error_value: str | None = None
    traceback: list[str] = Field(default_factory=list)
    raw_execution: Any = None
