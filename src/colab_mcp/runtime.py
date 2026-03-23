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

import json
import logging
import uuid
from collections.abc import Iterable

from fastmcp import FastMCP

import jupyter_kernel_client

from colab_mcp import auth
from colab_mcp import client
from colab_mcp.models import ColabExecutionResult, MLPipelineResult


ML_PIPELINE_RESULT_MARKER = "__COLAB_MCP_ML_PIPELINE__"


class ColabRuntimeTool(object):
    def __init__(self):
        self.__session = None
        self.__colab_prod_client = None
        self.__kernel_client = None
        self.__assignment = None
        self.__started = False
        # This is meant to be unique per each ColabRuntimeTool.
        self.__id = uuid.uuid4()
        # initialize MCP server bits
        self.mcp = FastMCP("runtime")
        self.mcp.tool(self.run_runtime_code)
        self.mcp.tool(self.execute_code)
        self.mcp.tool(self.setup_ml_workspace)
        self.mcp.tool(self.fetch_remote_dataset)
        self.mcp.tool(self.execute_ml_pipeline)

    @property
    def session(self):
        if not self.__session:
            # A bit cheeky - we passed the client secret the first time,
            # we don't need it this time because we should have a token config.
            # Not great, but keeps us from having to keep track of the client auth config
            # here.
            self.__session = auth.get_credentials(None)
        return self.__session

    @property
    def colab_prod_client(self):
        if not self.__colab_prod_client:
            self.__colab_prod_client = client.ColabClient(client.Prod, self.session)
        return self.__colab_prod_client

    @property
    def assignment(self):
        if not self.__assignment:
            self.__assignment = self.colab_prod_client.assign(self.__id)
        return self.__assignment

    @property
    def kernel_client(self):
        if not self.__kernel_client:
            url = self.assignment.runtime_proxy_info.url
            token = self.assignment.runtime_proxy_info.token

            self.__kernel_client = jupyter_kernel_client.KernelClient(
                server_url=url,
                token=token,
                client_kwargs={
                    "subprotocol": jupyter_kernel_client.JupyterSubprotocol.DEFAULT,
                    "extra_params": {"colab-runtime-proxy-token": token},
                },
                headers={
                    "X-Colab-Client-Agent": "colab-mcp",
                    "X-Colab-Runtime-Proxy-Token": token,
                },
            )
            # Note that start() checks to see if there is already a connection, so repeating it won't hurt.
            # See https://github.com/datalayer/jupyter-kernel-client/blob/786cdc38b7c97beaab751eee2a6836f25e010b06/jupyter_kernel_client/client.py#L413
            self.__kernel_client.start()
        return self.__kernel_client

    def start(self):
        """Start a Colab session. Fetch (assign) a VM, and initialize a Jupyter kernel."""
        if self.__started:
            return

        # All the resources in this class are lazily initialized, so touching the
        # kernel client here will cause use to get a VM assignment, then find the right
        # kernel to use.
        self.kernel_client.execute("_colab_mcp = True")
        self.__started = True
        logging.info("initialized - assigned %s", self.assignment.endpoint)

    def stop(self):
        """Stop the session. Unassign the VM."""
        if self.__assignment:
            if self.__kernel_client and hasattr(self.__kernel_client, "stop"):
                self.__kernel_client.stop()
            self.colab_prod_client.unassign(self.__assignment.endpoint)
            self.__started = False
            logging.info("unassigned %s", self.__assignment.endpoint)
            self.__kernel_client = None
            self.__assignment = None

    def run_runtime_code(self, code: str) -> ColabExecutionResult:
        """Evaluates Python code against the attached Colab runtime.

        Returns a normalized execution result with stdout, stderr, rich outputs,
        and error details.

        Arguments:
            - code (string): the code to execute.
        """
        logging.info(f"running code {code}")
        self.start()
        reply = self.kernel_client.execute(code)
        if not reply:
            return ColabExecutionResult(
                status="ok",
                raw_backend_payload=reply,
            )

        outputs = reply.get("outputs") or []
        execution_count = reply.get("execution_count")
        return ColabExecutionResult.from_outputs(
            outputs,
            status=str(reply.get("status", "ok")),
            execution_count=execution_count,
            raw_backend_payload=reply,
        )

    def execute_code(self, code: str) -> ColabExecutionResult:
        """Backward-compatible alias for runtime code execution."""
        return self.run_runtime_code(code)

    def _normalize_packages(self, packages) -> list[str]:
        flattened: list[str] = []

        def visit(value):
            if value is None:
                return
            if isinstance(value, (str, bytes)):
                flattened.append(str(value))
                return
            if isinstance(value, Iterable):
                for item in value:
                    visit(item)
                return
            flattened.append(str(value))

        visit(packages)
        return flattened

    def build_setup_ml_workspace_code(self, packages: list[str]) -> str:
        normalized_packages = self._normalize_packages(packages)
        return f"""
import json
import os
import subprocess
import sys

raw_packages = {json.dumps(normalized_packages)}

def _flatten_packages(values):
    normalized = []

    def visit(value):
        if value is None:
            return
        if isinstance(value, (str, bytes)):
            normalized.append(str(value))
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                visit(item)
            return
        normalized.append(str(value))

    visit(values)
    return normalized

packages = _flatten_packages(raw_packages)

if packages:
    install_spec = " ".join(packages)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *packages], check=True)
else:
    install_spec = ""

directories = ["/content/data/scour", "/content/data/concrete"]
for directory in directories:
    os.makedirs(directory, exist_ok=True)

print(json.dumps({{"installed_packages": packages, "install_spec": install_spec, "created_directories": directories}}))
""".strip()

    def build_fetch_remote_dataset_code(self, download_url: str, extract_to: str) -> str:
        return f"""
import json
import os
import pathlib
import shutil
import urllib.parse
import urllib.request
import zipfile

download_url = {json.dumps(download_url)}
extract_to = {json.dumps(extract_to)}
os.makedirs(extract_to, exist_ok=True)

parsed = urllib.parse.urlparse(download_url)
filename = pathlib.Path(parsed.path).name or "downloaded_dataset"
download_path = os.path.join(extract_to, filename)

with urllib.request.urlopen(download_url) as response, open(download_path, "wb") as output_file:
    shutil.copyfileobj(response, output_file)

extracted_paths = []
if zipfile.is_zipfile(download_path):
    with zipfile.ZipFile(download_path) as zip_file:
        zip_file.extractall(extract_to)
        extracted_paths = [os.path.join(extract_to, name) for name in zip_file.namelist()]

print(json.dumps({{
    "downloaded_path": download_path,
    "extract_to": extract_to,
    "extracted_paths": extracted_paths,
}}))
""".strip()

    def build_execute_ml_pipeline_code(self, code_block: str) -> str:
        return f"""
import contextlib
import io
import json
import pathlib
import traceback

_before_files = {{
    str(path): path.stat().st_mtime
    for path in pathlib.Path("/content").rglob("*")
    if path.is_file()
}}

_stdout_capture = io.StringIO()
_stderr_capture = io.StringIO()
_status = "ok"
_error_name = None
_error_value = None
_traceback = []

with contextlib.redirect_stdout(_stdout_capture), contextlib.redirect_stderr(_stderr_capture):
    try:
        exec(compile({json.dumps(code_block)}, "<colab_mcp_ml_pipeline>", "exec"), globals(), globals())
    except Exception as exc:
        _status = "error"
        _error_name = exc.__class__.__name__
        _error_value = str(exc)
        _traceback = traceback.format_exc().splitlines()

_generated_files = []
for path in pathlib.Path("/content").rglob("*"):
    if not path.is_file():
        continue
    path_str = str(path)
    modified_at = path.stat().st_mtime
    if path_str not in _before_files or modified_at > _before_files[path_str]:
        _generated_files.append(path_str)

print({json.dumps(ML_PIPELINE_RESULT_MARKER)} + json.dumps({{
    "status": _status,
    "stdout": _stdout_capture.getvalue(),
    "stderr": _stderr_capture.getvalue(),
    "generated_file_paths": sorted(_generated_files),
    "error_name": _error_name,
    "error_value": _error_value,
    "traceback": _traceback,
}}))
""".strip()

    def parse_ml_pipeline_result(
        self, execution_result: ColabExecutionResult
    ) -> MLPipelineResult:
        marker_line = None
        stdout_lines = execution_result.stdout.splitlines()
        clean_stdout_lines: list[str] = []
        for line in stdout_lines:
            if line.startswith(ML_PIPELINE_RESULT_MARKER):
                marker_line = line[len(ML_PIPELINE_RESULT_MARKER) :]
            else:
                clean_stdout_lines.append(line)

        if marker_line is None:
            return MLPipelineResult(
                status=execution_result.status,
                stdout=execution_result.stdout,
                stderr=execution_result.stderr,
                error_name=execution_result.error_name,
                error_value=execution_result.error_value,
                traceback=execution_result.traceback,
                raw_execution=execution_result,
            )

        payload = json.loads(marker_line)
        return MLPipelineResult(
            status=str(payload.get("status", execution_result.status)),
            stdout=str(payload.get("stdout", "\n".join(clean_stdout_lines))),
            stderr=str(payload.get("stderr", execution_result.stderr)),
            generated_file_paths=list(payload.get("generated_file_paths") or []),
            error_name=payload.get("error_name") or execution_result.error_name,
            error_value=payload.get("error_value") or execution_result.error_value,
            traceback=list(payload.get("traceback") or execution_result.traceback),
            raw_execution=execution_result,
        )

    def setup_ml_workspace(self, packages: list[str]) -> ColabExecutionResult:
        """Installs ML packages and prepares default dataset directories in /content.

        Arguments:
            - packages: list of pip package names to install in the Colab runtime.
        """
        return self.run_runtime_code(self.build_setup_ml_workspace_code(packages))

    def fetch_remote_dataset(
        self, download_url: str, extract_to: str
    ) -> ColabExecutionResult:
        """Downloads a remote dataset into /content and extracts zip archives when present."""
        return self.run_runtime_code(
            self.build_fetch_remote_dataset_code(download_url, extract_to)
        )

    def execute_ml_pipeline(self, code_block: str) -> MLPipelineResult:
        """Executes an ML-oriented Python code block and returns structured artifacts."""
        execution_result = self.run_runtime_code(
            self.build_execute_ml_pipeline_code(code_block)
        )
        return self.parse_ml_pipeline_result(execution_result)
