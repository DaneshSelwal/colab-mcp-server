import pytest
from colab_mcp.runtime import ColabRuntimeTool

def test_fetch_remote_dataset_code_security():
    tool = ColabRuntimeTool()
    code = tool.build_fetch_remote_dataset_code("http://example.com/test.zip", "/tmp/test_dir")

    assert "extract_to_abs = os.path.abspath(extract_to)" in code
    assert "os.path.commonpath([extract_to_abs, member_path]) != extract_to_abs" in code
    assert "ValueError" in code
