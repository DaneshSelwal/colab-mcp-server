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

import unittest

from colab_mcp.models import (
    CellDetail,
    CellSummary,
    ColabExecutionResult,
    ConnectionStatus,
    MLPipelineResult,
    SaveResult,
)


class TestColabExecutionResult(unittest.TestCase):
    def test_from_outputs_none_or_empty(self):
        result_none = ColabExecutionResult.from_outputs(None)
        self.assertEqual(result_none.status, "ok")
        self.assertEqual(result_none.display_items, [])
        self.assertEqual(result_none.stdout, "")
        self.assertIsNone(result_none.raw_backend_payload)

        result_empty = ColabExecutionResult.from_outputs([])
        self.assertEqual(result_empty.status, "ok")
        self.assertEqual(result_empty.display_items, [])
        self.assertEqual(result_empty.stdout, "")
        self.assertEqual(result_empty.raw_backend_payload, [])

    def test_from_outputs_stream_stdout(self):
        outputs = [
            {"output_type": "stream", "name": "stdout", "text": "hello "},
            {"output_type": "stream", "name": "stdout", "text": ["world", "!"]},
        ]
        result = ColabExecutionResult.from_outputs(outputs)
        self.assertEqual(result.stdout, "hello world!")
        self.assertEqual(result.stderr, "")

    def test_from_outputs_stream_stderr(self):
        outputs = [
            {"output_type": "stream", "name": "stderr", "text": "error "},
            {"output_type": "stream", "name": "stderr", "text": ["occurred"]},
        ]
        result = ColabExecutionResult.from_outputs(outputs)
        self.assertEqual(result.stderr, "error occurred")
        self.assertEqual(result.stdout, "")

    def test_from_outputs_error(self):
        outputs = [
            {
                "output_type": "error",
                "ename": "ValueError",
                "evalue": "bad value",
                "traceback": ["line 1", "line 2"],
            }
        ]
        result = ColabExecutionResult.from_outputs(outputs, status="ok")
        self.assertEqual(result.status, "error")
        self.assertEqual(result.error_name, "ValueError")
        self.assertEqual(result.error_value, "bad value")
        self.assertEqual(result.traceback, ["line 1", "line 2"])

    def test_from_outputs_data_text_plain(self):
        # Using list for text/plain
        outputs = [
            {
                "output_type": "display_data",
                "data": {"text/plain": ["line 1\n", "line 2"]},
            }
        ]
        result = ColabExecutionResult.from_outputs(outputs)
        self.assertEqual(result.text_result, "line 1\nline 2")
        self.assertEqual(result.display_items, outputs)

        # Using string for text/plain
        outputs2 = [
            {
                "output_type": "execute_result",
                "data": {"text/plain": "simple text"},
            }
        ]
        result2 = ColabExecutionResult.from_outputs(outputs2)
        self.assertEqual(result2.text_result, "simple text")
        self.assertEqual(result2.display_items, outputs2)

    def test_from_outputs_text(self):
        outputs = [
            {"output_type": "update_display_data", "text": "some text update"}
        ]
        result = ColabExecutionResult.from_outputs(outputs)
        self.assertEqual(result.text_result, "some text update")
        self.assertEqual(result.display_items, outputs)

        outputs2 = [
            {"output_type": "update_display_data", "text": ["text", " update"]}
        ]
        result2 = ColabExecutionResult.from_outputs(outputs2)
        self.assertEqual(result2.text_result, "text update")

    def test_from_outputs_non_dict(self):
        outputs = ["just a string", 42]
        result = ColabExecutionResult.from_outputs(outputs)
        self.assertEqual(result.display_items, outputs)


class TestModels(unittest.TestCase):
    def test_instantiation(self):
        cs = ConnectionStatus(connected=True)
        self.assertTrue(cs.connected)

        csum = CellSummary(cell_id="abc")
        self.assertEqual(csum.cell_id, "abc")
        self.assertEqual(csum.cell_type, "code")

        cdet = CellDetail(cell_id="def")
        self.assertEqual(cdet.cell_id, "def")
        self.assertEqual(cdet.code, "")

        sr = SaveResult(success=True)
        self.assertTrue(sr.success)

        cr = ColabExecutionResult(status="ok")
        self.assertEqual(cr.status, "ok")

        ml = MLPipelineResult(status="ok")
        self.assertEqual(ml.status, "ok")


if __name__ == "__main__":
    unittest.main()
