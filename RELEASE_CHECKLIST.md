# Release checklist

Use this before tagging or shipping a new version of `colab-mcp`.

## Code health
- [ ] Run the smoke test:
  - `PYTHONPATH=src python scripts/smoke_test.py`
- [ ] Run the full test suite:
  - `PYTHONPATH=src py -m pytest`
- [ ] Confirm the package imports cleanly:
  - `PYTHONPATH=src python -c "import colab_mcp"`

## Runtime checks
- [ ] Verify `--enable-proxy` starts the proxy and exposes a connect URL.
- [ ] Verify `--disable-proxy` still allows runtime-only execution when credentials are present.
- [ ] Confirm a missing log directory is created automatically.
- [ ] Confirm auth token cache paths are writable in the target environment.

## Notebook flow
- [ ] Confirm `connect_colab()` returns a connect URL when the proxy is available.
- [ ] Confirm notebook cell execution returns structured results.
- [ ] Confirm long-running proxy cell execution respects `timeout_seconds`.
- [ ] Confirm ML helper paths work:
  - `setup_ml_workspace`
  - `fetch_remote_dataset`
  - `execute_ml_pipeline`

## Packaging and release
- [ ] Verify `pyproject.toml` metadata is correct.
- [ ] Verify `uv sync` succeeds.
- [ ] Update the changelog or release notes.
- [ ] Tag the release after all checks pass.
