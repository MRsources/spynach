#!/bin/sh
# Run the test suite on the macOS host, outside the container.
# uv syncs .venv-mac with pyproject.toml before every run (a no-op when
# nothing changed), so dependency changes made in the container are picked
# up automatically. spynach itself is installed editable: source edits need
# no reinstall. The container's .venv is left alone.
cd "$(dirname "$0")/.." || exit 1
export UV_PROJECT_ENVIRONMENT=.venv-mac
exec uv run --python 3.12 --extra test --extra docs pytest -q "$@"
