#!/bin/sh
# Run the respinach side of the comparison cases on the macOS host, outside
# the container, so that both sides of a report are measured on the same
# machine (see comparison/README.md). uv syncs .venv-mac with pyproject.toml
# and the compare extra (Matplotlib for the PNG figures) before every run, a
# no-op when nothing changed. respinach itself is installed editable: source
# edits need no reinstall. Arguments go to comparison/run_respinach.py, e.g.
#
#     scripts/mac-compare.sh noesy_strychnine zulf_pf6_1d --report
cd "$(dirname "$0")/.." || exit 1
export UV_PROJECT_ENVIRONMENT=.venv-mac
exec uv run --python 3.12 --extra compare python comparison/run_respinach.py "$@"
