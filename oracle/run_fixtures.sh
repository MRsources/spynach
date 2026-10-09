#!/bin/sh
# Runs oracle/matlab/run_fixtures.m in MATLAB batch mode.
# Paths to Spinach, the repository and the fixture output are set in run_fixtures.m.
# Override the MATLAB executable with: MATLAB=/path/to/matlab oracle/run_fixtures.sh
set -eu

MATLAB="${MATLAB:-/Applications/MATLAB_R2026b.app/bin/matlab}"
SCRIPT="$(cd "$(dirname "$0")/matlab" && pwd)/run_fixtures.m"

if [ ! -x "$MATLAB" ]; then
    echo "MATLAB executable not found: $MATLAB (set MATLAB=...)" >&2
    exit 1
fi

exec "$MATLAB" -batch "run('$SCRIPT')"
