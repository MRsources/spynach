#!/bin/sh
# Runs comparison/matlab/run_comparison.m in MATLAB batch mode on the host.
# Paths and cases are set in run_comparison.m. Override the executable with MATLAB=/path/to/matlab.
set -eu
MATLAB="${MATLAB:-/Applications/MATLAB_R2026b.app/bin/matlab}"
SCRIPT="$(cd "$(dirname "$0")/matlab" && pwd)/run_comparison.m"
if [ ! -x "$MATLAB" ]; then
    echo "MATLAB executable not found: $MATLAB (set MATLAB=...)" >&2
    exit 1
fi
exec "$MATLAB" -batch "run('$SCRIPT')"
