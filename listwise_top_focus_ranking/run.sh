#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
if [[ "${1:-}" == '--background' ]]; then
  shift
  mkdir -p "$SCRIPT_DIR/logs"
  LOG_FILE="${LOG_FILE:-$SCRIPT_DIR/logs/train_$(date +%Y%m%d_%H%M%S)_$$.log}"
  mkdir -p "$(dirname "$LOG_FILE")"
  nohup "$PYTHON_BIN" -u "$SCRIPT_DIR/train.py" "$@" > "$LOG_FILE" 2>&1 < /dev/null &
  printf 'PID=%s\nLog=%s\n' "$!" "$LOG_FILE"
else
  exec "$PYTHON_BIN" -u "$SCRIPT_DIR/train.py" "$@"
fi
