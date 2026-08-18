#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_dir"

if [[ ! -x .venv/bin/python ]]; then
    printf '%s\n' \
        'The main WSL environment is missing.' \
        'Run ./scripts/install_wsl_runtime.sh first.' >&2
    exit 1
fi

export PYTHONPATH=src
exec .venv/bin/python -m voice_assistant.tts_benchmark \
    --config configs/wsl_cuda.yaml \
    "$@"
