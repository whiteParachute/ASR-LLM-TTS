#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

if ! grep -qi microsoft /proc/version 2>/dev/null; then
    printf 'This launcher is intended for WSL2.\n' >&2
    exit 1
fi

if [[ ! -x .venv/bin/voice-assistant-realtime ]]; then
    printf '%s\n' \
        'The WSL virtual environment is missing.' \
        'Run ./scripts/install_wsl_runtime.sh first.' >&2
    exit 2
fi

if [[ ! -x .venv-cosyvoice/bin/python ]]; then
    printf '%s\n' \
        'The isolated CosyVoice3 environment is missing.' \
        'Run ./scripts/install_wsl_cosyvoice_runtime.sh first.' >&2
    exit 3
fi

export PATH="/usr/lib/wsl/lib:$PATH"
if [[ -z "${PULSE_SERVER:-}" && -S /mnt/wslg/PulseServer ]]; then
    export PULSE_SERVER=unix:/mnt/wslg/PulseServer
fi

exec .venv/bin/voice-assistant-realtime \
    --config configs/wsl_cuda.yaml \
    "$@"
