#!/usr/bin/env bash
set -u

section() {
    printf '\n[%s]\n' "$1"
}

section "WSL"
if grep -qi microsoft /proc/version 2>/dev/null; then
    printf 'Detected WSL: yes\n'
else
    printf 'Detected WSL: no\n'
fi
uname -a

section "Linux distribution"
if [[ -r /etc/os-release ]]; then
    grep -E '^(PRETTY_NAME|VERSION_ID)=' /etc/os-release
fi

section "CPU and memory"
nproc
free -h

section "Linux filesystem"
df -h "$HOME"
case "$PWD" in
    /mnt/*)
        printf 'WARNING: repository is under /mnt; clone it under ~/projects for WSL performance.\n'
        ;;
    *)
        printf 'Repository is on the Linux filesystem.\n'
        ;;
esac

section "Python"
if command -v uv >/dev/null 2>&1; then
    uv --version
elif [[ -x "$HOME/.local/bin/uv" ]]; then
    "$HOME/.local/bin/uv" --version
fi
for python_bin in python3.11 python3.12 python3; do
    if command -v "$python_bin" >/dev/null 2>&1; then
        "$python_bin" --version
    fi
done

section "NVIDIA GPU exposed to WSL"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia_smi="$(command -v nvidia-smi)"
elif [[ -x /usr/lib/wsl/lib/nvidia-smi ]]; then
    nvidia_smi=/usr/lib/wsl/lib/nvidia-smi
else
    nvidia_smi=""
fi
if [[ -n "$nvidia_smi" ]]; then
    "$nvidia_smi" \
        --query-gpu=name,memory.total,driver_version \
        --format=csv,noheader
else
    printf 'nvidia-smi not found\n'
fi

section "WSLg audio"
if [[ -z "${PULSE_SERVER:-}" && -S /mnt/wslg/PulseServer ]]; then
    export PULSE_SERVER=unix:/mnt/wslg/PulseServer
fi
printf 'PULSE_SERVER=%s\n' "${PULSE_SERVER:-<unset>}"
if command -v pactl >/dev/null 2>&1; then
    pactl info 2>&1 | sed -n '1,20p'
    printf '\nSources:\n'
    pactl list short sources 2>&1
    printf '\nSinks:\n'
    pactl list short sinks 2>&1
else
    printf 'pactl not installed\n'
fi

section "Project virtual environment"
if [[ -x .venv/bin/python ]]; then
    .venv/bin/python - <<'PY'
import importlib.util

packages = (
    "torch",
    "torchaudio",
    "funasr",
    "transformers",
    "kokoro",
    "misaki",
    "sounddevice",
    "soundfile",
    "webrtcvad",
)
for package in packages:
    print(f"{package}: {bool(importlib.util.find_spec(package))}")

try:
    import torch
except ImportError:
    pass
else:
    print(f"torch_version: {torch.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    print(f"torch_cuda_version: {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"cuda_device: {torch.cuda.get_device_name(0)}")
PY
else
    printf '.venv does not exist\n'
fi
