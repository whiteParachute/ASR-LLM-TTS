#!/usr/bin/env bash
set -euo pipefail

if ! grep -qi microsoft /proc/version 2>/dev/null; then
    printf 'This installer is intended for WSL2.\n' >&2
    exit 1
fi

python_version="${PYTHON_VERSION:-3.11}"
pypi_index_url="${PYPI_INDEX_URL:-https://pypi.org/simple}"

if command -v uv >/dev/null 2>&1; then
    uv_bin="$(command -v uv)"
elif [[ -x "$HOME/.local/bin/uv" ]]; then
    uv_bin="$HOME/.local/bin/uv"
else
    printf 'uv is missing; run ./scripts/install_wsl_runtime.sh first.\n' >&2
    exit 2
fi

if [[ ! -x .venv/bin/python ]]; then
    printf '%s\n' \
        'The main .venv is missing; run ./scripts/install_wsl_runtime.sh first.' >&2
    exit 3
fi

main_site_packages="$(.venv/bin/python -c \
    'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
main_transformers_version="$(.venv/bin/python -c \
    'import transformers; print(transformers.__version__)')"

"$uv_bin" python install "$python_version"
"$uv_bin" venv --clear --python "$python_version" .venv-tts
tts_site_packages="$(.venv-tts/bin/python -c \
    'import sysconfig; print(sysconfig.get_paths()["purelib"])')"

# Reuse the already-tested CUDA runtime without sharing mutable package files.
# Qwen3-TTS dependency changes happen only inside the copied environment.
cp -a "$main_site_packages"/. "$tts_site_packages"/

"$uv_bin" pip install --python .venv-tts/bin/python \
    --requirement requirements-wsl-tts.txt \
    --index-url "$pypi_index_url"

.venv-tts/bin/python - "$main_transformers_version" <<'PY'
import torch
import transformers
from qwen_tts import Qwen3TTSModel
import subprocess
import sys

if not torch.cuda.is_available():
    raise SystemExit("Qwen3-TTS environment cannot access CUDA inside WSL")
if transformers.__version__ != "4.57.3":
    raise SystemExit(
        f"Qwen3-TTS requires Transformers 4.57.3, got {transformers.__version__}"
    )

main_version_after = subprocess.check_output(
    [".venv/bin/python", "-c", "import transformers; print(transformers.__version__)"],
    text=True,
).strip()
if main_version_after != sys.argv[1]:
    raise SystemExit(
        "Installing Qwen3-TTS unexpectedly changed the main environment"
    )

print(f"python={__import__('sys').version.split()[0]}")
print(f"torch={torch.__version__}")
print(f"transformers={transformers.__version__}")
print(f"main_transformers={main_version_after}")
print(f"gpu={torch.cuda.get_device_name(0)}")
print(f"qwen_tts_model={Qwen3TTSModel.__name__}")
PY

printf '\nQwen3-TTS isolated runtime installation completed.\n'
