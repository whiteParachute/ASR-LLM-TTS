#!/usr/bin/env bash
set -euo pipefail

if ! grep -qi microsoft /proc/version 2>/dev/null; then
    printf 'This installer is intended for WSL2.\n' >&2
    exit 1
fi

python_version="${PYTHON_VERSION:-3.11}"
torch_version="${TORCH_VERSION:-2.11.0}"
torch_build_suffix="${TORCH_BUILD_SUFFIX:-cu128}"
pytorch_index_url="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
pypi_index_url="${PYPI_INDEX_URL:-https://pypi.org/simple}"

if command -v uv >/dev/null 2>&1; then
    uv_bin="$(command -v uv)"
elif [[ -x "$HOME/.local/bin/uv" ]]; then
    uv_bin="$HOME/.local/bin/uv"
else
    printf 'uv is missing; run ./scripts/install_wsl_runtime.sh first.\n' >&2
    exit 2
fi

"$uv_bin" python install "$python_version"
"$uv_bin" venv --clear --python "$python_version" .venv-tts
"$uv_bin" pip install --python .venv-tts/bin/python \
    --upgrade pip setuptools wheel \
    --index-url "$pypi_index_url"
"$uv_bin" pip install --python .venv-tts/bin/python \
    "torch==${torch_version}+${torch_build_suffix}" \
    "torchaudio==${torch_version}+${torch_build_suffix}" \
    --index-url "$pytorch_index_url" \
    --extra-index-url "$pypi_index_url" \
    --index-strategy unsafe-best-match
"$uv_bin" pip install --python .venv-tts/bin/python \
    --requirement requirements-wsl-tts.txt \
    --index-url "$pypi_index_url"

.venv-tts/bin/python - <<'PY'
import torch
import transformers
from qwen_tts import Qwen3TTSModel

if not torch.cuda.is_available():
    raise SystemExit("Qwen3-TTS environment cannot access CUDA inside WSL")

print(f"python={__import__('sys').version.split()[0]}")
print(f"torch={torch.__version__}")
print(f"transformers={transformers.__version__}")
print(f"gpu={torch.cuda.get_device_name(0)}")
print(f"qwen_tts_model={Qwen3TTSModel.__name__}")
PY

printf '\nQwen3-TTS isolated runtime installation completed.\n'
