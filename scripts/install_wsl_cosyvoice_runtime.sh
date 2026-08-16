#!/usr/bin/env bash
set -euo pipefail

if ! grep -qi microsoft /proc/version 2>/dev/null; then
    printf 'This installer is intended for WSL2.\n' >&2
    exit 1
fi

python_version="${PYTHON_VERSION:-3.11}"
pypi_index_url="${PYPI_INDEX_URL:-https://pypi.org/simple}"
pytorch_index_url="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
cosyvoice_commit="${COSYVOICE_COMMIT:-074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc}"
runtime_dir="${COSYVOICE_RUNTIME_DIR:-.runtime/CosyVoice}"

if command -v uv >/dev/null 2>&1; then
    uv_bin="$(command -v uv)"
elif [[ -x "$HOME/.local/bin/uv" ]]; then
    uv_bin="$HOME/.local/bin/uv"
else
    printf 'uv is missing; run ./scripts/install_wsl_runtime.sh first.\n' >&2
    exit 2
fi

sudo apt-get update
sudo apt-get install -y \
    espeak-ng \
    ffmpeg \
    git \
    libsndfile1 \
    libsox-dev \
    sox

mkdir -p "$(dirname "$runtime_dir")"
if [[ -d "$runtime_dir/.git" ]]; then
    git -C "$runtime_dir" fetch origin "$cosyvoice_commit"
else
    if [[ -e "$runtime_dir" ]]; then
        printf 'CosyVoice runtime path exists but is not a Git clone: %s\n' \
            "$runtime_dir" >&2
        exit 3
    fi
    git clone --recursive \
        https://github.com/FunAudioLLM/CosyVoice.git \
        "$runtime_dir"
fi
git -C "$runtime_dir" checkout --detach "$cosyvoice_commit"
git -C "$runtime_dir" submodule update --init --recursive

"$uv_bin" python install "$python_version"
"$uv_bin" venv --clear --python "$python_version" .venv-cosyvoice
"$uv_bin" pip install --python .venv-cosyvoice/bin/python \
    "torch==2.3.1" \
    "torchaudio==2.3.1" \
    "torchvision==0.18.1" \
    --index-url "$pytorch_index_url" \
    --extra-index-url "$pypi_index_url" \
    --index-strategy unsafe-best-match
"$uv_bin" pip install --python .venv-cosyvoice/bin/python \
    --requirement requirements-wsl-cosyvoice.txt \
    --index-url "$pypi_index_url"

.venv-cosyvoice/bin/python - "$runtime_dir" <<'PY'
import sys
from pathlib import Path

runtime_dir = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(runtime_dir / "third_party/Matcha-TTS"))
sys.path.insert(0, str(runtime_dir))

import onnxruntime
import torch
import transformers
from cosyvoice.cli.cosyvoice import AutoModel

if not torch.cuda.is_available():
    raise SystemExit("CosyVoice3 environment cannot access CUDA inside WSL")

print(f"python={sys.version.split()[0]}")
print(f"torch={torch.__version__}")
print(f"transformers={transformers.__version__}")
print(f"onnx_providers={onnxruntime.get_available_providers()}")
print(f"gpu={torch.cuda.get_device_name(0)}")
print(f"cosyvoice_loader={AutoModel.__name__}")
PY

printf '\nCosyVoice3 isolated runtime installation completed.\n'
