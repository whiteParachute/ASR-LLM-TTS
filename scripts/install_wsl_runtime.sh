#!/usr/bin/env bash
set -euo pipefail

if ! grep -qi microsoft /proc/version 2>/dev/null; then
    printf 'This installer is intended for WSL2.\n' >&2
    exit 1
fi

python_version="${PYTHON_VERSION:-3.11}"
torch_version="${TORCH_VERSION:-2.11.0}"
torchvision_version="${TORCHVISION_VERSION:-0.26.0}"
pytorch_index_url="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
pypi_index_url="${PYPI_INDEX_URL:-https://pypi.org/simple}"
uv_version="${UV_VERSION:-0.11.32}"
install_proxy="${WSL_INSTALL_PROXY:-}"

apt_proxy_args=()
curl_proxy_args=()
if [[ -n "$install_proxy" ]]; then
    apt_proxy_args=(
        -o "Acquire::http::Proxy=$install_proxy"
        -o "Acquire::https::Proxy=$install_proxy"
    )
    curl_proxy_args=(--proxy "$install_proxy")
    export ALL_PROXY="$install_proxy"
    export all_proxy="$install_proxy"
fi

sudo apt-get "${apt_proxy_args[@]}" update
sudo apt-get "${apt_proxy_args[@]}" install -y \
    build-essential \
    ca-certificates \
    curl \
    espeak-ng \
    ffmpeg \
    git \
    libportaudio2 \
    libsndfile1 \
    portaudio19-dev \
    pulseaudio-utils \
    unzip

case "$python_version" in
    3.11|3.12) ;;
    *)
        printf 'Python %s is unsupported; use Python 3.11 or 3.12.\n' \
            "$python_version" >&2
        exit 3
        ;;
esac

if command -v uv >/dev/null 2>&1; then
    uv_bin="$(command -v uv)"
elif [[ -x "$HOME/.local/bin/uv" ]]; then
    uv_bin="$HOME/.local/bin/uv"
else
    uv_archive_dir="$(mktemp -d)"
    cleanup_uv_archive() {
        rm -rf -- "$uv_archive_dir"
    }
    trap cleanup_uv_archive EXIT

    curl "${curl_proxy_args[@]}" \
        --fail \
        --location \
        --retry 3 \
        --output "$uv_archive_dir/uv.tar.gz" \
        "https://github.com/astral-sh/uv/releases/download/${uv_version}/uv-x86_64-unknown-linux-gnu.tar.gz"
    tar -xzf "$uv_archive_dir/uv.tar.gz" -C "$uv_archive_dir"
    mkdir -p "$HOME/.local/bin"
    install -m 755 \
        "$uv_archive_dir/uv-x86_64-unknown-linux-gnu/uv" \
        "$HOME/.local/bin/uv"
    install -m 755 \
        "$uv_archive_dir/uv-x86_64-unknown-linux-gnu/uvx" \
        "$HOME/.local/bin/uvx"
    uv_bin="$HOME/.local/bin/uv"
fi

if [[ ! -x "$uv_bin" ]]; then
    printf 'uv installation failed: %s is not executable.\n' "$uv_bin" >&2
    exit 4
fi

"$uv_bin" python install "$python_version"
"$uv_bin" venv --clear --python "$python_version" .venv
"$uv_bin" pip install --python .venv/bin/python \
    --upgrade pip setuptools wheel \
    --index-url "$pypi_index_url"
"$uv_bin" pip install --python .venv/bin/python \
    "torch==${torch_version}" \
    "torchaudio==${torch_version}" \
    "torchvision==${torchvision_version}" \
    --find-links "$pytorch_index_url" \
    --index-url "$pypi_index_url"
"$uv_bin" pip install --python .venv/bin/python \
    --requirement requirements-wsl.txt \
    --index-url "$pypi_index_url"
"$uv_bin" pip install --python .venv/bin/python \
    --no-deps \
    --no-build-isolation \
    --editable .

.venv/bin/python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("PyTorch installed, but CUDA is not available inside WSL")

print(f"python={__import__('sys').version.split()[0]}")
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"gpu={torch.cuda.get_device_name(0)}")
PY

PYTHONPATH=src .venv/bin/python -m unittest discover \
    -s tests \
    -p 'test_*.py' \
    -v

printf '\nWSL runtime installation and unit tests completed.\n'
