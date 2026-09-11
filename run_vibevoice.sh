#!/usr/bin/env bash
set -euo pipefail

# Directory in cui si trova questo script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="${VENV_DIR:-.venv}"
SERVER_SCRIPT="${SERVER_SCRIPT:-vibevoice_openai_server_cuda.py}"

# La chiave può essere già esportata dall'esterno.
# Se vuoi impostarla qui, usa per esempio:
# export LOCAL_TTS_API_KEY="cambia-questa-chiave"

if [[ -z "${LOCAL_TTS_API_KEY:-}" ]]; then
    echo "[errore] LOCAL_TTS_API_KEY non impostata."
    echo 'Esempio: export LOCAL_TTS_API_KEY="supersecret"'
    exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
    echo "[setup] Creo virtualenv: $VENV_DIR"
    python3 -m venv "$VENV_DIR"

    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"

    echo "[setup] Aggiorno pip/setuptools/wheel..."
    python -m pip install --upgrade pip setuptools wheel

    # Se siamo nella root del repository VibeVoice, installiamo il progetto.
    if [[ -f "pyproject.toml" ]]; then
        echo "[setup] Installo VibeVoice + dipendenze streaming..."
        python -m pip install -e ".[streamingtts]"
    else
        echo "[warning] pyproject.toml non trovato."
        echo "[warning] Salto 'pip install -e \".[streamingtts]\"'."
    fi

    echo "[setup] Installo dipendenze del server..."
    python -m pip install \
        fastapi \
        "uvicorn[standard]" \
        huggingface_hub \
        numpy \
        pydantic

    echo "[setup] Virtualenv pronto."
else
    echo "[venv] Uso virtualenv esistente: $VENV_DIR"

    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
fi

if [[ ! -f "$SERVER_SCRIPT" ]]; then
    echo "[errore] Server non trovato: $SCRIPT_DIR/$SERVER_SCRIPT"
    exit 1
fi

echo "[check] Python: $(command -v python)"

# Il server richiesto è CUDA-only: falliamo subito se PyTorch/CUDA non funzionano.
python - <<'PY'
import sys

try:
    import torch
except ImportError:
    print("[errore] PyTorch non è installato nel virtualenv.", file=sys.stderr)
    sys.exit(1)

print(f"[check] PyTorch: {torch.__version__}")
print(f"[check] CUDA runtime PyTorch: {torch.version.cuda}")

if not torch.cuda.is_available():
    print(
        "[errore] CUDA non è disponibile in questo virtualenv.\n"
        "Controlla driver NVIDIA e installazione CUDA di PyTorch.",
        file=sys.stderr,
    )
    sys.exit(1)

print(f"[check] GPU: {torch.cuda.get_device_name(0)}")
PY

export VIBEVOICE_DDPM_STEPS="${VIBEVOICE_DDPM_STEPS:-1}"
export VIBEVOICE_DEFAULT_VOICE="${VIBEVOICE_DEFAULT_VOICE:-woman}"

echo "[start] Server: $SERVER_SCRIPT"
echo "[start] DDPM steps: $VIBEVOICE_DDPM_STEPS"
echo "[start] Voce default: $VIBEVOICE_DEFAULT_VOICE"

exec python "$SERVER_SCRIPT" "$@"
