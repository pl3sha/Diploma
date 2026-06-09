#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RESEARCH=0
DIPLOMA=0
DOWNLOAD_MODEL=0
ALL=0

for arg in "$@"; do
  case "$arg" in
    --research) RESEARCH=1 ;;
    --diploma) DIPLOMA=1 ;;
    --download-model) DOWNLOAD_MODEL=1 ;;
    --all) ALL=1 ;;
    -h|--help)
      echo "Usage: scripts/setup.sh [--research] [--diploma] [--download-model] [--all]"
      exit 0
      ;;
    *)
      echo "Неизвестный аргумент: $arg"
      exit 1
      ;;
  esac
done

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "Не найден $1"; exit 1; }
}

venv_python() {
  if [ -x ".venv/bin/python" ]; then
    .venv/bin/python "$@"
  else
    python "$@"
  fi
}

install_pytorch() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "  PyTorch + CUDA (cu124) ..."
    .venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
  else
    echo "  PyTorch CPU (nvidia-smi не найден) ..."
    .venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  fi
}

echo "=== 8-bit Character Generator — установка ==="
echo "Корень проекта: $ROOT"

require_cmd python3
require_cmd node
require_cmd npm

PYTHON=python3
if ! python3 -m venv --help >/dev/null 2>&1; then
  PYTHON=python
fi

if [ ! -d ".venv" ]; then
  echo ""
  echo "[1/4] Создание виртуального окружения .venv ..."
  "$PYTHON" -m venv .venv
else
  echo ""
  echo "[1/4] Виртуальное окружение .venv уже есть"
fi

echo "[2/4] Установка PyTorch и Python-зависимостей ..."
install_pytorch
.venv/bin/pip install -r backend/requirements.txt

echo "[3/4] Сборка фронтенда ..."
(cd frontend && npm install && npm run build)

if [ "$RESEARCH" = 1 ] || [ "$ALL" = 1 ]; then
  echo ""
  echo "[research] Установка зависимостей исследований ..."
  .venv/bin/pip install -r research/requirements.txt
  echo "[research] Разбивка датасета train/val (70/30) ..."
  venv_python research/split_dataset.py
fi

if [ "$DIPLOMA" = 1 ] || [ "$ALL" = 1 ]; then
  echo ""
  echo "[diploma] Установка зависимостей для генерации ВКР ..."
  .venv/bin/pip install -r scripts/requirements.txt
fi

if [ "$DOWNLOAD_MODEL" = 1 ] || [ "$ALL" = 1 ]; then
  echo ""
  echo "[model] Загрузка Stable Diffusion v1.5 с HuggingFace (~4 ГБ) ..."
  venv_python -c "from diffusers import StableDiffusionPipeline; StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5', safety_checker=None); print('OK')"
fi

echo ""
echo "=== Готово ==="
echo "Запуск:  scripts/run.sh"
echo "Режимы:  scripts/setup.sh --research | --download-model | --all"
