#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ ! -x ".venv/bin/uvicorn" ]; then
  echo "Сначала выполните: scripts/setup.sh"
  exit 1
fi

if [ ! -f "frontend/dist/index.html" ]; then
  echo "Сначала соберите фронтенд: scripts/setup.sh"
  exit 1
fi

echo "Запуск http://127.0.0.1:8000"
echo "Первый старт может скачать SD v1.5 с HuggingFace (~4 ГБ)"
.venv/bin/uvicorn backend.main:app --host 0.0.0.0 --port 8000
