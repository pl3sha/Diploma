# 8-bit Character Generator

Веб-приложение для генерации пиксельных спрайтов по текстовому описанию на базе Stable Diffusion v1.5 и LoRA.

**Режимы генерации:** `base` (SD v1.5) · `public` (Pixel Art LoRA) · `custom` (своя LoRA, 507 спрайтов)

## Быстрый старт

| Требование | Версия |
|---|---|
| Python | 3.10+ |
| Node.js | 18+ |
| GPU | опционально (без GPU: генерация ~3–8 мин, обучение очень медленное) |

```powershell
git clone git@github.com:pl3sha/Diploma.git
cd Diploma
scripts\setup.ps1
scripts\run.ps1
```

Linux / macOS:

```bash
chmod +x scripts/*.sh
scripts/setup.sh
scripts/run.sh
```

→ http://127.0.0.1:8000

`setup` создаёт `.venv`, ставит PyTorch (**CUDA cu124** если есть `nvidia-smi`, иначе CPU), зависимости бэкенда и собирает фронтенд.

### Флаги setup

| Windows | Linux/macOS | Действие |
|---|---|---|
| `-DownloadModel` | `--download-model` | Скачать SD v1.5 заранее (~4 ГБ) |
| `-Research` | `--research` | + `requests`, `bitsandbytes`, split train/val |
| `-All` | `--all` | research + предзагрузка модели |

При первом запуске без `-DownloadModel` модель качается автоматически с HuggingFace.

## Структура

```
backend/     FastAPI-сервер, LoRA, requirements.txt
frontend/    React + Vite
dataset/     507 пар PNG+TXT, скрипты предобработки
research/    обучение, валидация, оценка CLIP
scripts/     setup.ps1/.sh, run.ps1/.sh
```

В git только продакшн LoRA: `custom_8bit_v2.safetensors`, `public_pixel_art.safetensors`. Промежуточные чекпоинты — локально, в `.gitignore`.

## Исследования

Kohya_ss **не нужен** для запуска приложения.

```bash
scripts/setup.sh --research          # или setup.ps1 -Research
python research/split_dataset.py     # dataset/train + dataset/val (70/30)
python research/train_lora.py        # GPU
python research/train_lora.py --cpu --batch-size 1   # CPU
python research/evaluate.py        # сервер должен быть запущен (run.ps1)
```

`bitsandbytes` ставится только с флагом `-Research` (для AdamW8bit на GPU). На CPU и Windows без bnb используется обычный AdamW.

## API

| Метод | Эндпоинт | Описание |
|---|---|---|
| POST | `/generate` | Генерация по промпту |
| GET | `/history` | Последние 20 генераций |
| GET | `/health` | Статус сервера и LoRA |

```json
POST /generate
{
  "prompt": "warrior with red armor and sword",
  "model_type": "custom",
  "output_size": 128,
  "steps": 25
}
```

`model_type`: `base` | `public` | `custom` · `output_size`: `80` или `128`