# 8-bit Character Generator

Дипломная работа — веб-приложение для генерации пиксельных (8-bit) игровых персонажей с использованием нейросети Stable Diffusion v1.5 и дообученного LoRA-адаптера.

## Описание

Система позволяет генерировать пиксельные спрайты персонажей по текстовому описанию. Поддерживаются три режима:

- **Базовая SD v1.5** — стандартная Stable Diffusion без адаптации к пиксельному стилю
- **Публичная LoRA** — Pixel Art LoRA (PixelArtRedmond, CivitAI)
- **Обученная LoRA** — LoRA-адаптер, дообученный на собственном датасете пиксельных спрайтов

## Стек технологий

| Компонент | Технологии |
|---|---|
| Бэкенд | Python, FastAPI, uvicorn |
| ML | PyTorch, diffusers (SD v1.5), Pillow |
| Обучение | kohya_ss (sd-scripts), LoRA |
| Оценка | CLIP score (transformers) |
| Фронтенд | React 19, Vite, Axios |
| Деплой | cloudflared tunnel |

## Структура проекта

```
Diploma/
├── backend/
│   ├── main.py              # FastAPI сервер
│   ├── lora/                # LoRA-адаптеры
│   │   ├── custom_8bit_v2.safetensors        # Обученная LoRA (kohya_ss, 507 изображений, эпоха 14)
│   │   ├── custom_8bit_v2-000001..safetensors # Чекпоинты по эпохам
│   │   └── public_pixel_art.safetensors       # Публичная Pixel Art LoRA
│   ├── history/             # История генераций
│   └── requirements.txt
├── dataset/
│   ├── images/              # Полный датасет (PNG + TXT, 512×512, 507 пар)
│   ├── raw/                 # Исходные спрайты до предобработки
│   └── prepare_dataset.py   # Предобработка: 512×512, белый фон, NEAREST
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # Основной компонент
│   │   └── App.css          # Стили
│   └── package.json
├── research/                # Исследовательские скрипты
│   ├── train_lora.py        # Кастомный скрипт обучения с val_loss мониторингом
│   ├── split_dataset.py     # Разбивка датасета train/val (70/30, seed=42)
│   ├── evaluate.py          # Оценка моделей по CLIP score
│   ├── loss_log.csv         # Лог train/val loss по эпохам (research/train_lora.py)
│   ├── loss_plot.png        # График train/val loss
│   ├── convert_checkpoint.py
│   └── extract_unet_only.py
└── scripts/
    └── public-tunnel.bat    # Запуск cloudflare tunnel
```

## Датасет

- **507 пар** (изображение + текстовое описание) пиксельных спрайтов из открытых источников (itch.io, OpenGameArt, Kenney.nl)
- Классы: гуманоиды, демоны, нежить, монстры, животные и др.
- Предобработка (`dataset/prepare_dataset.py`): обрезка до квадрата, масштабирование до 512×512, замена прозрачности белым фоном, метод NEAREST для сохранения пиксельной чёткости
- Для исследовательской оценки обобщаемости применялась разбивка **70% / 30%** (см. `research/split_dataset.py`), seed=42 для воспроизводимости; продакшн-обучение (kohya_ss) использует полный датасет

## Обучение LoRA

Обучение выполнено с помощью **kohya_ss** (sd-scripts) — стандартного инструмента для LoRA fine-tuning диффузионных моделей.

### Параметры обучения

| Параметр | Значение |
|---|---|
| Базовая модель | runwayml/stable-diffusion-v1-5 |
| LoRA rank (network_dim) | 32 |
| LoRA alpha | 16 |
| Эпох | 15 |
| num_repeats | 5 |
| Learning rate | 1e-4 |
| Оптимизатор | AdamW8bit |
| Точность | fp16 |
| Датасет | dataset/images/ (507 пар, полный) |
| Шагов/эпоха | ~2535 |
| GPU | NVIDIA RTX 3070 (8 GB VRAM) |
| Активный чекпоинт | эпоха 14 (custom_8bit_v2.safetensors) |

### Валидация

Для верификации обобщающей способности модели реализован скрипт `research/train_lora.py`, который:
- использует разбивку **70/30** (модель обучается только на `train/`, не видя `val/`)
- после каждой эпохи вычисляет **val_loss** на отложенных данных
- строит график train/val loss (`research/loss_plot.png`) и сохраняет лог (`research/loss_log.csv`)

Дополнительно проведена оценка по метрике **CLIP score** на val-выборке (114 изображений):

| Модель | CLIP score |
|---|---|
| Base SD v1.5 | 30.75 |
| Public LoRA | 31.13 |
| **Custom LoRA (kohya_ss)** | **33.59** |

## Запуск приложения

### Бэкенд

```bash
pip install -r backend/requirements.txt
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### Фронтенд (продакшн-сборка)

```bash
cd frontend
npm install
npm run build
# фронтенд раздаётся бэкендом на /
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### Публичный доступ через туннель

```bash
scripts/public-tunnel.bat
```

## API

| Метод | Эндпоинт | Описание |
|---|---|---|
| POST | `/generate` | Генерация персонажа по промпту |
| GET | `/history` | История последних 20 генераций |
| GET | `/health` | Статус сервера и доступность LoRA |

### Пример запроса

```json
POST /generate
{
  "prompt": "warrior with red armor and sword",
  "model_type": "custom",
  "output_size": 128,
  "steps": 25
}
```
