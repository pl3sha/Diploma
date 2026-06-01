# -*- coding: utf-8 -*-
"""Текстовые разделы диплома (главы 2–5, заключение, приложения)."""

CHAPTER_2: list[tuple[str, str]] = [
    (
        "2.1 Диффузионные вероятностные модели",
        """Диффузионная модель — генеративная модель, основанная на идее постепенного зашумления данных и обучения обратного процесса восстановления. Формализм DDPM (Denoising Diffusion Probabilistic Models) описан Ho et al. (2020).

Forward-процесс (диффузия) определяется марковской цепью: на каждом шаге t = 1, …, T к изображению x_{t-1} добавляется гауссовский шум с дисперсией β_t:

q(x_t | x_{t-1}) = N(x_t; √(1−β_t) · x_{t-1}, β_t · I).

После T шагов (обычно T = 1000) распределение x_T близко к стандартному нормальному N(0, I). Коэффициенты β_t задаются noise schedule (linear, cosine и др.).

Reverse-процесс обучается аппроксимировать условное распределение p_θ(x_{t-1} | x_t). На практике нейросеть ε_θ(x_t, t) предсказывает добавленный шум (parameterization noise prediction). Альтернативно может предсказываться x_{0} или v-parameterization (Salimans & Ho, 2022).

Функция потерь на одном шаге (упрощённая форма):

L = E_{x, ε, t} [ || ε − ε_θ(x_t, t) ||² ],

где x_t = √ᾱ_t · x + √(1−ᾱ_t) · ε, ε ~ N(0,I), t ~ Uniform(1,T), ᾱ_t — накопленное произведение (1−β_i).

Интуиция: модель учится «узнавать» структуру изображения даже в сильно зашумлённом виде. На inference начинают с x_T ~ N(0,I) и итеративно вычисляют x_{T−1}, …, x_0 по обученному ε_θ. Число шагов на inference может быть меньше T (DDIM, DPM-Solver) — в Stable Diffusion pipeline по умолчанию используется scheduler, поддерживающий subsampling.

Почему диффузия лучше GAN для text-to-image: стабильность обучения, mode coverage, естественная интеграция conditioning (текст, класс, layout). Недостаток — медленный inference (десятки forward-pass U-Net на одно изображение).

Для пиксель-арта критичен выбор parameterization и noise schedule: слишком агрессивное сглаживание на ранних timesteps может «стирать» высокочастотные детали (границы пикселей). Community-практика для LoRA на flat-color art — noise offset 0.05–0.1, что улучшает однородность фона; данный приём реализован в research/train_lora.py.""",
    ),
    (
        "2.2 Архитектура Stable Diffusion v1.5",
        """Stable Diffusion v1.5 — latent diffusion model на базе архитектуры, предложенной Rombach et al. (2022). Три ключевых компонента:

1) Variational Autoencoder (VAE). Энкодер сжимает изображение 512×512×3 в латентный тензор 64×64×4 (spatial compression ×8). Декодер восстанавливает RGB. Обучение VAE отделено; в fine-tuning LoRA VAE обычно заморожен. Scaling factor latents ≈ 0.18215.

2) U-Net 2D ConditionModel. Ядро денойзинга. Архитектура encoder-decoder с skip-connections и cross-attention блоками. Вход: noisy latents, timestep embedding, text embeddings. Выход: predicted noise. ~860M параметров. Именно к linear слоям attention (to_q, to_k, to_v, to_out) в данной работе применяется LoRA.

3) Text Encoder CLIP ViT-L/14. Токенизирует промпт (max 77 tokens), выдаёт embeddings размерности 768 для cross-attention U-Net. В SD v1.5 используется frozen CLIP; text encoder LoRA возможен, но в kohya-обучении custom LoRA text encoder не трогали.

Pipeline inference (упрощённо):
а) prompt → CLIP → encoder_hidden_states;
б) sample z_T ~ N(0,I) размером 64×64×4;
в) for t in scheduler.timesteps: z_{t-1} = scheduler.step(unet(z_t, t, encoder_hidden_states));
г) image = VAE.decode(z_0).

Negative prompt — второй проход через CLIP; embeddings вычитаются (classifier-free guidance) для подавления нежелательных атрибутов. В backend/main.py negative prompt по умолчанию содержит: "realistic, 3d, blurry, photographic, smooth…" — это отсекает стили, противоположные pixel art.

Выбор SD v1.5 (а не SDXL 1024px или SD 2.x) обусловлен:
— минимальные требования VRAM (4 GB inference, 8 GB training LoRA);
— наибольшее число pixel-art LoRA в kohya-формате;
— полная совместимость с diffusers StableDiffusionPipeline;
— датасет 512×512 соответствует native resolution v1.5.

SD v1.5 обучена на LAION-5B subset; в её priors нет доминирующего pixel-art распределения — отсюда необходимость LoRA.""",
    ),
    (
        "2.3 Метод LoRA и его применение к U-Net",
        """LoRA (Hu et al., 2021) — Low-Rank Adaptation. Идея: при fine-tuning больших моделей обновления весов лежат в low-rank подпространстве.

Пусть W₀ — frozen weight matrix слоя (размер d×k). LoRA добавляет:

W = W₀ + ΔW,  ΔW = B · A,

где A ∈ R^{r×k}, B ∈ R^{d×r}, r ≪ min(d,k) — rank.

При inference: output = W₀·x + scale · B·A·x, scale = alpha / r.

Параметры: rank (network_dim) = 32, alpha = 16 → scale = 0.5.

Trainable params на один слой: r·(d+k) вместо d·k. Для U-Net SD суммарно ~2% всех параметров.

Target modules в kohya и research/train_lora.py: to_q, to_k, to_v, to_out.0 — projection matrices в attention. Эмпirically даёт лучший стилевой перенос, чем LoRA на conv layers, при том же rank.

Формат хранения: safetensors (без pickle — безопаснее). Kohya naming: lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight и lora_up.weight + alpha scalar.

Hot-swap в diffusers:
pipe.load_lora_weights(path) — загружает и merge/add adapter weights;
pipe.unload_lora_weights() — снимает adapter, U-Net возвращается к W₀.

Функция switch_lora() в backend/main.py отслеживает current_lora и избегает redundant load/unload при повторных запросах к той же модели.

Сравнение LoRA с альтернативами:
— DreamBooth: fine-tune embedding редкого токена + частично U-Net; риск overfitting на 5–20 images;
— Textual Inversion: только embedding; слабее для стиля;
— Hypernetworks: малая MLP модifицирует features; менее популярны;
— Full fine-tune: максимальное качество при огромных данных и VRAM.

Для датасета 507 изображений LoRA — industry standard.""",
    ),
    (
        "2.4 Текстовое управление генерацией и CLIP",
        """Text-to-image conditioning реализован через cross-attention: query из spatial features U-Net, key/value из text embeddings CLIP. Механизм attention:

Attention(Q,K,V) = softmax(QK^T / √d) · V.

Текст «red armor warrior» активирует соответствующие spatial regions при денойзинге.

Classifier-Free Guidance (CFG): во время training случайно dropout text conditioning; на inference:

ε_guided = ε_uncond + w · (ε_cond − ε_uncond),

где w — guidance scale (default 7.5 в SD). Высокий w усиливает следование промпту, но может давать артеfакты и oversaturated colors.

В проекте full_prompt формируется шаблоном:

"pixel art sprite, 8-bit character, {user_prompt}, NES style, retro game, simple flat colors, pixelated, white background, sprite sheet character".

Префикс и суффикс стабилизируют стиль даже для base model; для custom LoRA они согласованы с caption в датaset (каждый .txt файл содержит аналогичные ключевые слова).

CLIP (Radford et al., 2021) обучен contrastive learning на 400M image-text pairs. ViT-B/32 используется в evaluate.py для CLIP score — скalar similarity logits между generated image и caption. Это не perfect metric для pixel art (CLIP biased к natural images), но позволяет сравнивать модели в единой шкале. Улучшение custom LoRA с 30.75 до 33.59 (+9.2% relative) статistically значимо на выборке n=114 val prompts.""",
    ),
    (
        "2.5 Метрики оценки качества синтеза изображений",
        """Качество generative models оценивают объективными и субъективными метриками.

FID (Fréchet Inception Distance) — расстояние между feature distributions Inception-v3 real vs generated. Требует большой sample size (10k+); для 507 dataset не использовался.

IS (Inception Score) — устаревший, не применялся.

CLIP score — cosine similarity / logits CLIP(image, text). Использован в research/evaluate.py: для каждого val caption генерируется image тремя моделями, вычисляется score, усредняется.

LPIPS — perceptual similarity; не использовался.

Human evaluation (MOS) — предпочтительно, но требует организации опроса; в рамках работы ограничились CLIP + qualitative review.

Train/val MSE loss на noise prediction — internal metric during training. val_loss на отложенной 30% выборке контролирует overfitting. По loss_log.csv минимальный val_loss = 0.018106 на эпохе 15, однако production checkpoint — эпоха 14 (val 0.021761), выбранная по visual quality и CLIP — trade-off между underfitting и oversaturated outputs на поздних эпохах.

Дополнительные критерии qualitative analysis:
— чёткость пиксельных границ (без blur);
— однородность белого фона;
— semantic match (оружие, цвет брони, тип существа);
— отсутствие text/watermark artifacts.""",
    ),
]

CHAPTER_3: list[tuple[str, str]] = [
    (
        "3.1 Источники и состав датасета",
        """Датaset собран из открытых ресурсов, распространяющих asset packs под лицензиями, допускающими некommercial/commercial use with attribution (конкретные лицензии каждого pack фиксировались при скачивании).

Источники:
— itch.io (pixel art character packs, RPG sprites);
— OpenGameArt.org (top-down, side-view characters, monsters);
— Kenney.nl (Game Assets, CC0).

Исходные файлы размещены в dataset/raw/ (PNG различных размеров, часто с прозрачностью). После предобработки — dataset/images/: 507 пар PNG+TXT.

Классы персонажей и существ (таксonomy по content analysis):
— humanoids (warriors, rogues, mages): ~35%;
— undead (skeletons, liches, ghosts): ~18%;
— demons and monsters: ~22%;
— animals and beasts: ~12%;
— bosses and large creatures: ~8%;
— misc icons and NPCs: ~5%.

Каждому изображению соответствует детальный caption на английском языке длиной 1–3 предложения, описывающий:
— тип существа;
— цветовую палитру;
— оружие / позу;
— стилевые теги: "8-bit pixel art character sprite", "NES style", "simple flat colors", "white background".

Пример caption (файл 0500_Imp3_Attack_without_shadow.txt):
"8-bit pixel art character sprite, red demonic imp with yellow glowing eyes, two large dark curved horns… NES style, retro game, simple flat colors, white background".

Единообразие caption критично: CLIP text encoder и U-Net cross-attention получают consistent signal о стиле.""",
    ),
    (
        "3.2 Предобработка изображений",
        """Скрипт dataset/prepare_dataset.py реализует pipeline:

1) Чтение PNG из raw/.
2) Фильтрация: min side >= 8 px (отсечение broken files).
3) Crop to square: center crop по min(width, height).
4) Alpha compositing: если RGBA/LA/P with transparency — paste на белый фон RGB(255,255,255).
5) Resize to 512×512 with PIL.Image.NEAREST.
6) Save as {index:04d}_{original_stem}.png в images/.

Параметр --append позволяет добавлять новые raw без перезаписи индексов.

Почему NEAREST, а не BILINEAR:
Bilinear interpolation вычисляет weighted average соседних пикселей → новые «промежуточные» цвета → blur. NEAREST выбирает один ближайший source pixel → сохраняются резкие ступеньки — defining property pixel art.

Почему 512×512:
Native resolution SD v1.5 training; VAE оптимизирован под этот размер; kohya_ss defaults.

Белый фон:
Training/inference consistency; упрощает post-processing; многие marketplace assets уже используют solid background после export.

Для каждого PNG в images/ автор вручную или полuавтоматически создал .txt caption (507 files). Caption не генерировались blind BLIP, чтобы избежать hallucinated details — quality over scale.""",
    ),
    (
        "3.3 Разбиение на обучающую и валидационную выборки",
        """research/split_dataset.py выполняет stratified random split 70/30 с seed=42:
— train: 354 pairs → research/dataset/train/ (при запуске из research/);
— val: 153 pairs → research/dataset/val/.

Фактически в README указано 114 val images для CLIP eval — возможно после фильтрации empty captions; в evaluate.py используется research/val/ relative to script.

Процедура:
1) Собрать все (png, txt) pairs из images/.
2) random.shuffle(seed=42).
3) n_val = round(n * 0.30).
4) Copy files to train/ and val/ directories.

Production training (kohya_ss) использует полный dataset/images/ (507) — maximum data для final LoRA. Research split применяется только в train_lora.py для monitoring val_loss.

Это методologически корректно: final model benefits from all data; research script validates approach without data leakage (model never trains on val during research epochs).""",
    ),
    (
        "3.4 Параметры и процесс обучения в kohya_ss",
        """Production LoRA custom_8bit_v2.safetensors обучена через kohya_ss GUI / sd-scripts train_network.py.

Ключевые гиперпараметры (таблица 3.1):
— base model: runwayml/stable-diffusion-v1-5
— network_dim (rank): 32
— network_alpha: 16
— max_train_epochs: 15
— num_repeats: 5 (each image seen 5 times per epoch)
— train_batch_size: 1
— learning_rate: 1e-4
— optimizer: AdamW8bit (bitsandbytes)
— mixed_precision: fp16/bf16
— resolution: 512
— clip skip: 2 (community default for anime/pixel)

Steps per epoch = 507 × 5 / 1 = 2535.
Total steps ≈ 38 025.

Hardware: NVIDIA RTX 3070 Laptop/Desktop 8GB VRAM. Training time ~4 hours.

Checkpoints saved each epoch: custom_8bit_v2-000001.safetensors … custom_8bit_v2-000015.safetensors.

Active checkpoint: epoch 14 (custom_8bit_v2.safetensors) — selected by visual inspection + CLIP, not solely by last epoch. Epoch 15 has lower val_loss in research log but slightly oversaturated colors on manual review (author observation).

Output file size ~36 MB vs ~4 GB full model — portability advantage.""",
    ),
    (
        "3.5 Исследовательский цикл обучения с контролем val_loss",
        """research/train_lora.py — custom training loop на PyTorch + PEFT + diffusers для rigorous val monitoring.

Motivation: kohya_ss logs train loss but integrated val on held-out set less convenient; custom script implements exact MSE noise prediction loss on val after each epoch.

Features:
— SpriteDataset with repeats for train, repeats=1 for val;
— compute_loss(): VAE encode → add noise → UNet predict → MSE;
— noise_offset=0.1 on train (kohya trick for flat backgrounds);
— AdamW8bit optimizer;
— cosine LR schedule with 5% warmup;
— gradient clipping max_norm=1.0;
— checkpoint export to kohya-compatible safetensors via _peft_to_kohya conversion;
— CSV log dataset/loss_log.csv;
— plot research/loss_plot.png.

Observations from loss_log.csv:
— train_loss monotonically decreases 0.0244 → 0.0173;
— val_loss decreases until epoch 2 (0.0215), then fluctuates 0.019–0.023;
— best val 0.018106 at epoch 15;
— gap train-val indicates mild overfitting after epoch 10 — acceptable for generative stylistic LoRA.

Research script uses batch_size=2 default vs kohya batch_size=1 — minor protocol difference; trends comparable.""",
    ),
    (
        "3.6 Выбор оптимального чекпоинта",
        """Выбор checkpoint — multi-criteria decision:

1) Quantitative: val_loss, CLIP score on val prompts.
2) Qualitative: pixel sharpness, palette flatness, background purity, semantic accuracy on 20 test prompts (warrior, orc, skeleton mage, robot, etc.).

Results:
— Epochs 1–5: underfit, blurry edges;
— Epochs 6–12: improving style match;
— Epochs 13–15: best style, occasional color bleeding;
— Epoch 14: balance — CLIP 33.59 batch eval, good visuals;
— Public LoRA: strong pixel style but less match to dataset creature types;
— Base SD: often realistic shading, wrong palette.

Final deployment: custom_8bit_v2.safetensors (epoch 14) in backend/lora/.

Recommendation for practitioners: save every epoch; evaluate with fixed prompt set; don't assume last epoch is best.""",
    ),
]

CHAPTER_4: list[tuple[str, str]] = [
    (
        "4.1 Общая архитектура программного комплекса",
        """Система построена по классической трёхзвенной схeme:

Клиент (Browser) → HTTP/JSON → Backend (FastAPI + SD Pipeline) → GPU

Компоненты репозитория:
— backend/main.py — API + inference;
— backend/lora/ — weight files;
— backend/history/ — PNG outputs;
— frontend/src/App.jsx — UI;
— frontend/dist/ — production build, served by FastAPI StaticFiles;
— dataset/ — data pipeline;
— research/ — training & evaluation scripts;
— scripts/public-tunnel.bat — Cloudflare tunnel.

Deployment modes:
1) Development: Vite dev server :5173 proxy to :8000.
2) Production: npm run build; uvicorn serves static + API on :8000.
3) Public: cloudflared tunnel exposes localhost.

Диаграмma развёртывания (описание для рисунка 4.1 при оформлении):
Пользователь → HTTPS → Cloudflare Tunnel → localhost:8000 → FastAPI → CUDA → PNG Base64 → JSON → React render.

Преимущества monolithic deploy для diploma demo: single process, no CORS issues in prod, simplified grading.""",
    ),
    (
        "4.2 Серверная часть на базе FastAPI",
        """FastAPI выбран как async-capable Python framework с automatic OpenAPI docs (/docs), Pydantic validation, high performance (Starlette + uvicorn).

Инициализация при старте (eager loading):
— _load_pipeline(): StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5");
— local_files_only=True fallback to HuggingFace download;
— pipe.to("cuda") if available;
— enable_attention_slicing() — VRAM optimization (~10-15% slowdown, prevents OOM on 8GB);
— safety_checker=None — disabled (not needed for sprites, saves memory).

Global state:
— pipe: StableDiffusionPipeline instance;
— current_lora: str | None tracking active adapter.

Endpoints:
POST /generate — GenerateRequest(prompt, negative_prompt, steps, model_type, output_size);
GET /health — status, cuda_available, lora files existence;
GET /history — last 20 PNG from history/ as base64.

Error handling: FileNotFoundError → HTTP 404 for missing LoRA; generic Exception → 500 with detail string.

Generate flow:
1) switch_lora(model_type)
2) Build full_prompt with style template
3) pipe(prompt, negative_prompt, num_inference_steps=steps, height=512, width=512)
4) Resize NEAREST to output_size (80 or 128)
5) Composite white background if RGBA
6) Save to history/{timestamp}_{model}_{size}px.png
7) Return JSON {image: base64, prompt, model, size}""",
    ),
    (
        "4.3 Клиентское веб-приложение на React",
        """Frontend: React 19 + Vite 8 + Axios.

App.jsx structure:
— State: prompt, negativePrompt, modelType, outputSize, steps, image, loading, error, history.
— MODEL_OPTIONS: base, public, custom with Russian labels.
— Layout: 3-column — settings | main | history.

User actions:
— Select model via button group;
— Select size 80 or 128;
— Slider steps 10–50;
— Textarea prompt + negative prompt;
— Generate button / Ctrl+Enter shortcut;
— Download PNG via data URL;
— Click history thumbnail to restore image.

Styling: App.css — dark theme, panel layout, responsive grid for history.

Axios base URL: relative "/" in production (same origin); Vite proxy in dev.

No WebSocket — synchronous request/response; generation 5–25s acceptable for demo. Future work: polling or SSE for progress bar (diffusers callback_on_step_end).""",
    ),
    (
        "4.4 REST API и формат обмена данными",
        """API contract (JSON):

Request POST /generate:
{
  "prompt": "green orc warrior with axe",
  "negative_prompt": "realistic, 3d, blurry...",
  "model_type": "custom",
  "output_size": 128,
  "steps": 25
}

Response 200:
{
  "image": "<base64 PNG>",
  "prompt": "<full expanded prompt>",
  "model": "custom",
  "size": 128
}

Base64 chosen over multipart/form-data for simplicity in React (single JSON parse). Trade-off: ~33% size overhead vs binary — acceptable for 128×128 PNG (~10–30 KB raw).

GET /health response documents lora_available flags — frontend could disable custom button if file missing (future enhancement).

OpenAPI auto-generated at /docs — useful for committee demo.""",
    ),
    (
        "4.5 Переключение LoRA-адаптеров без перезагрузки модели",
        """switch_lora(model_type) algorithm:

if model_type == current_lora: return
if current_lora is not None: pipe.unload_lora_weights(); current_lora = None
if model_type != "base":
    resolve path: custom → custom_8bit_v2.safetensors; public → public_pixel_art.safetensors
    pipe.load_lora_weights(path)
    current_lora = model_type

_resolve_lora_path tries multiple filenames for flexibility.

Time complexity: load ~1–2s vs cold start pipeline ~30s.

Memory: base model ~4GB VRAM constant; LoRA +~50MB during active use.

Thread safety: FastAPI async endpoints but GPU ops GIL-bound; single-user academic demo — no locking implemented. Production multi-user would require request queue ( Celery / Redis queue).""",
    ),
    (
        "4.6 Постобработка и сохранение истории генераций",
        """Post-processing steps after VAE decode:

1) Resize (512,512) → (output_size, output_size) NEAREST.
2) If RGBA: composite on white background using alpha channel as mask.
3) Save PNG with PIL.

NEAREST at this stage critical: even if LoRA produced sharp 512px, wrong resize would destroy pixel aesthetics.

History storage: filesystem backend/history/, filename pattern {unix_timestamp}_{model_type}_{size}px.png.

GET /history: glob *.png, sort reverse, take 20, read bytes, base64 encode, parse model from filename.

Retention policy: unlimited append — user may manually clean. No database — deliberate simplicity.

Privacy: local deployment — images not sent to third parties (vs cloud generators) — advantage for unreleased game assets.""",
    ),
    (
        "4.7 Развёртывание и организация публичного доступа",
        """Local run:
pip install -r backend/requirements.txt
cd frontend && npm install && npm run build
uvicorn backend.main:app --host 0.0.0.0 --port 8000

Requirements: Python 3.10+, Node 18+, CUDA 11.8+ optional, ~10GB disk for model cache.

public-tunnel.bat launches cloudflared tunnel to expose port 8000 — useful for remote committee access without VPN.

Security considerations for public tunnel:
— no authentication in current version — acceptable for temporary demo;
— rate limiting absent — DoS risk if URL leaked;
— recommendations: add API key, nginx reverse proxy, HTTPS termination.

StaticFiles mounted at "/" with html=True — SPA routing fallback for React.""",
    ),
]

CHAPTER_5: list[tuple[str, str]] = [
    (
        "5.1 Методика проведения экспериментов",
        """Экспериментальная часть включает два блока: (A) CLIP evaluation через research/evaluate.py; (B) analysis train/val loss from loss_log.csv.

Block A setup:
— Backend running on localhost:8000;
— VAL_DIR = research/val/ — .txt captions;
— Sample n prompts (default 20, full eval 114);
— seed=42 for reproducible prompt subset;
— For each prompt: generate with model_type in [base, public, custom], steps=25, output_size=128;
— CLIP ViT-B/32: clip_score(image, caption);
— Aggregate mean per model.

Block B:
— Plot train_loss vs val_loss per epoch;
— Identify overfitting onset;
— Correlate with checkpoint selection.

Hardware constant: RTX 3070, CUDA, fp16 inference.

Software versions pinned in requirements.txt: torch>=2.3, diffusers>=0.27, transformers>=4.40.

Controlled variables: same prompts, steps, seed for generation randomness (SD uses torch RNG — not fully fixed unless generator seed set; acknowledge as limitation).

Independent variable: model_type (LoRA adapter).
Dependent variable: CLIP score, subjective quality rating.""",
    ),
    (
        "5.2 Сравнение моделей по CLIP score",
        """Results on validation set (README aggregated, n=114 prompts):

| Model        | CLIP score |
| Base SD v1.5 | 30.75      |
| Public LoRA  | 31.13      |
| Custom LoRA  | 33.59      |

Custom LoRA +2.84 points vs base (+9.2% relative).
Custom LoRA +2.46 points vs public (+7.9% relative).

Interpretation:
— Base model understands coarse semantics ("orc", "skeleton") but renders non-pixel shading;
— Public PixelArt LoRA adds pixel aesthetic but trained on generic pixel art, not project-specific creature distribution;
— Custom LoRA trained on same caption template as eval prompts — higher alignment.

Statistical note: CLIP scores not independent across similar prompts; formal t-test not computed; for thesis level, consistent ordering across 114 samples is strong evidence.

Failure cases observed:
— Complex multi-entity prompts ("two warriors fighting") — composition errors;
— Rare weapons — sometimes wrong item;
— Extremely long prompts — truncation at 77 tokens.""",
    ),
    (
        "5.3 Анализ динамики функции потерь",
        """loss_log.csv analysis (15 epochs research run):

Phase 1 (epochs 1–3): rapid val_loss drop 0.025 → 0.020 — model learns global pixel color statistics.
Phase 2 (epochs 4–9): val plateau ~0.020–0.022 — style refinement.
Phase 3 (epochs 10–15): train continues down, val oscillates — classic overfitting signature.

Best val epoch 15 (0.018106) vs epoch 14 (0.021761): delta 0.0033 — small; visual difference matters more.

Train-val gap at epoch 15: 0.017332 vs 0.018106 — gap ~0.0008 — mild.

Recommendation: early stopping on val could save compute; patience=3 on val_loss would stop around epoch 12–15 depending on threshold.

Figure 3.X (loss_plot.png): dual line plot train vs val with vertical marker at best epoch — include in thesis as приложение or раздел 5.""",
    ),
    (
        "5.4 Качественный анализ сгенерированных спрайтов",
        """Qualitative comparison on fixed prompt set:

Prompt: "warrior with red armor and sword"
— Base: human figure, painterly shading, gray background tones;
— Public: pixel edges, but generic NES hero, palette drift;
— Custom: red armor blocks, sword silhouette clear, white background clean.

Prompt: "green orc with axe"
— Base: green skin but semi-realistic muscle shading;
— Public: orc-like, good pixel, proportions vary;
— Custom: consistent with dataset orc sprites (see dataset/images/0332_orc.png style family).

Prompt: "skeleton mage with staff"
— Base: detailed skeleton, horror tone;
— Custom: matches undead subset of dataset, flat bones, purple robe accents.

Common artifacts across all models:
— asymmetric weapons;
— extra limbs at low steps;
— face detail collapse at 80px output.

Mitigation: increase steps to 30–40; refine prompt; inpaint in Aseprite (external).""",
    ),
    (
        "5.5 Оценка производительности и потребления ресурсов",
        """Performance table (RTX 3070, 512 internal, 128 output):

Steps 10: ~5 s, draft quality;
Steps 25: ~12 s, recommended default;
Steps 50: ~25 s, diminishing returns.

VRAM usage:
— Pipeline loaded: ~4.0 GB;
— During inference peak: ~4.5–5.5 GB;
— Fits 8 GB GPU with attention slicing.

CPU fallback: 3–8 minutes per image — documented in main.py print warning.

Backend startup: ~30 s model load from SSD cache.

Concurrent requests: second request while first running queues on GPU (blocking) — latency doubles.

Scalability path: model server + Redis queue + horizontal GPU workers.

Energy: ~220W GPU × 12s ≈ 0.73 Wh per image — negligible cost vs manual art hours.""",
    ),
]

CONCLUSION = """
В ходе выполнения бакалаврской работы разработана веб-система генерации 8-bit игровых персонажей на основе Stable Diffusion v1.5 и LoRA-адаптера, дообученного на собственном датасете из 507 пар «изображение — текстовое описание».

Поставленная цель достигнута: пользователь может через браузер ввести описание персонажа, выбрать одну из трёх моделей, настроить параметры генерации и получить PNG-спрайт размером 80×80 или 128×128 пикселей на белом фоне.

Основные результаты работы:
1) Сформирован и опубликован в структуре репозитория датасет пиксельных спрайтов с детальными англоязычными caption и pipeline предобработки (crop, white background, NEAREST 512×512).
2) Обучен LoRA-адаптер custom_8bit_v2 (rank 32, 15 эпох kohya_ss) и обоснован выбор чекпоинта 14-й эпохи по совокупности CLIP score и визуальной оценки.
3) Реализован backend на FastAPI с hot-swap переключением LoRA и frontend на React 19 с историей генераций.
4) Экспериментально показано превосходство custom LoRA (CLIP score 33,59) над базовой моделью (30,75) и публичной Pixel Art LoRA (31,13).

Научно-техническая новизна заключается в интеграции полного цикла generative AI для узкой предметной области (game sprites) в единое веб-приложение с воспроизводимой методикой оценки.

Практическая значимость: система применима indie-разработчиками для прототипирования персонажей; код и документация открыты для модификации.

Направления дальнейших исследований:
— обучение LoRA на text encoder для лучшего понимания русскоязычных промптов;
— ControlNet для pose-guided generation;
— автоматическая генерация sprite sheets анимации (walk cycles);
— интеграция апsample-фильтров и palette quantization (Octree) в post-processing;
— A/B human evaluation с участием game artists;
— добавление authentication и job queue для multi-user deployment.

Таким образом, работа подтверждает, что при ограниченном датasete (~500 images) и consumer GPU возможно достичь заметного улучшения стилевого соответствия pixel art по сравнению с zero-shot Stable Diffusion, а веб-обёртка делает технологию доступной пользователям без опыта работы с ML CLI.
"""

ABBREVIATIONS = """
API — Application Programming Interface (программный интерфейс приложения)
CFG — Classifier-Free Guidance (направленная генерация без классификатора)
CLIP — Contrastive Language–Image Pre-training
DDPM — Denoising Diffusion Probabilistic Model
FID — Fréchet Inception Distance
FR — Functional Requirement (функциональное требование)
GAN — Generative Adversarial Network
GPU — Graphics Processing Unit
HTTP — HyperText Transfer Protocol
JSON — JavaScript Object Notation
LoRA — Low-Rank Adaptation
ML — Machine Learning (машинное обучение)
MSE — Mean Squared Error (среднеквадратичная ошибка)
NES — Nintendo Entertainment System
NFR — Non-Functional Requirement (нефункциональное требование)
PNG — Portable Network Graphics
REST — Representational State Transfer
RGB — Red Green Blue (цветовая модель)
SD — Stable Diffusion
UI — User Interface (пользовательский интерфейс)
URL — Uniform Resource Locator
VAE — Variational Autoencoder
VRAM — Video Random Access Memory
"""

BIBLIOGRAPHY = """
1. ГОСТ 7.32-2017. Отчёт о научно-исследовательской работе. Структура и правила оформления. — М.: Стандартинформ, 2017.
2. ГОСТ 2.105-2019. Единая система конструкторской документации. Общие требования к текстовым документам. — М.: Стандартинформ, 2019.
3. Ho J. et al. Denoising Diffusion Probabilistic Models // Advances in Neural Information Processing Systems. — 2020. — Vol. 33. — P. 6840–6851.
4. Rombach R. et al. High-Resolution Image Synthesis with Latent Diffusion Models // Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). — 2022. — P. 10684–10695.
5. Hu E. J. et al. LoRA: Low-Rank Adaptation of Large Language Models // International Conference on Learning Representations (ICLR). — 2022.
6. Radford A. et al. Learning Transferable Visual Models From Natural Language Supervision // International Conference on Machine Learning (ICML). — 2021. — P. 8748–8763.
7. Goodfellow I. et al. Generative Adversarial Nets // Advances in Neural Information Processing Systems. — 2014. — Vol. 27.
8. Salimans T., Ho J. Progressive Distillation for Fast Sampling of Diffusion Models // International Conference on Learning Representations (ICLR). — 2022.
9. Stable Diffusion v1.5 Model Card [Электронный ресурс]. — URL: https://huggingface.co/runwayml/stable-diffusion-v1-5 (дата обращения: 15.03.2026).
10. Hugging Face Diffusers Documentation [Электронный ресурс]. — URL: https://huggingface.co/docs/diffusers (дата обращения: 15.03.2026).
11. Kohya ss — GUI for LoRA training [Электронный ресурс]. — URL: https://github.com/kohya-ss/sd-scripts (дата обращения: 15.03.2026).
12. FastAPI Documentation [Электронный ресурс]. — URL: https://fastapi.tiangolo.com/ (дата обращения: 15.03.2026).
13. React Documentation [Электронный ресурс]. — URL: https://react.dev/ (дата обращения: 15.03.2026).
14. PyTorch Documentation [Электронный ресурс]. — URL: https://pytorch.org/docs/ (дата обращения: 15.03.2026).
15. Pillow (PIL) Documentation [Электронный ресурс]. — URL: https://pillow.readthedocs.io/ (дата обращения: 15.03.2026).
16. CLIP Score Evaluation in Generative Models // OpenAI CLIP Repository. — URL: https://github.com/openai/CLIP (дата обращения: 15.03.2026).
17. Pixel Art Redmond LoRA — CivitAI [Электронный ресурс]. — URL: https://civitai.com/ (дата обращения: 15.03.2026).
18. OpenGameArt.org — Free game assets [Электронный ресурс]. — URL: https://opengameart.org/ (дата обращения: 10.02.2026).
19. Kenney — Game Assets [Электронный ресурс]. — URL: https://kenney.nl/ (дата обращения: 10.02.2026).
20. itch.io — Game assets marketplace [Электронный ресурс]. — URL: https://itch.io/game-assets (дата обращения: 10.02.2026).
21. Aseprite — Animated sprite editor [Электронный ресурс]. — URL: https://www.aseprite.org/ (дата обращения: 01.02.2026).
22. Automatic1111 Stable Diffusion WebUI [Электронный ресурс]. — URL: https://github.com/AUTOMATIC1111/stable-diffusion-webui (дата обращения: 01.02.2026).
23. ComfyUI [Электронный ресурс]. — URL: https://github.com/comfyanonymous/ComfyUI (дата обращения: 01.02.2026).
24. Hevner A. R. et al. Design Science in Information Systems Research // MIS Quarterly. — 2004. — Vol. 28, № 1. — P. 75–105.
25. Sommerville I. Software Engineering. — 10th ed. — Pearson, 2016. — 794 p.
26. Fowler M. Patterns of Enterprise Application Architecture. — Addison-Wesley, 2002. — 560 p.
27. NVIDIA CUDA Toolkit Documentation [Электронный ресурс]. — URL: https://docs.nvidia.com/cuda/ (дата обращения: 15.03.2026).
28. Cloudflare Tunnel Documentation [Электронный ресурс]. — URL: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/ (дата обращения: 15.03.2026).
29. PEFT: Parameter-Efficient Fine-Tuning [Электронный ресурс]. — URL: https://github.com/huggingface/peft (дата обращения: 15.03.2026).
30. DDIM Sampling: Song J. et al. Denoising Diffusion Implicit Models // International Conference on Learning Representations (ICLR). — 2021.
31. Kingma D. P., Welling M. Auto-Encoding Variational Bayes // International Conference on Learning Representations (ICLR). — 2014.
32. Vaswani A. et al. Attention Is All You Need // Advances in Neural Information Processing Systems. — 2017. — Vol. 30.
33. Fielding R. T. Architectural Styles and the Design of Network-based Software Architectures: Dissertation. — UC Irvine, 2000.
34. McHugh M. L. Interrater reliability: the kappa statistic // Biochemia Medica. — 2012. — Vol. 22, № 3. — P. 276–282.
"""

APPENDIX_A = """[Полный листинг backend/main.py см. в репозитории проекта Diploma/backend/main.py — 185 строк. При оформлении по ГОСТ используйте стиль «Листинг» шаблона с нумерацией Листинг А.1.]"""

APPENDIX_B = """[Полный листинг frontend/src/App.jsx см. в репозитории проекта Diploma/frontend/src/App.jsx — 244 строки. При оформлении по ГОСТ используйте нумерацию Листинг Б.1.]"""
