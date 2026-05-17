"""
Кастомный цикл обучения LoRA для Stable Diffusion v1.5.

Преимущества перед kohya_ss:
  - Реальный val_loss после каждой эпохи (модель НЕ видит val/ при обучении)
  - CSV лог: dataset/loss_log.csv
  - График: dataset/loss_plot.png
  - Полный контроль над процессом

Зависимости (уже в backend/requirements.txt):
    pip install peft safetensors tqdm matplotlib

Использование:
    python train_lora.py
    python train_lora.py --epochs 15 --repeats 14 --rank 32 --lr 1e-4
"""
from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

BASE_DIR   = Path(__file__).parent
TRAIN_DIR  = BASE_DIR / "dataset" / "train"
VAL_DIR    = BASE_DIR / "dataset" / "val"
OUTPUT_DIR = BASE_DIR / "backend" / "lora" / "custom_peft"
LOG_FILE   = BASE_DIR / "dataset" / "loss_log.csv"
PLOT_FILE  = BASE_DIR / "dataset" / "loss_plot.png"
MODEL_ID   = "runwayml/stable-diffusion-v1-5"

IMAGE_TRANSFORMS = transforms.Compose([
    transforms.Resize(512, interpolation=transforms.InterpolationMode.NEAREST),
    transforms.CenterCrop(512),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])


# ── Датасет ───────────────────────────────────────────────────────────

class SpriteDataset(Dataset):
    def __init__(self, image_dir: Path, tokenizer, repeats: int = 1) -> None:
        pairs: list[tuple[Path, str]] = []
        for img_path in sorted(image_dir.glob("*.png")):
            txt_path = img_path.with_suffix(".txt")
            if txt_path.exists():
                caption = txt_path.read_text(encoding="utf-8").strip()
                if caption:
                    pairs.append((img_path, caption))
        # repeats — аналог num_repeats в kohya: каждая картинка N раз за эпоху
        self.pairs = pairs * repeats
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        img_path, caption = self.pairs[idx]
        image = Image.open(img_path).convert("RGB")
        tokens = self.tokenizer(
            caption,
            max_length=77,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "pixel_values": IMAGE_TRANSFORMS(image),
            "input_ids": tokens.input_ids.squeeze(0),
        }


# ── Loss ──────────────────────────────────────────────────────────────

def compute_loss(batch, unet, vae, text_encoder, scheduler, device, dtype,
                 noise_offset: float = 0.1) -> torch.Tensor:
    pixel_values = batch["pixel_values"].to(device, dtype=dtype)
    input_ids    = batch["input_ids"].to(device)

    with torch.no_grad():
        latents               = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
        encoder_hidden_states = text_encoder(input_ids)[0]

    noise = torch.randn_like(latents)
    # noise offset улучшает качество фона и однородных областей (стандарт в kohya)
    if noise_offset > 0:
        noise += noise_offset * torch.randn(latents.shape[0], latents.shape[1], 1, 1,
                                             device=device, dtype=dtype)

    timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
    noisy     = scheduler.add_noise(latents, noise, timesteps)
    pred      = unet(noisy, timesteps, encoder_hidden_states).sample

    target = noise if scheduler.config.prediction_type == "epsilon" else \
             scheduler.get_velocity(latents, noise, timesteps)

    return F.mse_loss(pred.float(), target.float())


# ── График loss ───────────────────────────────────────────────────────

def save_plot(log_rows: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt

        epochs       = [r["epoch"]       for r in log_rows]
        train_losses = [r["train_loss"]  for r in log_rows]
        val_losses   = [r["val_loss"]    for r in log_rows]

        best_epoch = min(log_rows, key=lambda r: r["val_loss"])["epoch"]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(epochs, train_losses, marker="o", label="Train Loss")
        ax.plot(epochs, val_losses,   marker="s", label="Val Loss")
        ax.axvline(best_epoch, color="gray", linestyle="--", alpha=0.6, label=f"Лучший val (эпоха {best_epoch})")
        ax.set_xlabel("Эпоха")
        ax.set_ylabel("MSE Loss")
        ax.set_title("LoRA Training — Train / Val Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(PLOT_FILE), dpi=150)
        plt.close(fig)
        print(f"График сохранён: {PLOT_FILE}")
    except ImportError:
        print("matplotlib не найден, график не создан. Установите: pip install matplotlib")


# ── Конвертация PEFT → kohya ──────────────────────────────────────────

def _peft_to_kohya(peft_sd: dict, prefix: str, lora_alpha: int) -> dict:
    """
    Конвертирует PEFT LoRA state dict → kohya-формат.

    prefix: 'lora_unet' для UNet, 'lora_te' для text encoder.
    Входные ключи: [unet.|text_encoder.]base_model.model....lora_A.weight
    Выходные ключи: {prefix}_layer_name.lora_[down|up].weight

    Поддерживаемые форматы PEFT ключей:
      .lora_A.weight          (PEFT >= 0.6, текущий)
      .lora_A.default.weight  (PEFT < 0.6, устаревший)
      .lora.down.weight       (diffusers-native формат)
    """
    kohya: dict = {}
    seen: set = set()
    for key, value in peft_sd.items():
        key = (key
               .replace("unet.base_model.model.", "")
               .replace("text_encoder.base_model.model.", "")
               .replace("base_model.model.", ""))
        # Определяем тип ключа через endswith (точное совпадение суффикса)
        if key.endswith(".lora_A.weight") or key.endswith(".lora_A.default.weight") or key.endswith(".lora.down.weight"):
            layer = (key
                     .removesuffix(".lora_A.weight")
                     .removesuffix(".lora_A.default.weight")
                     .removesuffix(".lora.down.weight"))
            k = prefix + "_" + layer.replace(".", "_")
            kohya[k + ".lora_down.weight"] = value.contiguous().to(torch.float32)
            seen.add(k)
        elif key.endswith(".lora_B.weight") or key.endswith(".lora_B.default.weight") or key.endswith(".lora.up.weight"):
            layer = (key
                     .removesuffix(".lora_B.weight")
                     .removesuffix(".lora_B.default.weight")
                     .removesuffix(".lora.up.weight"))
            k = prefix + "_" + layer.replace(".", "_")
            kohya[k + ".lora_up.weight"] = value.contiguous().to(torch.float32)
    for k in seen:
        kohya[k + ".alpha"] = torch.tensor(float(lora_alpha))
    return kohya


def _save_kohya_unet(unet_peft, output_path: Path, lora_alpha: int) -> None:
    """Сохраняет UNet LoRA в kohya-совместимый safetensors файл."""
    from safetensors.torch import save_file
    from peft import get_peft_model_state_dict

    kohya_sd = _peft_to_kohya(get_peft_model_state_dict(unet_peft), "lora_unet", lora_alpha)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(kohya_sd, str(output_path))
    size_mb = output_path.stat().st_size // 1024 // 1024
    print(f"  Ключей: {len(kohya_sd)}  |  Размер: {size_mb} МБ")


# ── Обучение ──────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(args: argparse.Namespace) -> None:
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from peft import LoraConfig, get_peft_model
    from transformers import CLIPTextModel, CLIPTokenizer

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.float16
    print(f"Устройство : {device}")
    print(f"Train      : {len(list(TRAIN_DIR.glob('*.png')))} изображений")
    print(f"Val        : {len(list(VAL_DIR.glob('*.png')))} изображений")
    print(f"Seed       : {args.seed}")

    # ── Загрузка компонентов SD v1.5 ─────────────────────────────────
    print("\nЗагрузка моделей SD v1.5...")
    tokenizer    = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder").to(device, dtype=dtype)
    vae          = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae").to(device, dtype=dtype)
    unet         = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet").to(device, dtype=dtype)
    scheduler    = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)   # заморожен полностью (как в kohya по умолчанию)

    # ── LoRA только на UNet (attention-слои) — как kohya ─────────────
    unet_lora_cfg = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank // 2,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.0,
        bias="none",
    )
    unet = get_peft_model(unet, unet_lora_cfg)
    unet.print_trainable_parameters()

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # ── Датасеты ──────────────────────────────────────────────────────
    train_ds = SpriteDataset(TRAIN_DIR, tokenizer, repeats=args.repeats)
    val_ds   = SpriteDataset(VAL_DIR,   tokenizer, repeats=1)   # val без повторов
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    steps_per_epoch = len(train_loader)
    total_steps     = steps_per_epoch * args.epochs
    print(f"\nШагов за эпоху : {steps_per_epoch}")
    print(f"Всего шагов    : {total_steps}  ({args.epochs} эпох × {args.repeats} повторов)")

    # ── Оптимизатор ───────────────────────────────────────────────────
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(filter(lambda p: p.requires_grad, unet.parameters()), lr=args.lr)
        print(f"Оптимизатор    : AdamW8bit  lr={args.lr:.1e}")
    except ImportError:
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, unet.parameters()), lr=args.lr)
        print(f"Оптимизатор    : AdamW  lr={args.lr:.1e}")

    # ── Cosine LR scheduler с warmup (как в kohya) ───────────────────
    total_steps     = (len(list(TRAIN_DIR.glob("*.png"))) * args.repeats // args.batch_size) * args.epochs
    warmup_steps    = total_steps // 20   # 5% шагов — warmup
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    warmup_sched = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=args.lr * 0.1)
    scheduler_lr = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_steps])
    print(f"LR scheduler   : cosine с warmup ({warmup_steps} шагов), всего {total_steps} шагов")

    # ИСПРАВЛЕНИЕ: совместимость GradScaler с PyTorch < 2.0 и >= 2.0
    try:
        scaler = torch.amp.GradScaler("cuda")
    except TypeError:
        scaler = torch.cuda.amp.GradScaler()

    # ── Цикл обучения ─────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_rows: list[dict] = []

    print(f"\n{'─'*60}")
    print(f"{'Эпоха':>6}  {'train_loss':>12}  {'val_loss':>10}  {'время':>8}")
    print(f"{'─'*60}")

    for epoch in range(1, args.epochs + 1):

        # ── Train ────────────────────────────────────────────────────
        unet.train()
        train_sum = 0.0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Эпоха {epoch:>2}/{args.epochs} [train]", leave=False, ncols=80)
        for batch in pbar:
            with torch.autocast("cuda", dtype=dtype):
                loss = compute_loss(batch, unet, vae, text_encoder, scheduler, device, dtype,
                                    noise_offset=args.noise_offset)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler_lr.step()
            train_sum += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        pbar.close()

        train_loss = train_sum / len(train_loader)

        # ── Validation ───────────────────────────────────────────────
        unet.eval()
        val_sum = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Эпоха {epoch:>2}/{args.epochs} [val]  ", leave=False, ncols=80):
                with torch.autocast("cuda", dtype=dtype):
                    val_sum += compute_loss(batch, unet, vae, text_encoder, scheduler, device, dtype,
                                            noise_offset=0.0).item()

        val_loss = val_sum / len(val_loader)
        elapsed  = time.time() - t0

        print(f"{epoch:>6}  {train_loss:>12.4f}  {val_loss:>10.4f}  {elapsed:>6.0f}s")
        log_rows.append({"epoch": epoch, "train_loss": round(train_loss, 6), "val_loss": round(val_loss, 6)})

        # ── Чекпоинт в kohya-формат (UNet) ──────────────────────────
        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt_path = OUTPUT_DIR.parent / f"custom_peft_ep{epoch:02d}.safetensors"
            _save_kohya_unet(unet, ckpt_path, lora_alpha=args.rank // 2)
            print(f"         → чекпоинт: {ckpt_path.name}")

    # ── Финальное сохранение в kohya-формат ──────────────────────────
    # pipe.load_lora_weights() понимает kohya-формат надёжнее всего.
    # Конвертируем вручную: PEFT-ключи → kohya-ключи.
    print("\nКонвертация в kohya-формат (UNet)...")
    _save_kohya_unet(unet, OUTPUT_DIR / "pytorch_lora_weights.safetensors", lora_alpha=args.rank // 2)
    print(f"Модель сохранена : {OUTPUT_DIR}/pytorch_lora_weights.safetensors")

    # ── CSV лог ───────────────────────────────────────────────────────
    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"Лог              : {LOG_FILE}")

    # ── График ────────────────────────────────────────────────────────
    save_plot(log_rows)

    # ── Итоговая таблица ──────────────────────────────────────────────
    best_val = min(log_rows, key=lambda r: r["val_loss"])
    print(f"\n{'─'*60}")
    print(f"{'Эпоха':>6}  {'train_loss':>12}  {'val_loss':>10}")
    print(f"{'─'*60}")
    for r in log_rows:
        flag = "  ← лучший val" if r["epoch"] == best_val["epoch"] else ""
        print(f"{r['epoch']:>6}  {r['train_loss']:>12.4f}  {r['val_loss']:>10.4f}{flag}")
    print(f"\nЛучший val_loss {best_val['val_loss']:.4f} на эпохе {best_val['epoch']}")
    print(f"Используйте чекпоинт: custom_peft_ep{best_val['epoch']:02d}/")


# ── CLI ───────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Обучение LoRA с train/val мониторингом")
    p.add_argument("--epochs",     type=int,   default=15,   help="Число эпох")
    p.add_argument("--repeats",    type=int,   default=14,   help="Повторов каждой картинки за эпоху (num_repeats)")
    p.add_argument("--rank",       type=int,   default=32,   help="LoRA rank (network_dim)")
    p.add_argument("--lr",         type=float, default=5e-5, help="Learning rate")
    p.add_argument("--noise-offset", type=float, default=0.1, dest="noise_offset",
                   help="Noise offset для улучшения фона (0 = выкл)")
    p.add_argument("--batch-size", type=int,   default=2,    dest="batch_size")
    p.add_argument("--save-every", type=int,   default=5,    dest="save_every", help="Сохранять чекпоинт каждые N эпох")
    p.add_argument("--seed",       type=int,   default=42,   help="Random seed для воспроизводимости")
    p.add_argument("--no-gradient-checkpointing", action="store_false",
                   dest="gradient_checkpointing", default=True)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
