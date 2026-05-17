from __future__ import annotations

import argparse
import base64
import io
import random
import time
from pathlib import Path

import requests
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

VAL_DIR = Path(__file__).parent / "val"
API_URL = "http://127.0.0.1:8000/generate"
MODEL_TYPES = ["base", "public", "custom"]
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"


def load_clip() -> tuple[CLIPModel, CLIPProcessor]:
    print("Loading CLIP model...")
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    model = CLIPModel.from_pretrained(CLIP_MODEL_ID)
    model.eval()
    return model, processor


def clip_score(
    model: CLIPModel,
    processor: CLIPProcessor,
    image: Image.Image,
    text: str,
) -> float:
    inputs = processor(text=[text], images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        score = outputs.logits_per_image[0][0].item()
    return round(float(score), 2)


def generate_image(prompt: str, model_type: str, steps: int) -> Image.Image | None:
    payload = {
        "prompt": prompt,
        "model_type": model_type,
        "output_size": 128,
        "steps": steps,
    }
    try:
        r = requests.post(API_URL, json=payload, timeout=120)
        r.raise_for_status()
        img_bytes = base64.b64decode(r.json()["image"])
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        print(f"  Generation error ({model_type}): {e}")
        return None


def load_val_pairs(n: int, seed: int) -> list[tuple[str, str]]:
    txts = sorted(VAL_DIR.glob("*.txt"))
    if not txts:
        raise FileNotFoundError(f"No .txt files found in {VAL_DIR}")
    random.seed(seed)
    sample = random.sample(txts, min(n, len(txts)))
    pairs = []
    for txt in sample:
        caption = txt.read_text(encoding="utf-8").strip()
        if caption:
            pairs.append((txt.stem, caption))
    return pairs


def run_evaluation(n: int, steps: int, seed: int) -> None:
    if not VAL_DIR.exists():
        print(f"Validation directory not found: {VAL_DIR}")
        print("Run split_dataset.py first.")
        return

    clip_model, clip_processor = load_clip()
    pairs = load_val_pairs(n, seed)
    print(f"\nEvaluating {len(pairs)} prompt(s) from val/  steps={steps}\n")

    results: dict[str, list[float]] = {m: [] for m in MODEL_TYPES}
    col_w = 10

    header = f"{'File':<35}" + "".join(f"{m:>{col_w}}" for m in MODEL_TYPES)
    print(header)
    print("-" * len(header))

    for stem, caption in pairs:
        row = f"{stem[:33]:<35}"
        for model_type in MODEL_TYPES:
            img = generate_image(caption, model_type, steps)
            if img is None:
                row += f"{'ERR':>{col_w}}"
                continue
            score = clip_score(clip_model, clip_processor, img, caption)
            results[model_type].append(score)
            row += f"{score:>{col_w}.2f}"
            time.sleep(0.2)
        print(row)

    print("-" * len(header))
    avg_row = f"{'Average':<35}"
    for model_type in MODEL_TYPES:
        scores = results[model_type]
        avg = sum(scores) / len(scores) if scores else 0.0
        avg_row += f"{avg:>{col_w}.2f}"
    print(avg_row)
    print()

    best = max(MODEL_TYPES, key=lambda m: sum(results[m]) / len(results[m]) if results[m] else 0)
    print(f"Best model by CLIP score: {best}")


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate models using CLIP score on validation set")
    p.add_argument("--n", type=int, default=20, help="Number of prompts to evaluate (default: 20)")
    p.add_argument("--steps", type=int, default=25, help="Inference steps (default: 25)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = p.parse_args()
    run_evaluation(n=args.n, steps=args.steps, seed=args.seed)


if __name__ == "__main__":
    main()
