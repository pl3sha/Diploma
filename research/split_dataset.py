from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

IMAGES_DIR = Path("images")
TRAIN_DIR = Path("train")
VAL_DIR = Path("val")


def split(val_ratio: float = 0.30, seed: int = 42) -> None:
    pairs: list[tuple[Path, Path]] = []
    for img in sorted(IMAGES_DIR.glob("*.png")):
        txt = img.with_suffix(".txt")
        if txt.exists():
            pairs.append((img, txt))

    if not pairs:
        print(f"No image-caption pairs found in {IMAGES_DIR}/")
        return

    random.seed(seed)
    random.shuffle(pairs)

    n_val = max(1, round(len(pairs) * val_ratio))
    n_train = len(pairs) - n_val
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    for d in (TRAIN_DIR, VAL_DIR):
        d.mkdir(exist_ok=True)
        for f in d.iterdir():
            f.unlink()

    for img, txt in train_pairs:
        shutil.copy(img, TRAIN_DIR / img.name)
        shutil.copy(txt, TRAIN_DIR / txt.name)

    for img, txt in val_pairs:
        shutil.copy(img, VAL_DIR / img.name)
        shutil.copy(txt, VAL_DIR / txt.name)

    print(f"Total pairs : {len(pairs)}")
    print(f"Train ({1 - val_ratio:.0%})   : {n_train} pairs -> {TRAIN_DIR}/")
    print(f"Val   ({val_ratio:.0%})   : {n_val} pairs  -> {VAL_DIR}/")
    print(f"Seed        : {seed}")


def main() -> None:
    p = argparse.ArgumentParser(description="Split dataset into train/val subsets")
    p.add_argument("--val", type=float, default=0.30,
                   help="Validation ratio (default: 0.30)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility (default: 42)")
    args = p.parse_args()

    if not 0.0 < args.val < 1.0:
        print("Error: --val must be between 0 and 1")
        return

    split(val_ratio=args.val, seed=args.seed)


if __name__ == "__main__":
    main()
