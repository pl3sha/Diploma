from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

RAW_DIR = Path("raw")
OUT_DIR = Path("images")
MIN_SIDE = 8
TARGET = 512


def _preprocess(src: Path, min_side: int) -> Image.Image | None:
    img = Image.open(src)
    if img.width < min_side or img.height < min_side:
        return None
    if img.width != img.height:
        side = min(img.width, img.height)
        left = (img.width - side) // 2
        top = (img.height - side) // 2
        img = img.crop((left, top, left + side, top + side))
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        background = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "P":
            img = img.convert("RGBA")
        background.paste(img, mask=img.split()[-1])
        img = background
    else:
        img = img.convert("RGB")
    return img.resize((TARGET, TARGET), Image.NEAREST)


def main() -> None:
    p = argparse.ArgumentParser(description="raw PNG -> 512x512 RGB in images/")
    p.add_argument("--append", action="store_true")
    p.add_argument("--min-side", type=int, default=MIN_SIDE)
    args = p.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sources = sorted(RAW_DIR.glob("*.png"))
    start = len(list(OUT_DIR.glob("*.png"))) if args.append else 0
    saved = 0
    skipped = 0
    for src in sources:
        img = _preprocess(src, args.min_side)
        if img is None:
            skipped += 1
            continue
        img.save(OUT_DIR / f"{start + saved:04d}_{src.stem}.png")
        saved += 1
    print(f"raw: {len(sources)} saved: {saved} skipped: {skipped} -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
