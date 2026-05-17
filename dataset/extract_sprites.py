from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image


def extract_by_alpha(img: Image.Image, min_pixels: int = 16, padding: int = 2) -> list[Image.Image]:
    rgba = img.convert("RGBA")
    pixels = rgba.load()
    w, h = rgba.size

    visited = [[False] * h for _ in range(w)]
    sprites = []

    for start_x in range(w):
        for start_y in range(h):
            _, _, _, a = pixels[start_x, start_y]
            if a < 10 or visited[start_x][start_y]:
                continue

            stack = [(start_x, start_y)]
            xs, ys = [], []

            while stack:
                cx, cy = stack.pop()
                if cx < 0 or cy < 0 or cx >= w or cy >= h:
                    continue
                if visited[cx][cy]:
                    continue
                _, _, _, ca = pixels[cx, cy]
                if ca < 10:
                    continue
                visited[cx][cy] = True
                xs.append(cx)
                ys.append(cy)
                stack.extend([(cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)])

            if len(xs) < min_pixels:
                continue

            x0 = max(0, min(xs) - padding)
            y0 = max(0, min(ys) - padding)
            x1 = min(w, max(xs) + padding + 1)
            y1 = min(h, max(ys) + padding + 1)

            sprites.append(rgba.crop((x0, y0, x1, y1)))

    return sprites


def extract_by_grid(img: Image.Image, cell_w: int, cell_h: int) -> list[Image.Image]:
    sprites = []
    img_w, img_h = img.size
    rgba = img.convert("RGBA")

    for y in range(0, img_h - cell_h + 1, cell_h):
        for x in range(0, img_w - cell_w + 1, cell_w):
            cell = rgba.crop((x, y, x + cell_w, y + cell_h))
            non_transparent = sum(1 for px in cell.getdata() if px[3] >= 10)
            if non_transparent < 4:
                continue
            sprites.append(cell)

    return sprites


def save_sprites(sprites: list[Image.Image], out_dir: Path, prefix: str = "sprite") -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, sprite in enumerate(sprites):
        sprite.save(out_dir / f"{prefix}_{i:04d}.png")
    return len(sprites)


def main():
    parser = argparse.ArgumentParser(description="Extract sprites from a sprite sheet")
    parser.add_argument("image", help="Path to the sprite sheet (PNG)")
    parser.add_argument("--mode", choices=["alpha", "grid"], default="alpha",
                        help="alpha: flood-fill on opaque pixels (default); grid: fixed cell size")
    parser.add_argument("--out", default="raw", help="Output directory")
    parser.add_argument("--prefix", default="sprite", help="Output filename prefix")
    parser.add_argument("--min-pixels", type=int, default=16,
                        help="Minimum opaque pixel count to keep a sprite (alpha mode)")
    parser.add_argument("--padding", type=int, default=2,
                        help="Padding around bounding box in pixels (alpha mode)")
    parser.add_argument("--cell-w", type=int, default=16, help="Cell width in pixels (grid mode)")
    parser.add_argument("--cell-h", type=int, default=16, help="Cell height in pixels (grid mode)")

    args = parser.parse_args()

    src = Path(args.image)
    if not src.exists():
        print(f"Error: file not found: {src}")
        sys.exit(1)

    img = Image.open(src)
    print(f"Loaded: {src.name}  ({img.size[0]}x{img.size[1]})")

    if args.mode == "alpha":
        sprites = extract_by_alpha(img, args.min_pixels, args.padding)
        print(f"Mode: alpha flood-fill  min_pixels={args.min_pixels}  padding={args.padding}")
    else:
        sprites = extract_by_grid(img, args.cell_w, args.cell_h)
        print(f"Mode: grid  cell={args.cell_w}x{args.cell_h}")

    saved = save_sprites(sprites, Path(args.out), args.prefix)
    print(f"Saved {saved} sprite(s) -> {args.out}/")


if __name__ == "__main__":
    main()
