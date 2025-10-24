"""
Synthetic dataset generator for CI.
Creates a tiny weather dataset under the given data_path with 5 classes:
cloudy, foggy, rainy, snowy, sunny. Each class gets N images.
"""

from pathlib import Path
from typing import List, Tuple
import argparse
import random

from PIL import Image, ImageDraw, ImageFilter

CLASSES = ["cloudy", "foggy", "rainy", "snowy", "sunny"]


def make_image(size: Tuple[int, int], kind: str) -> Image.Image:
    w, h = size
    img = Image.new("RGB", size, (128, 128, 128))
    draw = ImageDraw.Draw(img)

    if kind == "sunny":
        # bright yellow sun
        img = Image.new("RGB", size, (255, 230, 100))
        draw = ImageDraw.Draw(img)
        r = min(w, h) // 4
        cx, cy = w // 2, h // 2
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(255, 210, 0))
    elif kind == "cloudy":
        # gray clouds
        img = Image.new("RGB", size, (180, 180, 180))
        draw = ImageDraw.Draw(img)
        for _ in range(8):
            x = random.randint(0, w)
            y = random.randint(h // 3, h)
            r = random.randint(10, 30)
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(200, 200, 200))
    elif kind == "rainy":
        # diagonal blue lines to mimic rain
        img = Image.new("RGB", size, (100, 100, 120))
        draw = ImageDraw.Draw(img)
        for x in range(0, w, 8):
            draw.line((x, 0, x - 10, h), fill=(80, 80, 200), width=1)
    elif kind == "foggy":
        # light gray with blur
        img = Image.new("RGB", size, (200, 200, 200))
        img = img.filter(ImageFilter.GaussianBlur(radius=3))
    elif kind == "snowy":
        # white background with small gray dots
        img = Image.new("RGB", size, (240, 240, 240))
        draw = ImageDraw.Draw(img)
        for _ in range(200):
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)
            draw.point((x, y), fill=(220, 220, 220))

    return img


def generate_dataset(data_path: Path, target_size: Tuple[int, int], images_per_class: int) -> None:
    data_path.mkdir(parents=True, exist_ok=True)
    for cls in CLASSES:
        cls_dir = data_path / cls
        cls_dir.mkdir(exist_ok=True)
        for i in range(images_per_class):
            img = make_image(target_size, cls)
            img.save(cls_dir / f"synthetic_{i+1}.jpg", format="JPEG", quality=90)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic weather dataset for CI")
    parser.add_argument("--data_path", type=str, default="../../data", help="Root data directory")
    parser.add_argument("--target_size", nargs=2, type=int, default=[128, 128], help="Image size W H")
    parser.add_argument("--images_per_class", type=int, default=20, help="Number of images per class")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    target_size = (args.target_size[0], args.target_size[1])

    # If data already exists with at least one image, skip generation
    if data_path.exists():
        existing = list(data_path.glob("**/*.jpg")) + list(data_path.glob("**/*.png"))
        if len(existing) > 0:
            print(f"Data exists at {data_path}, found {len(existing)} images. Skipping generation.")
            return

    generate_dataset(data_path, target_size, args.images_per_class)
    print(f"Synthetic dataset generated at {data_path}")


if __name__ == "__main__":
    main()