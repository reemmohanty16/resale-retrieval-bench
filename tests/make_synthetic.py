"""Create a small synthetic dataset in the DeepFashion Consumer-to-Shop layout for pipeline tests.

    python tests/make_synthetic.py --root /tmp/c2s_synth --items 60 [--identical]

Each item gets a distinctive colour/stripe pattern. Shop images are clean renders; consumer images are
perturbed (shift, brightness, noise) unless --identical, in which case consumer images are exact copies
of a shop image (so any deterministic embedder must reach Recall@1 = 1.0 — the evaluation check).
"""

from __future__ import annotations

import argparse
import os
import random

import numpy as np
from PIL import Image, ImageDraw

GROUPS = {"CLOTHING": ["Tees", "Dresses", "Jeans"], "TROUSERS": ["Pants"]}
TYPES = {"Tees": 1, "Dresses": 3, "Jeans": 2, "Pants": 2}


def render(seed: int, size=(120, 160)) -> Image.Image:
    rng = random.Random(seed)
    img = Image.new("RGB", size, (rng.randint(150, 255), rng.randint(150, 255), rng.randint(150, 255)))
    d = ImageDraw.Draw(img)
    base = (rng.randint(0, 200), rng.randint(0, 200), rng.randint(0, 200))
    d.rectangle([20, 30, 100, 140], fill=base)
    stripe = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
    for y in range(30, 140, rng.randint(8, 20)):
        d.line([(20, y), (100, y)], fill=stripe, width=rng.randint(1, 4))
    if rng.random() < 0.5:
        d.ellipse([40, 60, 80, 100], fill=(rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)))
    return img


def perturb(img: Image.Image, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    a = np.asarray(img).astype(np.float32)
    a = a * rng.uniform(0.7, 1.3) + rng.normal(0, 12, a.shape)
    a = np.clip(a, 0, 255).astype(np.uint8)
    out = Image.fromarray(a)
    dx, dy = int(rng.integers(-10, 11)), int(rng.integers(-10, 11))
    canvas = Image.new("RGB", out.size, (int(rng.integers(0, 255)),) * 3)
    canvas.paste(out, (dx, dy))
    return canvas


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--items", type=int, default=60)
    p.add_argument("--identical", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    rng = random.Random(a.seed)
    os.makedirs(os.path.join(a.root, "Anno"), exist_ok=True)
    os.makedirs(os.path.join(a.root, "Eval"), exist_ok=True)
    bbox_rows, part_rows = [], []
    cats = [(g, c) for g, cs in GROUPS.items() for c in cs]
    for i in range(a.items):
        item = f"id_{i + 1:08d}"
        g, c = cats[i % len(cats)]
        part = ["train", "val", "test"][i % 3] if i >= 6 else "test"   # ensure plenty of test items
        d = os.path.join(a.root, "img", g, c, item)
        os.makedirs(d, exist_ok=True)
        n_shop = rng.choice([1, 2, 2, 3])
        shop_imgs = []
        for s in range(n_shop):
            img = render(i * 100 + s) if s == 0 else perturb(render(i * 100), i * 100 + 50 + s)
            rel = f"img/{g}/{c}/{item}/shop_{s + 1:02d}.jpg"
            img.save(os.path.join(a.root, rel), quality=95)
            shop_imgs.append(rel)
            bbox_rows.append(f"{rel} {TYPES[c]} 1 20 30 100 140")
        n_cons = rng.choice([1, 1, 2])
        for k in range(n_cons):
            base = render(i * 100)
            img = base if a.identical else perturb(base, i * 100 + 10 + k)
            rel = f"img/{g}/{c}/{item}/consumer_{k + 1:02d}.jpg"
            img.save(os.path.join(a.root, rel), quality=95)
            bbox_rows.append(f"{rel} {TYPES[c]} 2 15 25 105 145")
            part_rows.append(f"{rel} {shop_imgs[k % len(shop_imgs)]} {item} {part}")

    with open(os.path.join(a.root, "Anno", "list_bbox_consumer2shop.txt"), "w") as f:
        f.write(f"{len(bbox_rows)}\nimage_name clothes_type source_type x_1 y_1 x_2 y_2\n" + "\n".join(bbox_rows) + "\n")
    with open(os.path.join(a.root, "Eval", "list_eval_partition.txt"), "w") as f:
        f.write(f"{len(part_rows)}\nimage_pair_name_1 image_pair_name_2 item_id evaluation_status\n" + "\n".join(part_rows) + "\n")
    print(f"synthetic dataset: {a.items} items, {len(bbox_rows)} images, {len(part_rows)} pairs -> {a.root}")


if __name__ == "__main__":
    main()
