"""Embed the query and gallery images of the benchmark with one model.

Usage:
    python -m rrb.embed --root /data/Consumer_to_shop --out out --model clip-vit-b32 [--crop bbox|none]

Writes:
    out/emb/<model>__<crop>/index.csv       one row per image (path, item_id, source, type, category, partition, bbox)
    out/emb/<model>__<crop>/embeddings.npy  float32 [N, D], L2-normalised, aligned with index.csv
    out/emb/<model>__<crop>/meta.json       model spec, device, timing, failures
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image

from . import data as D
from .models import DEFAULT_MODELS, REGISTRY, load_model


class ImageDataset:
    def __init__(self, root: str, records: list[D.ImageRecord], preprocess, crop: str, pad: float = 0.10):
        self.root = root
        self.records = records
        self.preprocess = preprocess
        self.crop = crop
        self.pad = pad
        self.failures: list[str] = []

    def __len__(self):
        return len(self.records)

    def _open(self, rec: D.ImageRecord) -> Image.Image:
        path = D.resolve_image_path(self.root, rec.path)
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:  # corrupt or missing file: substitute a neutral image, log it
            self.failures.append(f"{rec.path}\t{type(e).__name__}: {e}")
            return Image.new("RGB", (224, 224), (128, 128, 128))
        if self.crop == "bbox" and rec.bbox:
            x1, y1, x2, y2 = rec.bbox
            w, h = img.size
            bw, bh = max(x2 - x1, 1), max(y2 - y1, 1)
            px, py = int(bw * self.pad), int(bh * self.pad)
            box = (max(0, x1 - px), max(0, y1 - py), min(w, x2 + px), min(h, y2 + py))
            if box[2] - box[0] >= 8 and box[3] - box[1] >= 8:
                img = img.crop(box)
        return img

    def __getitem__(self, i):
        return self.preprocess(self._open(self.records[i]))


def run(root: str, out: str, model_name: str, crop: str = "bbox", batch_size: int = 128, workers: int = 4,
        device: str = "auto", limit: int = 0, query_partitions=("test",), gallery_partitions=("val", "test"),
        force: bool = False) -> str:
    import torch
    from torch.utils.data import DataLoader

    tag = f"{model_name}__{crop}"
    out_dir = os.path.join(out, "emb", tag)
    emb_path = os.path.join(out_dir, "embeddings.npy")
    if os.path.exists(emb_path) and not force:
        print(f"[embed] {tag}: embeddings exist, skipping (use --force to redo)")
        return out_dir

    t0 = time.time()
    records = D.build_records(root)
    queries, gallery = D.select_sets(records, query_partitions, gallery_partitions)
    needed = {r.path: r for r in queries}
    for r in gallery:
        needed.setdefault(r.path, r)
    recs = sorted(needed.values(), key=lambda r: r.path)
    if limit:
        recs = recs[:limit]
    print(f"[embed] {tag}: {len(recs)} images ({len(queries)} queries, {len(gallery)} gallery) | "
          f"dataset: {json.dumps(D.summarize(records))}")

    emb = load_model(model_name, device)
    print(f"[embed] device={emb.device} dim={emb.dim}")
    ds = ImageDataset(root, recs, emb.preprocess, crop)
    use_workers = 0 if emb.device == "mps" and workers > 0 and sys.platform == "darwin" and os.environ.get("RRB_MPS_WORKERS") is None else workers
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=use_workers, pin_memory=(emb.device == "cuda"))

    chunks = []
    done = 0
    t1 = time.time()
    for batch in dl:
        chunks.append(emb.embed_batch(batch))
        done += batch.shape[0]
        if done % (batch_size * 20) == 0 or done == len(ds):
            rate = done / max(time.time() - t1, 1e-6)
            print(f"[embed] {done}/{len(ds)}  {rate:.1f} img/s  eta {(len(ds) - done) / max(rate, 1e-6) / 60:.1f} min", flush=True)
    X = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, emb.dim or 0), dtype=np.float32)

    os.makedirs(out_dir, exist_ok=True)
    np.save(emb_path, X)
    D.write_index(recs, os.path.join(out_dir, "index.csv"))
    with open(os.path.join(out_dir, "failures.txt"), "w") as f:
        f.write("\n".join(ds.failures))
    meta = {
        "model": model_name, "spec": REGISTRY[model_name].__dict__, "crop": crop, "device": emb.device,
        "n_images": int(X.shape[0]), "dim": int(X.shape[1]) if X.ndim == 2 else None,
        "n_failures": len(ds.failures), "seconds": round(time.time() - t0, 1),
        "query_partitions": list(query_partitions), "gallery_partitions": list(gallery_partitions),
        "torch": torch.__version__,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[embed] wrote {emb_path} {X.shape} in {meta['seconds']}s, failures={len(ds.failures)}")
    return out_dir


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="dataset root containing Anno/, Eval/, img/")
    p.add_argument("--out", default="out")
    p.add_argument("--model", action="append", help="model name; repeat for several. Default: all standard models", default=None)
    p.add_argument("--crop", choices=["bbox", "none"], default="bbox")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    p.add_argument("--limit", type=int, default=0, help="embed only the first N images (smoke tests)")
    p.add_argument("--query-partitions", default="test")
    p.add_argument("--gallery-partitions", default="val,test")
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)
    models = a.model or DEFAULT_MODELS
    for m in models:
        run(a.root, a.out, m, a.crop, a.batch_size, a.workers, a.device, a.limit,
            tuple(a.query_partitions.split(",")), tuple(a.gallery_partitions.split(",")), a.force)


if __name__ == "__main__":
    main()
