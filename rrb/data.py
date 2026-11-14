"""Parsing and splitting for the DeepFashion Consumer-to-Shop Clothes Retrieval benchmark.

Dataset layout (as distributed by the authors; folder names may differ in case):

    <root>/
      Anno/list_bbox_consumer2shop.txt     # image, clothes_type, source_type, x1 y1 x2 y2
      Anno/list_item_consumer2shop.txt     # item ids (not needed here)
      Eval/list_eval_partition.txt         # consumer_img, shop_img, item_id, train|val|test
      img/<GROUP>/<Category>/id_XXXXXXXX/consumer_01.jpg | shop_01.jpg

Both annotation files start with two header rows (count, column names).

Codes (from the dataset README):
    clothes_type: 1 = upper-body, 2 = lower-body, 3 = full-body
    source_type:  1 = shop image,  2 = consumer image

Evaluation protocol used here (mirrors the benchmark's own definition):
    queries  = consumer images whose pair is in the chosen query partition(s) (default: test)
    gallery  = shop images in the val + test partitions
    a query is a hit at K if any top-K gallery image shares its item_id.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, asdict
from typing import Iterable

CLOTHES_TYPE = {1: "upper", 2: "lower", 3: "full"}
SOURCE_TYPE = {1: "shop", 2: "consumer"}


@dataclass(frozen=True)
class ImageRecord:
    path: str            # relative path as written in the annotation files, e.g. img/CLOTHING/Tees/id_00000001/shop_01.jpg
    item_id: str         # e.g. id_00000001
    source: str          # "consumer" | "shop"
    clothes_type: str    # "upper" | "lower" | "full" | "unknown"
    group: str           # top-level folder under img/, e.g. CLOTHING
    category: str        # second-level folder, e.g. Tees
    partition: str       # train | val | test
    bbox: tuple[int, int, int, int] | None  # x1, y1, x2, y2 or None


def _read_table(path: str) -> list[list[str]]:
    """Read a DeepFashion annotation file, skipping the two header rows."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()
    rows = []
    for line in lines[2:]:
        parts = line.split()
        if parts:
            rows.append(parts)
    return rows


def find_file(root: str, *candidates: str) -> str:
    """Return the first existing path among root/candidate, tolerating case differences in folder names."""
    for cand in candidates:
        p = os.path.join(root, cand)
        if os.path.exists(p):
            return p
    # case-insensitive fallback, one level at a time
    for cand in candidates:
        cur = root
        ok = True
        for part in cand.split("/"):
            if not os.path.isdir(cur):
                ok = False
                break
            matches = [e for e in os.listdir(cur) if e.lower() == part.lower()]
            if not matches:
                ok = False
                break
            cur = os.path.join(cur, matches[0])
        if ok and os.path.exists(cur):
            return cur
    raise FileNotFoundError(f"None of {candidates} found under {root}")


def resolve_image_path(root: str, rel: str) -> str:
    """Map an annotation path like img/CLOTHING/Tees/id_x/shop_01.jpg to a real file under root."""
    p = os.path.join(root, rel)
    if os.path.exists(p):
        return p
    # The distributed zip sometimes extracts to Img/ instead of img/
    first, _, rest = rel.partition("/")
    for alt in ("img", "Img", "IMG"):
        p = os.path.join(root, alt, rest)
        if os.path.exists(p):
            return p
    return os.path.join(root, rel)  # let the caller raise a clear error


def _parse_path(rel: str) -> tuple[str, str, str]:
    """Return (group, category, item_id) from img/<GROUP>/<Category>/<id>/<file>."""
    parts = rel.split("/")
    if len(parts) >= 5:
        return parts[1], parts[2], parts[3]
    if len(parts) == 4:
        return parts[1], "unknown", parts[2]
    return "unknown", "unknown", "unknown"


def load_bboxes(root: str) -> dict[str, tuple[int, int, int, int, int, int]]:
    """Return {path: (clothes_type, source_type, x1, y1, x2, y2)}."""
    path = find_file(root, "Anno/list_bbox_consumer2shop.txt", "anno/list_bbox_consumer2shop.txt")
    out = {}
    for parts in _read_table(path):
        img = parts[0]
        try:
            ct = int(parts[1])
            st = int(parts[2])
            x1, y1, x2, y2 = (int(float(v)) for v in parts[-4:])
        except (ValueError, IndexError):
            continue
        out[img] = (ct, st, x1, y1, x2, y2)
    return out


def load_partition(root: str) -> list[tuple[str, str, str, str]]:
    """Return rows of (consumer_img, shop_img, item_id, partition)."""
    path = find_file(root, "Eval/list_eval_partition.txt", "eval/list_eval_partition.txt")
    rows = []
    for parts in _read_table(path):
        if len(parts) < 4:
            continue
        rows.append((parts[0], parts[1], parts[2], parts[3]))
    return rows


def build_records(root: str) -> list[ImageRecord]:
    """Deduplicated image records with partition, source, type, category and bbox."""
    bboxes = load_bboxes(root)
    seen: dict[str, ImageRecord] = {}
    for consumer, shop, item_id, part in load_partition(root):
        for rel, default_source in ((consumer, "consumer"), (shop, "shop")):
            if rel in seen:
                continue
            group, category, pid = _parse_path(rel)
            if pid == "unknown":
                pid = item_id
            meta = bboxes.get(rel)
            if meta:
                ct, st, x1, y1, x2, y2 = meta
                clothes = CLOTHES_TYPE.get(ct, "unknown")
                source = SOURCE_TYPE.get(st, default_source)
                bbox = (x1, y1, x2, y2)
            else:
                clothes, source, bbox = "unknown", default_source, None
            # Filenames are authoritative when they say so explicitly. The released zip spells
            # consumer images "comsumer_XX.jpg" (sic), so accept both spellings.
            fname = os.path.basename(rel).lower()
            if fname.startswith(("consumer", "comsumer")):
                source = "consumer"
            elif fname.startswith("shop"):
                source = "shop"
            seen[rel] = ImageRecord(rel, pid, source, clothes, group, category, part, bbox)
    return list(seen.values())


def select_sets(records: Iterable[ImageRecord], query_partitions=("test",), gallery_partitions=("val", "test")):
    """Split records into (queries, gallery) following the benchmark protocol."""
    records = list(records)
    queries = [r for r in records if r.source == "consumer" and r.partition in query_partitions]
    gallery = [r for r in records if r.source == "shop" and r.partition in gallery_partitions]
    return queries, gallery


def write_index(records: Iterable[ImageRecord], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row", "path", "item_id", "source", "clothes_type", "group", "category", "partition", "x1", "y1", "x2", "y2"])
        for i, r in enumerate(records):
            bb = r.bbox if r.bbox else ("", "", "", "")
            w.writerow([i, r.path, r.item_id, r.source, r.clothes_type, r.group, r.category, r.partition, *bb])


def read_index(path: str) -> list[ImageRecord]:
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            bbox = None
            if row["x1"] != "":
                bbox = (int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"]))
            out.append(ImageRecord(row["path"], row["item_id"], row["source"], row["clothes_type"],
                                   row["group"], row["category"], row["partition"], bbox))
    return out


def summarize(records: Iterable[ImageRecord]) -> dict:
    records = list(records)
    parts: dict[str, int] = {}
    sources: dict[str, int] = {}
    types: dict[str, int] = {}
    items = set()
    for r in records:
        parts[r.partition] = parts.get(r.partition, 0) + 1
        sources[r.source] = sources.get(r.source, 0) + 1
        types[r.clothes_type] = types.get(r.clothes_type, 0) + 1
        items.add(r.item_id)
    return {"images": len(records), "items": len(items), "partition": parts, "source": sources, "clothes_type": types}
