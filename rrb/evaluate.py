"""Retrieval evaluation: consumer-to-shop (the task) and shop-to-shop (the control), with breakdowns.

Usage:
    python -m rrb.evaluate --out out                      # evaluates every out/emb/* folder
    python -m rrb.evaluate --out out --tag clip-vit-b32__bbox

Metrics
    recall@K   fraction of queries with at least one gallery image of the same item in the top K
    mrr        mean reciprocal rank of the first correct gallery image
    c2s        consumer query -> shop gallery (the benchmark task)
    s2s        shop query -> shop gallery excluding the query image itself, restricted to items
               with >= 2 shop images (a catalog-to-catalog control on the same items)
    c2s_paired consumer-to-shop restricted to the same items used for s2s, so the gap
               s2s - c2s_paired is measured on identical items
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np

from . import data as D

KS = (1, 5, 10, 20, 50)


def _topk_hits(Q: np.ndarray, G: np.ndarray, q_items: np.ndarray, g_items: np.ndarray, ks=KS,
               exclude_rows: np.ndarray | None = None, chunk: int = 2048):
    """Return (first_correct_rank[N] (1-based, 0 = not in top max(ks)), hits dict k -> bool array)."""
    kmax = max(ks)
    n = Q.shape[0]
    first_rank = np.zeros(n, dtype=np.int32)
    kmax_eff = min(kmax, G.shape[0] - (1 if exclude_rows is not None else 0))
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        sims = Q[s:e] @ G.T                                  # cosine (inputs are L2-normalised)
        if exclude_rows is not None:
            sims[np.arange(e - s), exclude_rows[s:e]] = -np.inf
        part = np.argpartition(-sims, kmax_eff - 1, axis=1)[:, :kmax_eff]
        part_sims = np.take_along_axis(sims, part, axis=1)
        order = np.argsort(-part_sims, axis=1)
        top = np.take_along_axis(part, order, axis=1)          # [b, kmax] gallery indices, best first
        correct = g_items[top] == q_items[s:e, None]
        any_hit = correct.any(axis=1)
        ranks = np.where(any_hit, correct.argmax(axis=1) + 1, 0)
        first_rank[s:e] = ranks
    hits = {k: (first_rank > 0) & (first_rank <= k) for k in ks}
    return first_rank, hits


def _metrics(first_rank: np.ndarray, hits: dict) -> dict:
    n = len(first_rank)
    out = {f"recall@{k}": float(h.mean()) if n else float("nan") for k, h in hits.items()}
    rr = np.where(first_rank > 0, 1.0 / np.maximum(first_rank, 1), 0.0)
    out["mrr"] = float(rr.mean()) if n else float("nan")
    out["n"] = int(n)
    return out


def _breakdown(first_rank, hits, keys: list[str], min_n: int = 20) -> dict:
    keys = np.asarray(keys)
    out = {}
    for key in sorted(set(keys.tolist())):
        m = keys == key
        if m.sum() < min_n:
            continue
        out[key] = _metrics(first_rank[m], {k: h[m] for k, h in hits.items()})
    return out


def evaluate_tag(out: str, tag: str, query_partitions=("test",), gallery_partitions=("val", "test")) -> dict:
    emb_dir = os.path.join(out, "emb", tag)
    X = np.load(os.path.join(emb_dir, "embeddings.npy"))
    recs = D.read_index(os.path.join(emb_dir, "index.csv"))
    assert len(recs) == X.shape[0], f"index/embedding mismatch in {tag}: {len(recs)} vs {X.shape[0]}"
    with open(os.path.join(emb_dir, "meta.json")) as f:
        meta = json.load(f)

    q_idx = np.array([i for i, r in enumerate(recs) if r.source == "consumer" and r.partition in query_partitions])
    g_idx = np.array([i for i, r in enumerate(recs) if r.source == "shop" and r.partition in gallery_partitions])
    if len(q_idx) == 0 or len(g_idx) == 0:
        raise RuntimeError(f"{tag}: empty query ({len(q_idx)}) or gallery ({len(g_idx)}) set")

    q_items = np.array([recs[i].item_id for i in q_idx])
    g_items = np.array([recs[i].item_id for i in g_idx])
    Q, G = X[q_idx], X[g_idx]

    # ---- consumer -> shop (task)
    fr, hits = _topk_hits(Q, G, q_items, g_items)
    res = {"tag": tag, "model": meta.get("model"), "crop": meta.get("crop"), "device": meta.get("device"),
           "n_queries": int(len(q_idx)), "n_gallery": int(len(g_idx)),
           "n_gallery_items": int(len(set(g_items.tolist()))),
           "c2s": _metrics(fr, hits)}
    res["c2s_by_clothes_type"] = _breakdown(fr, hits, [recs[i].clothes_type for i in q_idx])
    res["c2s_by_category"] = _breakdown(fr, hits, [recs[i].category for i in q_idx], min_n=50)
    res["c2s_by_group"] = _breakdown(fr, hits, [recs[i].group for i in q_idx], min_n=50)

    # ---- shop -> shop control on items with >= 2 shop images
    counts = defaultdict(int)
    for it in g_items:
        counts[it] += 1
    multi = np.array([counts[it] >= 2 for it in g_items])
    if multi.sum() >= 20:
        sq_rows = np.where(multi)[0]
        fr_s, hits_s = _topk_hits(G[sq_rows], G, g_items[sq_rows], g_items, exclude_rows=sq_rows)
        res["s2s"] = _metrics(fr_s, hits_s)
        res["s2s"]["n_items"] = int(len(set(g_items[sq_rows].tolist())))
        multi_items = set(g_items[sq_rows].tolist())
        pm = np.array([it in multi_items for it in q_items])
        res["c2s_paired"] = _metrics(fr[pm], {k: h[pm] for k, h in hits.items()})
        res["gap_paired"] = {f"recall@{k}": res["s2s"][f"recall@{k}"] - res["c2s_paired"][f"recall@{k}"] for k in KS}
        res["gap_paired"]["mrr"] = res["s2s"]["mrr"] - res["c2s_paired"]["mrr"]
    else:
        res["s2s"] = None
        res["c2s_paired"] = None
        res["gap_paired"] = None

    os.makedirs(os.path.join(out, "results"), exist_ok=True)
    with open(os.path.join(out, "results", f"{tag}.json"), "w") as f:
        json.dump(res, f, indent=2)
    np.save(os.path.join(out, "results", f"{tag}__c2s_first_rank.npy"), fr)
    return res


def write_summary(out: str) -> str:
    rows = []
    rdir = os.path.join(out, "results")
    for fn in sorted(os.listdir(rdir)):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(rdir, fn)) as f:
            r = json.load(f)
        row = {"tag": r["tag"], "model": r["model"], "crop": r["crop"], "n_queries": r["n_queries"], "n_gallery": r["n_gallery"]}
        for k in KS:
            row[f"c2s_r@{k}"] = r["c2s"][f"recall@{k}"]
        row["c2s_mrr"] = r["c2s"]["mrr"]
        if r.get("s2s"):
            for k in KS:
                row[f"s2s_r@{k}"] = r["s2s"][f"recall@{k}"]
                row[f"c2s_paired_r@{k}"] = r["c2s_paired"][f"recall@{k}"]
                row[f"gap_r@{k}"] = r["gap_paired"][f"recall@{k}"]
        for t in ("upper", "lower", "full"):
            b = r["c2s_by_clothes_type"].get(t)
            row[f"c2s_r@1_{t}"] = b["recall@1"] if b else ""
            row[f"c2s_r@10_{t}"] = b["recall@10"] if b else ""
        rows.append(row)
    path = os.path.join(rdir, "summary.csv")
    if rows:
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("tag", "model", "crop"), k))
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="out")
    p.add_argument("--tag", action="append", default=None, help="emb folder name(s); default: all")
    p.add_argument("--query-partitions", default="test")
    p.add_argument("--gallery-partitions", default="val,test")
    a = p.parse_args(argv)
    emb_root = os.path.join(a.out, "emb")
    tags = a.tag or sorted(d for d in os.listdir(emb_root) if os.path.exists(os.path.join(emb_root, d, "embeddings.npy")))
    for tag in tags:
        r = evaluate_tag(a.out, tag, tuple(a.query_partitions.split(",")), tuple(a.gallery_partitions.split(",")))
        c = r["c2s"]
        line = f"[eval] {tag:34s} c2s R@1={c['recall@1']:.3f} R@10={c['recall@10']:.3f} MRR={c['mrr']:.3f}"
        if r.get("s2s"):
            line += f" | s2s R@1={r['s2s']['recall@1']:.3f} gap@1={r['gap_paired']['recall@1']:+.3f}"
        print(line)
    print(f"[eval] summary -> {write_summary(a.out)}")


if __name__ == "__main__":
    main()
