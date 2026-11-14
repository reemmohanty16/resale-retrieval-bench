"""Figures and a Markdown results summary from out/results/*.json.

Usage:
    python -m rrb.report --out out

Writes out/report/fig1_c2s_recall.png, fig2_gap_dumbbell.png, fig3_type_heatmap.png, results.md
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .evaluate import KS  # noqa: E402
from .models import REGISTRY  # noqa: E402

# Validated categorical palette (light surface): one hue per model family, fixed order.
FAMILY_COLOR = {"general": "#2a78d6", "fashion": "#eb6834", "vision-only": "#1baf7a", "test": "#9a9a94"}
SURFACE = "#fcfcfb"
INK = "#1f1f1d"
INK2 = "#5f5f5a"
GRID = "#e4e4df"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.title.set_color(INK)


def load_results(out: str) -> list[dict]:
    rdir = os.path.join(out, "results")
    res = []
    for fn in sorted(os.listdir(rdir)):
        if fn.endswith(".json"):
            with open(os.path.join(rdir, fn)) as f:
                res.append(json.load(f))
    return res


def _family(r):
    spec = REGISTRY.get(r.get("model") or "", None)
    return spec.family if spec else "general"


def _label(r):
    return f"{r['model']} ({r['crop']})" if r.get("crop") not in (None, "bbox") else r["model"]


def fig_c2s(results, path):
    results = sorted(results, key=lambda r: r["c2s"]["recall@10"])
    labels = [_label(r) for r in results]
    r1 = np.array([r["c2s"]["recall@1"] for r in results]) * 100
    r10 = np.array([r["c2s"]["recall@10"] for r in results]) * 100
    colors = [FAMILY_COLOR.get(_family(r), FAMILY_COLOR["general"]) for r in results]
    y = np.arange(len(results))
    h = 0.34
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(results) + 1.6), facecolor=SURFACE)
    _style(ax)
    ax.barh(y + h / 2 + 0.02, r10, height=h, color=colors, edgecolor=SURFACE, linewidth=1)
    ax.barh(y - h / 2 - 0.02, r1, height=h, color=colors, alpha=0.45, edgecolor=SURFACE, linewidth=1)
    for yi, v1, v10 in zip(y, r1, r10):
        ax.text(v10 + 0.8, yi + h / 2 + 0.02, f"{v10:.1f}", va="center", fontsize=8.5, color=INK)
        ax.text(v1 + 0.8, yi - h / 2 - 0.02, f"{v1:.1f}", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=INK)
    ax.set_xlim(0, max(100, r10.max() * 1.15 if len(r10) else 100))
    ax.set_xlabel("Recall (%) on consumer-to-shop retrieval  —  dark bar: @10, light bar: @1")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for f, c in FAMILY_COLOR.items() if f != "test"]
    ax.legend(handles, ["general VLM", "fashion VLM", "vision-only"], frameon=False, fontsize=8.5, loc="lower right")
    ax.set_title("Zero-shot consumer-to-shop retrieval (DeepFashion C2S test queries)", fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig_gap(results, path, k=10):
    results = [r for r in results if r.get("s2s")]
    if not results:
        return False
    results = sorted(results, key=lambda r: r["c2s_paired"][f"recall@{k}"])
    labels = [_label(r) for r in results]
    c2s = np.array([r["c2s_paired"][f"recall@{k}"] for r in results]) * 100
    s2s = np.array([r["s2s"][f"recall@{k}"] for r in results]) * 100
    y = np.arange(len(results))
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(results) + 1.8), facecolor=SURFACE)
    _style(ax)
    for yi, a, b in zip(y, c2s, s2s):
        ax.plot([a, b], [yi, yi], color=GRID, linewidth=2, zorder=1)
        ax.text((a + b) / 2, yi + 0.22, f"gap {b - a:+.1f} pts", ha="center", fontsize=8, color=INK2)
    ax.scatter(s2s, y, s=60, color="#6b6b66", zorder=3, label="shop → shop (control)")
    ax.scatter(c2s, y, s=60, color=FAMILY_COLOR["general"], zorder=3, label="consumer → shop")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=INK)
    ax.set_xlim(0, 100)
    ax.set_xlabel(f"Recall@{k} (%) on the same items (items with ≥2 shop images)")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    ax.set_title("The photo gap: catalog queries vs. consumer photos, identical items", fontsize=11, loc="left", pad=28)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return True


def fig_types(results, path, k=10):
    types = ["upper", "lower", "full"]
    rows = []
    labels = []
    for r in sorted(results, key=lambda r: -r["c2s"][f"recall@{k}"]):
        b = r.get("c2s_by_clothes_type", {})
        if not any(t in b for t in types):
            continue
        rows.append([b[t][f"recall@{k}"] * 100 if t in b else np.nan for t in types])
        labels.append(_label(r))
    if not rows:
        return False
    M = np.array(rows)
    fig, ax = plt.subplots(figsize=(6, 0.5 * len(rows) + 1.6), facecolor=SURFACE)
    _style(ax)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blue_seq", ["#cde2fb", "#2a78d6", "#0d366b"])
    im = ax.imshow(M, cmap=cmap, aspect="auto", vmin=0, vmax=100)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=8.5,
                        color="white" if M[i, j] > 55 else INK)
    ax.set_xticks(range(len(types)))
    ax.set_xticklabels(["upper-body", "lower-body", "full-body"], color=INK)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, color=INK)
    ax.set_title(f"Consumer-to-shop Recall@{k} (%) by clothes type", fontsize=11, loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cb.ax.tick_params(colors=INK2, labelsize=8)
    cb.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return True


def results_md(results, out_dir) -> str:
    lines = ["# Results", ""]
    if results:
        r0 = results[0]
        lines += [f"Queries: {r0['n_queries']:,} consumer images · Gallery: {r0['n_gallery']:,} shop images "
                  f"({r0['n_gallery_items']:,} items)", ""]
    lines += ["## Consumer-to-shop retrieval (task)", "",
              "| Model | Crop | " + " | ".join(f"R@{k}" for k in KS) + " | MRR |",
              "|---|---|" + "---|" * (len(KS) + 1)]
    for r in sorted(results, key=lambda r: -r["c2s"]["recall@10"]):
        c = r["c2s"]
        lines.append(f"| {r['model']} | {r['crop']} | " + " | ".join(f"{c[f'recall@{k}']*100:.1f}" for k in KS) + f" | {c['mrr']:.3f} |")
    gap = [r for r in results if r.get("s2s")]
    if gap:
        lines += ["", "## Photo gap on identical items (items with ≥2 shop images)", "",
                  "| Model | s2s R@1 | c2s R@1 | gap@1 | s2s R@10 | c2s R@10 | gap@10 |", "|---|---|---|---|---|---|---|"]
        for r in sorted(gap, key=lambda r: -r["gap_paired"]["recall@10"]):
            s, c, g = r["s2s"], r["c2s_paired"], r["gap_paired"]
            lines.append(f"| {_label(r)} | {s['recall@1']*100:.1f} | {c['recall@1']*100:.1f} | {g['recall@1']*100:+.1f} | "
                         f"{s['recall@10']*100:.1f} | {c['recall@10']*100:.1f} | {g['recall@10']*100:+.1f} |")
    lines += ["", "## By clothes type (c2s Recall@10)", "", "| Model | upper | lower | full |", "|---|---|---|---|"]
    for r in sorted(results, key=lambda r: -r["c2s"]["recall@10"]):
        b = r.get("c2s_by_clothes_type", {})
        cells = [f"{b[t]['recall@10']*100:.1f} (n={b[t]['n']})" if t in b else "—" for t in ("upper", "lower", "full")]
        lines.append(f"| {_label(r)} | " + " | ".join(cells) + " |")
    path = os.path.join(out_dir, "results.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="out")
    a = p.parse_args(argv)
    results = load_results(a.out)
    rep = os.path.join(a.out, "report")
    os.makedirs(rep, exist_ok=True)
    fig_c2s(results, os.path.join(rep, "fig1_c2s_recall.png"))
    fig_gap(results, os.path.join(rep, "fig2_gap_dumbbell.png"))
    fig_types(results, os.path.join(rep, "fig3_type_heatmap.png"))
    print(f"[report] wrote figures and {results_md(results, rep)}")


if __name__ == "__main__":
    main()
