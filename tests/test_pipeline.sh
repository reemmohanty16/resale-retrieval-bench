#!/usr/bin/env bash
# End-to-end smoke test on synthetic data. No downloads: real-model backends run with tiny random weights
# (RRB_OFFLINE_TEST=1), which exercises the exact code paths; their numbers are meaningless by design.
# The `random` embedder on the --identical dataset must reach Recall@1 = 1.0, which checks the evaluation.
set -euo pipefail
cd "$(dirname "$0")/.."
WORK="${1:-/tmp/rrb_test}"
rm -rf "$WORK"; mkdir -p "$WORK"

echo "== 1. identical consumer/shop images: evaluation must return R@1 = 1.0"
python tests/make_synthetic.py --root "$WORK/ident" --items 240 --identical
python -m rrb.embed --root "$WORK/ident" --out "$WORK/out_ident" --model random --crop none --workers 0
python -m rrb.evaluate --out "$WORK/out_ident"
python - <<EOF
import json; r=json.load(open("$WORK/out_ident/results/random__none.json"))
assert r["c2s"]["recall@1"] >= 0.999, r["c2s"]
assert r["n_queries"] > 0 and r["n_gallery"] > 0
print("   OK: identical-image R@1 =", r["c2s"]["recall@1"], "| queries", r["n_queries"], "gallery", r["n_gallery"])
EOF

echo "== 2. perturbed images: pipeline runs for every backend (offline tiny models), both crop modes"
python tests/make_synthetic.py --root "$WORK/pert" --items 240
export RRB_OFFLINE_TEST=1
python -m rrb.embed --root "$WORK/pert" --out "$WORK/out_pert" --model random --model clip-vit-b32 --model fashionclip --model dinov2-b --crop bbox --workers 0 --batch-size 16
python -m rrb.embed --root "$WORK/pert" --out "$WORK/out_pert" --model random --crop none --workers 0
python -m rrb.evaluate --out "$WORK/out_pert"
python -m rrb.report --out "$WORK/out_pert"
python - <<EOF
import json, os, csv
out="$WORK/out_pert"
tags=sorted(os.listdir(os.path.join(out,"emb")))
assert set(tags) >= {"random__bbox","random__none","clip-vit-b32__bbox","fashionclip__bbox","dinov2-b__bbox"}, tags
for t in tags:
    r=json.load(open(os.path.join(out,"results",t+".json")))
    assert 0.0 <= r["c2s"]["recall@1"] <= 1.0 and r["s2s"] is not None, t
rows=list(csv.DictReader(open(os.path.join(out,"results","summary.csv"))))
assert len(rows)==len(tags), (len(rows), len(tags))
for f in ["fig1_c2s_recall.png","fig2_gap_dumbbell.png","fig3_type_heatmap.png","results.md"]:
    assert os.path.getsize(os.path.join(out,"report",f))>0, f
rnd=json.load(open(os.path.join(out,"results","random__none.json")))
print("   OK: tags", tags)
print("   random-projection on perturbed images: c2s R@1=%.2f R@10=%.2f | s2s R@1=%.2f" % (rnd["c2s"]["recall@1"], rnd["c2s"]["recall@10"], rnd["s2s"]["recall@1"]))
EOF
echo "== ALL TESTS PASSED  ($WORK)"
