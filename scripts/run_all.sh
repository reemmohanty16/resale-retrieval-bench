#!/usr/bin/env bash
# Full run: embed every standard model (bbox crop), plus a no-crop ablation for two models, then evaluate and report.
#   bash scripts/run_all.sh /path/to/Consumer_to_shop out
set -euo pipefail
ROOT="${1:?dataset root}"
OUT="${2:-out}"
cd "$(dirname "$0")/.."

MODELS=(clip-vit-b32 clip-vit-l14 siglip-b16 fashionclip marqo-fashionclip marqo-fashionsiglip dinov2-b)
for m in "${MODELS[@]}"; do
  python -m rrb.embed --root "$ROOT" --out "$OUT" --model "$m" --crop bbox --batch-size 128 --workers 4
done
# Ablation: does the bounding-box crop matter? (full image instead)
for m in clip-vit-b32 marqo-fashionsiglip; do
  python -m rrb.embed --root "$ROOT" --out "$OUT" --model "$m" --crop none --batch-size 128 --workers 4
done

python -m rrb.evaluate --out "$OUT"
python -m rrb.report --out "$OUT"
echo "done -> $OUT/report/results.md"
