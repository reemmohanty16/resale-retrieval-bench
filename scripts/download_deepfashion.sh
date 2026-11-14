#!/usr/bin/env bash
# Download the DeepFashion Consumer-to-Shop Clothes Retrieval benchmark from the authors' Google Drive folder.
# The dataset is released for non-commercial research use; cite Liu et al., CVPR 2016.
#
# Usage:  bash scripts/download_deepfashion.sh /data <password>  -> /data/Consumer_to_shop/{Anno,Eval,img}
#
# Google Drive sometimes refuses large public downloads with a quota error. If that happens, open the folder
# in a browser, "Add shortcut to Drive", mount Drive in Colab, and copy from the mounted path instead
# (see RUN_GUIDE.md, option B).
set -euo pipefail
DEST="${1:?destination directory}"
PASSWORD="${2:-}"   # the image zip is encrypted; the DeepFashion team sends the password after the signed agreement
FOLDER_URL="https://drive.google.com/drive/folders/0B7EVK8r0v71pQ2FuZ0k0QnhBQnc?resourcekey=0-NWldFxSChFuCpK4nzAIGsg"
mkdir -p "$DEST"; cd "$DEST"
python -m pip install -q gdown
gdown --folder "$FOLDER_URL" --remaining-ok -O ./c2s_download
# Flatten into the expected layout
mkdir -p Consumer_to_shop
find ./c2s_download -maxdepth 3 -type d \( -iname Anno -o -iname Eval \) -exec cp -r {} Consumer_to_shop/ \;
find ./c2s_download -type f -iname "*.zip" | while read -r z; do
  echo "unzipping $z"; unzip -q -o ${PASSWORD:+-P "$PASSWORD"} "$z" -d Consumer_to_shop/
done
ls Consumer_to_shop
echo "If Anno/ or Eval/ are missing, locate them under ./c2s_download and move them into Consumer_to_shop/."
