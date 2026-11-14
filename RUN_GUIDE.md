# Run guide (for Reem)

Everything runs in Google Colab. Your laptop only needs a browser.

## Status as of Nov 14, 2026
The full run is done (all seven models plus the crop ablation, one T4, ~1 h of GPU time). Results are in `results/` in this repo and in `My Drive/rrb/out/`. The steps below are what it takes to reproduce it.

Note: the image zip is password-protected. The password comes from the dataset authors after you send them a signed release agreement; keep it out of the notebook and the repo.

## Step 1 — Get the password (10 minutes of your time, then wait for them)
1. Download the agreement: https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/DeepFashionAgreement.pdf (sample: https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/DeepFashionAgreement_sample.pdf). Fill it in and sign it.
2. Open the draft in Gmail (hello@banabre.com), attach the signed PDF, delete the bracketed note at the bottom, and send. Recipients and the subject line ("DeepFashion Agreement") are already set as the authors require. They ask for an institutional email address; if you still have a university alumni address, send it from there instead.
3. When the password arrives, keep it with your paper files.

## Step 2 — Run the notebook (about an hour, mostly waiting)
1. Open Colab from Drive: `My Drive/rrb/run_colab.ipynb` → right-click → Open with → Google Colaboratory. Runtime → Change runtime type → **GPU**.
2. Run the cells top to bottom. The first code cell asks for permission to access Drive; allow it.
3. In section 2, use the **Option B** cell: paste the password into `PASSWORD = ""`, then run it. It fetches the annotation files, copies the 2.5 GB image zip to the Colab disk, and unzips it. The last line should print a count of about 239,000 images.
4. The "check the parse" cell should print about 33,881 items and tens of thousands of queries and gallery images. If it prints zeros or an error, stop and send me the output.
5. Section 3 embeds one model at a time and skips models already done, so if Colab disconnects, just re-run the cell.
6. Section 4 prints the results table and three figures. Everything is also saved in `My Drive/rrb/out/`.

## Step 3 — Send me `rrb/out/results/` and `rrb/out/report/`
That is all I need to write the paper draft. The embeddings can stay in your Drive.

## Expected timing on a T4
| Model | Approx. images/sec | Time for ~50k images |
|---|---|---|
| clip-vit-b32 | 800–1500 | 1 min |
| siglip-b16, fashionclip, marqo-* | 300–600 | 2–3 min |
| clip-vit-l14 | 80–150 | 6–10 min |
| dinov2-b | 200–400 | 2–4 min |

JPEG decoding is usually the bottleneck, so the whole run is normally under an hour once the data is unzipped.

## If something breaks
- **"unable to get password" / "unzip exit: 5" or 82:** the password is missing or wrong.
- **Anno or Eval shows MISSING:** the Drive mount hid the shared folders and gdown also failed. In Drive, open the authors' Anno and Eval folders, select all files, right-click → Make a copy, then move the copies into `My Drive/rrb/Consumer_to_shop/Anno` and `.../Eval`, and point `SHORTCUT` at `/content/drive/MyDrive/rrb/Consumer_to_shop`.
- **Out of memory:** lower `--batch-size` to 64 in the section 3 cell.
- **A Marqo model fails to load:** it needs `timm`; re-run the install cell, then that model alone.

## What NOT to do
- Don't run this on the Hearst laptop or store the dataset on it. Colab plus Drive keeps everything off the machine.
- Don't share the dataset, the Drive shortcut, or the password with anyone; the agreement forbids redistribution.
