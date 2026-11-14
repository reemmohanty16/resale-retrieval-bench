# 4. Benchmark and protocol (draft)

## 4.1 Data

We use the Consumer-to-Shop Clothes Retrieval benchmark of DeepFashion (Liu et al., 2016), which pairs photographs taken by shoppers ("consumer" images, typically phone photos of a garment being worn or laid out at home) with the retailer's catalog photographs of the same garment ("shop" images). The release contains 239,557 images of 33,881 distinct items, each with a clothing-type label (upper-body, lower-body, full-body), a category folder, and a garment bounding box. The authors' split assigns 120,927 images to train, 59,884 to validation and 58,746 to test. We do not use the training split at all: every model is evaluated zero-shot, exactly as released by its authors.

Following the benchmark's own definition of the task, the query set is the consumer images whose pair falls in the test split (47,434 queries) and the gallery is the set of shop images in the validation and test splits (22,669 images covering N_ITEMS items). A query is counted as a hit at rank K if any of the top-K gallery images shares its item identifier.

The dataset is released for non-commercial research and was obtained under the authors' release agreement. It was used only for this evaluation.

## 4.2 Models

Seven publicly released image encoders, grouped into three families:

- General-purpose vision-language models: CLIP ViT-B/32 and CLIP ViT-L/14 (Radford et al., 2021; OpenAI weights), SigLIP ViT-B/16 trained on WebLI (Zhai et al., 2023).
- Fashion-specialised vision-language models: FashionCLIP (Chia et al., 2022), Marqo-FashionCLIP and Marqo-FashionSigLIP (Marqo, 2024), all fine-tuned from general CLIP/SigLIP checkpoints on fashion product data.
- A vision-only self-supervised model: DINOv2 ViT-B/14 (Oquab et al., 2023), using the CLS token.

Each model is used with the preprocessing shipped with its weights (open_clip for CLIP, SigLIP and the Marqo models; Hugging Face transformers for FashionCLIP and DINOv2). Image embeddings are L2-normalised and compared by cosine similarity. No text is used anywhere in the protocol; the models' text towers are not loaded.

## 4.3 Cropping

The benchmark provides a garment bounding box for every image. In the main setting we crop each image to its box, expanded by 10% of the box width and height on each side, before preprocessing. This removes most of the background, body and clutter that distinguishes a consumer photo from a catalog photo and is the setting that favours the consumer side most. A second setting feeds the full image. Both settings are run for CLIP ViT-B/32 and Marqo-FashionSigLIP so that the effect of cropping can be separated from the effect of the photo itself.

## 4.4 Metrics

Recall@K for K in {1, 5, 10, 20, 50} and mean reciprocal rank (MRR) of the first correct gallery image. Recall@K is the fraction of queries for which a correct item appears in the top K, which is the quantity a user of a "find this garment" feature experiences: whether the right listing is on the first screen.

## 4.5 The control: shop-to-shop retrieval on the same items

A consumer-to-shop recall number on its own cannot say how much of the loss is due to the photo and how much is simply the difficulty of telling 20,000 similar garments apart. To separate the two we add a control that holds the items fixed and changes only the photograph. For every item with at least two shop images, each shop image is used in turn as a query against the full gallery with itself removed; a hit is another shop image of the same item. This is retrieval between two catalog photos of one garment, the easiest version of the task, and it is scored with the same gallery and the same metrics.

We then restrict the consumer-to-shop queries to that same subset of items (the paired consumer-to-shop score) and report the difference

    gap@K = Recall@K(shop → shop) − Recall@K(consumer → shop, same items).

Because the two numbers are computed on identical items against an identical gallery, the gap isolates what the consumer photograph costs the encoder. We call it the photo gap.

## 4.6 Breakdowns

Consumer-to-shop recall is also reported by clothing type (upper, lower, full-body) and by category folder, for categories with at least 50 test queries.

## 4.7 Implementation

All runs use a single NVIDIA T4 in Google Colab, batch size 256, fp32 inference. Embedding the 70,103 query and gallery images takes between about 4 minutes (CLIP ViT-B/32) and 12 minutes (CLIP ViT-L/14) per model; JPEG decoding, not the GPU, is the bottleneck for the smaller models. Retrieval is exact (full similarity matrix in 2,048-query chunks, top-50 by partial sort), so there is no approximate-search error in any number. The code, including a synthetic-data test that checks the whole pipeline end to end, is released at [repo URL].
