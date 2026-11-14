# Paper skeleton

**Working title:** The Photo Gap: How Zero-Shot Vision-Language Embeddings Degrade on Consumer Garment Photos, and What It Means for Secondhand Fashion Search

**Author:** Reem Mohanty (Banabre / Glo AI Inc.)

**Target venues:** arXiv (cs.CV, cs.IR) first; then a fashion/retail workshop at CVPR or RecSys; a practitioner version for Towards Data Science.

**Length:** 4 pages + references (workshop format), ~2,500 words.

---

## Abstract (150 words, written last)
One sentence each: the deployment reality (off-the-shelf embeddings power AI shopping tools) → the open question (how much do they lose on consumer photos?) → what we did (seven zero-shot models, DeepFashion C2S, paired control) → headline numbers (to fill: R@10 drop of X points; fashion-specific training closes Y of the gap / does not) → implication for secondhand, where every listing photo is a consumer photo.

## 1. Introduction
- Secondhand fashion is growing faster than firsthand (cite ThredUp Resale Report, current year; find one more independent source).
- Every secondhand listing is a *consumer photo*: hangers, flat lays, mirrors, phone cameras. Every AI styling or search product sits on top of an image embedding model, usually used zero-shot.
- Consumer-to-shop retrieval has a decade of supervised work (Street2Shop 2015; DeepFashion 2016; later metric-learning papers). The zero-shot question is different: it asks about the models people actually deploy without training data.
- **Contributions:** (1) a zero-shot benchmark of seven current embedding models on DeepFashion C2S; (2) the paired *photo gap* measurement on identical items; (3) breakdowns by garment type and crop; (4) [phase 2, if done] a small secondhand-photo set collected in-store.

## 2. Related work
- Consumer-to-shop / street-to-shop retrieval: Kiapour et al. 2015 (Where to Buy It); Liu et al. 2016 (DeepFashion); Ge et al. 2019 (DeepFashion2); representative metric-learning follow-ups.
- Vision-language embeddings: CLIP (Radford 2021), SigLIP (Zhai 2023), SigLIP 2 (2025).
- Fashion-specific VLMs: FashionCLIP (Chia 2022), Marqo-FashionCLIP/SigLIP (2024).
- Fashion retrieval benchmarks: Fashion200K; LookBench (Gao et al. 2026) — live catalog retrieval, no consumer queries. Position this paper as filling the consumer-query gap for zero-shot models.
- Domain-gap studies in retrieval generally (one or two citations).

## 3. Setup
- **Data:** DeepFashion Consumer-to-Shop: 239,557 images, 195,540 pairs, 33,881 items. Queries = test-partition consumer images; gallery = val+test shop images. Report exact counts from `results/*.json`.
- **Models:** table with family, params, training data, embedding dim (fill from model cards).
- **Protocol:** cosine similarity, Recall@K, MRR; bbox crop with 10 % padding (default) vs. full image; no fine-tuning, no re-ranking, no text.
- **Control and gap:** shop→shop on items with ≥2 shop images, query excluded; paired gap on the same items. Explain why this isolates the photo effect from item difficulty.
- **Compute:** one GPU, hours. Reproducibility: repository link, fixed seeds, exact model tags.

## 4. Results
- **Table 1:** c2s Recall@{1,5,10,20,50} and MRR for all models (from `report/results.md`).
- **Figure 1:** c2s recall bars by model, coloured by family.
- **Table 2 / Figure 2:** paired gap (s2s vs c2s on identical items). This is the headline figure.
- **Figure 3:** heatmap by clothes type.
- **Crop ablation:** bbox vs full image for two models.
- Guidance for writing: report what the numbers say, including if fashion-specific models do *not* beat general ones, or if the gap is smaller than expected. A null result is still a result.

## 5. What this means for secondhand search
- Translate the gap into product terms: with a gallery of N items, Recall@10 of X % means roughly one in Y queries fails to surface the right item in the first screen.
- Which of the three levers matters most given the data: fashion-specific pretraining, cropping/detection, model size.
- Why secondhand is harder than this benchmark: one-of-a-kind inventory (no second chance via another listing of the same SKU), inventory churn, and photos that are worse than DeepFashion's consumer images. Keep this section grounded in the measured numbers; speculative claims go in Limitations.

## 6. Limitations
- DeepFashion consumer photos are street/selfie photos, not resale listing photos (hangers, flat lays). [Phase 2 addresses this.]
- Zero-shot only; fine-tuned systems will do better. The paper is about deployed defaults.
- Dataset from 2016; styles and photo quality have shifted.
- Model training data may include DeepFashion images (contamination). Note which model cards disclose training sources.

## 7. Conclusion
Two sentences.

## Phase 2 (optional, strengthens the paper a lot)
A **secondhand listing-style** test set: 150–300 garments photographed in-store in four styles (hanger, flat lay, mannequin, mirror selfie) with a phone, paired with the brand's catalog image where one exists. Get written permission from the store, no people in frame, no price tags with store info. Run the same models; report the gap against the DeepFashion gap.

## Checklist before arXiv
- [ ] Numbers in text match `results/summary.csv`
- [ ] Every model citation has the exact checkpoint tag
- [ ] Dataset licence terms respected; no images redistributed
- [ ] Code released; README reproduces Table 1
- [ ] Affiliation and author name as they appear on your immigration paperwork
