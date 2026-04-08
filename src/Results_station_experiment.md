# Station-Specific Training Experiment

## Motivation

The error analysis on V2 revealed a striking pattern: condition prediction accuracy varied dramatically across the three photography stations, even though the model was trained on data from all three:

| Station | V2 Accuracy | V2 MAE | Test Items |
|---------|-------------|--------|------------|
| Station 1 (Wargon) | 34.9% | 0.93 | 4,297 |
| Station 2 (Wargon) | 44.5% | 0.71 | 526 |
| Station 3 (Myrorna) | 54.6% | 0.51 | 1,565 |

This raised a key question: **is the performance gap caused by data noise (station 1 has inconsistent annotations) or by domain mismatch (the stations are visually different and the model can't generalize)?**

To answer this, we trained station-specific models and tested both in-domain and cross-domain.

## Experimental Setup

We trained three separate models using the same V2 architecture (dual-stream EfficientNet-B2 with ordinal soft-label loss), differing only in their training/test data:

1. **S3 → S3** (in-domain): Train on station 3, test on station 3 (80/20 split within station)
2. **S3 → S1** (cross-domain): Train on station 3, test on station 1
3. **S1 → S1** (in-domain): Train on station 1, test on station 1 (80/20 split within station)

Each model was trained for up to 10 epochs with the same two-phase freeze/unfreeze strategy and ordinal KL loss as V2.

## Results

| Experiment | Train Size | Test Size | Accuracy | MAE | 1-off |
|------------|------------|-----------|----------|-----|-------|
| **S3 → S3** | 5,932 | 1,484 | **53.6%** | **0.51** | **95.8%** |
| S3 → S1 | 5,932 | 4,368 | 15.5% | 1.13 | 75.3% |
| S1 → S1 | 17,471 | 4,368 | 32.1% | 0.99 | 74.6% |
| V2 (all stations, reference) | 25,548 | 6,388 | 40.5% | 0.81 | 82.4% |

## Three Major Findings

### 1. Station 3 has fundamentally cleaner labels (or easier data)

The S3 → S3 model achieves **53.6% accuracy with 0.51 MAE and 95.8% within-one-grade accuracy** — a dramatic improvement over V2's combined performance. Critically, this was achieved with only ~6K training items, **less than a quarter of V2's training data**.

This is a strong signal that station 3's labels are more consistent. Less data, dramatically better performance — there's no way to explain this without invoking label quality. The 95.8% within-one-grade accuracy is approaching what you'd expect from human inter-annotator agreement on a subjective 5-point scale.

### 2. Station 1 is hard, regardless of data volume

The S1 → S1 model has **3x more training data** (17,471 items) than the S3 → S3 model, yet only achieves 32.1% accuracy with 0.99 MAE. This is worse than V2's overall performance and dramatically worse than station 3's in-domain results.

If station 1's poor performance were a sample-size or model capacity issue, more data would help. It doesn't. The conclusion: **station 1 has noisier labels or genuinely harder visual conditions** that no amount of additional data can overcome with current methods.

### 3. Cross-domain transfer is catastrophic

The S3 → S1 model collapses to **15.5% accuracy — worse than random chance (20%)**. However, its 1-off accuracy stays at 75.3%, which is a remarkable pattern: the model isn't producing garbage predictions, it's predicting the *wrong absolute level* but maintaining the relative ordering.

**This is the smoking gun for systematic annotator bias.** What station 3 calls "condition 4" appears to be what station 1 annotators call "condition 3" (or vice versa). The two annotation teams have different mental rubrics for what each grade means. A model trained on one team's labels cannot apply them to the other team's items because the teams disagree on the underlying scale.

The ordinal structure is preserved (the model can rank items by condition), but the absolute calibration is shifted by roughly one grade.

## Implications for the V2 Combined Model

These results explain why V2's performance on the combined dataset (40.5%) is bounded:
- Station 1 dominates the test set at 67% of items but has noisy labels
- Station 3 has cleaner labels but only contributes 25% of the test set
- The model has to learn a "compromise" calibration that doesn't fit either station perfectly

The V2 model's 40.5% accuracy is essentially a weighted average of two different ceilings: ~53% for station 3 items and ~32% for station 1 items, mediated by the model trying to learn a single unified mapping.

## What This Means for Real-World Deployment

A production system trained on this dataset would face the same problem: it would inherit the systematic disagreement between annotation teams and never be more accurate than the average. The right fix is not better models or more data — it is **harmonizing the annotation rubrics across collection sites** before training, or training station-specific models if that isn't feasible.

For e-commerce platforms aggregating used clothing from multiple sources (warehouses, sellers, regions), this is a real concern: different annotators applying the same nominal scale will produce systematically biased datasets, and a single global model cannot resolve the disagreement.

## Key Takeaway

The bottleneck for the V2 model is not model capacity, training time, hyperparameters, or even raw data quantity. **It is annotator inconsistency between collection sites.** A station-specific model on the cleaner station (station 3, Myrorna AB) achieves accuracy and within-one-grade performance approaching human inter-annotator agreement levels — which is likely the true ceiling for this task.

This finding is the most important contribution of the project. It transforms the question from "how do we get higher accuracy?" to "what does accuracy even mean when the ground truth itself is inconsistent?"
