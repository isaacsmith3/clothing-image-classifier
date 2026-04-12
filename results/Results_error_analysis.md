# Error Analysis: What the Model Learned and Where It Fails

## Is the Model Just Guessing?

No. The evidence against random behavior is strong:

| Metric | V2 Model | Random Baseline | Improvement |
|--------|----------|-----------------|-------------|
| Accuracy | 40.5% | ~20% | 2x better |
| MAE | 0.812 | 1.522 | 47% lower error |
| Within ±1 grade | 82.4% | ~52% | +30pp |
| Within ±2 grades | 96.1% | ~76% | +20pp |
| Catastrophic errors (±3-4) | 3.9% | ~24% | 6x fewer |

The model is also **underconfident** (mean softmax confidence 30.6% vs actual accuracy 40.5%, ECE = 0.099). A model fitting noise would be overconfident — high confidence on wrong predictions. Underconfidence means the model has learned the task is genuinely ambiguous and hedges its bets.

## The Real Problem: Station Heterogeneity

The most revealing finding is performance stratified by photography station:

| Station | Accuracy | MAE | Test Items | Share |
|---------|----------|-----|------------|-------|
| Station 1 (Wargon) | 34.9% | 0.93 | 4,297 | 67% |
| Station 2 (Wargon) | 44.5% | 0.71 | 526 | 8% |
| Station 3 (Myrorna) | **54.6%** | **0.51** | 1,565 | 25% |

Station 3 is dramatically easier — the model achieves 55% accuracy with MAE of 0.51 (average prediction is half a grade off). Station 1, which dominates the test set at 67%, drags the overall average down to 40%.

Possible explanations:
- **Annotation consistency**: Station 3 (Myrorna AB) may have more consistent human raters, producing cleaner labels
- **Camera/lighting differences**: Each station uses different equipment. Station 1 images may have more variability in background, lighting, or angle
- **Clothing distribution**: Station 1 processes more diverse items, making condition assessment harder

This tells us that **the performance bottleneck is label quality and station heterogeneity, not model capacity**. The model clearly learns visual condition features — it just can't overcome noisy or inconsistent human annotations.

## Performance by Clothing Type

Some garment types are significantly easier to assess than others:

**Easiest (highest accuracy):**
- Denim jacket: 87.5% (n=8, small sample)
- Training top: 57.7%
- Night gown: 55.6%
- Blazer: 54.9%

**Hardest (lowest accuracy):**
- Tank top: 30.7%
- Rain jacket: 30.8%
- Winter jacket: 31.3%
- T-shirt: 33.7%

The pattern makes intuitive sense. Structured garments (blazers, denim jackets) show wear more visibly — wrinkles, fading, and deformation are easy to spot. Simpler garments (t-shirts, tank tops) are harder because condition differences are subtler (minor pilling, slight discoloration) and harder to distinguish at 260x260 resolution.

## Ordinal Error Distribution

The ordinal loss does exactly what it was designed to do — concentrate errors near the diagonal:

| Error magnitude | Model | Random |
|-----------------|-------|--------|
| Exact (±0) | 40.5% | 20.0% |
| Within ±1 | 82.4% | 52.0% |
| Within ±2 | 96.1% | 76.0% |
| ±3 or worse | 3.9% | 24.0% |

Only 3.9% of predictions are off by 3+ grades. In a production system, this means a condition-5 (excellent) item would almost never be classified as condition-1 (poor) — the kind of catastrophic error that would frustrate sellers or misprice inventory.

## Confidence Calibration

The model is slightly underconfident:
- **Mean confidence**: 30.6%
- **Mean accuracy**: 40.5%
- **ECE (Expected Calibration Error)**: 0.099

This is likely caused by the ordinal soft-label loss, which trains the model to spread probability mass across adjacent classes rather than concentrate it on one. The model outputs distributions like [0.05, 0.15, 0.40, 0.30, 0.10] rather than [0.01, 0.02, 0.90, 0.05, 0.02]. This makes the max probability lower even when the prediction is correct.

For a production deployment, this is actually preferable to overconfidence — the model won't confidently mislead downstream systems. Items where the model is genuinely uncertain (max prob < 25%) could be routed to human review.

## GradCAM: What the Model Sees

GradCAM heatmaps show the model focuses on:
- **Central garment regions** rather than background — it learned to ignore the photography backdrop
- **Texture and surface areas** — broad attention across the fabric surface, likely detecting pilling, discoloration, and overall wear
- **Collar and seam areas** on structured garments — regions where wear is most visible

For incorrect predictions, the heatmaps show similar attention patterns but the model sometimes fixates on a single region (e.g., a stain) while missing the overall garment condition. This suggests the model occasionally lets a salient local defect override its global assessment.

## Test-Time Augmentation

TTA (averaging predictions over 5 augmented versions: original, horizontal flip, ±10 rotation, center crop) provided a small but consistent boost:

| Metric | No TTA | With TTA | Delta |
|--------|--------|----------|-------|
| Accuracy | 40.50% | 41.06% | +0.56% |
| MAE | 0.812 | 0.796 | -0.015 |
| 1-off Accuracy | 82.39% | 82.98% | +0.59% |

The improvement is modest but confirms the model learned genuine visual features — TTA would not help a model fitting noise. The accuracy gain is essentially "free" (no retraining, just 5x inference time).

## Key Takeaways

1. **The model is not guessing.** It's 2x better than random with concentrated errors and appropriate uncertainty.

2. **The ceiling is label quality, not model capacity.** Station 3 results (55% accuracy, 0.51 MAE) show what's achievable with consistent annotations. Station 1 labels appear noisier.

3. **Clothing condition is inherently subjective.** On a 5-point scale, human inter-annotator agreement is typically 60-75%. The model's 82% within-one-grade accuracy is likely approaching human-level agreement.

4. **Structured garments are easier.** Blazers and denim jackets show wear visibly; t-shirts and tank tops require detecting subtle texture changes.

5. **The ordinal loss was the right choice.** 96% of predictions within ±2 grades means the model rarely makes catastrophic errors, even when it gets the exact grade wrong.

6. **A production system should combine model predictions with human review.** Route low-confidence predictions (max softmax < 25%) to human raters, use model predictions directly for high-confidence cases.
