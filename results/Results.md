# Clothing Condition Classifier — Results & Technical Summary

## Problem

Predict the **condition grade** (1–5, poor to excellent) of used clothing items from front and back photos. Secondary task: detect **fraud candidates**. This is a supervised, multi-task classification problem.

## Dataset

- ~32K labeled clothing items from 3 inspection stations
- 25,548 train / 6,388 test (stratified split)
- Each item has: front photo, back photo, condition (1–5), stains, holes, fraud flag
- Fraud is extremely rare (~95 items / 0.3%) — severe class imbalance
- Images vary in resolution across stations but are resized to a fixed input size

## Architecture — Dual-Stream EfficientNet-B2

- **Backbone:** EfficientNet-B2 pretrained on ImageNet (~7.7M params), used as a shared feature extractor
- **Dual-stream:** The same backbone processes front and back images independently, producing a 1408-d feature vector each
- **Concatenation:** The two feature vectors are concatenated (2816-d) so the model sees both sides
- **Classification head:** 2816 -> 256 -> 5 (with dropout 0.4 and 0.3)
- **Fraud head (autoresearch version):** 2816 -> 128 -> 1 (binary, BCE loss)
- **Total params:** ~8.4M

## Training Strategy

### Two-Phase Transfer Learning
1. **Phase 1 (epochs 1–2):** Backbone frozen, only the classification head trains. Lets the head learn to map ImageNet features to condition grades without destroying pretrained knowledge.
2. **Phase 2 (epochs 2+):** Backbone unfrozen with 10x lower learning rate (1e-4 vs 1e-3). The backbone slowly adapts to clothing-specific patterns while the head keeps learning faster.

### Regularization
- **Mixup (alpha=0.2):** Blends pairs of images and their labels to learn smoother decision boundaries
- **Label smoothing (0.1):** Softens targets (90% true class, 2.5% each other) to prevent overconfidence
- **Data augmentation:** Random horizontal flip, affine transforms (rotation, scale, shear), color jitter, random erasing
- **Dropout:** 0.4 before the first linear layer, 0.3 before the output layer
- **Weight decay:** 1e-4 (AdamW)

### Learning Rate Schedule
- **OneCycleLR:** Warmup (10% of training), then cosine decay. When backbone unfreezes, a new schedule starts with separate max LRs per param group.
- **Early stopping:** Patience of 3 epochs on val loss

### Mixed Precision
- FP16 autocast on MPS for faster training

## Results (Notebook — 15 epoch run)

| Metric | Value |
|--------|-------|
| Best val loss | 1.3417 (epoch 7) |
| Best val accuracy | ~45% (5-class) |
| Early stopped at | Epoch 10 |
| Random baseline | 20% |

### Observations
- **The model is learning:** 45% vs 20% random baseline on a subjective 5-class task
- **Overfitting emerges around epoch 8–10:** Train loss keeps dropping while val loss plateaus/rises
- The model likely favors the most common condition classes
- Condition grading is inherently subjective (inter-annotator agreement across stations is imperfect)

## Autoresearch Experiments

An automated system ran multiple 5-minute training experiments varying hyperparameters:
- Added a fraud detection head (multi-task learning: 0.8 * condition_loss + 0.2 * fraud_loss)
- Tried aggressive LR (head=5e-3, backbone=1e-3) with gradient clipping and constant LR schedule
- Tried unfreezing backbone immediately (no frozen phase)
- Smaller image size (224 vs 260) for faster iteration

## What Could Improve

- Larger/better backbone (EfficientNet-B3, ConvNeXt, ViT)
- More training data or better augmentation
- Ordinal regression instead of flat classification (condition is ordered)
- Address class imbalance (weighted loss, oversampling)
- Longer training with better LR schedules
- Ensemble front+back at a later fusion point
