# Vision-as-Auditor: Fraud Detection via Model-Seller Disagreement

## Motivation

V2's fraud pipeline established that in this dataset, **fraud is a metadata-contradiction problem, not a visual one**. The heuristic rule "condition ≥ 4 AND (major stains OR major holes)" achieved 100% recall at 22.6% precision (F1 0.368), dominating both IsolationForest on learned embeddings (F1 0) and CLIP zero-shot discrepancy (F1 0.005).

But that conclusion had a gap. The heuristic works because fraud in this dataset is defined as a contradiction between the `condition` field and the `stains`/`holes` fields. If a seller falsifies *both* the condition score and the defect labels, the contradiction disappears and the heuristic goes silent. A vision model looking at the actual pixels would still see the damage.

The V2 writeup proposed an experiment for this but never ran it (takeaway #4): use a trained condition model as an **independent visual auditor**, predict condition from images, and flag items where the model disagrees with the seller's claim. The CLIP result (F1 0.005) closed the door on *zero-shot* auditors. This experiment asks whether a fine-tuned auditor does better.

The key design question: which auditor? An all-stations model (V2) is high-coverage but noisy (MAE 0.81). A station-3-only model is cleaner (MAE 0.51) but trained on only one annotation team. I test both, and I add per-station bias calibration for the cross-station case (because the station experiment showed S3→S1 has a systematic ~1-grade offset).

## Method

**Signed discrepancy** as the fraud score: `signed = claimed_condition − predicted_condition`. Positive = the model thinks the item is worse than the seller claims. This is the directional fraud signal; `|pred − claimed|` turns out to be actively misleading (it's symmetric in a problem that isn't).

Predicted condition is the softmax expected value `Σ k·P(k)`, not the argmax — this gives a continuous ranking signal instead of an integer.

**Auditors evaluated:**
1. **V2 (all-stations)** — pretrained multi-task model from iteration 2, no new training. Zero-cost baseline.
2. **S3-raw** — newly trained single-head condition model on station-3 train split only (5,265 items), ordinal KL loss, same dual-stream EfficientNet-B2.
3. **S3-calibrated** — S3-raw predictions with per-station median bias subtracted. This removes the systematic annotator-scale offset between stations without supervision.

**Primary metrics:** ROC-AUC and PR-AUC on the full V2 test set (6,388 items, 23 fraud). At a base rate of 0.36%, threshold-based F1 is dominated by base-rate effects; ranking metrics are more informative.

**Secondary metric:** precision@k (for k = 23, 50, 100) — the top-k-by-suspicion view. This matches how a real review queue would operate.

**Combination:** heuristic as gate, auditor as reranker over the 102 flagged items. This asks: "of the items the heuristic already caught, can the auditor tell us which to review first?"

## Auditor training

Station-3 train split: 5,265 items train / 586 val (90/10 internal split). Architecture: dual-stream EfficientNet-B2 single condition head, ordinal KL soft-label loss (σ=1.0). Two-phase schedule: 2 epochs head-only at lr=1e-3, then backbone unfrozen at lr=1e-4. OneCycleLR, AdamW (wd=1e-4), mixed-precision on MPS, 15 epoch budget, patience 3.

**Training result:** Best val loss 0.1661 at epoch 8 (acc 0.573, MAE 0.471, 1-off 0.961). Early stopped at epoch 11 (patience 3). Training loss continued dropping (0.092 at stop) while val loss diverged — classic overfitting signal, but the checkpoint from epoch 8 is clean.

## Results

### Per-station auditor accuracy (non-fraud items)

| Auditor | Station 1 (n=4,279) |  | Station 2 (n=525) |  | Station 3 (n=1,561) |  |
|---|---|---|---|---|---|---|
|  | Acc | MAE | Acc | MAE | Acc | MAE |
| V2 (all-stations) | 0.320 | 0.935 | 0.474 | 0.703 | 0.502 | 0.577 |
| S3-raw | 0.220 | 1.051 | 0.255 | 0.957 | 0.510 | 0.580 |
| S3-calibrated | 0.294 | 1.022 | 0.442 | 0.823 | 0.533 | 0.537 |

The V2 multi-task model generalizes better across stations (trained on all three), while S3-raw overfits to station-3 annotation conventions. Calibration partially closes the gap for S3 on stations 1 and 2, but V2 still leads on station-1 accuracy.

### Fraud detection — standalone

| Method | ROC-AUC | PR-AUC | P@23 | P@100 | F1 (best) |
|---|---|---|---|---|---|
| Heuristic rules (reference) | — | — | — | — | 0.368 |
| V2 auditor | 0.679 | 0.011 | 0.043 | 0.010 | 0.063 |
| S3 auditor (raw) | 0.730 | 0.009 | 0.043 | 0.010 | 0.047 |
| S3 auditor (calibrated) | 0.744 | 0.009 | 0.043 | 0.010 | 0.044 |

### Fraud detection — auditor as reranker over heuristic flags

The heuristic catches all 23 fraud items but at 22.6% precision — for every true fraud in the 102 flags, there are ~3.4 false positives that still need review. If the auditor can rank the true fraud to the top of that list, a reviewer can spend attention where it matters.

| Auditor | n=102 flags, 23 fraud | ROC-AUC | PR-AUC | top-23 caught | top-50 caught |
|---|---|---|---|---|---|
| V2 | | 0.458 | 0.273 | 5/23 | 12/23 |
| S3-raw | | 0.555 | 0.291 | 8/23 | 12/23 |
| S3-calibrated | | 0.711 | 0.386 | 10/23 | 18/23 |

## Interpretation

**Standalone auditors cannot replace the heuristic.** All three auditors have ROC-AUC well above chance (0.68–0.74), confirming that vision-predicted condition carries a real fraud signal. But at a 0.36% base rate, that signal is far too weak for standalone use — precision@23 is 4.3% for every auditor (1/23 caught), and the best F1 is 0.063 (V2). The heuristic's F1 of 0.368 with perfect recall is untouchable as a first-pass filter.

**The cleaner auditor does beat the noisier one.** S3-raw (ROC-AUC 0.730) outperforms V2 (0.679) despite training on one-third the data. This confirms the station experiment finding: station-3 annotations are cleaner, and a model trained on clean labels produces a better-calibrated condition signal. The V2 model's multi-task heads and multi-station noise dilute its condition accuracy.

**Calibration matters for cross-station deployment.** S3-calibrated (ROC-AUC 0.744) improves over S3-raw (0.730) standalone, and the gap widens dramatically in reranking: within heuristic flags, S3-calibrated achieves ROC-AUC 0.711 and catches 10/23 in the top-23, vs. S3-raw's 0.555 and 8/23. The per-station median bias subtraction removes the ~1-grade systematic offset between station-3 and station-1 annotators, turning what would be false alarms on station-1 items into properly calibrated scores.

**V2 is the worst reranker despite being the best general model.** V2's reranking ROC-AUC (0.458) is barely above chance — worse than random ordering within the flagged set. The multi-task model's condition predictions are too noisy across stations to provide useful prioritization, even though it has the lowest cross-station MAE overall.

**The practical pipeline is: heuristic filter → S3-calibrated reranker.** The heuristic catches all 23 fraud items in 102 flags. The S3-calibrated auditor pushes 10 of those 23 into the top-23 review slots (43.5% precision, vs. the base 22.5%), and 18/23 into the top-50. A human reviewer examining just the top-50 by auditor score would catch 78% of fraud while reviewing fewer than half the flagged items.

## Limitations

1. **Test set has only 23 fraud items.** Every metric is noisy. The 95% CI on recall@k is wide enough that small differences between auditors should not be over-interpreted.
2. **Fraud is defined as a metadata contradiction.** By construction, the heuristic rules are nearly optimal against this specific definition because they directly implement it. The vision auditor is being asked to detect fraud through a fundamentally different channel (pixels), and it inherits any label noise in `condition`.
3. **No "adversarial" fraud.** A real-world test would include fraud items where *both* condition and defect labels are falsified. In that setting the heuristic would go silent and the vision auditor would be the only remaining signal. We don't have that test set, so we can only show what the auditor *can* rank, not what it *alone can detect*.

## Takeaway

Vision-based auditing is a **reranker, not a detector**. The fine-tuned condition model cannot replace simple metadata heuristics for fraud detection — the base rate is too low and the condition prediction too noisy for standalone use. But as a second-stage prioritizer over heuristic flags, the S3-calibrated auditor nearly doubles precision in the top-23 review slots (43.5% vs. 22.5% base) and catches 78% of fraud in the top-50.

This closes the loop on V2's takeaway #4: a task-specific fine-tuned model does better than CLIP zero-shot (ROC-AUC 0.744 vs. the CLIP F1 of 0.005), but the improvement manifests as reranking quality, not as a standalone detector. The remaining gap — the auditor still misses fraud items where both condition and defect labels are falsified in ways that also fool the visual model — would require adversarial test data to evaluate, which this dataset does not contain.
