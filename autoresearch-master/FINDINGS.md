# Autoresearch Findings Report

## Summary

Adapted Karpathy's autoresearch (autonomous LLM training loop) for clothing condition classification (5-class) + fraud detection (binary) on Apple M3 Max with MPS. Ran ~14 experiments over 1 session. Best combined_score improved from **0.2202** (baseline) to **0.2318** (+5.3%).

**Current best commit:** `cc89ae6` (exp1)

## What Was Built

### Files Created/Modified in `autoresearch-master/`

| File | Action | Purpose |
|------|--------|---------|
| `prepare.py` | Complete rewrite | Immutable data harness: ClothingDataset, transforms, `evaluate()` with combined_score metric |
| `train.py` | Complete rewrite | Agent-editable: dual-stream EfficientNet-B2, multi-task heads, MPS training loop |
| `program.md` | Complete rewrite | Agent instructions for clothing domain, higher-is-better metric, MPS constraints |
| `requirements.txt` | New | Vision deps (torch, timm, torchvision, etc.) replacing NLP deps |
| `data/cleaned_metadata.csv` | Copied | Self-contained data reference |
| `.gitignore` | Updated | Added run.log, *.pt, *.pth |

### Key Design Decisions

- **Metric**: `combined_score = 0.6 * condition_acc + 0.4 * fraud_f1` (higher is better)
- **Model contract**: `forward(front, back) -> {"condition": (B,5), "fraud": (B,1)}`
- **evaluate()** creates its own test dataloader internally (fixed eval conditions)
- **pip** instead of uv for package management
- **python3** command (macOS has `python3` not `python`)
- **Line buffering** (`sys.stdout.reconfigure(line_buffering=True)`) so logs appear in real-time when redirected to file

## Experiment Results

| # | Score | Status | What Changed | Key Finding |
|---|-------|--------|-------------|-------------|
| baseline | 0.2202 | keep | Heads-only, frozen backbone, img260, batch32, mixup | Starting point |
| **exp1** | **0.2318** | **keep** | Unfreeze backbone, img224, batch48, higher LR | **Best result** |
| exp2 | 0.2016 | discard | MobileNetV3 backbone, fraud pos_weight=50 | Weaker features |
| exp3 | 0.1953 | discard | Class-weighted CE + fraud pos_weight=30 | Weights hurt condition acc |
| exp4 | 0.2291 | discard | No mixup, wider head 512, COND_W=0.8 | Marginal regression |
| exp5 | 0.1939 | discard | Image 160px, batch64 | Too low resolution |
| exp5b | 0.1974 | discard | img224, batch64, no mixup | Mixup matters |
| exp6 | 0.2209 | discard | Equal task weights, fraud pos_weight=10 | Didn't help fraud |
| (other) | 0.1974 | discard | Feature extraction + heads (582 epochs) | Massive overfit |

## Lessons Learned

### What Works
1. **EfficientNet-B2 is the right backbone** for this task. MobileNetV3 (lighter) and EfficientNet-B0 both performed worse.
2. **Immediate full fine-tuning** (UNFREEZE_AT_EPOCH=1) beats heads-only training with a 5-min budget.
3. **Mixup helps** even with less than 1 epoch of training. Disabling it consistently hurt performance.
4. **224px images** are the sweet spot. 260px is too slow (fewer steps), 160px loses too much detail.
5. **Batch size 48** works well. 64 doesn't improve things and 96 doesn't either.

### What Doesn't Work
1. **Fraud detection remains at F1=0.0** in almost every experiment. The class is too rare (95/31,936 = 0.3%) and 5 minutes of training isn't enough to learn the pattern.
2. **Class-weighted losses** hurt condition accuracy without helping fraud. The model has too few training steps for weighted losses to converge.
3. **Fraud pos_weight** (10, 30, or 50) doesn't improve fraud F1 either. The model needs more training iterations to actually learn from the rare positives.
4. **Feature extraction + cached heads** (from concurrent session) got 99.8% train accuracy but only 0.1974 combined score — massive overfitting on the extracted features.
5. **num_workers>0** crashes on macOS because the train script runs at module level (no `if __name__ == '__main__':` guard). Would need to restructure the script.

### Infrastructure Issues Encountered
1. **MPS timing**: Wall-clock timing with `time.time()` sometimes drifts due to MPS async operations. Fixed with deadline-based approach (`deadline = time.time() + TIME_BUDGET`) and SIGALRM hard timeout.
2. **Output buffering**: Python buffers stdout when redirecting to file. Fixed with `sys.stdout.reconfigure(line_buffering=True)`.
3. **OneCycleLR breaks** when adding param groups mid-training (missing `initial_lr` key). Fixed by switching to CosineAnnealingLR.
4. **timm `num_features` inaccuracy**: For some models (MobileNetV3), `model.num_features` doesn't match the actual output dimension. Fixed with dummy forward pass detection.
5. **Multiple runs in log**: When `>` (overwrite) is used in the shell command but the agent runs from a different working dir, stale log entries mix with new ones.

## The Fundamental Bottleneck

With 25,548 training images, batch_size=48, and `num_workers=0` on MPS, each epoch takes ~10-15 minutes. The 5-minute training budget only allows **~150-260 steps** (less than 1 full epoch). This means:

- The model sees only ~35-50% of the training data per run
- No multi-epoch convergence
- Loss is still decreasing when training stops
- Fraud positives (72 in training set) may not even appear in many batches

### Potential Solutions for Future Experiments
1. **Pre-extract backbone features** (but avoid overfitting — needs regularization)
2. **Increase TIME_BUDGET** to 600s or 900s to allow 1-2 full epochs
3. **Aggressive data subsampling**: Train on a balanced subset (downsample majority classes)
4. **Pre-resize images** to 224px once and cache them on disk (eliminates resize overhead each batch)
5. **Use `fork` multiprocessing start method** to enable num_workers>0 without restructuring the script
6. **For fraud**: Use a separate binary classifier trained only on fraud features, or explore anomaly detection approaches since the class is too rare for standard classification

## Current State

- **Branch**: `autoresearch` at commit `cc89ae6`
- **Best score**: 0.2318 (condition_acc=0.3863, fraud_f1=0.0000)
- **All dependencies installed** via pip
- **results.tsv** has full experiment log
- **program.md** is ready for autonomous agent use

To resume the loop: `cd autoresearch-master && python3 train.py > run.log 2>&1`
