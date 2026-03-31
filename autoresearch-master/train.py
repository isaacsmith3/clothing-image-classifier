"""
Autoresearch training script for clothing condition + fraud detection.
Single-file, single-device (MPS). This is the file the agent modifies.

Usage: python3 train.py
"""

import sys
# Ensure unbuffered output so logs appear in real-time when redirected
if not sys.stdout.line_buffering:
    sys.stdout.reconfigure(line_buffering=True)

import time
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from prepare import (
    TIME_BUDGET, IMG_SIZE, NUM_CONDITION_CLASSES,
    make_dataloaders, evaluate,
)

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly)
# ---------------------------------------------------------------------------

# Model
BACKBONE_NAME = "efficientnet_b2"
PRETRAINED = True

# Data
BATCH_SIZE = 48
IMAGE_SIZE = 224            # smaller for faster batches

# Optimization
HEAD_LR = 2e-3
BACKBONE_LR = 3e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1

# Training strategy
UNFREEZE_AT_EPOCH = 1       # unfreeze backbone immediately
MIXUP_ALPHA = 0.2           # set to 0 to disable mixup

# Multi-task loss weights
CONDITION_WEIGHT = 0.7
FRAUD_WEIGHT = 0.3

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ClothingModel(nn.Module):
    """
    Dual-stream model: shared backbone processes front and back images
    independently. Features are concatenated and fed to two task heads.

    Returns dict: {"condition": (B, 5), "fraud": (B, 1)}
    """

    def __init__(self, backbone_name=BACKBONE_NAME, pretrained=PRETRAINED):
        super().__init__()

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=3,
            num_classes=0,
            global_pool="avg",
        )
        feat_dim = self.backbone.num_features  # 1408 for efficientnet_b2

        # Condition head (5-class classification)
        self.condition_head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(feat_dim * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, NUM_CONDITION_CLASSES),
        )

        # Fraud head (binary classification)
        self.fraud_head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(feat_dim * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, front, back):
        f_front = self.backbone(front)
        f_back = self.backbone(back)
        combined = torch.cat([f_front, f_back], dim=1)
        return {
            "condition": self.condition_head(combined),
            "fraud": self.fraud_head(combined),
        }

# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def mixup_data(x1, x2, y_cond, y_fraud, alpha=MIXUP_ALPHA):
    """Apply mixup to a batch. Returns mixed inputs and targets."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    idx = torch.randperm(x1.size(0), device=x1.device)
    mixed_x1 = lam * x1 + (1 - lam) * x1[idx]
    mixed_x2 = lam * x2 + (1 - lam) * x2[idx]
    return mixed_x1, mixed_x2, y_cond, y_cond[idx], y_fraud, y_fraud[idx], lam

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
    print("WARNING: MPS not available, using CPU")

print(f"Device: {device}")
print(f"Time budget: {TIME_BUDGET}s")

# Data
train_loader, _ = make_dataloaders(BATCH_SIZE, IMAGE_SIZE)
print(f"Train batches: {len(train_loader)}")

# Model
model = ClothingModel(BACKBONE_NAME, PRETRAINED).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# Loss functions
condition_criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
fraud_criterion = nn.BCEWithLogitsLoss()

# Freeze backbone initially (if UNFREEZE_AT_EPOCH > 1)
if UNFREEZE_AT_EPOCH > 1:
    for param in model.backbone.parameters():
        param.requires_grad = False
    head_params = [p for n, p in model.named_parameters() if "backbone" not in n]
    optimizer = torch.optim.AdamW(head_params, lr=HEAD_LR, weight_decay=WEIGHT_DECAY)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters (heads only): {trainable:,}")
else:
    # All params trainable from start
    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n], "lr": HEAD_LR},
        {"params": list(model.backbone.parameters()), "lr": BACKBONE_LR},
    ], weight_decay=WEIGHT_DECAY)
    print(f"All parameters trainable from start: {total_params:,}")

# Scheduler — use CosineAnnealingLR (robust to param group changes)
steps_per_epoch = len(train_loader)
estimated_total_steps = steps_per_epoch * 5  # rough estimate
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=estimated_total_steps, eta_min=1e-6
)

# ---------------------------------------------------------------------------
# Training loop (wall-clock budget)
# ---------------------------------------------------------------------------

def unfreeze_backbone():
    """Unfreeze backbone and rebuild optimizer + scheduler."""
    global optimizer, scheduler
    for param in model.backbone.parameters():
        param.requires_grad = True
    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n], "lr": HEAD_LR * 0.5},
        {"params": list(model.backbone.parameters()), "lr": BACKBONE_LR},
    ], weight_decay=WEIGHT_DECAY)
    remaining_steps = max(100, estimated_total_steps - step)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=remaining_steps, eta_min=1e-6
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Backbone unfrozen — trainable params: {trainable:,}")

epoch = 0
step = 0
backbone_unfrozen = False

print(f"\nTraining started (budget: {TIME_BUDGET}s)")
t_train_start = time.time()

def elapsed():
    return time.time() - t_train_start

while elapsed() < TIME_BUDGET:
    epoch += 1

    if epoch == UNFREEZE_AT_EPOCH and not backbone_unfrozen:
        unfreeze_backbone()
        backbone_unfrozen = True

    model.train()
    epoch_loss = 0.0
    epoch_correct = 0
    epoch_total = 0

    for fronts, backs, labels in train_loader:
        fronts = fronts.to(device)
        backs = backs.to(device)
        cond_targets = labels["condition"].to(device)
        fraud_targets = labels["fraud"].to(device)

        # Mixup
        if MIXUP_ALPHA > 0:
            fronts, backs, cond_a, cond_b, fraud_a, fraud_b, lam = mixup_data(
                fronts, backs, cond_targets, fraud_targets
            )
        else:
            cond_a, cond_b, fraud_a, fraud_b, lam = cond_targets, cond_targets, fraud_targets, fraud_targets, 1.0

        # Forward
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
            outputs = model(fronts, backs)

            # Condition loss (with mixup)
            cond_loss = lam * condition_criterion(outputs["condition"], cond_a) + \
                        (1 - lam) * condition_criterion(outputs["condition"], cond_b)

            # Fraud loss (with mixup)
            fraud_logits = outputs["fraud"].squeeze(-1)
            fraud_loss = lam * fraud_criterion(fraud_logits, fraud_a) + \
                         (1 - lam) * fraud_criterion(fraud_logits, fraud_b)

            loss = CONDITION_WEIGHT * cond_loss + FRAUD_WEIGHT * fraud_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Track accuracy (on original targets, not mixup)
        with torch.no_grad():
            preds = outputs["condition"].argmax(dim=1)
            epoch_correct += (preds == cond_targets).sum().item()
            epoch_total += fronts.size(0)
            epoch_loss += loss.item() * fronts.size(0)

        if device.type == "mps":
            torch.mps.synchronize()

        step += 1

        if elapsed() >= TIME_BUDGET:
            break

    # Epoch summary
    avg_loss = epoch_loss / max(epoch_total, 1)
    avg_acc = epoch_correct / max(epoch_total, 1)
    remaining = max(0, TIME_BUDGET - elapsed())
    print(f"  epoch {epoch} | loss: {avg_loss:.4f} | acc: {avg_acc:.4f} | remaining: {remaining:.0f}s")

total_training_time = elapsed()
print(f"\nTraining done: {epoch} epochs, {step} steps, {total_training_time:.1f}s")

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

print("Evaluating...")
results = evaluate(model, device)

t_end = time.time()

# Memory tracking
if device.type == "mps":
    peak_memory_mb = torch.mps.driver_allocated_memory() / 1024 / 1024
else:
    peak_memory_mb = 0.0

# ---------------------------------------------------------------------------
# Results output (DO NOT CHANGE FORMAT — parsed by the experiment loop)
# ---------------------------------------------------------------------------

print("=== RESULTS ===")
print(f"condition_acc: {results['condition_acc']:.4f}")
print(f"fraud_f1: {results['fraud_f1']:.4f}")
print(f"combined_score: {results['combined_score']:.4f}")
print(f"peak_memory_mb: {peak_memory_mb:.1f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds: {t_end - t_start:.1f}")
print(f"num_epochs_completed: {epoch}")
print("=== END RESULTS ===")
