"""
Autoresearch training script for clothing condition + fraud detection.
Single-file, single-device (MPS). This is the file the agent modifies.

Usage: python3 train.py
"""

import sys
import signal
if not sys.stdout.line_buffering:
    sys.stdout.reconfigure(line_buffering=True)

def _timeout_handler(signum, frame):
    print("\nHARD TIMEOUT — killing process")
    sys.exit(1)
signal.signal(signal.SIGALRM, _timeout_handler)
signal.alarm(600)

import time
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
# Hyperparameters
# ---------------------------------------------------------------------------

BACKBONE_NAME = "efficientnet_b2"
PRETRAINED = True
BATCH_SIZE = 48
IMAGE_SIZE = 224
HEAD_LR = 5e-3              # aggressive LR for fast learning
BACKBONE_LR = 1e-3          # higher backbone LR too
WEIGHT_DECAY = 1e-3         # stronger regularization to compensate high LR
LABEL_SMOOTHING = 0.15
MIXUP_ALPHA = 0.0           # disabled for speed
CONDITION_WEIGHT = 0.8
FRAUD_WEIGHT = 0.2
GRAD_CLIP = 1.0             # gradient clipping for stability with high LR

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ClothingModel(nn.Module):
    def __init__(self, backbone_name=BACKBONE_NAME, pretrained=PRETRAINED):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained,
            in_chans=3, num_classes=0, global_pool="avg",
        )
        with torch.no_grad():
            feat_dim = self.backbone(torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)).shape[1]

        self.condition_head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(feat_dim * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, NUM_CONDITION_CLASSES),
        )
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
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Device: {device}")
print(f"Time budget: {TIME_BUDGET}s")

train_loader, _ = make_dataloaders(BATCH_SIZE, IMAGE_SIZE)
print(f"Train batches: {len(train_loader)}")

model = ClothingModel(BACKBONE_NAME, PRETRAINED).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

condition_criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
fraud_criterion = nn.BCEWithLogitsLoss()

# All params trainable from start with aggressive LR
optimizer = torch.optim.AdamW([
    {"params": [p for n, p in model.named_parameters() if "backbone" not in n], "lr": HEAD_LR},
    {"params": list(model.backbone.parameters()), "lr": BACKBONE_LR},
], weight_decay=WEIGHT_DECAY)

# Constant LR — no warmup, no decay. Maximize learning in limited steps.

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

epoch = 0
step = 0
deadline = time.time() + TIME_BUDGET
print(f"\nTraining started")

while time.time() < deadline:
    epoch += 1
    model.train()
    epoch_loss = 0.0
    epoch_correct = 0
    epoch_total = 0

    for fronts, backs, labels in train_loader:
        if time.time() >= deadline:
            break

        fronts = fronts.to(device)
        backs = backs.to(device)
        cond_targets = labels["condition"].to(device)
        fraud_targets = labels["fraud"].to(device)

        with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
            outputs = model(fronts, backs)
            cond_loss = condition_criterion(outputs["condition"], cond_targets)
            fraud_logits = outputs["fraud"].squeeze(-1)
            fraud_loss = fraud_criterion(fraud_logits, fraud_targets)
            loss = CONDITION_WEIGHT * cond_loss + FRAUD_WEIGHT * fraud_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        with torch.no_grad():
            preds = outputs["condition"].argmax(dim=1)
            epoch_correct += (preds == cond_targets).sum().item()
            epoch_total += fronts.size(0)
            epoch_loss += loss.item() * fronts.size(0)

        if device.type == "mps":
            torch.mps.synchronize()
        step += 1

        if step % 50 == 0:
            remaining = max(0, deadline - time.time())
            print(f"    step {step} | loss: {loss.item():.4f} | remaining: {remaining:.0f}s")

    avg_loss = epoch_loss / max(epoch_total, 1)
    avg_acc = epoch_correct / max(epoch_total, 1)
    remaining = max(0, deadline - time.time())
    print(f"  epoch {epoch} | loss: {avg_loss:.4f} | acc: {avg_acc:.4f} | steps: {step} | remaining: {remaining:.0f}s")

total_training_time = time.time() - (deadline - TIME_BUDGET)
print(f"\nTraining done: {epoch} epochs, {step} steps, {total_training_time:.1f}s")

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

print("Evaluating...")
results = evaluate(model, device)
t_end = time.time()

if device.type == "mps":
    peak_memory_mb = torch.mps.driver_allocated_memory() / 1024 / 1024
else:
    peak_memory_mb = 0.0

print("=== RESULTS ===")
print(f"condition_acc: {results['condition_acc']:.4f}")
print(f"fraud_f1: {results['fraud_f1']:.4f}")
print(f"combined_score: {results['combined_score']:.4f}")
print(f"peak_memory_mb: {peak_memory_mb:.1f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds: {t_end - t_start:.1f}")
print(f"num_epochs_completed: {epoch}")
print("=== END RESULTS ===")
