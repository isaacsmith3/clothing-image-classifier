"""
fraud_auditor.py
----------------
Vision-as-auditor experiment: use a trained condition model as an independent
judge of each item's condition, and flag items where the model's prediction
disagrees with the seller's stated condition by more than a threshold.

This is the experiment proposed in Results_v2.md takeaway #4: a condition model
could detect fraud cases where BOTH the condition and defect labels have been
falsified — the images still reveal the truth. CLIP was too noisy to do this
(F1 0.005). The hypothesis here is that a task-specific fine-tuned model can.

Phases:
  1. Load pretrained V2 multi-task model, run it as auditor on the full test
     set. Compute |pred - claimed| discrepancy and sweep thresholds for PR/F1.
  2. Train a dedicated station-3-clean auditor (cleanest labels). Evaluate with
     and without per-station bias correction (the station experiment showed
     S3->S1 has systematic ~1 grade offset).
  3. Compare all auditors against heuristic-rules and CLIP baselines.

Usage:
    python src/fraud_auditor.py --phase {v2,train,eval,all}

Outputs:
    checkpoints_auditor/best_auditor.pt   station-3 auditor weights
    results/auditor_predictions.csv       per-item predictions from each auditor
    report/figures/auditor_*.png          PR curves, discrepancy histograms
    results/Results_auditor.md            written separately
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

import timm

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = PROJECT_ROOT / "data" / "cleaned_metadata_v2.csv"
V2_CKPT = PROJECT_ROOT / "checkpoints_v2" / "best_model_v2.pt"
AUDITOR_CKPT_DIR = PROJECT_ROOT / "checkpoints_auditor"
AUDITOR_CKPT = AUDITOR_CKPT_DIR / "best_auditor.pt"
PREDS_CSV = PROJECT_ROOT / "results" / "auditor_predictions.csv"
FIGURES_DIR = PROJECT_ROOT / "report" / "figures"

AUDITOR_CKPT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
USE_AMP = DEVICE.type in ("cuda", "mps")

IMG_SIZE = 260
BATCH_SIZE = 32

# ── Transforms / Dataset ──────────────────────────────────────────────────────
eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=20, scale=(0.85, 1.15), shear=10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.08),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
])


class ClothingDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None, return_labels=True):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.return_labels = return_labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        front = Image.open(row["front_path"]).convert("RGB")
        back = Image.open(row["back_path"]).convert("RGB")
        if self.transform:
            front = self.transform(front)
            back = self.transform(back)
        if self.return_labels:
            return front, back, torch.tensor(row["condition"] - 1, dtype=torch.long)
        return front, back


# ── Models ────────────────────────────────────────────────────────────────────
class ClothingMultiTaskModel(nn.Module):
    """V2 multi-task model — matches checkpoints_v2/best_model_v2.pt exactly."""

    def __init__(self, backbone_name="efficientnet_b2", pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained,
            in_chans=3, num_classes=0, global_pool="avg",
        )
        feat_dim = self.backbone.num_features
        cd = feat_dim * 2
        self.condition_head = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(cd, 256),
            nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(256, 5),
        )
        self.stains_head = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(cd, 64),
            nn.ReLU(inplace=True), nn.Linear(64, 3),
        )
        self.holes_head = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(cd, 64),
            nn.ReLU(inplace=True), nn.Linear(64, 3),
        )
        self.pilling_head = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(cd, 64),
            nn.ReLU(inplace=True), nn.Linear(64, 5),
        )

    def forward(self, front, back):
        ff = self.backbone(front)
        fb = self.backbone(back)
        combined = torch.cat([ff, fb], dim=1)
        return {
            "condition": self.condition_head(combined),
            "stains": self.stains_head(combined),
            "holes": self.holes_head(combined),
            "pilling": self.pilling_head(combined),
        }


class SimpleConditionModel(nn.Module):
    """Single-head condition model — used to train the station-3 auditor."""

    def __init__(self, backbone_name="efficientnet_b2", pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained,
            in_chans=3, num_classes=0, global_pool="avg",
        )
        cd = self.backbone.num_features * 2
        self.head = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(cd, 256),
            nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(256, 5),
        )

    def forward(self, front, back):
        return self.head(torch.cat([self.backbone(front), self.backbone(back)], dim=1))


# ── Ordinal soft-label loss (from V2/station_experiment) ──────────────────────
def ordinal_soft_labels(targets, num_classes=5, sigma=1.0):
    classes = torch.arange(num_classes, device=targets.device).float()
    tgt = targets.float().unsqueeze(1)
    soft = torch.exp(-0.5 * ((classes - tgt) / sigma) ** 2)
    return soft / soft.sum(dim=1, keepdim=True)


def ordinal_kl_loss(logits, targets, num_classes=5, sigma=1.0):
    soft = ordinal_soft_labels(targets, num_classes, sigma)
    log_probs = F.log_softmax(logits, dim=1)
    return F.kl_div(log_probs, soft, reduction="batchmean")


# ── Inference helpers ─────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(model, df: pd.DataFrame, kind: str) -> dict:
    """
    Run a model on df and return:
      argmax_pred  (int 0-4)
      expected_pred (float 0-4) = sum_k k * softmax_k, the ordinal expected value
      softmax     (N, 5)
    `kind` is "multitask" (returns dict) or "simple" (returns logits).
    """
    loader = DataLoader(
        ClothingDataset(df, transform=eval_tf, return_labels=False),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )
    model.eval()
    all_argmax, all_expected, all_softmax = [], [], []
    classes = torch.arange(5, device=DEVICE).float()

    for batch in tqdm(loader, desc=f"infer ({kind})", leave=False):
        fronts, backs = batch
        fronts, backs = fronts.to(DEVICE), backs.to(DEVICE)
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=USE_AMP):
            out = model(fronts, backs)
            logits = out["condition"] if kind == "multitask" else out
        probs = F.softmax(logits.float(), dim=1)
        argmax = probs.argmax(1)
        expected = (probs * classes).sum(1)
        all_argmax.append(argmax.cpu().numpy())
        all_expected.append(expected.cpu().numpy())
        all_softmax.append(probs.cpu().numpy())
        if DEVICE.type == "mps":
            torch.mps.synchronize()

    return {
        "argmax": np.concatenate(all_argmax),
        "expected": np.concatenate(all_expected),
        "softmax": np.concatenate(all_softmax),
    }


# ── Phase 1: use V2 as auditor (no new training) ──────────────────────────────
def run_v2_auditor(test_df: pd.DataFrame) -> dict:
    print("\n" + "=" * 60)
    print("  Phase 1: V2 multi-task model as auditor (no training)")
    print("=" * 60)
    model = ClothingMultiTaskModel(pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(V2_CKPT, map_location=DEVICE, weights_only=False))
    print(f"  Loaded weights from {V2_CKPT.name}")
    out = run_inference(model, test_df, kind="multitask")
    del model
    if DEVICE.type == "mps":
        torch.mps.empty_cache()
    return out


# ── Phase 2: train station-3-clean auditor ────────────────────────────────────
def train_station3_auditor(train_df: pd.DataFrame, val_df: pd.DataFrame,
                           num_epochs: int = 15, patience: int = 3) -> dict:
    print("\n" + "=" * 60)
    print("  Phase 2: training station-3-clean condition auditor")
    print(f"  Train: {len(train_df):,}  Val: {len(val_df):,}")
    print("=" * 60)

    train_loader = DataLoader(
        ClothingDataset(train_df, transform=train_tf),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        ClothingDataset(val_df, transform=eval_tf),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )

    model = SimpleConditionModel(pretrained=True).to(DEVICE)
    for p in model.backbone.parameters():
        p.requires_grad = False

    head_params = [p for n, p in model.named_parameters() if "backbone" not in n]
    optimizer = torch.optim.AdamW(head_params, lr=1e-3, weight_decay=1e-4)
    spe = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, epochs=num_epochs, steps_per_epoch=spe,
        pct_start=0.1, anneal_strategy="cos",
    )

    unfreeze_at = 3
    best_val_loss = float("inf")
    patience_ctr = 0
    history = []

    for epoch in range(1, num_epochs + 1):
        if epoch == unfreeze_at:
            for p in model.backbone.parameters():
                p.requires_grad = True
            optimizer.add_param_group({
                "params": list(model.backbone.parameters()),
                "lr": 1e-4, "weight_decay": 1e-4,
            })
            remaining = num_epochs - epoch + 1
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=[5e-4, 1e-4],
                epochs=remaining, steps_per_epoch=spe,
                pct_start=0.05, anneal_strategy="cos",
            )
            print(f"  backbone unfrozen at epoch {epoch}")

        # Train
        model.train()
        train_loss = 0.0
        n_train = 0
        for fronts, backs, tgt in tqdm(train_loader, desc=f"e{epoch} train", leave=False):
            fronts, backs, tgt = fronts.to(DEVICE), backs.to(DEVICE), tgt.to(DEVICE)
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=USE_AMP):
                logits = model(fronts, backs)
                loss = ordinal_kl_loss(logits, tgt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item() * fronts.size(0)
            n_train += fronts.size(0)
            if DEVICE.type == "mps":
                torch.mps.synchronize()
        train_loss /= n_train

        # Val
        model.eval()
        val_loss = 0.0
        n_val = 0
        all_p, all_t = [], []
        with torch.no_grad():
            for fronts, backs, tgt in val_loader:
                fronts, backs, tgt = fronts.to(DEVICE), backs.to(DEVICE), tgt.to(DEVICE)
                with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=USE_AMP):
                    logits = model(fronts, backs)
                    loss = ordinal_kl_loss(logits, tgt)
                val_loss += loss.item() * fronts.size(0)
                n_val += fronts.size(0)
                all_p.append(logits.argmax(1).cpu().numpy())
                all_t.append(tgt.cpu().numpy())
        val_loss /= n_val
        all_p = np.concatenate(all_p)
        all_t = np.concatenate(all_t)
        acc = (all_p == all_t).mean()
        mae = np.abs(all_p - all_t).mean()
        one_off = (np.abs(all_p - all_t) <= 1).mean()

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            patience_ctr = 0
            torch.save(model.state_dict(), AUDITOR_CKPT)
        else:
            patience_ctr += 1
        mark = "✓ saved" if improved else f"  ({patience_ctr}/{patience})"
        print(f"  e{epoch:02d}  train={train_loss:.4f}  val={val_loss:.4f}  "
              f"acc={acc:.4f}  MAE={mae:.3f}  1-off={one_off:.4f}  {mark}")
        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "acc": float(acc), "mae": float(mae), "one_off": float(one_off),
        })
        if patience_ctr >= patience:
            print(f"  early stopped at epoch {epoch}")
            break

    with open(AUDITOR_CKPT_DIR / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"  saved auditor to {AUDITOR_CKPT}")
    del model
    if DEVICE.type == "mps":
        torch.mps.empty_cache()
    return {"history": history, "best_val_loss": best_val_loss}


def run_auditor_inference(test_df: pd.DataFrame) -> dict:
    print("\n  Running station-3 auditor on full test set...")
    model = SimpleConditionModel(pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(AUDITOR_CKPT, map_location=DEVICE, weights_only=False))
    out = run_inference(model, test_df, kind="simple")
    del model
    if DEVICE.type == "mps":
        torch.mps.empty_cache()
    return out


# ── Fraud detection evaluation ────────────────────────────────────────────────
def heuristic_flags(df: pd.DataFrame) -> np.ndarray:
    r1 = (df["condition"] >= 4) & ((df["stains"] == 2) | (df["holes"] == 2))
    r2 = (df["condition"] >= 4) & (df["pilling"] <= 2)
    r3 = (df["condition"] == 5) & ((df["stains"] >= 1) | (df["holes"] >= 1))
    return (r1 | r2 | r3).values.astype(int)


def sweep_threshold(score: np.ndarray, y_true: np.ndarray,
                    thresholds: np.ndarray) -> pd.DataFrame:
    """Sweep thresholds on a ranking score (higher = more suspicious)."""
    rows = []
    for t in thresholds:
        y_pred = (score >= t).astype(int)
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0,
        )
        rows.append({"threshold": float(t), "precision": float(p),
                     "recall": float(r), "f1": float(f),
                     "flagged": int(y_pred.sum())})
    return pd.DataFrame(rows)


def precision_at_k(score: np.ndarray, y_true: np.ndarray, ks: list[int]) -> dict:
    """Take the top-k items by score. Return precision@k and recall@k for each k."""
    order = np.argsort(-score)  # descending
    out = {}
    total_pos = int(y_true.sum())
    for k in ks:
        top = order[:k]
        caught = int(y_true[top].sum())
        out[k] = {
            "precision": caught / k if k > 0 else 0.0,
            "recall": caught / total_pos if total_pos > 0 else 0.0,
            "caught": caught,
        }
    return out


def ranker_metrics(score: np.ndarray, y_true: np.ndarray) -> dict:
    """ROC-AUC, PR-AUC, and precision@k for the key flag budgets."""
    return {
        "roc_auc": float(roc_auc_score(y_true, score)),
        "pr_auc": float(average_precision_score(y_true, score)),
        "base_rate": float(y_true.mean()),
        "precision_at_k": precision_at_k(
            score, y_true, ks=[23, 50, 100, 200, 500]
        ),
    }


def per_station_calibrate(predictions: np.ndarray, stations: np.ndarray,
                          claimed: np.ndarray) -> np.ndarray:
    """
    Subtract per-station median bias. Target: median(pred - claimed) = 0 per
    station among the majority (non-fraud) population. This removes the
    systematic ~1 grade offset between station-3 and station-1 annotators.
    """
    calibrated = predictions.astype(float).copy()
    for s in np.unique(stations):
        mask = stations == s
        bias = float(np.median(calibrated[mask] - claimed[mask]))
        calibrated[mask] -= bias
    return calibrated


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["v2", "train", "eval", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    df = pd.read_csv(DATA_CSV)
    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy().reset_index(drop=True)
    s3_train = train_df[train_df["station"] == "station3"].copy()
    s3_test_mask = (test_df["station"] == "station3")

    print(f"train total={len(train_df):,}  test total={len(test_df):,}  "
          f"s3 train={len(s3_train):,}")
    print(f"test fraud: {int(test_df['is_fraud_candidate'].sum())} / {len(test_df):,}")

    y_true = test_df["is_fraud_candidate"].values.astype(int)
    claimed = (test_df["condition"].values - 1).astype(float)  # 0-4
    stations = test_df["station"].values

    results = {"fraud_ground_truth": int(y_true.sum())}
    preds_out = test_df[["timestamp", "station", "condition",
                         "stains", "holes", "pilling", "is_fraud_candidate"]].copy()

    # Phase 1: V2 as auditor
    if args.phase in ("v2", "all"):
        v2_out = run_v2_auditor(test_df)
        preds_out["v2_pred_argmax"] = v2_out["argmax"]
        preds_out["v2_pred_expected"] = v2_out["expected"]

    # Phase 2: train the station-3 auditor
    if args.phase in ("train", "all"):
        # Split s3_train 90/10 for validation during training
        rng = np.random.RandomState(42)
        idx = rng.permutation(len(s3_train))
        cut = int(0.9 * len(idx))
        s3_tr = s3_train.iloc[idx[:cut]].copy()
        s3_val = s3_train.iloc[idx[cut:]].copy()
        train_station3_auditor(s3_tr, s3_val, num_epochs=args.epochs)

    # Inference with station-3 auditor
    if args.phase in ("eval", "all") and AUDITOR_CKPT.exists():
        a_out = run_auditor_inference(test_df)
        preds_out["s3_pred_argmax"] = a_out["argmax"]
        preds_out["s3_pred_expected"] = a_out["expected"]

    # Save predictions
    preds_out.to_csv(PREDS_CSV, index=False)
    print(f"\n  predictions saved to {PREDS_CSV}")

    # Evaluation: discrepancy-based fraud detection
    print("\n" + "=" * 60)
    print("  Fraud detection evaluation")
    print("=" * 60)

    # Heuristic baseline (test set only)
    y_heur = heuristic_flags(test_df)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_heur, average="binary", zero_division=0)
    print(f"  Heuristic rules:     P={p:.4f}  R={r:.4f}  F1={f:.4f}  flagged={int(y_heur.sum())}")
    results["heuristic"] = {"precision": float(p), "recall": float(r),
                            "f1": float(f), "flagged": int(y_heur.sum())}

    # For each auditor, compute signed discrepancy (claimed - pred). Positive =
    # model thinks item is worse than seller claims = suspicious. This is the
    # fraud-specific directional signal, different from |pred - claimed|.
    auditors = []
    if "v2_pred_expected" in preds_out.columns:
        auditors.append(("v2", preds_out["v2_pred_expected"].values))
    if "s3_pred_expected" in preds_out.columns:
        auditors.append(("s3_raw", preds_out["s3_pred_expected"].values))
        cal = per_station_calibrate(preds_out["s3_pred_expected"].values, stations, claimed)
        preds_out["s3_pred_calibrated"] = cal
        auditors.append(("s3_calibrated", cal))

    thresholds = np.linspace(-2.0, 4.0, 121)
    sweep_results = {}
    for name, pred in auditors:
        signed = claimed - pred  # higher = more suspicious
        preds_out[f"{name}_signed_disc"] = signed

        # Threshold sweep for best-F1 operating point
        sweep = sweep_threshold(signed, y_true, thresholds)
        best = sweep.iloc[sweep["f1"].idxmax()]
        sweep_results[name] = sweep
        sweep.to_csv(PROJECT_ROOT / "results" / f"auditor_sweep_{name}.csv", index=False)

        # Ranking metrics (the primary view given 0.36% base rate)
        rm = ranker_metrics(signed, y_true)
        print(f"\n  {name} auditor (signed discrepancy = claimed - pred):")
        print(f"    ROC-AUC = {rm['roc_auc']:.4f}   PR-AUC = {rm['pr_auc']:.4f}   "
              f"(base rate {rm['base_rate']:.4f})")
        print(f"    best-F1 threshold = {best['threshold']:+.2f}  "
              f"P={best['precision']:.4f}  R={best['recall']:.4f}  "
              f"F1={best['f1']:.4f}  flagged={int(best['flagged'])}")
        print(f"    Precision@k:")
        for k, v in rm["precision_at_k"].items():
            print(f"      k={k:>4}: P={v['precision']:.3f}  R={v['recall']:.3f}  "
                  f"caught={v['caught']}/{int(y_true.sum())}")

        results[name] = {
            "roc_auc": rm["roc_auc"],
            "pr_auc": rm["pr_auc"],
            "best_f1_threshold": float(best["threshold"]),
            "best_f1_precision": float(best["precision"]),
            "best_f1_recall": float(best["recall"]),
            "best_f1": float(best["f1"]),
            "best_f1_flagged": int(best["flagged"]),
            "precision_at_k": rm["precision_at_k"],
        }

    # Auditor-as-reranker over heuristic flags: of the 102 heuristic flags,
    # can the auditor rank the 23 true fraud to the top?
    print("\n" + "=" * 60)
    print("  Auditor as reranker over heuristic flags")
    print("=" * 60)
    heur_mask = y_heur.astype(bool)
    heur_idx = np.where(heur_mask)[0]
    print(f"  Heuristic flags: {heur_mask.sum()}  "
          f"(of which {int(y_true[heur_mask].sum())} are true fraud)")
    for name, pred in auditors:
        signed_within = (claimed - pred)[heur_mask]
        y_within = y_true[heur_mask]
        if len(np.unique(y_within)) < 2:
            continue
        rm = ranker_metrics(signed_within, y_within)
        print(f"\n  {name} reranking within heuristic flags:")
        print(f"    ROC-AUC = {rm['roc_auc']:.4f}  PR-AUC = {rm['pr_auc']:.4f}")
        for k, v in rm["precision_at_k"].items():
            if k <= len(y_within):
                print(f"      top-{k:>3}: P={v['precision']:.3f}  "
                      f"R={v['recall']:.3f}  caught={v['caught']}/{int(y_within.sum())}")
        results[f"{name}_rerank_heuristic"] = rm

    # Save final predictions with calibration column
    preds_out.to_csv(PREDS_CSV, index=False)
    with open(PROJECT_ROOT / "results" / "auditor_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Quick per-station report: auditor accuracy on non-fraud items
    print("\n" + "=" * 60)
    print("  Per-station auditor accuracy (non-fraud items only)")
    print("=" * 60)
    nonfraud_mask = y_true == 0
    for name, pred in auditors:
        print(f"\n  {name}:")
        for s in ["station1", "station2", "station3"]:
            m = nonfraud_mask & (stations == s)
            if m.sum() == 0:
                continue
            pred_argmax = np.round(pred[m]).clip(0, 4).astype(int)
            true_c = claimed[m].astype(int)
            acc = (pred_argmax == true_c).mean()
            mae = np.abs(pred[m] - claimed[m]).mean()
            print(f"    {s}: n={int(m.sum())}  acc={acc:.3f}  MAE={mae:.3f}")

    print("\n  done.")


if __name__ == "__main__":
    main()
