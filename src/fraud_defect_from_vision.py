"""
fraud_defect_from_vision.py
---------------------------
Extracts V2's stains/holes predictions on the test set and evaluates:

  1. How accurate are V2's defect predictions vs the (presumed honest) labels?
  2. If we plug V2's defect *predictions* into the same heuristic rules
     (in place of the seller-reported defect fields), does the fraud
     detection still work?

This tests whether a vision model can substitute for trustworthy defect
reporting. The standard heuristic assumes the seller honestly reports
defects. That's a strong assumption — a real fraudster would falsify
both fields. The vision heuristic does not rely on seller honesty for
the defect fields at all.
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

# Reuse machinery from fraud_auditor.py
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fraud_auditor import (  # noqa: E402
    ClothingMultiTaskModel, ClothingDataset, eval_tf,
    DEVICE, USE_AMP, BATCH_SIZE, V2_CKPT, DATA_CSV,
    heuristic_flags,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_CSV = PROJECT_ROOT / "results" / "v2_defect_predictions.csv"
RESULTS_JSON = PROJECT_ROOT / "results" / "vision_heuristic_results.json"


@torch.no_grad()
def extract_all_heads(model, df):
    loader = DataLoader(
        ClothingDataset(df, transform=eval_tf, return_labels=False),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )
    model.eval()
    classes5 = torch.arange(5, device=DEVICE).float()
    classes3 = torch.arange(3, device=DEVICE).float()
    out_cond, out_stains, out_holes, out_pilling = [], [], [], []
    out_stains_arg, out_holes_arg = [], []
    for fronts, backs in tqdm(loader, desc="infer V2 all-heads", leave=False):
        fronts, backs = fronts.to(DEVICE), backs.to(DEVICE)
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=USE_AMP):
            out = model(fronts, backs)
        p_cond = F.softmax(out["condition"].float(), dim=1)
        p_stains = F.softmax(out["stains"].float(), dim=1)
        p_holes = F.softmax(out["holes"].float(), dim=1)
        p_pill = F.softmax(out["pilling"].float(), dim=1)
        out_cond.append((p_cond * classes5).sum(1).cpu().numpy())
        out_stains.append((p_stains * classes3).sum(1).cpu().numpy())
        out_holes.append((p_holes * classes3).sum(1).cpu().numpy())
        out_pilling.append((p_pill * classes5).sum(1).cpu().numpy())
        out_stains_arg.append(p_stains.argmax(1).cpu().numpy())
        out_holes_arg.append(p_holes.argmax(1).cpu().numpy())
        if DEVICE.type == "mps":
            torch.mps.synchronize()
    return {
        "cond_expected": np.concatenate(out_cond),
        "stains_expected": np.concatenate(out_stains),
        "holes_expected": np.concatenate(out_holes),
        "pilling_expected": np.concatenate(out_pilling),
        "stains_argmax": np.concatenate(out_stains_arg),
        "holes_argmax": np.concatenate(out_holes_arg),
    }


def vision_heuristic_flags(test_df: pd.DataFrame, stains_pred: np.ndarray,
                           holes_pred: np.ndarray) -> np.ndarray:
    """
    Same rules as heuristic_flags, but defect fields come from the vision model.
    We only replace stains/holes; pilling stays as the claimed field because
    pilling is claimed alongside condition in the same label block.
    """
    tmp = test_df.copy()
    tmp["stains"] = stains_pred
    tmp["holes"] = holes_pred
    return heuristic_flags(tmp)


def main():
    df = pd.read_csv(DATA_CSV)
    test_df = df[df["split"] == "test"].copy().reset_index(drop=True)
    y_true = test_df["is_fraud_candidate"].values.astype(int)

    print(f"Loading V2 model from {V2_CKPT}")
    model = ClothingMultiTaskModel(pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(V2_CKPT, map_location=DEVICE, weights_only=False))

    preds = extract_all_heads(model, test_df)

    # Save per-item table
    out_df = test_df[["timestamp", "station", "condition", "stains", "holes",
                      "pilling", "is_fraud_candidate"]].copy()
    out_df["v2_cond_expected"] = preds["cond_expected"]
    out_df["v2_stains_expected"] = preds["stains_expected"]
    out_df["v2_holes_expected"] = preds["holes_expected"]
    out_df["v2_stains_argmax"] = preds["stains_argmax"]
    out_df["v2_holes_argmax"] = preds["holes_argmax"]
    out_df["v2_pilling_expected"] = preds["pilling_expected"]
    out_df.to_csv(OUT_CSV, index=False)
    print(f"  saved {OUT_CSV}")

    results = {}

    # --- Accuracy of V2 defect heads on non-fraud items
    print("\n=== V2 defect-head accuracy ===")
    nonfraud = y_true == 0
    for name, (pred, true) in {
        "stains": (preds["stains_argmax"][nonfraud], test_df["stains"][nonfraud].values),
        "holes": (preds["holes_argmax"][nonfraud], test_df["holes"][nonfraud].values),
    }.items():
        acc = float((pred == true).mean())
        cm = confusion_matrix(true, pred, labels=[0, 1, 2])
        print(f"  {name}: acc={acc:.3f}  n={len(pred)}")
        print(f"    confusion (rows=true, cols=pred):\n{cm}")
        results[f"v2_{name}_accuracy"] = acc
        results[f"v2_{name}_confusion"] = cm.tolist()

    # --- Baseline heuristic (for reference)
    y_baseline = heuristic_flags(test_df)
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_baseline, average="binary", zero_division=0)
    print(f"\nBaseline heuristic (seller-reported defects):")
    print(f"  P={p:.4f}  R={r:.4f}  F1={f:.4f}  flagged={int(y_baseline.sum())}")
    results["baseline_heuristic"] = {
        "precision": float(p), "recall": float(r), "f1": float(f),
        "flagged": int(y_baseline.sum()),
    }

    # --- Vision-only heuristic (V2-predicted defects)
    y_vision = vision_heuristic_flags(
        test_df, preds["stains_argmax"], preds["holes_argmax"])
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_vision, average="binary", zero_division=0)
    print(f"\nVision heuristic (V2-predicted defects substituted for claimed defects):")
    print(f"  P={p:.4f}  R={r:.4f}  F1={f:.4f}  flagged={int(y_vision.sum())}")
    print(f"  Fraud caught: {int((y_vision & y_true).sum())} / {int(y_true.sum())}")
    results["vision_heuristic"] = {
        "precision": float(p), "recall": float(r), "f1": float(f),
        "flagged": int(y_vision.sum()),
        "fraud_caught": int((y_vision & y_true).sum()),
    }

    # --- Combined (either flag)
    y_comb = (y_baseline | y_vision).astype(int)
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_comb, average="binary", zero_division=0)
    print(f"\nCombined (seller OR vision heuristic):")
    print(f"  P={p:.4f}  R={r:.4f}  F1={f:.4f}  flagged={int(y_comb.sum())}")
    results["combined_heuristic"] = {
        "precision": float(p), "recall": float(r), "f1": float(f),
        "flagged": int(y_comb.sum()),
    }

    # --- Continuous vision fraud score: expected defect severity under high claim
    # score = (claimed condition >= 4) * (stains_expected + holes_expected)
    high_claim = (test_df["condition"] >= 4).values.astype(float)
    defect_score = preds["stains_expected"] + preds["holes_expected"]
    vision_score = high_claim * defect_score  # only positive when claim is high

    if y_true.sum() > 0:
        auc = float(roc_auc_score(y_true, vision_score))
        ap = float(average_precision_score(y_true, vision_score))
        print(f"\nContinuous vision score: ROC-AUC={auc:.4f}  PR-AUC={ap:.4f}")
        results["continuous_vision_score"] = {"roc_auc": auc, "pr_auc": ap}

        # top-k
        order = np.argsort(-vision_score)
        for k in [23, 50, 100]:
            caught = int(y_true[order[:k]].sum())
            print(f"  top-{k}: caught {caught}/{int(y_true.sum())} "
                  f"(P={caught/k:.3f})")
            results.setdefault("vision_score_topk", {})[str(k)] = {
                "caught": caught, "precision": caught / k,
            }

    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {RESULTS_JSON}")


if __name__ == "__main__":
    main()
