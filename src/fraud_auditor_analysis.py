"""
fraud_auditor_analysis.py
-------------------------
Runs after fraud_auditor.py has written src/auditor_predictions.csv with
both V2 and S3-auditor columns. Produces:
  - PR curve figure comparing V2, S3-raw, S3-calibrated, heuristic, CLIP
  - Discrepancy histogram (fraud vs non-fraud)
  - Per-station auditor accuracy table
  - Combined ranker: heuristic gate -> S3-calibrated rerank
  - results/auditor_results.json with final numbers
  - results/Results_auditor.md  (populated from template)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREDS_CSV = PROJECT_ROOT / "results" / "auditor_predictions.csv"
FIGURES_DIR = PROJECT_ROOT / "report" / "figures"
RESULTS_JSON = PROJECT_ROOT / "results" / "auditor_results.json"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def heuristic_flags(df: pd.DataFrame) -> np.ndarray:
    r1 = (df["condition"] >= 4) & ((df["stains"] == 2) | (df["holes"] == 2))
    r2 = (df["condition"] >= 4) & (df["pilling"] <= 2)
    r3 = (df["condition"] == 5) & ((df["stains"] >= 1) | (df["holes"] >= 1))
    return (r1 | r2 | r3).values.astype(int)


def per_station_calibrate(predictions: np.ndarray, stations: np.ndarray,
                          claimed: np.ndarray) -> np.ndarray:
    """Remove per-station median (pred - claimed) bias."""
    calibrated = predictions.astype(float).copy()
    for s in np.unique(stations):
        mask = stations == s
        bias = float(np.median(calibrated[mask] - claimed[mask]))
        calibrated[mask] -= bias
    return calibrated


def precision_at_k(score, y_true, ks):
    order = np.argsort(-score)
    total = int(y_true.sum())
    return {
        k: {
            "precision": float(y_true[order[:k]].sum() / k) if k else 0.0,
            "recall": float(y_true[order[:k]].sum() / total) if total else 0.0,
            "caught": int(y_true[order[:k]].sum()),
        }
        for k in ks
    }


def ranker_metrics(score, y_true):
    return {
        "roc_auc": float(roc_auc_score(y_true, score)),
        "pr_auc": float(average_precision_score(y_true, score)),
        "precision_at_k": precision_at_k(score, y_true, [23, 50, 100, 200]),
    }


def summarize_auditor(name: str, signed: np.ndarray, y_true: np.ndarray,
                      heur_mask: np.ndarray) -> dict:
    print(f"\n  [{name}]")
    rm = ranker_metrics(signed, y_true)
    print(f"    standalone  ROC-AUC={rm['roc_auc']:.4f}  "
          f"PR-AUC={rm['pr_auc']:.4f}")
    for k, v in rm["precision_at_k"].items():
        print(f"      top-{k:>3}: caught {v['caught']}/{int(y_true.sum())} "
              f"(P={v['precision']:.3f} R={v['recall']:.3f})")

    rerank = None
    signed_in = signed[heur_mask]
    y_in = y_true[heur_mask]
    if len(np.unique(y_in)) > 1:
        rerank = {
            "roc_auc": float(roc_auc_score(y_in, signed_in)),
            "pr_auc": float(average_precision_score(y_in, signed_in)),
            "precision_at_k": precision_at_k(signed_in, y_in, [10, 23, 50]),
        }
        print(f"    rerank(heur) ROC-AUC={rerank['roc_auc']:.4f}  "
              f"PR-AUC={rerank['pr_auc']:.4f}")
        for k, v in rerank["precision_at_k"].items():
            if k <= len(y_in):
                print(f"      top-{k:>3}: caught {v['caught']}/{int(y_in.sum())} "
                      f"(P={v['precision']:.3f} R={v['recall']:.3f})")
    return {"standalone": rm, "rerank_heuristic": rerank}


def plot_pr_curves(auditors: dict, y_true: np.ndarray, heuristic_point: tuple,
                   out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, score in auditors.items():
        precs, recs, _ = precision_recall_curve(y_true, score)
        ap = average_precision_score(y_true, score)
        ax.plot(recs, precs, label=f"{name}  (AP={ap:.3f})", linewidth=2)
    h_p, h_r = heuristic_point
    ax.scatter([h_r], [h_p], color="black", s=100, marker="*",
               label=f"heuristic rules (P={h_p:.2f}, R={h_r:.2f})", zorder=5)
    ax.axhline(float(y_true.mean()), color="grey", linestyle=":",
               label=f"random baseline (P={y_true.mean():.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Fraud detection PR curves: auditor vs heuristic")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  saved {out_path}")


def plot_discrepancy_hist(auditors: dict, y_true: np.ndarray, out_path: Path):
    fig, axes = plt.subplots(1, len(auditors), figsize=(5.5 * len(auditors), 4.5),
                              squeeze=False)
    axes = axes[0]
    fraud = y_true.astype(bool)
    for ax, (name, signed) in zip(axes, auditors.items()):
        bins = np.linspace(-4, 4, 40)
        ax.hist(signed[~fraud], bins=bins, density=True, alpha=0.5,
                color="steelblue", label=f"non-fraud (n={int((~fraud).sum())})")
        ax.hist(signed[fraud], bins=bins, density=True, alpha=0.7,
                color="crimson", label=f"fraud (n={int(fraud.sum())})")
        ax.axvline(0, color="black", linestyle="--", alpha=0.5)
        ax.set_xlabel("signed discrepancy (claimed − predicted)")
        ax.set_ylabel("density")
        ax.set_title(f"{name}")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  saved {out_path}")


def per_station_accuracy(preds_df: pd.DataFrame, col: str) -> pd.DataFrame:
    mask = ~preds_df["is_fraud_candidate"].astype(bool)
    rows = []
    for s in ["station1", "station2", "station3"]:
        m = mask & (preds_df["station"] == s)
        if m.sum() == 0:
            continue
        pred_argmax = preds_df.loc[m, col].round().clip(0, 4).astype(int)
        true_c = preds_df.loc[m, "condition"] - 1
        acc = float((pred_argmax == true_c).mean())
        mae = float(np.abs(preds_df.loc[m, col] - true_c).mean())
        rows.append({"station": s, "n": int(m.sum()),
                     "acc": acc, "mae": mae})
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(PREDS_CSV)
    print(f"Loaded {len(df):,} predictions from {PREDS_CSV}")

    y_true = df["is_fraud_candidate"].values.astype(int)
    claimed = (df["condition"].values - 1).astype(float)
    stations = df["station"].values

    # Heuristic baseline
    y_heur = heuristic_flags(df)
    p_h, r_h, f1_h, _ = precision_recall_fscore_support(
        y_true, y_heur, average="binary", zero_division=0)
    print(f"\nHeuristic: P={p_h:.4f} R={r_h:.4f} F1={f1_h:.4f} "
          f"flagged={int(y_heur.sum())}")

    # Assemble the auditor signals (signed = claimed - pred; higher is suspicious)
    auditor_signals = {}
    if "v2_pred_expected" in df.columns:
        auditor_signals["V2"] = claimed - df["v2_pred_expected"].values
    if "s3_pred_expected" in df.columns:
        raw = df["s3_pred_expected"].values
        auditor_signals["S3-raw"] = claimed - raw
        cal = per_station_calibrate(raw, stations, claimed)
        df["s3_pred_calibrated"] = cal
        auditor_signals["S3-calibrated"] = claimed - cal

    # Per-auditor summary
    results = {
        "n_test": len(df),
        "n_fraud": int(y_true.sum()),
        "heuristic": {"precision": float(p_h), "recall": float(r_h),
                      "f1": float(f1_h), "flagged": int(y_heur.sum())},
    }
    heur_mask = y_heur.astype(bool)
    print("\n=== Auditor results ===")
    for name, signed in auditor_signals.items():
        results[name] = summarize_auditor(name, signed, y_true, heur_mask)

    # Per-station accuracy for each auditor
    station_tables = {}
    if "v2_pred_expected" in df.columns:
        station_tables["V2"] = per_station_accuracy(df, "v2_pred_expected")
    if "s3_pred_expected" in df.columns:
        station_tables["S3-raw"] = per_station_accuracy(df, "s3_pred_expected")
        station_tables["S3-calibrated"] = per_station_accuracy(df, "s3_pred_calibrated")
    print("\n=== Per-station accuracy (non-fraud only) ===")
    for name, t in station_tables.items():
        print(f"\n{name}:")
        print(t.to_string(index=False))
        results[f"{name}_per_station"] = t.to_dict(orient="records")

    # Figures
    plot_pr_curves(auditor_signals, y_true, (p_h, r_h),
                   FIGURES_DIR / "auditor_pr_curves.png")
    plot_discrepancy_hist(auditor_signals, y_true,
                          FIGURES_DIR / "auditor_discrepancy_hist.png")

    # Save JSON
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {RESULTS_JSON}")

    # Also re-save the predictions with calibrated column
    df.to_csv(PREDS_CSV, index=False)


if __name__ == "__main__":
    main()
