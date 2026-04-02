"""
data_cleaning.py
----------------
Processes raw data/clothing_v3 into a single cleaned master CSV
optimised for a Multi-Task Learning model.

Usage (from project root):
    python src/data_cleaning.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split # type: ignore

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "clothing_v3"
OUTPUT_CSV   = PROJECT_ROOT / "data" / "cleaned_metadata_v2.csv"

# ── Normalisation maps ─────────────────────────────────────────────────────────
DEFECT_MAP: dict[str, int] = {
    "none":  0,
    "no":    0,
    "minor": 1,
    "yes":   1,
    "major": 2,
}

USAGE_BUCKET: dict[str, str] = {
    "reuse":           "Pristine",
    "export":          "Discount",
    "repair":          "Damaged",
    "remake":          "Damaged",
    "recycle":         "Waste",
    "rcycle":          "Waste",
    "energy recovery": "Waste",
}

STATION_RE = re.compile(r"(station\d+)", re.IGNORECASE)

# ── Category / Type mappings (from labels.py) ─────────────────────────────────
CATEGORY_LIST = ['Ladies', 'Men', 'Children', 'Unisex']
CATEGORY_MAP = {v.strip().lower(): i for i, v in enumerate(CATEGORY_LIST)}

TYPE_LIST = [
    'Blazer', 'Blouse', 'Cardigan', 'Denim jacket', 'Dress', 'Hoodie',
    'Jacket', 'Jeans', 'Night gown', 'Outerwear', 'Pajamas', 'Rain jacket',
    'Rain trousers', 'Robe', 'Shirt', 'Shorts', 'Skirt', 'Sweater',
    'T-shirt', 'Tank top', 'Tights', 'Top', 'Training top', 'Trousers',
    'Tunic', 'Vest', 'Winter jacket', 'Winter trousers',
]
TYPE_MAP = {v.strip().lower(): i for i, v in enumerate(TYPE_LIST)}


# ── Helpers ────────────────────────────────────────────────────────────────────
def normalise_defect(value: str | int) -> int:
    """Map stains / holes / smell → 0 / 1 / 2.  Returns -1 on unknown."""
    return DEFECT_MAP.get(str(value).strip().lower(), -1)


def normalise_usage(value: str) -> str:
    """Map raw usage strings into four business-logic buckets."""
    return USAGE_BUCKET.get(value.strip().lower(), "Unknown")


def extract_station(path: Path) -> str:
    """Pull 'stationN' from the directory path."""
    for part in path.parts:
        m = STATION_RE.search(part)
        if m:
            return m.group(1).lower()
    return "unknown"


def parse_label(json_path: Path) -> dict | None:
    """
    Parse one label JSON into a cleaned row dict.
    Returns None if the file is malformed or missing required fields.
    """
    try:
        with open(json_path) as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    condition = raw.get("condition")
    pilling   = raw.get("pilling")
    if condition is None or pilling is None:
        return None

    stains = normalise_defect(raw.get("stains", "none"))
    holes  = normalise_defect(raw.get("holes",  "none"))
    smell  = normalise_defect(raw.get("smell",  "none"))
    usage  = normalise_usage(raw.get("usage", ""))

    # Category and type as integer codes (-1 = unknown)
    raw_cat = str(raw.get("category", "")).strip().lower()
    category = CATEGORY_MAP.get(raw_cat, -1)

    raw_type = str(raw.get("type", "")).strip().lower()
    clothing_type = TYPE_MAP.get(raw_type, -1)

    timestamp = json_path.stem.replace("labels_", "")

    return {
        "timestamp":  timestamp,
        "station":    extract_station(json_path),
        "front_path": str(json_path.parent / f"front_{timestamp}.jpg"),
        "back_path":  str(json_path.parent / f"back_{timestamp}.jpg"),
        "condition":  int(condition),
        "pilling":    int(pilling),
        "stains":     stains,
        "holes":      holes,
        "smell":      smell,
        "usage":      usage,
        "material":   raw.get("material", ""),
        "category":   category,
        "clothing_type": clothing_type,
    }


# ── Core pipeline ──────────────────────────────────────────────────────────────
def build_master_dataframe() -> tuple[pd.DataFrame, int, int]:
    """
    Walk RAW_DATA_DIR, parse every labels_*.json, verify image existence.

    Returns
    -------
    df               : cleaned DataFrame
    n_missing_images : rows dropped because front or back image was absent
    n_bad_json       : rows dropped due to malformed / incomplete JSON
    """
    rows: list[dict] = []
    n_missing_images = 0
    n_bad_json       = 0

    for json_path in sorted(RAW_DATA_DIR.rglob("labels_*.json")):
        timestamp = json_path.stem.replace("labels_", "")
        front = json_path.parent / f"front_{timestamp}.jpg"
        back  = json_path.parent / f"back_{timestamp}.jpg"

        if not front.exists() or not back.exists():
            n_missing_images += 1
            continue

        row = parse_label(json_path)
        if row is None:
            n_bad_json += 1
            continue

        rows.append(row)

    return pd.DataFrame(rows), n_missing_images, n_bad_json


def add_fraud_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag items whose stated condition is high (>=4) but have a Major
    defect in stains or holes — these are the prime fraud candidates.
    """
    df["is_fraud_candidate"] = (
        (df["condition"] >= 4) & ((df["stains"] == 2) | (df["holes"] == 2))
    )
    return df


def stratified_split(
    df: pd.DataFrame,
    test_size: float = 0.20,
    seed: int = 42,
) -> pd.DataFrame:
    """
    80/20 stratified split on the `holes` column so rare Major-damage
    examples appear in both train and test sets.
    Falls back to a plain shuffle split if any stratum is too small.
    """
    try:
        train_idx, test_idx = train_test_split(
            df.index,
            test_size=test_size,
            random_state=seed,
            stratify=df["holes"],
        )
    except ValueError:
        train_idx, test_idx = train_test_split(
            df.index,
            test_size=test_size,
            random_state=seed,
        )

    df["split"] = "train"
    df.loc[test_idx, "split"] = "test"
    return df


def print_summary(df: pd.DataFrame, n_missing: int, n_bad: int) -> None:
    total_scanned = len(df) + n_missing + n_bad
    print("\n" + "=" * 55)
    print("  DATA CLEANING SUMMARY")
    print("=" * 55)
    print(f"  JSON files scanned       : {total_scanned:>7,}")
    print(f"  Dropped – missing images : {n_missing:>7,}")
    print(f"  Dropped – bad JSON       : {n_bad:>7,}")
    print(f"  Rows in output CSV       : {len(df):>7,}")
    print()
    print("  Train / test split")
    print(f"    train : {(df['split'] == 'train').sum():>6,}")
    print(f"    test  : {(df['split'] == 'test').sum():>6,}")
    print()
    print("  Fraud candidate flag")
    vc = df["is_fraud_candidate"].value_counts()
    print(f"    True  : {vc.get(True,  0):>6,}  "
          f"({vc.get(True, 0) / len(df) * 100:.2f}%)")
    print(f"    False : {vc.get(False, 0):>6,}")
    print()
    print("  Usage bucket distribution")
    for bucket, count in df["usage"].value_counts().items():
        print(f"    {bucket:<12} : {count:>6,}")
    print()
    print("  Condition distribution")
    for rating, count in sorted(df["condition"].value_counts().items()):
        print(f"    {rating}  : {count:>6,}")
    print("=" * 55)
    print(f"\n  Output saved → {OUTPUT_CSV}\n")


# ── Entry point ────────────────────────────────────────────────────────────────
def main() -> None:
    if not RAW_DATA_DIR.exists():
        sys.exit(f"ERROR: raw data directory not found:\n  {RAW_DATA_DIR}")

    print(f"Scanning {RAW_DATA_DIR} …")
    df, n_missing, n_bad = build_master_dataframe()

    if df.empty:
        sys.exit("ERROR: No valid rows found — check RAW_DATA_DIR.")

    df = add_fraud_flag(df)
    df = stratified_split(df)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print_summary(df, n_missing, n_bad)


if __name__ == "__main__":
    main()
