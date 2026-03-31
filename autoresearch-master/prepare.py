"""
Fixed data loading and evaluation harness for clothing autoresearch.
DO NOT MODIFY — this file is read-only for the agent.

Provides:
  - Constants (TIME_BUDGET, IMG_SIZE, etc.)
  - ClothingDataset class
  - Train/test transforms
  - make_dataloaders(batch_size, img_size) -> (train_loader, test_loader)
  - evaluate(model, device) -> dict with condition_acc, fraud_f1, combined_score
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import f1_score

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

TIME_BUDGET = 300          # training time budget in seconds (5 minutes)
IMG_SIZE = 260             # default image size
NUM_CONDITION_CLASSES = 5  # condition grades 1-5 (stored as 0-4)
EVAL_BATCH_SIZE = 32       # fixed batch size for evaluation

# Data paths (relative to autoresearch-master/)
CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "cleaned_metadata.csv")

# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_train_transforms(img_size=IMG_SIZE):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=20, scale=(0.85, 1.15), shear=10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.08),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
    ])

def get_test_transforms(img_size=IMG_SIZE):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ClothingDataset(Dataset):
    """
    Returns front and back images as separate 3-channel tensors,
    plus a label dict with condition, fraud, stains, and holes.
    """

    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        front = Image.open(row["front_path"]).convert("RGB")
        back = Image.open(row["back_path"]).convert("RGB")

        if self.transform:
            front = self.transform(front)
            back = self.transform(back)

        labels = {
            "condition": torch.tensor(row["condition"] - 1, dtype=torch.long),
            "fraud": torch.tensor(float(row["is_fraud_candidate"]), dtype=torch.float),
            "stains": torch.tensor(row["stains"], dtype=torch.long),
            "holes": torch.tensor(row["holes"], dtype=torch.long),
        }
        return front, back, labels

# ---------------------------------------------------------------------------
# Dataloaders
# ---------------------------------------------------------------------------

def make_dataloaders(batch_size, img_size=IMG_SIZE):
    """Create train and test dataloaders from the cleaned CSV."""
    df = pd.read_csv(CSV_PATH)
    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]

    train_dataset = ClothingDataset(train_df, transform=get_train_transforms(img_size))
    test_dataset = ClothingDataset(test_df, transform=get_test_transforms(img_size))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    return train_loader, test_loader

# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, device):
    """
    Fixed evaluation function. Creates its own test dataloader with
    deterministic transforms and fixed batch size.

    The model must:
      - Accept (front_images, back_images) as input
      - Return a dict {"condition": tensor(B, 5), "fraud": tensor(B, 1)}

    Returns:
      dict with condition_acc, fraud_f1, combined_score
      combined_score = 0.6 * condition_acc + 0.4 * fraud_f1
    """
    df = pd.read_csv(CSV_PATH)
    test_df = df[df["split"] == "test"]
    test_dataset = ClothingDataset(test_df, transform=get_test_transforms(IMG_SIZE))
    test_loader = DataLoader(
        test_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    model.eval()
    all_cond_preds = []
    all_cond_targets = []
    all_fraud_preds = []
    all_fraud_targets = []

    for fronts, backs, labels in test_loader:
        fronts = fronts.to(device)
        backs = backs.to(device)

        outputs = model(fronts, backs)

        # Condition predictions
        cond_preds = outputs["condition"].argmax(dim=1).cpu()
        all_cond_preds.append(cond_preds)
        all_cond_targets.append(labels["condition"])

        # Fraud predictions
        fraud_probs = torch.sigmoid(outputs["fraud"]).squeeze(-1).cpu()
        fraud_preds = (fraud_probs > 0.5).long()
        all_fraud_preds.append(fraud_preds)
        all_fraud_targets.append(labels["fraud"].long())

    all_cond_preds = torch.cat(all_cond_preds)
    all_cond_targets = torch.cat(all_cond_targets)
    all_fraud_preds = torch.cat(all_fraud_preds)
    all_fraud_targets = torch.cat(all_fraud_targets)

    condition_acc = (all_cond_preds == all_cond_targets).float().mean().item()
    fraud_f1 = f1_score(
        all_fraud_targets.numpy(),
        all_fraud_preds.numpy(),
        zero_division=0.0,
    )

    combined_score = 0.6 * condition_acc + 0.4 * fraud_f1

    return {
        "condition_acc": condition_acc,
        "fraud_f1": float(fraud_f1),
        "combined_score": combined_score,
    }

# ---------------------------------------------------------------------------
# Main — sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"CSV path: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"Total rows: {len(df):,}")
    print(f"Train: {len(df[df['split'] == 'train']):,}")
    print(f"Test:  {len(df[df['split'] == 'test']):,}")

    print(f"\nCondition distribution:")
    print(df["condition"].value_counts().sort_index().to_string())

    print(f"\nFraud distribution:")
    print(df["is_fraud_candidate"].value_counts().to_string())

    # Verify a few image paths exist
    sample = df.sample(5, random_state=42)
    missing = 0
    for _, row in sample.iterrows():
        for col in ["front_path", "back_path"]:
            if not os.path.exists(row[col]):
                print(f"  MISSING: {row[col]}")
                missing += 1
    if missing == 0:
        print(f"\nImage paths verified (5 samples checked)")

    # Load one batch
    train_loader, test_loader = make_dataloaders(batch_size=4, img_size=IMG_SIZE)
    fronts, backs, labels = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  Front: {fronts.shape}")
    print(f"  Back:  {backs.shape}")
    print(f"  Condition: {labels['condition']}")
    print(f"  Fraud: {labels['fraud']}")

    print(f"\nSanity check passed. Ready to train.")
