"""
train_evaluate.py
------------------
Implements the common unsupervised evaluation protocol for all four models:

1. Split sequences into train / val / test.
   - Train set: ONLY normal sequences (autoencoders never see anomalies).
   - Val set:   ONLY normal sequences, held out from training, used purely
                to select the anomaly-score threshold (no test-set leakage).
   - Test set:  BOTH normal and anomalous sequences, used only for final
                evaluation (never seen during training or thresholding).
2. Train each autoencoder to minimise reconstruction error on the train set.
3. Compute reconstruction error (MSE) on the val set -> set threshold at the
   99th percentile of NORMAL validation error (approximating the known ~8%
   anomaly rate without touching test labels).
4. On the test set: predict anomaly if reconstruction_error > threshold.
   Report Precision, Recall, F1-score, and PR-AUC (using the continuous
   reconstruction-error score, not just the thresholded prediction).

Usage:
    python train_evaluate.py --model mlp
    python train_evaluate.py --model cnn
    python train_evaluate.py --model lstm
    python train_evaluate.py --model attention
    python train_evaluate.py --model all      # runs all four, saves comparison table
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
import json
import os

from models import MODEL_REGISTRY

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data():
    X = np.load("data/sequences_X.npy")
    y = np.load("data/sequences_y.npy")
    return X, y


def split_data(X, y, seed=SEED):
    # First split off a labelled test set (stratified, keeps both classes)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )
    # From the remaining normal-only pool, carve train vs. val
    normal_mask = y_trainval == 0
    X_normal = X_trainval[normal_mask]
    X_train, X_val = train_test_split(X_normal, test_size=0.15, random_state=seed)
    return X_train, X_val, X_test, y_test


def make_loader(X, batch_size=64, shuffle=True):
    tensor_X = torch.tensor(X, dtype=torch.float32)
    ds = TensorDataset(tensor_X)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_model(model, train_loader, epochs=15, lr=1e-3):
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for (batch_x,) in train_loader:
            batch_x = batch_x.to(DEVICE)
            optimizer.zero_grad()
            recon = model(batch_x)
            loss = criterion(recon, batch_x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)
        avg_loss = total_loss / len(train_loader.dataset)
        print(f"  epoch {epoch+1}/{epochs}  train_MSE={avg_loss:.5f}")
    return model


def reconstruction_errors(model, X):
    model.eval()
    errors = []
    loader = make_loader(X, batch_size=128, shuffle=False)
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(DEVICE)
            recon = model(batch_x)
            err = ((recon - batch_x) ** 2).mean(dim=(1, 2))  # per-sequence MSE
            errors.append(err.cpu().numpy())
    return np.concatenate(errors)


def run_one_model(model_name, X_train, X_val, X_test, y_test, epochs=15):
    print(f"\n=== Training {model_name.upper()} Autoencoder ===")
    seq_len, num_features = X_train.shape[1], X_train.shape[2]
    model_cls = MODEL_REGISTRY[model_name]
    model = model_cls(seq_len, num_features)

    train_loader = make_loader(X_train, batch_size=64, shuffle=True)
    model = train_model(model, train_loader, epochs=epochs)

    # Threshold selection on held-out NORMAL validation data (no test leakage)
    val_errors = reconstruction_errors(model, X_val)
    threshold = float(np.percentile(val_errors, 99))

    # Final evaluation on labelled test set (normal + anomalous)
    test_errors = reconstruction_errors(model, X_test)
    y_pred = (test_errors > threshold).astype(int)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    pr_auc = average_precision_score(y_test, test_errors)  # uses continuous score

    result = {
        "model": model_name,
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "pr_auc": pr_auc,
        "n_test": int(len(y_test)),
        "n_test_anomalous": int(y_test.sum()),
    }
    print(f"  -> Precision={precision:.4f}  Recall={recall:.4f}  F1={f1:.4f}  PR-AUC={pr_auc:.4f}")
    return result, model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_REGISTRY.keys()) + ["all"], default="all")
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()

    X, y = load_data()
    X_train, X_val, X_test, y_test = split_data(X, y)
    print(f"Train (normal only): {len(X_train)} | Val (normal only): {len(X_val)} | "
          f"Test: {len(X_test)} ({y_test.sum()} anomalous)")

    os.makedirs("results", exist_ok=True)
    models_to_run = list(MODEL_REGISTRY.keys()) if args.model == "all" else [args.model]

    all_results = []
    for name in models_to_run:
        result, _ = run_one_model(name, X_train, X_val, X_test, y_test, epochs=args.epochs)
        all_results.append(result)

    with open("results/comparison_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n=== Comparison Table ===")
    print(f"{'Model':<12}{'Precision':<12}{'Recall':<10}{'F1':<10}{'PR-AUC':<10}")
    for r in all_results:
        print(f"{r['model']:<12}{r['precision']:<12.4f}{r['recall']:<10.4f}{r['f1_score']:<10.4f}{r['pr_auc']:<10.4f}")

    print("\nSaved: results/comparison_results.json")
