"""
analysis_by_attack.py
Re-runs the four autoencoders (same order and seed handling as `train_evaluate.py --model all`) (train_evaluate.py,
models.py) and additionally reports per-attack-category recall, so the Final
report's error analysis is reproducible. Writes current results and checkpoints.

Usage: python analysis_by_attack.py --epochs 25
"""
import argparse, json
import numpy as np, torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
import train_evaluate as te
from models import MODEL_REGISTRY

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=25)
args = ap.parse_args()

X, y = te.load_data()
attack = np.load("data/sequences_attack.npy", allow_pickle=True)
X_train, X_val, X_test, y_test = te.split_data(X, y)
atk_test = attack[np.load("data/sequences_split.npy") == "test"]
cats = ["Brute Force", "Credential Stuffing", "Impossible Travel", "Device Spoofing", "Privilege Escalation"]

# Capture the test-set errors that run_one_model already computes, instead of calling
# reconstruction_errors() again: an extra DataLoader would consume torch RNG state and
# change the models trained after it, so the numbers would no longer match
# `train_evaluate.py --model all`.
captured = {}
_orig = te.reconstruction_errors
def _wrapped(model, X):
    e = _orig(model, X)
    if X is X_test:
        captured["err"] = e
    return e
te.reconstruction_errors = _wrapped

out = []
for name in MODEL_REGISTRY:
    res, model = te.run_one_model(name, X_train, X_val, X_test, y_test, epochs=args.epochs)
    pred = (captured["err"] > res["threshold"]).astype(int)
    res["epochs"] = args.epochs
    res["recall_by_attack"] = {c: float(pred[atk_test == c].mean()) if (atk_test == c).any() else None for c in cats}
    res["n_by_attack"] = {c: int((atk_test == c).sum()) for c in cats}
    res["false_positive_rate"] = float(pred[atk_test == "Normal"].mean())
    res["true_positives"] = int(pred[y_test == 1].sum())
    res["false_positives"] = int(pred[y_test == 0].sum())
    out.append(res)

json.dump(out, open("results/final_run_%dep.json" % args.epochs, "w"), indent=2)
print(json.dumps(out, indent=2))
