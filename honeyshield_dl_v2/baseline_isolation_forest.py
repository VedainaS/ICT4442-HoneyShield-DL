"""
baseline_isolation_forest.py
-----------------------------
Classical (non-Deep-Learning) baseline for comparison. Operates on the
LAST event of each sequence (i.e., the same event each autoencoder is
ultimately judged on), using the same train/val/test split and the same
evaluation metrics, so it is directly comparable to the four DL models.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
import json

from train_evaluate import load_data, split_data, SEED


def run_isolation_forest():
    X, y = load_data()
    X_train, X_val, X_test, y_test = split_data(X, y)

    # Use only the last event's feature vector in each sequence (point-level
    # features), consistent with Isolation Forest's standard per-event usage.
    X_train_last = X_train[:, -1, :]
    X_test_last = X_test[:, -1, :]

    clf = IsolationForest(contamination=0.08, random_state=SEED)
    clf.fit(X_train_last)

    # score_samples: higher = more normal, so we negate for an anomaly score
    anomaly_scores = -clf.score_samples(X_test_last)
    y_pred = (clf.predict(X_test_last) == -1).astype(int)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    pr_auc = average_precision_score(y_test, anomaly_scores)

    result = {
        "model": "isolation_forest (baseline)",
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "pr_auc": pr_auc,
        "n_test": int(len(y_test)),
        "n_test_anomalous": int(y_test.sum()),
    }
    print(f"Isolation Forest -> Precision={precision:.4f}  Recall={recall:.4f}  "
          f"F1={f1:.4f}  PR-AUC={pr_auc:.4f}")
    return result


if __name__ == "__main__":
    result = run_isolation_forest()
    with open("results/isolation_forest_result.json", "w") as f:
        json.dump(result, f, indent=2)
