"""
build_sequences.py
-------------------
Preprocesses raw_logs.csv and restructures it into fixed-length, per-user
temporal sequences — the SAME input representation used by all four models
(MLP, CNN, LSTM, Attention), so that architecture is the only variable being
compared (fair-comparison requirement).

Steps:
1. Clean + encode categorical/behavioural features per event.
2. Group events by user, order by timestamp.
3. Slide a fixed-length window (N=10 events) per user to build sequences.
   Users with fewer than N events are zero-padded at the start.
4. Save:
   - data/sequences_X.npy        shape (num_sequences, N, num_features)
   - data/sequences_y.npy        shape (num_sequences,)  1 if ANY event in
                                   the window is a true anomaly, else 0
   - data/sequences_attack.npy   dominant attack type label per sequence (for analysis only)

Ground-truth labels are NOT used for training (only for threshold selection
and final evaluation, as per the unsupervised protocol).
"""

import numpy as np
import pandas as pd
import os

SEQ_LEN = 10  # fixed-length window per user (N=10 events)


def frequency_encode(series):
    freq = series.value_counts(normalize=True)
    return series.map(freq).astype(float)


def build_features(df):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["login_hour"] = df["timestamp"].dt.hour
    df["is_weekend"] = df["timestamp"].dt.weekday.isin([5, 6]).astype(int)

    # Behavioural outcome flags
    df["login_success_flag"] = df["login_success"].astype(int)
    df["is_failed_login"] = (df["failed_attempts"] > 0).astype(int)
    df["is_sensitive_resource"] = df["resource_accessed"].isin(
        ["admin_panel", "payroll_system", "hr_database", "finance_console"]
    ).astype(int)
    df["unusual_login_hour"] = (~df["login_hour"].between(6, 22)).astype(int)

    # Statistical deviation feature
    df["session_duration_zscore"] = (
        (df["session_duration"] - df["session_duration"].mean())
        / df["session_duration"].std()
    )

    # Frequency-encoded rarity features (fit on this dataset only;
    # in a train/test split these should be fit on train only to avoid leakage)
    for col in ["country", "resource_accessed", "browser", "device_type", "ip_address"]:
        df[f"{col}_freq"] = frequency_encode(df[col])

    feature_cols = [
        "login_success_flag", "is_failed_login", "is_sensitive_resource",
        "unusual_login_hour", "session_duration_zscore",
        "country_freq", "resource_accessed_freq", "browser_freq",
        "device_type_freq", "ip_address_freq", "failed_attempts",
    ]
    return df, feature_cols


def build_sequences(df, feature_cols, seq_len=SEQ_LEN):
    X_list, y_list, attack_list = [], [], []
    num_features = len(feature_cols)

    for user_id, group in df.groupby("user_id"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        feats = group[feature_cols].values.astype(np.float32)
        labels = group["is_anomaly_ground_truth"].values
        attacks = group["attack_type"].values

        n = len(group)
        for end_idx in range(1, n + 1):
            start_idx = max(0, end_idx - seq_len)
            window_feats = feats[start_idx:end_idx]
            window_labels = labels[start_idx:end_idx]
            window_attacks = attacks[start_idx:end_idx]

            pad_len = seq_len - len(window_feats)
            if pad_len > 0:
                pad = np.zeros((pad_len, num_features), dtype=np.float32)
                window_feats = np.vstack([pad, window_feats])

            # Label corresponds to the LAST event in the window (the event being
            # classified); earlier events in the window provide temporal context only.
            # This keeps the sequence-level anomaly rate consistent with the
            # original per-event rate (~8%), unlike an "any event in window"
            # rule which would inflate it.
            seq_label = int(window_labels[-1])
            dominant_attack = window_attacks[-1]

            X_list.append(window_feats)
            y_list.append(seq_label)
            attack_list.append(dominant_attack)

    X = np.stack(X_list)
    y = np.array(y_list)
    attacks = np.array(attack_list)
    return X, y, attacks


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = pd.read_csv("data/raw_logs.csv")
    df, feature_cols = build_features(df)
    X, y, attacks = build_sequences(df, feature_cols, seq_len=SEQ_LEN)

    np.save("data/sequences_X.npy", X)
    np.save("data/sequences_y.npy", y)
    np.save("data/sequences_attack.npy", attacks)

    print(f"Built {X.shape[0]} sequences of shape ({X.shape[1]}, {X.shape[2]})")
    print(f"Anomalous sequences: {y.sum()} ({100*y.mean():.2f}%)")
    print(f"Feature columns ({len(feature_cols)}): {feature_cols}")
